"""Unit tests for base preparation, run isolation, and snapshot experiment ordering."""

import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import harness, prepare_world, test_e1, test_e2


class FakeControl:
    """Simulate control requests and persist fake world state for experiment tests."""
    def __init__(self, directory, events):
        """Initialize this test fixture without launching a real server."""
        self.directory = directory
        self.events = events
        self.world = directory / "world"
        state_file = self.world / "fake-state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {"gameTime": 0, "bot": None}
        self.current = state["gameTime"]
        self.saved_bot = state["bot"]
        self.bot = None
        self.rules = {"minecraft:advance_weather": True, "minecraft:spawn_mobs": True,
                      "minecraft:random_tick_speed": 3, "minecraft:advance_time": True}

    def save(self):
        """Persist the fake game time and bot state for later restore requests."""
        self.world.mkdir(exist_ok=True)
        (self.world / "fake-state.json").write_text(json.dumps({"gameTime": self.current, "bot": self.bot}))

    def request(self, command, args=None):
        """Handle a scripted control request using the enclosing test fixture."""
        args = args or {}
        self.events.append((command, self.directory.name, self.current, copy.deepcopy(args)))
        if command == "ping":
            return {"gameTime": self.current}
        if command == "hash":
            return {"gameTime": self.current, **{key: "0" * 64 for key in harness.HASH_KEYS}}
        if command == "command":
            parts = args["command"].split()
            if parts[0] == "gamerule":
                if len(parts) == 2:
                    return {"value": int(self.rules[parts[1]])}
                value = {"true": True, "false": False}.get(parts[2])
                if value is None:
                    value = int(parts[2])
                if self.rules[parts[1]] == value:
                    raise harness.CommandError("gamerule value already current")
                self.rules[parts[1]] = value
                return {"value": int(value)}
            return {"value": 16}
        if command in ("lockstep", "recording", "observe", "bot_select", "bot_break"):
            return {}
        if command == "spawn_bot":
            if self.bot is not None:
                raise AssertionError("duplicate spawn")
            self.bot = {"pos": [args["x"], args["y"], args["z"]], "yRot": args["yaw"],
                        "xRot": args["pitch"], "deltaMovement": [0, 0, 0], "onGround": True,
                        "health": 20, "food": 20, "inventory": 0, "input": {}}
            return {}
        if command == "restore_bots":
            self.bot = copy.deepcopy(self.saved_bot)
            if self.bot is None:
                raise AssertionError("snapshot has no saved bot")
            return {}
        if command == "give":
            self.bot["inventory"] += args["count"]
            return {}
        if command == "bot_place":
            self.bot["inventory"] -= 1
            return {}
        if command == "bot_input":
            self.bot["input"] = dict(args)
            self.bot["yRot"] = args["yaw"]
            return {}
        if command == "bot_state":
            if self.bot is None:
                raise AssertionError("bot_state before spawn/restore")
            return copy.deepcopy(self.bot)
        if command == "save_snapshot":
            self.save()
            return {}
        if command == "await_entities":
            return {"gameTime": self.current, "chunks": 400, "waitedMs": 0.0}
        if command == "step":
            hashes = []
            for _tick in range(args["n"]):
                if self.bot is not None:
                    inputs = self.bot["input"]
                    speed = inputs.get("forward", 0) * 0.01
                    yaw = math.radians(inputs.get("yaw", 0))
                    self.bot["pos"][0] -= math.sin(yaw) * speed
                    self.bot["pos"][2] += math.cos(yaw) * speed
                self.current += 1
                if args.get("hash_every") and self.current % args["hash_every"] == 0:
                    payload = json.dumps({"gameTime": self.current, "bot": self.bot}, sort_keys=True)
                    components = {key: hashlib.sha256((key + payload).encode()).hexdigest()
                                  for key in harness.HASH_KEYS}
                    hashes.append({"gameTime": self.current,
                                   "hash": {"gameTime": self.current, **components}})
            return {"gameTime": self.current, "hashes": hashes}
        raise AssertionError(f"Unexpected fake command: {command}")


class ExperimentTests(unittest.TestCase):
    """Exercise preparation and experiment ordering without launching a JVM."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name)
        self.events = []
        self.servers = []

        def server_factory(directory, **options):
            """Return a fake server that records lifecycle and control events."""
            events = self.events
            control = FakeControl(Path(directory), events)

            class FakeServer:
                """Expose a fake client and record context-managed server lifecycle events."""
                client = control

                def __enter__(server):
                    """Enter the existing fixture context and return its managed object."""
                    events.append(("start", Path(directory).name, control.current, options))
                    return server

                def __exit__(server, *_exc):
                    """Apply the existing fixture cleanup without changing exception handling."""
                    control.save()
                    events.append(("stop", Path(directory).name, control.current, {}))

            server = FakeServer()
            self.servers.append(server)
            return server

        self.patchers = [patch.object(harness, "Server", side_effect=server_factory),
                         patch.object(harness, "clone_apfs", side_effect=self.fake_clone),
                         patch.object(harness, "disk_usage", return_value="Filesystem df -k fixture")]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_setup(self, name, runs_dir, *, bootstrap_base=False):
        """Create an isolated fake server directory for an experiment."""
        directory = Path(runs_dir) / name
        directory.mkdir()
        (directory / "mods").mkdir()
        (directory / "fabric-server-launch.jar").write_text("shell-only")
        self.events.append(("setup", name, None, {}))
        return directory

    def fake_clone(self, source, destination):
        """Copy the fixture tree and record the simulated clone operation."""
        state_file = Path(source) / ("fake-state.json" if Path(source).name == "world" else "world/fake-state.json")
        current = json.loads(state_file.read_text())["gameTime"]
        self.events.append(("clone", str(source), current, str(destination)))
        shutil.copytree(source, destination)

    def base(self):
        """Prepare the base fixture used by the experiment tests."""
        directory = self.fake_setup("base", self.runs)
        world = directory / "world"
        world.mkdir()
        (world / "fake-state.json").write_text(json.dumps({"gameTime": 100, "bot": None}))
        (directory / "fabric-server-launch.jar").write_text("base-shell")

    def invoke(self, module):
        """Run the gate with patched dependencies and capture its output."""
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = module.main(["--runs-dir", str(self.runs)])
        self.assertEqual(status, 0, output.getvalue())
        return output.getvalue()

    def test_prepare_uses_exact_gamerules_and_100_lockstep_ticks(self):
        """Verify prepare uses exact gamerules and 100 lockstep ticks."""
        with patch.object(prepare_world, "setup_server", side_effect=self.fake_setup) as setup:
            output = self.invoke(prepare_world)
        setup.assert_called_once_with("base", self.runs, bootstrap_base=True)
        self.assertIn("game time 100", output)
        commands = [event[3]["command"] for event in self.events if event[0] == "command"]
        self.assertEqual(commands, ["gamerule minecraft:advance_weather",
                                    "gamerule minecraft:advance_weather false",
                                    "gamerule minecraft:spawn_mobs",
                                    "gamerule minecraft:spawn_mobs false",
                                    "gamerule minecraft:random_tick_speed",
                                    "gamerule minecraft:random_tick_speed 0",
                                    "gamerule minecraft:advance_time",
                                    "forceload add -160 -160 -1 159", "forceload add 0 -160 159 159",
                                    "forceload query"])
        lockstep = next(event for event in self.events if event[0] == "lockstep")
        self.assertEqual(lockstep[3], {"enabled": True})
        names = [event[0] for event in self.events]
        forceload = max(index for index, event in enumerate(self.events)
                        if event[0] == "command" and event[3]["command"].startswith("forceload add"))
        self.assertGreater(names.index("await_entities"), forceload)  # a fresh world awaits only after forceload
        steps = [event for event in self.events if event[0] == "step"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0][3], {"n": 100})
        hashes = [event for event in self.events if event[0] == "hash"]
        self.assertEqual(hashes, [("hash", "base", 100, {"region": harness.REGION})])
        self.assertLess(self.events.index(hashes[0]), next(
            index for index, event in enumerate(self.events) if event[0] == "stop"))
        metadata = json.loads((self.runs / "base/fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["world_hash"], "0" * 64)
        self.assertIsNone(metadata["manifest"])

    def test_gamerule_configuration_is_idempotent_with_26_3_setter_semantics(self):
        """Verify gamerule configuration is idempotent with 26 3 setter semantics."""
        client = FakeControl(self.runs, self.events)
        prepare_world.configure_gamerules(client)
        self.assertEqual(client.rules, dict(prepare_world.GAME_RULES))
        self.events.clear()
        prepare_world.configure_gamerules(client)
        self.assertEqual([event[3]["command"] for event in self.events],
                         [f"gamerule {name}" for name, _desired in prepare_world.GAME_RULES])

    def test_advance_time_is_enabled_only_when_query_returns_zero(self):
        """Verify advance time is enabled only when query returns zero."""
        client = FakeControl(self.runs, self.events)
        client.rules = dict(prepare_world.GAME_RULES)
        client.rules["minecraft:advance_time"] = False
        prepare_world.configure_gamerules(client)
        setters = [event[3]["command"] for event in self.events
                   if len(event[3]["command"].split()) == 3]
        self.assertEqual(setters, ["gamerule minecraft:advance_time true"])

    def test_gamerule_query_and_set_failures_propagate_without_suppression(self):
        """Verify gamerule query and set failures propagate without suppression."""
        for responses in ([harness.CommandError("query failed")],
                          [{"value": 1}, harness.CommandError("set failed")]):
            client = Mock()
            client.request.side_effect = responses
            with self.assertRaises(harness.CommandError):
                prepare_world.configure_gamerules(client)
            self.assertEqual(client.request.call_count, len(responses))

    def test_invalid_gamerule_query_value_does_not_trigger_setter(self):
        """Verify invalid gamerule query value does not trigger setter."""
        for value in (None, True, 1.0, "1", 2):
            client = Mock()
            client.request.return_value = {"value": value}
            with self.subTest(value=value), self.assertRaises(harness.ProtocolError):
                prepare_world.configure_gamerules(client)
            client.request.assert_called_once_with("command", {"command": "gamerule minecraft:advance_weather"})

    def test_e1_full_length_absolute_schedule_and_movement_checks(self):
        """Verify e1 full length absolute schedule and movement checks."""
        self.base()
        output = self.invoke(test_e1)
        self.assertIn("PASS E1: 1200 matching hashes", output)
        hashes_a = json.loads((self.runs / "e1_a/hashes.json").read_text())
        self.assertEqual(hashes_a, json.loads((self.runs / "e1_b/hashes.json").read_text()))
        self.assertEqual([record["gameTime"] for record in hashes_a], list(range(120, 24101, 20)))
        for name in ("e1_a", "e1_b"):
            events = [event for event in self.events if event[1] == name]
            self.assertEqual(sum(event[0] == "give" for event in events), 1)
            self.assertEqual(sum(event[0] == "spawn_bot" for event in events), 1)
            self.assertGreater(sum(event[0] == "bot_state" for event in events), 100)
            inputs = [event for event in events if event[0] == "bot_input"]
            self.assertEqual(inputs[0][2], 100)
            self.assertEqual(inputs[0][3], harness.routine.actions_for(100)[0]["args"])
            for event in events:
                if event[0] == "step":
                    self.assertEqual(event[3]["region"], harness.REGION)
                    self.assertEqual(event[3]["hash_every"], 20)

    def test_e2_saves_live_world_only_and_never_respawns_or_refills_c(self):
        """Verify e2 saves live world only and never respawns or refills c."""
        self.base()
        with patch.object(test_e2, "setup_server", side_effect=self.fake_setup) as setup:
            output = self.invoke(test_e2)
        setup.assert_called_once_with("e2_c", self.runs)
        self.assertIn("PASS E2: 600 matching hashes", output)
        self.assertIn("Fork latency (save + APFS world clone)", output)
        self.assertIn("df -k before clone", output)
        self.assertIn("df -k after clone", output)
        clones = [event for event in self.events if event[0] == "clone"]
        self.assertEqual(clones, [("clone", str(self.runs / "base"), 100, str(self.runs / "e2_a")),
                                  ("clone", str(self.runs / "e2_a/world"), 12100, str(self.runs / "e2_c/world"))])
        snapshot_index = next(index for index, event in enumerate(self.events) if event[0] == "save_snapshot")
        clone_index = self.events.index(clones[1])
        stop_a_index = next(index for index, event in enumerate(self.events) if event[:2] == ("stop", "e2_a"))
        self.assertLess(snapshot_index, clone_index)
        self.assertLess(clone_index, stop_a_index)
        self.assertFalse(any(event[0] in ("step", "bot_input")
                             for event in self.events[snapshot_index + 1:clone_index]))
        restored = [event for event in self.events if event[1] == "e2_c"]
        self.assertEqual(sum(event[0] == "restore_bots" for event in restored), 1)
        self.assertFalse(any(event[0] in ("give", "spawn_bot") for event in restored))
        self.assertEqual(next(event[2] for event in restored if event[0] == "bot_input"), 12100)
        self.assertEqual((self.runs / "e2_c/fabric-server-launch.jar").read_text(), "shell-only")
        hashes_a = json.loads((self.runs / "e2_a/hashes.json").read_text())
        hashes_c = json.loads((self.runs / "e2_c/hashes.json").read_text())
        self.assertEqual(hashes_c, [record for record in hashes_a if record["gameTime"] > 12100])
        self.assertEqual(self.servers[-1].client.bot["inventory"], 4)

    def test_existing_run_preflight_makes_no_changes(self):
        """Verify existing run preflight makes no changes."""
        self.base()
        (self.runs / "e1_b").mkdir()
        with self.assertRaises(FileExistsError):
            test_e1.main(["--runs-dir", str(self.runs)])
        self.assertFalse((self.runs / "e1_a").exists())
        (self.runs / "e2_a").mkdir()
        with patch.object(test_e2, "setup_server") as setup:
            with self.assertRaises(FileExistsError):
                test_e2.main(["--runs-dir", str(self.runs)])
            setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
