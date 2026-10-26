"""Server-free IT-22 checks using scripted protocol replies, never Minecraft or APFS clones."""

from collections import deque
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.scenario import ScenarioManifest
from scripts import harness, test_it15_skills, test_it22_night as probe, village_session

NPC = "adult_02"
ORIGINAL_POSITION = [-18.2, -59.0625, -7.8]
ORIGIN = [-19, -60, -8]
LOAD = {"item": "minecraft:bread", "count_band": "sack"}
OBSERVE_ARGS = {"name": probe.OBSERVER, "include_terrain": False}


def observation(tick, visible=True, load=None, distance=12):
    """Describe settled bots on the probe's gravel floor, distinct from the NPC's original fractional position."""
    entities = [{"name": "Bramble", "npcId": NPC, "pos": list(ORIGINAL_POSITION), "distance": 1.5}]
    if visible:
        entity = {"type": "minecraft:player", "name": probe.TARGET,
                  "pos": [ORIGIN[0] + distance + 0.5, ORIGIN[1], ORIGIN[2] + 0.5], "distance": distance}
        if load is not None:
            entity["load"] = deepcopy(load)
        entities.append(entity)
    return {"gameTime": tick, "self": {"name": probe.OBSERVER, "pos": [-18.5, -60, -6.5],
                                     "health": 20, "food": 20, "inventory": []}, "entities": entities,
            "heard": [], "changes": []}


class FakeNightClient:
    """Require the exact command order and arguments, returning only predeclared detached protocol replies."""

    def __init__(self, replies):
        """Keep expected requests separate from the actual call trace."""
        self.replies = deque(deepcopy(replies))
        self.calls = []

    def request(self, command, args=None):
        """Fail unexpected calls rather than silently simulating an untested engine behaviour."""
        self.calls.append((command, deepcopy(args)))
        if not self.replies:
            raise AssertionError(f"Unexpected request: {self.calls[-1]!r}")
        expected_command, expected_args, result = self.replies.popleft()
        if (command, args) != (expected_command, expected_args):
            raise AssertionError(f"Expected {(expected_command, expected_args)!r}, got {(command, args)!r}")
        return deepcopy(result)


class NightProbeTest(unittest.TestCase):
    """Share real constructor arguments and small, explicit protocol scripts."""

    def setUp(self):
        """Keep every output in a temporary run root and capture the probe's ordinary check messages."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.args = harness.experiment_parser("").parse_args(["--runs-dir", str(self.root), "--port", "28000"])
        self.experiment = probe.IT22NightExperiment(self.args)

    def attach(self, replies):
        """Use the existing probe with a scripted client and the fixed stationary NPC geometry."""
        client = FakeNightClient(replies)
        self.experiment.client = client
        self.experiment.npc = NPC
        self.experiment.origin = list(ORIGIN)
        return client

    @staticmethod
    def sample_replies(start, visible, load=None, distance=12):
        """Admit one villager sight record in the same twenty-tick interval as the bot's view."""
        view = observation(start + 20, visible, load, distance)
        record = {"gameTime": start + 20, "npc": NPC, "entities": deepcopy(view["entities"])}
        return [("ping", None, {"gameTime": start}), ("step", {"n": 20}, {"gameTime": start + 20}),
                ("observe", OBSERVE_ARGS, view),
                ("npc_testimony", {"id": NPC, "since": start + 1}, {"records": [record]})]

    @staticmethod
    def prepare_replies(disabled=None):
        """Include a travelling NPC, fractional dwelling coordinates and a deliberately dirty first observation."""
        baseline = observation(6020)
        dirty = deepcopy(baseline)
        dirty["changes"] = [{"from": "minecraft:stone", "to": "minecraft:air"}]
        commands = ["fill -25 -60 -14 -1 -52 -2 minecraft:air",
                    "fill -25 -61 -14 -1 -61 -2 minecraft:gravel"]
        replies = [("npc_state", {"id": "adult_01"}, {"trip": {"to": "work"}, "pos": [90, 64, 90]}),
                   ("npc_state", {"id": NPC}, {"trip": None, "pos": list(ORIGINAL_POSITION)})]
        replies.extend(("command", {"command": command}, {}) for command in commands)
        replies.extend(("skill_cancel", {"name": name}, {}) for name in (probe.OBSERVER, probe.TARGET))
        for command in ("tp investigator -18.5 -60 -6.5 0 0", "tp envoy -6.5 -60 -7.5 0 0",
                        "clear envoy", "time set 18000"):
            replies.append(("command", {"command": command}, {}))
        replies.extend([("step", {"n": 20}, {"gameTime": 6020}), ("observe", OBSERVE_ARGS, dirty),
                        ("observe", OBSERVE_ARGS, baseline),
                        ("config_set", {"night_sight": False, "visible_loads": {}}, {}),
                        ("observe", OBSERVE_ARGS, baseline if disabled is None else disabled),
                        ("config_set", {"night_sight": True, "visible_loads": {"minecraft:bread": 9}}, {})])
        return replies

    def use_prepare_client(self, disabled=None):
        """Read NPC definitions through the production manifest API, not a fictional npcs attribute."""
        self.experiment.manifest = ScenarioManifest({"npcs": [{"id": "adult_01"}, {"id": NPC}]})
        self.experiment.client = FakeNightClient(self.prepare_replies(disabled))
        return self.experiment.client

    def condition_replies(self):
        """Script all nine physical conditions, including eight then nine loaves and the separate load testimony."""
        conditions = [([], False, None, 12),
                      (["setblock -7 -57 -8 minecraft:stone",
                        "setblock -7 -58 -8 minecraft:lantern[hanging=true]"], True, None, 12),
                      (["setblock -7 -58 -8 minecraft:air",
                        "item replace entity envoy weapon.mainhand with minecraft:lantern"], True, None, 12),
                      (["clear envoy"], False, None, 12),
                      (["tp envoy -12.5 -60 -7.5 0 0"], True, None, 6),
                      (["tp envoy -6.5 -60 -7.5 0 0", "time set 1000"], True, None, 12),
                      (["give envoy minecraft:bread 8"], True, None, 12),
                      (["give envoy minecraft:bread 1"], True, LOAD, 12),
                      (["time set 18000"], False, None, 12)]
        replies = []
        for index, (commands, visible, load, distance) in enumerate(conditions):
            replies.extend(("command", {"command": command}, {}) for command in commands)
            sample = self.sample_replies(6020 + index * 20, visible, load, distance)
            replies.extend(sample)
            if index == 7:
                replies.append(("ping", None, {"gameTime": 6180}))
                replies.append(deepcopy(sample[-1]))
        return replies

    def segment_replies(self, start):
        """Require the same three lamp transitions, twenty-tick samples, testimony windows and hashes; the lantern's
        support was placed by the conditions and stays (vanilla setblock fails when nothing changes)."""
        replies = []
        transitions = [("setblock -7 -58 -8 minecraft:lantern[hanging=true]",),
                       ("setblock -7 -58 -8 minecraft:air",),
                       ("setblock -7 -58 -8 minecraft:lantern[hanging=true]",)]
        for index, commands in enumerate(transitions):
            replies.extend(("command", {"command": command}, {}) for command in commands)
            replies.extend(self.sample_replies(start + index * 20, index != 1, LOAD))
            replies.append(("hash", {"region": self.experiment.region}, {"total": f"hash-{start + index * 20 + 20}"}))
        return replies


class NightProbeConstructionTest(NightProbeTest):
    """Both import modes and the real constructor must remain entirely server-free."""

    def test_constructor_loads_real_manifest_region_and_ports_without_creating_a_session(self):
        """Validate the actual fixture schema while forbidding server and session construction."""
        with patch.object(probe, "VillageSession", side_effect=AssertionError("session started")), \
                patch.object(harness, "Server", side_effect=AssertionError("server started")):
            experiment = probe.IT22NightExperiment(self.args)
        self.assertEqual(experiment.options["port"], 28000)
        self.assertEqual(experiment.options["game_port"], 28001)
        self.assertEqual(experiment.region, experiment.manifest.data["layout"]["fixture_hash_region"])
        self.assertTrue(experiment.manifest.data["npcs"])
        self.assertEqual(experiment.directory, self.root / "it22_night")
        self.assertEqual(experiment.snapshot, self.root / "it22_night_fork")
        self.assertIsNone(experiment.client)
        self.assertEqual((experiment.npc, experiment.origin, experiment.checks), ("", [], []))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_imports_work_as_a_package_and_as_a_standalone_script_without_running_main(self):
        """Use script-style import aliases without invoking the server-owning main entry point."""
        aliases = {"harness": harness, "test_it15_skills": test_it15_skills, "village_session": village_session}
        with patch.dict(sys.modules, aliases), \
                patch.object(village_session, "VillageSession", side_effect=AssertionError("session started")), \
                patch.object(harness, "Server", side_effect=AssertionError("server started")):
            for package in ("scripts", ""):
                with self.subTest(package=package):
                    namespace = {"__name__": "night_probe_import", "__package__": package,
                                 "__file__": str(probe.ROOT / "scripts/test_it22_night.py")}
                    source = Path(namespace["__file__"]).read_text(encoding="utf-8")
                    exec(compile(source, namespace["__file__"], "exec"), namespace)
                    self.assertIs(namespace["harness"], harness)
                    self.assertEqual(namespace["IT22NightExperiment"](self.args).region, self.experiment.region)

    def test_invalid_port_allocation_is_rejected_before_a_run(self):
        """The control and game listeners must fit the same eight-port allocation."""
        for control, game in ((0, None), (65529, None), (28000, 28000), (28000, 28008)):
            with self.subTest(control=control, game=game):
                self.args.port, self.args.game_port = control, game
                with self.assertRaises(ValueError):
                    probe.IT22NightExperiment(self.args)


class NightProbePreparationTest(NightProbeTest):
    """Preparation uses floored original NPC coordinates, settled bots and same-tick default-off comparisons."""

    def test_prepare_uses_manifest_data_and_floors_the_original_npc_position(self):
        """Do not round the negative path height or confuse the NPC's original position with settled bot feet."""
        client = self.use_prepare_client()
        self.experiment._prepare()
        self.assertFalse(client.replies)
        self.assertEqual(self.experiment.npc, NPC)
        self.assertEqual(self.experiment.origin, ORIGIN)
        self.assertEqual([check["passed"] for check in self.experiment.checks], [True, True])
        self.assertEqual(len([call for call in client.calls if call[0] == "step"]), 1)
        self.assertEqual(len([call for call in client.calls if call[0] == "observe"]), 3)

    def test_disabled_comparison_detects_changed_bytes_and_missing_target(self):
        """A changed field, reordered JSON keys or missing dark target cannot silently pass the compatibility check."""
        baseline = observation(6020)
        reordered = dict(reversed(list(baseline.items())))
        for disabled, expected in (({**baseline, "extra": True}, [False, True]),
                                   (reordered, [False, True]), (observation(6020, False), [False, False])):
            with self.subTest(expected=expected, disabled=disabled):
                self.experiment.checks.clear()
                self.experiment.npc = ""
                client = self.use_prepare_client(disabled)
                self.experiment._prepare()
                self.assertFalse(client.replies)
                self.assertEqual([check["passed"] for check in self.experiment.checks], expected)

    def test_no_stationary_npc_fails_before_altering_the_world(self):
        """An empty or entirely travelling NPC population must not create a test corridor."""
        for definitions in ([], [{"id": NPC}]):
            with self.subTest(definitions=definitions):
                self.experiment.manifest = ScenarioManifest({"npcs": definitions})
                replies = [("npc_state", {"id": NPC}, {"trip": {"to": "work"}})] if definitions else []
                client = FakeNightClient(replies)
                self.experiment.client = client
                with self.assertRaisesRegex(ValueError, "no dwelling villager"):
                    self.experiment._prepare()
                self.assertFalse(client.replies)


class NightProbeSamplingTest(NightProbeTest):
    """Sight checks compare the same interval and distinguish bot, NPC and missing-testimony failures."""

    def test_sample_checks_bot_and_latest_npc_record_independently(self):
        """Both expected visibility values and either observer's disagreement must be reflected in the checks."""
        for expected, bot_visible, npc_visible in ((False, False, False), (True, True, True),
                                                  (True, False, True), (False, False, True)):
            with self.subTest(expected=expected, bot=bot_visible, npc=npc_visible):
                self.experiment.checks.clear()
                replies = self.sample_replies(6020, bot_visible)
                records = [{"entities": observation(6030, not npc_visible)["entities"]},
                           {"entities": observation(6040, npc_visible)["entities"]}]
                replies[-1][2]["records"] = records
                client = self.attach(replies)
                result = self.experiment._sample("test condition", expected)
                self.assertEqual(result, observation(6040, bot_visible))
                self.assertFalse(client.replies)
                self.assertEqual([check["passed"] for check in self.experiment.checks],
                                 [bot_visible == expected, True, npc_visible == expected])

    def test_missing_testimony_fails_sampling_instead_of_counting_as_invisible(self):
        """No villager record is a failed sample even when both expected and bot visibility are false."""
        replies = self.sample_replies(6020, False)
        replies[-1][2]["records"] = []
        client = self.attach(replies)
        self.experiment._sample("no records", False)
        self.assertFalse(client.replies)
        self.assertEqual([check["passed"] for check in self.experiment.checks], [True, False])


class NightProbeConditionsTest(NightProbeTest):
    """Every visibility condition and both count-free load sources contribute checks to the gate result."""

    def test_lamp_places_support_before_hanging_and_removes_only_the_lantern(self):
        """Vanilla block updates must not pop an unsupported lantern or remove its non-emitting support."""
        client = self.attach([
            ("command", {"command": "setblock -7 -57 -8 minecraft:stone"}, {}),
            ("command", {"command": "setblock -7 -58 -8 minecraft:lantern[hanging=true]"}, {}),
            ("command", {"command": "setblock -7 -58 -8 minecraft:air"}, {})])
        self.experiment._lamp("minecraft:lantern")
        self.experiment._lamp("minecraft:air")
        self.assertFalse(client.replies)

    def test_conditions_check_all_nine_samples_and_four_load_assertions(self):
        """Exercise the real condition script, commands and sack renderer without an engine."""
        client = self.attach(self.condition_replies())
        self.experiment._conditions()
        self.assertFalse(client.replies)
        self.assertEqual(len(self.experiment.checks), 31)
        self.assertTrue(all(check["passed"] for check in self.experiment.checks))
        self.assertEqual([call for call in client.calls if call[0] == "step"], [("step", {"n": 20})] * 9)
        self.assertEqual([call for call in client.calls if call[0] == "command" and
                          call[1]["command"].startswith("give ")],
                         [("command", {"command": "give envoy minecraft:bread 8"}),
                          ("command", {"command": "give envoy minecraft:bread 1"})])

    def test_load_checks_reject_early_sacks_missing_sacks_and_exact_count_fields(self):
        """The threshold and count-free shape must be checked for the bot and villager independently."""
        cases = [("eight bread is not a sack", "observe", 6, LOAD),
                 ("nine bread reveals only a sack", "observe", 7, None),
                 ("nine bread reveals only a sack", "observe", 7, {**LOAD, "count": 9}),
                 ("testimony records the same count-free load", "npc_testimony", 8, {**LOAD, "count": 9})]
        for label, command, occurrence, load in cases:
            with self.subTest(label=label, load=load):
                self.experiment.checks.clear()
                self.experiment._support_placed = False  # each run starts from a fresh clone
                replies = self.condition_replies()
                result = [result for name, args, result in replies if name == command][occurrence]
                entities = result["entities"] if command == "observe" else result["records"][-1]["entities"]
                target = probe.IT22NightExperiment._target(entities)
                if load is None:
                    target.pop("load", None)
                else:
                    target["load"] = deepcopy(load)
                client = self.attach(replies)
                self.experiment._conditions()
                self.assertFalse(client.replies)
                checks = {check["check"]: check["passed"] for check in self.experiment.checks}
                self.assertFalse(checks[label])
                if load is None:
                    self.assertFalse(checks["prompt renders the sack"])


class NightProbeSegmentTest(NightProbeTest):
    """A replay must preserve all three observations and complete testimony, not only world hashes."""

    def test_segment_repeats_the_exact_requests_and_captures_all_evidence(self):
        """Restoring the starting tick produces the same twenty-request script and sixty-tick evidence."""
        runs = []
        traces = []
        self.experiment._support_placed = True  # as after the conditions
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                client = self.attach(self.segment_replies(6200))
                runs.append(self.experiment._segment())
                traces.append(client.calls)
                self.assertFalse(client.replies)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(len(traces[0]), 18)
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0]), 3)
        self.assertEqual([sample["observation"]["gameTime"] for sample in runs[0]], [6220, 6240, 6260])
        self.assertEqual([sample["hash"] for sample in runs[0]], ["hash-6220", "hash-6240", "hash-6260"])
        for sample in runs[0]:
            self.assertEqual(set(sample), {"hash", "observation", "testimony"})
            self.assertEqual(sample["testimony"]["records"][0]["entities"], sample["observation"]["entities"])
        self.assertEqual(self.experiment.checks, [])


class NightProbeRunTest(NightProbeTest):
    """Replace only session ownership and filesystem cloning while running all real probe methods."""

    def run_probe(self, mismatch=None):
        """Run factual and restored scripted clients and read the genuine report without starting any servers."""
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        self.args.runs_dir = Path(temporary.name)
        self.experiment = probe.IT22NightExperiment(self.args)
        self.experiment.manifest = ScenarioManifest({"npcs": [{"id": "adult_01"}, {"id": NPC}]})
        factual_replies = self.prepare_replies() + self.condition_replies()
        factual_replies.extend([("save_snapshot", None, {}), ("ping", None, {"gameTime": 6200})])
        factual_replies.extend(self.segment_replies(6200))
        replay_start = 6201 if mismatch == "tick" else 6200
        replay_replies = [("ping", None, {"gameTime": replay_start})] + self.segment_replies(replay_start)
        if mismatch in {"observe", "npc_testimony", "hash"}:
            result = next(result for command, args, result in replay_replies if command == mismatch)
            if mismatch == "hash":
                result["total"] = "different-hash"
            elif mismatch == "observe":
                result["entities"].pop()
            else:
                result["records"][0]["entities"].pop()
        clients = deque([FakeNightClient(factual_replies), FakeNightClient(replay_replies)])
        sessions = []
        events = []

        class OfflineSession:
            """Own temporary paths and an already scripted client, with no Minecraft or real clone implementation."""

            def __init__(self, directory, source, manifest, options, *, fresh=True):
                """Retain constructor arguments for assertions about fixture and fork wiring."""
                self.directory, self.source, self.manifest, self.options = directory, source, manifest, options
                self.fresh = fresh
                self.client = clients.popleft()
                sessions.append(self)

            def clone(self):
                """Create only the empty directories the actual report and snapshot code require."""
                events.append(("clone", self.directory.name))
                (self.directory / "world").mkdir(parents=True)

            def __enter__(self):
                """Expose the scripted client without creating a server."""
                events.append(("enter", self.directory.name))
                return self

            def __exit__(self, *exc):
                """Record that the factual session is closed before the resumed one begins."""
                events.append(("exit", self.directory.name))

        def snapshot(source, destination):
            """Stand in for the one requested APFS clone, preserving its source/destination relationship."""
            self.assertEqual(source, self.experiment.directory / "world")
            self.assertEqual(destination, self.experiment.snapshot / "world")
            events.append(("snapshot", self.experiment.snapshot.name))
            destination.mkdir(parents=True)

        with patch.object(probe, "VillageSession", OfflineSession), patch.object(harness, "clone_apfs", snapshot), \
                patch.object(harness, "Server", side_effect=AssertionError("server started")):
            passed = self.experiment.run()
        for session in sessions:
            self.assertFalse(session.client.replies)
        self.assertFalse(clients)
        self.assertEqual(events, [("clone", "it22_night"), ("enter", "it22_night"),
                                  ("snapshot", "it22_night_fork"), ("exit", "it22_night"),
                                  ("clone", "it22_night_resumed"), ("enter", "it22_night_resumed"),
                                  ("exit", "it22_night_resumed")])
        self.assertEqual([session.fresh for session in sessions], [True, False])
        self.assertEqual([session.source for session in sessions],
                         [self.args.runs_dir / "village_sabotage", self.experiment.snapshot])
        self.assertTrue(all(session.manifest is self.experiment.manifest for session in sessions))
        self.assertTrue(all(session.options == self.experiment.options for session in sessions))
        self.assertTrue((self.experiment.snapshot / "orchestrator").is_dir())
        report = json.loads((self.experiment.directory / "it22_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report, {"passed": passed, "checks": self.experiment.checks})
        self.assertEqual(len(report["checks"]), 35)
        return report

    def test_full_script_counts_every_check_and_passes_exact_replay(self):
        """All default-off, nine-sample, four-load, restored-tick and exact-evidence checks must pass."""
        report = self.run_probe()
        self.assertTrue(report["passed"])
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        self.assertEqual(report["checks"][-1]["check"],
                         "short night replay: observations, testimony and hashes match exactly")

    def test_replay_rejects_any_evidence_or_restored_tick_mismatch(self):
        """Identical hashes alone cannot mask differing bot observations or villager testimony."""
        for mismatch in ("observe", "npc_testimony", "hash", "tick"):
            with self.subTest(mismatch=mismatch):
                report = self.run_probe(mismatch)
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"][-1]["passed"])
                self.assertEqual(report["checks"][-2]["passed"], mismatch != "tick")


if __name__ == "__main__":
    unittest.main()
