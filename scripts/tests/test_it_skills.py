"""Offline IT-15 checks for exact path footing, action feedback, runtime rest and bounded CLI options.

The existing FakeWorld exercises runtime plumbing only, never claims to simulate village pathfinding or item
physics. The owner still runs the real gate and its scratch-branch feet-position negative control separately.
"""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest

from orchestrator.decisions import Decision
from orchestrator.llm import ToolCall
from orchestrator.runtime import NIGHT_REST, POLL, failure_hint
from scripts import harness
from scripts.build_village import ScenarioConfig
from scripts.test_it15_skills import (
    IT15SkillsExperiment,
    OVERLONG_SPEECH,
    ROAD_CELLS,
    WAKE_TEXT,
    feedback_checks,
    inventory_delta,
    main,
    on_dirt_path,
    port_options,
    rest_checks,
)
from scripts.tests.fake_world import FakeWorld


def arguments(*values: str):
    """Parse actual shared CLI options without creating a session or opening either listener."""
    return harness.experiment_parser("IT-15 unit tests").parse_args(list(values))


def decision(tick: int, call: ToolCall | None = None, *, prompt: str = "", outcome: str = "") -> Decision:
    """Create a genuine immutable decision record without inventing a separate fake log interface."""
    return Decision(tick, "builder", "unused", False, "model", call.name if call else None,
                    call.args if call else None, "scripted", 0, 0, prompt, outcome)


def feedback_pair(call: ToolCall, raw: str, observation: dict | None = None) -> tuple[Decision, Decision]:
    """Supply one recorded failure and the exact next-prompt action line for focused checker tests."""
    outcome = raw + failure_hint(call, raw, observation or {})
    first = decision(0, call, outcome=outcome)
    prompt = f"{call.name} {json.dumps(call.args, sort_keys=True)} -> {outcome}"
    return first, decision(120, prompt=prompt)


class IT15PortsTests(unittest.TestCase):
    """Keep all game/control listeners inside the gate runner's allocated eight ports."""

    def test_default_game_port_is_control_plus_one_not_25565(self):
        """Omitting --game-port must never inherit the harness's global Minecraft port."""
        options = port_options(arguments("--port", "30100"))
        self.assertEqual(options["port"], 30100)
        self.assertEqual(options["game_port"], 30101)

    def test_explicit_allocated_ports_preserve_other_server_options(self):
        """Every distinct in-allocation game port is accepted without losing seed, heap or timeouts."""
        for offset in range(1, 8):
            with self.subTest(offset=offset):
                args = arguments("--port", "30100", "--game-port", str(30100 + offset), "--heap", "2G",
                                 "--base-seed", "17", "--startup-timeout", "11", "--request-timeout", "12")
                options = port_options(args)
                self.assertEqual(options, {**harness.server_options(args), "game_port": 30100 + offset})

    def test_colliding_and_outside_game_ports_are_rejected(self):
        """Both the default 25565 trap and boundary/collision mistakes fail before any fixture is cloned."""
        for game_port in (25565, 30099, 30100, 30108, 65536, -1):
            with self.subTest(game_port=game_port), self.assertRaisesRegex(ValueError, "--game-port"):
                port_options(arguments("--port", "30100", "--game-port", str(game_port)))

    def test_control_port_reserves_a_valid_eight_port_range(self):
        """Reject invalid TCP allocations, including an overflow near the top of the port range."""
        for port in (0, -1, 65529, 65536):
            with self.subTest(port=port), self.assertRaisesRegex(ValueError, "--port"):
                port_options(arguments("--port", str(port)))
        self.assertEqual(port_options(arguments("--port", "65528", "--game-port", "65535"))["game_port"], 65535)

    def test_invalid_cli_ports_print_bad_and_fail_without_starting(self):
        """Match IT-04's failure contract instead of returning argparse's success-looking help or a traceback."""
        output = io.StringIO()
        previous = signal.getsignal(signal.SIGTERM)
        try:
            with contextlib.redirect_stdout(output):
                result = main(["--port", "30100", "--game-port", "25565"])
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(result, 1)
        self.assertIn("BAD", output.getvalue())
        self.assertIn("FAIL IT-15:", output.getvalue())
        self.assertNotIn("PASS IT-15:", output.getvalue())

    def test_existing_run_is_preserved_and_reported_as_failure(self):
        """A collision fails in VillageSession.clone before a fixture read or any possible server startup."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "it15_skills"
            directory.mkdir()
            marker = directory / "keep.txt"
            marker.write_text("another run", encoding="utf-8")
            output = io.StringIO()
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with contextlib.redirect_stdout(output):
                    result = main(["--runs-dir", temporary])
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(result, 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "another run")
            self.assertIn("Refusing to overwrite", output.getvalue())
            self.assertIn("FAIL IT-15:", output.getvalue())


class IT15RoadTests(unittest.TestCase):
    """Make a road regression impossible to hide behind gravel, integer-height teleports or no movement."""

    def test_selected_corridor_is_not_a_builder_gravel_crosswalk(self):
        """The actual village generator leaves every selected road cell as dirt_path, not its collar or crossings."""
        config = ScenarioConfig.create("sabotage")
        road = next(road for road in config.roads if road["name"] == "west_spine")
        left, north, right, south = road["bounds"]
        for depth in range(ROAD_CELLS[0][1], ROAD_CELLS[-1][1] + 1):
            cell = (ROAD_CELLS[0][0], depth)
            self.assertTrue(left <= cell[0] <= right and north <= cell[1] <= south)
            self.assertNotIn(cell, config.pedestrian_cells())
        self.assertEqual(len(ROAD_CELLS), 3)
        self.assertTrue(all(abs(after[1] - before[1]) >= 8 for before, after in zip(ROAD_CELLS, ROAD_CELLS[1:])))

    def test_only_settled_fractional_feet_satisfy_the_path_precondition(self):
        """The old feet calculation floors -60.0625 into the solid path; the fixed calculation reaches foot air."""
        state = {"pos": [-11.5, -60.0625, 16.5], "onGround": True}
        self.assertTrue(on_dirt_path(state, ROAD_CELLS[0], -61))
        self.assertFalse(on_dirt_path({**state, "pos": [-11.5, -60, 16.5]}, ROAD_CELLS[0], -61))
        self.assertFalse(on_dirt_path({**state, "pos": [-12.5, -60, 16.5]}, ROAD_CELLS[0], -61))
        self.assertFalse(on_dirt_path({**state, "onGround": False}, ROAD_CELLS[0], -61))

    def test_wrong_cell_height_and_nonfinite_coordinates_fail(self):
        """An old location, nearby gravel height, malformed position or nonfinite coordinate cannot count as arrival."""
        for position in ([-11.5, -60.0625, 24.5], [-11.5, -60.1, 16.5], [float("nan"), -60.0625, 16.5],
                         [-11.5, float("inf"), 16.5], []):
            with self.subTest(position=position):
                self.assertFalse(on_dirt_path({"pos": position, "onGround": True}, ROAD_CELLS[0], -61))


class IT15FeedbackTests(unittest.TestCase):
    """Require actual final outcomes and failure hints in the same actor's immediate following prompt."""

    def test_every_planted_failure_has_a_nonempty_hint_in_the_next_prompt(self):
        """Cover missing drops, both absent-villager reasons, the 650-character speech and the missing bill."""
        cases = [
            (ToolCall("collect", {}), "not possible: target_not_found"),
            (ToolCall("ask", {"npc": "it15_absent"}), "not possible: target_not_found"),
            (ToolCall("ask", {"npc": "adult_01"}), "not possible: out_of_reach"),
            (ToolCall("speak", {"text": OVERLONG_SPEECH}),
             "failed: Speech needs nonempty text of at most 600 characters and talk/shout volume"),
            (ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [-6, -61, 0]}),
             "not possible: missing_items"),
        ]
        self.assertEqual(len(OVERLONG_SPEECH), 650)
        for call, raw in cases:
            with self.subTest(tool=call.name, raw=raw):
                first, following = feedback_pair(call, raw)
                self.assertTrue(all(feedback_checks(first, following, first.outcome, {}).values()))

    def test_started_is_not_a_completed_skill_outcome(self):
        """Checking just a started action would miss a later no_path or failed transfer."""
        call = ToolCall("move_to", {"x": -12, "y": -60, "z": 24})
        first = decision(0, call, outcome="started")
        prompt = f"move_to {json.dumps(call.args, sort_keys=True)} -> succeeded"
        following = decision(80, prompt=prompt)
        self.assertTrue(all(feedback_checks(first, following, "succeeded", {}).values()))
        self.assertFalse(feedback_checks(first, following, "failed (no_path)", {})["outcome in next prompt"])
        self.assertFalse(feedback_checks(first, replace(following, user_prompt="started"), "succeeded", {})[
            "outcome in next prompt"])

    def test_bare_failure_or_hint_only_does_not_pass(self):
        """A reason code or hint fragment alone is not the actual action outcome returned to the agent."""
        first, following = feedback_pair(ToolCall("collect", {}), "not possible: target_not_found")
        for prompt in ("not possible: target_not_found", failure_hint(first.tool_call, first.outcome, {}), ""):
            with self.subTest(prompt=prompt):
                checks = feedback_checks(first, replace(following, user_prompt=prompt), first.outcome, {})
                self.assertFalse(checks["outcome in next prompt"])
                self.assertFalse(checks["failure_hint in next prompt"])

    def test_wrong_actor_stale_prompt_and_missing_runtime_hint_fail(self):
        """A correct fragment in another actor's prompt or an old turn cannot satisfy next-agent feedback."""
        first, following = feedback_pair(ToolCall("collect", {}), "not possible: target_not_found")
        for changed in (replace(following, agent="mayor"), replace(following, game_time=0)):
            checks = feedback_checks(first, changed, first.outcome, {})
            self.assertFalse(checks["outcome in next prompt"])
            self.assertFalse(checks["failure_hint in next prompt"])
        bare = replace(first, outcome="not possible: target_not_found")
        self.assertFalse(feedback_checks(bare, following, bare.outcome, {})["failure_hint in next prompt"])

    def test_missing_hint_and_too_short_or_too_long_thinking_delays_fail(self):
        """Unknown failure text must not pass vacuously, and a failure cannot immediately retry or stall forever."""
        first, following = feedback_pair(ToolCall("collect", {}), "not possible: unknown_reason")
        self.assertFalse(feedback_checks(first, following, first.outcome, {})["failure_hint in next prompt"])
        first, following = feedback_pair(ToolCall("collect", {}), "not possible: target_not_found")
        for tick in (40, 80, 160, 1200):
            with self.subTest(tick=tick):
                self.assertFalse(feedback_checks(first, replace(following, game_time=tick), first.outcome, {})[
                    "failure thinking time"])

    def test_speech_listener_feedback_requires_the_final_outcome(self):
        """Both who heard and nobody-heard messages must be in the speaker's next actual prompt."""
        call = ToolCall("speak", {"text": "IT15 hello"})
        first = decision(0, call, outcome="started")
        for ending in ("succeeded (heard by: mayor)", "succeeded (nobody was close enough to hear)"):
            with self.subTest(ending=ending):
                following = decision(40, prompt=f'speak {{"text": "IT15 hello"}} -> {ending}')
                self.assertTrue(feedback_checks(first, following, ending, {})["outcome in next prompt"])
                self.assertFalse(feedback_checks(first, replace(following, user_prompt="succeeded"), ending, {})[
                    "outcome in next prompt"])

    def test_inventory_feedback_sums_separate_construction_polls_and_items(self):
        """A six-plank bill can be spent over several polls alongside unrelated item changes."""
        prompt = ("  [day 0, 08:00] your inventory: -2 oak_planks, +1 oak_log\n"
                  "  [day 0, 08:01] your inventory: -4 oak_planks\n"
                  "Inventory: 12 oak_planks\n"
                  "  [day 0, 08:02] your inventory: +3 emerald")
        self.assertEqual(inventory_delta(prompt, "minecraft:oak_planks"), -6)
        self.assertEqual(inventory_delta(prompt, "minecraft:oak_log"), 1)
        self.assertEqual(inventory_delta(prompt, "minecraft:emerald"), 3)
        self.assertEqual(inventory_delta(prompt, "minecraft:oak"), 0)
        self.assertEqual(inventory_delta("Inventory: 6 oak_planks", "minecraft:oak_planks"), 0)

    def test_real_runtime_exercise_checks_final_success_and_rejected_actions(self):
        """Use the one existing FakeWorld with the actual experiment/runtime rather than a mock framework."""
        cases = [
            (ToolCall("move_to", {"x": 4, "y": -60, "z": 0}), {}, "succeeded"),
            (ToolCall("collect", {}), {"collect": "target_not_found"}, "not possible: target_not_found"),
            (ToolCall("ask", {"npc": "adult_01"}), {"ask": "out_of_reach"}, "not possible: out_of_reach"),
            (ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [-6, -61, 0]}),
             {"place_section": "missing_items"}, "not possible: missing_items"),
        ]
        for call, rejects, expected in cases:
            with self.subTest(tool=call.name), tempfile.TemporaryDirectory() as temporary:
                experiment = IT15SkillsExperiment(arguments("--runs-dir", temporary))
                experiment.client = FakeWorld({"builder": [0.5, -60, 0.5]}, rejects=rejects,
                                              skill_ticks={"move_to": 80})
                with contextlib.redirect_stdout(io.StringIO()):
                    prompt = experiment._exercise("case", call, expected)
                self.assertTrue(all(check["passed"] for check in experiment.checks), experiment.checks)
                if expected == "succeeded":
                    self.assertEqual(experiment.last_status["state"], "succeeded")
                    self.assertEqual(experiment.last_status["skill"], call.name)
                self.assertIn(expected, prompt)
                records = [json.loads(line) for line in
                           (experiment.directory / "orchestrator/case/decisions.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertEqual(records[0]["tool"], call.name)
                self.assertIsNone(records[1]["tool"])
                self.assertEqual(records[1]["game_time"], 80 if expected == "succeeded" else 120)

    def test_unexpected_engine_failure_fails_outcome_and_next_prompt_checks(self):
        """A logged no_path hint cannot make a road walk pass, even when an incremental prompt omits it."""
        with tempfile.TemporaryDirectory() as temporary:
            experiment = IT15SkillsExperiment(arguments("--runs-dir", temporary))
            experiment.client = FakeWorld({"builder": [0.5, -60, 0.5]}, rejects={"move_to": "no_path"})
            with contextlib.redirect_stdout(io.StringIO()):
                experiment._exercise("road", ToolCall("move_to", {"x": 4, "y": -60, "z": 0}))
            outcomes = {check["check"]: check["passed"] for check in experiment.checks}
            self.assertFalse(outcomes["road: expected outcome"])
            self.assertFalse(outcomes["road: outcome in next prompt"])
            self.assertFalse(outcomes["road: failure_hint in next prompt"])


class IT15MiningProbeTests(unittest.TestCase):
    """Probe diagnostics use bounded pickup-adjacent cells without advancing time or starting skills."""

    def test_drop_paths_are_dry_runs_to_nearby_standing_cells(self):
        """FakeWorld checks request wiring only; real IT-15 establishes standability and pickup physics."""
        experiment = IT15SkillsExperiment(arguments())
        experiment.client = FakeWorld({"builder": [-106.5, -60, 8.5]})
        routes = experiment._drop_paths([-108.1, -60.0, 8.9])
        self.assertEqual([route["pos"] for route in routes if route["pos"][1] == -60],
                         [[-108.5, -60, 8.5], [-108.5, -60, 9.5], [-107.5, -60, 8.5], [-107.5, -60, 9.5]])
        self.assertEqual({route["pos"][1] for route in routes}, {-60, -59, -61, -62, -63})
        self.assertTrue(all(route["feasible"] for route in routes))
        self.assertTrue(all(command == "skill_start" and args["dry_run"] is True and args["skill"] == "move_to"
                            for command, args in experiment.client.requests))
        self.assertEqual(experiment.client.skills["builder"], [])
        self.assertEqual(experiment.client.game_time, 0)


class IT15RestTests(unittest.TestCase):
    """Pin nonvacuous idle rest, the spoken-to exception and the first dawn poll without a sleeping wait skill."""

    def records(self) -> list[Decision]:
        """Return one daytime idle decision, a correctly heard night wake and the first dawn decision."""
        return [decision(15_960), decision(18_440, prompt=f'  [day 0, 00:24] mayor (2 blocks away): "{WAKE_TEXT}"'),
                decision(24_000)]

    def test_complete_idle_night_with_one_heard_wake_passes(self):
        """Both endpoints and exactly one speech-backed decision are necessary to establish night coverage."""
        self.assertTrue(all(rest_checks(self.records(), 16_000, 18_440).values()))

    def test_empty_missing_wake_or_unspoken_night_decision_fails(self):
        """Silence from a missing runtime or a long-running wait is not proof of production rest behavior."""
        records = self.records()
        for changed in ([], records[1:], records[:-1], [records[0], records[-1]],
                        [records[0], decision(16_120), *records[1:]],
                        [records[0], replace(records[1], user_prompt="unrelated speech"), records[-1]]):
            with self.subTest(changed=changed):
                self.assertFalse(all(rest_checks(changed, 16_000, 18_440).values()))

    def test_a_wait_skill_does_not_count_as_idle_night_coverage(self):
        """The pre-night decision must choose no action, not a skill that would suppress decisions anyway."""
        records = self.records()
        records[0] = replace(records[0], tool="wait", args={"ticks": 1200})
        self.assertFalse(rest_checks(records, 16_000, 18_440)["idle before rest"])

    def test_exact_dawn_is_outside_rest_and_later_days_use_absolute_ticks(self):
        """Use a half-open night window, independent of which day the action cases finish on."""
        records = [replace(record, game_time=record.game_time + 48_000) for record in self.records()]
        self.assertTrue(all(rest_checks(records, 64_000, 66_440).values()))
        records.insert(1, decision(64_000))
        self.assertFalse(rest_checks(records, 64_000, 66_440)["no night decisions unless spoken to"])

    def test_experiment_runtime_rests_wakes_and_resumes_without_wait_skills(self):
        """Exercise the actual agent's prompts and production timing settings for an entire fake-world night."""
        with tempfile.TemporaryDirectory() as temporary:
            experiment = IT15SkillsExperiment(arguments("--runs-dir", temporary))
            world = FakeWorld({"builder": [100.5, -60, 100.5]}, game_time=NIGHT_REST[0] - POLL)
            experiment.client = world

            def idle(system: str, user: str) -> None:
                """Keep the runtime truly idle before, during and after the rest window."""
                return None

            runtime = experiment._runtime("night", idle)
            runtime.run(18_400)
            self.assertEqual([record.game_time for record in runtime.log.decisions()], [15_960])
            world.hear("builder", "mayor", WAKE_TEXT)
            runtime.run(24_040)
            self.assertTrue(all(rest_checks(runtime.log.decisions(), 16_000, 18_400).values()))
            self.assertFalse(any(command == "skill_start" for command, args in world.requests))

    def test_completed_skill_status_is_not_an_active_night_wait(self):
        """The engine retains last-completed status rather than returning idle after construction finishes."""
        world = FakeWorld({"builder": [100.5, -60, 100.5]})
        world.request("skill_start", {"name": "builder", "skill": "wait", "args": {"ticks": 40}})
        world.request("step", {"n": 40})
        state = world.request("skill_status", {"name": "builder"})["state"]
        self.assertEqual(state, "succeeded")
        self.assertIn(state, {"idle", "succeeded", "failed", "cancelled"})


if __name__ == "__main__":
    unittest.main()
