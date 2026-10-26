"""Pin IT-11's prompt-driven policy, rule-removal control and scorer checks without starting an engine."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import ENGINE_RULES, GameMaster, NarratedRuntime, narration_error
from scripts import harness
from scripts.agent_scenario import WorkshopScenario
from scripts.test_it11_narration import NarrationExperiment, PLANS, classification, planned_action, planted_checks
from scripts.test_it11_narration import main, scripted_ruling


def scored_acts() -> tuple[list[dict], list[dict]]:
    """Supply scorer-shaped records for each documented real-engine case, including prior speech compression."""
    acts = [
        {"agent": "ana", "tool": "speak", "args": {}, "gameTime": 0,
         "gm": {"accepted": True, "duration_ticks": 0}, "engine": "started", "engine_final": "running",
         "engine_check": {"feasible": True, "ticks": 400}},
        {"agent": "ben", "tool": "mine", "args": {}, "gameTime": 0,
         "gm": {"accepted": True, "duration_ticks": 0}, "engine": "not possible: out_of_reach"},
        {"agent": "cy", "tool": "wait", "args": {"ticks": 80}, "gameTime": 0,
         "gm": {"accepted": True, "duration_ticks": 80}, "engine": "started", "engine_final": "succeeded",
         "engine_check": {"feasible": True, "ticks": 80}},
        {"agent": "ana", "tool": "move_to", "args": {}, "gameTime": 40,
         "gm": {"accepted": True, "duration_ticks": 0}, "engine": "not possible: busy"},
        {"agent": "ben", "tool": "place", "args": {}, "gameTime": 40,
         "gm": {"accepted": True, "duration_ticks": 0}, "engine": "not possible: missing_items"},
        {"agent": "cy", "tool": "ask", "args": {"npc": "ben"}, "gameTime": 80,
         "gm": {"accepted": False, "duration_ticks": 0},
         "engine_check": {"feasible": False, "reason": "target_not_found", "ticks": 0}},
        {"agent": "cy", "tool": "wait", "args": {"ticks": 200}, "gameTime": 120,
         "gm": {"accepted": False, "duration_ticks": 0}, "engine_check": {"feasible": True, "ticks": 200}},
    ]
    control = [{"agent": "cy", "tool": "ask", "args": {"npc": "ben"}, "gameTime": 160,
                "gm": {"accepted": True, "duration_ticks": 0}, "engine": "not possible: target_not_found"}]
    return acts, control


class NarrationControlTest(unittest.TestCase):
    """Rules are read from GameMaster's actual system prompt and errors are scored by production narration_error."""

    def test_removing_only_engine_rules_changes_the_same_ask_ruling(self):
        """Normal rules reject an agent; an otherwise identical GameMaster without those rules accepts it."""
        for rules, expected in ((True, False), (False, True)):
            with self.subTest(rules=rules):
                gm = GameMaster(ScriptedClient(scripted_ruling), "People: ana, ben, cy.", {"cy": "By the market."})
                if not rules:
                    gm._system = gm._system.replace("\n\n" + ENGINE_RULES, "")
                verdict = gm.resolve(80, "cy", ToolCall("ask", {"npc": "ben"}))
                self.assertIs(verdict.accepted, expected)
                self.assertEqual(gm.calls, 1)

    def test_the_prompt_driven_schedule_reaches_every_planted_case(self):
        """The real narrated loop emits exactly the planned faults, despite GM waiting and actor-local notes."""
        with tempfile.TemporaryDirectory() as temporary:
            scenario = WorkshopScenario()
            gm = GameMaster(ScriptedClient(scripted_ruling), scenario.world_description(),
                            {name: f"You are {name} at (0, -60, 0)." for name in scenario.BOTS})
            runtime = NarratedRuntime(scenario.agents(), ScriptedClient(planned_action), gm, Path(temporary))
            runtime.run(0, 160)
            for name, plan in PLANS.items():
                self.assertEqual([act["tool"] for act in runtime.acts if act["agent"] == name],
                                 [call.name for call in plan])
            self.assertEqual([act["gameTime"] for act in runtime.acts if act["agent"] == "cy"], [0, 80, 120])

    def test_scorer_checks_require_every_case_and_exact_reason(self):
        """Missing acts, changed engine verdicts and absent controls all fail instead of passing vacuously."""
        acts, control = scored_acts()
        self.assertTrue(all(planted_checks(acts, control).values()))
        summary = narration_error(acts)
        self.assertEqual((summary["infeasible_first"], summary["infeasible_downstream"]), (1, 2))
        self.assertEqual(summary["false_rejections"], 1)
        self.assertEqual(classification(acts, 3, "infeasible_downstream"), 1)
        self.assertFalse(all(planted_checks(acts, []).values()))
        for index in range(len(acts)):
            with self.subTest(missing=index):
                self.assertFalse(all(planted_checks(acts[:index] + acts[index + 1:], control).values()))
        changed = deepcopy(acts)
        changed[1]["engine"] = "not possible: target_not_found"
        self.assertFalse(all(planted_checks(changed, control).values()))

    def test_ports_stay_inside_the_gate_allocation(self):
        """Defaults use the next game port and explicit out-of-allocation or overlapping ports are refused."""
        args = harness.experiment_parser("").parse_args(["--port", "25700"])
        self.assertEqual(NarrationExperiment(args).options["game_port"], 25701)
        for port in (25699, 25700, 25708):
            with self.subTest(port=port):
                args.game_port = port
                with self.assertRaises(ValueError):
                    NarrationExperiment(args)
        args.game_port = None
        for port in (0, -1, 65529, 65536):
            with self.subTest(base=port):
                args.port = port
                with self.assertRaises(ValueError):
                    NarrationExperiment(args)

    def test_invalid_options_print_a_failure_verdict_without_cloning(self):
        """A rejected socket allocation prints BAD and FAIL, returns one, and leaves the run root untouched."""
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            status = main(["--runs-dir", temporary, "--port", "25700", "--game-port", "25708"])
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(status, 1)
        self.assertIn("BAD", output.getvalue())
        self.assertIn("FAIL IT-11:", output.getvalue())
