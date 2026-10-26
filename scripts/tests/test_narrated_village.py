"""Test the naive village GM and safe scripted setup without launching a village or model server."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import GameMaster, Resolution
from scripts.run_narrated_village import NaiveVillageGM, NarratedVillage, main


class NaiveVillageGMTest(unittest.TestCase):
    """The naive policy parses GameMaster's actual prompts and broadcasts only speech."""

    def setUp(self):
        """Connect the real naive policy through ScriptedClient to a three-person GameMaster."""
        self.views = {"alice": "Beside the bell.\nInventory:\n  two loaves and a map.",
                      "ben": "In the shop.", "cy": "At the market."}
        self.gm = GameMaster(ScriptedClient(NaiveVillageGM(list(self.views)).resolve), "A village.", self.views)

    def test_speech_is_instant_preserves_multiline_view_and_is_heard_by_every_other_person(self):
        """The actual speech prompt preserves the actor's multiline view and delivers exact words to everyone else."""
        text = 'Meet by the bell.\nBring the map marked {"route": [1, 2]} and say "hello".'
        resolution = self.gm.resolve(80, "alice", ToolCall("speak", {"text": text, "mode": "talk",
                                                                   "note": "A private planning note."}))
        self.assertEqual(resolution, Resolution(True, "", 0, ["alice did speak"], self.views["alice"],
                                               {"ben": text, "cy": text}))
        self.assertEqual({person: self.gm.view(person) for person in self.views}, self.views)
        self.assertEqual(self.gm.calls, 1)

    def test_non_speech_with_nested_lists_preserves_view_and_emits_no_heard_observations(self):
        """Non-speech prompts with nested lists parse after prior acts and never produce heard observations."""
        calls = [ToolCall("wait", {"ticks": 40}),
                 ToolCall("build_section", {"schematic": "bridge_span", "origin": [1, 64, 2],
                                            "materials": [["oak_planks", [1, 2]], ["stone", []]]}),
                 ToolCall("write_book", {"slot": 2, "title": "Routes", "pages": ["First\nsecond", "[1, 2]"]})]
        for tick, call in enumerate(calls):
            with self.subTest(tool=call.name):
                resolution = self.gm.resolve(tick * 40, "alice", call)
                self.assertEqual(resolution, Resolution(True, "", 0, [f"alice did {call.name}"],
                                                       self.views["alice"], {}))
                self.assertEqual({person: self.gm.view(person) for person in self.views}, self.views)
        self.assertEqual(self.gm.calls, len(calls))


class NarratedVillageScriptedTest(unittest.TestCase):
    """Scripted setup uses the village policy and refuses existing run directories before startup."""

    def test_main_scripted_rejects_zero_acts_or_gm_calls_and_prints_each_reason(self):
        """Scripted runs require both narrated acts and GM calls, with all empty counters explained."""
        for acts, gm_calls in ((0, 1), (1, 0), (0, 0)):
            report = {"acts": acts, "gm_calls": gm_calls, "stopped": None}
            with self.subTest(acts=acts, gm_calls=gm_calls), \
                    patch.object(NarratedVillage, "run", return_value=report), \
                    patch("scripts.run_narrated_village.harness.stop_on_sigterm"), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--scripted"]), 1)
            printed = output.getvalue()
            self.assertEqual("acts == 0" in printed, acts == 0)
            self.assertEqual("gm_calls == 0" in printed, gm_calls == 0)
            self.assertIn(json.dumps(report, indent=1), printed)

    def test_main_scripted_rejects_early_stop_even_after_activity(self):
        """Budget exhaustion fails a scripted run even when turns and GM calls already occurred."""
        for stopped in ("Model-call budget 4 would be exceeded at tick 6080", ""):
            report = {"acts": 2, "gm_calls": 2, "stopped": stopped}
            with self.subTest(stopped=stopped), patch.object(NarratedVillage, "run", return_value=report), \
                    patch("scripts.run_narrated_village.harness.stop_on_sigterm"), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--scripted"]), 1)
            self.assertIn(f"FAIL VG5-scripted: stopped early: {stopped}", output.getvalue())

    def test_main_scripted_accepts_a_completed_run_with_activity(self):
        """A nonempty scripted run without an early stop still succeeds and prints its report."""
        report = {"acts": 1, "gm_calls": 1, "stopped": None}
        with patch.object(NarratedVillage, "run", return_value=report), \
                patch("scripts.run_narrated_village.harness.stop_on_sigterm"), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--scripted"]), 0)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_main_live_keeps_success_status_for_empty_or_early_stopped_runs(self):
        """Non-scripted exit status and JSON output remain unchanged for empty or stopped reports."""
        for acts, gm_calls, stopped in ((0, 0, None), (0, 1, None), (1, 0, None), (2, 2, "budget exhausted")):
            report = {"acts": acts, "gm_calls": gm_calls, "stopped": stopped}
            with self.subTest(report=report), patch.object(NarratedVillage, "run", return_value=report), \
                    patch("scripts.run_narrated_village.harness.stop_on_sigterm"), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([]), 0)
            self.assertEqual(json.loads(output.getvalue()), report)

    def test_main_scripted_refuses_existing_default_and_explicit_names_without_changing_files(self):
        """The scripted entry point raises FileExistsError for existing runs without cloning or altering their files."""
        for name, options in (("narrated_scripted_sabotage", []), ("already_narrated", ["--name", "already_narrated"])):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                runs_dir = Path(directory)
                run_dir = runs_dir / name
                run_dir.mkdir()
                marker = run_dir / "keep.txt"
                marker.write_text("Do not replace this run.\n", encoding="utf-8")
                with self.assertRaises(FileExistsError) as caught:
                    main(["--scripted", "--runs-dir", str(runs_dir), *options])
                self.assertEqual(str(caught.exception), f"Refusing to overwrite {run_dir}")
                self.assertEqual(marker.read_text(encoding="utf-8"), "Do not replace this run.\n")
                self.assertEqual(list(runs_dir.iterdir()), [run_dir])
                self.assertEqual(list(run_dir.iterdir()), [marker])

    def test_scripted_models_use_village_actions_and_a_naive_gm(self):
        """Scripted setup returns the village action policy and an instant, broadcasting GM named naive-gm."""
        village = NarratedVillage(argparse.Namespace(scripted=True, variant="sabotage"))
        agent_model, gm_model = village._models()
        self.assertIsInstance(agent_model, ScriptedClient)
        self.assertIsInstance(gm_model, ScriptedClient)
        self.assertEqual((agent_model.model, gm_model.model), ("scripted", "naive-gm"))
        builder_prompt = "Your notes:\n  (none)\n\nYou are builder at (0, 64, 0)."
        self.assertEqual(agent_model.complete("", builder_prompt, []).tool_call,
                         ToolCall("move_to", {"x": -107, "y": -60, "z": 8, "note": "step 1"}))
        security_prompt = "Your notes:\n  (none)\n\nYou are security at (0, 64, 0)."
        self.assertEqual(agent_model.complete("", security_prompt, []).tool_call, ToolCall("wait", {"ticks": 1200}))
        views = {role["name"]: "Beside the bell." for role in village._manifest.roles}
        gm = GameMaster(gm_model, "A village.", views)
        self.assertEqual(gm.resolve(0, "builder", ToolCall("speak", {"text": "The bridge is ready.", "mode": "talk"})),
                         Resolution(True, "", 0, ["builder did speak"], views["builder"],
                                    {person: "The bridge is ready." for person in views if person != "builder"}))


class NarratedVillageLiveModelsTest(unittest.TestCase):
    """Live narrated runs give the GM its own model when one is named, else the agents' model."""

    def test_gm_model_defaults_to_the_agents_model(self):
        """Without `--gm-model` the GM uses the agents' model; with it, the named one."""
        options = {"scripted": False, "variant": "sabotage", "provider": "openai-compatible", "model": "agent-model",
                   "base_url": "http://127.0.0.1:9/v1", "temperature": 0.7, "reasoning_effort": None}
        for gm_model, expected in ((None, "agent-model"), ("gm-model", "gm-model")):
            with self.subTest(gm_model=gm_model):
                agents, gm = NarratedVillage(argparse.Namespace(**options, gm_model=gm_model))._models()
                self.assertEqual((agents.model, gm.model), ("agent-model", expected))
