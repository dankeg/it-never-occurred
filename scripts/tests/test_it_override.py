"""Server-free synthetic-log checks for IT-04's reuse assertions and real same-choice guard."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.decisions import Decision, DecisionLog
from scripts.fork_decision import DecisionFork
from scripts.test_it04_override import reuse_checks


class OverrideReuseTest(unittest.TestCase):
    """Require pointwise replay while prompts match and sampling as soon as they change."""

    def test_replay_until_digest_changes_then_model(self):
        """A genuine matched replay followed by a changed prompt satisfies both gate assertions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, swapped = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "swapped.jsonl")
            decision = Decision(40, "mayor", "same", False, "model", "wait", {"ticks": 200}, "scripted", 0, 0,
                                "prompt")
            factual.append(decision)
            factual.append(replace(decision, game_time=80))
            swapped.append(replace(decision, source="replay"))
            swapped.append(replace(decision, game_time=80, digest="changed"))
            swapped.append(replace(decision, agent="builder", source="forced", args={"ticks": 400}))
            self.assertTrue(all(reuse_checks(factual, swapped, "builder", 40, 120).values()))

    def test_wrong_action_or_wrong_source_is_not_reuse(self):
        """Matching digests cannot excuse a changed tool/args, a model call, or an unrelated forced decision."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual = DecisionLog(root / "factual.jsonl")
            decision = Decision(40, "mayor", "same", False, "model", "wait", {"ticks": 200}, "scripted", 0, 0,
                                "prompt")
            factual.append(decision)
            for index, changes in enumerate(({"args": {"ticks": 201}}, {"tool": "keep_going"},
                                              {"source": "model"}, {"source": "forced"})):
                with self.subTest(changes=changes):
                    swapped = DecisionLog(root / f"swapped_{index}.jsonl")
                    swapped.append(replace(decision, **({"source": "replay"} | changes)))
                    self.assertFalse(all(reuse_checks(factual, swapped, "builder", 40, 120).values()))

    def test_changed_or_missing_prompt_cannot_claim_replay(self):
        """A replay entry with no matching source digest is an error even after an earlier legitimate replay."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, swapped = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "swapped.jsonl")
            decision = Decision(40, "mayor", "same", False, "model", "wait", {"ticks": 200}, "scripted", 0, 0,
                                "prompt")
            factual.append(decision)
            swapped.append(replace(decision, source="replay"))
            swapped.append(replace(decision, game_time=80, digest="unmatched", source="replay"))
            self.assertFalse(all(reuse_checks(factual, swapped, "builder", 40, 120).values()))

    def test_empty_other_agent_suffix_is_not_vacuous_success(self):
        """A swap alone, or decisions entirely outside the continuation, proves nothing about reuse."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, swapped = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "swapped.jsonl")
            self.assertFalse(all(reuse_checks(factual, swapped, "builder", 40, 120).values()))
            swapped.append(Decision(0, "mayor", "same", False, "replay", "wait", {"ticks": 200},
                                    "scripted", 0, 0, "prompt"))
            self.assertFalse(all(reuse_checks(factual, swapped, "builder", 40, 120).values()))


class IdenticalOverrideTest(unittest.TestCase):
    """The negative control invokes DecisionFork itself and must not need a server or a complete fork namespace."""

    def test_identical_act_is_refused_without_creating_an_arm(self):
        """The production same_choice check rejects the logged act before any cloning or server setup."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = DecisionLog(root / "factual/orchestrator/decisions.jsonl")
            log.append(Decision(40, "builder", "same", False, "model", "mine", {"x": 1, "y": -60, "z": 2},
                                "scripted", 0, 0, "prompt"))
            args = argparse.Namespace(runs_dir=root, variant="sabotage", run="factual", agent="builder", tick=40,
                                      swap=json.dumps({"name": "mine", "args": {"x": 1, "y": -60, "z": 2}}))
            before = log.path.read_bytes()
            with self.assertRaisesRegex(ValueError, "matches the logged choice"):
                DecisionFork(args)
            self.assertEqual(log.path.read_bytes(), before)
            self.assertEqual([path.name for path in root.iterdir()], ["factual"])


if __name__ == "__main__":
    unittest.main()
