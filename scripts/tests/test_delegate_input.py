"""Unit tests for the delegate's input in Kev's format: the flattened state document and the question row."""

import json
from pathlib import Path
import unittest

from orchestrator.decisions import Decision
from orchestrator.delegate_input import kev_record, state_text
from scripts.build_decision_dataset import StateDocuments

ROOT = Path(__file__).resolve().parents[2]


class StateTextTest(unittest.TestCase):
    """The state text is deterministic, keeps every byte of the state document, and only grows within an epoch."""

    def setUp(self):
        """The envoy's first four Sol decisions (one epoch) as state documents."""
        fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        decisions = [Decision(**entry) for entry in fixture["decisions"]]
        documents = StateDocuments(decisions, fixture["system"], fixture["tools"])
        self.states = [documents.states[(decision.agent, decision.game_time)] for decision in decisions]

    def test_each_decision_extends_the_previous_text_within_an_epoch(self):
        texts = [state_text(state) for state in self.states]
        for earlier, later in zip(texts, texts[1:]):
            self.assertTrue(later.startswith(earlier))
            self.assertGreater(len(later), len(earlier))

    def test_turns_show_the_call_in_sorted_key_json_and_its_result(self):
        text = state_text(self.states[1])
        turn = self.states[1]["history"][0]
        self.assertIn(f"Your action: open_container {turn['arguments']}\nResult: done", text)
        self.assertTrue(text.startswith(self.states[1]["system"] + "\n\n"))
        self.assertTrue(text.endswith(self.states[1]["user"]))

    def test_a_turn_without_a_call_says_none(self):
        state = {"system": "S", "history": [{"user": "U1", "tool": None, "arguments": None, "result": "no action"}],
                 "user": "U2"}
        self.assertEqual(state_text(state), "S\n\nU1\n\nYour action: (none)\nResult: no action\n\nU2")


class KevRecordTest(unittest.TestCase):
    """A dataset record becomes one Kev question whose options are the candidate labels in order."""

    def test_question_options_and_label(self):
        record = {"agent": "mayor", "frontier_index": 1,
                  "state": {"system": "S", "history": [], "user": "U"},
                  "candidates": [{"label": "wait {\"ticks\":100} | wait 100 ticks"},
                                 {"label": "keep_going {} | carry on"}]}
        self.assertEqual(kev_record(record), {"state": "S\n\nU", "questions": [
            {"instr": "Which act does mayor take next?",
             "options": ["wait {\"ticks\":100} | wait 100 ticks", "keep_going {} | carry on"], "label": 1}]})
        self.assertEqual(kev_record(record, ["A", "B"])["questions"][0]["options"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
