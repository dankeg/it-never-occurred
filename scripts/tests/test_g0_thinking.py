"""Unit tests for gate G0's bookkeeping: prompts rebuilt exactly as Sol saw them, and the paired summary."""

import json
from pathlib import Path
import unittest

from orchestrator.decisions import Decision
from scripts.g0_thinking import ThinkingGate, states

ROOT = Path(__file__).resolve().parents[2]


class StatesTest(unittest.TestCase):
    """Each logged decision is re-asked with the history and prompt that reproduce its digest."""

    def setUp(self):
        """The envoy's first four Sol decisions with the prompts of the commit that recorded them."""
        self.fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        self.decisions = [Decision(**entry) for entry in self.fixture["decisions"]]

    def test_histories_grow_within_the_epoch(self):
        cases = states(self.decisions, self.fixture["system"], self.fixture["tools"])
        self.assertEqual([len(history) for _, history in cases], [0, 1, 2, 3])

    def test_a_prompt_that_does_not_reproduce_its_digest_is_refused(self):
        with self.assertRaises(ValueError):
            states(self.decisions, {"envoy": "another system prompt"}, self.fixture["tools"])


class SummaryTest(unittest.TestCase):
    """Agreement per mode and the paired difference over decisions asked in both modes."""

    def test_paired_difference_and_non_speech_share(self):
        def row(mode, tick, agree, free_text=False):
            return {"mode": mode, "agent": "mayor", "game_time": tick, "agree": agree, "free_text": free_text,
                    "call": {"name": "wait", "args": {}}, "seconds": 2.0, "output_tokens": 10}
        rows = [row("off", 40, False), row("off", 80, True), row("off", 120, False, True),
                row("on", 40, True), row("on", 80, True), row("on", 120, True, True)]
        summary = ThinkingGate.summary(rows)
        self.assertAlmostEqual(summary["off"]["agreement"], 1 / 3)
        self.assertEqual(summary["on"]["non_speech_agreement"], 1.0)
        self.assertAlmostEqual(summary["difference_all"]["on_minus_off"], 2 / 3)
        self.assertEqual(summary["difference_non_speech"]["pairs"], 2)


if __name__ == "__main__":
    unittest.main()
