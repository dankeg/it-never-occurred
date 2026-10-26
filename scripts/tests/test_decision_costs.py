"""Synthetic decision logs pin cost estimates and behavioural summaries."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.decision_costs import behaviour, least_squares, main, summarize


class LeastSquaresTest(unittest.TestCase):
    """The timing fit recovers noiseless coefficients and rejects singular designs."""

    def test_noiseless_rows_recover_fixed_prefill_and_decode_coefficients(self):
        """Independent noiseless rows recover the exact fixed and token-time coefficients."""
        self.assertEqual(least_squares([(0, 0, 2), (1, 0, 5), (0, 1, 7), (1, 1, 10)]), (2.0, 3.0, 5.0))

    def test_singular_rows_have_no_fit(self):
        """Empty, identical and collinear timing rows have no unique fit."""
        for rows in ([], [(1, 1, 2)] * 3, [(1, 2, 3), (2, 4, 6), (3, 6, 9)]):
            with self.subTest(rows=rows):
                self.assertIsNone(least_squares(rows))


class DecisionBehaviourTest(unittest.TestCase):
    """Failure and repeat shares use each agent's time order while ignoring notes."""

    def test_failures_and_repeats_are_counted_per_agent_without_notes(self):
        """Three failure prefixes count and changed notes do not prevent a repeated action."""
        rows = [
            {"agent": "ben", "game_time": 80, "tool": "wait", "args": {"ticks": 20}, "outcome": "started"},
            {"agent": "cy", "game_time": 40, "tool": "wait", "args": {"ticks": 20}, "outcome": "failed: busy"},
            {"agent": "ben", "game_time": 0, "tool": "wait", "args": {"note": "first", "ticks": 20},
             "outcome": "not possible: blocked"},
            {"agent": "cy", "game_time": 0, "tool": "keep_going", "args": None, "outcome": "invalid action"},
            {"agent": "ben", "game_time": 40, "tool": "wait", "args": {"ticks": 20, "note": "second"},
             "outcome": "started"}]
        self.assertEqual(behaviour(rows), {"failedShare": 0.6, "repeatShare": 0.667})

    def test_first_actions_have_no_repeat_comparison(self):
        """An agent's first action cannot contribute a repeat comparison."""
        self.assertEqual(behaviour([{"agent": "ben", "game_time": 0, "tool": "wait", "args": {},
                                     "outcome": "done"}]), {"failedShare": 0.0, "repeatShare": None})


class DecisionCostSummaryTest(unittest.TestCase):
    """Temporary run logs separate model costs, inclusive histories and fitted timings."""

    def _write_run(self, root, name):
        """Write four independent timed model rows and one expensive replay row."""
        run = Path(root) / name
        (run / "orchestrator").mkdir(parents=True)
        rows = []
        for tick, input_tokens, output_tokens, seconds in ((0, 100, 10, 5), (2400, 200, 10, 6),
                                                         (4800, 100, 20, 8), (7200, 200, 20, 9)):
            rows.append({"agent": "ben", "game_time": tick, "source": "model", "tool": "wait",
                         "args": {"ticks": 20}, "outcome": "failed: blocked" if tick == 0 else "started",
                         "input_tokens": input_tokens, "cached_tokens": 50, "output_tokens": output_tokens,
                         "seconds": seconds})
        rows.append({**rows[-1], "game_time": 9600, "source": "replay", "input_tokens": 999999,
                     "output_tokens": 999999, "seconds": 999999, "outcome": "invalid action"})
        (run / "orchestrator/decisions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
        return run

    def test_only_model_rows_are_costed_and_timings_are_split(self):
        """Replays affect behaviour but not token costs or the fixed/prefill/decode timing split."""
        with tempfile.TemporaryDirectory() as folder:
            run = self._write_run(folder, "costs")
            self.assertEqual(summarize(run, None, (1000, 100, 10000)), {
                "run": "costs", "decisions": 4, "lastGameTime": 7200, "failedShare": 0.4, "repeatShare": 1.0,
                "uncached": 150, "cached": 50, "output": 15.0, "costPerDecision": 0.305, "costTotal": 1.22,
                "costPerGameDay": 4.1, "seconds": 7.0, "fixedSeconds": 1.0,
                "prefillTokensPerSecond": 100, "decodeTokensPerSecond": 3})

    def test_until_excludes_the_boundary_and_later_rows(self):
        """The until cutoff is exclusive and recomputes rates from only earlier model decisions."""
        with tempfile.TemporaryDirectory() as folder:
            run = self._write_run(folder, "early")
            self.assertEqual(summarize(run, 4800, (1000, 100, 10000)), {
                "run": "early", "decisions": 2, "lastGameTime": 2400, "failedShare": 0.5, "repeatShare": 1.0,
                "uncached": 150, "cached": 50, "output": 10.0, "costPerDecision": 0.255, "costTotal": 0.51,
                "costPerGameDay": 5.1, "seconds": 5.5})
            self.assertEqual(summarize(run, 0), {"run": "early", "decisions": 0, "lastGameTime": 0})

    def test_main_prints_one_json_summary_per_run(self):
        """The CLI emits exactly one JSON line per run using its until and rates arguments."""
        with tempfile.TemporaryDirectory() as folder:
            runs = [self._write_run(folder, name) for name in ("first", "second")]
            output = io.StringIO()
            with redirect_stdout(output):
                result = main([*(str(run) for run in runs), "--until", "4800", "--rates", "1000,100,10000"])
            self.assertEqual(result, 0)
            self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()],
                             [summarize(run, 4800, (1000, 100, 10000)) for run in runs])
