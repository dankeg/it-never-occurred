"""Server-free checks for IT-03's live fork selection and full suffix comparison."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from orchestrator.decisions import Decision, DecisionLog
from scripts.test_it03_forks import snapshot_labels, suffix_checks


class ForkSelectionTest(unittest.TestCase):
    """Select real running activities, thinking gaps, and the exact nightly interval."""

    def test_running_move_and_speech_are_distinct_from_wait_and_finished_skills(self):
        """An idle or waiting agent must not count as mid-skill or mid-speech coverage."""
        self.assertEqual(snapshot_labels(20, {"ana": {"state": "running", "skill": "move_to"}}, {}), {"skill"})
        self.assertEqual(snapshot_labels(20, {"ana": {"state": "running", "skill": "speak"}}, {}), {"speech"})
        for status in ({"state": "running", "skill": "wait"}, {"state": "succeeded", "skill": "speak"},
                       {"state": "running"}):
            with self.subTest(status=status):
                self.assertEqual(snapshot_labels(20, {"ana": status}, {}), set())

    def test_thinking_requires_an_idle_agent_with_future_ready_time(self):
        """The timing negative control needs a genuine gap, not a busy agent or an expired deadline."""
        self.assertEqual(snapshot_labels(60, {"ana": {"state": "failed"}}, {"ana": 100}), {"thinking"})
        self.assertEqual(snapshot_labels(100, {"ana": {"state": "failed"}}, {"ana": 100}), set())
        self.assertEqual(snapshot_labels(60, {"ana": {"state": "running", "skill": "wait"}}, {"ana": 100}), set())

    def test_night_is_half_open_and_repeats_each_day(self):
        """The saved night snapshot is strictly in the 16,000-to-24,000 rest window."""
        for tick, expected in ((15_980, False), (16_000, True), (16_020, True), (23_980, True),
                               (24_000, False), (40_020, True)):
            with self.subTest(tick=tick):
                self.assertEqual("night" in snapshot_labels(tick, {}, {}), expected)


class ForkSuffixTest(unittest.TestCase):
    """Use real synthetic decision logs and harness-shaped hash records to reject incomplete exactness claims."""

    def test_fork_hash_is_excluded_and_decision_at_fork_is_included(self):
        """A snapshot is before its poll: replay includes that decision but only strictly later hashes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, replay = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "replay.jsonl")
            before = Decision(0, "ana", "prefix", False, "model", "wait", {"ticks": 40}, "scripted", 0, 0, "before")
            after = replace(before, game_time=40, digest="suffix", user_prompt="after", outcome="started")
            factual.append(before)
            factual.append(after)
            replay.append(replace(after, source="replay"))
            hashes = [{"gameTime": tick, "total": str(tick)} for tick in (20, 40, 60, 80)]
            recorded = SimpleNamespace(log=factual, hashes=hashes)
            resumed = SimpleNamespace(log=replay, hashes=hashes[2:], stats=SimpleNamespace(model_calls=0))
            self.assertTrue(all(suffix_checks(recorded, resumed, 40, 80).values()))
            resumed.hashes = hashes[1:]
            self.assertFalse(suffix_checks(recorded, resumed, 40, 80)["identical complete world hashes"])

    def test_changed_hash_or_decision_and_model_calls_fail_independently(self):
        """Detect the missing-timing control's changed hash as well as changed args, outcome, or digest."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual = DecisionLog(root / "factual.jsonl")
            decision = Decision(40, "ana", "digest", False, "model", "wait", {"ticks": 40}, "scripted", 0, 0,
                                "prompt", outcome="started")
            factual.append(decision)
            hashes = [{"gameTime": 60, "total": "world"}, {"gameTime": 80, "total": "world"}]
            recorded = SimpleNamespace(log=factual, hashes=hashes)
            for index, changes in enumerate(({"digest": "different"}, {"args": {"ticks": 80}},
                                              {"outcome": "not possible"}, {"tool": "keep_going"})):
                with self.subTest(changes=changes):
                    replay = DecisionLog(root / f"replay_{index}.jsonl")
                    replay.append(replace(decision, source="replay", **changes))
                    resumed = SimpleNamespace(log=replay, hashes=hashes, stats=SimpleNamespace(model_calls=0))
                    self.assertFalse(suffix_checks(recorded, resumed, 40, 80)["identical decisions"])
            resumed = SimpleNamespace(log=factual, hashes=[hashes[0], {"gameTime": 80, "total": "changed"}],
                                      stats=SimpleNamespace(model_calls=1))
            checks = suffix_checks(recorded, resumed, 40, 80)
            self.assertFalse(checks["identical complete world hashes"])
            self.assertFalse(checks["zero model calls"])

    def test_equal_empty_or_truncated_suffixes_do_not_pass(self):
        """Equality alone cannot hide a run with no decisions or missing checkpoints."""
        with tempfile.TemporaryDirectory() as temporary:
            empty = DecisionLog(Path(temporary) / "empty.jsonl")
            recorded = SimpleNamespace(log=empty, hashes=[])
            resumed = SimpleNamespace(log=empty, hashes=[], stats=SimpleNamespace(model_calls=0))
            checks = suffix_checks(recorded, resumed, 40, 80)
            self.assertFalse(checks["identical decisions"])
            self.assertFalse(checks["identical complete world hashes"])

    def test_night_suffix_can_have_no_decisions_but_still_requires_complete_hashes(self):
        """Sleeping through the end is valid: an empty suffix is not the same as an empty factual run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, replay = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "replay.jsonl")
            factual.append(Decision(0, "ana", "awake", False, "model", "wait", {"ticks": 400}, "scripted", 0, 0,
                                    "prompt"))
            hashes = [{"gameTime": 16_040, "total": "first"}, {"gameTime": 16_060, "total": "second"}]
            recorded = SimpleNamespace(log=factual, hashes=hashes)
            resumed = SimpleNamespace(log=replay, hashes=hashes, stats=SimpleNamespace(model_calls=0))
            self.assertTrue(all(suffix_checks(recorded, resumed, 16_020, 16_060).values()))
            resumed.hashes = hashes[:1]
            self.assertFalse(suffix_checks(recorded, resumed, 16_020, 16_060)["identical complete world hashes"])

    def test_sparse_hashes_count_absolute_boundaries_after_an_off_grid_fork(self):
        """A fork at tick 20 still has both the tick-200 and tick-400 checkpoints, not just one interval."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factual, replay = DecisionLog(root / "factual.jsonl"), DecisionLog(root / "replay.jsonl")
            decision = Decision(40, "ana", "same", False, "model", "wait", {"ticks": 400}, "scripted", 0, 0,
                                "prompt")
            factual.append(decision)
            replay.append(replace(decision, source="replay"))
            hashes = [{"gameTime": 200, "total": "first"}, {"gameTime": 400, "total": "second"}]
            recorded = SimpleNamespace(log=factual, hashes=hashes)
            resumed = SimpleNamespace(log=replay, hashes=hashes, stats=SimpleNamespace(model_calls=0))
            self.assertTrue(all(suffix_checks(recorded, resumed, 20, 400, 200).values()))


if __name__ == "__main__":
    unittest.main()
