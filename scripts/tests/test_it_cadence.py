"""Test the cadence gate's measured-duration and timing comparisons without a server."""

from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import Decision, DecisionLog
from orchestrator.llm import ToolCall
from orchestrator.narration import NarratedRuntime
from scripts import harness
from scripts.test_it18_cadence import (
    EngineFaithfulGM, EngineRecord, changed_tick, decision_ticks, skill_duration, thinking_exercised,
)
from scripts.tests.fake_world import FakeWorld


class CadenceTest(unittest.TestCase):
    """Keep the oracle grounded in actual skill endings and actual decisions, including negative controls."""

    def decision(self, tick=40, agent="ana", outcome="started"):
        """Construct one decision record without executing an engine or policy."""
        return Decision(tick, agent, "recorded-digest", False, "model", "wait", {"ticks": 80},
                        "scripted", 0, 0, "prompt", outcome)

    def test_actual_end_not_poll_rounded_duration(self):
        """A skill ending between polls uses its final tick, not an estimated or next-decision duration."""
        endings = {"ana:1": {"gameTime": 73, "outcome": "succeeded"}}
        self.assertEqual(skill_duration(self.decision(), {"skillId": "ana:1"}, endings), 33)

    def test_post_poll_cancellation_preserves_interruption_at_that_poll(self):
        """A skill cancelled by a replacement at tick 80 was still running when tick 80's prompt was built."""
        endings = {"ana:1": {"gameTime": 80, "outcome": "cancelled"}}
        self.assertGreater(40 + skill_duration(self.decision(), {"skillId": "ana:1"}, endings), 80)

    def test_instant_and_rejected_actions_take_zero_engine_ticks(self):
        """Thinking delay belongs to the runtime and must not be smuggled into the GM's durations."""
        for outcome in ("done", "not possible: target_not_found", "failed: busy", "carried on"):
            with self.subTest(outcome=outcome):
                self.assertEqual(skill_duration(self.decision(outcome=outcome), None, {}), 0)

    def test_missing_and_impossible_endings_are_not_guessed(self):
        """Absent or backwards actual skill endings fail clearly instead of inventing an action duration."""
        for start, endings in ((None, {}), ({"skillId": "ana:1"}, {}),
                               ({"skillId": "ana:1"}, {"ana:1": {"gameTime": 39, "outcome": "succeeded"}})):
            with self.subTest(start=start, endings=endings), self.assertRaises(harness.HarnessError):
                skill_duration(self.decision(), start, endings)

    def test_ticks_are_grouped_by_agent_in_decision_order(self):
        """A real file-backed log, including simultaneous decisions, retains every exact decision tick."""
        with tempfile.TemporaryDirectory() as temporary:
            log = DecisionLog(Path(temporary) / "decisions.jsonl")
            for tick, agent in ((80, "ana"), (40, "ben"), (40, "ana")):
                log.append(self.decision(tick=tick, agent=agent))
            self.assertEqual(decision_ticks(log), {"ana": [40, 80], "ben": [40]})

    def test_negative_control_requires_a_mistimed_decision_not_truncation(self):
        """A partial log alone is not a valid negative control; an actual early decision is."""
        expected = {"ana": [40, 160, 280], "ben": [40, 80]}
        self.assertFalse(changed_tick(expected, {"ana": [40], "ben": [40, 80]}))
        self.assertFalse(changed_tick(expected, {}))
        self.assertTrue(changed_tick(expected, {"ana": [40, 80]}))

    def test_thinking_coverage_needs_an_actual_delayed_instant_action(self):
        """Simply doing an instant act does not prove thinking time was exercised before the next poll."""
        with tempfile.TemporaryDirectory() as temporary:
            log = DecisionLog(Path(temporary) / "decisions.jsonl")
            log.append(self.decision(outcome="done"))
            self.assertFalse(thinking_exercised(log))
            log.append(self.decision(tick=160, outcome="done"))
            self.assertTrue(thinking_exercised(log))

    def test_scripted_gm_uses_measured_speech_duration_and_actual_recipients(self):
        """Only the utterance actually heard by another agent is routed, using the real recorded skill end."""
        with tempfile.TemporaryDirectory() as temporary:
            log = DecisionLog(Path(temporary) / "decisions.jsonl")
            log.append(Decision(0, "ana", "digest", False, "model", "speak", {"text": "hello"},
                                "scripted", 0, 0, "prompt", "started"))
            record = EngineRecord(FakeWorld({"ana": [0, 64, 0], "ben": [1, 64, 0]}))
            record.request("observe", {"name": "ana"})
            record.request("observe", {"name": "ben"})
            record.starts[("ana", 0)] = {"skillId": "ana:1", "uttId": 7}
            record.observations[("ben", 40)] = {"heard": [
                {"prov": "utt:6", "text": "unrelated"},
                {"prov": "utt:7", "text": "hello", "speaker": "ana", "gameTime": 7}]}
            gm = EngineFaithfulGM(log, record, {"ana:1": {"gameTime": 7, "outcome": "succeeded"}})
            resolution = gm.resolve(0, "ana", ToolCall("speak", {"text": "hello"}))
            self.assertTrue(resolution.accepted)
            self.assertEqual(resolution.duration_ticks, 7)
            self.assertEqual(resolution.observations, {})
            self.assertEqual(gm.calls, 1)
            agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in ("ana", "ben")]
            runtime = NarratedRuntime(agents, None, gm, Path(temporary) / "narrated")
            gm.deliver(runtime, 0)
            self.assertFalse(runtime._interrupted["ben"])
            gm.deliver(runtime, 40)
            self.assertTrue(runtime._interrupted["ben"])
            heard = runtime.agents[1].memory.to_json()["heard"]
            self.assertEqual(len(heard), 1)
            self.assertEqual(heard[0]["text"], "hello")
            self.assertEqual(heard[0]["gameTime"], 7)
            gm.deliver(runtime, 80)
            self.assertEqual(len(runtime.agents[1].memory.to_json()["heard"]), 1)
