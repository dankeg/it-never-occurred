"""Jubilee outcomes and leak evidence from a synthetic run directory (no server, no model)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.jubilee_leaks import JubileeLeaks
from scripts.jubilee_outcomes import HallState, JubileeOutcomes

HALL = [-60, -60, -10, -45, -58, -1]
MANIFEST = {
    "layout": {"hall": {"bounds": HALL, "feast_tables": ["feast_table_1"]}},
    "containers": {"feast_table_1": {"pos": [-52, -60, -5]}},
    "jubilee": {"committee": {"members": ["innkeeper", "freight", "builder"]}, "reveal_game_time": 36_000},
}


class StubJudge:
    """Reveals exactly the texts that mention the feast; records the batch it was asked to judge."""

    def __init__(self):
        self.batches = []

    def judge(self, texts: list[str]) -> None:
        self.batches.append(sorted(texts))

    def reveals(self, text: str) -> bool:
        return "feast" in text


class FakeClient:
    """Answers `container_items` and `blocks` like the harness."""

    def request(self, command, args):
        if command == "container_items":
            return {"gameTime": 36_000, "items": [{"item": "minecraft:bread", "count": 64},
                                                  {"item": "minecraft:bread", "count": 10, "mouldy": True}]}
        lanterns = 3
        return {"gameTime": 36_000, "blocks": ["minecraft:lantern[hanging=false]"] * lanterns
                + ["minecraft:air"] * (len(args["positions"]) - lanterns)}


def write_lines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class JubileeOutcomeTest(unittest.TestCase):
    """Fresh bread and lights are counted from the world; attendance from the mayor's polls."""

    def test_hall_and_attendance(self):
        outcomes = JubileeOutcomes(MANIFEST)
        hall = outcomes.hall(FakeClient())
        self.assertEqual(hall, HallState(36_000, 64, 10, 3))
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.jsonl"
            write_lines(trajectory, [{"t": 35_960, "agent": "mayor", "pos": [-50.5, -60, -5.5]},
                                     {"t": 36_400, "agent": "mayor", "pos": [-50.5, -60, -5.5]},
                                     {"t": 36_400, "agent": "baker", "pos": [-50.5, -60, -5.5]},
                                     {"t": 37_200, "agent": "mayor", "pos": [-50.5, -60, -5.5]}])
            result = outcomes.result(hall, trajectory)
        self.assertEqual(result.mayor_in_hall, [36_400])
        self.assertEqual((result.feast, result.lit, result.attended), (False, False, True))


class JubileeLeakTest(unittest.TestCase):
    """Each channel is found, committee members are left out of the summary, and nothing after the reveal counts."""

    def test_channels_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "scenario.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
            write_lines(run / "orchestrator/perceptions.jsonl", [
                {"t": 1_000, "agent": "mayor", "entities": [
                    {"name": "freight", "pos": [10, -60, 3], "load": {"item": "minecraft:bread", "count_band": "sack"}},
                    {"name": "baker", "pos": [12, -60, 3],  # the baker's own delivery is ordinary business
                     "load": {"item": "minecraft:bread", "count_band": "sack"}}]},
                {"t": 2_000, "agent": "neighbour", "heard": [{"speaker": "freight", "text": "the feast is at six"},
                                                             {"speaker": "baker", "text": "lovely weather"}]},
                {"t": 3_000, "agent": "secretary", "changes": [{"pos": [-50, -59, -5], "to": "minecraft:lantern"}]},
                {"t": 3_100, "agent": "builder", "heard": [{"speaker": "innkeeper", "text": "feast plans"}]},
                {"t": 40_000, "agent": "mayor", "heard": [{"speaker": "innkeeper", "text": "welcome to your feast"}]},
            ])
            write_lines(run / "world/simharness/events.jsonl", [
                {"type": "block_change", "gameTime": 3_000, "pos": [-50, -59, -5], "to": "minecraft:lantern"}])
            write_lines(run / "orchestrator/trajectory.jsonl", [
                {"t": 2_900, "agent": "mayor", "pos": [-50.5, -60, -5.5]},
                {"t": 3_040, "agent": "mayor", "pos": [-50.5, -60, -5.5]}])
            judge = StubJudge()
            leaks = JubileeLeaks(run, judge)
            evidence = leaks.evidence()
        channels = [(item.agent, item.channel) for item in evidence]
        # what members hear among themselves is not judged; nothing after the reveal counts
        self.assertEqual(channels, [("mayor", "sack"), ("neighbour", "speech"), ("secretary", "hall_change"),
                                    ("mayor", "hall_entry")])
        self.assertEqual(judge.batches, [["lovely weather", "the feast is at six"]])
        summary = leaks.summary(evidence)
        self.assertEqual(set(summary), {"mayor", "neighbour", "secretary"})
        self.assertEqual(summary["mayor"]["first"]["channel"], "sack")
        self.assertFalse(leaks.mayor_kept_unaware(evidence))



class VantageViewTest(unittest.TestCase):
    """The observer's view lists sightings on change, and everything perceived, from the perception log only."""

    def test_sightings_on_change_and_perceived_rows(self):
        from scripts.run_views import RunViews
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "scenario.json").write_text(json.dumps({"roles": [{"name": "mayor"}, {"name": "freight"}]}),
                                               encoding="utf-8")
            sack = {"item": "minecraft:bread", "count_band": "sack"}
            write_lines(run / "orchestrator/perceptions.jsonl", [
                {"t": 40, "agent": "mayor", "entities": [{"name": "freight", "pos": [1, -60, 1]}]},
                {"t": 80, "agent": "mayor", "entities": [{"name": "freight", "pos": [2, -60, 1]}]},
                {"t": 120, "agent": "mayor", "entities": [{"name": "freight", "pos": [3, -60, 1], "load": sack}],
                 "heard": [{"speaker": "freight", "text": "Good evening"}]},
                {"t": 120, "agent": "freight", "heard": [{"speaker": "mayor", "text": "hidden from the mayor view"}]},
                {"t": 160, "agent": "mayor", "read": "book Appointments"}])
            text = RunViews(run).render("vantage:mayor")
        lines = text.splitlines()
        self.assertEqual([line.split("] ")[1].split(" ")[0] for line in lines], ["SAW", "SAW", "HEARD", "READ"])
        self.assertIn('"load"', lines[1])
        self.assertNotIn("hidden from the mayor view", text)


class CompactViewTest(unittest.TestCase):
    """Compact lines keep the evidence (who, what, where, words) and drop ids; `since` drops setup events."""

    def test_compact_speech_action_and_effect(self):
        from scripts.run_views import RunViews
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "scenario.json").write_text(json.dumps({"roles": [{"name": "freight"}, {"name": "baker"}]}),
                                               encoding="utf-8")
            write_lines(run / "world/simharness/events.jsonl", [
                {"type": "doc_write", "gameTime": 100, "id": 2, "author": "freight", "kind": "sign"},
                {"type": "utterance", "gameTime": 2100, "id": 9, "speaker": "freight", "to": "baker",
                 "volume": "whisper", "text": "Can you bake for us?", "listeners": ["baker"]},
                {"type": "transfer", "gameTime": 2205, "id": 10, "from": "container:40,-60,44", "to": "freight",
                 "count": 32, "item": "minecraft:bread"}])
            write_lines(run / "orchestrator/decisions.jsonl", [
                {"game_time": 2200, "agent": "freight", "tool": "chest_take",
                 "args": {"x": 40, "y": -60, "z": 44, "item": "minecraft:bread", "count": 32}, "outcome": "started"}])
            text = RunViews(run).render("both", compact=True, since=2000)
        self.assertNotIn("sign", text)
        self.assertIn("freight to baker (whisper; heard by baker): Can you bake for us?", text)
        self.assertIn("freight chest_take (40, -60, 44) item=bread count=32 -> started", text)
        self.assertIn("container:40,-60,44 -> freight: 32 bread", text)
        self.assertNotIn('"id"', text)

if __name__ == "__main__":
    unittest.main()
