"""Stdlib coverage for tiny viewer runs, live log caches, and local HTTP."""

import json
from io import BytesIO
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.viewer_server import RunData, RunIndex, ViewerHandler


class ViewerServerTests(unittest.TestCase):
    """Exercise public viewer APIs without Minecraft or production run data."""

    def setUp(self):
        """Create an isolated run with an empty decision log."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs_dir = self.root / "runs"
        self.run_dir = self.runs_dir / "tiny_sabotage"
        self.scenario_dir = self.root / "scenario"
        self.scenario_dir.mkdir()
        self.cache_dir = self.root / "cache"
        self.write_jsonl("orchestrator/decisions.jsonl", [])

    def write_jsonl(self, relative, records):
        """Write a small UTF-8 source log and return its path."""
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                        encoding="utf-8")
        return path

    def data(self):
        """Open the fixture using an explicit per-run cache directory."""
        return RunData(self.run_dir, scenario_dir=self.scenario_dir, cache_dir=self.cache_dir)

    @staticmethod
    def decision(tick, prompt):
        """Return a decision containing an on-demand prompt."""
        return {"game_time": tick, "agent": "Ada", "tool": "say", "args": {"text": "hello"},
                "outcome": "ok", "user_prompt": prompt}

    def test_prompt_includes_the_conversation_since_the_epoch_start(self):
        self.write_jsonl("orchestrator/decisions.jsonl", [
            self.decision(40, "Your notes:\n  (none)\nfirst full prompt"),
            self.decision(80, "Since your last action:\n  (nothing new)"),
            self.decision(120, "Your notes:\n  (none)\nsecond epoch"),
            self.decision(160, "Since your last action:\n  one update"),
            self.decision(200, "Since your last action:\n  another update")])
        result = self.data().prompt("Ada", 200)
        self.assertEqual([turn["t"] for turn in result["history"]], [120, 160])
        self.assertEqual(result["history"][0]["tool"], "say")
        self.assertEqual(self.data().prompt("Ada", 120)["history"], [])  # a full prompt starts its own epoch

    def test_decisions_fall_back_to_numeric_prompt_positions_and_filter_events(self):
        """Normalize and sort events while recovering signed fractional positions."""
        self.write_jsonl("orchestrator/decisions.jsonl", [
            self.decision(200, "You are Ada at (3, -61, -4.5)."),
            self.decision(100, "You are Ada at (-1.25, -61.0, 2.5).\nYou saw a bridge."),
        ])
        source = self.write_jsonl("world/simharness/events.jsonl", [
            {"type": "origin", "gameTime": 9000},
            {"type": "npc_day", "gameTime": 10000},
            {"type": "utterance", "gameTime": 150, "speaker": "Villager", "text": "hello"},
        ])
        with source.open("ab") as handle:
            handle.write(b"not valid JSON\n")
        run = self.data()
        events = run.events()
        self.assertEqual([(event["t"], event["type"]) for event in events],
                         [(100, "decision"), (150, "utterance"), (200, "decision")])
        self.assertTrue(all("user_prompt" not in event for event in events))
        self.assertEqual(run.events(start=150, end=150), [events[1]])
        points = run.tracks()["agents"]["Ada"]
        self.assertEqual([point[:3] for point in points], [[100, -1.25, 2.5], [200, 3, -4.5]])
        self.assertTrue(all(len(point) == 5 for point in points))
        self.assertTrue(any("malformed complete lines" in warning for warning in run.meta()["warnings"]))

    def test_trajectory_npc_sampling_and_extended_trip_rows(self):
        """Prefer trajectory positions and retain independent NPC tracks and trips."""
        self.write_jsonl("orchestrator/decisions.jsonl", [
            self.decision(100, "You are Ada at (99, -61, 99)."),
        ])
        self.write_jsonl("orchestrator/trajectory.jsonl", [
            {"t": 140, "agent": "Ada", "pos": [-2.5, -61, 4], "skill": None, "state": "idle"},
            {"t": 100, "agent": "Ada", "pos": [-3, -61, 2.25], "skill": "walk", "state": "running"},
        ])
        self.write_jsonl("world/simharness/npc_observations.jsonl", [
            {"gameTime": tick, "npc": identity, "observer": "npc:" + identity,
             "observerEye": [position, -59.38, -position]}
            for tick, identity, position in ((100, "adult_01", 1), (100, "adult_02", 8),
                                             (120, "adult_01", 5), (180, "adult_02", 6),
                                             (200, "adult_01", 2), (200, "adult_02", 9))
        ])
        self.write_jsonl("world/simharness/welfare.jsonl", [
            {"npc": "adult_01", "day": 0, "activity": "work", "from": "home", "to": "shop",
             "departTick": 100, "arriveTick": 180, "onTime": True, "outcome": "arrived", "path": []},
            {"npc": "adult_01", "day": 0, "activity": "home", "from": "shop", "to": "home",
             "departTick": 240, "arriveTick": None, "onTime": False, "outcome": "unreachable", "path": []},
            {"type": "npc_day", "npc": "adult_01", "day": 0},
        ])
        tracks = self.data().tracks()
        self.assertEqual(tracks["agents"]["Ada"],
                         [[100, -3, 2.25, "walk", "running"], [140, -2.5, 4, None, "idle"]])
        self.assertEqual(tracks["npcs"], {"adult_01": [[100, 1, -1], [200, 2, -2]],
                                          "adult_02": [[100, 8, -8], [200, 9, -9]]})
        self.assertEqual(tracks["trips"]["adult_01"],
                         [[100, 180, "work", "shop", True, "arrived", "home"],
                          [240, None, "home", "home", False, "unreachable", "shop"]])

    def test_changes_respect_all_map_layers_and_horizontal_bounds(self):
        """Keep sorted changes on either layer inside the map's half-open bounds."""
        base_map = {"origin": -1, "size": 2, "palette": ["minecraft:air", "minecraft:stone"],
                    "layers": [{"y": -60, "grid": [0, 0, 0, 0]}, {"y": -61, "grid": [1, 1, 1, 1]}]}
        base_dir = self.runs_dir / "village_sabotage"
        base_dir.mkdir()
        (base_dir / "viewer_map.json").write_text(json.dumps(base_map), encoding="utf-8")
        self.write_jsonl("world/simharness/events.jsonl", [
            {"type": "block_change", "gameTime": tick, "pos": position,
             "from": "minecraft:stone", "to": "minecraft:air", "actor": "Ada"}
            for tick, position in ((200, [0, -61, 0]), (100, [-1, -60, -1]),
                                   (300, [1, -60, 0]), (310, [0, -60, 1]),
                                   (320, [-2, -60, 0]), (330, [0, -60, -2]), (340, [0, -62, 0]))
        ])
        run = self.data()
        self.assertEqual(run.map(), base_map)
        self.assertEqual(run.changes(), [[100, -1, -60, -1, "minecraft:air"],
                                          [200, 0, -61, 0, "minecraft:air"]])

    def test_live_cache_resumes_partial_lines_and_invalidates_rewritten_sources(self):
        """Resume complete lines after reopening and discard stale rewritten records."""
        first = {"type": "transfer", "gameTime": 100, "from": "Ada", "to": "shop", "item": "oak", "count": 1}
        source = self.write_jsonl("world/simharness/events.jsonl", [first])
        run = self.data()
        self.assertEqual([event["t"] for event in run.events()], [100])
        pending = json.dumps(dict(first, gameTime=200)).encode("utf-8")
        split = len(pending) // 2
        with source.open("ab") as handle:
            handle.write(pending[:split])
        self.assertEqual([event["t"] for event in run.events()], [100])
        run = self.data()
        self.assertEqual([event["t"] for event in run.events()], [100])
        with source.open("ab") as handle:
            handle.write(pending[split:])
        self.assertEqual([event["t"] for event in run.events()], [100])
        with source.open("ab") as handle:
            handle.write(b"\n" + json.dumps(dict(first, gameTime=300)).encode("utf-8") + b"\n")
        self.assertEqual([event["t"] for event in run.events()], [100, 200, 300])
        previous = source.stat()
        contents = source.read_bytes()
        source.write_bytes(contents.replace(b'"gameTime": 100', b'"gameTime": 105'))
        os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        self.assertEqual(source.stat().st_size, previous.st_size)
        self.assertEqual([event["t"] for event in run.events()], [105, 200, 300])
        previous = source.stat()
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(contents.replace(b'"gameTime": 100', b'"gameTime": 109'))
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(source)
        self.assertEqual([event["t"] for event in run.events()], [109, 200, 300])
        self.write_jsonl("world/simharness/events.jsonl", [dict(first, gameTime=50)])
        self.assertEqual([event["t"] for event in run.events()], [50])
        self.assertEqual([event["t"] for event in self.data().events()], [50])

    def test_warm_cache_avoids_source_redecoding_and_keeps_prompts_on_demand(self):
        """Reuse persisted indexes without copying prompt bodies into derived caches."""
        prompts = ["You are Ada at (1, -61, 2).\nCafé — 見た", "You are Ada at (3, -61, 4).\nPRIVATE_PROMPT_BODY"]
        records = [self.decision(100, prompts[0]), self.decision(200, prompts[1])]
        self.write_jsonl("orchestrator/decisions.jsonl", records)
        run = self.data()
        expected_events, expected_tracks = run.events(), run.tracks()
        cache_files = list(self.cache_dir.glob("*.json"))
        self.assertTrue(cache_files)
        for path in cache_files:
            self.assertNotIn("PRIVATE_PROMPT_BODY", path.read_text(encoding="utf-8"))
        with patch("json.loads", wraps=json.loads) as decode:
            warm = self.data()
            self.assertEqual(warm.events(), expected_events)
            self.assertEqual(warm.tracks(), expected_tracks)
        raw_records = [json.dumps(record, ensure_ascii=False) for record in records]
        decoded = [call.args[0].decode("utf-8") if isinstance(call.args[0], bytes) else call.args[0]
                   for call in decode.call_args_list]
        self.assertFalse(any(text.strip() in raw_records for text in decoded))
        self.assertIsNone(warm.prompt("Ada", 99))
        self.assertIsNone(warm.prompt("missing", 300))
        for tick, expected in ((100, prompts[0]), (199, prompts[0]), (200, prompts[1]), (300, prompts[1])):
            with self.subTest(tick=tick):
                result = warm.prompt("Ada", tick)
                self.assertEqual(result["prompt"], expected)
                self.assertEqual(result["decision"]["tool"], "say")

    def test_missing_optional_sources_and_sorted_safe_run_index(self):
        """Handle missing optional data, list newest runs first, and reject escaping paths."""
        run = self.data()
        self.assertEqual(run.events(), [])
        self.assertEqual(run.changes(), [])
        self.assertEqual(run.tracks(), {"agents": {}, "npcs": {}, "trips": {}})
        self.assertIsNone(run.prompt("Ada", 0))
        self.assertEqual(run.meta()["variant"], "sabotage")
        with self.assertRaises(FileNotFoundError):
            run.map()
        os.utime(self.run_dir / "orchestrator/decisions.jsonl", (1000, 1000))
        for tick, name in ((2000, "alpha_accident"), (3000, "zeta_sabotage")):
            log = self.runs_dir / name / "world/simharness/events.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text("", encoding="utf-8")
            os.utime(log, (tick, tick))
        (self.runs_dir / "not_a_run").mkdir()
        index = RunIndex(self.runs_dir, cache_dir=self.cache_dir, scenario_dir=self.scenario_dir)
        summaries = index.runs()
        self.assertEqual([summary["name"] for summary in summaries],
                         ["zeta_sabotage", "alpha_accident", "tiny_sabotage"])
        self.assertTrue(all(set(summary) >= {"name", "variant", "start", "end", "modified", "live"}
                            for summary in summaries))
        self.assertEqual(index.run("tiny_sabotage", variant="accident").summary()["variant"], "accident")
        self.assertTrue((self.cache_dir / "tiny_sabotage").is_dir())
        observations = self.write_jsonl("world/simharness/npc_observations.jsonl", [
            {"npc": "adult_01", "gameTime": tick, "observerEye": [0, -59.38, 0]} for tick in (120, 900)
        ])
        self.assertEqual(index.run("tiny_sabotage").summary()["end"], 900)
        index.run("tiny_sabotage").tracks()
        observations.write_text('{"npc":"adult_01","gameTime":180,"observerEye":[0,-59.38,0]}\n')
        self.assertEqual(index.run("tiny_sabotage").summary()["end"], 180)
        for name in ("../outside", "/absolute", "a/b", "missing"):
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                index.run(name)

    def test_http_windows_prompts_and_traversal_refusal(self):
        """Serve filtered JSON and prompt lookups while refusing encoded traversal."""
        first_prompt = "You are Ada at (-1, -61, 2)."
        self.write_jsonl("orchestrator/decisions.jsonl", [
            self.decision(100, first_prompt), self.decision(200, "You are Ada at (3, -61, 4)."),
        ])
        self.write_jsonl("world/simharness/events.jsonl", [
            {"type": "utterance", "gameTime": 150, "speaker": "Villager", "text": "hello"},
        ])
        index = RunIndex(self.runs_dir, cache_dir=self.cache_dir, scenario_dir=self.scenario_dir)

        def get(target):
            """Exercise the real HTTP parser and response writer without binding a socket."""
            output = BytesIO()
            connection = Mock()
            connection.makefile.return_value = BytesIO(f"GET {target} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
            connection.sendall.side_effect = output.write
            with patch.object(ViewerHandler, "log_message"):
                ViewerHandler(connection, ("127.0.0.1", 0), SimpleNamespace(index=index))
            headers, body = output.getvalue().split(b"\r\n\r\n", 1)
            return headers, json.loads(body)

        base = "/api/runs/"
        headers, events = get(base + "tiny_sabotage/events?from=150&to=200")
        self.assertIn(b"200 OK", headers)
        self.assertIn(b"Content-Type: application/json", headers)
        self.assertEqual([(event["t"], event["type"]) for event in events],
                         [(150, "utterance"), (200, "decision")])
        self.assertTrue(all("user_prompt" not in event for event in events))
        _, result = get(base + "tiny_sabotage/prompt?agent=Ada&t=199")
        self.assertEqual(result["prompt"], first_prompt)
        headers, result = get(base + "%2e%2e%2foutside/events")
        self.assertIn(b"404 Not Found", headers)
        self.assertIn("error", result)
        headers, result = get(base + "tiny_sabotage/events?from=200&to=150")
        self.assertIn(b"400 Bad Request", headers)
        self.assertIn("error", result)
        for endpoint, kind in (("meta", dict), ("tracks", dict), ("changes", list)):
            headers, result = get(base + "tiny_sabotage/" + endpoint)
            self.assertIn(b"200 OK", headers)
            self.assertIsInstance(result, kind)
        headers, result = get(base.rstrip("/"))
        self.assertIn(b"200 OK", headers)
        self.assertEqual(result[0]["name"], "tiny_sabotage")
        for target in (base + "tiny_sabotage/map", "/STATUS.md"):
            headers, result = get(target)
            self.assertIn(b"404 Not Found", headers)
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
