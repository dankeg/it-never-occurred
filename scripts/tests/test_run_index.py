"""Offline DuckDB indexing and queries over small real-format synthetic runs."""

from contextlib import chdir, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb

from scripts.run_index import CHILD_TABLES, RunIndex, main
from scripts.run_views import RunViews


def write_json(directory: Path, name: str, value: object) -> None:
    """Write one synthetic JSON artifact without touching any real run."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(directory: Path, name: str, rows: list[dict]) -> None:
    """Write a small log in its provided order, including deliberate out-of-order ticks."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def decision(tick: int, agent: str = "builder", tool: str = "wait") -> dict:
    """Make a decision with private fields that must never enter the index."""
    return {"game_time": tick, "agent": agent, "tool": tool, "args": {"ticks": 10, "note": "PRIVATE_NOTE"},
            "outcome": "started", "model": "test-model", "source": "model", "input_tokens": 40,
            "cached_tokens": 10, "output_tokens": 5, "seconds": 0.25, "user_prompt": "PRIVATE_PROMPT",
            "future_private_field": "PRIVATE_FUTURE"}


def event(tick: int, event_type: str, **fields) -> dict:
    """Build one engine event with its actual event/time field names."""
    return {"id": tick, "gameTime": tick, "type": event_type, **fields}


def sample(tick: int, agent: str, position: list | dict | None) -> dict:
    """Build one recorded trajectory row."""
    return {"t": tick, "agent": agent, "pos": position, "skill": None, "state": "idle"}


class RunIndexTests(unittest.TestCase):
    """Exercise replacement, rollback, privacy and offline query semantics through persisted logs."""

    def setUp(self):
        """Isolate both source runs and the database in a temporary directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "index" / "runs.duckdb"
        self.index = RunIndex(self.database)
        self.addCleanup(self.index.close)

    def run_directory(self, name: str = "original", roles: tuple[str, ...] = ("builder", "mayor")) -> Path:
        """Create a complete tiny run with optional empty logs and actual report/provenance shapes."""
        directory = self.root / name
        directory.mkdir()
        write_json(directory, "scenario.json", {
            "variant": "sabotage", "roles": [{"name": role, "brief": "PRIVATE_BRIEF"} for role in roles]})
        write_jsonl(directory, "orchestrator/decisions.jsonl", [decision(20), decision(10)])
        write_jsonl(directory, "world/simharness/events.jsonl", [
            event(15, "skill_end", bot=roles[0], skill="wait", skillId=1, outcome="done", reason="finished"),
            event(12, "utterance", speaker=roles[0], speakerKey=roles[0], volume="talk", to=roles[1],
                  text="Public message", listeners=[roles[1]], startedAt=11)])
        write_jsonl(directory, "orchestrator/trajectory.jsonl", [sample(0, roles[0], [0, 0, 0]),
                                                                  sample(0, roles[1], [5, 0, 0])])
        write_jsonl(directory, "world/simharness/observations.jsonl", [
            {"gameTime": 30, "observer": roles[0], "items": [
                {"category": "self", "name": roles[0], "pos": [0, 0, 0], "skill": {"skillId": 1}}]}])
        write_json(directory, "provenance.json", {
            "commit": "legacy", "starts": [{"commit": "first", "time": "earlier"},
                                            {"commit": "latest", "time": "later"}]})
        write_json(directory, "m5_report.json", {
            "end": 48000, "outcomes": {roles[0]: True, roles[1]: None},
            "stats": {"transfers": 1}, "first_true": {"bridge_complete": 12000}})
        return directory

    def snapshot(self) -> dict:
        """Read every persisted row to detect partial replacement across all six tables."""
        return {table: self.index.connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
                for table in ("runs", *CHILD_TABLES)}

    def test_metadata_reports_and_decision_privacy(self):
        """Last provenance commit and real report end win; private decision/scenario content is absent."""
        directory = self.run_directory()
        write_json(directory, "outcome.json", {"builder": {"success": False}, "mayor": True})
        rows = [decision(20), decision(10)]
        rows[0]["args"]["steps"] = [{"tool": "wait", "args": {"note": "PRIVATE_NESTED", "ticks": 2}}]
        write_jsonl(directory, "orchestrator/decisions.jsonl", rows)
        metadata = self.index.index_run(directory)
        self.assertEqual((metadata["name"], metadata["variant"], metadata["model"]),
                         ("original", "sabotage", "test-model"))
        self.assertEqual((metadata["start_tick"], metadata["end_tick"], metadata["code_commit"]),
                         (0, 48000, "latest"))
        outcomes = self.index._rows("SELECT source, agent, outcome, raw_json FROM outcomes", [])
        self.assertEqual(len(outcomes), 6)
        summary = next(row for row in outcomes if row["source"] == "m5_report.json" and row["agent"] is None)
        self.assertEqual(summary["raw_json"]["first_true"], {"bridge_complete": 12000})
        self.assertEqual(summary["raw_json"]["stats"], {"transfers": 1})
        values = {(row["source"], row["agent"]): row["outcome"] for row in outcomes if row["agent"]}
        self.assertEqual(values[("m5_report.json", "builder")], True)
        self.assertIsNone(values[("m5_report.json", "mayor")])
        self.assertEqual(values[("outcome.json", "builder")], {"success": False})
        public = self.index._rows("SELECT args, raw_json, input_tokens, cached_tokens FROM decisions", [])
        self.assertEqual(public[0]["input_tokens"], 40)
        self.assertEqual(public[0]["cached_tokens"], 10)
        self.assertEqual(public[1]["args"]["steps"][0]["args"], {"ticks": 2})
        self.assertNotIn("PRIVATE_", repr(self.snapshot()))

    def test_reindex_replaces_all_rows_and_preserves_other_runs(self):
        """Repeated indexing is idempotent; removed files remove stale rows without touching another run."""
        directory = self.run_directory()
        other = self.run_directory("other")
        self.index.index_run(directory)
        self.index.index_run(other)
        before = self.snapshot()
        self.index.index_run(directory)
        self.assertEqual(self.snapshot(), before)
        write_jsonl(directory, "orchestrator/decisions.jsonl", [decision(40, tool="keep_going")])
        for name in ("world/simharness/events.jsonl", "orchestrator/trajectory.jsonl", "m5_report.json"):
            (directory / name).unlink()
        write_json(directory, "provenance.json", {"commit": "direct"})
        metadata = self.index.index_run(directory)
        self.assertEqual((metadata["code_commit"], metadata["start_tick"], metadata["end_tick"]), ("direct", 30, 40))
        for table in CHILD_TABLES:
            count = self.index.connection.execute(f"SELECT count(*) FROM {table} WHERE run_name = ?",
                                                  [directory.name]).fetchone()[0]
            self.assertEqual(count, 1 if table == "decisions" else 0)
            self.assertEqual([row for row in self.snapshot()[table] if row[0] == other.name],
                             [row for row in before[table] if row[0] == other.name])
        with RunIndex(self.database) as reopened:
            self.assertEqual(reopened.per_agent_timeline("original", "builder")[0]["data"]["tool"], "keep_going")
        with self.assertRaises(duckdb.ConnectionException):
            reopened.connection.execute("SELECT 1")

    def test_invalid_replacement_rolls_back_every_table(self):
        """A typed insertion failure after run/decision/event writes restores the full previous transaction."""
        directory = self.run_directory()
        self.index.index_run(directory)
        before = self.snapshot()
        write_jsonl(directory, "orchestrator/decisions.jsonl", [decision(100, tool="new-tool")])
        write_jsonl(directory, "orchestrator/trajectory.jsonl", [sample(100, "builder", ["not-a-number", 0, 0])])
        write_json(directory, "scenario.json", {"variant": "changed", "roles": [{"name": "builder"}]})
        with self.assertRaises(duckdb.ConversionException):
            self.index.index_run(directory)
        self.assertEqual(self.snapshot(), before)
        write_jsonl(directory, "orchestrator/trajectory.jsonl", [])
        self.index.index_run(directory)
        self.assertEqual(self.index.per_agent_timeline(directory, "builder", 100)[0]["data"]["tool"], "new-tool")

    def test_malformed_input_and_missing_scenario_leave_previous_rows(self):
        """Invalid source logs and mistyped run paths cannot silently erase an indexed run."""
        directory = self.run_directory()
        self.index.index_run(directory)
        before = self.snapshot()
        (directory / "orchestrator/decisions.jsonl").write_text("{broken\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.index.index_run(directory)
        with self.assertRaises(ValueError):
            self.index.index_run(self.root / "missing")
        self.assertEqual(self.snapshot(), before)

    def test_timeline_attribution_and_cutoff(self):
        """All actual actor fields, transfer receivers and heard/directed speech enter chronological timelines."""
        directory = self.run_directory()
        events = [event(22, "transfer", **{"from": "mayor", "to": "builder", "item": "bread", "count": 2}),
                  event(18, "purchase", buyer="builder", item="bread", count=1),
                  event(19, "doc_write", author="builder", title="Report", text="public"),
                  event(21, "ferry_trip", rider="builder"),
                  event(20, "block_change", actor="builder", pos=[0, 0, 0]),
                  event(15, "skill_end", bot="builder", skill="wait", skillId=1, outcome="done"),
                  event(23, "utterance", speaker="mayor", speakerKey="mayor", volume="talk",
                        to=None, text="Heard", listeners=["builder"]),
                  event(24, "utterance", speaker="mayor", speakerKey="mayor", volume="talk",
                        to="builder", text="Directed but not heard", listeners=[]),
                  event(25, "purchase", buyer="mayor", item="bread", count=1)]
        write_jsonl(directory, "world/simharness/events.jsonl", events)
        write_jsonl(directory, "orchestrator/trajectory.jsonl", [sample(20, "builder", [0, 0, 0])])
        self.index.index_run(directory)
        timeline = self.index.per_agent_timeline(directory, "builder", 15)
        self.assertEqual([row["tick"] for row in timeline], [15, 18, 19, 20, 20, 20, 21, 22, 23, 24])
        self.assertEqual([row["kind"] for row in timeline if row["tick"] == 20],
                         ["decision", "event", "trajectory"])
        self.assertNotIn("PRIVATE_", json.dumps(timeline))
        self.assertEqual(self.index.per_agent_timeline(directory, "agent:builder", 15), timeline)
        self.assertEqual(self.index.per_agent_timeline("unknown", "builder"), [])

    def test_utterances_match_listeners_addressees_and_npc_keys(self):
        """Broadcast hearers and directed targets match in both directions; unrelated/co-heard speech does not."""
        directory = self.run_directory()
        specs = [(5, "mayor", "mayor", None, ["builder"]),
                 (6, "builder", "builder", None, ["npc:oak_1"]),
                 (7, "Oak", "npc:oak_1", "builder", []),
                 (8, "builder", "builder", "mayor", []),
                 (9, "mayor", "mayor", "third", ["third"]),
                 (10, "third", "third", None, ["mayor", "builder"]),
                 (11, "builder", "builder", "third", ["mayor"])]
        write_jsonl(directory, "world/simharness/events.jsonl", [
            event(tick, "utterance", speaker=speaker, speakerKey=key, to=recipient, listeners=listeners,
                  text=f"public {tick}", volume="talk", startedAt=tick - 1)
            for tick, speaker, key, recipient, listeners in reversed(specs)])
        self.index.index_run(directory)
        self.assertEqual([row["tick"] for row in self.index.utterances_between(directory, "mayor", "builder")],
                         [5, 8, 11])
        self.assertEqual([row["tick"] for row in self.index.utterances_between(directory, "builder", "mayor", 8)],
                         [8, 11])
        by_key = self.index.utterances_between(directory, "builder", "npc:oak_1")
        self.assertEqual([row["tick"] for row in by_key], [6, 7])
        self.assertEqual(by_key, self.index.utterances_between(directory, "Oak", "builder"))
        self.assertEqual(by_key[0]["listeners"], ["npc:oak_1"])
        self.assertEqual(by_key[1]["speaker_key"], "npc:oak_1")
        self.assertEqual([row["tick"] for row in self.index.per_agent_timeline(directory, "Oak", 5)], [6, 7])

    def test_meetings_use_indexed_simultaneous_role_samples(self):
        """Pair intervals include boundary/final ticks, split on missing positions, and exclude NPCs."""
        directory = self.run_directory()
        rows = []
        for tick, distance in ((0, 8), (10, 5), (20, 9), (30, 0), (50, 0)):
            rows += [sample(tick, "mayor", [distance, 0, 0]), sample(tick, "builder", {"x": 0, "y": 0, "z": 0}),
                     sample(tick, "npc:oak_1", [0, 0, 0])]
        rows.append(sample(40, "builder", [0, 0, 0]))
        write_jsonl(directory, "orchestrator/trajectory.jsonl", list(reversed(rows)))
        self.index.index_run(directory)
        expected = [{"start_tick": start, "end_tick": end, "agents": ["builder", "mayor"]}
                    for start, end in ((0, 10), (30, 30), (50, 50))]
        self.assertEqual(self.index.meetings(directory), expected)
        self.assertEqual(self.index.meetings(directory, 0), expected[1:])
        self.assertEqual(self.index.meetings(directory, 4), expected[1:])
        (directory / "orchestrator/trajectory.jsonl").unlink()
        self.assertEqual(self.index.meetings(directory), expected)
        for radius in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.index.meetings(directory, radius)

    def test_branch_diff_pairs_ordinal_sequences_not_ticks(self):
        """Different ticks pair by sequence, with stable equal-tick ordering and None padding."""
        first = self.run_directory("first")
        second = self.run_directory("second")
        write_jsonl(first, "orchestrator/decisions.jsonl", [decision(30, tool="last"), decision(5),
                                                          decision(10, tool="first")])
        write_jsonl(second, "orchestrator/decisions.jsonl", [decision(35, tool="tail"), decision(20, tool="one"),
                                                           decision(20, tool="two"), decision(15, "mayor")])
        self.index.index_run(first)
        self.index.index_run(second)
        diff = self.index.paired_branch_diff(first, second, "builder", 10)
        self.assertEqual([row["sequence"] for row in diff], [0, 1, 2])
        self.assertEqual([(row["first"]["tool"] if row["first"] else None, row["second"]["tool"]) for row in diff],
                         [("first", "one"), ("last", "two"), (None, "tail")])
        self.assertEqual((diff[0]["first"]["game_time"], diff[0]["second"]["game_time"]), (10, 20))
        self.assertNotIn("PRIVATE_", json.dumps(diff))

    def test_three_role_meeting_order_matches_views(self):
        """Shared-start intervals sort by role pair before end tick, identically to per-run views."""
        directory = self.run_directory(roles=("builder", "mayor", "envoy"))
        rows = [sample(tick, agent, [position, 0, 0])
                for tick, positions in ((0, (0, 2, 4)), (10, (0, 1, 20)), (20, (30, 0, 20)))
                for agent, position in zip(("builder", "envoy", "mayor"), positions)]
        write_jsonl(directory, "orchestrator/trajectory.jsonl", rows)
        self.index.index_run(directory)
        meetings = self.index.meetings(directory)
        self.assertEqual(meetings, RunViews(directory).meetings())
        self.assertEqual(meetings[0], {"start_tick": 0, "end_tick": 10, "agents": ["builder", "envoy"]})

    def test_sql_quoting_in_run_and_agent_names(self):
        """Names containing SQL syntax remain bound data in replacements and every query."""
        agent = "O'Brien'; DROP TABLE runs; --"
        directory = self.run_directory("run'; DELETE FROM decisions; --", (agent, "mayor"))
        write_jsonl(directory, "orchestrator/decisions.jsonl", [decision(10, agent)])
        self.index.index_run(directory)
        self.index.index_run(directory)
        self.assertEqual(self.index.per_agent_timeline(directory, agent, 10)[0]["data"]["agent"], agent)
        self.assertEqual(len(self.index.utterances_between(directory, agent, "mayor")), 1)
        self.assertEqual(len(self.index.meetings(directory)), 1)
        self.assertEqual(len(self.index.paired_branch_diff(directory, directory, agent)), 1)
        self.assertEqual(self.index.connection.execute("SELECT count(*) FROM runs").fetchone()[0], 1)

    def test_cli_default_database_and_stored_name_queries(self):
        """The default database is worktree-relative, not placed in the source run; queries can use saved names."""
        directory = self.run_directory()
        source_before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*")
                         if path.is_file()}
        stdout = StringIO()
        with chdir(self.root), redirect_stdout(stdout):
            with patch("scripts.run_index.RunIndex", wraps=RunIndex) as constructor:
                self.assertEqual(main([str(directory)]), 0)
                constructor.assert_called_once_with(Path("runs/index.duckdb"))
        self.assertEqual(json.loads(stdout.getvalue())[0]["end_tick"], 48000)
        default_database = self.root / "runs/index.duckdb"
        self.assertTrue(default_database.is_file())
        self.assertEqual({path.relative_to(directory): path.read_bytes() for path in directory.rglob("*")
                          if path.is_file()}, source_before)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([directory.name, "--db", str(default_database), "--query", "timeline",
                                   "--agent", "builder", "--from-tick", "20"]), 0)
        self.assertEqual([row["tick"] for row in json.loads(stdout.getvalue())], [20])

    def test_cli_query_modes(self):
        """All four documented query choices produce JSON with the promised command-line flags."""
        directory = self.run_directory()
        self.index.index_run(directory)
        cases = [("utterances", ["--agent", "builder", "--other-agent", "mayor"], 1),
                 ("meetings", ["--radius", "8"], 1),
                 ("diff", ["--agent", "builder", "--from-tick", "20"], 1)]
        for query, flags, expected_count in cases:
            with self.subTest(query=query):
                stdout = StringIO()
                runs = [directory.name] * (2 if query == "diff" else 1)
                with redirect_stdout(stdout):
                    self.assertEqual(main([*runs, "--db", str(self.database), "--query", query, *flags]), 0)
                self.assertEqual(len(json.loads(stdout.getvalue())), expected_count)

    def test_empty_optional_files_and_nested_outcome_mapping(self):
        """A scenario alone is indexable; optional per-agent outcome mappings remain typed JSON."""
        directory = self.root / "empty"
        directory.mkdir()
        write_json(directory, "scenario.json", {"roles": [{"name": "builder"}], "variant": "sabotage"})
        metadata = self.index.index_run(directory)
        self.assertEqual((metadata["start_tick"], metadata["end_tick"], metadata["code_commit"], metadata["model"]),
                         (0, 0, None, None))
        self.assertEqual(self.index.per_agent_timeline(directory, "builder"), [])
        self.assertEqual(self.index.meetings(directory), [])
        write_json(directory, "outcome.json", {"end": 50, "agents": {"builder": False}})
        metadata = self.index.index_run(directory)
        self.assertEqual(metadata["end_tick"], 50)
        self.assertEqual(self.index._rows("SELECT outcome FROM outcomes WHERE agent = ?", ["builder"]),
                         [{"outcome": False}])


if __name__ == "__main__":
    unittest.main()
