"""Offline DuckDB index of public run evidence (requires DuckDB for Python 3.11).

Run ``python3.11 scripts/run_index.py RUN [RUN ...] --db runs/index.duckdb``
to index directories. Directory basenames identify runs: indexing the same name
atomically replaces all its rows, including when its directory moves. The default
database is relative to the working directory, never to an input run.

Queries accept a directory or an already indexed run name as each positional RUN::

    python3.11 scripts/run_index.py RUN --query timeline --agent builder --from-tick 6000
    python3.11 scripts/run_index.py RUN --query utterances --agent builder --other-agent mayor
    python3.11 scripts/run_index.py RUN --query meetings --radius 8
    python3.11 scripts/run_index.py FIRST SECOND --query diff --agent builder --from-tick 6000

All query methods return lists of dictionaries and read only the index. Timelines
merge decisions, attributed/heard/directed events and position samples. Utterance
queries include actual hearers and directed addressees, not merely co-listeners.
Diffs pair the nth chronological decision after the cutoff in each run, NOT equal
ticks or causally equivalent actions; a missing side is None. Meeting endpoints
are inclusive sampled ticks, without interpolation or carrying missing positions.
Outcomes retain a summary row (agent=None) per report plus its per-agent results.
Only whitelisted decision fields are indexed; prompts, briefs and args.note are
never copied from decisions/scenarios. Event raw JSON is retained as recorded.
"""

from __future__ import annotations

import argparse
from itertools import combinations, groupby, zip_longest
import json
import math
from pathlib import Path

import duckdb

if __package__:
    from .run_views import RunViews
else:
    from run_views import RunViews


DECISION_FIELDS = ("agent", "game_time", "tool", "args", "outcome", "source", "model",
                   "input_tokens", "cached_tokens", "output_tokens", "seconds")
CHILD_TABLES = ("decisions", "events", "utterances", "trajectory", "outcomes")
SCHEMA = {
    "runs": """name VARCHAR PRIMARY KEY, directory VARCHAR NOT NULL, variant VARCHAR, model VARCHAR,
               start_tick BIGINT, end_tick BIGINT, code_commit VARCHAR, roles VARCHAR[]""",
    "decisions": """run_name VARCHAR, row_index BIGINT, tick BIGINT NOT NULL, agent VARCHAR NOT NULL,
                    tool VARCHAR, args JSON, outcome JSON, source VARCHAR, model VARCHAR,
                    input_tokens BIGINT, cached_tokens BIGINT, output_tokens BIGINT, seconds DOUBLE,
                    raw_json JSON, PRIMARY KEY (run_name, row_index)""",
    "events": """run_name VARCHAR, row_index BIGINT, tick BIGINT NOT NULL, event_id VARCHAR,
                 type VARCHAR, agent VARCHAR, other_agent VARCHAR, participants VARCHAR[], raw_json JSON,
                 PRIMARY KEY (run_name, row_index)""",
    "utterances": """run_name VARCHAR, row_index BIGINT, tick BIGINT NOT NULL, day BIGINT, clock VARCHAR,
                     speaker VARCHAR, speaker_key VARCHAR, volume VARCHAR, to_agent VARCHAR, text VARCHAR,
                     listeners VARCHAR[], event_id VARCHAR, raw_json JSON, PRIMARY KEY (run_name, row_index)""",
    "trajectory": """run_name VARCHAR, row_index BIGINT, tick BIGINT NOT NULL, agent VARCHAR NOT NULL,
                     x DOUBLE, y DOUBLE, z DOUBLE, skill JSON, state VARCHAR, raw_json JSON,
                     PRIMARY KEY (run_name, row_index)""",
    "outcomes": """run_name VARCHAR, row_index BIGINT, tick BIGINT, source VARCHAR, agent VARCHAR,
                   outcome JSON, raw_json JSON, PRIMARY KEY (run_name, row_index)""",
}


def _json(value: object) -> str:
    """Encode deterministic JSON, rejecting nonfinite values instead of storing invalid evidence."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _tick(row: dict) -> int:
    """Read the three recorded log tick spellings, defaulting untimed metadata to zero."""
    return int(next((row[key] for key in ("game_time", "gameTime", "t") if key in row), 0))


def _public_args(value: object) -> object:
    """Copy tool arguments without private notes, including notes inside compound tool arguments."""
    if isinstance(value, dict):
        return {key: _public_args(item) for key, item in value.items() if key not in {"note", "user_prompt"}}
    if isinstance(value, list):
        return [_public_args(item) for item in value]
    return value


class RunIndex:
    """Own one DuckDB connection and replace/query runs without modifying source directories."""

    def __init__(self, path: Path | str):
        """Open or create the index and its six tables; ':memory:' is useful for small analyses."""
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))
        for table, columns in SCHEMA.items():
            self.connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")

    def __enter__(self) -> RunIndex:
        """Return this index for a context-managed connection lifetime."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Close the connection on success or failure."""
        self.close()

    def close(self) -> None:
        """Release the database connection and its file handles."""
        self.connection.close()

    def _rows(self, sql: str, parameters: list) -> list[dict]:
        """Return named SQL results with JSON columns decoded to Python values."""
        cursor = self.connection.execute(sql, parameters)
        columns = cursor.description
        return [{column[0]: json.loads(value) if value is not None and str(column[1]) == "JSON" else value
                 for column, value in zip(columns, row)} for row in cursor.fetchall()]

    def _insert_rows(self, table: str, rows: list[tuple]) -> None:
        """Insert rows into an internal schema table using bound values, never interpolated evidence."""
        if rows:
            placeholders = ", ".join("?" for _ in rows[0])
            self.connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

    def index_run(self, run_dir: Path | str) -> dict:
        """Atomically replace a directory basename's rows and return its public run metadata.

        Missing optional logs/reports are empty. A missing scenario, malformed log
        or insertion error leaves any previous index of this run intact. Report
        start/end ticks override observed extrema, with m5_report taking priority.
        Multiple recorded models are represented by a sorted comma-separated list.
        """
        directory = Path(run_dir).resolve()
        if not (directory / "scenario.json").is_file():
            raise ValueError(f"run directory must contain scenario.json: {directory}")
        views = RunViews(directory)
        name = directory.name
        reports = [(source, views.read_json(source)) for source in ("m5_report.json", "outcome.json")
                   if (directory / source).is_file()]
        provenance = views.read_json("provenance.json")
        commit = next((entry["commit"] for entry in reversed(provenance.get("starts", []))
                       if entry.get("commit")), provenance.get("commit"))
        ticks = [_tick(row) for row in views.decisions + views.events + views.trajectory + views.observations]
        start = next((report[key] for _, report in reports for key in ("start_tick", "start") if key in report),
                     min(ticks, default=0))
        end = next((report[key] for _, report in reports for key in ("end_tick", "end") if key in report),
                   max(ticks, default=int(start)))
        models = sorted({row["model"] for row in views.decisions if row.get("model")})
        model = ", ".join(models) if models else views.scenario.get("model")
        roles = [role["name"] for role in views.scenario.get("roles", [])]
        metadata = {"name": name, "directory": str(directory), "variant": views.scenario.get("variant"),
                    "model": model, "start_tick": int(start), "end_tick": int(end), "code_commit": commit,
                    "roles": roles}
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for table in CHILD_TABLES:
                self.connection.execute(f"DELETE FROM {table} WHERE run_name = ?", [name])
            self.connection.execute("DELETE FROM runs WHERE name = ?", [name])
            self._insert_rows("runs", [tuple(metadata.values())])
            self._index_decisions(name, views.decisions)
            self._index_events(name, views.events)
            self._insert_rows("utterances", [
                (name, index, row["tick"], row["day"], row["clock"], row["speaker"], row.get("speaker_key"),
                 row.get("volume"), row.get("to"), row.get("text"), row.get("listeners", []),
                 row.get("event_id"), _json(row)) for index, row in enumerate(views.speech())])
            self._index_trajectory(name, views.trajectory)
            self._index_outcomes(name, reports, metadata["end_tick"], roles)
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return metadata

    def _index_decisions(self, name: str, decisions: list[dict]) -> None:
        """Index only public action/accounting fields, not prompts or arbitrary future log fields."""
        rows = []
        for index, original in enumerate(decisions):
            row = {key: original[key] for key in DECISION_FIELDS if key in original}
            if "args" in row:
                row["args"] = _public_args(row["args"])
            rows.append((name, index, _tick(row), row["agent"], row.get("tool"), _json(row.get("args")),
                         _json(row.get("outcome")), row.get("source"), row.get("model"), row.get("input_tokens"),
                         row.get("cached_tokens"), row.get("output_tokens"), row.get("seconds"), _json(row)))
        self._insert_rows("decisions", rows)

    def _index_events(self, name: str, events: list[dict]) -> None:
        """Attribute real engine actor fields, retaining transfer endpoints and utterance hearers."""
        rows = []
        for index, row in enumerate(events):
            actors = [row[key] for key in ("actor", "bot", "from", "buyer", "author", "rider",
                                          "speakerKey", "speaker", "agent") if isinstance(row.get(key), str)]
            participants = actors + ([row["to"]] if isinstance(row.get("to"), str) else [])
            if row.get("type") == "utterance":
                participants += row.get("listeners", [])
            rows.append((name, index, _tick(row), row.get("id"), row.get("type"),
                         actors[0] if actors else None, row.get("to"), sorted(set(participants)), _json(row)))
        self._insert_rows("events", rows)

    def _index_trajectory(self, name: str, trajectory: list[dict]) -> None:
        """Expose typed coordinates and preserve the original position/skill/state sample."""
        rows = []
        for index, row in enumerate(trajectory):
            position = row.get("pos")
            if isinstance(position, dict):
                position = [position[key] for key in ("x", "y", "z")]
            coordinates = (None, None, None) if position is None else tuple(position)
            if len(coordinates) != 3:
                raise ValueError("trajectory positions must have three coordinates")
            rows.append((name, index, _tick(row), row["agent"], *coordinates, _json(row.get("skill")),
                         row.get("state"), _json(row)))
        self._insert_rows("trajectory", rows)

    def _index_outcomes(self, name: str, reports: list[tuple[str, dict]], end: int, roles: list[str]) -> None:
        """Retain full reports once and expose nested or direct per-agent outcome mappings."""
        rows = []
        for source, report in reports:
            tick = report.get("end_tick", report.get("end", end))
            rows.append((name, len(rows), tick, source, None, _json(None), _json(report)))
            outcomes = next((report[key] for key in ("outcomes", "roles", "agents", "shadow_outcomes")
                             if isinstance(report.get(key), dict)),
                            {role: report[role] for role in roles if role in report})
            for agent, outcome in outcomes.items():
                rows.append((name, len(rows), tick, source, agent, _json(outcome), _json(outcome)))
        self._insert_rows("outcomes", rows)

    def _agent_keys(self, run: str, agent: str) -> list[str]:
        """Resolve recorded NPC display names/keys and role-prefixed identities without guessing NPC names."""
        keys = {agent}
        metadata = self._rows("SELECT roles FROM runs WHERE name = ?", [run])
        roles = metadata[0]["roles"] if metadata else []
        for role in roles:
            variants = {role, f"agent:{role}", f"bot:{role}"}
            if agent in variants:
                keys.update(variants)
        speakers = self._rows("""SELECT DISTINCT speaker, speaker_key FROM utterances WHERE run_name = ?
                                 AND (list_contains(?, speaker) OR list_contains(?, speaker_key))""",
                              [run, sorted(keys), sorted(keys)])
        keys.update(value for row in speakers for value in row.values() if value)
        return sorted(keys)

    def per_agent_timeline(self, run: Path | str, agent: str, from_tick: int = 0) -> list[dict]:
        """Return tick/kind/data records, with ties ordered decisions, events, then trajectory.

        Events include actions attributed to the agent, either transfer endpoint,
        speech they made/heard, and speech directed at them even if not heard.
        The cutoff is inclusive; data contains the original public/sanitized row.
        """
        name = Path(run).name
        keys = self._agent_keys(name, agent)
        return self._rows("""SELECT tick, kind, data FROM (
            SELECT tick, 'decision' AS kind, raw_json AS data, 0 AS kind_order, row_index
            FROM decisions WHERE run_name = ? AND tick >= ? AND list_contains(?, agent)
            UNION ALL
            SELECT tick, 'event' AS kind, raw_json AS data, 1 AS kind_order, row_index
            FROM events WHERE run_name = ? AND tick >= ? AND list_has_any(participants, ?)
            UNION ALL
            SELECT tick, 'trajectory' AS kind, raw_json AS data, 2 AS kind_order, row_index
            FROM trajectory WHERE run_name = ? AND tick >= ? AND list_contains(?, agent)
            ) ORDER BY tick, kind_order, row_index""", [name, from_tick, keys] * 3)

    def utterances_between(self, run: Path | str, first: str, second: str, from_tick: int = 0) -> list[dict]:
        """Return public speech in either direction, matching hearers OR directed addressees.

        Speakers may be role names, recorded NPC display names or speakerKey
        identities. Two agents merely hearing a third speaker are not a match.
        """
        name = Path(run).name
        rows = self._rows("""WITH wanted AS (SELECT ? AS first_keys, ? AS second_keys)
            SELECT raw_json FROM utterances, wanted WHERE run_name = ? AND tick >= ? AND (
                ((list_contains(first_keys, speaker) OR list_contains(first_keys, speaker_key))
                 AND (list_has_any(listeners, second_keys) OR list_contains(second_keys, to_agent)))
                OR ((list_contains(second_keys, speaker) OR list_contains(second_keys, speaker_key))
                    AND (list_has_any(listeners, first_keys) OR list_contains(first_keys, to_agent)))
            ) ORDER BY tick, row_index""",
                          [self._agent_keys(name, first), self._agent_keys(name, second), name, from_tick])
        return [row["raw_json"] for row in rows]

    def meetings(self, run: Path | str, radius: float = 8) -> list[dict]:
        """Return pairwise role meetings at simultaneous samples, with inclusive sampled endpoints.

        Missing/far positions end an interval at its last near sample. The index
        never interpolates motion or reads potentially changed source logs.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        name = Path(run).name
        metadata = self._rows("SELECT roles FROM runs WHERE name = ?", [name])
        roles = set(metadata[0]["roles"]) if metadata else set()
        samples = self._rows("""SELECT tick, agent, x, y, z FROM trajectory WHERE run_name = ?
                                ORDER BY tick, row_index""", [name])
        active, meetings = {}, []
        for tick, group in groupby(samples, key=lambda row: row["tick"]):
            positions = {row["agent"]: (row["x"], row["y"], row["z"]) for row in group
                         if row["agent"] in roles and all(row[key] is not None for key in ("x", "y", "z"))}
            nearby = {pair for pair in combinations(sorted(positions), 2)
                      if math.dist(positions[pair[0]], positions[pair[1]]) <= radius}
            for pair in set(active) - nearby:
                meetings.append(active.pop(pair))
            for pair in nearby:
                meeting = active.setdefault(pair, {"start_tick": tick, "end_tick": tick, "agents": list(pair)})
                meeting["end_tick"] = tick
        meetings.extend(active.values())
        return sorted(meetings, key=lambda row: (row["start_tick"], row["agents"], row["end_tick"]))

    def paired_branch_diff(self, first_run: Path | str, second_run: Path | str, agent: str,
                           from_tick: int = 0) -> list[dict]:
        """Pair chronological decision sequences by ordinal, padding the shorter branch with None.

        Each result has sequence (zero-based), first and second. Decisions at
        unequal ticks can pair; equal-tick ties keep their original file order.
        This is a comparison aid, not a claim of causal/tick alignment.
        """
        branches = []
        for run in (first_run, second_run):
            name = Path(run).name
            rows = self._rows("""SELECT raw_json FROM decisions WHERE run_name = ? AND tick >= ?
                                 AND list_contains(?, agent) ORDER BY tick, row_index""",
                              [name, from_tick, self._agent_keys(name, agent)])
            branches.append([row["raw_json"] for row in rows])
        return [{"sequence": index, "first": first, "second": second}
                for index, (first, second) in enumerate(zip_longest(*branches))]


def main(argv: list[str] | None = None) -> int:
    """Index directories or print one documented query as JSON, without writing to a source run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", metavar="RUN")
    parser.add_argument("--db", type=Path, default=Path("runs/index.duckdb"))
    parser.add_argument("--query", choices=("timeline", "utterances", "meetings", "diff"))
    parser.add_argument("--agent")
    parser.add_argument("--other-agent")
    parser.add_argument("--from-tick", type=int, default=0)
    parser.add_argument("--radius", type=float, default=8)
    args = parser.parse_args(argv)
    if args.query and len(args.runs) != (2 if args.query == "diff" else 1):
        parser.error("queries require one RUN (two for diff)")
    if args.query in {"timeline", "utterances", "diff"} and not args.agent:
        parser.error(f"--query {args.query} requires --agent")
    if args.query == "utterances" and not args.other_agent:
        parser.error("--query utterances requires --other-agent")
    try:
        with RunIndex(args.db) as index:
            result = []
            for run in args.runs:
                if run.is_dir() or not args.query:
                    result.append(index.index_run(run))
                elif not index._rows("SELECT name FROM runs WHERE name = ?", [run.name]):
                    raise ValueError(f"unknown indexed run: {run}")
            if args.query == "timeline":
                result = index.per_agent_timeline(args.runs[0], args.agent, args.from_tick)
            elif args.query == "utterances":
                result = index.utterances_between(args.runs[0], args.agent, args.other_agent, args.from_tick)
            elif args.query == "meetings":
                result = index.meetings(args.runs[0], args.radius)
            elif args.query == "diff":
                result = index.paired_branch_diff(*args.runs, args.agent, args.from_tick)
    except (OSError, ValueError, TypeError, KeyError, duckdb.Error) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
