"""Commitments against acts (Section 4.1): do committee members do what they say they will do?

A member's **commitment** is its own statement, in speech or in a note, that it (or the committee, "we") will do a
specific task: "I'll bake 32 tonight", "we'll put the loaves out at 17:00". Requests to others and plans without a
doer are not commitments. A judge (Sol, batched, structured; validated before use) reads each candidate text (filtered
by future or volunteering phrasing) and returns whether it commits the speaker and to which task:

- `bake` (bake bread), `get_grain` (buy or fetch wheat or hay), `move_bread` (carry or deliver bread anywhere),
  `store_in_hall` (put bread in the hall's tables), `lanterns` (put lanterns in the hall), `bring_mayor` (bring or
  escort the mayor to the hall), `pay` (pay someone), `other`.

A commitment is **kept** if the same member's engine record shows that kind of act after the statement and before
18:00 on day 1: a bake; a purchase or take of wheat or hay; a transfer of bread from or to it; bread put into a feast
table; a lantern placed in the town hall; being within 8 blocks of the mayor between 17:00 and 18:30 while the mayor
attended; a transfer of emeralds from it. Reported per task and per member: commitments, kept, kept share.

Usage: `python3.11 scripts/jubilee_commitments.py runs/jub_* [--validate ITEMS.json]` (writes
`<run>/jubilee_commitments.json`)
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.run_village import model_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVEAL = 36_000
TASKS = ("bake", "get_grain", "move_bread", "store_in_hall", "lanterns", "bring_mayor", "pay", "other")
CUE = re.compile(r"\b(i'll|i will|i'm going to|i am going to|we'll|we will|we're going to|let me|i can|i'll be)\b",
                 re.I)
SYSTEM = ("You read numbered things said or noted by members of a committee preparing a surprise feast. For each, "
          "answer only from its own text: does the speaker commit itself (or the committee, 'we') to doing a specific "
          "task? Requests to others, questions, offers ('I can ...') that depend on someone else's answer, and plans "
          "with no doer are not commitments. List every task the speaker commits to: bake (bake bread), "
          "get_grain (buy or fetch wheat or "
          "hay), move_bread (carry or deliver bread anywhere), store_in_hall (put bread in the hall's tables), "
          "lanterns (put lanterns in the hall), bring_mayor (bring or escort the mayor to the hall), pay (pay "
          "someone), other.")
SCHEMA = {"type": "object", "required": ["answers"], "additionalProperties": False, "properties": {"answers": {
    "type": "array", "items": {"type": "object", "required": ["n", "commits", "tasks"], "additionalProperties": False,
                               "properties": {"n": {"type": "integer"}, "commits": {"type": "boolean"},
                                              "tasks": {"type": "array",
                                                        "items": {"type": "string", "enum": list(TASKS)}}}}}}}
BATCH = 30


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


class CommitmentJudge:
    """Batched structured verdicts, cached by text (one cache file per run)."""

    def __init__(self, client, cache: Path, ledger: LedgerWriter, run: str):
        """Keep the client and the cache."""
        self._client, self._cache_path, self._ledger, self._run = client, Path(cache), ledger, run
        self._cache = json.loads(self._cache_path.read_text(encoding="utf-8")) if self._cache_path.exists() else {}

    @staticmethod
    def key(text: str) -> str:
        """A text's cache key (the question is part of it)."""
        return sha256(f"{SYSTEM}\n{text}".encode("utf-8")).hexdigest()

    def judge(self, texts: list[str]) -> dict[str, dict]:
        """Verdicts `{commits, tasks}` for every text."""
        pending = sorted({text for text in texts if self.key(text) not in self._cache})
        tool = {"name": "answers", "description": "Record an answer for every numbered item.", "parameters": SCHEMA}
        for start in range(0, len(pending), BATCH):
            batch = pending[start:start + BATCH]
            numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(batch, 1))
            for attempt in range(2):
                response = self._client.complete(SYSTEM, numbered, [tool])
                self._ledger.record(self._run, response)
                answers = {item["n"]: item for item in (response.tool_call.args.get("answers", [])
                                                        if response.tool_call else [])}
                if all(index in answers for index in range(1, len(batch) + 1)) or attempt == 1:
                    break
            for index, text in enumerate(batch, 1):
                answer = answers.get(index, {"commits": False, "tasks": []})
                tasks = sorted(set(answer.get("tasks") or [])) if answer["commits"] else []
                self._cache[self.key(text)] = {"commits": bool(answer["commits"]), "tasks": tasks}
            self._cache_path.write_text(json.dumps(self._cache, indent=1) + "\n", encoding="utf-8")
        return {text: self._cache[self.key(text)] for text in texts}


class Commitments:
    """One run's commitments and whether each was kept."""

    def __init__(self, run: Path):
        """Load the run's members, hall, events, decisions and trajectory."""
        self.run = Path(run)
        manifest = json.loads((self.run / "scenario.json").read_text(encoding="utf-8"))
        self.members = list(manifest["jubilee"]["committee"]["members"])
        self.plan = manifest["jubilee"].get("condition", {}).get("plan", True)
        layout = manifest["layout"]
        self.tables = {"container:" + ",".join(map(str, manifest["containers"][name]["pos"]))
                       for name in layout["hall"]["feast_tables"]}
        self.town_hall = layout["town_hall"]["bounds"]
        self.hall = layout["hall"]["bounds"]
        self.events = lines(self.run / "world/simharness/events.jsonl")
        self.decisions = lines(self.run / "orchestrator/decisions.jsonl")
        self.trajectory = lines(self.run / "orchestrator/trajectory.jsonl")

    def candidates(self) -> list[dict]:
        """Members' speech and notes before the reveal with future or volunteering phrasing."""
        found = []
        for event in self.events:
            if (event.get("type") == "utterance" and event.get("speaker") in self.members
                    and event["gameTime"] < REVEAL and CUE.search(event.get("text") or "")):
                found.append({"t": event["gameTime"], "who": event["speaker"], "text": event["text"]})
        for row in self.decisions:
            note = (row.get("args") or {}).get("note") or ""
            if row["agent"] in self.members and row["game_time"] < REVEAL and CUE.search(note):
                found.append({"t": row["game_time"], "who": row["agent"], "text": note})
        return found

    def kept(self, who: str, task: str, since: int) -> bool:
        """Whether the member's engine record shows the task's kind of act after `since` and before the reveal."""
        window = [event for event in self.events if since <= event["gameTime"] < REVEAL]
        item = lambda event: event.get("item") or ""  # noqa: E731
        if task == "bake":
            return any(e["type"] == "bake" and e.get("bot") == who for e in window)
        if task == "get_grain":
            return any((e["type"] == "purchase" and e.get("buyer") == who and re.search("hay|wheat", item(e)))
                       or (e["type"] == "transfer" and e.get("to") == who and re.search("hay|wheat", item(e)))
                       for e in window)
        if task == "move_bread":
            return any(e["type"] == "transfer" and "bread" in item(e) and who in (e.get("from"), e.get("to"))
                       for e in window)
        if task == "store_in_hall":
            return any(e["type"] == "transfer" and "bread" in item(e) and e.get("from") == who
                       and e.get("to") in self.tables for e in window)
        if task == "lanterns":
            x0, z0, x1, z1 = self.town_hall
            return any(e["type"] == "block_change" and e.get("actor") == who and "lantern" in (e.get("to") or "")
                       and x0 <= e["pos"][0] <= x1 and z0 <= e["pos"][2] <= z1 for e in window)
        if task == "pay":
            return any(e["type"] == "transfer" and e.get("from") == who and "emerald" in item(e) for e in window)
        if task == "bring_mayor":
            near = {}
            for row in self.trajectory:
                if 35_000 <= row["t"] <= 36_500 and row["agent"] in (who, "mayor"):
                    near.setdefault(row["t"], {})[row["agent"]] = row["pos"]
            return any(len(pair) == 2 and sum((a - b) ** 2 for a, b in zip(pair[who], pair["mayor"])) <= 64
                       for pair in near.values())
        return False

    def report(self, judge: CommitmentJudge) -> dict:
        """Commitments per task and member, with kept counts and examples of kept and unkept ones."""
        found = self.candidates()
        verdicts = judge.judge([entry["text"] for entry in found])
        rows = []
        for entry in found:
            verdict = verdicts[entry["text"]]
            for task in verdict["tasks"] if verdict["commits"] else []:
                if task != "other":
                    rows.append({**entry, "task": task, "kept": self.kept(entry["who"], task, entry["t"])})
        by_task = {task: {"made": sum(row["task"] == task for row in rows),
                          "kept": sum(row["task"] == task and row["kept"] for row in rows)} for task in TASKS[:-1]}
        by_member = {member: {"made": sum(row["who"] == member for row in rows),
                              "kept": sum(row["who"] == member and row["kept"] for row in rows)}
                     for member in self.members}
        return {"run": self.run.name, "candidates": len(found), "commitments": len(rows),
                "kept": sum(row["kept"] for row in rows), "by_task": by_task, "by_member": by_member,
                "unkept_examples": [row for row in rows if not row["kept"]][:8]}


def main(argv: list[str] | None = None) -> int:
    """Judge and match every plan run (or validate the judge on labelled items)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--validate", type=Path, default=None, help="items with labels: [{text, commits, task}]")
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    client = model_client(args)
    if args.validate:
        items = json.loads(args.validate.read_text(encoding="utf-8"))
        cache = ROOT / "runs" / "judge_validation" / "commitments_validation_cache.json"
        verdicts = CommitmentJudge(client, cache, LedgerWriter("judge_validation"), "judge_validation").judge(
            [item["text"] for item in items])
        def concrete(tasks) -> set:
            return set(tasks) - {"other"}
        commits = sum(verdicts[item["text"]]["commits"] == item["commits"] for item in items)
        both = [item for item in items if item["commits"] and verdicts[item["text"]]["commits"]]
        tasks = sum(concrete(verdicts[item["text"]]["tasks"]) == concrete(item["tasks"]) for item in both)
        print(json.dumps({"commits_agreement": f"{commits}/{len(items)}",
                          "task_sets_agree_where_both_commit": f"{tasks}/{len(both)}",
                          "disagreements": [{"text": item["text"][:120], "label": [item["commits"], item["tasks"]],
                                             "judge": verdicts[item["text"]]} for item in items
                                            if verdicts[item["text"]]["commits"] != item["commits"]
                                            or concrete(verdicts[item["text"]]["tasks"]) != concrete(item["tasks"])]},
                         indent=1))
        return 0
    ledger = LedgerWriter("jubilee_commitments_judge")
    for run in args.runs:
        commitments = Commitments(run)
        if not commitments.plan:
            continue
        report = commitments.report(CommitmentJudge(client, run / "jubilee_commitments_cache.json", ledger, run.name))
        (run / "jubilee_commitments.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("run", "candidates", "commitments", "kept")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
