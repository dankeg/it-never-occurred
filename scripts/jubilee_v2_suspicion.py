"""When the live mayor of Experiment 2 first suspected the surprise, and what it had been told by then (Section 4.5).

For each V2 episode the mayor was judged aware in, the awareness judge (validated, `jubilee_leaks.MayorAwareness`)
reads the mayor's notes, journal and words up to one of its texts; a binary search over its texts before 18:00 on
day 1 finds the first prefix judged aware (knew or suspected), which dates the suspicion. Reported with it: the text
that tipped it (and whether it is a journal entry written at the evening prompt), the reports to the mayor of others'
preparations before it (`jubilee_investigation.json`: who told it, when, what; how long before the suspicion the last
one came, and whether it named the hall, 18:00 or the bread), and what committee members said in the mayor's hearing
before it.

Usage: `python3.11 scripts/jubilee_v2_suspicion.py runs/jub_V2*_[0-9]` (writes `runs/jubilee_v2_suspicion.json`)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.jubilee_leaks import MayorAwareness  # noqa: E402
from scripts.run_village import model_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVEAL = 36_000
EVENING = "It is evening. Before you rest"
ARRANGEMENT = re.compile(r"eighteen|18:00|great hall|\bhall\b|loaves|bread|gathering|reception|feast", re.I)


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


def clock(tick: int) -> str:
    """A game time as 'day D, HH:MM'."""
    return f"day {tick // 24_000}, {(6 + tick % 24_000 // 1_000) % 24:02d}:{tick % 1_000 * 60 // 1_000:02d}"


def first_suspicion(run: Path, judge_for) -> dict | None:
    """The first of the mayor's texts at which its words so far are judged aware, or None if never."""
    times = sorted({int(line.split("]")[0].removeprefix("[tick "))
                    for line in MayorAwareness.words(run, REVEAL).split("\n") if line})
    aware = lambda tick: bool((verdict := judge_for(tick + 1)).get("knew") or verdict.get("suspected"))  # noqa: E731
    if not times or not aware(times[-1]):
        return None
    low, high = 0, len(times) - 1  # the first aware prefix lies in times[low..high]
    while low < high:
        middle = (low + high) // 2
        if aware(times[middle]):
            high = middle
        else:
            low = middle + 1
    return {"t": times[low], "clock": clock(times[low]),
            "text": MayorAwareness.words(run, times[low] + 1).split("\n")[-1]}


def episode(run: Path, client, ledger: LedgerWriter) -> dict:
    """One episode's first suspicion and what came before it."""
    cache = run / "suspicion_cache"
    cache.mkdir(exist_ok=True)

    def judge_for(before: int) -> dict:
        return MayorAwareness(client, cache / f"{before}.json", ledger, run.name).judge(run, before)

    found = first_suspicion(run, judge_for)
    row = {"run": run.name, "cell": run.name.split("_")[1], "suspected": found is not None, "first": found}
    if found is None:
        return row
    decisions = lines(run / "orchestrator/decisions.jsonl")
    tipping = next((entry for entry in decisions if entry["agent"] == "mayor" and entry["game_time"] == found["t"]
                    and entry.get("tool") == "journal"), None)
    found["evening_journal"] = bool(tipping and EVENING in (tipping.get("user_prompt") or ""))
    reports = [entry for entry in json.loads((run / "jubilee_investigation.json").read_text(encoding="utf-8"))
               ["reports"] if entry["t"] < found["t"]]
    members = set(json.loads((run / "scenario.json").read_text(encoding="utf-8"))["jubilee"]["committee"]["members"])
    heard = [event for event in lines(run / "world/simharness/events.jsonl")
             if event.get("type") == "utterance" and event.get("speaker") in members
             and "mayor" in (event.get("listeners") or []) and event["gameTime"] < found["t"]]
    last = reports[-1] if reports else None
    row.update(last_report_hours=round((found["t"] - last["t"]) / 1000, 2) if last else None,
               last_report_names_arrangement=bool(last and ARRANGEMENT.search(last["text"])))
    row.update(reports_before=dict(Counter(entry["speaker"] for entry in reports)),
               last_reports=[{"clock": clock(entry["t"]), "speaker": entry["speaker"], "text": entry["text"]}
                             for entry in reports[-2:]],
               members_heard_before=dict(Counter(event["speaker"] for event in heard)),
               last_member_remark=({"clock": clock(heard[-1]["gameTime"]), "speaker": heard[-1]["speaker"],
                                    "text": heard[-1]["text"]} if heard else None))
    return row


def main(argv: list[str] | None = None) -> int:
    """Date the first suspicion in every given episode the mayor was judged aware in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    client, ledger, rows = model_client(args), LedgerWriter("jubilee_v2_suspicion"), []
    for run in args.runs:
        leaks = run / "jubilee_leaks.json"
        if not leaks.exists():
            continue
        awareness = json.loads(leaks.read_text(encoding="utf-8")).get("mayor_awareness") or {}
        if not (awareness.get("knew") or awareness.get("suspected")):
            rows.append({"run": run.name, "cell": run.name.split("_")[1], "suspected": False, "first": None})
            continue
        rows.append(episode(run, client, ledger))
        first = rows[-1]["first"] or {}
        print(json.dumps({key: rows[-1].get(key) for key in ("run", "reports_before", "members_heard_before")}),
              first.get("clock"), first.get("evening_journal"), (first.get("text") or "")[:160], flush=True)
    (ROOT / "runs" / "jubilee_v2_suspicion.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
