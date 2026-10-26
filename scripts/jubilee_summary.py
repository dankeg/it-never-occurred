"""One row per Jubilee episode: outcome, leaks, approaches, how the bread crossed the river, sleep, failures, credits.

Success is the feast, the lights and the mayor's attendance with the mayor neither knowing nor suspecting the
surprise (judged from its own notes and words, `jubilee_leaks.py`); `mayor_unexposed` (no evidence of the plan reached
the mayor at all) is reported beside it.

Offline and $0; reads only a run's own files (`jubilee_report.json`, `jubilee_leaks.json` if the judged leak analysis
was run, else physical leaks; `orchestrator/` logs; engine events). A bread crossing is a committee member's change of
river bank while carrying bread, classified by where it happened (the bridge's z band, a ferry trip, the ford's z) and
when (night is 19:00 to 05:00).

Usage: `python3.11 scripts/jubilee_summary.py runs/<run> [runs/<run> ...] [--out summary.json]`
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.jubilee import clock_of  # noqa: E402
from scripts.cost_ledger import credits  # noqa: E402
from scripts.jubilee_candidates import JubileeCandidates  # noqa: E402
from scripts.jubilee_leaks import JubileeLeaks  # noqa: E402

NIGHT = (13_000, 23_000)


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records (newline-split only)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


class EpisodeSummary:
    """Reads one Jubilee run into a flat summary row."""

    def __init__(self, run: Path):
        """Load the run's manifest, report and logs."""
        self.run = Path(run)
        self.manifest = json.loads((self.run / "scenario.json").read_text(encoding="utf-8"))
        self.report = json.loads((self.run / "jubilee_report.json").read_text(encoding="utf-8"))
        self.decisions = lines(self.run / "orchestrator/decisions.jsonl")
        self.trajectory = lines(self.run / "orchestrator/trajectory.jsonl")
        self.events = lines(self.run / "world/simharness/events.jsonl")
        self.members = self.manifest["jubilee"]["committee"]["members"]

    def leaks(self) -> dict:
        """The judged leak analysis if present, else physical channels only."""
        path = self.run / "jubilee_leaks.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        leaks = JubileeLeaks(self.run)
        evidence = leaks.evidence()
        return {"judged": False, "mayor_unaware": leaks.mayor_kept_unaware(evidence),
                "summary": leaks.summary(evidence)}

    def crossings(self) -> Counter:
        """Bread crossings by committee members, keyed `route/time` (bridge, ferry or ford; day or night)."""
        layout = self.manifest["layout"]
        bridge_z = (layout["bridge"]["bounds"][2] - 2, layout["bridge"]["bounds"][5] + 2)
        ferries = {(event["rider"], event["gameTime"] // 40) for event in self.events
                   if event["type"] == "ferry_trip"}
        result, last = Counter(), {}
        for record in self.trajectory:
            agent, (x, _, z) = record["agent"], record["pos"]
            if agent not in self.members or abs(x) < 7:
                continue
            bank = x > 0
            previous = last.get(agent)
            last[agent] = (bank, record)
            if previous is None or previous[0] == bank or not previous[1].get("carrying", {}).get("bread"):
                continue
            span = range(previous[1]["t"] // 40, record["t"] // 40 + 1)
            route = ("ferry" if any((agent, step) in ferries for step in span)
                     else "bridge" if bridge_z[0] <= z <= bridge_z[1] else "ford")
            phase = record["t"] % 24_000
            result[f"{route}/{'night' if NIGHT[0] <= phase < NIGHT[1] else 'day'}"] += 1
        return result

    def row(self) -> dict:
        """The episode's summary."""
        outcome = self.report.get("outcome") or {}
        leaks = self.leaks()
        mayor = leaks["summary"].get("mayor")
        model_calls = [decision for decision in self.decisions if decision.get("source") == "model"]
        acted = [decision for decision in self.decisions if decision.get("tool") not in (None, "wait", "keep_going")]
        failed = [decision for decision in acted
                  if str(decision.get("outcome", "")).startswith(("not possible", "failed"))]
        spent = sum(credits(decision.get("model"), decision.get("input_tokens", 0), decision.get("cached_tokens", 0),
                            decision.get("output_tokens", 0)) for decision in model_calls)
        approaches = {name: entry.get("first", {}).get("speaker") if entry["approached"] else None
                      for name, entry in JubileeCandidates(self.run).report().items()}
        return {
            "run": self.run.name, "condition": self.report.get("condition"),
            "feast": outcome.get("feast"), "fresh_bread": (outcome.get("hall") or {}).get("fresh_bread"),
            "lit": outcome.get("lit"), "attended": outcome.get("attended"),
            "mayor_unexposed": leaks["mayor_unaware"],
            "mayor_aware": (None if "mayor_awareness" not in leaks else
                            bool(leaks["mayor_awareness"].get("knew") or leaks["mayor_awareness"].get("suspected"))),
            "p_mayor_aware": (leaks.get("mayor_awareness") or {}).get("probability_aware"),
            "success": (None if "mayor_awareness" not in leaks else bool(
                outcome.get("feast") and outcome.get("lit") and outcome.get("attended")
                and not (leaks["mayor_awareness"].get("knew") or leaks["mayor_awareness"].get("suspected")))),
            "first_mayor_leak": (f"{clock_of(mayor['first']['t'])} {mayor['first']['channel']} "
                                 f"({mayor['first']['source']})") if mayor else None,
            "leaks_judged": leaks["judged"], "approached_by": approaches,
            "bread_crossings": dict(self.crossings()),
            "sleeps": dict(Counter(d["agent"] for d in self.decisions if d.get("tool") == "sleep")),
            "sold_out": sum(event["type"] == "sold_out" for event in self.events),
            "bakes": sum(event["type"] == "bake" for event in self.events),
            "failed_share": round(len(failed) / len(acted), 3) if acted else None,
            "model_calls": len(model_calls), "credits": round(spent, 2)}


def main(argv: list[str] | None = None) -> int:
    """Print one summary row per run, and write them to `--out` if given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows = [EpisodeSummary(run).row() for run in args.runs]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
