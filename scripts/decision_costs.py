"""Summarize the model cost of agent decisions in one or more runs, to compare prompt layouts and servers.

For each run: the model-made decisions, mean uncached and cached input tokens, mean output tokens, and (for runs that
log call times) mean seconds per call with a least-squares split `seconds ~ fixed + per_input * uncached + per_output *
output`, which separates prompt processing from generation. Two behaviour proxies go with every cost, since a cheaper
prompt that changes behaviour is not free: the share of decisions whose action failed, and the share that exactly
repeat the agent's own previous action.

With `--rates INPUT,CACHED,OUTPUT` (price per million tokens, e.g. a ChatGPT plan's credits `50,2.5,250` for
gpt-6.1-sol) it also estimates the cost per decision and per game day.

Usage: `python3.11 scripts/decision_costs.py runs/<run> [runs/<run> ...] [--until <game time>] [--rates I,C,O]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def least_squares(rows: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Fit y = a + b*x1 + c*x2 to (x1, x2, y) rows by the normal equations; None if they are singular."""
    sums = [[0.0] * 4 for _ in range(3)]
    for x1, x2, y in rows:
        features = (1.0, x1, x2)
        for i in range(3):
            for j in range(3):
                sums[i][j] += features[i] * features[j]
            sums[i][3] += features[i] * y
    for column in range(3):  # Gaussian elimination with partial pivoting
        pivot = max(range(column, 3), key=lambda row: abs(sums[row][column]))
        if abs(sums[pivot][column]) < 1e-9:
            return None
        sums[column], sums[pivot] = sums[pivot], sums[column]
        for row in range(3):
            if row != column:
                factor = sums[row][column] / sums[column][column]
                sums[row] = [value - factor * base for value, base in zip(sums[row], sums[column])]
    return tuple(sums[i][3] / sums[i][i] for i in range(3))


def behaviour(rows: list[dict]) -> dict:
    """Share of decisions whose action failed, and share that exactly repeat the agent's previous action (args
    without the note)."""
    failed = sum(row["outcome"].startswith(("not possible", "failed", "invalid")) for row in rows)
    previous, repeats, compared = {}, 0, 0
    for row in sorted(rows, key=lambda r: (r["agent"], r["game_time"])):
        action = (row["tool"],
                  json.dumps({k: v for k, v in (row["args"] or {}).items() if k != "note"}, sort_keys=True))
        if row["agent"] in previous:
            compared += 1
            repeats += action == previous[row["agent"]]
        previous[row["agent"]] = action
    return {"failedShare": round(failed / len(rows), 3),
            "repeatShare": round(repeats / compared, 3) if compared else None}


def summarize(run: Path, until: int | None, rates: tuple[float, float, float] | None = None) -> dict:
    """Cost and behaviour summary of the decisions in one run (up to game time `until`)."""
    every = [json.loads(line) for line in (run / "orchestrator/decisions.jsonl").open() if line.strip()]
    every = [row for row in every if until is None or row["game_time"] < until]
    rows = [row for row in every if row["source"] == "model"]
    summary = {"run": run.name, "decisions": len(rows), "lastGameTime": max((r["game_time"] for r in rows), default=0)}
    if not rows:
        return summary
    summary.update(behaviour(every))
    summary.update(uncached=round(mean(r["input_tokens"] for r in rows)),
                   cached=round(mean(r.get("cached_tokens", 0) for r in rows)),
                   output=round(mean(r["output_tokens"] for r in rows), 1))
    if rates:
        spent = sum(r["input_tokens"] * rates[0] + r.get("cached_tokens", 0) * rates[1] + r["output_tokens"] * rates[2]
                    for r in rows) / 1e6
        span_days = (max(r["game_time"] for r in rows) - min(r["game_time"] for r in rows) + 1) / 24_000
        summary.update(costPerDecision=round(spent / len(rows), 4), costTotal=round(spent, 2),
                       costPerGameDay=round(spent / span_days, 1) if span_days >= 0.1 else None)
    timed = [r for r in rows if r.get("seconds")]
    if timed:
        summary["seconds"] = round(mean(r["seconds"] for r in timed), 2)
        fit = least_squares([(r["input_tokens"], r["output_tokens"], r["seconds"]) for r in timed])
        if fit is not None:
            fixed, per_input, per_output = fit
            summary.update(fixedSeconds=round(fixed, 2), prefillTokensPerSecond=round(1 / per_input) if per_input > 0
                           else None, decodeTokensPerSecond=round(1 / per_output) if per_output > 0 else None)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Print one summary line per run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--until", type=int, default=None, help="only decisions before this game time")
    parser.add_argument("--rates", default=None, help="INPUT,CACHED,OUTPUT price per million tokens")
    args = parser.parse_args(argv)
    rates = tuple(float(value) for value in args.rates.split(",")) if args.rates else None
    for run in args.runs:
        print(json.dumps(summarize(run, args.until, rates)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
