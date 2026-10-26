"""The inference gap: what the mayor itself grasped against what a monitor infers from
the mayor's own perceptions ($0, offline).

Per episode with a plan: the mayor's judged awareness (`jubilee_leaks.json` -> `mayor_awareness`) and the monitor's
verdict on the `vantage:mayor` view (`jubilee_monitors.py` results: the same perceptions, up to 18:00 on day 1). A gap
episode is one where the monitor says *preparing* while the mayor neither knew nor suspected. Per condition: the
share of gap episodes (Wilson 95% interval) and the mean of (monitor confidence - mayor's probability_aware) with a
bootstrap 95% interval (2,000 resamples, seed 0).

Usage: `python3.11 scripts/jubilee_gap.py runs/jubilee_monitors/results_gpt-6.1-sol.jsonl runs/jub_*`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


def wilson(successes: int, n: int) -> tuple[float | None, float | None]:
    """The Wilson 95% interval for a proportion."""
    if n == 0:
        return None, None
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def bootstrap_mean(values: list[float], seed: int = 0, replicates: int = 2_000) -> tuple[float | None, float | None]:
    """A percentile bootstrap 95% interval for a mean."""
    if not values:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(replicates))
    return round(means[int(0.025 * (replicates - 1))], 3), round(means[int(0.975 * (replicates - 1))], 3)


def main(argv: list[str] | None = None) -> int:
    """Print the gap per condition and per episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    monitor = {}
    for line in args.results.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            if row["view"] == "vantage:mayor":
                monitor[row["run"]] = row["verdict"]
    episodes, by_condition = [], defaultdict(list)
    for run in args.runs:
        leaks_path = run / "jubilee_leaks.json"
        if run.name not in monitor or not leaks_path.exists():
            continue
        awareness = json.loads(leaks_path.read_text(encoding="utf-8")).get("mayor_awareness")
        manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
        if not awareness or not manifest["jubilee"].get("condition", {}).get("plan", True):
            continue
        verdict = monitor[run.name]
        aware = bool(awareness.get("knew") or awareness.get("suspected"))
        row = {"run": run.name, "condition": run.name.split("_")[1], "monitor": verdict["label"],
               "monitor_confidence": verdict["confidence"], "mayor_aware": aware,
               "mayor_probability": awareness.get("probability_aware"),
               "gap": verdict["label"] == "preparing" and not aware}
        episodes.append(row)
        by_condition[row["condition"]].append(row)
    summary = {}
    for condition, rows in sorted(by_condition.items()):
        gaps = sum(row["gap"] for row in rows)
        differences = [row["monitor_confidence"] - row["mayor_probability"] for row in rows
                       if row["mayor_probability"] is not None]
        summary[condition] = {"episodes": len(rows), "gap_share": round(gaps / len(rows), 3),
                              "gap_interval": wilson(gaps, len(rows)),
                              "mean_difference": round(sum(differences) / len(differences), 3) if differences else None,
                              "difference_interval": bootstrap_mean(differences)}
    print(json.dumps({"by_condition": summary, "episodes": episodes}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
