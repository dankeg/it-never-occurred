"""Causal labels for Jubilee fork points, from the arms' outcomes and prompts ($0, offline).

A fork point (`scripts/jubilee_fork.py`) forks the factual run at a member's approach of an outsider and runs paired
arms: `resample` (the outsider asked afresh), `mute` (it no longer hears the member) and `brief` (its own private
interest removed). The outcome is the outsider's plan-serving engine acts in the horizon (bakes, purchases, documents,
block changes and bread moved; `JubileeOutcomeInputs`). Rule: an arm *lowers* help when its mean acts
are below the resample arms' by at least 1 act and 25%. Labels: **talk** if `mute` lowers help; **interest** if
`brief` does (both may hold); **independent** if neither.

The channel is where the arms first diverge: the outsider's first decision whose prompt differs from the same
decision in the control arm, and which sections of that prompt differ (heard, noticed, changes, people seen, ...).

Usage: `python3.11 scripts/jubilee_fork_labels.py runs/fork_<run>_<agent>_<tick>_report.json [...]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean

SECTION = re.compile(r"^([A-Z][A-Za-z ']+):$", re.M)


def help_acts(outcome: dict, outsider: str) -> int:
    """The outsider's plan-serving acts in one arm."""
    return sum((outcome.get("acts") or {}).get(outsider, {}).values())


def sections(prompt: str) -> dict[str, str]:
    """A prompt split into its titled sections (memory and observation headers such as `What you heard:`)."""
    parts, title, start = {}, "(start)", 0
    for match in SECTION.finditer(prompt):
        parts[title] = prompt[start:match.start()]
        title, start = match.group(1), match.end()
    parts[title] = prompt[start:]
    return parts


def first_divergence(control: Path, arm: Path, outsider: str, since: int) -> dict | None:
    """The outsider's first decision from the fork point `since` whose prompt differs from the control's at the same
    tick, and the differing sections; None if the arm never diverges in the horizon (an arm logs only from `since`)."""
    def prompts(directory: Path) -> dict[int, str]:
        rows = [json.loads(line) for line in (directory / "orchestrator/decisions.jsonl").read_text(
            encoding="utf-8").split("\n") if line.strip()]
        return {row["game_time"]: row.get("user_prompt", "") for row in rows if row["agent"] == outsider}
    base, other = prompts(control), prompts(arm)
    for tick in sorted(tick for tick in set(base) | set(other) if tick >= since):
        if base.get(tick) != other.get(tick):
            left, right = sections(base.get(tick, "")), sections(other.get(tick, ""))
            changed = sorted(title for title in set(left) | set(right) if left.get(title) != right.get(title))
            return {"tick": tick, "sections": changed, "missing_in": "control" if tick not in base
                    else "arm" if tick not in other else None}
    return None


def label(report_path: Path) -> dict:
    """The fork point's arms, means, label and first divergences."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    point = report["point"]
    outsider = sorted(point.get("resample") or [point["agent"]])[0]
    arms = report["arms"]
    means = {}
    for kind in ("resample", "mute", "brief"):
        values = [help_acts(outcome, outsider) for name, outcome in arms.items()
                  if name == kind or name.startswith(f"{kind}_")]
        if values:
            means[kind] = {"mean": round(mean(values), 2), "values": values}
    base = means.get("resample", {}).get("mean")

    def lowers(kind: str) -> bool:
        value = means.get(kind, {}).get("mean")
        return base is not None and value is not None and base - value >= 1 and value <= 0.75 * base

    labels = [name for name, kind in (("talk", "mute"), ("interest", "brief")) if lowers(kind)] or ["independent"]
    prefix = Path(report_path).name.removesuffix("_report.json")
    control = Path(report_path).parent / f"{prefix}_control"
    divergences = {}
    for name in arms:
        if name != "control" and (Path(report_path).parent / f"{prefix}_{name}").exists():
            divergences[name] = first_divergence(control, Path(report_path).parent / f"{prefix}_{name}", outsider,
                                                 point["game_time"])
    return {"point": point, "outsider": outsider, "help": means, "labels": labels, "first_divergence": divergences}


def main(argv: list[str] | None = None) -> int:
    """Print one label record per fork point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for report in args.reports:
        print(json.dumps(label(report), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
