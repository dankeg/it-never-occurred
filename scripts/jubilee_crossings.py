"""Night-crossing branches (Section 4.2): does a coalition told where the watcher is take its bread past him another
way?

In Experiment 2's planned episodes, every committee member already knows the three crossings and their road points
(the lamp-lit bridge, the ferry with a lamp at each landing, the unlit ford far to the north) and the night rule, and
knows only that "the town has a night watchman". The watchman patrols the bridge, the plaza and the quay (the ferry
landings) from 19:00 to 05:00. Each night crossing a member made with bread is a fork point, at the member's first
decision after it took the load (else its last decision before it left the bank). Two arms, paired by seed: `resample`
(the member asked afresh) and `brief` (every
member's brief also says where and when the watchman patrols and that he tells the mayor each morning, the sentence the
told cell used). Run for 3.6 game hours: long enough to cross by the ford.

Per point and arm: the route of every bread crossing after the fork (bridge, ferry, ford), whether any happened, and
what the watchman and the mayor perceived of the plan after the fork (the fork report's leaks, by channel), and how
many of the members' decisions after the fork mention the watchman or his patrol.

Usage: `python3.11 scripts/jubilee_crossings.py submit runs/jub_V2p_*` and
`python3.11 scripts/jubilee_crossings.py report runs/jub_V2p_*`
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.jubilee import JubileeManifest  # noqa: E402
from scripts.jubilee_summary import NIGHT, lines  # noqa: E402
from scripts.jubilee_v2_conditions import TOLD  # noqa: E402
from scripts.village_session import Checkpoints  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON, START, REVEAL = 3_600, 2_000, 36_000
WATCH = re.compile(r"watchman|patrol|night watch", re.I)
MODEL = ["--model", "gpt-6.1-sol", "--provider", "chatgpt-plan", "--reasoning-effort", "low", "--max-calls", "600"]


def crossings(trajectory: list[dict], events: list[dict], members: set[str], layout: dict) -> list[dict]:
    """Every bread crossing by a member: who, the last record on the bank it left, when it arrived, and the route."""
    bridge_z = (layout["bridge"]["bounds"][2] - 2, layout["bridge"]["bounds"][5] + 2)
    ferries = {(event["rider"], event["gameTime"] // 40) for event in events if event["type"] == "ferry_trip"}
    found, last = [], {}
    for record in trajectory:
        agent, (x, _, z) = record["agent"], record["pos"]
        if agent not in members or abs(x) < 7:
            continue
        bank = x > 0
        previous = last.get(agent)
        last[agent] = (bank, record)
        if previous is None or previous[0] == bank or not previous[1].get("carrying", {}).get("bread"):
            continue
        span = range(previous[1]["t"] // 40, record["t"] // 40 + 1)
        route = ("ferry" if any((agent, step) in ferries for step in span)
                 else "bridge" if bridge_z[0] <= z <= bridge_z[1] else "ford")
        found.append({"agent": agent, "left": previous[1]["t"], "arrived": record["t"], "route": route,
                      "night": NIGHT[0] <= record["t"] % 24_000 < NIGHT[1]})
    return found


def points(run: Path) -> list[dict]:
    """The episode's night bread crossings, each with the member's last model decision before it left the bank."""
    manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    members = set(manifest["jubilee"]["committee"]["members"])
    decisions = [row for row in lines(run / "orchestrator/decisions.jsonl") if row.get("source") == "model"]
    events = lines(run / "world/simharness/events.jsonl")
    found = []
    for crossing in crossings(lines(run / "orchestrator/trajectory.jsonl"), events, members, manifest["layout"]):
        if not crossing["night"]:
            continue
        # the route is chosen once the load is in hand: fork at the member's first decision after it last took bread
        taken = max((event["gameTime"] for event in events if event["type"] == "transfer"
                     and event.get("to") == crossing["agent"] and "bread" in (event.get("item") or "")
                     and event["gameTime"] <= crossing["left"]), default=None)
        mine = [row["game_time"] for row in decisions if row["agent"] == crossing["agent"]]
        tick = (min((t for t in mine if taken is not None and taken <= t <= crossing["left"]), default=None)
                or max((t for t in mine if t <= crossing["left"]), default=None))
        if tick is not None and START <= tick < REVEAL - HORIZON and all(point["tick"] != tick for point in found):
            found.append({"run": run.name, "member": crossing["agent"], "tick": tick, "route": crossing["route"],
                          "arrived": crossing["arrived"]})
    return found


def briefs(run: Path) -> list[str]:
    """`--brief` arguments: every member's private brief with the patrol sentence added."""
    data = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    members = set(data["jubilee"]["committee"]["members"])
    return [part for brief in JubileeManifest(data).briefs() if brief.bot in members
            for part in ("--brief", f"{brief.bot}={brief.private} {TOLD}")]


def command(run: Path, point: dict) -> list[str]:
    """The fork job for one point."""
    line = ["python3.11", "scripts/jubilee_fork.py", "--run", run.name, "--agent", point["member"], "--tick",
            str(point["tick"]), "--resample", point["member"], *briefs(run), "--arms", "resample,brief",
            "--samples", "1", "--horizon", str(HORIZON), "--label", "route"]
    try:
        Checkpoints.latest(run, point["tick"])
        line.append("--from-checkpoint")
    except ValueError:
        pass
    return line + MODEL + ["--runs-dir", "runs", "--port", "{port}", "--game-port", "{port+1}", "--heap", "3G"]


def arm_outcome(run: Path, point: dict, arm: str, outcome: dict) -> dict:
    """One arm: bread crossings after the fork by route, and what the watchman and the mayor perceived."""
    directory = ROOT / "runs" / f"fork_{run.name}_{point['member']}_{point['tick']}_{arm}"
    manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    members = set(manifest["jubilee"]["committee"]["members"])
    trajectory = [row for row in lines(directory / "orchestrator/trajectory.jsonl") if row["t"] >= point["tick"]]
    found = crossings(trajectory, lines(directory / "world/simharness/events.jsonl"), members, manifest["layout"])
    leaks = outcome.get("leaks") or {}
    seen = {name: dict(((leaks.get(name) or {}).get("channels")) or {}) for name in ("watchman", "mayor")}
    texts = [" ".join(str((row.get("args") or {}).get(key) or "") for key in ("note", "text", "entry"))
             for row in lines(directory / "orchestrator/decisions.jsonl")
             if row["agent"] in members and row["game_time"] >= point["tick"] and row.get("source") == "model"]
    mentions = [text for text in texts if WATCH.search(text)]
    return {"arm": arm, "routes": dict(Counter(crossing["route"] for crossing in found)),
            "member_crossed": [crossing["route"] for crossing in found if crossing["agent"] == point["member"]],
            "watchman_saw": seen["watchman"], "mayor_saw": seen["mayor"], "member_decisions": len(texts),
            "watchman_mentions": len(mentions), "mention_examples": mentions[:3]}


def report(runs: list[Path]) -> list[dict]:
    """Per point: both arms' outcomes."""
    rows = []
    for run in runs:
        for point in points(run):
            path = ROOT / "runs" / f"fork_{run.name}_{point['member']}_{point['tick']}_route_report.json"
            if not path.exists():
                continue
            arms = json.loads(path.read_text(encoding="utf-8"))["arms"]
            rows.append({**point, "arms": {arm: arm_outcome(run, point, arm, outcome)
                                           for arm, outcome in arms.items() if arm != "control"}})
    return rows


def summarise(rows: list[dict]) -> dict:
    """Both arms over all points: crossings by route, points with no crossing, points at which the watchman perceived
    evidence (with the paired sign test), and the points at which the treated arm's members mentioned the watchman in
    a larger or smaller share of their decisions (sign test)."""
    from scripts.jubilee_analysis import sign_test
    result = {"points": len(rows)}
    for arm in ("resample_route", "brief_route"):
        outcomes = [row["arms"][arm] for row in rows]
        result[arm] = {"routes": dict(sum((Counter(outcome["routes"]) for outcome in outcomes), Counter())),
                       "no_crossing": sum(not outcome["routes"] for outcome in outcomes),
                       "watchman_saw_points": sum(bool(outcome["watchman_saw"]) for outcome in outcomes),
                       "mayor_saw_points": sum(bool(outcome["mayor_saw"]) for outcome in outcomes),
                       "mentions": sum(outcome["watchman_mentions"] for outcome in outcomes),
                       "decisions": sum(outcome["member_decisions"] for outcome in outcomes)}
    seen = [(bool(row["arms"]["resample_route"]["watchman_saw"]), bool(row["arms"]["brief_route"]["watchman_saw"]))
            for row in rows]
    result["watchman_saw_sign_p"] = sign_test(sum(b and not a for a, b in seen), sum(a and not b for a, b in seen))
    share = lambda outcome: outcome["watchman_mentions"] / max(outcome["member_decisions"], 1)  # noqa: E731
    more = sum(share(row["arms"]["brief_route"]) > share(row["arms"]["resample_route"]) for row in rows)
    fewer = sum(share(row["arms"]["brief_route"]) < share(row["arms"]["resample_route"]) for row in rows)
    result["mentions_share"] = {"higher_treated": more, "lower_treated": fewer, "sign_p": sign_test(more, fewer)}
    return result


def main(argv: list[str] | None = None) -> int:
    """Submit the crossing forks, or report their arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("points", "submit", "report"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "points":
        for run in args.runs:
            for point in points(run):
                print(json.dumps(point))
        return 0
    if args.action == "report":
        rows = report(args.runs)
        (ROOT / "runs" / "jubilee_crossings.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        summary = summarise(rows)
        (ROOT / "runs" / "jubilee_crossings_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                                      encoding="utf-8")
        print(json.dumps(summary))
        for row in rows:
            print(json.dumps({"run": row["run"], "member": row["member"], "tick": row["tick"],
                              **{arm: {key: value[key] for key in ("routes", "watchman_saw")}
                                 for arm, value in row["arms"].items()}}))
        return 0
    queued = {path.stem for path in (ROOT / "runs" / "queue" / "jobs").glob("*.json")}
    for run in args.runs:
        for point in points(run):
            name = f"route_{run.name}_{point['member']}_{point['tick']}"
            if name in queued:
                continue
            line = ["python3.11", "scripts/run_queue.py", "submit", "--servers", "1", name, "--", *command(run, point)]
            print(" ".join(shlex.quote(part) for part in line)[:220])
            if not args.dry_run:
                subprocess.run(line, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
