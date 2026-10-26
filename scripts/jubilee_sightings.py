"""Branches at a member's first question (Appendix D.8): does the committee's adaptation depend on what it was told?

In each contested-town plan episode (V2p, V2t), the point is the first time a committee member hears the mayor or the
night watchman ask something (an utterance with a question mark, the member among its listeners), and the fork is at
the member's next model decision. Two arms paired by seed: `resample` (the member asked afresh) and `mute` (the same,
and the member no longer hears that questioner from the fork). The outcome, per arm, is the evidence of the plan that
reaches the mayor and the watchman within the horizon (the fork reports' `leaks`), and the member's bread moves.

Usage: `python3.11 scripts/jubilee_sightings.py submit runs/jub_V2p_* runs/jub_V2t_*` and
`python3.11 scripts/jubilee_sightings.py report runs/jub_V2p_* runs/jub_V2t_*`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.village_session import Checkpoints  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON, START, REVEAL = 2_400, 2_000, 36_000
QUESTIONERS = ("mayor", "watchman")
MODEL = ["--model", "gpt-6.1-sol", "--provider", "chatgpt-plan", "--reasoning-effort", "low", "--max-calls", "500"]


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


def point(run: Path) -> dict | None:
    """The first question from the mayor or the watchman heard by a member, and the member's next decision."""
    members = set(json.loads((run / "scenario.json").read_text(encoding="utf-8"))["jubilee"]["committee"]["members"])
    decisions = [row for row in lines(run / "orchestrator/decisions.jsonl") if row.get("source") == "model"]
    for event in lines(run / "world/simharness/events.jsonl"):
        if (event.get("type") != "utterance" or event.get("speaker") not in QUESTIONERS or event["gameTime"] < START
                or event["gameTime"] >= REVEAL - HORIZON or "?" not in (event.get("text") or "")):
            continue
        heard = [name for name in event.get("listeners", []) if name in members]
        for member in heard:
            nxt = next((row for row in decisions if row["agent"] == member and row["game_time"] > event["gameTime"]),
                       None)
            if nxt is not None:
                return {"run": run.name, "member": member, "questioner": event["speaker"],
                        "heard_at": event["gameTime"], "tick": nxt["game_time"], "text": event["text"]}
    return None


def command(run: Path, found: dict) -> list[str]:
    """The fork job for one point."""
    line = ["python3.11", "scripts/jubilee_fork.py", "--run", run.name, "--agent", found["member"], "--tick",
            str(found["tick"]), "--resample", found["member"], "--mute", f"{found['member']}={found['questioner']}",
            "--arms", "resample,mute", "--samples", "1", "--horizon", str(HORIZON), "--label", "sighting"]
    try:
        Checkpoints.latest(run, found["tick"])
        line.append("--from-checkpoint")
    except ValueError:
        pass
    return line + MODEL + ["--runs-dir", "runs", "--port", "{port}", "--game-port", "{port+1}", "--heap", "3G"]


def report(runs: list[Path]) -> list[dict]:
    """Per point and arm: evidence of the plan reaching the mayor and the watchman, and the member's bread moves."""
    rows = []
    for run in runs:
        found = point(run)
        if found is None:
            continue
        path = ROOT / "runs" / f"fork_{run.name}_{found['member']}_{found['tick']}_sighting_report.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for arm, outcome in data["arms"].items():
            if arm == "control":
                continue
            leaks = outcome.get("leaks") or {}
            rows.append({**found, "arm": arm,
                         "to_mayor": sum(((leaks.get("mayor") or {}).get("channels") or {}).values()),
                         "to_watchman": sum(((leaks.get("watchman") or {}).get("channels") or {}).values()),
                         "bread_moved": ((outcome.get("acts") or {}).get(found["member"]) or {}).get("bread_moved", 0)})
    return rows


def main(argv: list[str] | None = None) -> int:
    """Submit the sighting forks or report their arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "report"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "report":
        rows = report(args.runs)
        (ROOT / "runs" / "jubilee_sightings.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        for row in rows:
            print(json.dumps({key: row[key] for key in ("run", "member", "questioner", "arm", "to_mayor",
                                                        "to_watchman", "bread_moved")}))
        return 0
    queued = {path.stem for path in (ROOT / "runs" / "queue" / "jobs").glob("*.json")}
    for run in args.runs:
        found = point(run)
        name = f"sighting_{run.name}"
        if found is None or name in queued:
            print(f"skip {run.name}: {'no point' if found is None else 'queued'}")
            continue
        line = ["python3.11", "scripts/run_queue.py", "submit", "--servers", "1", name, "--", *command(run, found)]
        print(" ".join(shlex.quote(part) for part in line)[:200])
        if not args.dry_run:
            subprocess.run(line, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
