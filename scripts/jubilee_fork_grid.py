"""Submit branches at each outsider's first approach in finished episodes to the run queue (Appendix D.8).

For each finished episode whose exact replay passed, `jubilee_candidates.py` finds each outsider's first approach by
a member; the baker and the secretary in condition A get one queue job each running the fork command
(`resample`, `mute` and `brief` arms, 2 samples, horizon 1,200 ticks, from the latest checkpoint) with Sol. A fork
point already submitted or finished is skipped. `--dry-run` prints the commands only.

Usage: `python3.11 scripts/jubilee_fork_grid.py runs/jub_A_* runs/jub_C_* [--dry-run]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.jubilee_candidates import JubileeCandidates  # noqa: E402
from scripts.village_session import Checkpoints  # noqa: E402

MODEL = ["--model", "gpt-6.1-sol", "--provider", "chatgpt-plan", "--reasoning-effort", "low", "--max-calls", "400"]
# the baker's and the secretary's first approach, in condition A, 2 samples, horizon 1,200 ticks
FORKED = {"A": ("baker", "secretary")}
DESIGN = {"--samples": "2", "--horizon": "1200"}


def replay_passed(run: Path) -> bool:
    """Whether the episode finished and its queue log (or a later replay-only job's log) reports a passing replay."""
    logs = [ROOT / "runs" / "queue" / f"{name}.log" for name in (run.name, f"replay_{run.name}")]
    return (run / "jubilee_report.json").exists() and any(
        log.exists() and "PASS replay" in log.read_text(errors="ignore") for log in logs)


def main(argv: list[str] | None = None) -> int:
    """Submit (or print) one queue job per approached outsider of each eligible episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    queued = {path.stem for path in (ROOT / "runs" / "queue" / "jobs").glob("*.json")}
    for run in args.runs:
        if not replay_passed(run):
            print(f"skip {run.name}: not finished or replay not passed")
            continue
        report = JubileeCandidates(run).report()
        (run / "jubilee_candidates.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
        forked = FORKED.get(run.name.split("_")[1], ())
        for outsider, entry in report.items():
            name = f"fork_{run.name}_{outsider}"
            if outsider not in forked or not entry["approached"] or name in queued:
                continue
            command = shlex.split(entry["fork"])
            for flag, value in DESIGN.items():
                command[command.index(flag) + 1] = value
            try:  # an approach before the first checkpoint forks from the fixture (the runner replays from start)
                Checkpoints.latest(run, entry["first"]["decision_tick"])
            except ValueError:
                command.remove("--from-checkpoint")
            command += MODEL + ["--runs-dir", "runs", "--port", "{port}", "--game-port", "{port+1}", "--heap", "3G"]
            line = ["python3.11", "scripts/run_queue.py", "submit", "--servers", "1", name, "--", *command]
            print(" ".join(shlex.quote(part) for part in line)[:300])
            if not args.dry_run:
                subprocess.run(line, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
