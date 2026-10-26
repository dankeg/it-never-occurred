"""Judge finished episodes and queue their branches while the run queue works (as used during the study).

Every `--every` seconds, for the grid's episodes (`runs/jub_*`):
- a finished episode whose replay passed gets its leak and awareness judgement (`jubilee_leaks.py`, Luna for speech,
  Sol for the mayor's awareness) if it has none yet;
- in conditions A and C, its fork points are submitted to the queue (`jubilee_fork_grid.py`; already-queued ones are
  skipped);
- in conditions B, D and E (never forked), its checkpoints are removed once judged, to save disk.
It exits when every expected episode has been judged and every fork submission attempted.

Usage: `python3.11 scripts/jubilee_pipeline.py --expect 25 [--every 300]`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def passed(run: Path) -> bool:
    """Finished, with a passing replay in its queue log or in a later replay-only job's log."""
    logs = [RUNS / "queue" / f"{name}.log" for name in (run.name, f"replay_{run.name}")]
    return (run / "jubilee_report.json").exists() and any(
        log.exists() and "PASS replay" in log.read_text(errors="ignore") for log in logs)


def step(forks: bool) -> tuple[int, int]:
    """One pass (fork points submitted only when `forks`); returns (episodes judged, episodes found)."""
    runs = sorted(path for path in RUNS.glob("jub_*") if path.is_dir() and not path.name.endswith("_replay"))
    judged = 0
    for run in runs:
        if not passed(run):
            continue
        leaks = run / "jubilee_leaks.json"
        if not leaks.exists() or "mayor_awareness" not in leaks.read_text(encoding="utf-8"):
            subprocess.run([sys.executable, "scripts/jubilee_leaks.py", str(run.relative_to(ROOT)),
                            "--model", "gpt-6.1-sol", "--reasoning-effort", "low"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, check=False)
        if (run / "jubilee_leaks.json").exists():
            judged += 1
        condition = run.name.split("_")[1]
        if condition in ("A", "C"):
            if not forks:
                continue
            subprocess.run([sys.executable, "scripts/jubilee_fork_grid.py", str(run.relative_to(ROOT))], cwd=ROOT,
                           stdout=subprocess.DEVNULL, check=False)
        elif (run / "checkpoints").exists() and (run / "jubilee_leaks.json").exists():
            shutil.rmtree(run / "checkpoints")
    return judged, len(runs)


def main(argv: list[str] | None = None) -> int:
    """Loop until the expected episodes are judged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", type=int, required=True)
    parser.add_argument("--every", type=int, default=300)
    parser.add_argument("--forks", action="store_true", help="also submit A and C fork points")
    args = parser.parse_args(argv)
    while True:
        judged, found = step(args.forks)
        print(f"{time.strftime('%H:%M:%S')} judged {judged} of {args.expect} ({found} started)", flush=True)
        if judged >= args.expect:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
