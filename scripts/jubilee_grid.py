"""Submit the Jubilee episodes to the run queue (`scripts/run_queue.py`); nothing runs until `serve`.

Each episode is one queue job: `run_jubilee.py` on the built town with its condition file, Sol at low reasoning,
checkpoints every 4,800 ticks, and the exact-replay check. Names are `jub_<condition>_<n>`; an existing finished run is
never resubmitted. `--dry-run` prints the commands only.

Usage: `python3.11 scripts/jubilee_grid.py [--dry-run] A_committee=6 B_watched=6 C_separate=6 D_ordinary=3
E_physics_off=4`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ROOT / "scenario" / "jubilee_conditions"


def command(name: str, condition: str, max_calls: int) -> list[str]:
    """The queue command for one episode (`{port}` and `{port+1}` are filled by the queue)."""
    return ["python3.11", "scripts/run_jubilee.py", "--runs-dir", "runs", "--provider", "chatgpt-plan",
            "--model", "gpt-6.1-sol", "--reasoning-effort", "low", "--max-calls", str(max_calls),
            "--name", name, "--condition", str(CONDITIONS.relative_to(ROOT) / f"{condition}.json"),
            "--checkpoint-every", "4800", "--port", "{port}", "--game-port", "{port+1}", "--heap", "3G", "--replay"]


def main(argv: list[str] | None = None) -> int:
    """Submit (or print) one job per episode, interleaving conditions so every condition advances together."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", nargs="+", help="CONDITION=N, e.g. A_committee=6")
    parser.add_argument("--max-calls", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    counts = {}
    for entry in args.counts:
        condition, count = entry.split("=")
        if not (CONDITIONS / f"{condition}.json").exists():
            parser.error(f"no condition file {condition}.json")
        counts[condition] = int(count)
    jobs = [(f"jub_{condition.split('_')[0]}_{n}", condition) for n in range(1, max(counts.values()) + 1)
            for condition, count in counts.items() if n <= count]
    submitted = {path.stem for path in (ROOT / "runs" / "queue" / "jobs").glob("*.json")}
    for name, condition in jobs:
        if (ROOT / "runs" / name / "jubilee_report.json").exists() or name in submitted:
            print(f"skip {name}: finished or already submitted")
            continue
        line = ["python3.11", "scripts/run_queue.py", "submit", "--servers", "1", name, "--",
                *command(name, condition, args.max_calls)]
        print(" ".join(line))
        if not args.dry_run:
            subprocess.run(line, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
