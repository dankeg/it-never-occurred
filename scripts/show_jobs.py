"""The exact command of every episode, replay and branch in the paper, from the run queue's job log ($0).

`runs/queue/jobs/<name>.json` records each job's command line as it was submitted; `{port}` and `{port+N}` were
filled in by the queue. With no argument this lists the jobs by kind; with a name it prints that job's command, with
the ports filled in, ready to paste (or rerun it through the queue: `scripts/run_queue.py submit NAME -- ...`).

Usage: `python3 scripts/show_jobs.py` and `python3 scripts/show_jobs.py jub_L2_1 [--port 25600]`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import shlex

JOBS = Path(__file__).resolve().parents[1] / "runs" / "queue" / "jobs"
KINDS = {"jub": "episode", "replay": "replay-only check of an episode", "asked": "noon branch: told to ask itself",
         "jubilee": "noon branch: told of its fortieth year", "candid": "noon branch: dislikes surprises",
         "self": "noon branch: likes to know what concerns it", "drive": "noon branch: raised motivation",
         "duty": "noon branch: secretary obliged to report (alone, or with the anniversary)",
         "counsellor": "Experiment 2: the monitors' question every hour", "route": "night crossing: told of the patrol",
         "sighting": "Experiment 2: branch at a member's first question", "fork": "branch at a first approach"}


def main(argv: list[str] | None = None) -> int:
    """List the jobs, or print one job's command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--port", type=int, default=25600, help="first port of the block the job may use")
    args = parser.parse_args(argv)
    if args.name is None:
        groups = defaultdict(list)
        for path in sorted(JOBS.glob("*.json")):
            kind = "duty" if path.stem.startswith("duty") else path.stem.split("_")[0]
            groups[kind].append(path.stem)
        for kind, names in groups.items():
            print(f"{KINDS.get(kind, kind)} ({len(names)}): {', '.join(names[:4])}{', ...' if len(names) > 4 else ''}")
        return 0
    command = json.loads((JOBS / f"{args.name}.json").read_text(encoding="utf-8"))["command"]
    fill = lambda part: re.sub(r"\{port(?:\+(\d+))?\}", lambda m: str(args.port + int(m.group(1) or 0)), part)  # noqa: E731
    print(" ".join(shlex.quote(fill(part)) for part in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
