"""Free disk space by stripping finished runs down to the logs that analysis and the viewer use.

A finished run keeps its orchestrator logs (decisions, trajectory, hashes, stats), its reports, and the harness logs
in `world/simharness/` (events, villager observations and trips), except the exposure audit `observations.jsonl`.
Everything else in `world/` (region files, entities, player data) is deleted: it is only needed to resume or fork the
run, and a run's world can always be recreated by replaying its decision log from the built village.

Usage: `python3.11 scripts/prune_runs.py runs/<run> [...]`, or `--all-finished` for every run in `runs/` that is not
live (no log written in the last 10 minutes) and is not a built village or base world.

`--plan` (C7, retention) deletes nothing: it lists the finished runs that pruning would strip (top-level runs and runs
one level down, as in `gates_*` directories), with the size of what pruning removes, and keeps out of the list every run
named, or whose parent is named, in `STATUS.md`, `DECISIONS.md` or a document under `docs/`. Sizes are apparent sizes:
APFS clones share blocks, so the disk actually freed can be smaller. Nothing is pruned without the owner's approval of
such a plan.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

KEEP_IN_HARNESS_LOGS = {"events.jsonl", "npc_observations.jsonl", "welfare.jsonl", "harness_state.json", "schematics"}
PROTECTED = ("base", "base_v1", "village_", "_cache")


def prunable(run: Path) -> list[Path]:
    """What pruning a run removes: its world data except the harness logs that analysis and the viewer use."""
    world = Path(run) / "world"
    if not world.is_dir():
        return []
    paths = []
    for entry in world.iterdir():
        if entry.name == "simharness":
            paths += [log for log in entry.iterdir() if log.name not in KEEP_IN_HARNESS_LOGS]
        else:
            paths.append(entry)
    return paths


def prune_run(run: Path) -> int:
    """Delete a run's world data except its harness logs; return the bytes freed (approximately)."""
    freed = 0
    for path in prunable(run):
        freed += _size(path)
        _remove(path)
    return freed


def _size(path: Path) -> int:
    """Total size of a file or directory tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _remove(path: Path) -> None:
    """Delete a file or a directory tree."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def finished_runs(runs_dir: Path, idle_seconds: float = 600.0) -> list[Path]:
    """Runs in `runs_dir` that have a world, are not protected, and have not been written to recently."""
    result = []
    now = time.time()
    for run in sorted(runs_dir.iterdir()):
        if not (run / "world").is_dir() or run.name.startswith(PROTECTED):
            continue
        logs = [path for path in (run / "world/simharness").glob("*.jsonl")] + list((run / "orchestrator").glob("*"))
        if logs and now - max(path.stat().st_mtime for path in logs) < idle_seconds:
            continue
        result.append(run)
    return result


def referenced_names(repo: Path) -> set[str]:
    """Run-directory-like names (`runs/<name>` or a bare `<name>` word) mentioned in the record and the docs."""
    texts = [repo / "STATUS.md", repo / "DECISIONS.md", *sorted((repo / "docs").rglob("*.md"))]
    names = set()
    for path in texts:
        if path.exists():
            names.update(re.findall(r"[A-Za-z0-9_.-]+", path.read_text(encoding="utf-8")))
    return names


def retention_plan(runs_dir: Path, repo: Path, idle_seconds: float = 600.0) -> dict:
    """Finished runs (top-level ones, and runs nested one level down, as in `gates_*` and probe directories) split
    into prune candidates and runs kept because the record or a document names them or their parent. Top-level
    fixtures are never listed; fixture clones inside a gate directory are scratch and are."""
    cited = referenced_names(repo)
    now = time.time()
    candidates, kept = [], []
    for top in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        nested = [] if (top / "world").is_dir() else sorted(path for path in top.iterdir() if (path / "world").is_dir())
        for run in ([top] if (top / "world").is_dir() else nested):
            if run == top and top.name.startswith(PROTECTED):
                continue
            logs = list((run / "world/simharness").glob("*.jsonl")) + list((run / "orchestrator").glob("*"))
            if logs and now - max(path.stat().st_mtime for path in logs) < idle_seconds:
                continue
            entry = {"run": str(run.relative_to(runs_dir)), "world_bytes": sum(_size(path) for path in prunable(run))}
            (kept if {top.name, run.name} & cited else candidates).append(entry)
    return {"candidates": candidates, "kept_cited": kept,
            "reclaimable_bytes": sum(entry["world_bytes"] for entry in candidates)}


def main(argv: list[str] | None = None) -> int:
    """Prune the given runs, or every finished run; with `--plan`, only list what pruning would remove."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--all-finished", action="store_true")
    parser.add_argument("--plan", action="store_true", help="list prune candidates and cited runs; delete nothing")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    args = parser.parse_args(argv)
    if args.plan:
        plan = retention_plan(args.runs_dir, Path(__file__).resolve().parents[1])
        for entry in sorted(plan["candidates"], key=lambda item: -item["world_bytes"]):
            print(f"prune  {entry['run']}: {entry['world_bytes'] / 1e9:.2f} GB")
        for entry in plan["kept_cited"]:
            print(f"keep   {entry['run']} (cited): {entry['world_bytes'] / 1e9:.2f} GB")
        print(f"{len(plan['candidates'])} candidates, {plan['reclaimable_bytes'] / 1e9:.2f} GB of world data; "
              f"{len(plan['kept_cited'])} cited runs kept")
        return 0
    runs = finished_runs(args.runs_dir) if args.all_finished else args.runs
    total = 0
    for run in runs:
        freed = prune_run(run)
        total += freed
        print(f"{run.name}: freed {freed / 1e9:.2f} GB")
    print(f"total freed: {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
