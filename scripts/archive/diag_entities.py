"""Diagnostic: run two fresh copies of runs/base to a target game time and diff per-entity canonical dumps.

Archived (C5, 2026-10-18): a one-off that found the entity-order determinism faults fixed in M0b. Kept for reference;
run it with `PYTHONPATH=scripts`.
"""

from __future__ import annotations

import sys

import harness
import scripted_routine


def dump_at(directory, target_offset: int) -> list[dict]:
    """Run the routine for target_offset ticks and return the regional entity dump."""
    with harness.Server(directory) as server:
        start = harness.enable_lockstep(server.client)
        harness.apply_commands(server.client, scripted_routine.initial_commands())
        current = start
        end = start + target_offset
        while current < end:
            harness.apply_commands(server.client, scripted_routine.actions_for(current))
            after = min(end, scripted_routine.next_action_time(current))
            server.client.request("step", {"n": after - current})
            current = after
        return server.client.request("dump_entities", {"region": harness.REGION})["entities"]


def main() -> int:
    """Compare two fresh runs at the optional tick offset and print entity differences."""
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    runs = harness.Path(__file__).resolve().parents[1] / "runs"
    dumps = []
    for name in ("diag_a", "diag_b"):
        directory = runs / name
        if directory.exists():
            raise FileExistsError(directory)
        harness.clone_apfs(runs / "base", directory)
        dumps.append(dump_at(directory, offset))
    a, b = dumps
    print(f"entities: A={len(a)} B={len(b)}")
    for ea, eb in zip(a, b):
        for key in ("uuid", "type", "pos", "motion", "rot"):
            if ea[key] != eb[key]:
                print(f"DIFF {ea['uuid']} {key}: A={ea[key]} B={eb[key]}")
        if ea["nbt"] != eb["nbt"]:
            na, nb = ea["nbt"], eb["nbt"]
            i = next(k for k in range(min(len(na), len(nb))) if na[k] != nb[k])
            print(f"DIFF {ea['uuid']} nbt at char {i}:\n  A: ...{na[max(0, i - 150):i + 150]}"
                  f"\n  B: ...{nb[max(0, i - 150):i + 150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
