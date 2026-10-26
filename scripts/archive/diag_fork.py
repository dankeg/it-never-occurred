"""Diagnostic for E2: compare continuous run A with restored run C right at the fork point and after k ticks.

Archived (C5, 2026-10-18): a one-off that found the E2 fork faults (D4, D5). Kept for reference; run it with
`PYTHONPATH=scripts`.
"""

from __future__ import annotations

import sys

import harness
import scripted_routine


def drive(client, start, end):
    """Advance from start to end at routine input boundaries and return the final tick."""
    current = start
    while current < end:
        harness.apply_commands(client, scripted_routine.actions_for(current))
        after = min(end, scripted_routine.next_action_time(current))
        client.request("step", {"n": after - current})
        current = after
    return current


def dump(client):
    """Return canonical entities in the fixed wide diagnostic region."""
    return client.request("dump_entities", {"region": {"x0": -160, "z0": -160, "x1": 159, "z1": 159,
                                                       "y0": -64, "y1": 0}})["entities"]


def diff(label, a, b):
    """Print entity and NBT differences between continuous and restored dumps."""
    print(f"[{label}] entities A={len(a)} C={len(b)}")
    bu = {e["uuid"]: e for e in b}
    for ea in a:
        eb = bu.get(ea["uuid"])
        if eb is None:
            print(f"  missing in C: {ea['uuid']} {ea['type']}")
            continue
        for key in ("pos", "motion", "rot"):
            if ea[key] != eb[key]:
                print(f"  {ea['type']} {key}: A={ea[key]} C={eb[key]}")
        na, nb = ea["nbt"], eb["nbt"]
        if na != nb:
            i = next((k for k in range(min(len(na), len(nb))) if na[k] != nb[k]), min(len(na), len(nb)))
            print(f"  {ea['type']} nbt differs at {i}:\n    A: ...{na[max(0, i - 120):i + 160]}"
                  f"\n    C: ...{nb[max(0, i - 120):i + 160]}")


def main() -> int:
    """Compare snapshot and continuous runs at the fork and optional later offset."""
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    runs = harness.Path(__file__).resolve().parents[1] / "runs"
    a_dir, c_dir = runs / "fork_a", runs / "fork_c"
    for d in (a_dir, c_dir):
        if d.exists():
            raise FileExistsError(d)
    harness.clone_apfs(runs / "base", a_dir)
    with harness.Server(a_dir) as server:
        start = harness.enable_lockstep(server.client)
        harness.apply_commands(server.client, scripted_routine.initial_commands())
        snap = drive(server.client, start, start + offset)
        server.client.request("save_snapshot")
        harness.clone_apfs(a_dir / "world", c_dir / "world") if False else None
        a0 = dump(server.client)
        # clone the whole run dir (server files + saved world) for C
        harness.clone_apfs(a_dir, c_dir)
        a1_time = drive(server.client, snap, snap + k)
        a1 = dump(server.client)
    with harness.Server(c_dir) as server:
        t = harness.enable_lockstep(server.client)
        assert t == snap, (t, snap)
        server.client.request("restore_bots")
        c0 = dump(server.client)
        drive(server.client, snap, snap + k)
        c1 = dump(server.client)
    diff(f"at fork t={snap}", a0, c0)
    diff(f"after {k} ticks t={a1_time}", a1, c1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
