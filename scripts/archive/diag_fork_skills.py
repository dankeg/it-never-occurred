"""Diagnostic for test_e2_skills: entity dumps of continuous A vs restored C at the fork and k ticks later.

Archived (C5, 2026-10-18): a one-off that found the E2-skills fork fault (D6). Kept for reference; run it with
`PYTHONPATH=scripts`.
"""

from __future__ import annotations

import sys

import harness
import test_e2_skills as e2s
import test_s1 as skills
from diag_fork import diff
from setup_server import setup_server

WIDE = {"x0": -160, "z0": -160, "x1": 159, "z1": 159, "y0": -64, "y1": 0}


def dump(client):
    """Return canonical entities in the skill diagnostic's wide region."""
    return client.request("dump_entities", {"region": WIDE})["entities"]


def main() -> int:
    """Print continuous-versus-restored entity differences around the skill fork."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    runs = harness.Path(__file__).resolve().parents[1] / "runs"
    a_dir = runs / "dfs_a"
    c_dir = setup_server("dfs_c", runs)
    harness.clone_apfs(runs / "base", a_dir)
    skills.write_schematic(a_dir, "e2_skills_deck")
    with harness.Server(a_dir) as server:
        start = harness.enable_lockstep(server.client)
        e2s.setup(server.client)
        e2s.run_segment(server.client, start, 0, e2s.FORK_TICKS, [], [])
        server.client.request("save_snapshot")
        a0 = dump(server.client)
        harness.clone_apfs(a_dir / "world", c_dir / "world")
        server.client.request("step", {"n": k})
        a1 = dump(server.client)
    with harness.Server(c_dir) as server:
        harness.enable_lockstep(server.client)
        server.client.request("restore_bots")
        c0 = dump(server.client)
        server.client.request("step", {"n": k})
        c1 = dump(server.client)
    diff("at fork", a0, c0)
    diff(f"after {k}", a1, c1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
