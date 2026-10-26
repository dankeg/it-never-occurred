"""Calibrate the village's villager schedule from a counterfactual pair: the bridge out against the bridge intact.

Each run clones the built village, optionally fills in the missing bridge deck at the start, and runs the villagers
(bots idle) to the end of day 1. It prints commute durations per activity for day 1, so deadlines can be set: intact
commutes should be on time, and the detour via the ford should be what makes cross-river villagers late.

Archived (C5, 2026-10-18): a one-off that calibrated the villager schedule now fixed in build_village.py. Kept for
reference; run it with `PYTHONPATH=scripts`.
"""

from __future__ import annotations

import json

if __package__:
    from . import harness
else:
    import harness

END_TICK = 48_000
DECK = "fill -6 -61 0 6 -61 1 minecraft:oak_planks"


def durations(day_record: dict) -> dict[str, list[int]]:
    """Return sorted trip durations (ticks) per activity for one day's NPC record."""
    result: dict[str, list[int]] = {}
    for trip in day_record["trips"]:
        if trip.get("arriveTick") is not None:
            result.setdefault(trip["activity"], []).append(trip["arriveTick"] - trip["departTick"])
    return {activity: sorted(values) for activity, values in result.items()}


def main(argv: list[str] | None = None) -> int:
    """Run one arm (`--bridge out|in`) and write its day-1 commute durations."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--bridge", choices=("out", "in"), required=True)
    args = parser.parse_args(argv)
    directory = args.runs_dir / f"calibrate_{args.variant}_bridge_{args.bridge}"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    harness.clone_apfs(args.runs_dir / f"village_{args.variant}", directory)
    with harness.Server(directory, **harness.server_options(args)) as server:
        client = server.client
        now = harness.enable_lockstep(client, recording=True)
        client.request("restore_bots")
        if args.bridge == "in":
            client.request("command", {"command": DECK})
        while now < END_TICK:
            step = min(2_000, END_TICK - now)
            now = harness.game_time(client.request("step", {"n": step}))
        record = next(day for day in client.request("npc_metrics", {"day": 1})["days"] if day["day"] == 1)
    result = {"bridge": args.bridge, "durations": durations(record),
              "trips": [{key: trip.get(key) for key in ("npc", "activity", "from", "to", "departTick", "arriveTick")}
                        for trip in record["trips"]]}
    (directory / "calibration.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for activity, values in result["durations"].items():
        print(f"{args.bridge} {activity}: n={len(values)} min={values[0]} median={values[len(values) // 2]} "
              f"max={values[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
