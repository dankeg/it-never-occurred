"""Create and save runs/base after exactly 100 lockstep ticks."""

from __future__ import annotations

if __package__:
    from . import harness
    from .provenance import CodeState, FixtureMetadata
    from .setup_server import setup_server
else:
    import harness
    from provenance import CodeState, FixtureMetadata
    from setup_server import setup_server


GAME_RULES = (
    ("minecraft:advance_weather", False),
    ("minecraft:spawn_mobs", False),
    ("minecraft:random_tick_speed", 0),
    ("minecraft:advance_time", True),
)


def configure_gamerules(client: harness.Client) -> None:
    """Query before setting: 26.3 rejects assignments of the current value."""
    for name, desired in GAME_RULES:
        command = f"gamerule {name}"
        current = client.request("command", {"command": command}).get("value")
        if type(current) is not int or (type(desired) is bool and current not in (0, 1)):
            raise harness.ProtocolError(f"Invalid gamerule query result for {name}: {current!r}")
        if current != int(desired):
            value = str(desired).lower()
            client.request("command", {"command": f"{command} {value}"})


def main(argv: list[str] | None = None) -> int:
    """Parse argv, prepare the base world, and return zero after exactly 100 ticks."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    args = harness.experiment_parser(__doc__).parse_args(argv)
    directory = setup_server("base", args.runs_dir, bootstrap_base=True)
    with harness.Server(directory, **harness.server_options(args)) as server:
        start = harness.enable_lockstep(server.client, await_entities=False)  # nothing is force-loaded yet
        configure_gamerules(server.client)
        # forceload is capped at 256 chunks per command: two halves of the 320x320-block (20x20-chunk) area.
        for area in ("-160 -160 -1 159", "0 -160 159 159"):
            server.client.request("command", {"command": f"forceload add {area}"})
        query = server.client.request("command", {"command": "forceload query"})
        print(f"forceload query result: {query}")
        server.client.request("await_entities", harness.FORCELOADED_CHUNKS)  # generated without ticking
        end = harness.game_time(server.client.request("step", {"n": 100}))
        if end != start + 100:
            raise harness.ProtocolError(f"World preparation advanced {end - start} ticks, expected 100")
        world_hash = harness.hash_components(server.client.request("hash", {"region": harness.REGION}), end)["total"]
    FixtureMetadata.write(directory, CodeState.capture(harness.MOD_JAR), world_hash=world_hash)
    print(f"Prepared {directory / 'world'} at game time {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
