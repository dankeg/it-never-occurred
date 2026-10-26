"""Build the Jubilee fixture on an APFS clone, or write its server-free review map.

Use ``python scripts/build_town.py --plan-only`` for the manifest and PNG only.
Running without that flag explicitly starts a server; fixture builds and IT-24
are reserved for the integration reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import harness
    from .build_village import VillageBuilder
    from .provenance import CodeState, FixtureMetadata
    from .setup_server import REPO_ROOT
    from .town_kit import town_commands
    from .town_plan import TownPlan
else:
    import harness
    from build_village import VillageBuilder
    from provenance import CodeState, FixtureMetadata
    from setup_server import REPO_ROOT
    from town_kit import town_commands
    from town_plan import TownPlan


class TownBuilder(VillageBuilder):
    """Apply a TownPlan using unchanged legacy protocol helpers, not legacy geometry."""

    def __init__(self, plan: TownPlan):
        """Retain a pure plan without starting Minecraft or touching any fixture."""
        super().__init__(plan)
        self.plan = plan

    def commands(self) -> list[str]:
        """Return the deterministic static construction stream for review and tests."""
        return town_commands(self.plan)

    def _command(self, command: str) -> dict:
        """Allow verified no-op block placements without hiding command or material errors.

        The harness rejects vanilla's zero-change fills, including clearing the
        already-empty base and shared room walls. Only well-formed, bounded block
        commands qualify; an engine block predicate must also confirm the target
        material and properties after the rejection.
        """
        try:
            return super()._command(command)
        except harness.CommandError as error:
            parts = command.split()
            if not ((parts[0] == "fill" and len(parts) == 8) or
                    (parts[0] == "setblock" and len(parts) == 5)):
                raise
            coordinates = list(map(int, parts[1:-1]))
            first, last = coordinates[:3], coordinates[-3:]
            region = self.plan.layout["fixture_hash_region"]
            for axis, low, high in zip(("x", "y", "z"), first, last):
                if not region[axis + "0"] <= low <= high <= region[axis + "1"]:
                    raise error
            if ((last[0] - first[0] + 1) * (last[1] - first[1] + 1)
                    * (last[2] - first[2] + 1) > 32_768):
                raise error
            predicate = "execute if block " + " ".join(map(str, first)) + " " + parts[-1]
            try:
                result = super()._command(predicate)
            except harness.CommandError:
                raise error
            if result.get("value", 0) <= 0:
                raise error
            return {"value": 0, "unchanged": True}

    def _stock_commands(self, container: dict) -> list[str]:
        """Split content stock into real stacks, retaining the exact bread-day component."""
        position = " ".join(map(str, container["pos"]))
        commands = [f"setblock {position} minecraft:chest[facing=south,type=single]"]
        slot = 2
        for item, stock in container["stock"].items():
            count = stock["count"] if isinstance(stock, dict) else stock
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid stock count for {item}: {stock!r}")
            item_spec = item
            if isinstance(stock, dict) and "baked_day" in stock:
                day = stock["baked_day"]
                if type(day) is not int:
                    raise ValueError(f"Invalid baked_day for {item}: {day!r}")
                item_spec += f"[minecraft:custom_data={{simharness_baked_day:{day}}}]"
            stack_size = 1 if item in ("minecraft:writable_book", "minecraft:iron_pickaxe") else 64
            while count:
                if slot >= 27:
                    raise ValueError("Content stock exceeds the chest capacity")
                amount = min(count, stack_size)
                commands.append(f"item replace block {position} container.{slot} with {item_spec} {amount}")
                count -= amount
                slot += 1
        return commands

    def place_documents(self) -> None:
        """Stock declared chests, author actual books/signs, and apply owner-only unlocked ACLs."""
        for container in self.plan.containers.values():
            for command in self._stock_commands(container):
                self._command(command)
        origins = self.plan.layout.setdefault("setup", {}).setdefault("document_origins", {})
        for document in self.plan.documents:
            author = document["author"]
            self._command(f"item replace entity {author} hotbar.8 with minecraft:writable_book")
            result = self.client.request("write_book", {
                "name": author, "slot": 8, "title": document["title"], "pages": document["pages"]})
            origins[document["id"]] = result["prov"]
            position = " ".join(map(str, self.plan.containers[document["container"]]["pos"]))
            self._command(f"item replace block {position} container.{document['slot']} from entity {author} hotbar.8")
            self._command(f"item replace entity {author} hotbar.8 with minecraft:air")
        for name in self.plan.content["stalls"]:
            stall = self.plan.containers[name]
            self._sign(tuple(stall["sign_pos"]), stall["sign"])
            self.client.request("shop_set", {"pos": stall["pos"], "prices": stall["prices"]})
        for container in self.plan.containers.values():
            if container["allowed"] is not None or "owner" in container:
                arguments = {"pos": container["pos"], "allowed": container["allowed"]}
                if "owner" in container:
                    arguments["owner"] = container["owner"]
                self.client.request("acl_set", arguments)
        mayor = next(role for role in self.plan.roles if role.name == "mayor")
        self._command("tp mayor " + " ".join(map(str, mayor.spawn)))

    def build_port(self) -> None:
        """Register an intact ferry using the same boat and landing protocol as the village."""
        boat = self.plan.layout["port"]["boat"]
        position = " ".join(map(str, boat["pos"]))
        self._command(f'summon {boat["type"]} {position} '
                      '{UUID:[I;0,0,0,703],Tags:["town_boat"],NoGravity:1b,Invulnerable:1b}')
        self.client.request("ferry_setup", self.plan.layout["ferry"])

    def build(self, client: harness.Client) -> None:
        """Settle 100 ticks, align daylight to game time, then record the paused fixture."""
        self.client = client
        before = harness.enable_lockstep(client, recording=False)
        if before >= 900:
            raise harness.HarnessError("Town requires the prepared day-0 base before NPC departures")
        client.request("reseed", {"base_seed": self.plan.seed})
        self.plan.layout["setup"] = {"start_game_time": before}
        defaults = client.request("config_get")
        notable = {"minecraft:lantern", "minecraft:smoker", "minecraft:oak_trapdoor",
                   "minecraft:oak_door", "minecraft:chest", "minecraft:crafting_table", "minecraft:hay_block",
                   "minecraft:bell"}
        client.request("config_set", {**self.plan.layout["config"],
                                      "notable_blocks": sorted(set(defaults["notable_blocks"]) | notable)})
        for command in self.commands():
            self._command(command)
        self.build_port()
        self.spawn_roles()
        self.place_documents()
        self.equip_roles()
        self.populate()
        after = harness.game_time(client.request("step", {"n": 100}))
        if after != before + 100:
            raise harness.ProtocolError(f"Town setup advanced {after - before} ticks instead of 100")
        # the day-time clock must match the game time; `time set` fails when nothing changes, and `time add 0` reads it
        if self.client.request("command", {"command": "time add 0"})["value"] % 24_000 != after % 24_000:
            self._command(f"time set {after % 24_000}")
        self.plan.layout["setup"]["recording_start_game_time"] = after
        client.request("recording", {"enabled": True})
        client.request("save_snapshot")

    def run(self, args, *, manifest_path: Path | None = None) -> Path:
        """Clone without overwriting, then save both manifests, the map and fixture provenance."""
        directory = args.runs_dir / self.plan.fixture_name
        harness.clone_apfs(args.runs_dir / "base", directory)
        options = harness.server_options(args)
        options["base_seed"] = self.plan.seed
        with harness.Server(directory, **options) as server:
            self.build(server.client)
            result = server.client.request("hash", {"region": self.plan.layout["fixture_hash_region"]})
            world_hash = harness.hash_components(result, harness.game_time(result))["total"]
        self.plan.write(directory / "scenario.json")
        manifest = manifest_path or REPO_ROOT / "scenario/town_jubilee.json"
        self.plan.write(manifest)
        self.write_map(manifest.with_suffix(".png"))
        FixtureMetadata.write(directory, CodeState.capture(harness.MOD_JAR), manifest=manifest, world_hash=world_hash)
        return directory

    def write_map(self, path: Path) -> Path:
        """Render the pure plan alongside its manifest with no server dependency."""
        if __package__:
            from .town_map import render_map
        else:
            from town_map import render_map
        return render_map(self.plan, path)


def main(argv: list[str] | None = None) -> int:
    """Write a plan-only preview, or explicitly build a new Jubilee fixture."""
    parser = harness.experiment_parser(__doc__)
    parser.set_defaults(base_seed=None)
    parser.add_argument("--seed", dest="base_seed", type=int)
    parser.add_argument("--content", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "scenario/town_jubilee.json")
    parser.add_argument("--plan-only", action="store_true", help="write JSON and PNG only; never launch a server")
    args = parser.parse_args(argv)
    builder = TownBuilder(TownPlan.create(args.base_seed, args.content))
    if args.plan_only:
        builder.plan.write(args.manifest)
        builder.write_map(args.manifest.with_suffix(".png"))
        print(json.dumps({"manifest": str(args.manifest), "map": str(args.manifest.with_suffix('.png'))}))
    else:
        harness.stop_on_sigterm()
        args.base_seed = builder.plan.seed
        print(builder.run(args, manifest_path=args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
