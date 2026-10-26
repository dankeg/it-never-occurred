"""Report M7a village probes on a fresh APFS scratch clone, never the built run.

Only explicit CLI execution starts Minecraft. Bridge checks use grounded dry-runs
but ADMIN-STAGE preceding sections on the clone; they do not certify placement
execution. The original gap and ford are restored before the NPC-day sample.
Ordinary failed checks are reports (exit 0); lifecycle/protocol failures exit 1.
"""

from __future__ import annotations

from collections import Counter, deque
import json
import math
from pathlib import Path
import re

if __package__:
    from . import harness
    from .setup_server import REPO_ROOT
else:
    import harness
    from setup_server import REPO_ROOT


class VillageProbe:
    """Inspect one manifest against an isolated copy using existing harness APIs."""

    def __init__(self, config: dict, source: Path, scratch: Path):
        """Retain the asdict-shaped manifest and paths without starting a server."""
        self.config = config
        self.source = source.expanduser().resolve()
        self.scratch = scratch.expanduser().absolute()
        self.places = {place["name"]: place for place in config["places"]}
        self.bot = "village_probe"
        self.client: harness.Client | None = None
        self.results: list[tuple[str, bool, str]] = []
        self.log_offset = 0

    def _check(self, category: str, label: str, passed: bool, detail: str = "") -> bool:
        self.results.append((category, bool(passed), label))
        if not passed:
            print(f"FAIL {category}: {label}" + (f"; {detail}" if detail else ""))
        return bool(passed)

    def _category(self, category: str, action) -> None:
        try:
            action()
        except harness.ProtocolError:
            raise
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            self._check(category, "probe could not complete", False, str(error))

    def _command(self, text: str) -> dict:
        return self.client.request("command", {"command": text})

    def _condition(self, condition: str) -> int:
        try:
            return self._command("execute if " + condition)["value"]
        except harness.CommandError:
            self._command("execute unless " + condition)
            return 0

    @staticmethod
    def _coords(position) -> str:
        return " ".join(map(str, position))

    def _teleport(self, position) -> None:
        self.client.request("skill_cancel", {"name": self.bot})
        self._command(f"tp {self.bot} {self._coords(position)} 0 0")

    def _clear(self) -> None:
        for item in self.client.request("bot_state", {"name": self.bot})["inventory"]:
            self._command(f"item replace entity {self.bot} {self._slot(item['slot'])} with minecraft:air")

    @staticmethod
    def _slot(slot: int) -> str:
        if 0 <= slot < 9:
            return f"hotbar.{slot}"
        if 9 <= slot < 36:
            return f"inventory.{slot - 9}"
        raise harness.HarnessError(f"Unexpected probe inventory slot {slot}")

    def _dry(self, skill: str, args: dict) -> dict:
        return self.client.request("skill_start", {
            "name": self.bot, "skill": skill, "args": args, "dry_run": True})

    def _move(self, category: str, label: str, start, finish, feasible: bool = True) -> bool:
        self._teleport(start)
        result = self._dry("move_to", dict(zip(("x", "y", "z"), finish)))
        passed = result.get("feasible") is feasible and (feasible or result.get("reason") == "no_path")
        return self._check(category, label, passed, json.dumps(result, sort_keys=True))

    def _feet(self, name: str) -> tuple:
        horizontal, height, depth = self.places[name]["pos"]
        return horizontal + 0.5, height + 1, depth + 0.5

    def _step(self, ticks: int) -> None:
        before = harness.game_time(self.client.request("ping"))
        remaining = ticks
        while remaining:
            count = min(200, remaining)
            after = harness.game_time(self.client.request("step", {"n": count}))
            if after != before + count:
                raise harness.ProtocolError(f"step advanced from {before} to {after}, expected {count} ticks")
            before, remaining = after, remaining - count

    def _snapshot(self) -> dict:
        self.client.request("save_snapshot")
        path = self.scratch / "world/simharness/harness_state.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved.get("npcs", {}).get("graph", {}).get("nodes"), list):
            raise harness.HarnessError("Snapshot has no persisted NPC graph nodes")
        return saved["npcs"]

    @staticmethod
    def _node(value) -> tuple:
        if isinstance(value, dict):
            return value["x"], value["y"], value["z"]
        if len(value) != 3:
            raise harness.HarnessError(f"Invalid road node: {value!r}")
        return tuple(value)

    @staticmethod
    def _walk(nodes: set, start: tuple) -> dict:
        parents = {start: None} if start in nodes else {}
        pending = deque(parents)
        while pending:
            horizontal, height, depth = pending.popleft()
            for delta_x, delta_z in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                for delta_y in (-1, 0, 1):
                    target = horizontal + delta_x, height + delta_y, depth + delta_z
                    if target in nodes and target not in parents:
                        parents[target] = (horizontal, height, depth)
                        pending.append(target)
        return parents

    def _commuters(self) -> list[dict]:
        return [npc for npc in self.config["npcs"] if npc.get("work")
                and self.places[npc["home"]]["bank"] != self.places[npc["work"]]["bank"]]

    def _through_ford(self, path: list) -> bool:
        river = self.config["layout"]["river"]["bounds"]
        ford_z = self.config["layout"]["ford"]["bounds"][2]
        return any(river[0] <= node[0] <= river[3] and node[2] == ford_z for node in map(self._node, path))

    def probe_movement(self) -> None:
        """Dry-run same-bank journeys and independent short legs via the ford."""
        for bank in ("west", "east"):
            anchor = bank + "_bridge"
            for name, place in self.places.items():
                if place["bank"] == bank and name != anchor:
                    self._move("movement", f"{anchor} -> {name}", self._feet(anchor), self._feet(name))
        waypoints = ("west_bridge", "west_ford", "east_ford", "east_bridge")
        for start, finish in zip(waypoints, waypoints[1:]):
            self._move("movement", f"ford waypoint {start} -> {finish}", self._feet(start), self._feet(finish))

    def probe_graph(self) -> None:
        """BFS actual persisted nodes with RoadGraph's cardinal, +/-1-height edges."""
        saved = self._snapshot()
        nodes = {self._node(node) for node in saved["graph"]["nodes"]}
        river = self.config["layout"]["river"]["bounds"]
        for bank in ("west", "east"):
            subset = {node for node in nodes if (node[0] < river[0] if bank == "west" else node[0] > river[3])}
            anchor = tuple(self.places[bank + "_bridge"]["pos"])
            visited = self._walk(subset, anchor)
            missing = [name for name, place in self.places.items()
                       if place["bank"] == bank and tuple(place["pos"]) not in visited]
            self._check("graph", f"{bank} bank connected", bool(subset) and len(visited) == len(subset) and not missing,
                        f"reachable={len(visited)}/{len(subset)}, missing places={missing}")
        for npc in self.config["npcs"]:
            destination = npc.get("work") or npc.get("school")
            if not destination or self.places[npc["home"]]["bank"] == self.places[destination]["bank"]:
                continue
            start, finish = tuple(self.places[npc["home"]]["pos"]), tuple(self.places[destination]["pos"])
            parents = self._walk(nodes, start)
            path, current = [], finish
            if finish in parents:
                while current is not None:
                    path.append(current)
                    current = parents[current]
            self._check("graph", f"{npc['id']} cross-bank route uses ford", bool(path) and self._through_ford(path),
                        f"BFS edges={max(0, len(path) - 1)}")

    def _capture(self, positions, candidates: tuple[str, ...]) -> dict:
        captured = {}
        for position in positions:
            coordinate = tuple(position)
            for block in candidates:
                if self._condition(f"block {self._coords(coordinate)} {block}"):
                    captured[coordinate] = block
                    break
            else:
                raise harness.HarnessError(f"Unknown original block at {coordinate}; refusing to overwrite it")
        return captured

    def _set_block(self, position, block: str) -> None:
        if not self._condition(f"block {self._coords(position)} {block}"):
            self._command(f"setblock {self._coords(position)} {block}")

    def _restore(self, original: dict) -> None:
        for position, block in original.items():
            self._set_block(position, block)

    def _restore_baseline(self, original: dict, before: set, category: str) -> None:
        try:
            self._restore(original)
            self._step(1)
            after = {self._node(node) for node in self._snapshot()["graph"]["nodes"]}
            if not self._check(category, "original blocks and graph restored", before == after):
                raise harness.HarnessError("Restored node set differs from baseline")
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError) as error:
            raise harness.ProtocolError(
                f"{category}: restoration failed; refusing an altered NPC baseline: {error}") from error

    def probe_ford_closure(self) -> None:
        """Temporarily flood the complete gravel column, then restore exact surfaces."""
        bounds = self.config["layout"]["ford"]["bounds"]
        positions = ((horizontal, height, depth) for horizontal in range(bounds[0], bounds[3] + 1)
                     for height in range(bounds[1], bounds[4] + 1) for depth in range(bounds[2], bounds[5] + 1))
        original = self._capture(positions, ("minecraft:gravel", "minecraft:dirt_path", "minecraft:stone_bricks"))
        before = {self._node(node) for node in self._snapshot()["graph"]["nodes"]}
        try:
            self._command(f"fill {self._coords(bounds)} minecraft:water")
            self._step(1)
            for crossing in ("ford", "bridge"):
                endpoints = [self._feet(bank + "_" + crossing) for bank in ("west", "east")]
                standable = [self._move("ford closure", f"{bank} {crossing} endpoint standable", endpoint, endpoint)
                             for bank, endpoint in zip(("west", "east"), endpoints)]
                if all(standable):
                    self._move("ford closure", f"short {crossing} crossing has no_path", *endpoints, False)
                else:
                    self._check("ford closure", f"{crossing} no_path test inconclusive", False,
                                "An unstandable endpoint cannot prove a water barrier")
            nodes = {self._node(node) for node in self._snapshot()["graph"]["nodes"]}
            visited = self._walk(nodes, tuple(self.places["west_bridge"]["pos"]))
            east = {node for node in nodes if node[0] > self.config["layout"]["river"]["bounds"][3]}
            self._check("ford closure", "NPC banks disconnected",
                        bool(visited) and bool(east) and not east.intersection(visited))
        finally:
            self._restore_baseline(original, before, "ford closure")

    @staticmethod
    def _targets(section: dict) -> list[tuple]:
        return [tuple(section["origin"][index] + block[key] for index, key in enumerate(("dx", "dy", "dz")))
                for block in section["blocks"]]

    def _section(self, name: str, section: dict) -> None:
        path = self.scratch / "world/simharness/schematics" / (name + ".json")
        if json.loads(path.read_text(encoding="utf-8"))["blocks"] != section["blocks"]:
            raise harness.HarnessError(f"{name}: schematic file differs from manifest; refusing administrative staging")
        bill = section["bill"]
        computed = dict(Counter(block["block"] for block in section["blocks"]))
        if computed != bill or not bill:
            raise harness.HarnessError(f"{name}: invalid or mismatched material bill")
        self._teleport(section["stand"])
        self._clear()
        for item, count in bill.items():
            self.client.request("give", {"name": self.bot, "item": item, "count": count})
        args = {"schematic": name, "origin": section["origin"]}
        result = self._dry("place_section", args)
        self._check("schematics", f"{name} full-bill dry-run", result.get("feasible") is True
                    and result.get("estimate", {}).get("materials") == bill, json.dumps(result, sort_keys=True))
        material = next(iter(bill))
        self._command(f"clear {self.bot} {material} 1")
        short = self._dry("place_section", args)
        self._check("schematics", f"{name} one-item-short rejected", short.get("feasible") is False
                    and short.get("reason") == "missing_items", json.dumps(short, sort_keys=True))

    def probe_schematics(self) -> None:
        """Dry-run each section, ADMIN-STAGE its blocks, and finally restore the gap."""
        sections = [(name, section) for name, section in self.config["schematics"].items()
                    if name.startswith("bridge_span_")]
        sections.sort(key=lambda entry: int(entry[0].rsplit("_", 1)[1]))
        self._check("schematics", "five bridge sections", len(sections) == 5)
        originals = self._capture((position for _, section in sections for position in self._targets(section)),
                                  ("minecraft:water", "minecraft:air"))
        before = {self._node(node) for node in self._snapshot()["graph"]["nodes"]}
        print("NOTE schematics: preceding sections are ADMIN-STAGED even after a failed dry-run; "
              "no placement-execution claim.")
        try:
            for name, section in sections:
                self._category("schematics", lambda: self._section(name, section))
                path = self.scratch / "world/simharness/schematics" / (name + ".json")
                if json.loads(path.read_text(encoding="utf-8"))["blocks"] != section["blocks"]:
                    raise harness.HarnessError(f"{name}: refusing to stage a mismatched file")
                for position, block in zip(self._targets(section), section["blocks"]):
                    self._set_block(position, block["block"])
        finally:
            self._restore_baseline(originals, before, "schematics")
            self._clear()
        self._category("schematics", lambda: self._section("port_repair", self.config["schematics"]["port_repair"]))
        self._clear()

    def _skill(self, skill: str, args: dict) -> dict:
        started = self.client.request("skill_start", {"name": self.bot, "skill": skill, "args": args})
        if not started.get("accepted"):
            raise harness.HarnessError(f"{skill} rejected: {started}")
        for _ in range(100):
            self._step(5)
            status = self.client.request("skill_status", {"name": self.bot})
            if status.get("state") != "running":
                if status.get("state") != "succeeded":
                    raise harness.HarnessError(f"{skill} failed: {status}")
                return status
        self.client.request("skill_cancel", {"name": self.bot})
        raise harness.HarnessError(f"{skill} did not finish within 500 probe ticks")

    def _ledger(self) -> None:
        document = next(entry for entry in self.config["documents"] if entry["id"] == "market_ledger")
        position = self.config["containers"][document["container"]]["pos"]
        horizontal, height, depth = position
        contents = None
        for delta_x, delta_z in ((0, 1), (1, 0), (-1, 0), (0, -1)):
            self._teleport((horizontal + 0.5 + delta_x, height, depth + 0.5 + delta_z))
            try:
                contents = self.client.request("open_container", {"name": self.bot, "pos": position})
                break
            except harness.CommandError:
                continue
        if contents is None:
            raise harness.HarnessError("Ledger chest has no reachable visible adjacent reading position")
        books = [item for item in contents["items"] if item["item"] == "minecraft:written_book"]
        if (len(books) != 1 or books[0]["slot"] != document["slot"]
                or books[0].get("doc", {}).get("title") != document["title"]):
            raise harness.HarnessError("Ledger metadata differs, "
                                       "or item-only chest_take cannot uniquely select the ledger")
        self._clear()
        self._skill("chest_take", {"pos": position, "item": "minecraft:written_book", "count": 1})
        try:
            inventory = self.client.request("bot_state", {"name": self.bot})["inventory"]
            slot = next(item["slot"] for item in inventory if item["item"] == "minecraft:written_book")
            self.client.request("bot_select", {"name": self.bot, "slot": slot})
            actual = self.client.request("read_item_doc", {"name": self.bot, "slot": slot})
            self._check("evidence", "actual ledger title and exact pages", actual.get("title") == document["title"]
                        and actual.get("pages") == document["pages"], json.dumps(actual.get("pages")))
            entry = self.config["evidence"]["ledger_entry"]
            self._check("evidence", "variant ledger sale presence/absence", entry in actual.get("pages", []) if entry
                        else "sold 1 iron_pickaxe to the envoy, day 0" not in actual.get("pages", []))
        finally:
            self._skill("chest_put", {"pos": position, "item": "minecraft:written_book", "count": 1})

    def probe_evidence(self) -> None:
        """Query real block/entity conditions and take/read/return the actual ledger."""
        evidence = self.config["evidence"]
        for pos, block in zip(self.config["layout"]["bridge"]["debris"],
                              ("minecraft:cobblestone", "minecraft:oak_planks")):
            self._check("evidence", "submerged bridge debris " + self._coords(pos),
                        bool(self._condition(f"block {self._coords(pos)} {block}")))
        self._check("evidence", "one usable tagged oak boat", self._condition(
            "entity @e[type=minecraft:oak_boat,tag=village_boat]") == 1)
        for support in evidence["supports"]:
            self._check("evidence", support["id"], bool(self._condition(
                f"block {self._coords(support['pos'])} {support['block']}")))
        tool = evidence["tool"]
        tag = tool["tag"] if tool else "village_evidence_pickaxe"
        tagged = self._condition(f"entity @e[type=minecraft:item,tag={tag}]")
        self._check("evidence", "dropped tool count", tagged == (1 if tool else 0), f"tagged entities={tagged}")
        if tool:
            item = json.dumps(tool["item"])
            horizontal, height, depth = tool["pos"]
            selector = (f'@e[type=minecraft:item,tag={tag},x={horizontal},y={height},z={depth},distance=..0.75,'
                        f'nbt={{Item:{{id:{item},count:1}}}}]')
            self._check("evidence", "dropped tool actual item, count and position",
                        self._condition("entity " + selector) == 1)
            self._check("evidence", "dropped tool configured UUID", self._condition("entity " + tool["uuid"]) == 1)
        self._category("evidence", self._ledger)

    def probe_npc_day(self) -> None:
        """Collect 24,000 ticks and inspect thirty states and eight ford work trips."""
        self._teleport(self._feet("west_bridge"))
        self._clear()
        start = harness.game_time(self.client.request("ping"))
        if start % 24_000 >= 1_000:
            print("NOTE NPC day: advancing to the next day boundary to avoid pre-probe departures.")
            self._step(24_000 - start % 24_000)
            start = harness.game_time(self.client.request("ping"))
        day = start // 24_000
        self._step(24_000)
        days = self.client.request("npc_metrics", {"day": day})["days"]
        metrics = next(record for record in days if record["day"] == day)
        identities = [npc["id"] for npc in self.config["npcs"]]
        self._check("NPC day", "30 configured NPCs and daily records", len(identities) == len(set(identities)) == 30
                    and Counter(row["npc"] for row in metrics["npcs"]) == Counter(identities))
        for identity in identities:
            state = self.client.request("npc_state", {"id": identity})
            valid = (state.get("id") == identity and len(state.get("pos", [])) == 3
                     and all(math.isfinite(value) for value in state["pos"])
                     and state.get("activity") not in (None, "stranded", "error", "failed")
                     and isinstance(state.get("path"), list))
            self._check("NPC day", identity + " valid state", valid)
        trips = metrics["trips"]
        errors = [f"{trip.get('npc')}/{trip.get('activity')}" for trip in trips
                  if trip.get("npc") not in identities or trip.get("arriveTick") is None or not trip.get("path")
                  or trip.get("day") != day or trip.get("outcome") in ("unreachable", "failed", "error")]
        self._check("NPC day", "no unreachable/error trips", bool(trips) and not errors, str(errors))
        incomplete = []
        for npc in self.config["npcs"]:
            activity = "work" if npc["role"] == "adult" else "school"
            destination = npc.get(activity)
            primary = [trip for trip in trips if trip.get("npc") == npc["id"] and trip.get("activity") == activity]
            homes = [trip for trip in trips if trip.get("npc") == npc["id"] and trip.get("activity") == "home"
                     and trip.get("to") == npc["home"] and trip.get("arriveTick") is not None]
            if (destination and (len(primary) != 1 or primary[0].get("to") != destination
                                 or primary[0].get("from") != npc["home"])) or not homes:
                incomplete.append(npc["id"])
        self._check("NPC day", "complete primary and home itineraries", not incomplete, str(incomplete))
        commuters = self._commuters()
        self._check("NPC day", "eight cross-river workers configured", len(commuters) == 8)
        for npc in commuters:
            matching = [trip for trip in trips if trip.get("npc") == npc["id"] and trip.get("activity") == "work"]
            valid = len(matching) == 1
            if valid:
                trip = matching[0]
                valid = (trip.get("from") == npc["home"] and trip.get("to") == npc["work"]
                         and trip.get("arriveTick") is not None and self._through_ford(trip.get("path", [])))
            self._check("NPC day", npc["id"] + " completed work trip via ford", valid)
        with (self.scratch / "server.log").open(encoding="utf-8", errors="replace") as stream:
            stream.seek(self.log_offset)
            errors = [line.strip() for line in stream if re.search(
                r"npc.*(?:error|exception|failed)|(?:error|exception).*npc"
                r"|at org\.minecraftsim\.harness\.(?:Villagers|RoadGraph)\.",
                line, re.IGNORECASE)]
        self._check("NPC day", "no new NPC error log entries", not errors, " | ".join(errors[:3]))
        print(f"NOTE NPC day {day}: 24000 ticks collected; welfare=" + json.dumps({key: metrics.get(key) for key in
              ("meanCommute", "maxCommute", "lateShare", "missedMarketShare", "absentShare")}, sort_keys=True))

    def report(self) -> None:
        """Print category totals and explicit scope/access limitations without asserting."""
        print("\nVillage probe report (observations only; failed checks do not make this a gate):")
        for category in dict.fromkeys(category for category, _, _ in self.results):
            values = [passed for name, passed, _ in self.results if name == category]
            print(f"  {category}: {sum(values)} passed, {len(values) - sum(values)} failed")
        print(f"Scratch retained: {self.scratch}; built source was never opened by a server: {self.source}")
        print("LIMIT: bot teleports, supplies, master-key access, ford closure and prior-section staging "
              "are scratch-only admin setup.")
        print("LIMIT: dry-runs do not verify vanilla placement execution; "
              "BFS reads saved nodes, not a public graph-query API.")
        print("LIMIT: ledger read uses grounded take/read/return but may change its scratch chest slot; "
              "witness incident provenance is not fabricated.")
        print("LIMIT: no_path is bounded by 256 edges/20000 expansions; "
              "same-bank/ford legs and separate NPC BFS are reported independently.")
        print("Design deviations: see layout.deviations in the scenario manifest and docs/codex/M7a_VILLAGE_LAYOUT.md.")

    def run(self, options: dict) -> int:
        """Clone once and launch only the clone; return 1 only for unrecoverable setup/runtime errors."""
        fatal = False
        try:
            destination = self.scratch.resolve()
            if self.scratch.exists() or self.scratch.is_symlink():
                raise FileExistsError(f"Refusing to overwrite scratch run {self.scratch}")
            if self.source == destination or self.source in destination.parents or destination in self.source.parents:
                raise ValueError("Source and scratch must be separate, non-nested directories")
            harness.clone_apfs(self.source, self.scratch)
            log = self.scratch / "server.log"
            self.log_offset = log.stat().st_size if log.exists() else 0
            options = {**options, "base_seed": self.config["seed"]}
            with harness.Server(self.scratch, **options) as server:
                self.client = server.client
                harness.enable_lockstep(self.client, recording=False)
                self.client.request("restore_bots")
                self.client.request("recording", {"enabled": False})
                spawn = next(role["spawn"] for role in self.config["roles"] if role["name"] == "builder")
                self.client.request("spawn_bot", {"name": self.bot, "yaw": 0, "pitch": 0,
                                                    **dict(zip(("x", "y", "z"), spawn))})
                self.client.request("bot_flag", {"name": self.bot, "master_key": True})
                for category, action in (("movement", self.probe_movement), ("graph", self.probe_graph),
                                         ("ford closure", self.probe_ford_closure),
                                         ("schematics", self.probe_schematics),
                                         ("evidence", self.probe_evidence), ("NPC day", self.probe_npc_day)):
                    self._category(category, action)
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            fatal = True
            self._check("runtime", "unrecoverable setup/lifecycle/protocol failure", False, str(error))
        self.report()
        return 1 if fatal else 0


def main(argv: list[str] | None = None) -> int:
    """Parse an explicit runtime request; importing this module has no launch side effects."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="accident")
    parser.add_argument("--scratch", type=Path, help="New APFS clone; defaults to <runs-dir>/probe_village_<variant>")
    args = parser.parse_args(argv)
    path = REPO_ROOT / "scenario" / f"village_{args.variant}.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["variant"] != args.variant:
            raise ValueError("Manifest variant differs from --variant")
        source = args.runs_dir / f"village_{args.variant}"
        scratch = args.scratch or args.runs_dir / f"probe_village_{args.variant}"
        return VillageProbe(config, source, scratch).run(harness.server_options(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL setup: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
