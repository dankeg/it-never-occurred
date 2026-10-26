"""IT-24: two independent Jubilee town builds and live geometry/routine checks ($0).

Only explicit CLI execution starts Minecraft. Build twice from separate APFS clones
of --runs-dir/base using TownBuilder.run with private manifest paths. Preserve both fixtures and
inspect further clones, never a pre-existing town. Require equal recorded hashes
and equal live hashes over layout.fixture_hash_region before any probe mutation.
Check fixture-start daylight against game time on both clones. Require every
building's news landmark to contain its doors, and inspect the actual plaza
frontages: only declared street/lane mouths may interrupt their walls.

On the first inspection clone, finish day 0 before spawning observers or changing
blocks/configuration. Check all forty primary, market and home itineraries. Then
ask SkillPath for every directed pair of place/interior endpoints, and require a
direct two-edge route through each town-hall, inn and bakery doorway. Check the
registered ferry boat through the actual cottage glass, with an opaque negative
control. Daylight controls precede night observations of the plaza, east square,
main street, west quay, bridge, ferry/back/residential lanes and the ford; held
lights cannot rescue a missing street lamp.

All artifacts are retained below --scratch (default --runs-dir/it24_town). Existing
scratch directories are refused. Only --port + 0..7 are used, sequentially. Failed
checks and runtime errors both exit 1. This gate does not certify actual movement,
ferry operation, the positive control or replay; those have separate gates. The
owner runs this after merging night/goods APIs and building the harness jar.
"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import copy
import json
import math
from pathlib import Path

if __package__:
    from . import harness
    from .test_it15_skills import port_options
else:
    import harness
    from test_it15_skills import port_options

from orchestrator.scenario import ScenarioManifest  # noqa: E402

DAY = 24_000
NIGHT = 13_000
BUILD_TIMEOUT = 600.0
VIEWER = "it24_observer"
TARGET = "it24_target"
LIMITS = (
    "Requires an existing prepared day-0 base, merged Jubilee APIs and an already-built harness jar.",
    "APFS clones and admin teleports/block substitutions are confined to new scratch runs.",
    "Route evidence is SkillPath dry-run feasibility, not executed movement or ferry transport.",
    "The current planner is bounded by 1024 edges and 200000 expansions; no_path is a gate failure, not proof "
    "of topological disconnection.",
    "Day 0 includes the builder's recorded setup interval; a fixture past the first departure is rejected.",
    "Lighting tests recorded plaza/square/street/quay/bridge lamp pools and the central bridge lane, not blanket "
    "illumination. Dim samples exclude all recorded five-block light pools; occupied cells are omitted.",
    "Both builds publish manifests and maps under scratch only; canonical scenario files are not rewritten.",
)


class IT24TownExperiment:
    """Own two builds, isolated inspection servers, checks and their retained evidence."""

    def __init__(self, args):
        """Validate options and retain paths without starting processes or creating files."""
        self.args = args
        self.options = port_options(args)
        self.options["request_timeout"] = max(BUILD_TIMEOUT, args.request_timeout)
        self.directory = (args.scratch or args.runs_dir / "it24_town").expanduser().absolute()
        self.client: harness.Client | None = None
        self.checks: list[dict] = []
        self.config: dict = {}
        self.plan = None
        self.places: dict[str, dict] = {}
        self.layout: dict = {}
        self.artifacts: dict[str, str] = {}
        self.created = False

    def _check(self, label: str, passed: bool, detail="") -> bool:
        """Print one strict gate check and retain full, JSON-serializable diagnostics."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if not passed else ""), flush=True)
        return bool(passed)

    def _write(self, name: str, value) -> None:
        """Retain one evidence file beneath this gate's newly reserved directory."""
        path = self.directory / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.artifacts[name] = str(path)

    def _phase(self, label: str, action) -> None:
        """Continue after ordinary failed probes, but never reuse a broken protocol connection."""
        try:
            action()
        except harness.ProtocolError:
            raise
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            self._check(label + " completed", False, str(error))

    def _command(self, command: str) -> dict:
        """Issue the existing administrative command API on an inspection clone only."""
        return self.client.request("command", {"command": command})

    @staticmethod
    def _coords(position) -> str:
        """Render a protocol position for a vanilla command."""
        return " ".join(map(str, position))

    def _condition(self, condition: str) -> bool:
        """Distinguish a false vanilla predicate from a malformed/unsupported command."""
        try:
            return self._command("execute if " + condition)["value"] > 0
        except harness.CommandError:
            self._command("execute unless " + condition)
            return False

    def _teleport(self, name: str, position) -> None:
        """Place a scratch observer without advancing game time or enabling infinite materials."""
        self.client.request("skill_cancel", {"name": name})
        self._command(f"tp {name} {self._coords(position)} 0 0")

    def _step_to(self, finish: int) -> None:
        """Advance bounded lockstep batches and require the exact requested game-time interval."""
        current = harness.game_time(self.client.request("ping"))
        if finish < current:
            raise ValueError(f"Refusing to rewind game time from {current} to {finish}")
        while current < finish:
            count = min(200, finish - current)
            after = harness.game_time(self.client.request("step", {"n": count}))
            if after != current + count:
                raise harness.ProtocolError(f"step advanced from {current} to {after}, expected {count} ticks")
            current = after

    @staticmethod
    def _center(position) -> tuple:
        """Center a feet cell horizontally without mistaking it for its supporting block."""
        return math.floor(position[0]) + 0.5, position[1], math.floor(position[2]) + 0.5

    def _feet(self, name: str) -> tuple:
        """Convert the legacy Place.pos support-block convention to standing feet."""
        horizontal, height, depth = self.places[name]["pos"]
        return horizontal + 0.5, height + 1, depth + 0.5

    def _walkable(self, feet) -> bool:
        """Find conservative air/air/solid standing samples, never snapping onto a neighbouring room."""
        cell = tuple(math.floor(value) for value in feet)
        head = (cell[0], cell[1] + 1, cell[2])
        below = (cell[0], cell[1] - 1, cell[2])
        return (self._condition(f"block {self._coords(cell)} minecraft:air")
                and self._condition(f"block {self._coords(head)} minecraft:air")
                and not self._condition(f"block {self._coords(below)} minecraft:air")
                and not self._condition(f"block {self._coords(below)} minecraft:water"))

    @staticmethod
    def _bounds(section: dict) -> tuple:
        """Read either horizontal building bounds or an inclusive six-coordinate room volume."""
        bounds = section["bounds"]
        if isinstance(bounds, dict):
            return bounds["x0"], bounds["z0"], bounds["x1"], bounds["z1"]
        if len(bounds) == 4:
            return tuple(bounds)
        if len(bounds) == 6:
            return bounds[0], bounds[2], bounds[3], bounds[5]
        raise ValueError(f"Invalid bounds: {bounds}")

    def _builds(self) -> list[Path]:
        """Run the normal builder twice, retaining its logs, fixtures and manifest independently."""
        if __package__:
            from .build_town import TownBuilder
            from .town_plan import TownPlan
        else:
            from build_town import TownBuilder
            from town_plan import TownPlan

        source = (self.args.runs_dir / "base").expanduser().resolve()
        destination = self.directory.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Base and scratch must be separate, non-nested directories")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.created = True
        fixtures, hashes, manifests = [], [], []
        for number in (1, 2):
            root = self.directory / f"build_{number}"
            root.mkdir()
            harness.clone_apfs(source, root / "base")
            options = copy(self.args)
            options.runs_dir = root
            options.game_port = self.options["game_port"]
            options.request_timeout = self.options["request_timeout"]
            log_path = root / "build.log"
            self.artifacts[f"build_{number}"] = str(root)
            with log_path.open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    plan = TownPlan.create(self.args.base_seed)
                    fixture = TownBuilder(plan).run(options, manifest_path=root / "town_jubilee.json")
            if number == 1:
                self.plan = plan
            manifest = ScenarioManifest.load(fixture / "scenario.json").data
            metadata = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
            fixtures.append(fixture)
            manifests.append(manifest)
            hashes.append(metadata.get("world_hash"))
        valid = all(isinstance(value, str) and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value) for value in hashes)
        self._check("two complete builds have identical recorded world hashes",
                    valid and hashes[0] == hashes[1], hashes)
        self._check("independent built manifests are identical", manifests[0] == manifests[1])
        self._write("build_hashes.json", hashes)
        self.config = manifests[0]
        self.places = {place["name"]: place for place in self.config["places"]}
        self.layout = self.config["layout"]
        region = self.layout["fixture_hash_region"]
        chunks = harness.FORCELOADED_CHUNKS
        self._check("live hash covers the entire force-loaded horizontal area",
                    (region["x0"], region["z0"], region["x1"], region["z1"])
                    == (chunks["cx0"] * 16, chunks["cz0"] * 16,
                        chunks["cx1"] * 16 + 15, chunks["cz1"] * 16 + 15), region)
        return fixtures

    def probe_fixture_clock(self, number: int) -> None:
        """Read both clocks before advancing the clone beyond its saved fixture start."""
        start = harness.game_time(self.client.request("ping"))
        result = self._command("time add 0")  # reads the day-time clock (time query daytime fails in 26.3)
        evidence = {"game_time": start, "daytime": result["value"],
                    "recording_start_game_time": self.layout["setup"]["recording_start_game_time"]}
        self._check(f"inspection {number}: fixture-start daytime equals game time modulo 24000",
                    type(result["value"]) is int and result["value"] == start % DAY
                    and harness.game_time(result) == start == evidence["recording_start_game_time"], evidence)
        self._write(f"fixture_clock_{number}.json", evidence)

    def probe_landmarks(self) -> None:
        """Check each named building's actual npc_setup landmark, including both door halves."""
        landmarks = self.plan.npc_payload()["landmarks"]
        evidence = []
        for building in self.plan.buildings:
            name = building["name"]
            matches = [landmark for landmark in landmarks
                       if landmark["name"].lower().removeprefix("the ").replace(" ", "_") == name]
            doors = self.plan.layout["doors"].get(name, [])
            outside = list(doors)
            if len(matches) == 1:
                region = matches[0]["region"]
                minimum, maximum = region["min"], region["max"]
                if len(minimum) == len(maximum) == 3:
                    outside = [door for door in doors
                               if not all(minimum[axis] <= door[axis]
                                          and door[axis] + int(axis == 1) <= maximum[axis] for axis in range(3))]
            record = {"building": name, "landmarks": matches, "doors": doors, "outside": outside}
            self._check(f"{name}: one news landmark contains every complete door",
                        len(matches) == 1 and bool(doors) and not outside, record)
            evidence.append(record)
        self._check("news landmark checks cover every building", bool(evidence))
        self._write("building_landmarks.json", evidence)

    def probe_plaza(self) -> None:
        """Check immediate facade coverage and real walls, with only narrow paved road mouths."""
        bounds = self._bounds(self.layout["plaza"])
        if not self._check("plaza retains its specified bounds", bounds == (-46, 25, -11, 52), bounds):
            return
        overlaps = []
        for index, building in enumerate(self.plan.buildings):
            left, north, right, south = self._bounds(building)
            for other in self.plan.buildings[index + 1:]:
                other_left, other_north, other_right, other_south = self._bounds(other)
                if left <= other_right and other_left <= right and north <= other_south and other_north <= south:
                    overlaps.append([building["name"], other["name"]])
        self._check("plaza frontage plan has no overlapping building footprints", not overlaps, overlaps)
        left, north, right, south = bounds
        faces = (("west", 0, left - 1, north, south, 2),
                 ("east", 0, right + 1, north, south, 0),
                 ("north", 1, north - 1, left, right, 3),
                 ("south", 1, south + 1, left, right, 1))
        height = math.floor(self._feet("plaza")[1])
        walls = {"minecraft:stone_bricks", "minecraft:cobblestone", "minecraft:bricks", "minecraft:stone",
                 "minecraft:polished_andesite", "minecraft:white_concrete", "minecraft:calcite",
                 "minecraft:glass", "minecraft:glass_pane"}
        evidence = []
        for side, axis, face, first, last, facade_index in faces:
            cells = {}
            main_cells = set()
            outside = face + (-1 if side in ("west", "north") else 1)
            for along in range(first, last + 1):
                horizontal, depth = (face, along) if axis == 0 else (along, face)
                owners = [building["name"] for building in self.plan.buildings
                          if self._bounds(building)[facade_index] == face
                          and self._bounds(building)[1 - axis] <= along <= self._bounds(building)[3 - axis]]
                roads = []
                for road in self.plan.roads:
                    road_bounds = self._bounds(road)
                    if (road_bounds[axis] <= min(face, outside) <= max(face, outside) <= road_bounds[axis + 2]
                            and road_bounds[1 - axis] <= along <= road_bounds[3 - axis]):
                        roads.append(road["name"])
                        if side == "west" and road["name"] == "main_street":
                            main_cells.add(along)
                cells[along] = {"pos": [horizontal, height, depth], "buildings": owners, "roads": roads}
            invalid = [cell for cell in cells.values()
                       if (not cell["roads"] and len(cell["buildings"]) != 1)
                       or (cell["roads"] and cell["buildings"])]
            self._check(f"plaza {side}: continuous immediate building frontage except declared road mouths",
                        any(cell["buildings"] for cell in cells.values()) and not invalid, invalid)
            openings = []
            for along, cell in cells.items():
                if cell["roads"]:
                    if not openings or along != openings[-1][-1] + 1:
                        openings.append([])
                    openings[-1].append(along)
            wide = [opening for opening in openings
                    if len(opening) > (5 if all(along in main_cells for along in opening) else 3)]
            self._check(f"plaza {side}: road mouths are at most five blocks for main street, three otherwise",
                        not wide, {"openings": openings, "too_wide": wide})
            positions = [[cell["pos"][0], height + offset, cell["pos"][2]]
                         for cell in cells.values() for offset in (-1, 0, 1)]
            blocks = self.client.request("blocks", {"positions": positions})["blocks"]
            if len(blocks) != len(positions):
                raise harness.ProtocolError("Plaza block query returned the wrong number of blocks")
            physical = []
            for index, cell in enumerate(cells.values()):
                support, feet, head = blocks[index * 3:index * 3 + 3]
                cell["blocks"] = [support, feet, head]
                if cell["roads"]:
                    valid = support in self.layout["road_surfaces"] and feet == head == "minecraft:air"
                else:
                    door = any(cell["pos"] in self.plan.layout["doors"].get(name, []) for name in cell["buildings"])
                    valid = all(block in walls or block.endswith(("_log", "_wood"))
                                or (door and block == "minecraft:oak_door") for block in (feet, head))
                if not valid:
                    physical.append(cell)
            self._check(f"plaza {side}: live walls enclose the plaza and road mouths remain paved and clear",
                        not physical, physical)
            evidence.append({"side": side, "cells": list(cells.values()), "openings": openings})
        self._write("plaza_frontages.json", evidence)

    def probe_npc_day(self) -> None:
        """Require every scheduled day-0 trip, deadline and home arrival, not just aggregate metrics."""
        start = harness.game_time(self.client.request("ping"))
        if not self._check("inspection begins before day-0 villager departures", 0 <= start < 1000, start):
            return
        self._step_to(DAY)
        records = self.client.request("npc_metrics", {"day": 0})["days"]
        metrics = next(record for record in records if record["day"] == 0)
        npcs = self.config["npcs"]
        identities = [npc["id"] for npc in npcs]
        self._check("40 distinct villagers: 30 adults and 10 children",
                    len(identities) == len(set(identities)) == 40
                    and Counter(npc["role"] for npc in npcs) == {"adult": 30, "child": 10})
        self._check("day-0 metrics cover each configured villager once",
                    Counter(row["npc"] for row in metrics["npcs"]) == Counter(identities))
        trips = metrics["trips"]
        failures = [trip for trip in trips if trip.get("npc") not in identities or trip.get("day") != 0
                    or trip.get("arriveTick") is None or not trip.get("path") or trip.get("outcome") != "arrived"]
        self._check("all day-0 trips arrived without failure, absence or missed market", bool(trips) and not failures,
                    failures)
        rows = {row["npc"]: row for row in metrics["npcs"]}
        states = {}
        for npc in npcs:
            identity = npc["id"]
            activity = "work" if npc["role"] == "adult" else "school"
            primary = [trip for trip in trips if trip["npc"] == identity and trip["activity"] == activity]
            market = [trip for trip in trips if trip["npc"] == identity and trip["activity"] == "market"]
            homes = [trip for trip in trips if trip["npc"] == identity and trip["activity"] == "home"
                     and trip["to"] == npc["home"] and trip.get("arriveTick") is not None]
            scheduled_market = npc["role"] == "adult" and 0 in npc.get("market_days", range(7))
            state = self.client.request("npc_state", {"id": identity})
            states[identity] = state
            row = rows.get(identity, {})
            valid = (bool(npc.get(activity)) and len(primary) == 1
                     and primary[0].get("from") == npc["home"] and primary[0].get("to") == npc[activity]
                     and row.get(activity + "Status") == "on_time" and bool(homes)
                     and len(market) == int(scheduled_market)
                     and (not scheduled_market or row.get("marketStatus") == "visited")
                     and state.get("id") == identity and state.get("activity") == "home"
                     and len(state.get("pos", [])) == 3
                     and all(math.isfinite(value) for value in state["pos"])
                     and math.dist(state["pos"], self._feet(npc["home"])) <= 1.5)
            self._check(f"{identity}: complete primary/market/home day-0 itinerary", valid,
                        {"metrics": row, "state": state})
        self._write("npc_day_0.json", {"metrics": metrics, "states": states, "inspection_start": start})

    def _interiors(self) -> dict[str, tuple]:
        """Find actual empty cells in the recorded functional room bounds, including both storerooms."""
        required = {"town_hall": ("great_hall", "anteroom", "office"), "inn": ("common", "storeroom"),
                    "bakery": ("shop", "storeroom")}
        preferred = {"town_hall/great_hall": self.layout["hall"]["feet"],
                     "town_hall/anteroom": self.layout["anteroom"]["feet"],
                     "town_hall/office": self.layout["office"]["feet"],
                     "inn/storeroom": self.layout["inn_storeroom"]["feet"],
                     "bakery/shop": self.layout["bakery"]["feet"]}
        targets = {}
        for building, rooms in required.items():
            for room in rooms:
                label = f"{building}/{room}"
                section = self.layout[building]["rooms"][room]
                left, north, right, south = self._bounds(section)
                height = self._feet(building)[1]
                candidates = [(horizontal + 0.5, height, depth + 0.5)
                              for horizontal in range(left + 1, right) for depth in range(north + 1, south)]
                candidates.sort(key=lambda point: (abs(point[0] - (left + right + 1) / 2)
                                                    + abs(point[2] - (north + south + 1) / 2), point))
                if label in preferred:
                    candidates.insert(0, self._center(preferred[label]))
                targets["interior/" + label] = next(point for point in candidates if self._walkable(point))
                self._check(label + " has an actual walkable interior endpoint", True)
        return targets

    def _dry_move(self, target) -> dict:
        """Use the exact supported grounded planner API, with no replacement graph or movement estimates."""
        return self.client.request("skill_start", {"name": VIEWER, "skill": "move_to", "dry_run": True,
                                                   "args": dict(zip(("x", "y", "z"), target))})

    def probe_routes(self) -> None:
        """Check every ordered pair and prove the short route crosses each important open doorway."""
        targets = {"place/" + name: self._feet(name) for name in self.places}
        targets.update(self._interiors())
        doors = self.layout["doors"]
        closed = [detail["pos"] for detail in self.layout.get("door_details", []) if detail.get("open") is False]
        doorway_results = []
        for building in ("town_hall", "inn", "bakery"):
            positions = doors[building]
            self._check(f"{building} has recorded functional doors", bool(positions))
            for index, position in enumerate(positions):
                coordinate = self._coords(position)
                opened = self._condition(f"block {coordinate} #minecraft:doors[open=true,half=lower]")
                if position in closed:  # a service door that starts closed (the hall's): a wall until opened
                    self._check(f"{building} door {index} starts closed", not opened)
                    continue
                self._check(f"{building} door {index} starts open", opened)
                if not opened:
                    continue
                north_south = any(self._condition(f"block {coordinate} #minecraft:doors[facing={facing}]")
                                  for facing in ("north", "south"))
                horizontal, height, depth = position
                offset = (0, 1) if north_south else (1, 0)
                sides = [(horizontal + 0.5 + sign * offset[0], height, depth + 0.5 + sign * offset[1])
                         for sign in (-1, 1)]
                for side, point in enumerate(sides):
                    targets[f"door/{building}/{index}/{side}"] = point
                for start, finish in (sides, list(reversed(sides))):
                    self._teleport(VIEWER, start)
                    result = self._dry_move(finish)
                    doorway_results.append({"building": building, "door": position, "from": start,
                                            "to": finish, "result": result})
                    self._check(f"{building} door {index} permits direct two-edge passage {start} -> {finish}",
                                result.get("feasible") is True and result.get("estimate", {}).get("ticks") == 20,
                                result)
        routes, failures = [], []
        for origin, start in targets.items():
            self._teleport(VIEWER, start)
            for destination, finish in targets.items():
                if destination == origin:
                    continue
                result = self._dry_move(finish)
                record = {"from": origin, "to": destination, "result": result}
                routes.append(record)
                if result.get("feasible") is not True:
                    failures.append(record)
        expected = len(targets) * (len(targets) - 1)
        self._check(f"all {expected} directed place/interior/door planner routes exist",
                    expected > 0 and len(routes) == expected and not failures, failures[:20])
        self._write("routes.json", {"targets": targets, "doors": doorway_results, "routes": routes,
                                    "failed_routes": len(failures)})

    def _seen(self, uuid: str | None = None) -> dict | None:
        """Read actual engine Sight admission, not a coordinate-only visibility approximation."""
        entities = self.client.request("observe", {"name": VIEWER, "include_terrain": False})["entities"]
        key, value = ("uuid", uuid) if uuid is not None else ("name", TARGET)
        return next((entity for entity in entities if entity.get(key) == value), None)

    def probe_window(self) -> None:
        """Require glass/opaque/glass visibility of the actual registered boat at the east mooring."""
        window = self.layout["neighbour_window"]
        position = window["pos"] if isinstance(window, dict) else window
        left, north, right, south = self._bounds(self.places["cottage"])
        if isinstance(window, dict) and "observer" in window:
            observer = self._center(window["observer"])
        else:
            horizontal, _, depth = position
            observer = self._center((horizontal + (1 if horizontal == left else -1 if horizontal == right else 0),
                                     self._feet("cottage")[1],
                                     depth + (1 if depth == north else -1 if depth == south else 0)))
        moor = self.layout["ferry"]["landings"]["east"]["moor"]
        distance = math.dist(observer, moor)
        self._check("cottage window is 6-10 blocks from the east mooring", 6 <= distance <= 10, distance)
        self._check("window observer is inside the cottage on a walkable cell",
                    left < observer[0] < right and north < observer[2] < south and self._walkable(observer), observer)
        self._teleport(VIEWER, observer)
        boat = self.layout["ferry"]["boat"]
        entities = self.client.request("dump_entities", {"region": self.layout["fixture_hash_region"]})["entities"]
        registered = [entity for entity in entities if entity.get("uuid") == boat]
        at_moor = (len(registered) == 1 and registered[0].get("type") == "minecraft:oak_boat"
                   and [float(value) for value in registered[0]["pos"].split(",")] == moor)
        self._check("registered ferry boat is at the east mooring", at_moor, registered)
        original, pending = {}, [tuple(position)]
        while pending:
            cell = pending.pop()
            if cell in original or not (left <= cell[0] <= right and north <= cell[2] <= south
                                        and observer[1] <= cell[1] <= observer[1] + 3):
                continue
            coordinate = self._coords(cell)
            if not self._condition(f"block {coordinate} minecraft:glass_pane"):
                continue
            properties = []
            for key in ("east", "north", "south", "west", "waterlogged"):
                value = self._condition(f"block {coordinate} minecraft:glass_pane[{key}=true]")
                properties.append(f"{key}={str(value).lower()}")
            original[cell] = "minecraft:glass_pane[" + ",".join(properties) + "]"
            pending.extend((cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                           for offset in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)))
        if not self._check("declared neighbour window contains actual glass panes", bool(original), position):
            return
        clear = self._seen(boat)
        opaque = None
        try:
            for cell in original:
                self._command(f"setblock {self._coords(cell)} minecraft:stone")
            opaque = self._seen(boat)
        finally:
            for cell, state in original.items():
                self._command(f"setblock {self._coords(cell)} {state}")
        restored = self._seen(boat)
        self._check("Sight sees the moored boat through glass, not opaque replacement, then glass again",
                    at_moor and clear is not None and opaque is None and restored is not None,
                    {"glass": clear, "opaque": opaque, "restored": restored})
        self._write("window.json", {"observer": observer, "moor": moor, "distance": distance,
                                    "glass_cells": [list(cell) for cell in original],
                                    "glass": clear, "opaque": opaque, "restored": restored})

    def _road_points(self) -> list[tuple]:
        """Enumerate existing road feet for reproducible daylight sightline controls."""
        height = self._feet("plaza")[1]
        points = set()
        for road in self.config["roads"]:
            left, north, right, south = self._bounds(road)
            points.update((horizontal + 0.5, height, depth + 0.5)
                          for horizontal in range(left, right + 1) for depth in range(north, south + 1))
        return sorted(points)

    def _lighting_samples(self) -> list[dict]:
        """Sample named lit areas and the bridge centre, excluding every lamp pool from dim controls."""
        height = self._feet("plaza")[1]
        samples = []
        left, north, right, south = self._bounds(self.places["plaza"])
        west, east = self._feet("west_bridge"), self._feet("east_bridge")
        lamps = [tuple(lamp["pos"] if isinstance(lamp, dict) else lamp) for lamp in self.layout["lamps"]]
        lit_areas = [("east_square", self._bounds(self.layout["east_square"]))]
        lit_areas.extend((road["name"], self._bounds(road)) for road in self.config["roads"]
                         if road["name"] in ("main_street", "west_quay"))
        road_points = set(self._road_points())
        first, top, last, bottom = self._bounds(self.layout["bridge"])
        road_points.update((horizontal + 0.5, height, depth + 0.5)
                           for horizontal in range(first, last + 1) for depth in range(top, bottom + 1))
        for lamp in lamps:
            self._check(f"recorded lamp {lamp} is an actual lantern",
                        self._condition(f"block {self._coords(lamp)} minecraft:lantern"))
            group = None
            landing = next((name for name in ("west_landing", "east_landing")
                            if math.dist((lamp[0], lamp[2]), (self._feet(name)[0], self._feet(name)[2])) <= 6), None)
            if west[0] - 3 <= lamp[0] <= east[0] + 3 and abs(lamp[2] + 0.5 - west[2]) <= 4:
                group = "bridge"
            elif left <= lamp[0] <= right and north <= lamp[2] <= south:
                group = "plaza"
            elif landing is not None:
                group = "landing"
            else:
                group = next((name for name, (area_left, area_north, area_right, area_south) in lit_areas
                              if area_left <= lamp[0] <= area_right and area_north <= lamp[2] <= area_south), None)
            if group is None:
                continue
            candidates = [(lamp[0] + 0.5 + offset_x, height, lamp[2] + 0.5 + offset_z)
                          for offset_x in (-1, 0, 1) for offset_z in (-1, 0, 1)
                          if offset_x or offset_z]
            candidates.sort(key=lambda point: (self._lamp_distance(point, lamp), point))
            feet = next((point for point in candidates if point in road_points
                         and self._lamp_distance(point, lamp) <= 5 and self._walkable(point)), None)
            self._check(f"{group} lamp {lamp} has a walkable sample in its five-block pool", feet is not None)
            if feet is not None:
                label = f"landing/{landing}" if group == "landing" else group
                samples.append({"label": f"{label}/lamp/{self._coords(lamp)}", "lit": True,
                                "feet": feet, "source_lamp": lamp})
        central = self._center(((west[0] + east[0]) / 2, height, (west[2] + east[2]) / 2))
        mid_lamp = min(lamps, key=lambda lamp: math.dist((lamp[0] + 0.5, lamp[2] + 0.5),
                                                        (central[0], central[2])))
        distance = self._lamp_distance(central, mid_lamp)
        self._check("central bridge lane is within five blocks of its mid-span lamp", distance <= 5,
                    {"feet": central, "lamp": mid_lamp, "distance": distance})
        self._check("central bridge lane has an actual walkable standing cell", self._walkable(central), central)
        samples.append({"label": "bridge/central_lane", "lit": True, "feet": central, "source_lamp": mid_lamp})
        for name in ("west_landing", "east_landing"):
            self._check(name + " has a representative recorded lamp pool",
                        any(sample["label"].startswith(f"landing/{name}/") for sample in samples))
        areas = [(name, {"bounds": bounds}) for name, bounds in self.layout["dim_areas"].items()]
        areas.extend((road["name"], road) for road in self.config["roads"]
                     if road.get("dim") and ("residential" in road["name"] or "homes" in road["name"]))
        groups = Counter()
        for name, section in areas:
            group = ("ferry" if "ferry" in name or "landing" in name else "back" if "back" in name
                     else "residential" if "residential" in name or "home" in name else "ford")
            first, top, last, bottom = self._bounds(section)
            for horizontal in range(first, last + 1, 4):
                for depth in range(top, bottom + 1, 4):
                    feet = (horizontal + 0.5, height, depth + 0.5)
                    if any(self._lamp_distance(feet, lamp) <= 5 for lamp in lamps):
                        continue
                    samples.append({"label": f"{group}/{name}/{horizontal}/{depth}", "lit": False, "feet": feet})
                    groups[group] += 1
        for group in ("ferry", "back", "residential"):
            self._check(f"{group} lanes have dark-area samples", groups[group] > 0)
        samples.extend({"label": name, "lit": False, "feet": self._feet(name)}
                       for name in ("west_ford", "east_ford"))
        samples.append({"label": "ford/midstream", "lit": False,
                        "feet": (0.5, height, self._feet("west_ford")[2])})
        return samples

    @staticmethod
    def _lamp_distance(feet, lamp) -> float:
        """Measure player-body centre to block centre, matching the night-rule geometry."""
        return math.dist((feet[0], feet[1] + 0.9, feet[2]), tuple(value + 0.5 for value in lamp))

    def probe_lighting(self) -> None:
        """Require a daylight sightline control for each >8-block night admission/rejection."""
        self._check("town enables engine night sight", self.client.request("config_get").get("night_sight") is True)
        roads = self._road_points()
        pairs = []
        covered = Counter()
        for sample in self._lighting_samples():
            target = sample["feet"]
            if not self._walkable(target):
                continue
            self._teleport(TARGET, target)
            candidates = [point for point in roads if 12 <= math.dist(point, target) <= 18]
            candidates.sort(key=lambda point: (math.dist(point, target), point))
            observer = None
            for point in candidates:
                if not self._walkable(point):
                    continue
                self._teleport(VIEWER, point)
                if self._seen() is not None:
                    observer = point
                    break
            self._check(sample["label"] + " has a daylight sightline beyond eight blocks", observer is not None)
            if observer is not None:
                pairs.append({**sample, "observer": observer})
                covered[sample["label"].split("/")[0]] += 1
        for group in ("plaza", "east_square", "main_street", "west_quay", "bridge", "landing", "ferry", "back",
                      "residential", "west_ford", "east_ford"):
            self._check(group + " has walkable daylight-controlled lighting coverage", covered[group] > 0)
        current = harness.game_time(self.client.request("ping"))
        if current % DAY >= NIGHT:
            raise ValueError("Lighting daylight controls must precede the night phase")
        self._step_to(current // DAY * DAY + NIGHT)
        self._check("night target has no inventory that could supply held light",
                    not self.client.request("bot_state", {"name": TARGET})["inventory"])
        for pair in pairs:
            self._teleport(TARGET, pair["feet"])
            self._teleport(VIEWER, pair["observer"])
            seen = self._seen()
            pair["night_entity"] = seen
            self._check(pair["label"] + (" is lit at night" if pair["lit"] else " remains dim at night"),
                        (seen is not None) == pair["lit"], pair)
        self._write("lighting.json", pairs)

    def _inspect(self, fixture: Path, number: int) -> dict:
        """Hash the complete fresh clone first; run all mutating probes only on inspection one."""
        directory = self.directory / f"inspection_{number}"
        harness.clone_apfs(fixture, directory)
        self.artifacts[f"inspection_{number}"] = str(directory)
        with harness.Server(directory, **{**self.options, "base_seed": self.config["seed"]}) as server:
            self.client = server.client
            harness.enable_lockstep(self.client, recording=False)
            self.client.request("restore_bots")
            result = self.client.request("hash", {"region": self.layout["fixture_hash_region"]})
            initial = harness.hash_components(result, harness.game_time(result))
            self._write(f"live_hash_{number}.json", initial)
            self._phase(f"inspection {number} fixture-start clock", lambda: self.probe_fixture_clock(number))
            if number == 1:
                self._phase("building news landmarks", self.probe_landmarks)
                self._phase("physical plaza enclosure", self.probe_plaza)
                self._phase("full villager day 0", self.probe_npc_day)
                for name in (VIEWER, TARGET):
                    self.client.request("spawn_bot", {"name": name, "yaw": 0, "pitch": 0,
                                                       **dict(zip(("x", "y", "z"), self._feet("plaza")))})
                    self._check(name + " starts with empty inventory, not a held light",
                                not self.client.request("bot_state", {"name": name})["inventory"])
                self._phase("all planner routes and interiors", self.probe_routes)
                self._phase("glass sightline to mooring", self.probe_window)
                self._phase("street lighting", self.probe_lighting)
                self.client.request("save_snapshot")
        self.client = None
        return initial

    def run(self) -> int:
        """Execute only on request and return failure for any missing or negative evidence."""
        try:
            fixtures = self._builds()
            hashes = [self._inspect(fixture, number) for number, fixture in enumerate(fixtures, 1)]
            self._check("fresh live full-volume hashes and game times match before probe mutations",
                        hashes[0] == hashes[1], hashes)
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            self._check("runtime/build completed", False, str(error))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        if self.created:
            self._write("report.json", {"passed": passed, "checks": self.checks, "limits": LIMITS,
                                        "artifacts": self.artifacts})
        print(f"\n{'PASS' if passed else 'FAIL'} IT-24: {sum(check['passed'] for check in self.checks)}"
              f"/{len(self.checks)} checks", flush=True)
        for name, path in self.artifacts.items():
            print(f"  {name}: {path}")
        for limit in LIMITS:
            print("LIMIT: " + limit)
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse the shared runtime CLI; imports and --help never start a server."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--scratch", type=Path, help="New artifact root; defaults to <runs-dir>/it24_town")
    args = parser.parse_args(argv)
    try:
        return IT24TownExperiment(args).run()
    except (OSError, ValueError) as error:
        print(f"FAIL setup: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
