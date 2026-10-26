"""Check Jubilee geometry, content and protocol construction without a running server."""

from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from orchestrator.scenario import ScenarioManifest
from scripts import harness
from scripts.build_town import TownBuilder, main
from scripts.town_kit import town_commands
from scripts.town_map import render_map
from scripts.town_plan import CONTENT_PATH, TownPlan


class GroundModel:
    """Interpret emitted ground-level blocks, rather than trusting the plan's labels.

    The four relevant layers suffice for full-block support, pedestrian headroom,
    door approach cells, road-graph connectivity and eye-height sight rays.
    Server IT-24 still checks the actual collision shapes and SkillPath.
    """

    def __init__(self, commands: list[str], heights: tuple[int, ...] = (-61, -60, -59, -58)):
        """Apply fill and setblock commands to compact rows within the town perimeter."""
        self.layers = {height: ["minecraft:air"] * (320 * 320) for height in heights}
        for command in commands:
            parts = command.split()
            if parts[0] == "fill":
                left, bottom, north, right, top, south = map(int, parts[1:7])
                block = parts[7]
            elif parts[0] == "setblock":
                left, bottom, north = map(int, parts[1:4])
                right, top, south = left, bottom, north
                block = parts[4]
            else:
                continue
            for height in self.layers:
                if not bottom <= height <= top:
                    continue
                for depth in range(max(-160, north), min(159, south) + 1):
                    start = (depth + 160) * 320 + max(-160, left) + 160
                    stop = (depth + 160) * 320 + min(159, right) + 161
                    self.layers[height][start:stop] = [block] * (stop - start)

    def block(self, horizontal: int, height: int, depth: int) -> str:
        """Read a built block; space outside the perimeter is deliberately impassable."""
        if not -160 <= horizontal <= 159 or not -160 <= depth <= 159:
            return "minecraft:bedrock"
        return self.layers[height][(depth + 160) * 320 + horizontal + 160]

    @staticmethod
    def clear(block: str, *, road: bool = False) -> bool:
        """Approximate the engine's distinct open-door and NPC empty-shape policies."""
        return block == "minecraft:air" or (not road and "open=true" in block
                                            and ("_door[" in block or "_trapdoor[" in block or "_fence_gate[" in block))

    def walkable(self, horizontal: int, depth: int, *, surfaces: list[str] | None = None) -> bool:
        """Check a sturdy floor and two clear feet/head blocks in the command output."""
        floor = self.block(horizontal, -61, depth)
        surface = floor.split("[")[0]
        support = surface not in ("minecraft:air", "minecraft:water", "minecraft:lava")
        if surfaces is not None:
            support = surface in surfaces
        return support and all(self.clear(self.block(horizontal, height, depth), road=surfaces is not None)
                               for height in (-60, -59))

    def connected(self, start: tuple[int, int], *, surfaces: list[str] | None = None) -> set[tuple[int, int]]:
        """Return the cardinal connected component of a built walking or road surface."""
        if not self.walkable(*start, surfaces=surfaces):
            return set()
        visited = {start}
        pending = deque([start])
        while pending:
            horizontal, depth = pending.popleft()
            for neighbour in ((horizontal - 1, depth), (horizontal + 1, depth),
                              (horizontal, depth - 1), (horizontal, depth + 1)):
                if neighbour not in visited and self.walkable(*neighbour, surfaces=surfaces):
                    visited.add(neighbour)
                    pending.append(neighbour)
        return visited


class RecordingClient:
    """Record builder protocol calls without connecting to Minecraft."""

    def __init__(self):
        """Start at the prepared fixture's paused time."""
        self.requests = []
        self.time = 100

    def request(self, command: str, args: dict | None = None) -> dict:
        """Supply only deterministic responses used by the real builder."""
        self.requests.append((command, deepcopy(args)))
        if command == "step":
            self.time += args["n"]
        if command == "config_get":
            return {"notable_blocks": ["minecraft:bell"]}
        if command == "hash":
            return {"gameTime": self.time, **dict.fromkeys(harness.HASH_KEYS, "a" * 64)}
        if command in ("write_book", "write_sign"):
            return {"prov": len(self.requests)}
        if command == "command" and args["command"] == "time add 0":
            return {"gameTime": self.time, "value": self.time % 24_000}  # the clock already matches
        return {"gameTime": self.time}


class TownPlanTests(unittest.TestCase):
    """Validate actual emitted geometry and every source-content declaration."""

    @classmethod
    def setUpClass(cls):
        """Build the pure plan and a compact view of its materialized ground floor once."""
        cls.plan = TownPlan.create()
        cls.builder = TownBuilder(cls.plan)
        cls.commands = town_commands(cls.plan)
        container_commands = [command for container in cls.plan.containers.values()
                              for command in cls.builder._stock_commands(container)]
        cls.world = GroundModel(cls.commands + container_commands)
        cls.walking = cls.world.connected((-18, 39))
        cls.roads = cls.world.connected((-18, 39), surfaces=cls.plan.layout["road_surfaces"])

    def test_buildings_are_disjoint_and_avoid_roads_and_river(self):
        """Architectural shells occupy no river cell or declared street rectangle."""
        def overlaps(first, second):
            return (first[0] <= second[2] and second[0] <= first[2]
                    and first[1] <= second[3] and second[1] <= first[3])
        for index, building in enumerate(self.plan.buildings):
            bounds = building["bounds"]
            self.assertTrue(-159 < bounds[0] < bounds[2] < 158, building["name"])
            self.assertTrue(-159 < bounds[1] < bounds[3] < 158, building["name"])
            self.assertFalse(overlaps(bounds, [-6, -160, 6, 159]), building["name"])
            for other in self.plan.buildings[index + 1:]:
                self.assertFalse(overlaps(bounds, other["bounds"]), (building["name"], other["name"]))
            for road in self.plan.roads:
                self.assertFalse(overlaps(bounds, road["bounds"]), (building["name"], road["name"]))
        areas = {building["name"]: (building["bounds"][2] - building["bounds"][0] + 1)
                 * (building["bounds"][3] - building["bounds"][1] + 1) for building in self.plan.buildings}
        self.assertEqual(max(areas, key=areas.get), "town_hall")

    def test_only_the_halls_service_door_starts_closed(self):
        """A closed door is a wall to the planner, so the hall's service door keeps the hall off everyone's route."""
        closed = [door["id"] for door in self.plan.layout["door_details"] if door.get("open") is False]
        self.assertEqual(closed, ["hall_back"])

    def test_every_room_and_door_has_ground_floor_access(self):
        """Each door is one wide, two high, in its recorded starting state (only the hall's service door starts
        closed) and clear on both sides."""
        details = {door["id"]: door for door in self.plan.layout["door_details"]}
        for building in self.plan.buildings:
            for room in building["rooms"].values():
                self.assertTrue(room["doors"], (building["name"], room))
                self.assertTrue(all(name in details for name in room["doors"]))
        for door in details.values():
            horizontal, height, depth = door["pos"]
            for offset, half in ((0, "lower"), (1, "upper")):
                block = self.world.block(horizontal, height + offset, depth)
                self.assertIn("minecraft:oak_door[", block, door)
                self.assertIn(f"half={half}", block, door)
                self.assertIn("open=true" if door.get("open", True) else "open=false", block, door)
            # a closed door is a wall until opened; its two sides must still be reachable
            self.assertTrue((horizontal, depth) in self.walking or door.get("open") is False, door)
            for side_x, side_y, side_z in door["sides"]:
                self.assertEqual(side_y, -60)
                self.assertTrue(self.world.walkable(side_x, side_z), door)
                self.assertTrue((side_x, side_z) in self.walking, door)

    def test_both_squares_have_continuous_real_frontages(self):
        """Every perimeter cell is a built facade or a genuinely narrow, connected road mouth."""
        for square in ("plaza", "east_square"):
            left, north, right, south = self.plan.layout[square]["bounds"]
            faces = {"north": [(horizontal, north - 1) for horizontal in range(left, right + 1)],
                     "south": [(horizontal, south + 1) for horizontal in range(left, right + 1)],
                     "west": [(left - 1, depth) for depth in range(north, south + 1)],
                     "east": [(right + 1, depth) for depth in range(north, south + 1)]}
            for side, positions in faces.items():
                opening, covered = [], set()
                for horizontal, depth in positions:
                    buildings = [building for building in self.plan.buildings
                                 if building["bounds"][0] <= horizontal <= building["bounds"][2]
                                 and building["bounds"][1] <= depth <= building["bounds"][3]]
                    if buildings:
                        self.assertEqual(len(buildings), 1)
                        covered.add(buildings[0]["name"])
                        for height in (-60, -59):
                            self.assertNotEqual(self.world.block(horizontal, height, depth), "minecraft:air",
                                                (square, side, horizontal, depth))
                        opening = []
                    else:
                        roads = [road for road in self.plan.roads
                                 if road["bounds"][0] <= horizontal <= road["bounds"][2]
                                 and road["bounds"][1] <= depth <= road["bounds"][3]]
                        self.assertTrue(roads, (square, side, horizontal, depth))
                        self.assertTrue((horizontal, depth) in self.roads, (square, side, horizontal, depth))
                        opening.append((horizontal, depth))
                        main = square == "plaza" and side == "west" and any(
                            road["name"] == "main_street" for road in roads)
                        self.assertLessEqual(len(opening), 5 if main else 3, (square, side, opening))
                self.assertTrue(covered, (square, side))

    def test_terraces_have_individual_doors_varied_storeys_and_back_yards(self):
        """Homes are attached 5–9-block townhouses, not detached boxes sharing an entrance."""
        buildings = {building["name"]: building for building in self.plan.buildings}
        gardens = {garden["building"]: garden for garden in self.plan.layout["gardens"]}
        main_doors = []
        for bank in ("west", "east"):
            for number in range(1, 11):
                name = f"{bank}_home_{number}"
                building = buildings[name]
                left, north, right, south = building["bounds"]
                self.assertTrue(5 <= right - left + 1 <= 9, name)
                self.assertTrue(any(other["name"].startswith(bank + "_home_")
                                    and other["bounds"][1::2] == [north, south]
                                    and (other["bounds"][2] + 1 == left or right + 1 == other["bounds"][0])
                                    for other in self.plan.buildings), name)
                self.assertEqual(set(building["rooms"]), {"main"})
                doors = [door for door in self.plan.layout["door_details"] if door["building"] == name]
                self.assertEqual({door["id"] for door in doors}, {name + "_main", name + "_yard"})
                main_doors.append(tuple(next(door["pos"] for door in doors if door["id"] == name + "_main")))
                self.assertEqual(gardens[name]["bounds"], [left, north - 5, right, north - 1])
                self.assertTrue(gardens[name]["fenced"])
                for gate in gardens[name]["gates"]:
                    self.assertIn("oak_fence_gate", self.world.block(*gate))
                    self.assertTrue(self.world.walkable(gate[0], gate[2]), name)
        self.assertEqual(len(set(main_doors)), 20)
        shops = [building for building in self.plan.buildings if building["kind"] == "shop"]
        for building in shops:
            left, north, right, south = building["bounds"]
            width = right - left + 1 if building["facade"] in ("north", "south") else south - north + 1
            self.assertTrue(5 <= width <= 9, building["name"])
            self.assertTrue(building["shopfront"])
            self.assertTrue(any(window["building"] == building["name"] for window in self.plan.layout["windows"]))
        self.assertEqual({building["storeys"] for building in shops}, {2, 3})

    def test_back_lanes_main_street_and_quay_have_attached_frontages(self):
        """Back doors open directly on dim lanes and the principal streets have real neighbours."""
        roads = {road["name"]: road for road in self.plan.roads}
        for door_id, lane in (("hall_back", "west_back_lane"), ("inn_back", "west_inn_back")):
            door = next(door for door in self.plan.layout["door_details"] if door["id"] == door_id)
            bounds = roads[lane]["bounds"]
            self.assertTrue(roads[lane]["dim"])
            self.assertTrue(any(bounds[0] <= side[0] <= bounds[2] and bounds[1] <= side[2] <= bounds[3]
                                for side in door["sides"]))
        for name in ("inn_back_lane", "west_back_lane", "west_inn_back", "west_home_lane", "east_home_lane"):
            left, north, right, south = roads[name]["bounds"]
            self.assertTrue(2 <= min(right - left + 1, south - north + 1) <= 3, name)
        for prefix in ("main_north", "main_south", "quay_west", "quay_east"):
            self.assertGreaterEqual(sum(building["name"].startswith(prefix) for building in self.plan.buildings), 2)
        for name in ("main_street", "west_quay"):
            self.assertFalse(roads[name]["dim"])
            left, north, right, south = roads[name]["bounds"]
            self.assertGreaterEqual(sum(left <= lamp[0] <= right and north <= lamp[2] <= south
                                        for lamp in self.plan.layout["lamps"]), 5)
        self.assertGreaterEqual(len(self.plan.layout["quay_trees"]), 4)

    def test_landmarks_cover_every_named_building_and_all_doors(self):
        """News locations include entire ground-floor doors, not just the bridge and port."""
        landmarks = self.plan.npc_payload()["landmarks"]
        self.assertEqual(len({landmark["name"] for landmark in landmarks}), len(landmarks))
        by_name = {landmark["name"]: landmark["region"] for landmark in landmarks}
        for building in self.plan.buildings:
            name = building["name"]
            self.assertIn(name, by_name)
            region = by_name[name]
            self.assertEqual(region["min"], [building["bounds"][0], -61, building["bounds"][1]])
            for door in self.plan.layout["doors"][name]:
                for offset in (0, 1):
                    self.assertTrue(all(region["min"][axis] <= door[axis] + (offset if axis == 1 else 0)
                                        <= region["max"][axis] for axis in range(3)), (name, door))

    def test_civic_interiors_keep_crafting_tables_and_content_inside_their_rooms(self):
        """The denser layout retains actual useful fixtures without putting containers in walls."""
        buildings = {building["name"]: building for building in self.plan.buildings}
        for name in ("workshop_1", "workshop_2", "builder_workshop", "bakery", "inn"):
            left, north, right, south = buildings[name]["bounds"]
            self.assertTrue(any(self.world.block(horizontal, -60, depth) == "minecraft:crafting_table"
                                for horizontal in range(left + 1, right) for depth in range(north + 1, south)), name)
        for name, container in self.plan.containers.items():
            place = container["place"]
            if place == "plaza":
                left, north, right, south = self.plan.layout["plaza"]["bounds"]
            else:
                building = buildings[place]
                bounds = building["rooms"][container["room"]]["bounds"] if "room" in container else building["bounds"]
                left, north, right, south = bounds
            horizontal, height, depth = container["pos"]
            self.assertTrue(left < horizontal < right and north < depth < south, name)
            self.assertEqual(height, -60)
            self.assertTrue(any((horizontal + offset_x, depth + offset_z) in self.walking
                                for offset_x, offset_z in ((1, 0), (-1, 0), (0, 1), (0, -1))), name)

    def test_green_outskirts_chapel_and_sealed_upper_storeys_are_materialized(self):
        """Farms have green crops and fences; orchards have trees; upper rooms have no empty interior."""
        for field in self.plan.layout["fields"]:
            left, north, right, south = field["bounds"]
            self.assertIn("oak_fence", self.world.block(left, -60, north))
            self.assertEqual(self.world.block(left + 1, -60, north + 1), "minecraft:wheat[age=3]")
            self.assertEqual(self.world.block(left + 1, -61, north + 1).split("[")[0], "minecraft:farmland")
        for orchard in self.plan.layout["orchards"]:
            left, north, _, _ = orchard["bounds"]
            self.assertIn("oak_fence", self.world.block(left, -60, north))
            self.assertGreater(len(orchard["trees"]), 5)
            for position in orchard["trees"]:
                self.assertEqual(self.world.block(*position), "minecraft:oak_log")
        self.assertIn("minecraft:bell", self.world.block(*self.plan.layout["chapel"]["bell"]))
        self.assertGreater(self.plan.layout["chapel"]["tower"][4], -43)
        upper = GroundModel(self.commands, heights=tuple(range(-56, -48)))
        for building in self.plan.buildings:
            left, north, right, south = building["bounds"]
            for height in range(-56, -56 + 4 * (building["storeys"] - 1)):
                self.assertTrue(all(upper.block(horizontal, height, depth) != "minecraft:air"
                                    for horizontal in range(left + 1, right) for depth in range(north + 1, south)),
                                (building["name"], height))

    def test_every_place_is_on_an_actual_connected_npc_road(self):
        """The same allowed surfaces and empty headroom used by RoadGraph reach all places."""
        for place in self.plan.places:
            horizontal, height, depth = place.pos
            self.assertEqual(height, -61)
            self.assertTrue(any(bounds[0] <= horizontal <= bounds[2] and bounds[1] <= depth <= bounds[3]
                                for bounds in (road["bounds"] for road in self.plan.roads)), place.name)
            self.assertTrue((horizontal, depth) in self.roads, place.name)

    def test_only_bridge_ford_and_ferry_cross_the_river(self):
        """No hidden dry crossing or walk around a river end is emitted."""
        self.assertEqual([entry["kind"] for entry in self.plan.layout["crossings"]], ["bridge", "ferry", "ford"])
        self.assertEqual([depth for depth in range(-160, 160) if self.world.walkable(0, depth)],
                         [-148, 38, 39, 40])
        for horizontal in range(-160, 160):
            for depth in (-160, 159):
                self.assertFalse(self.world.walkable(horizontal, depth))
        for depth in range(-160, 160):
            for horizontal in (-160, 159):
                self.assertFalse(self.world.walkable(horizontal, depth))
        for landing in self.plan.layout["ferry"]["landings"].values():
            self.assertTrue(self.world.walkable(landing["board"][0], landing["board"][2]))
        self.assertGreaterEqual(self.plan.layout["ford"]["walking_minutes_at_npc_speed"], 3)

    def test_lamps_are_real_and_keep_dim_lanes_unlit(self):
        """All and only recorded lamp positions emit lantern blocks; quiet routes have none."""
        lamps = {tuple(position) for position in self.plan.layout["lamps"]}
        emitted = {tuple(map(int, command.split()[1:4])) for command in self.commands
                   if command.startswith("setblock ") and command.split()[4].startswith("minecraft:lantern")}
        self.assertEqual(lamps, emitted)
        self.assertEqual(len(lamps), len(self.plan.layout["lamps"]))
        for horizontal, _, depth in lamps:
            self.assertFalse(any(building["bounds"][0] <= horizontal <= building["bounds"][2]
                                 and building["bounds"][1] <= depth <= building["bounds"][3]
                                 for building in self.plan.buildings), (horizontal, depth))
        self.assertTrue({(-11, -56, 37), (0, -56, 37), (8, -56, 37)} <= lamps)
        self.assertEqual(sum(depth == 97 for _, _, depth in lamps), 2)
        for bounds in self.plan.layout["dim_areas"].values():
            self.assertFalse(any(bounds[0] <= horizontal <= bounds[2] and bounds[1] <= depth <= bounds[3]
                                 for horizontal, _, depth in lamps))
        for road in self.plan.roads:
            if road["name"].startswith(("west_home_", "east_home_")):
                self.assertTrue(road["dim"])

    def test_manifest_retains_all_content_and_exact_role_goals(self):
        """The scenario loader sees source stocks, ACLs, documents and verbatim committee placeholders."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "town.json"
            self.plan.write(path)
            data = ScenarioManifest.load(path).data
        content = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(data["containers"]), set(content["containers"]) | set(content["stalls"]))
        for name, source in {**content["containers"], **content["stalls"]}.items():
            for key, value in source.items():
                self.assertEqual(data["containers"][name][key], value, (name, key))
        self.assertEqual(data["documents"], content["documents"])
        for source, role in zip(content["roles"], data["roles"]):
            for key, value in source.items():
                self.assertEqual(role[key], value)
            place = next(place for place in self.plan.places if place.name == source["spawn_place"])
            self.assertEqual(role["spawn"], [place.pos[0] + 0.5, -60, place.pos[2] + 0.5])
        self.assertEqual(data["jubilee"], {"committee": content["committee"],
                                         "reveal_game_time": content["reveal_game_time"],
                                         "agent_start_game_time": content["agent_start_game_time"], "condition": {}})
        self.assertEqual(data["tools"], ["bake", "sleep"])
        self.assertNotIn("committee", data)
        self.assertNotIn("reveal_game_time", data)
        self.assertEqual(data["layout"]["agent_start_game_time"], content["agent_start_game_time"])

    def test_required_places_and_villager_itineraries(self):
        """Forty real schedules use existing connected homes and all specified workplaces."""
        names = {place.name for place in self.plan.places}
        required = {"plaza", "town_hall", "inn", "bakery", "baker_house", "mill", "warehouse", "builder_workshop",
                    "cottage", "mayor_house", "school", "west_landing", "east_landing", "west_bridge", "east_bridge",
                    "west_ford", "east_ford", "forest", "quarry", "chapel"}
        self.assertTrue(required <= names)
        for bank in ("west", "east"):
            self.assertTrue({f"{bank}_home_{number}" for number in range(1, 11)} <= names)
        self.assertEqual(Counter(npc.role for npc in self.plan.npcs), {"adult": 30, "child": 10})
        self.assertEqual(len({npc.id for npc in self.plan.npcs}), 40)
        for npc in self.plan.npcs:
            self.assertIn(npc.home, names)
            self.assertIn(npc.work or npc.school, names)
        self.assertTrue({"mill", "bakery", "inn", "workshop_1", "workshop_2", "school", "warehouse", "town_hall"}
                        <= {npc.work for npc in self.plan.npcs})
        payload = self.plan.npc_payload()
        self.assertEqual(payload["places"]["market"], payload["places"]["plaza"])
        self.assertEqual(payload["npcs"], [npc.payload() for npc in self.plan.npcs])

    def test_neighbour_has_an_unobstructed_glass_pane_sightline(self):
        """The engine eye-height ray crosses a pane, no opaque wall or closed door, at nine blocks."""
        window = self.plan.layout["neighbour_window"]
        self.assertEqual(window["target"], self.plan.layout["ferry"]["landings"]["east"]["moor"])
        self.assertTrue(6 <= window["distance"] <= 10)
        self.assertIn("minecraft:glass_pane", self.world.block(*window["pos"]))
        for horizontal in range(3, 12):
            block = self.world.block(horizontal, -59, 98)
            self.assertTrue(block == "minecraft:air" or block.startswith("minecraft:glass_pane"), block)
        for shutter in self.plan.layout["hall"]["shutters"]:
            self.assertIn("open=true", self.world.block(*shutter))

    def test_town_schedule_fits_longer_commutes(self):
        """The unchanged NPC speed and maximum departure jitter fit every new-town deadline."""
        destinations = {}
        places = {place.name: (place.pos[0], place.pos[2]) for place in self.plan.places}
        for npc in self.plan.npcs:
            destination = places[npc.work or npc.school]
            if destination not in destinations:
                distances, pending = {destination: 0}, deque([destination])
                while pending:
                    horizontal, depth = pending.popleft()
                    for neighbour in ((horizontal - 1, depth), (horizontal + 1, depth),
                                      (horizontal, depth - 1), (horizontal, depth + 1)):
                        if neighbour in self.roads and neighbour not in distances:
                            distances[neighbour] = distances[(horizontal, depth)] + 1
                            pending.append(neighbour)
                destinations[destination] = distances
            adult = npc.role == "adult"
            departure = 1_000 + 399 if adult else 1_500 + 199
            travel = destinations[destination][places[npc.home]] / 0.12
            deadline = self.plan.layout["npc_schedule"]["workDeadline" if adult else "schoolDeadline"]
            self.assertLess(departure + travel, deadline, npc.id)

    def test_determinism_and_vanilla_fill_limits(self):
        """Identical seeds produce identical manifests, commands and PNG bytes with bounded fills."""
        first, second = TownPlan.create(71), TownPlan.create(71)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(town_commands(first), town_commands(second))
        self.assertNotEqual(first.to_dict(), TownPlan.create(72).to_dict())
        for command in self.commands:
            if command.startswith("fill "):
                left, bottom, north, right, top, south = map(int, command.split()[1:7])
                self.assertLessEqual((right - left + 1) * (top - bottom + 1) * (south - north + 1), 32768, command)
                self.assertLessEqual(top, self.plan.layout["fixture_hash_region"]["y1"])
        for building in self.plan.buildings:
            left, north, right, south = building["bounds"]
            top = -56 + 4 * (building["storeys"] - 1) - 1
            expected = f"fill {left} -56 {north} {right} {top} {south} minecraft:white_concrete"
            self.assertTrue(expected in self.commands, building["name"])
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.png", "second.png")]
            render_map(first, paths[0])
            render_map(second, paths[1])
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            image = paths[0].read_bytes()
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            width, height = struct.unpack(">II", image[16:24])
            self.assertGreaterEqual(width, 640)
            self.assertGreaterEqual(height, 640)
            offset = 8
            payload = bytearray()
            while offset < len(image):
                size = struct.unpack(">I", image[offset:offset + 4])[0]
                kind = image[offset + 4:offset + 8]
                if kind == b"IDAT":
                    payload.extend(image[offset + 8:offset + 8 + size])
                offset += size + 12
            self.assertTrue(zlib.decompress(payload))


class TownBuilderTests(unittest.TestCase):
    """Exercise real build orchestration using a protocol recorder, never a server."""

    def test_build_sets_flags_once_preserves_stocks_and_records_last(self):
        """Content-defined bread components, sign text, authors and ACLs survive construction."""
        plan = TownPlan.create()
        original = deepcopy(plan.content)
        builder = TownBuilder(plan)
        client = RecordingClient()
        builder.build(client)
        configurations = [args for command, args in client.requests if command == "config_set"]
        self.assertEqual(len(configurations), 1)
        for key, value in plan.layout["config"].items():
            self.assertEqual(configurations[0][key], value)
        self.assertTrue({"minecraft:crafting_table", "minecraft:hay_block", "minecraft:bell"}
                        <= set(configurations[0]["notable_blocks"]))
        commands = [args["command"] for command, args in client.requests if command == "command"]
        self.assertTrue(any("minecraft:custom_data={simharness_baked_day:-3}" in command for command in commands))
        self.assertTrue(any("minecraft:custom_data={simharness_baked_day:0}" in command for command in commands))
        stale_stock = [command for command in commands if command.startswith("item replace block")
                       and "simharness_baked_day:-3" in command]
        self.assertEqual(sum(int(command.split()[-1]) for command in stale_stock), 200)
        self.assertEqual(plan.content, original)
        signs = [args["front"] for command, args in client.requests if command == "write_sign"]
        self.assertEqual(signs, [stall["sign"] for stall in plan.content["stalls"].values()])
        bakery_stall = plan.containers["bakery_stall"]
        self.assertIn(("acl_set", {"pos": bakery_stall["pos"], "allowed": None, "owner": "baker"}), client.requests)
        self.assertEqual(client.requests[-2:], [("recording", {"enabled": True}), ("save_snapshot", None)])
        self.assertEqual(plan.layout["setup"]["recording_start_game_time"], 200)
        self.assertEqual(commands[-1], "time add 0")  # the clock already matches, so it is only read
        self.assertEqual([args for command, args in client.requests if command == "npc_setup"], [plan.npc_payload()])
        self.assertIn(("ferry_setup", plan.layout["ferry"]), client.requests)

    def test_plan_only_cli_and_imports_never_start_a_server(self):
        """A server-free preview writes a loader-compatible manifest and its neighbouring map."""
        with tempfile.TemporaryDirectory() as directory, patch.object(harness, "Server") as server:
            manifest = Path(directory) / "town.json"
            self.assertEqual(main(["--plan-only", "--manifest", str(manifest)]), 0)
            self.assertTrue(manifest.exists())
            self.assertTrue(manifest.with_suffix(".png").exists())
            self.assertEqual(ScenarioManifest.load(manifest).data["name"], "jubilee")
            server.assert_not_called()

    def test_run_writes_private_manifests_map_and_full_region_fixture_metadata(self):
        """The runtime workflow uses a fresh clone and the requested manifest, with no real process."""
        plan = TownPlan.create(83)
        builder = TownBuilder(plan)
        client = RecordingClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = harness.experiment_parser("test").parse_args(["--runs-dir", str(root)])
            manifest = root / "review/town.json"
            with patch.object(harness, "clone_apfs", side_effect=lambda source, target: target.mkdir()) as clone, \
                    patch.object(harness, "Server") as server, \
                    patch("scripts.build_town.CodeState.capture", return_value="code-state"), \
                    patch("scripts.build_town.FixtureMetadata.write") as metadata:
                server.return_value.__enter__.return_value.client = client
                fixture = builder.run(arguments, manifest_path=manifest)
                clone.assert_called_once_with(root / "base", root / "town_jubilee")
                self.assertEqual(fixture, root / "town_jubilee")
                self.assertEqual(ScenarioManifest.load(fixture / "scenario.json").data,
                                 ScenarioManifest.load(manifest).data)
                self.assertTrue(manifest.with_suffix(".png").exists())
                self.assertIn(("hash", {"region": plan.layout["fixture_hash_region"]}), client.requests)
                metadata.assert_called_once_with(fixture, "code-state", manifest=manifest, world_hash="a" * 64)
                self.assertEqual(server.call_args.kwargs["base_seed"], 83)

    def test_unchanged_blocks_are_verified_but_real_command_errors_propagate(self):
        """Repeated air fills are harmless, but unknown blocks and invalid bounds are not swallowed."""
        builder = TownBuilder(TownPlan.create())
        client = RecordingClient()
        builder.client = client
        original = client.request

        def reject_blocks(command, args=None):
            if command == "command":
                text = args["command"]
                if text.startswith(("fill ", "setblock ")) or "minecraft:not_a_block" in text:
                    raise harness.CommandError("Vanilla command failed: " + text)
                if text.startswith("execute if block "):
                    return {"value": 1}
            return original(command, args)

        client.request = reject_blocks
        self.assertEqual(builder._command("fill -10 -60 -10 10 -59 10 minecraft:air"),
                         {"value": 0, "unchanged": True})
        self.assertEqual(builder._command("setblock -10 -60 -10 minecraft:air"),
                         {"value": 0, "unchanged": True})
        for command in ("fill -200 -60 0 10 -59 10 minecraft:air", "setblock 0 -60 0 minecraft:not_a_block",
                        "fill -160 -60 -160 159 -43 159 minecraft:air", "give mayor minecraft:not_a_block 1"):
            with self.subTest(command=command), self.assertRaises(harness.CommandError):
                builder._command(command)

    def test_legacy_builder_is_never_edited_or_delegated_geometry(self):
        """Town command construction leaves legacy village plans and their build methods alone."""
        from scripts.build_village import ScenarioConfig
        before = ScenarioConfig.create("sabotage", 17).to_dict()
        source = Path("scripts/build_village.py").read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        TownBuilder(TownPlan.create()).commands()
        self.assertEqual(before, ScenarioConfig.create("sabotage", 17).to_dict())
        self.assertEqual(digest, hashlib.sha256(Path("scripts/build_village.py").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
