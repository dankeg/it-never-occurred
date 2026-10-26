"""Describe the Jubilee town without connecting to a server or changing legacy fixtures."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

if __package__:
    from .build_village import NpcSpec, Place, Role, SCHEDULE
    from .setup_server import BASE_SEED, REPO_ROOT
else:
    from build_village import NpcSpec, Place, Role, SCHEDULE
    from setup_server import BASE_SEED, REPO_ROOT


CONTENT_PATH = REPO_ROOT / "scenario/jubilee.json"
SURFACE_Y = -61
TOWN_SCHEDULE = {**SCHEDULE, "workDeadline": 5_000, "schoolDeadline": 5_200, "schoolAbsent": 6_500}


@dataclass
class TownPlan:
    """Own the geometry and content-derived manifest of one deterministic town.

    Place positions name exterior road surfaces, as required by RoadGraph; feet
    and role spawns are one block above them. Functional rooms are ground-floor
    interiors, independent of the sealed decorative upper storeys.
    """

    seed: int
    content: dict
    places: list[Place]
    roles: list[Role]
    npcs: list[NpcSpec]
    roads: list[dict]
    buildings: list[dict]
    containers: dict[str, dict]
    documents: list[dict]
    layout: dict

    @classmethod
    def create(cls, seed: int | None = None, content_path: Path | None = None) -> TownPlan:
        """Read scenario content and assemble a repeatable, server-free plan."""
        content = json.loads(Path(content_path or CONTENT_PATH).read_text(encoding="utf-8"))
        seed = content.get("seed", BASE_SEED) if seed is None else seed
        plan = cls(seed, content, [], [], [], [], [], {}, deepcopy(content["documents"]), {
            "agent_start_game_time": content["agent_start_game_time"],
            "region": {"x0": -160, "y0": -64, "z0": -160, "x1": 159, "y1": -30, "z1": 159},
            "fixture_hash_region": {
                "x0": -160, "y0": -64, "z0": -160, "x1": 159, "y1": -30, "z1": 159},
            "boundary": {"min": -160, "max": 159},
            "river": {"bounds": [-6, -63, -160, 6, -61, 159]},
            "doors": {}, "door_details": [], "windows": [], "lamps": [],
            "road_surfaces": ["minecraft:stone_bricks", "minecraft:polished_andesite", "minecraft:oak_planks",
                              "minecraft:gravel"],
            "npc_schedule": dict(TOWN_SCHEDULE),
            "gardens": [], "furniture": [], "fields": [], "orchards": [], "quay_trees": [],
        })
        rng = random.Random(seed)
        plan._districts(rng)
        plan._crossings()
        plan._residences(rng)
        plan._resources()
        plan._content()
        plan._population()
        plan._lighting()
        plan.layout["buildings"] = plan.buildings
        plan.layout["crossings"] = [
            {"kind": "bridge", "places": ["west_bridge", "east_bridge"]},
            {"kind": "ferry", "places": ["west_landing", "east_landing"]},
            {"kind": "ford", "places": ["west_ford", "east_ford"]},
        ]
        return plan

    @property
    def fixture_name(self) -> str:
        """Keep the new fixture separate from every existing village variant."""
        return "town_jubilee"

    def _road(self, name: str, bounds: list[int], *, dim: bool = True) -> None:
        """Register one rectangular full-block road in the NPC surface network."""
        self.roads.append({"name": name, "bounds": bounds, "dim": dim,
                           "surface": "minecraft:polished_andesite" if dim else "minecraft:stone_bricks"})

    def _place(self, name: str, horizontal: int, depth: int, bank: str, kind: str,
               bounds: list[int] | None = None) -> None:
        """Register an exterior, physically paved destination."""
        self.places.append(Place(name, (horizontal, SURFACE_Y, depth), bank, kind,
                                 tuple(bounds) if bounds else None))

    def _building(self, name: str, kind: str, bounds: list[int], rooms: dict | None = None,
                  roof: str = "spruce") -> dict:
        """Register a stone-and-timber shell with explicit functional interiors."""
        building = {"name": name, "kind": kind, "bounds": bounds, "roof": roof, "storeys": 2,
                    "rooms": rooms or {"main": {"bounds": bounds[:], "doors": []}}}
        self.buildings.append(building)
        self.layout["doors"][name] = []
        return building

    def _door(self, building: dict, name: str, horizontal: int, depth: int, facing: str,
              rooms: tuple[str, ...], *, open_: bool = True) -> None:
        """Record a one-wide door and its two clear approach cells; `open_` is its state at the start (a closed
        door is a wall to the planner until someone opens it)."""
        axis = "x" if facing in ("east", "west") else "z"
        sides = ([[horizontal - 1, -60, depth], [horizontal + 1, -60, depth]] if axis == "x" else
                 [[horizontal, -60, depth - 1], [horizontal, -60, depth + 1]])
        position = [horizontal, -60, depth]
        self.layout["doors"][building["name"]].append(position)
        self.layout["door_details"].append({
            "id": name, "building": building["name"], "pos": position, "facing": facing,
            "axis": axis, "rooms": list(rooms), "sides": sides, "open": open_})
        for room in rooms:
            building["rooms"][room]["doors"].append(name)

    def _window(self, building: str, horizontal: int, depth: int, facing: str,
                *, shuttered: bool = False) -> dict:
        """Record a glass-pane window and optional adjacent oak shutters.

        Shutters fold beside the opening, rather than covering its sight ray
        when open. Their positions remain explicit for grounded interactions.
        """
        outward = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}[facing]
        shutters = []
        if shuttered:
            for offset in (-1, 1):
                shutters.append([horizontal + outward[0] + (offset if outward[1] else 0), -59,
                                 depth + outward[1] + (offset if outward[0] else 0)])
        window = {"building": building, "pos": [horizontal, -59, depth], "facing": facing,
                  "shutters": shutters}
        self.layout["windows"].append(window)
        return window

    def _fronted(self, name: str, kind: str, bounds: list[int], facade: str, bank: str,
                 *, storeys: int = 2, roof: str = "spruce", shopfront: bool = False) -> dict:
        """Attach a ground-floor entrance and overlooking windows to a street-facing shell."""
        building = self._building(name, kind, bounds, roof=roof)
        building.update(storeys=storeys, facade=facade, shopfront=shopfront)
        left, north, right, south = bounds
        horizontal, depth = (left + right) // 2, (north + south) // 2
        if facade in ("north", "south"):
            depth = north if facade == "north" else south
            outside = [horizontal, depth + (-1 if facade == "north" else 1)]
            windows = [(left + 1, depth), (right - 1, depth)]
        else:
            horizontal = left if facade == "west" else right
            outside = [horizontal + (-1 if facade == "west" else 1), depth]
            windows = [(horizontal, north + 1), (horizontal, south - 1)]
        self._door(building, name + "_main", horizontal, depth, facade, ("main",))
        self._place(name, *outside, bank, kind, bounds)
        for window_x, window_z in windows:
            if abs(window_x - horizontal) + abs(window_z - depth) > 0:
                self._window(name, window_x, window_z, facade)
        return building

    def _terrace(self, prefix: str, bounds: list[int], facade: str, bank: str, rng: random.Random,
                 *, count: int | None = None, start: int = 1, homes: bool = False, yards: bool = False) -> None:
        """Divide one continuous frontage into attached, individually accessible 5–9-block lots."""
        left, north, right, south = bounds
        along_x = facade in ("north", "south")
        cursor, end = (left, right) if along_x else (north, south)
        count = count or (end - cursor + 9) // 9
        for index in range(count):
            remaining = count - index - 1
            width = rng.randint(max(5, end - cursor + 1 - 9 * remaining),
                                min(9, end - cursor + 1 - 5 * remaining))
            lot = [cursor, north, cursor + width - 1, south] if along_x else [left, cursor, right, cursor + width - 1]
            name = f"{prefix}_{start + index}"
            building = self._fronted(name, "house" if homes else "shop", lot, facade, bank,
                                     storeys=rng.choice((2, 3)), roof=rng.choice(("spruce", "dark_oak", "brick")),
                                     shopfront=not homes)
            if yards:
                centre = (lot[0] + lot[2]) // 2
                self._door(building, name + "_yard", centre, north, "north", ("main",))
                self._window(name, lot[0] + 1, north, "north")
                self.layout["gardens"].append({"bounds": [lot[0], north - 5, lot[2], north - 1],
                                                "building": name, "fenced": True,
                                                "gates": [[centre, -60, north - 5], [centre, -60, north - 1]]})
            cursor += width

    def _districts(self, rng: random.Random) -> None:
        """Enclose both squares and connect their continuous frontages with narrow lanes."""
        for name, bounds in (
            ("east_north_quay", [8, -148, 10, 12]), ("east_ferry_lane", [8, 49, 9, 110]),
            ("west_ferry_lane", [-10, 62, -8, 100]), ("west_back_lane", [-69, -105, -67, 149]),
            ("west_upstream_lane", [-14, -148, -12, -63]),
            ("west_outer_lane", [-145, -33, -143, 68]), ("west_residential", [-145, -15, -15, -13]),
            ("west_homes_back", [-145, -33, -67, -31]), ("west_home_lane", [-108, -33, -106, -13]),
            ("west_inn_back", [-69, 2, -7, 4]), ("inn_back_lane", [-23, 2, -21, 25]),
            ("mayor_lane", [-145, 25, -67, 27]), ("west_south", [-145, 66, -8, 68]),
            ("quay_ferry_entry", [-14, 60, -8, 62]),
            ("east_residential", [8, -30, 143, -28]), ("east_homes_back", [55, -50, 143, -49]),
            ("east_home_lane", [93, -50, 95, -28]), ("east_outer_lane", [141, -50, 143, 110]),
            ("east_trade_lane", [55, -50, 57, 88]), ("east_shops", [55, 28, 143, 30]),
            ("bakery_back_lane", [31, 49, 57, 51]), ("east_south", [8, 61, 143, 63]),
            ("east_square_lane", [20, 13, 22, 29]), ("east_square_north", [8, 10, 57, 12]),
            ("east_square_south_lane", [20, 48, 22, 63]),
            ("warehouse_lane", [78, 61, 80, 110]), ("dock_street", [8, 86, 143, 88]),
            ("warehouse_front", [17, 105, 143, 107]), ("east_dock_back", [8, 108, 143, 110]),
        ):
            self._road(name, bounds)
        self._road("main_street", [-145, 48, -47, 52], dim=False)
        self._road("west_quay", [-14, -62, -12, 62], dim=False)
        self.layout["plaza"] = {"bounds": [-46, 25, -11, 52]}
        self.layout["east_square"] = {"bounds": [15, 29, 30, 48]}
        self._road("plaza", self.layout["plaza"]["bounds"], dim=False)
        self._road("east_square", self.layout["east_square"]["bounds"], dim=False)
        self._place("plaza", -18, 39, "west", "plaza", self.layout["plaza"]["bounds"])
        self._place("market", -18, 39, "west", "market")
        hall = self._building("town_hall", "town_hall", [-66, 25, -47, 47], {
            "great_hall": {"bounds": [-66, 25, -47, 36], "doors": []},
            "anteroom": {"bounds": [-66, 36, -47, 42], "doors": []},
            "office": {"bounds": [-66, 42, -47, 47], "doors": []},
        }, "dark_oak")
        hall.update(storeys=3, facade="east", shopfront=False)
        self._door(hall, "hall_main", -47, 39, "east", ("anteroom",))
        self._door(hall, "hall_great_hall", -57, 36, "south", ("great_hall", "anteroom"))
        self._door(hall, "hall_office", -57, 42, "south", ("anteroom", "office"))
        # the hall's service door starts closed, so the planner never routes anyone through the hall as a shortcut
        self._door(hall, "hall_back", -66, 30, "west", ("great_hall",), open_=False)
        hall_windows = [self._window("town_hall", -47, depth, "east", shuttered=True) for depth in (28, 32)]
        office_window = self._window("town_hall", -47, 45, "east")
        self._place("town_hall", -46, 39, "west", "town_hall", hall["bounds"])
        self.layout["town_hall"] = {"bounds": hall["bounds"], "rooms": hall["rooms"]}
        self.layout["hall"] = {
            "bounds": [-65, -60, 26, -48, -58, 35], "feet": [-57, -60, 32],
            "feast_tables": [f"feast_table_{number}" for number in range(1, 5)],
            "windows": [window["pos"] for window in hall_windows],
            "shutters": [shutter for window in hall_windows for shutter in window["shutters"]],
        }
        self.layout["office"] = {"bounds": [-65, -60, 43, -48, -58, 46], "feet": [-57, -60, 45],
                                  "window": office_window["pos"], "desk": [-63, -60, 44]}
        self.layout["anteroom"] = {"bounds": [-65, -60, 37, -48, -58, 41], "feet": [-57, -60, 39],
                                    "desk": [-63, -60, 40], "appointments": "appointments"}
        inn = self._building("inn", "inn", [-38, 5, -24, 24], {
            "common": {"bounds": [-38, 13, -24, 24], "doors": []},
            "storeroom": {"bounds": [-38, 5, -24, 13], "doors": []},
        })
        inn.update(storeys=3, facade="south", shopfront=True)
        self._door(inn, "inn_main", -31, 24, "south", ("common",))
        self._door(inn, "inn_store", -31, 13, "south", ("common", "storeroom"))
        self._door(inn, "inn_back", -31, 5, "north", ("storeroom",))
        self._window("inn", -35, 24, "south")
        self._window("inn", -27, 24, "south")
        self._place("inn", -31, 25, "west", "inn", inn["bounds"])
        self.layout["inn"] = {"bounds": inn["bounds"], "rooms": inn["rooms"]}
        self.layout["inn_storeroom"] = {"bounds": [-37, -60, 6, -25, -58, 12],
                                         "feet": [-34, -60, 8], "container": "inn_storeroom"}
        bakery = self._building("bakery", "bakery", [31, 29, 44, 48], {
            "shop": {"bounds": [31, 29, 44, 38], "doors": []},
            "storeroom": {"bounds": [31, 38, 44, 48], "doors": []},
        }, "brick")
        bakery.update(storeys=3, facade="west", shopfront=True)
        self._door(bakery, "bakery_main", 31, 33, "west", ("shop",))
        self._door(bakery, "bakery_store", 37, 38, "south", ("shop", "storeroom"))
        self._door(bakery, "bakery_back", 40, 48, "south", ("storeroom",))
        self._window("bakery", 31, 31, "west")
        self._window("bakery", 31, 35, "west")
        self._place("bakery", 30, 33, "east", "bakery", bakery["bounds"])
        self.layout["bakery"] = {"bounds": bakery["bounds"], "rooms": bakery["rooms"],
                                  "oven": [42, -60, 31], "feet": [37, -60, 33]}
        civic = [
            ("baker_house", "house", [45, 29, 53, 48], "south", "east"),
            ("mill", "mill", [23, 13, 35, 28], "south", "east"),
            ("builder_workshop", "workshop", [58, 64, 73, 78], "north", "east"),
            ("warehouse", "warehouse", [43, 89, 58, 104], "south", "east"),
            ("cottage", "house", [10, 91, 22, 104], "south", "east"),
            ("mayor_house", "house", [-108, 8, -91, 24], "south", "west"),
            ("school", "school", [-142, 35, -125, 47], "south", "west"),
            ("workshop_1", "workshop", [-124, 35, -111, 47], "south", "west"),
            ("workshop_2", "workshop", [-96, 35, -80, 47], "south", "west"),
            ("chapel", "chapel", [-90, 7, -73, 24], "south", "west"),
        ]
        for name, kind, bounds, facade, bank in civic:
            self._fronted(name, kind, bounds, facade, bank, roof="brick" if bank == "east" else "spruce",
                          shopfront=kind in ("mill", "workshop"))
        self._road("cottage_entry", [16, 105, 16, 108])
        self.layout["chapel"] = {"tower": [-80, -56, 8, -74, -35, 14], "bell": [-86, -60, 11]}
        neighbour_window = self._window("cottage", 10, 98, "west")
        self.layout["neighbour_window"] = {"pos": neighbour_window["pos"], "observer": [11.5, -60, 98.5],
                                           "target": [2.5, -60.5, 98.5], "landing": "east", "distance": 9.0}
        self.layout["mill"] = {"bounds": [23, 13, 35, 28], "stall": "mill_stall"}
        for prefix, bounds, facade, bank in (
            ("plaza_north", [-46, 10, -39, 24], "south", "west"),
            ("plaza_north_shop", [-20, 10, -15, 24], "south", "west"),
            ("plaza_river_north", [-10, 25, -7, 37], "west", "west"),
            ("plaza_river_south", [-10, 41, -7, 52], "west", "west"),
            ("plaza_south", [-66, 53, -15, 65], "north", "west"),
            ("main_north", [-110, 35, -97, 47], "south", "west"),
            ("main_chapel_end", [-79, 35, -70, 47], "south", "west"),
            ("main_south", [-142, 53, -70, 65], "north", "west"),
            ("quay_west", [-23, -62, -15, -19], "east", "west"),
            ("quay_east", [-11, -62, -7, -19], "west", "west"),
            ("quay_west_inner", [-23, -12, -15, 1], "east", "west"),
            ("quay_east_inner", [-11, -12, -7, 1], "west", "west"),
            ("square_north", [15, 13, 19, 28], "south", "east"),
            ("square_south_west", [15, 49, 19, 60], "north", "east"),
            ("square_south_east", [23, 49, 30, 60], "north", "east"),
            ("square_west_north", [10, 29, 14, 37], "east", "east"),
            ("square_west_south", [10, 41, 14, 48], "east", "east"),
            ("east_trade_north", [58, 14, 140, 27], "south", "east"),
            ("east_trade_south", [58, 31, 140, 43], "north", "east"),
        ):
            self._terrace(prefix, bounds, facade, bank, rng)
        self._terrace("quay_corner_north", [-11, 5, -7, 24], "west", "west", rng)
        self._fronted("quay_corner_south", "shop", [-11, 53, -7, 59], "west", "west", shopfront=True)
        for number, bounds in enumerate(([59, 89, 67, 104], [68, 89, 76, 104]), 1):
            self._fronted(f"warehouse_annex_{number}", "warehouse", bounds, "south", "east", storeys=3)
        for number, bounds in enumerate(([23, 91, 31, 104], [32, 91, 40, 104],
                                        [10, 74, 18, 85], [19, 74, 27, 85]), 1):
            self._fronted(f"dock_cottage_{number}", "house", bounds, "south", "east")
        self.layout["fountain"] = {"bounds": [-31, -61, 36, -27, -60, 40]}
        self.layout["furniture"] = [
            {"pos": self.layout["office"]["desk"], "block": "minecraft:dark_oak_planks"},
            {"pos": self.layout["anteroom"]["desk"], "block": "minecraft:oak_planks"},
            {"pos": self.layout["bakery"]["oven"], "block": "minecraft:smoker[facing=west,lit=false]"},
            {"pos": [-138, -60, 38], "block": "minecraft:lectern"},
            {"pos": self.layout["chapel"]["bell"], "block": "minecraft:bell[attachment=floor,facing=south]"},
            *({"pos": position, "block": "minecraft:crafting_table"}
              for position in ([-121, -60, 38], [-93, -60, 38], [60, -60, 75],
                               [42, -60, 35], [-36, -60, 20])),
        ]
        for horizontal in (-44, -36, -28, -20, -12):
            for depth in (25, 52):
                self.layout["lamps"].append([horizontal, -56, depth])
        for horizontal in (-44, -12):
            for depth in (33, 41):
                self.layout["lamps"].append([horizontal, -56, depth])
        self.layout["lamps"].extend([[horizontal, -56, depth]
                                       for horizontal in (-38, -20) for depth in (35, 44)])
        self.layout["lamps"].extend([[17, -56, 31], [28, -56, 31], [17, -56, 46], [28, -56, 46]])
        self.layout["lamps"].extend([[horizontal, -56, 50] for horizontal in range(-140, -47, 10)])
        self.layout["lamps"].extend([[-12, -56, depth] for depth in range(-62, 25, 10)])
        self.layout["lamps"].append([-12, -56, 58])
        self.layout["quay_trees"] = [[-12, -60, depth] for depth in (-57, -37, -17, -7)]
        self.layout["dim_areas"] = {
            "west_ferry_lane": [-10, 62, -8, 88], "east_ferry_lane": [8, 62, 9, 88],
            "hall_back_lane": [-69, 25, -67, 47], "inn_back_lane": [-38, 2, -24, 4],
            "west_side_street": [-145, -15, -24, -13], "east_side_street": [55, -30, 143, -28],
            "ford": [-20, -151, 20, -136],
        }

    def _crossings(self) -> None:
        """Declare the sole three river crossings and the existing ferry schema."""
        self.layout["bridge"] = {
            "bounds": [-8, -61, 38, 8, -61, 40], "deck_y": -61, "width": 3,
            "parapets": [[-6, -60, depth, 6, -60, depth] for depth in (37, 41)],
        }
        self._road("west_bridge_road", [-11, 38, -7, 40], dim=False)
        self._road("east_bridge_road", [7, 38, 15, 40], dim=False)
        self._place("west_bridge", -8, 39, "west", "bridge")
        self._place("east_bridge", 8, 39, "east", "bridge")
        self.layout["lamps"].extend([[horizontal, -56, 37] for horizontal in (-11, 0, 8)])
        self.layout["ford"] = {
            "bounds": [-7, -63, -148, 7, -61, -148], "width": 1,
            "route_endpoints": ["town_hall", "bakery"],
            "detour_blocks": 444, "walking_minutes_at_npc_speed": 444 / (0.12 * 20 * 60),
        }
        self._road("west_ford_entry", [-14, -148, -7, -148])
        self._road("east_ford_entry", [7, -148, 10, -148])
        self._place("west_ford", -8, -148, "west", "ford")
        self._place("east_ford", 8, -148, "east", "ford")
        boat_id = "00000000-0000-0000-0000-0000000002bf"
        self.layout["port"] = {
            "deck_y": -61, "platforms": [[-9, 96, -5, 100], [4, 96, 9, 100]],
            "boat": {"pos": [2.5, -60.5, 98.5], "uuid": boat_id, "tag": "town_boat",
                     "type": "minecraft:oak_boat"},
        }
        self.layout["ferry"] = {
            "boat": boat_id, "landings": {
                "west": {"board": [-5, -61, 98], "moor": [-2.5, -60.5, 98.5]},
                "east": {"board": [4, -61, 98], "moor": [2.5, -60.5, 98.5]},
            }, "speed": 0.4,
        }
        self._road("west_landing_entry", [-9, 97, -5, 99])
        self._road("east_landing_entry", [4, 97, 9, 99])
        self._place("west_landing", -8, 98, "west", "landing")
        self._place("east_landing", 8, 98, "east", "landing")
        self.layout["lamps"].extend([[-7, -56, 97], [6, -56, 97]])

    def _residences(self, rng: random.Random) -> None:
        """Give each bank ten attached homes with separate front doors and fenced back yards."""
        for bank, first, second in (
            ("west", [-142, -25, -109, -16], [-105, -25, -70, -16]),
            ("east", [58, -40, 92, -31], [96, -40, 130, -31]),
        ):
            for start, bounds in ((1, first), (6, second)):
                self._terrace(bank + "_home", bounds, "south", bank, rng,
                              count=5, start=start, homes=True, yards=True)
        self._road("east_yard_entry", [55, -48, 57, -44])
        self._road("east_yard_back", [55, -48, 143, -46])

    def _resources(self) -> None:
        """Provide harvestable wood and stone at the town edges, clear of roads."""
        self.layout["forest"] = {"trees": [[horizontal, -60, depth]
                                            for horizontal in (-149, -141, -133)
                                            for depth in (-140, -132, -124, -116)]}
        self.layout["quarry"] = {"bounds": [-152, -63, 140, -124, -61, 155],
                                  "iron": [[-151, -60, depth] for depth in range(143, 151)]}
        self._road("forest_lane", [-151, -106, -8, -104])
        self._road("quarry_lane", [-123, 147, -68, 149])
        self._place("forest", -145, -105, "west", "forest", [-153, -144, -129, -112])
        self._place("quarry", -122, 148, "west", "quarry", [-152, 140, -124, 155])
        self.layout["fields"] = [{"bounds": [-125, -100, -80, -68]}, {"bounds": [30, -130, 115, -92]}]
        self.layout["orchards"] = [
            {"bounds": [-120, -138, -80, -112],
             "trees": [[horizontal, -60, depth] for horizontal in range(-116, -81, 7)
                       for depth in range(-134, -114, 7)]},
            {"bounds": [60, 119, 133, 146],
             "trees": [[horizontal, -60, depth] for horizontal in range(64, 131, 8)
                       for depth in range(123, 144, 8)]},
        ]

    def _content(self) -> None:
        """Assign geometry to content containers without rewriting their stocks or goals."""
        positions = {
            "appointments": [-64, -60, 38], "treasury": [-64, -60, 45],
            **{f"feast_table_{number}": [-66 + number * 4, -60, 30] for number in range(1, 5)},
            "inn_storeroom": [-36, -60, 8], "bakery_storeroom": [42, -60, 44],
            "warehouse": [55, -60, 100], "builder_store": [69, -60, 75],
            "bakery_stall": [-40, -60, 27], "mill_stall": [32, -60, 25],
            "stall_planks": [-30, -60, 27], "stall_books": [-20, -60, 27], "stall_lanterns": [-44, -60, 47],
        }
        for name, source in {**self.content["containers"], **self.content["stalls"]}.items():
            if name not in positions:
                raise ValueError(f"No town position assigned to content container {name!r}")
            entry = deepcopy(source)
            entry.update(pos=positions[name], allowed=source.get("allowed"))
            if name in self.content["stalls"]:
                horizontal, height, depth = entry["pos"]
                entry["sign_pos"] = [horizontal, height, depth + 2]
            self.containers[name] = entry
        self.layout["config"] = {
            "night_sight": True,
            "visible_loads": {"minecraft:bread": 9, "minecraft:hay_block": 1, "minecraft:wheat": 16},
            "carry_limits": {"minecraft:bread": 32, "minecraft:wheat": 96, "minecraft:hay_block": 8},
            "fresh_days": 2, "bake_ticks": 40,
            "villager_purchases": [{"stall": self.containers["bakery_stall"]["pos"],
                                    "item": "minecraft:bread", "count": 1}],
        }
        places = {place.name: place for place in self.places}
        for source in self.content["roles"]:
            horizontal, height, depth = places[source["spawn_place"]].pos
            self.roles.append(Role(source["name"], (horizontal + 0.5, height + 1, depth + 0.5),
                                   deepcopy(source["kit"]), source["public_goal"], source["private_goal"]))

    def _population(self) -> None:
        """Distribute thirty workers and ten schoolchildren between both banks."""
        workplaces = ("mill", "bakery", "inn", "workshop_1", "workshop_2", "school", "warehouse",
                      "town_hall", "builder_workshop", "plaza")
        for index in range(30):
            bank = "west" if index % 2 == 0 else "east"
            home = f"{bank}_home_{index // 2 % 10 + 1}"
            self.npcs.append(NpcSpec(f"adult_{index + 1:02d}", f"Townsfolk {index + 1:02d}", home, "adult",
                                     work=workplaces[index % len(workplaces)]))
        for index in range(10):
            bank = "west" if index % 2 == 0 else "east"
            self.npcs.append(NpcSpec(f"child_{index + 1:02d}", f"Pupil {index + 1:02d}",
                                     f"{bank}_home_{index // 2 + 1}", "child", school="school", market_days=()))

    def _lighting(self) -> None:
        """Keep frontage doorsteps clear without adding lights to any quiet route."""
        reserved = {(place.pos[0], place.pos[2]) for place in self.places}
        reserved.update((side[0], side[2]) for door in self.layout["door_details"] for side in door["sides"])
        reserved.update((position[0], position[2]) for position in self.layout["quay_trees"])
        for lamp in self.layout["lamps"]:
            horizontal, height, depth = lamp
            if (horizontal, depth) not in reserved:
                reserved.add((horizontal, depth))
                continue
            for offset_x, offset_z in ((-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, 1)):
                candidate_x, candidate_z = horizontal + offset_x, depth + offset_z
                if (candidate_x, candidate_z) in reserved:
                    continue
                if not any(not road["dim"] and road["bounds"][0] <= candidate_x <= road["bounds"][2]
                           and road["bounds"][1] <= candidate_z <= road["bounds"][3] for road in self.roads):
                    continue
                if any(bounds[0] <= candidate_x <= bounds[2] and bounds[1] <= candidate_z <= bounds[3]
                       for bounds in self.layout["dim_areas"].values()):
                    continue
                lamp[:] = [candidate_x, height, candidate_z]
                reserved.add((candidate_x, candidate_z))
                break
            else:
                raise ValueError(f"No clear lit-street position for lamp {lamp}")

    def npc_payload(self) -> dict:
        """Name every building, including its doors, in the news landmark catalogue."""
        landmarks = []
        for building in self.buildings:
            left, north, right, south = building["bounds"]
            landmarks.append({"name": building["name"], "region": {
                "min": [left, -61, north], "max": [right, self.layout["region"]["y1"], south]}})
        return {"region": self.layout["region"], "surfaces": self.layout["road_surfaces"],
                "places": {place.name: list(place.pos) for place in self.places},
                "npcs": [npc.payload() for npc in self.npcs], "speed": 0.12, "schedule": self.layout["npc_schedule"],
                "landmarks": [*landmarks,
                    {"name": "the bridge", "region": {
                        "min": [-10, -63, 36], "max": [10, -55, 42]}},
                    {"name": "the port", "region": {
                        "min": [-10, -63, 94], "max": [10, -55, 102]}},
                ]}

    def to_dict(self) -> dict:
        """Serialize geometry plus the original role briefs, stocks, documents and committee."""
        roles = [{**deepcopy(source), **asdict(role)} for source, role in zip(self.content["roles"], self.roles)]
        return {"name": "jubilee", "variant": "jubilee", "seed": self.seed, "roles": roles,
                "places": [{**asdict(place), "feet": [place.pos[0], place.pos[1] + 1, place.pos[2]]}
                           for place in self.places], "npcs": [asdict(npc) for npc in self.npcs],
                "roads": deepcopy(self.roads), "buildings": deepcopy(self.buildings),
                "containers": deepcopy(self.containers), "documents": deepcopy(self.documents),
                "stalls": {name: deepcopy(self.containers[name]) for name in self.content["stalls"]},
                "jubilee": {"committee": deepcopy(self.content["committee"]),
                            "reveal_game_time": self.content["reveal_game_time"],
                            "agent_start_game_time": self.content["agent_start_game_time"], "condition": {}},
                "tools": ["bake", "sleep"], "schematics": {}, "success_predicates": {},
                "layout": deepcopy(self.layout)}

    def write(self, path: Path) -> None:
        """Write a reviewable manifest without launching Minecraft."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
