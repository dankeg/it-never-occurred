"""Pure, deterministic block-command components for the Jubilee town.

Components never contact a server. Surface blocks occupy y=-61, leaving the
ground-floor walking cells at y=-60; visual upper storeys remain solid and sealed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


SURFACE_Y = -61
FILL_LIMIT = 32_768
Bounds = tuple[int, int, int, int]
Volume = tuple[int, int, int, int, int, int]


def _fill(bounds: Sequence[int], block: str) -> list[str]:
    """Emit inclusive fills, splitting large volumes below vanilla's block limit."""
    coordinates = tuple(bounds)
    sizes = [coordinates[axis + 3] - coordinates[axis] + 1 for axis in range(3)]
    if any(size < 1 for size in sizes):
        return []
    if sizes[0] * sizes[1] * sizes[2] <= FILL_LIMIT:
        return ["fill " + " ".join(map(str, coordinates)) + " " + block]
    axis = max(range(3), key=lambda index: sizes[index])
    middle = (coordinates[axis] + coordinates[axis + 3]) // 2
    lower, upper = list(coordinates), list(coordinates)
    lower[axis + 3], upper[axis] = middle, middle + 1
    return [*_fill(lower, block), *_fill(upper, block)]


def _block(pos: Sequence[int], block: str) -> str:
    """Format a single vanilla block placement without executing it."""
    return "setblock " + " ".join(map(str, pos)) + " " + block


def _walls(bounds: Sequence[int], bottom: int, top: int, block: str) -> list[str]:
    """Outline a rectangle without filling its walkable interior."""
    left, north, right, south = bounds
    commands = []
    for depth in (north, south):
        commands.extend(_fill((left, bottom, depth, right, top, depth), block))
    for horizontal in (left, right):
        commands.extend(_fill((horizontal, bottom, north, horizontal, top, south), block))
    return commands


def _value(plan: object, key: str, default=None):
    """Read a plan object or its serialized manifest using the same schema."""
    return plan.get(key, default) if isinstance(plan, Mapping) else getattr(plan, key, default)


@dataclass(frozen=True)
class Roof:
    """A pitched stair roof with a slab ridge, contained within its footprint."""

    bounds: Bounds
    material: str = "spruce"
    eaves_y: int = -53
    max_y: int = -30

    def commands(self) -> list[str]:
        """Run the ridge along the long axis while respecting the fixture height."""
        left, north, right, south = self.bounds
        material = self.material.removeprefix("minecraft:")
        material = material.removesuffix("_stairs").removesuffix("_slab").removesuffix("_planks")
        solids = {"brick": "bricks", "stone_brick": "stone_bricks",
                  "deepslate_tile": "deepslate_tiles", "deepslate_brick": "deepslate_bricks"}
        solid = "minecraft:" + solids.get(material, material + "_planks")
        stairs, slab = f"minecraft:{material}_stairs", f"minecraft:{material}_slab[type=bottom]"
        across_x = right - left < south - north
        start, end = (left, right) if across_x else (north, south)
        commands = []
        for inset in range((end - start) // 2 + 1):
            low, high, height = start + inset, end - inset, self.eaves_y + inset
            ridge = high - low <= 1 or height >= self.max_y
            if across_x:
                layer = (low, height, north, high, height, south)
                edges = ((low, height, north, low, height, south),
                         (high, height, north, high, height, south))
                facings = ("east", "west")
            else:
                layer = (left, height, low, right, height, high)
                edges = ((left, height, low, right, height, low),
                         (left, height, high, right, height, high))
                facings = ("south", "north")
            commands.extend(_fill(layer, slab if ridge else solid))
            if ridge:
                break
            for edge, facing in zip(edges, facings):
                commands.extend(_fill(edge, f"{stairs}[facing={facing},half=bottom,shape=straight]"))
        return commands


@dataclass(frozen=True)
class Building:
    """A stone ground floor beneath two- or three-storey sealed timber-framed facades."""

    bounds: Bounds
    rooms: Mapping[str, Mapping] = field(default_factory=dict)
    doors: Sequence[Mapping] = ()
    windows: Sequence[Mapping] = ()
    roof: str = "spruce"
    height: int = -56
    name: str = ""
    keep_clear: Sequence[Bounds] = ()
    storeys: int = 2
    facade: str | None = None
    shopfront: bool = False

    def commands(self) -> list[str]:
        """Finish partitions before cutting every one-wide door and glass window."""
        left, north, right, south = self.bounds
        commands = _fill((left, SURFACE_Y, north, right, SURFACE_Y, south), "minecraft:oak_planks")
        commands.extend(_walls(self.bounds, SURFACE_Y + 1, self.height - 1, "minecraft:stone_bricks"))
        commands.extend(_fill((left + 1, SURFACE_Y + 1, north + 1, right - 1, self.height - 1, south - 1),
                              "minecraft:air"))
        for room in self.rooms.values():
            commands.extend(_walls(room["bounds"], SURFACE_Y + 1, self.height - 1, "minecraft:stone_bricks"))
        upper_top = self.height + 4 * (self.storeys - 1) - 1
        commands.extend(_fill((left, self.height, north, right, upper_top, south),
                              "minecraft:white_concrete"))
        for horizontal in (left, right):
            for depth in (north, south):
                commands.extend(_fill((horizontal, SURFACE_Y + 1, depth, horizontal, upper_top, depth),
                                      "minecraft:stripped_spruce_log[axis=y]"))
        for height in range(self.height, upper_top + 1, 4):
            commands.extend(_walls(self.bounds, height, height, "minecraft:stripped_spruce_log[axis=x]"))
        facades = []
        for horizontal in range(left + 2, right - 1, 4):
            facades.extend(((horizontal, north, "north", 0, -1), (horizontal, south, "south", 0, 1)))
        for depth in range(north + 2, south - 1, 4):
            facades.extend(((left, depth, "west", -1, 0), (right, depth, "east", 1, 0)))
        for horizontal, depth, facing, offset_x, offset_z in facades:
            if self.facade and facing != self.facade:
                continue
            for height in range(self.height + 1, upper_top, 4):
                commands.append(_block((horizontal, height, depth), "minecraft:stripped_spruce_log[axis=y]"))
                commands.append(_block((horizontal, height + 1, depth), "minecraft:glass_pane"))
            box_x, box_z = horizontal + offset_x, depth + offset_z
            if not any(bounds[0] <= box_x <= bounds[2] and bounds[1] <= box_z <= bounds[3]
                       for bounds in self.keep_clear):
                commands.append(_block((box_x, self.height + 1, box_z), "minecraft:composter[level=0]"))
                commands.append(_block((box_x, self.height + 2, box_z), "minecraft:potted_poppy"))
        if self.shopfront:
            frontage = {"north": (left, -57, north, right, -57, north),
                        "south": (left, -57, south, right, -57, south),
                        "west": (left, -57, north, left, -57, south),
                        "east": (right, -57, north, right, -57, south)}[self.facade or "south"]
            commands.extend(_fill(frontage, "minecraft:stripped_dark_oak_log[axis=x]"))
        commands.extend(Roof(self.bounds, self.roof, upper_top + 1).commands())
        for window in self.windows:
            horizontal, height, depth = window["pos"]
            facing = window.get("facing", "south")
            connections = "east=true,west=true" if facing in ("north", "south") else "north=true,south=true"
            commands.append(_block((horizontal, height, depth), f"minecraft:glass_pane[{connections}]"))
            for shutter in window.get("shutters", []):
                commands.append(_block(shutter, f"minecraft:oak_trapdoor[facing={facing},half=top,"
                                       "open=true,powered=false,waterlogged=false]"))
        for door in self.doors:
            horizontal, bottom, depth = door["pos"]
            commands.extend(_fill((horizontal, bottom, depth, horizontal, bottom + 1, depth), "minecraft:air"))
            for offset, half in ((0, "lower"), (1, "upper")):
                state = "true" if door.get("open", True) else "false"
                commands.append(_block((horizontal, bottom + offset, depth),
                                       f"minecraft:oak_door[facing={door['facing']},half={half},"
                                       f"hinge=left,open={state},powered=false]"))
        return commands


@dataclass(frozen=True)
class Street:
    """A full-block paved road with flush kerbs and no protruding path hazards."""

    bounds: Bounds
    surface: str = "minecraft:stone_bricks"

    def commands(self) -> list[str]:
        """Pave one rectangle at the common ground surface."""
        left, north, right, south = self.bounds
        commands = _fill((left, SURFACE_Y, north, right, SURFACE_Y, south), self.surface)
        commands.extend(_walls(self.bounds, SURFACE_Y, SURFACE_Y, "minecraft:polished_andesite"))
        return commands


@dataclass(frozen=True)
class LampPost:
    """A recorded lantern above a four-block fence pole, never an implicit light."""

    pos: tuple[int, int, int]

    def commands(self) -> list[str]:
        """Place only the lamp specified by the plan, with no extra ground clutter."""
        horizontal, top, depth = self.pos
        return [*_fill((horizontal, SURFACE_Y + 1, depth, horizontal, top - 1, depth),
                       "minecraft:spruce_fence"),
                _block(self.pos, "minecraft:lantern[hanging=false,waterlogged=false]")]


@dataclass(frozen=True)
class Bridge:
    """A three-clear-lane stone span with parapets outside its walking deck."""

    bounds: Volume
    parapets: Sequence[Sequence[int]] = ()

    def commands(self) -> list[str]:
        """Emit the declared deck and its solid side protection."""
        left, bottom, north, right, top, south = self.bounds
        commands = _fill(self.bounds, "minecraft:stone_bricks")
        parapets = self.parapets or ((left, top + 1, north - 1, right, top + 1, north - 1),
                                     (left, top + 1, south + 1, right, top + 1, south + 1))
        for bounds in parapets:
            commands.extend(_fill(bounds, "minecraft:stone_brick_wall"))
        for horizontal in (left, right):
            commands.extend(_fill((horizontal, bottom - 2, north, horizontal, bottom - 1, south),
                                  "minecraft:stone_bricks"))
        return commands


@dataclass(frozen=True)
class Fountain:
    """A contained stone basin and central spout without any light source."""

    bounds: Volume

    def commands(self) -> list[str]:
        """Keep water level with the rim so it never floods the plaza."""
        left, bottom, north, right, top, south = self.bounds
        horizontal, depth = (left + right) // 2, (north + south) // 2
        commands = _fill(self.bounds, "minecraft:stone_bricks")
        commands.extend(_fill((left + 1, top, north + 1, right - 1, top, south - 1), "minecraft:water"))
        commands.extend(_fill((horizontal, top, depth, horizontal, top + 2, depth), "minecraft:cobblestone_wall"))
        commands.append(_block((horizontal, top + 3, depth), "minecraft:stone_brick_slab[type=bottom]"))
        return commands


@dataclass(frozen=True)
class Garden:
    """A green back yard or orchard, with optional walkable gates and perimeter fences."""

    bounds: Bounds
    fenced: bool = False
    gates: Sequence[Sequence[int]] = ()
    flowers: bool = True

    def commands(self) -> list[str]:
        """Use a flush stone edging and coordinate-selected flowers, never randomness."""
        left, north, right, south = self.bounds
        commands = _fill((left, SURFACE_Y, north, right, SURFACE_Y, south), "minecraft:grass_block")
        commands.extend(_walls(self.bounds, SURFACE_Y, SURFACE_Y, "minecraft:mossy_cobblestone"))
        flowers = ("minecraft:poppy", "minecraft:dandelion", "minecraft:oxeye_daisy")
        if self.flowers:
            for horizontal in range(left + 1, right, 2):
                for depth in range(north + 1, south, 2):
                    if any(gate[0] == horizontal for gate in self.gates):
                        continue
                    commands.append(_block((horizontal, SURFACE_Y + 1, depth),
                                           flowers[(horizontal + depth) % len(flowers)]))
        if self.fenced:
            commands.extend(_walls(self.bounds, SURFACE_Y + 1, SURFACE_Y + 1, "minecraft:oak_fence"))
            for gate in self.gates:
                commands.append(_block(gate, "minecraft:oak_fence_gate[facing=north,open=true]"))
        return commands


@dataclass(frozen=True)
class Tree:
    """A compact persistent tree whose crown never replaces an adjoining building."""

    pos: tuple[int, int, int]
    keep_clear: Sequence[Bounds] = ()

    def commands(self) -> list[str]:
        """Emit a trunk and a clipped crown above pedestrian headroom."""
        horizontal, bottom, depth = self.pos
        commands = []
        if not self.keep_clear:
            commands.extend(_fill((horizontal - 2, bottom + 3, depth - 2, horizontal + 2, bottom + 4, depth + 2),
                                  "minecraft:oak_leaves[persistent=true]"))
        else:
            for canopy_x in range(horizontal - 2, horizontal + 3):
                for canopy_z in range(depth - 2, depth + 3):
                    if not any(bounds[0] <= canopy_x <= bounds[2] and bounds[1] <= canopy_z <= bounds[3]
                               for bounds in self.keep_clear):
                        commands.extend(_fill((canopy_x, bottom + 3, canopy_z, canopy_x, bottom + 4, canopy_z),
                                              "minecraft:oak_leaves[persistent=true]"))
        commands.extend(_fill((horizontal, bottom, depth, horizontal, bottom + 3, depth), "minecraft:oak_log"))
        return commands


@dataclass(frozen=True)
class Field:
    """A fenced, irrigated green wheat field rather than an empty outskirts lawn."""

    bounds: Bounds

    def commands(self) -> list[str]:
        """Plant deterministic young wheat around contained irrigation channels."""
        left, north, right, south = self.bounds
        commands = Garden(self.bounds, fenced=True, flowers=False).commands()
        commands.extend(_fill((left + 1, SURFACE_Y, north + 1, right - 1, SURFACE_Y, south - 1),
                              "minecraft:farmland[moisture=7]"))
        commands.extend(_fill((left + 1, SURFACE_Y + 1, north + 1, right - 1, SURFACE_Y + 1, south - 1),
                              "minecraft:wheat[age=3]"))
        for horizontal in range(left + 4, right, 8):
            commands.extend(_fill((horizontal, SURFACE_Y + 1, north + 1, horizontal, SURFACE_Y + 1, south - 1),
                                  "minecraft:air"))
            commands.extend(_fill((horizontal, SURFACE_Y, north + 1, horizontal, SURFACE_Y, south - 1),
                                  "minecraft:water"))
        return commands


def town_commands(plan: object) -> list[str]:
    """Compile a complete town's geometry, leaving stocks and documents to its builder.

    Road rectangles are clipped to the banks before the three explicit crossings
    are placed; in particular a ferry platform can never become a fourth bridge.
    All command lists are stable and all bulk fills fit the vanilla command limit.
    """
    layout = _value(plan, "layout", {})
    boundary = layout.get("boundary", {"min": -160, "max": 159})
    minimum, maximum = boundary["min"], boundary["max"]
    region = layout.get("fixture_hash_region", {"y1": -43})
    top = region.get("y1", -43)
    commands = _fill((minimum, SURFACE_Y + 1, minimum, maximum, top, maximum), "minecraft:air")
    commands.extend(_fill((minimum, -64, minimum, maximum, -64, maximum), "minecraft:bedrock"))
    commands.extend(_fill((minimum, -63, minimum, maximum, SURFACE_Y - 1, maximum), "minecraft:dirt"))
    commands.extend(_fill((minimum, SURFACE_Y, minimum, maximum, SURFACE_Y, maximum), "minecraft:grass_block"))
    river = layout.get("river", {"bounds": [-6, -63, minimum, 6, SURFACE_Y, maximum]})
    commands.extend(_fill(river["bounds"], "minecraft:water"))
    river_left, _, _, river_right, _, _ = river["bounds"]
    for road in _value(plan, "roads", []):
        left, north, right, south = road["bounds"]
        for bank_left, bank_right in ((left, min(right, river_left - 1)), (max(left, river_right + 1), right)):
            if bank_left <= bank_right:
                commands.extend(Street((bank_left, north, bank_right, south),
                                       road.get("surface", "minecraft:stone_bricks")).commands())
    for name in ("plaza", "east_square"):
        if name in layout:
            commands.extend(Street(tuple(layout[name]["bounds"])).commands())
    if "bridge" in layout:
        bridge = layout["bridge"]
        commands.extend(Bridge(tuple(bridge["bounds"]), bridge.get("parapets", ())).commands())
    if "ford" in layout:
        commands.extend(_fill(layout["ford"]["bounds"], "minecraft:gravel"))
    port = layout.get("port", {})
    for platform in port.get("platforms", []):
        left, north, right, south = platform
        commands.extend(_fill((left, port["deck_y"], north, right, port["deck_y"], south), "minecraft:oak_planks"))
    buildings = _value(plan, "buildings", layout.get("buildings", []))
    details = layout.get("door_details", [])
    windows = layout.get("windows", [])
    keep_clear = [tuple(road["bounds"]) for road in _value(plan, "roads", [])]
    keep_clear.append((river_left, minimum, river_right, maximum))
    keep_clear.extend(tuple(building["bounds"]) for building in buildings)
    for building in buildings:
        name = building["name"]
        doors = [door for door in details if door["building"] == name]
        if not doors:
            left, north, right, south = building["bounds"]
            for pos in layout.get("doors", {}).get(name, []):
                facing = ("west" if pos[0] == left else "east" if pos[0] == right
                          else "north" if pos[2] == north else "south")
                doors.append({"pos": pos, "facing": facing})
        commands.extend(Building(tuple(building["bounds"]), building.get("rooms", {}), doors,
                                 [window for window in windows if window["building"] == name],
                                 building.get("roof", "spruce"), building.get("height", -56), name,
                                 keep_clear, building.get("storeys", 2), building.get("facade"),
                                 building.get("shopfront", False)).commands())
    if "chapel" in layout:
        left, bottom, north, right, top, south = layout["chapel"]["tower"]
        commands.extend(_fill((left, bottom, north, right, top, south), "minecraft:stone_bricks"))
        for height in (top - 8, top - 4):
            for horizontal in (left, right):
                commands.append(_block((horizontal, height, (north + south) // 2), "minecraft:glass_pane"))
        commands.extend(Roof((left, north, right, south), "dark_oak", top + 1, -30).commands())
    if "fountain" in layout:
        commands.extend(Fountain(tuple(layout["fountain"]["bounds"])).commands())
    for garden in layout.get("gardens", []):
        commands.extend(Garden(tuple(garden["bounds"]), garden.get("fenced", False),
                               garden.get("gates", ())).commands())
    for field_plan in layout.get("fields", []):
        commands.extend(Field(tuple(field_plan["bounds"])).commands())
    for orchard in layout.get("orchards", []):
        commands.extend(Garden(tuple(orchard["bounds"]), fenced=True, flowers=False).commands())
        for position in orchard["trees"]:
            commands.extend(Tree(tuple(position)).commands())
    for position in layout.get("quay_trees", []):
        commands.extend(Tree(tuple(position), [tuple(building["bounds"]) for building in buildings]).commands())
    for horizontal, bottom, depth in layout.get("forest", {}).get("trees", []):
        commands.extend(_fill((max(minimum, horizontal - 2), bottom + 3, max(minimum, depth - 2),
                               min(maximum, horizontal + 2), bottom + 4, min(maximum, depth + 2)),
                              "minecraft:oak_leaves[persistent=true]"))
        commands.extend(_fill((horizontal, bottom, depth, horizontal, bottom + 3, depth), "minecraft:oak_log"))
    quarry = layout.get("quarry", {})
    if "bounds" in quarry:
        commands.extend(_fill(quarry["bounds"], "minecraft:stone"))
        for pos in quarry.get("iron", []):
            commands.append(_block(pos, "minecraft:iron_ore"))
    plaza = layout.get("plaza", {}).get("bounds")
    for number, container in enumerate(_value(plan, "containers", {}).values()):
        horizontal, _, depth = container["pos"]
        if container.get("prices") and plaza and plaza[0] <= horizontal <= plaza[2] and plaza[1] <= depth <= plaza[3]:
            for post_x in (horizontal - 1, horizontal + 1):
                commands.extend(_fill((post_x, SURFACE_Y + 1, depth - 1, post_x, -58, depth - 1),
                                      "minecraft:stripped_spruce_log[axis=y]"))
            color = ("white", "red", "green", "yellow")[number % 4]
            commands.extend(_fill((horizontal - 1, -57, depth - 1, horizontal + 1, -57, depth + 1),
                                  f"minecraft:{color}_wool"))
    for furnishing in layout.get("furniture", []):
        commands.append(_block(furnishing["pos"], furnishing["block"]))
    for lamp in layout.get("lamps", []):
        commands.extend(LampPost(tuple(lamp)).commands())
    for depth in (minimum, maximum):
        commands.extend(_fill((minimum, -63, depth, maximum, -57, depth), "minecraft:stone_bricks"))
    for horizontal in (minimum, maximum):
        commands.extend(_fill((horizontal, -63, minimum, horizontal, -57, maximum), "minecraft:stone_bricks"))
    return commands
