"""Render the Jubilee plan as a labeled top-down PNG using only the standard library.

The renderer accepts either a TownPlan or a serialized manifest. It does not
connect to Minecraft, invoke an image converter or depend on installed fonts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import struct
import zlib


FONT = {
    "A": (14, 17, 17, 31, 17, 17, 17), "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14), "D": (30, 17, 17, 17, 17, 17, 30),
    "E": (31, 16, 16, 30, 16, 16, 31), "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15), "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14), "J": (7, 2, 2, 2, 18, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17), "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17), "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14), "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13), "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30), "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14), "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 21, 10), "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4), "Z": (31, 1, 2, 4, 8, 16, 31),
    "0": (14, 17, 19, 21, 25, 17, 14), "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31), "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14), "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14), "9": (14, 17, 17, 15, 1, 1, 14),
    " ": (0, 0, 0, 0, 0, 0, 0), "-": (0, 0, 0, 31, 0, 0, 0),
    "_": (0, 0, 0, 0, 0, 0, 31), ".": (0, 0, 0, 0, 0, 12, 12),
    ":": (0, 12, 12, 0, 12, 12, 0), "/": (1, 2, 2, 4, 8, 8, 16),
    "(": (2, 4, 8, 8, 8, 4, 2), ")": (8, 4, 2, 2, 2, 4, 8),
    "+": (0, 4, 4, 31, 4, 4, 0), "?": (14, 17, 1, 2, 4, 0, 4),
    "=": (0, 0, 31, 0, 31, 0, 0),
}
COLORS = {
    "paper": (250, 248, 240), "ground": (218, 229, 204), "ink": (42, 48, 46),
    "river": (113, 172, 196), "road": (193, 181, 151), "dim": (180, 188, 190),
    "lane": (126, 140, 145), "fence": (121, 105, 75), "field": (194, 203, 128),
    "furrow": (148, 165, 86), "orchard": (180, 207, 153), "shopfront": (231, 169, 56),
    "bridge": (132, 143, 145), "ford": (180, 160, 127), "landing": (160, 127, 91),
    "lamp": (255, 202, 47), "door": (190, 52, 42), "window": (30, 165, 197),
    "container": (108, 66, 33), "garden": (119, 162, 91), "forest": (68, 120, 72),
    "quarry": (141, 145, 146), "house": (202, 169, 131), "town_hall": (178, 118, 103),
    "inn": (191, 142, 106), "bakery": (222, 183, 105), "mill": (184, 183, 132),
    "workshop": (155, 161, 178), "warehouse": (161, 152, 137), "school": (179, 158, 188),
    "chapel": (177, 168, 194), "tower": (124, 112, 146), "shop": (219, 193, 154),
}


class _Canvas:
    """A tiny RGB raster with deterministic PNG encoding and bitmap lettering."""

    def __init__(self, width: int, height: int):
        """Allocate an opaque paper-colored RGB image."""
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(COLORS["paper"]) * width * height)

    def rectangle(self, bounds: Sequence[int], color: tuple[int, int, int]) -> None:
        """Fill an inclusive rectangle, clipping safely to the canvas."""
        left, top, right, bottom = bounds
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width - 1, right), min(self.height - 1, bottom)
        if left > right or top > bottom:
            return
        row = bytes(color) * (right - left + 1)
        for vertical in range(top, bottom + 1):
            offset = (vertical * self.width + left) * 3
            self.pixels[offset:offset + len(row)] = row

    def outline(self, bounds: Sequence[int], color: tuple[int, int, int]) -> None:
        """Stroke a one-pixel rectangular outline."""
        left, top, right, bottom = bounds
        for edge in ((left, top, right, top), (left, bottom, right, bottom),
                     (left, top, left, bottom), (right, top, right, bottom)):
            self.rectangle(edge, color)

    def line(self, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        """Draw an integer line with no floating-point rounding differences."""
        start_x, start_y = start
        end_x, end_y = end
        delta_x, delta_y = abs(end_x - start_x), -abs(end_y - start_y)
        step_x, step_y = (1 if start_x < end_x else -1), (1 if start_y < end_y else -1)
        error = delta_x + delta_y
        while True:
            self.rectangle((start_x, start_y, start_x, start_y), color)
            if start_x == end_x and start_y == end_y:
                return
            double = 2 * error
            if double >= delta_y:
                error += delta_y
                start_x += step_x
            if double <= delta_x:
                error += delta_x
                start_y += step_y

    def text(self, left: int, top: int, text: str, *, scale: int = 1) -> None:
        """Render all place-name characters without loading a system font."""
        for index, character in enumerate(text.upper()):
            for vertical, bits in enumerate(FONT.get(character, FONT["?"])):
                for horizontal in range(5):
                    if bits & (1 << (4 - horizontal)):
                        pixel_x, pixel_y = left + (index * 6 + horizontal) * scale, top + vertical * scale
                        self.rectangle((pixel_x, pixel_y, pixel_x + scale - 1, pixel_y + scale - 1), COLORS["ink"])

    def write(self, path: Path) -> Path:
        """Write RGB PNG chunks with no timestamps or nondeterministic metadata."""
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

        stride = self.width * 3
        scanlines = b"".join(b"\0" + self.pixels[offset:offset + stride]
                             for offset in range(0, len(self.pixels), stride))
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines, 9))
                         + chunk(b"IEND", b""))
        return path


def _value(plan: object, key: str, default=None):
    """Accept a live plan, a Place dataclass or a manifest dictionary."""
    return plan.get(key, default) if isinstance(plan, Mapping) else getattr(plan, key, default)


def _overlaps(bounds: Sequence[int], other: Sequence[int], padding: int = 2) -> bool:
    """Keep labels and ground-floor symbols separated by a small clear gutter."""
    return not (bounds[2] + padding < other[0] or bounds[0] - padding > other[2] or
                bounds[3] + padding < other[1] or bounds[1] - padding > other[3])


def _badge_box(anchor: tuple[int, int], width: int, map_bounds: Sequence[int],
               occupied: list[Sequence[int]]) -> tuple[int, int, int, int]:
    """Find a nonoverlapping numbered callout, expanding beyond a crowded frontage."""
    anchor_x, anchor_y = anchor
    left, top, right, bottom = map_bounds
    candidates = [(anchor_x - width // 2, anchor_y - 5)]
    for distance in (8, 16, 24, 40, 64, 96):
        candidates.extend((anchor_x - width // 2 + offset_x, anchor_y - 5 + offset_y)
                          for offset_x, offset_y in ((0, -distance), (0, distance), (-distance, 0),
                                                     (distance, 0), (-distance, -distance), (distance, -distance),
                                                     (-distance, distance), (distance, distance)))

    def available():
        """Try nearby sites first, then the entire map rather than cover another label."""
        yield from candidates
        for pixel_y in range(top + 3, bottom - 10, 14):
            for pixel_x in range(left + 3, right - width, width + 4):
                yield pixel_x, pixel_y

    for pixel_x, pixel_y in available():
        box = (pixel_x, pixel_y, pixel_x + width, pixel_y + 10)
        if (left + 2 <= box[0] and top + 2 <= box[1] and box[2] < right - 2 and box[3] < bottom - 2
                and not any(_overlaps(box, other) for other in occupied)):
            return box
    raise ValueError("Town map has no clear space for a place callout")


def render_map(plan: object, path: Path | str) -> Path:
    """Draw dense frontages and every marker, with full place names in keyed margins."""
    layout = _value(plan, "layout", {})
    region = layout.get("region", {"x0": -160, "z0": -160, "x1": 159, "z1": 159})
    minimum_x, minimum_z, maximum_x, maximum_z = (region[key] for key in ("x0", "z0", "x1", "z1"))
    scale, margin, header = 4, 36, 80
    map_width, map_height = (maximum_x - minimum_x + 1) * scale, (maximum_z - minimum_z + 1) * scale
    buildings = list(_value(plan, "buildings", layout.get("buildings", [])))
    places = list(_value(plan, "places", []))
    banks = {"west": [], "east": []}
    for place in places:
        position = _value(place, "feet", _value(place, "pos"))
        bank = _value(place, "bank")
        if bank not in banks:
            bank = "west" if position[0] < (minimum_x + maximum_x) / 2 else "east"
        banks[bank].append((position, _value(place, "name")))
    for entries in banks.values():
        entries.sort(key=lambda entry: (entry[0][2], entry[0][0], entry[1]))
    label_width = max((len(name.replace("_", " ")) * 12 for entries in banks.values()
                       for _, name in entries), default=120)
    rail_width = max(264, label_width + len(str(max(1, len(places)))) * 12 + 32)
    rail_height = 52 + max(map(len, banks.values())) * 24
    body_height = max(map_height, rail_height)
    legend = [("river", "RIVER"), ("road", "LIT MAIN STREETS"), ("dim", "DIM LANES / SIDE STREETS"),
              ("bridge", "STONE BRIDGE"), ("ford", "UNLIT FORD"), ("landing", "FERRY LANDINGS"),
              ("garden", "GARDENS / BACK YARDS"), ("field", "FENCED FIELDS"), ("orchard", "FENCED ORCHARDS"),
              ("forest", "FOREST / QUAY TREES"), ("quarry", "QUARRY")]
    legend.extend((kind if kind in COLORS else "house", kind.replace("_", " ").upper())
                  for kind in sorted({building["kind"] for building in buildings}))
    legend.extend((("lamp", "LANTERN POST"), ("door", "OPEN DOOR"), ("window", "GLASS WINDOW"),
                   ("container", "CONTAINER / STALL"), ("shopfront", "SHOPFRONT FACADE"), ("tower", "CHAPEL TOWER")))
    legend_rows = (len(legend) + 3) // 4
    canvas = _Canvas(map_width + margin * 2 + rail_width * 2,
                     header + margin + body_height + margin + 108 + legend_rows * 24)
    map_left, map_top = rail_width + margin, header + margin
    map_bounds = (map_left, map_top, map_left + map_width - 1, map_top + map_height - 1)

    def point(horizontal: float, depth: float) -> tuple[int, int]:
        return map_left + int((horizontal - minimum_x) * scale), map_top + int((depth - minimum_z) * scale)

    def rectangle(bounds: Sequence[int], color: tuple[int, int, int], *, outline: bool = False) -> tuple[int, ...]:
        """Project either a ground rectangle or a six-coordinate block volume."""
        left, north, right, south = bounds if len(bounds) == 4 else (bounds[0], bounds[2], bounds[3], bounds[5])
        pixel_left, pixel_top = point(left, north)
        pixel_right, pixel_bottom = point(right + 1, south + 1)
        pixels = (pixel_left, pixel_top, pixel_right - 1, pixel_bottom - 1)
        canvas.rectangle(pixels, color)
        if outline:
            canvas.outline(pixels, COLORS["ink"])
        return pixels

    def fenced(bounds: Sequence[int], color: str) -> tuple[int, ...]:
        """Show cultivated plots with fences and regularly spaced fence posts."""
        pixels = rectangle(bounds, COLORS[color])
        left, top, right, bottom = pixels
        canvas.outline(pixels, COLORS["fence"])
        for pixel_x in range(left, right + 1, 4 * scale):
            for pixel_y in (top, bottom):
                canvas.rectangle((pixel_x - 1, pixel_y - 1, pixel_x + 1, pixel_y + 1), COLORS["fence"])
        for pixel_y in range(top, bottom + 1, 4 * scale):
            for pixel_x in (left, right):
                canvas.rectangle((pixel_x - 1, pixel_y - 1, pixel_x + 1, pixel_y + 1), COLORS["fence"])
        return pixels

    def tree(position: Sequence[int]) -> None:
        """Draw a small crown and trunk, keeping individual quay trees visible."""
        pixel_x, pixel_y = point(position[0] + 0.5, position[2] + 0.5)
        radius = 2 * scale
        canvas.rectangle((pixel_x - radius, pixel_y - radius + 3, pixel_x + radius, pixel_y + radius - 3),
                         COLORS["forest"])
        canvas.rectangle((pixel_x - radius + 3, pixel_y - radius, pixel_x + radius - 3, pixel_y + radius),
                         COLORS["forest"])
        canvas.rectangle((pixel_x - 1, pixel_y - 1, pixel_x + 1, pixel_y + 1), COLORS["fence"])

    canvas.text(margin, 18, "THE JUBILEE - RIVERSIDE TOWN", scale=2)
    canvas.text(margin, 43, f"PLAN ONLY / SEED {_value(plan, 'seed', '?')} / NORTH UP / 1 BLOCK = {scale} PIXELS")
    canvas.text(margin, 59, f"{len(buildings)} BUILDINGS / {len(places)} NAMED PLACES / FULL NAMES IN NUMBERED MARGINS")
    rectangle((minimum_x, minimum_z, maximum_x, maximum_z), COLORS["ground"])
    for garden in layout.get("gardens", []):
        rectangle(garden["bounds"], COLORS["garden"])
    for field in layout.get("fields", []):
        left, top, right, bottom = fenced(field["bounds"], "field")
        for pixel_y in range(top + scale, bottom, scale * 3):
            canvas.line((left + 2, pixel_y), (right - 2, pixel_y), COLORS["furrow"])
    for orchard in layout.get("orchards", []):
        fenced(orchard["bounds"], "orchard")
        for position in orchard.get("trees", []):
            tree(position)
    for position in layout.get("forest", {}).get("trees", []):
        tree(position)
    if "river" in layout:
        rectangle(layout["river"]["bounds"], COLORS["river"])
    for road in sorted(_value(plan, "roads", []), key=lambda road: not road.get("dim", True)):
        left, top, right, bottom = rectangle(road["bounds"], COLORS["dim" if road.get("dim", True) else "road"])
        if road.get("dim", True):
            if right - left >= bottom - top:
                for pixel_x in range(left + 2, right - 1, 12):
                    canvas.line((pixel_x, (top + bottom) // 2), (min(pixel_x + 4, right), (top + bottom) // 2),
                                COLORS["lane"])
            else:
                for pixel_y in range(top + 2, bottom - 1, 12):
                    canvas.line(((left + right) // 2, pixel_y), ((left + right) // 2, min(pixel_y + 4, bottom)),
                                COLORS["lane"])
    for name in ("plaza", "east_square"):
        if name in layout:
            rectangle(layout[name]["bounds"], COLORS["road"])
    for name, color in (("bridge", "bridge"), ("ford", "ford"), ("quarry", "quarry")):
        if name in layout:
            rectangle(layout[name]["bounds"], COLORS[color], outline=True)
    for platform in layout.get("port", {}).get("platforms", []):
        rectangle(platform, COLORS["landing"], outline=True)
    for position in layout.get("quay_trees", []):
        tree(position)
    roof_ridges = []
    for building in buildings:
        left, top, right, bottom = rectangle(building["bounds"], COLORS.get(building["kind"], COLORS["house"]),
                                              outline=True)
        if "storeys" in building:
            horizontal = building.get("facade", "south") in ("north", "south")
            for offset in ((0, 3) if building["storeys"] == 3 else (0,)):
                start = ((left + 3, (top + bottom) // 2 + offset) if horizontal else
                         ((left + right) // 2 + offset, top + 3))
                end = (right - 3, start[1]) if horizontal else (start[0], bottom - 3)
                canvas.line(start, end, COLORS["fence"])
                roof_ridges.append((*start, *end))
        if building.get("shopfront"):
            facades = {"north": (left + 1, top + 1, right - 1, top + 2),
                       "south": (left + 1, bottom - 2, right - 1, bottom - 1),
                       "west": (left + 1, top + 1, left + 2, bottom - 1),
                       "east": (right - 2, top + 1, right - 1, bottom - 1)}
            canvas.rectangle(facades.get(building.get("facade"), facades["south"]), COLORS["shopfront"])
        for room in building.get("rooms", {}).values():
            left, north, right, south = room["bounds"]
            start, end = point(left, north), point(right + 1, south + 1)
            canvas.outline((*start, end[0] - 1, end[1] - 1), COLORS["ink"])
    tower = layout.get("chapel", {}).get("tower")
    if tower:
        left, top, right, bottom = rectangle(tower, COLORS["tower"], outline=True)
        centre_x, centre_y = (left + right) // 2, (top + bottom) // 2
        canvas.line((centre_x, top + 3), (centre_x, bottom - 3), COLORS["paper"])
        canvas.line((left + 3, centre_y), (right - 3, centre_y), COLORS["paper"])
    if "fountain" in layout:
        rectangle(layout["fountain"]["bounds"], COLORS["river"], outline=True)
    markers = []

    def marker(position: Sequence[int], color: str, radius: int = 2) -> None:
        """Reserve important symbols before positioning any numbered labels."""
        pixel_x, pixel_y = point(position[0] + 0.5, position[2] + 0.5)
        markers.append(((pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius), color))

    for window in layout.get("windows", []):
        marker(window["pos"], "window")
    doors = layout.get("door_details", [])
    if not doors:
        doors = [{"pos": pos} for positions in layout.get("doors", {}).values() for pos in positions]
    for door in doors:
        marker(door["pos"], "door")
    for container in _value(plan, "containers", {}).values():
        marker(container["pos"], "container")
    for position in layout.get("lamps", []):
        marker(position, "ink", radius=3)
        marker(position, "lamp")
    occupied = [bounds for bounds, _ in markers] + roof_ridges
    if tower:
        start, end = point(tower[0], tower[2]), point(tower[3] + 1, tower[5] + 1)
        occupied.append((*start, end[0] - 1, end[1] - 1))
    numbered = []
    number = 0
    building_bounds = {building["name"]: building["bounds"] for building in buildings}
    for bank, entries in banks.items():
        rail_left = margin // 2 if bank == "west" else map_left + map_width + margin
        canvas.text(rail_left, map_top, f"{bank.upper()} BANK", scale=2)
        canvas.text(rail_left, map_top + 24, "NUMBERED PLACES / NORTH TO SOUTH")
        for row, (position, name) in enumerate(entries):
            number += 1
            label_top = map_top + 48 + row * 24
            canvas.text(rail_left, label_top, str(number), scale=2)
            canvas.text(rail_left + len(str(max(1, len(places)))) * 12 + 12, label_top,
                        name.replace("_", " "), scale=2)
            anchor = point(position[0] + 0.5, position[2] + 0.5)
            if name in building_bounds:
                left, north, right, south = building_bounds[name]
                anchor = point((left + right + 1) / 2, (north + south + 1) / 2)
            box = _badge_box(anchor, len(str(number)) * 6 + 3, map_bounds, occupied)
            occupied.append(box)
            numbered.append((position, str(number), box))
    for position, number, box in numbered:
        pixel_x, pixel_y = point(position[0] + 0.5, position[2] + 0.5)
        canvas.line((pixel_x, pixel_y), (max(box[0], min(pixel_x, box[2])),
                                       max(box[1], min(pixel_y, box[3]))), COLORS["ink"])
    for _, number, box in numbered:
        canvas.rectangle(box, COLORS["paper"])
        canvas.outline(box, COLORS["fence"])
        canvas.text(box[0] + 2, box[1] + 2, number)
    for bounds, color in markers:
        canvas.rectangle(bounds, COLORS[color])
    border_start, border_end = point(minimum_x, minimum_z), point(maximum_x + 1, maximum_z + 1)
    canvas.outline((*border_start, border_end[0] - 1, border_end[1] - 1), COLORS["ink"])
    legend_left, legend_top = margin, map_top + body_height + margin
    canvas.text(legend_left, legend_top, "KEY", scale=2)
    legend_top += 30
    column_width = (canvas.width - margin * 2) // 4
    for index, (color, label) in enumerate(legend):
        left, top = legend_left + (index % 4) * column_width, legend_top + (index // 4) * 24
        canvas.rectangle((left, top, left + 12, top + 12), COLORS[color])
        canvas.outline((left, top, left + 12, top + 12), COLORS["ink"])
        canvas.text(left + 20, top, label, scale=2)
    legend_top += legend_rows * 24 + 12
    canvas.text(legend_left, legend_top,
                "ROOF RIDGES: ONE = 2 STOREYS / TWO = 3 STOREYS. DOORS AND WINDOWS: GROUND FLOOR.")
    canvas.text(legend_left, legend_top + 14, "UPPER FLOORS SEALED. DASHED ROADS ARE DIM. FERRY IS NOT A ROAD.")
    canvas.line((canvas.width - margin - 20 * scale, legend_top + 6),
                (canvas.width - margin, legend_top + 6), COLORS["ink"])
    canvas.text(canvas.width - margin - 20 * scale, legend_top + 16, "20 BLOCKS")
    for coordinate in range((minimum_x + 39) // 40 * 40, maximum_x + 1, 40):
        pixel_x, _ = point(coordinate, minimum_z)
        canvas.text(pixel_x + 1, map_top + map_height + 9, str(coordinate))
    for coordinate in range((minimum_z + 39) // 40 * 40, maximum_z + 1, 40):
        _, pixel_y = point(minimum_x, coordinate)
        canvas.text(map_left - 30, pixel_y + 1, str(coordinate))
    return canvas.write(Path(path))


def main(argv: list[str] | None = None) -> int:
    """Render a saved manifest or construct a plan without starting a server."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "scenario/town_jubilee.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--content", type=Path, help="construct a plan from this scenario content instead")
    parser.add_argument("--seed", type=int, help="construct a plan with this deterministic seed")
    args = parser.parse_args(argv)
    if args.content is not None or args.seed is not None or not args.manifest.exists():
        if __package__:
            from .town_plan import TownPlan
        else:
            from town_plan import TownPlan
        plan = TownPlan.create(seed=args.seed, content_path=args.content)
    else:
        plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    path = render_map(plan, args.output or args.manifest.with_suffix(".png"))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
