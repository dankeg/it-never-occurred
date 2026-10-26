"""Render a built world from above as a PNG: the highest block in every column, shaded by its height ($0).

It reads a disposable clone of a fixture through the harness's read-only `blocks` query (so the fixture is never
touched) and colours each column by its top block, darker where lower, with lamps marked. It shows what was really
built, unlike a plan diagram, so it is how the town is reviewed.

Usage: `python3.11 scripts/render_town.py --fixture town_jubilee --out runs/town_jubilee/town.png [--scale 3]`
"""

from __future__ import annotations

import json
import shutil
import zlib

if __package__:
    from . import harness
else:
    import harness

BATCH = 4_096
COLOURS = {  # block id (without properties) -> RGB; anything else gets a stable colour from its name
    "minecraft:water": (52, 92, 196), "minecraft:grass_block": (106, 152, 70), "minecraft:dirt": (134, 96, 67),
    "minecraft:dirt_path": (148, 122, 65), "minecraft:gravel": (136, 126, 126), "minecraft:stone": (125, 125, 125),
    "minecraft:stone_bricks": (122, 121, 122), "minecraft:polished_andesite": (132, 135, 134),
    "minecraft:cobblestone": (110, 110, 110), "minecraft:oak_planks": (162, 131, 79),
    "minecraft:spruce_planks": (115, 85, 49), "minecraft:oak_log": (109, 85, 51), "minecraft:oak_leaves": (60, 110, 40),
    "minecraft:bricks": (150, 74, 58), "minecraft:white_concrete": (207, 213, 214),
    "minecraft:calcite": (223, 224, 220),
    "minecraft:lantern": (255, 214, 92), "minecraft:sand": (219, 207, 163), "minecraft:glass_pane": (190, 220, 230),
}
LIGHT = (255, 214, 92)
DYES = {"light_blue": (58, 175, 217), "light_gray": (142, 142, 134), "white": (233, 236, 236),
        "orange": (240, 118, 19), "magenta": (189, 68, 179), "yellow": (248, 197, 39), "lime": (112, 185, 25),
        "pink": (237, 141, 172), "gray": (62, 68, 71), "cyan": (21, 137, 145), "purple": (121, 42, 172),
        "blue": (53, 57, 157), "brown": (114, 71, 40), "green": (84, 109, 27), "red": (161, 39, 34),
        "black": (20, 21, 25)}
MATERIALS = (("dark_oak", (66, 43, 20)), ("spruce", (115, 85, 49)), ("birch", (196, 179, 123)),
             ("oak", (162, 131, 79)), ("stone_brick", (122, 121, 122)), ("brick", (150, 74, 58)),
             ("cobble", (110, 110, 110)), ("deepslate", (80, 80, 82)), ("andesite", (132, 135, 134)),
             ("calcite", (223, 224, 220)), ("glass", (190, 220, 230)), ("leaves", (60, 110, 40)),
             ("grass", (106, 152, 70)), ("farmland", (115, 75, 45)), ("wheat", (200, 180, 80)), ("hay", (200, 170, 50)),
             ("path", (148, 122, 65)), ("fence", (162, 131, 79)), ("stone", (125, 125, 125)), ("log", (109, 85, 51)))


class TownRender:
    """Top blocks of a square region, read layer by layer from the top down."""

    def __init__(self, bounds: tuple[int, int, int, int], top: int, bottom: int):
        """Columns x0..x1, z0..z1; layers from `top` down to `bottom` (the first non-air block wins)."""
        self._bounds, self._top, self._bottom = bounds, top, bottom

    def columns(self, client) -> dict[tuple[int, int], tuple[str, int]]:
        """Map each column to its top block id and height."""
        x0, z0, x1, z1 = self._bounds
        pending = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        found: dict[tuple[int, int], tuple[str, int]] = {}
        for y in range(self._top, self._bottom - 1, -1):
            remaining = []
            for start in range(0, len(pending), BATCH):
                batch = pending[start:start + BATCH]
                blocks = client.request("blocks", {"positions": [[x, y, z] for x, z in batch]})["blocks"]
                for column, block in zip(batch, blocks):
                    block = block.split("[")[0]
                    if block in ("minecraft:air", "minecraft:cave_air", "minecraft:void_air"):
                        remaining.append(column)
                    else:
                        found[column] = (block, y)
            pending = remaining
            if not pending:
                break
        return found

    @staticmethod
    def colour(block: str) -> tuple[int, int, int]:
        """A block's colour: the palette's, or a stable muted colour derived from its id."""
        if block in COLOURS:
            return COLOURS[block]
        name = block.removeprefix("minecraft:")
        for dye in sorted(DYES, key=len, reverse=True):  # light_blue before blue
            if name.startswith(dye + "_"):
                return DYES[dye]
        for material, colour in MATERIALS:
            if material in name:
                return colour
        value = zlib.crc32(block.encode())
        return (90 + value % 120, 90 + (value >> 8) % 120, 90 + (value >> 16) % 120)

    def image(self, columns: dict, scale: int):
        """The PNG image: one `scale`-pixel square per column, shaded by height, lamps drawn on top."""
        from PIL import Image  # optional dependency, only for rendering
        x0, z0, x1, z1 = self._bounds
        picture = Image.new("RGB", ((x1 - x0 + 1) * scale, (z1 - z0 + 1) * scale), (0, 0, 0))
        span = max(1, self._top - self._bottom)
        for (x, z), (block, y) in columns.items():
            shade = 0.55 + 0.45 * (y - self._bottom) / span
            colour = LIGHT if "lantern" in block or "torch" in block else tuple(
                min(255, int(channel * shade)) for channel in self.colour(block))
            for dx in range(scale):
                for dz in range(scale):
                    picture.putpixel(((x - x0) * scale + dx, (z - z0) * scale + dz), colour)
        return picture


def main(argv: list[str] | None = None) -> int:
    """Clone the fixture, read its top blocks, and write the PNG."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--fixture", default="town_jubilee")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--top", type=int, default=-40)
    parser.add_argument("--bottom", type=int, default=-64)
    args = parser.parse_args(argv)
    source = args.runs_dir / args.fixture
    manifest = json.loads((source / "scenario.json").read_text(encoding="utf-8"))
    x0, _, z0, x1, _, z1 = (manifest["layout"]["region"][key] for key in ("x0", "y0", "z0", "x1", "y1", "z1"))
    scratch = args.runs_dir / f"{args.fixture}_render"
    if scratch.exists():
        shutil.rmtree(scratch)
    harness.clone_apfs(source, scratch)
    try:
        with harness.Server(scratch, **harness.server_options(args)) as server:
            render = TownRender((x0, z0, x1, z1), args.top, args.bottom)
            render.image(render.columns(server.client), args.scale).save(args.out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
