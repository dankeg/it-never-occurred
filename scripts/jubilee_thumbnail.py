"""The submission thumbnail: one planned episode drawn on the town plan ($0, offline).

The plan is the scenario's own map with the place labels left out. On it: every stretch a committee member walked
carrying a visible sack of bread (orange), the mayor's own path (blue), and each moment the mayor saw a sack, as a
ring on the carrier joined to where the mayor stood. The mayor's question to the baker on day 1 is quoted as said.
Everything is read from the episode's logs, up to 18:00 on day 1.

Usage: `python3.11 scripts/jubilee_thumbnail.py [--run jub_L2_1] [--out docs/report/thumbnail.png]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

from scripts.town_map import render_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAST = 36_000
BREAD, MAYOR, INK, MUTED, PAPER = "#e0662a", "#1f5fbf", "#16130f", "#5c564c", "#fbf8f1"
VIEW = (-112, 62, -6, 92)  # world x0, x1, z0, z1 in blocks: the old town, the bridge and the bakery
QUESTION = "Who is the feast celebrating, and who arranged it?"
def old_standard() -> Path:
    """The folder holding the Old Standard fonts that TeX Live ships (the report's typeface)."""
    found = subprocess.run(["kpsewhich", "OldStandard-Regular.otf"], capture_output=True, text=True).stdout.strip()
    if not found:
        raise SystemExit("Old Standard not found: install TeX Live (the report build needs it too)")
    return Path(found).parent


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def plan_image() -> Image.Image:
    """The scenario's plan without place labels, cropped to the view (4 pixels per block)."""
    plan = json.loads((ROOT / "scenario" / "town_jubilee.json").read_text(encoding="utf-8"))
    plan["places"] = []
    with tempfile.TemporaryDirectory() as folder:
        image = Image.open(render_map(plan, Path(folder) / "plan.png")).convert("RGB")
        image.load()
    x0, x1, z0, z1 = VIEW
    # with no place labels the plan maps world (x, z) to pixel (300 + 4 (x + 160), 116 + 4 (z + 160))
    image = image.crop((300 + 4 * (x0 + 160), 116 + 4 * (z0 + 160), 300 + 4 * (x1 + 160), 116 + 4 * (z1 + 160)))
    muted = ImageEnhance.Color(image).enhance(0.45)
    return Image.blend(muted, Image.new("RGB", muted.size, (255, 255, 255)), 0.3)


def bread_runs(trajectory: list[dict], members: set[str]) -> list[list[tuple[float, float]]]:
    """Each stretch a member walked with a visible sack of bread (nine loaves or more), as world (x, z) points."""
    runs, current = [], {}
    for record in sorted(trajectory, key=lambda row: (row["agent"], row["t"])):
        agent = record["agent"]
        if agent not in members or record["t"] >= FEAST:
            continue
        loaded = (record.get("carrying") or {}).get("bread", 0) >= 9
        if loaded:
            current.setdefault(agent, []).append((record["pos"][0], record["pos"][2]))
        elif agent in current:
            runs.append(current.pop(agent))
    return runs + list(current.values())


def position(trajectory: list[dict], agent: str, tick: int) -> tuple[float, float]:
    """An agent's (x, z) at the record nearest a tick."""
    record = min((row for row in trajectory if row["agent"] == agent), key=lambda row: abs(row["t"] - tick))
    return record["pos"][0], record["pos"][2]


def draw(run: Path, out: Path) -> None:
    """Compose the thumbnail at 1600 by 900 pixels."""
    fonts = old_standard()
    for face in ("Regular", "Bold", "Italic"):
        font_manager.fontManager.addfont(str(fonts / f"OldStandard-{face}.otf"))
    manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    members = set(manifest["jubilee"]["committee"]["members"])
    trajectory = lines(run / "orchestrator" / "trajectory.jsonl")
    leaks = json.loads((run / "jubilee_leaks.json").read_text(encoding="utf-8"))
    sightings = [event for event in leaks["evidence"] if event["agent"] == "mayor" and event["channel"] == "sack"
                 and event["t"] < FEAST]
    asked = next(row["game_time"] for row in lines(run / "orchestrator" / "decisions.jsonl")
                 if row["agent"] == "mayor" and QUESTION in str((row.get("args") or {}).get("text", "")))

    x0, x1, z0, z1 = VIEW
    figure = plt.figure(figsize=(16, 9), dpi=100)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(plan_image(), extent=(x0, x1, z1, z0), interpolation="nearest")
    halo = [patheffects.withStroke(linewidth=4, foreground="white", alpha=0.9)]
    path = [(row["pos"][0], row["pos"][2]) for row in trajectory if row["agent"] == "mayor" and row["t"] < FEAST]
    axis.plot(*zip(*path), color=MAYOR, lw=1.8, alpha=0.7, zorder=3, path_effects=halo)
    for points in bread_runs(trajectory, members):
        axis.plot(*zip(*points), color=BREAD, lw=6, alpha=0.95, solid_capstyle="round", zorder=4,
                  path_effects=[patheffects.withStroke(linewidth=10, foreground="white", alpha=0.9)])
    for event in sightings:
        carrier = [float(value) for value in event["detail"].removeprefix("at [").rstrip("]").split(",")]
        seen, stood = (carrier[0], carrier[2]), position(trajectory, "mayor", event["t"])
        axis.plot(*zip(stood, seen), color=MAYOR, lw=2, ls=(0, (2, 2)), zorder=5)
        axis.scatter(*seen, s=420, facecolors="white", edgecolors=MAYOR, linewidths=3.5, zorder=6)
    for label, (x, z) in (("great hall", (-57, 23)), ("bridge", (0, 48)), ("bakery", (38, 54))):
        axis.text(x, z, label, ha="center", va="center", fontsize=22, fontstyle="italic", family="Old Standard",
                  color=INK, zorder=7, path_effects=[patheffects.withStroke(linewidth=6, foreground="white")])
    where = position(trajectory, "mayor", asked)
    axis.scatter(*where, s=200, color=MAYOR, edgecolors="white", linewidths=2.5, zorder=8)
    quote = VPacker(children=[
        TextArea("“Who is the feast celebrating,\nand who arranged it?”",
                 textprops={"fontsize": 27, "fontstyle": "italic", "family": "Old Standard", "color": INK,
                            "linespacing": 1.15, "ha": "center"}),
        TextArea("the mayor, to the baker, on the morning of the feast",
                 textprops={"fontsize": 16, "family": "Old Standard", "color": MUTED})], align="center", pad=0, sep=8)
    axis.add_artist(AnnotationBbox(quote, where, xybox=(14, 6), boxcoords="data", zorder=9, pad=1.0,
                                   bboxprops={"boxstyle": "round,pad=0.6", "fc": PAPER, "ec": MAYOR, "lw": 2.5},
                                   arrowprops={"arrowstyle": "-", "color": MAYOR, "lw": 2.5, "shrinkB": 9}))
    axis.set_xlim(x0, x1)
    axis.set_ylim(z1, z0)
    axis.axis("off")

    band = figure.add_axes((0, 0, 1, 0.235))
    band.set_facecolor(INK)
    band.patch.set_alpha(0.9)
    band.set_xticks([])
    band.set_yticks([])
    for spine in band.spines.values():
        spine.set_visible(False)
    band.text(0.035, 0.6, "It Never Occurred to the Mayor", fontsize=54, fontweight="bold", family="Old Standard",
              color=PAPER, va="center")
    band.text(0.036, 0.24, "Collusion and Oversight among LLM Agents in a Physical Town", fontsize=25,
              family="Old Standard", color="#d9d2c3", va="center")
    for y, colour, label, style in ((0.7, BREAD, "committee carrying bread", "line"),
                                    (0.47, MAYOR, "the mayor's path", "line"),
                                    (0.24, MAYOR, "the mayor sees a sack", "ring")):
        if style == "line":
            band.plot([0.745, 0.785], [y, y], color=colour, lw=5, transform=band.transAxes, solid_capstyle="round")
        else:
            band.plot(0.765, y, "o", ms=15, mfc="white", mec=colour, mew=3, transform=band.transAxes)
        band.text(0.8, y, label, fontsize=19, family="Old Standard", color=PAPER, va="center",
                  transform=band.transAxes)
    band.set_xlim(0, 1)
    band.set_ylim(0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=100, facecolor=PAPER)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Draw the thumbnail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="jub_L2_1")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "report" / "thumbnail.png")
    args = parser.parse_args(argv)
    draw(ROOT / "runs" / args.run, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
