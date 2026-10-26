"""The report's figures, from `runs/jubilee_analysis.json`, the probe rows and the episodes' judged leaks ($0, offline).

- `fig_town.png`: the town plan (`scenario/town_jubilee.png`) cropped to the built area, with the places the story
  turns on labelled;
- `fig_chain.png`: the same-model chain (the mayor's own model at every stage), share saying a preparation was under
  way on planned episodes and on control episodes, with Wilson 95% intervals;
- `fig_probe.png`: the in-context query through the day (median and middle half of planned and control episodes;
  Figure 4), the evidence reaching the mayor (ticks), and the live mayor's own judged probability at 18:00 (squares);
- `fig_drive.png`: the live mayor by drive level: aware (planned and control episodes) and its requests for news;
- `fig_channels.png`: how the plan reached the mayor with the physical rules on and off, by kind of evidence;
- `fig_readers.png`: every observer of the evidence (Experiment 1, its branches, Experiment 2): hits in planned
  episodes and false alarms in control episodes, with Wilson 95% intervals.

Colours are the first three slots of the validated reference palette (all-pairs safe); text stays neutral.

Usage: `python3.11 scripts/jubilee_figures.py [--out docs/report]`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import sys

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.jubilee_analysis import score, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN_COLOUR, NOPLAN_COLOUR, SPARE_COLOUR = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLAN_CELLS, NOPLAN_CELLS = {"A", "L1", "L2", "L3"}, {"D", "N3"}
def hour(tick: int) -> float:
    """Hours since 00:00 on day 0 (tick 0 is 06:00)."""
    return tick / 1000 + 6

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


TEXT_WIDTH = 6.7  # inches: the report's text width (Letter, 0.9-inch margins); figures drawn at it print 1:1


def save_at_size(figure, axis, path: Path, right: float, top: float, bottom: float) -> None:
    """Save at the figure's exact size, with the left margin fitted to the widest y label and the other margins given
    in inches, so that the figure prints at the size its fonts were set for."""
    figure.canvas.draw()
    width, height = figure.get_size_inches()
    labels = max(label.get_window_extent().width for label in axis.get_yticklabels()) / figure.dpi
    figure.subplots_adjust(left=(labels + 0.1) / width, right=1 - right / width, top=1 - top / height,
                           bottom=bottom / height)
    figure.savefig(path, bbox_inches=figure.bbox_inches)


def jsonl(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


PLACES = {  # letter -> (place, world x, z); the key is printed in the report's caption
    "A": ("town hall (great hall, office)", -56, 36), "B": ("inn", -31, 14), "C": ("plaza", -28, 38),
    "D": ("mayor's house", -99, 16), "E": ("bridge (lamp-lit)", 0, 39), "F": ("ferry", 0, 98),
    "G": ("ford (unlit)", 0, -148), "H": ("bakery", 37, 38), "I": ("mill", 29, 20),
    "J": ("neighbour's cottage", 16, 97), "K": ("warehouse (freight)", 50, 96), "L": ("builder's workshop", 65, 71)}


def town(out: Path) -> None:
    """The town plan cropped to the built area, with the story's places marked by letter (world x, z in blocks)."""
    image = Image.open(ROOT / "scenario" / "town_jubilee.png")
    # the plan maps world (x, z) to pixel (332 + 4 (x + 160), 116 + 4 (z + 160)) at 4 pixels per block
    x0, x1, z0, z1 = -118, 92, -154, 110
    crop = image.crop((332 + 4 * (x0 + 160), 116 + 4 * (z0 + 160), 332 + 4 * (x1 + 160), 116 + 4 * (z1 + 160)))
    figure, axis = plt.subplots(figsize=(4.4, 4.4 * (z1 - z0) / (x1 - x0)))
    axis.imshow(crop, extent=(x0, x1, z1, z0), interpolation="lanczos")
    for letter, (_, x, z) in PLACES.items():
        axis.text(x, z, letter, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white",
                  bbox={"boxstyle": "circle,pad=0.22", "fc": INK, "ec": "white", "lw": 0.8})
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(out / "fig_town.png")
    plt.close(figure)


def chain(analysis: dict, out: Path) -> None:
    """The same-model chain: share saying *preparing* (or aware) on planned and control episodes, Wilson intervals."""
    stages = [("R1 monitor", "Outside monitor,\nthe mayor's perceptions"),
              ("R2 auditor", "Outside auditor,\nthe mayor's prompts"),
              ("R2 as mayor", "Asked as the mayor,\nthe mayor's prompts"),
              ("probe", "The mayor's model, asked\nin its own context (last hour)"),
              ("live mayor, believed", "The mayor, live: believed\nsomething was kept from it"),
              ("live mayor", "The mayor, live: suspected\na surprise for itself")]
    stages = [(key, label) for key, label in stages if key in analysis["chain"]]
    figure, axis = plt.subplots(figsize=(6.4, 2.9))
    for index, (key, _) in enumerate(stages):
        entry = analysis["chain"][key]["all"]
        for offset, rate_key, interval_key, n_key, colour in ((-0.13, "hit", "hit_interval", "n_plan", PLAN_COLOUR),
                                                              (0.13, "fa", "fa_interval", "n_noplan", NOPLAN_COLOUR)):
            if entry[rate_key] is None:
                continue
            low, high = entry[interval_key]
            y = index + offset
            axis.plot([low, high], [y, y], color=colour, lw=2, solid_capstyle="round")
            axis.plot(entry[rate_key], y, "o", ms=6, color=colour, mec="white", mew=1.2)
            axis.text(1.02, y, f"{entry[rate_key]:.0%} (n={entry[n_key]})", va="center", fontsize=7, color=INK)
    axis.set_yticks(range(len(stages)), [label for _, label in stages], fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    axis.xaxis.grid(True, color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel("share saying a preparation was under way (live mayor: judged aware)")
    axis.plot([], [], "o", color=PLAN_COLOUR, label="plan days")
    axis.plot([], [], "o", color=NOPLAN_COLOUR, label="ordinary days (no plan)")
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5)
    figure.savefig(out / "fig_chain.png")
    plt.close(figure)


def clock_hour(clock: str) -> float:
    """Hours since 00:00 on day 0 from a clock string such as 'day 1, 07:38'."""
    day, time_of_day = clock.removeprefix("day ").split(", ")
    hours, minutes = time_of_day.split(":")
    return 24 * int(day) + int(hours) + int(minutes) / 60


def probe(analysis: dict, out: Path, model: str) -> None:
    """The mayor's model questioned in context through Experiment 1, on a time axis broken over the night (when the
    mayor made almost no decisions). Top: the probability it gave that a preparation was under way, at the first
    question of each hour, as the median and interquartile range of planned and of control episodes, with a tick where
    evidence of the plan first reached the mayor in each planned episode. Bottom: the share of episodes in which it
    had named the mayor as the target by each hour, against the mayor itself by 18:00 on day 1."""
    cells = {row["run"]: row["cell"] for row in analysis["episodes"]}
    hourly = defaultdict(dict)  # run -> hour since 00:00 on day 0 -> the score at that hour's first question
    for row in sorted(jsonl(ROOT / "runs" / f"jubilee_probe_{model}.jsonl"), key=lambda row: row["t"]):
        hourly[row["run"]].setdefault(int(hour(row["t"])), score(row))
    groups = ((PLAN_CELLS, PLAN_COLOUR, "planned episodes"), (NOPLAN_CELLS, NOPLAN_COLOUR, "control episodes"))
    days = ((0, range(8, 20), (7.6, 19.4), [8, 11, 14, 17]), (1, range(30, 42), (29.6, 42.2), [30, 33, 36, 39, 42]))
    figure = plt.figure(figsize=(TEXT_WIDTH, 3.9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.32, wspace=0.05, left=0.115, right=0.985,
                               top=0.9, bottom=0.11)
    episodes = analysis["probe"].get("episodes", {})
    firsts = {run: clock_hour(episodes[run]["probe_self_first"]) for run in episodes
              if episodes[run].get("probe_self_first")}
    evidence = [clock_hour(" ".join(row["first_mayor_leak"].split(" ")[:3]))  # 'day 0, 13:19 sack (freight)'
                for row in analysis["episodes"] if row["cell"] in PLAN_CELLS and row.get("first_mayor_leak")]
    plan_rows = [row for row in analysis["episodes"] if row["cell"] in PLAN_CELLS]
    suspected = sum(bool(row["mayor_aware"]) for row in plan_rows)
    for column, (day, bins, limits, ticks) in enumerate(days):
        top, bottom = figure.add_subplot(grid[0, column]), figure.add_subplot(grid[1, column])
        for cells_in, colour, label in groups:
            runs = [run for run in hourly if cells.get(run) in cells_in]
            spread = [(h, sorted(hourly[run][h] for run in runs if h in hourly[run])) for h in bins]
            spread = [(h, values) for h, values in spread if len(values) >= len(runs) / 2]
            quartile = lambda values, q: values[min(len(values) - 1, round(q * (len(values) - 1)))]  # noqa: E731
            xs = [h for h, _ in spread]
            top.fill_between(xs, [quartile(v, 0.25) for _, v in spread], [quartile(v, 0.75) for _, v in spread],
                             color=colour, alpha=0.18, lw=0)
            top.plot(xs, [quartile(v, 0.5) for _, v in spread], color=colour, lw=1.8, label=f"{label} ({len(runs)})")
            times = [bins[0] + step / 12 for step in range(round(12 * (limits[1] - bins[0])) + 1)]
            bottom.plot(times, [sum(firsts.get(run, 99) <= t for run in runs) / len(runs) for t in times],
                        drawstyle="steps-post", color=colour, lw=1.8, clip_on=False)
        seen = [h for h in evidence if limits[0] <= h <= limits[1]]
        top.plot(seen, [0.015] * len(seen), "|", ms=7, mew=1.1, color=INK, clip_on=False,
                 label="evidence first reaches the mayor" if column == 0 else None)
        if day == 1:
            bottom.plot(42, suspected / len(plan_rows), "D", ms=5, mfc="white", mec=INK, mew=1.1, clip_on=False)
            bottom.annotate(f"the mayor itself, by 18:00:\nsuspected a surprise for\nitself in {suspected} of "
                            f"{len(plan_rows)}", (42, suspected / len(plan_rows)), xytext=(-6, 12),
                            textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK)
        for axis in (top, bottom):
            axis.set_xlim(*limits)
            axis.set_xticks(ticks, [f"{h % 24:02d}:00" for h in ticks], fontsize=7)
            axis.yaxis.grid(True, color=GRID, lw=0.6)
            axis.set_axisbelow(True)
            axis.tick_params(axis="y", labelsize=7)
            if column:
                axis.spines["left"].set_visible(False)
                axis.tick_params(axis="y", length=0, labelleft=False)
        top.set_ylim(0, 1)
        bottom.set_ylim(-0.03, 1.03)
        bottom.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        top.set_title(f"Day {day}" + (" (the feast is at 18:00)" if day else ""), fontsize=7.5, color=INK, loc="left",
                      pad=3)
        if column == 0:
            top.set_ylabel("probability that a\npreparation is under way", fontsize=7.5)
            bottom.set_ylabel("share that has\nnamed the mayor", fontsize=7.5)
            top.text(-0.2, 1.08, "A", transform=top.transAxes, fontsize=9, color=INK, va="bottom")
            bottom.text(-0.2, 1.08, "B", transform=bottom.transAxes, fontsize=9, color=INK, va="bottom")
            handles, labels = top.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.11, 1.0), ncol=3, frameon=False, fontsize=7.5,
                  handlelength=1.6, columnspacing=1.8)
    figure.align_ylabels()
    figure.savefig(out / "fig_probe.png", bbox_inches=figure.bbox_inches)
    plt.close(figure)


def drive(analysis: dict, out: Path) -> None:
    """The live mayor by drive level: share aware on planned and control episodes, and requests for news per episode."""
    levels = {"L0": ("A", "D"), "L1": ("L1", None), "L2": ("L2", None), "L3": ("L3", "N3")}
    cells = analysis["cells"]
    figure, (aware, asks) = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for index, (level, (plan, noplan)) in enumerate(levels.items()):
        for offset, cell, colour in ((-0.1, plan, PLAN_COLOUR), (0.1, noplan, NOPLAN_COLOUR)):
            if cell not in cells:
                continue
            entry = cells[cell]["mayor_aware"]
            low, high = entry["interval"]
            aware.plot([index + offset] * 2, [low, high], color=colour, lw=2, solid_capstyle="round")
            aware.plot(index + offset, entry["share"], "o", ms=5.5, color=colour, mec="white", mew=1.1)
            requests = cells[cell]["requests"]
            low, high = requests["interval"]
            asks.plot([index + offset] * 2, [low, high], color=colour, lw=2, solid_capstyle="round")
            asks.plot(index + offset, requests["mean"], "o", ms=5.5, color=colour, mec="white", mew=1.1)
    for axis, label in ((aware, "mayor knew or suspected"), (asks, "mayor's requests for news")):
        axis.set_xticks(range(4), list(levels))
        axis.set_xlabel("the mayor's drive")
        axis.set_ylabel(label)
        axis.yaxis.grid(True, color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    aware.set_ylim(0, 1)
    aware.plot([], [], "o", color=PLAN_COLOUR, label="plan days")
    aware.plot([], [], "o", color=NOPLAN_COLOUR, label="ordinary days")
    aware.legend(loc="upper left", frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "fig_drive.png")
    plt.close(figure)


def readers(analysis: dict, v2: dict | None, out: Path) -> None:
    """Every observer of the evidence, one row each: hits in planned and false alarms in control episodes with Wilson
    intervals, grouped: Experiment 1 (Table 2), the branches at noon on day 1 (Table 4) and Experiment 2 (Table 5
    and its observers). Each branch arm is shown by the measure it moved (the self-questioning arm by belief, the
    others by a suspected surprise for itself)."""
    def counts(k_plan, n_plan, k_ordinary, n_ordinary):
        return (k_plan, n_plan), (k_ordinary, n_ordinary)

    def from_rates(entry):
        return counts(round(entry["hit"] * entry["n_plan"]), entry["n_plan"],
                      round(entry["fa"] * entry["n_noplan"]), entry["n_noplan"])

    def from_strings(entry):
        (k_plan, n_plan), (k_ordinary, n_ordinary) = (map(int, entry[key].split("/")) for key in ("plan", "ordinary"))
        return counts(k_plan, n_plan, k_ordinary, n_ordinary)

    def from_fork(name, measure):
        data = (analysis.get("forks") or {}).get(name) or {}
        plan, ordinary = data.get("plan"), data.get("noplan")
        if not plan or not ordinary or measure not in plan or measure not in ordinary:
            return None
        return counts(plan[measure]["asked"], plan["points"], ordinary[measure]["asked"], ordinary["points"])

    chain, self_named = analysis["chain"], analysis["probe"]["all"]["self"]
    groups = [("Experiment 1", [
        ("Monitor given only the mayor's perceptions", from_rates(chain["R1 monitor"]["all"])),
        ("Mayor's model, queried in context: plan under way", from_rates(chain["probe"]["all"])),
        ("... and names the mayor as its target", counts(self_named["plan"]["k"], self_named["plan"]["n"],
                                                     self_named["noplan"]["k"], self_named["noplan"]["n"])),
        ("The mayor itself: believed something was kept from it", from_rates(chain["live mayor, believed"]["all"])),
        ("The mayor itself: suspected a surprise for itself", from_rates(chain["live mayor"]["all"]))]),
        ("Experiment 1: branches at noon on day 1", [
            ("Told to ask itself: believed something was kept", from_fork("asked", "believed")),
            ("Told of its fortieth year: suspected a surprise", from_fork("jubilee", "aware")),
            ("Told it dislikes surprises: suspected a surprise", from_fork("candid", "aware")),
            ("Told it likes to know what concerns it: suspected a surprise", from_fork("self", "aware")),
            ("Made determined to find out: suspected a surprise", from_fork("drive", "aware")),
            ("Secretary obliged to report: suspected a surprise", from_fork("duty", "aware")),
            ("... and told of its fortieth year: suspected a surprise", from_fork("duty_jubilee", "aware"))])]
    if v2:
        town = v2["chain"]
        groups.append(("Experiment 2", [
            ("Monitor given only the mayor's perceptions", from_strings(town["monitor on the mayor's record"])),
            ("Mayor's model, queried in context: plan under way", from_strings(town["asked, last hour"])),
            ("The mayor itself: suspected a surprise for itself", from_strings(town["live mayor, aware"])),
            ("Asked the question each hour: suspected a surprise", from_fork("counsellor", "aware")),
            ("... believed something was kept from it", from_fork("counsellor", "believed"))]))
    rows = [(group, label, value) for group, entries in groups for label, value in entries if value is not None]
    pitch = 0.165  # inches per row
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH, pitch * (len(rows) + 1.4 * len(groups)) + 0.62))
    header = transforms.blended_transform_factory(figure.transFigure, axis.transData)
    y, ticks, labels = 0, [], []
    for group, _ in groups:
        members = [(label, value) for group_, label, value in rows if group_ == group]
        if not members:
            continue
        axis.text(0.004, y, group, transform=header, ha="left", va="center", fontsize=8.5, color=INK)
        y += 1
        for label, ((k_plan, n_plan), (k_ordinary, n_ordinary)) in members:
            for offset, k, n, colour in ((-0.2, k_plan, n_plan, PLAN_COLOUR),
                                         (0.2, k_ordinary, n_ordinary, NOPLAN_COLOUR)):
                low, high = wilson(k, n)
                axis.plot([low, high], [y + offset] * 2, color=colour, lw=1.4, solid_capstyle="round", alpha=0.55)
                axis.plot(k / n, y + offset, "o", ms=4.2, color=colour, mec="white", mew=0.8, clip_on=False)
            axis.text(1.03, y, f"{k_plan}/{n_plan} · {k_ordinary}/{n_ordinary}", va="center", fontsize=6.5,
                      color=INK, transform=axis.get_yaxis_transform())
            ticks.append(y)
            labels.append(label)
            y += 1
        y += 0.4
    axis.set_yticks(ticks, labels, fontsize=7.5, color=INK)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(y - 0.6, -0.8)
    axis.set_xlim(-0.015, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"], fontsize=7.5)
    axis.xaxis.grid(True, color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.spines["left"].set_visible(False)
    axis.set_xlabel("share of episodes", fontsize=7.5)
    axis.plot([], [], "o", color=PLAN_COLOUR, label="planned episodes")
    axis.plot([], [], "o", color=NOPLAN_COLOUR, label="control episodes (no plan)")
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5, borderaxespad=0.2)
    save_at_size(figure, axis, out / "fig_readers.png", right=0.62, top=0.24, bottom=0.38)
    plt.close(figure)


def channels(analysis: dict, out: Path) -> None:
    """How the plan reached the mayor with the physical rules on and off: mean pieces of evidence per planned episode,
    by kind (bread seen in transit, the hall seen being prepared, words heard or read)."""
    kinds = (("sacks of bread seen", ("sack",), PLAN_COLOUR), ("the hall seen being prepared",
                                                                ("hall_entry", "hall_change"), SPARE_COLOUR),
             ("words heard or read", ("speech", "read"), NOPLAN_COLOUR))
    groups = list(analysis["channels"].items())
    figure, axis = plt.subplots(figsize=(3.15, 2.45))
    height = 0.25
    for row, (group, entry) in enumerate(groups):
        for index, (label, keys, colour) in enumerate(kinds):
            value = sum(entry["per_episode"].get(key, 0) for key in keys)
            y = row + (index - 1) * (height + 0.04)
            axis.barh(y, value, height=height, color=colour, label=label if row == 0 else None)
            axis.text(value + 0.12, y, f"{value:.1f}", va="center", fontsize=6.5, color=INK)
    experiment, rules = zip(*(group.rsplit(", ", 1) for group, _ in groups))
    axis.set_yticks(range(len(groups)), [f"{name}\n{rule} ({entry['n']})" for name, rule, (_, entry)
                                         in zip(experiment, rules, groups)], fontsize=7, color=INK)
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=7)
    axis.spines["left"].set_visible(False)
    axis.xaxis.grid(True, color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel("pieces of evidence per planned episode", fontsize=7)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, fontsize=7, handlelength=1.0,
                borderaxespad=0.15, labelspacing=0.25)
    save_at_size(figure, axis, out / "fig_channels.png", right=0.08, top=0.5, bottom=0.36)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "report")
    parser.add_argument("--probe-model", default="gpt-6.1-sol")
    args = parser.parse_args(argv)
    analysis = json.loads((ROOT / "runs" / "jubilee_analysis.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    town(args.out)
    chain(analysis, args.out)
    probe(analysis, args.out, args.probe_model)
    drive(analysis, args.out)
    channels(analysis, args.out)
    v2_path = ROOT / "runs" / "jubilee_v2_analysis.json"
    readers(analysis, json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.exists() else None, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
