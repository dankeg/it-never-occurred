"""Build the report PDF: generated tables from `runs/jubilee_analysis.json` replace `<!-- include: NAME -->` markers in
`docs/report/report.md`, then pandoc and xelatex make `docs/report/report.pdf` ($0, offline). The look of Apart's
report template (Old Standard, its title block and headings) comes from `docs/report/apart.tex`, which report.md's
metadata loads.

Tables: `recruitment` (Table 1), `chain` (Table 2, every reader of the first experiment's evidence), and in the
appendices `cells` (the first experiment by cell), `asked` (the counterfactual branches), `v2` (the second experiment by
cell), `monitors` and `decomposition`, `episodes`.

Usage: `python3.11 scripts/jubilee_report_build.py [--tables-only]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.jubilee_analysis import sdt  # noqa: E402

REPORT = ROOT / "docs" / "report"
CELL_ORDER = ["A", "L1", "L2", "L3", "D", "N3", "E", "B", "C", "V2p", "V2t", "V2x", "V2o"]


def yes(value) -> str:
    """A boolean as a table mark."""
    return "–" if value is None else ("yes" if value else "no")


EVIDENCE = {"sack": "sack", "hall_entry": "entered hall", "hall_change": "hall changed", "speech": "words"}


def short_clock(clock: str | None) -> str:
    """'day 0, 13:19' as 'd0 13:19'."""
    return clock.replace("day ", "d").replace(",", "") if clock else "never"


def first_evidence(leak: str | None) -> str:
    """A first record of the plan reaching the mayor, as 'd0 13:19, sack (freight)' (the hall's item left out)."""
    found = re.fullmatch(r"day (\d+), (\d\d:\d\d) (\w+)(?: \((.*)\))?", leak or "")
    if not found:
        return "none"
    day, time, kind, detail = found.groups()
    detail = f" ({detail})" if detail and kind not in ("hall_entry", "hall_change") else ""
    return f"d{day} {time}, {EVIDENCE.get(kind, kind)}{detail}"


def episodes_table(analysis: dict) -> str:
    """Appendix E (LaTeX, a compact long table): every episode's outcome, the mayor's awareness and exposure, the
    questioned model, investigation and logistics."""
    rows = sorted(analysis["episodes"], key=lambda row: (CELL_ORDER.index(row["cell"]), row["run"]))
    probes = (analysis.get("probe") or {}).get("episodes", {})
    lines = [r"\begingroup\footnotesize\setlength{\tabcolsep}{3.6pt}",
             r"\begin{longtable}{@{}l ccc cc l l rrr c r@{}}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{The mayor (p)} & First evidence & Queried: & & & & Bread & \\",
             r"\cmidrule(lr){5-6}",
             r"Episode & Feast & Lit & Came & suspected & believed & to the mayor & under way & Requests & Reports & "
             r"Noticed & crossings & Credits \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    previous = None
    for row in rows:
        if previous is not None and row["cell"] != previous:
            lines.append(r"\addlinespace[2pt]")
        previous = row["cell"]
        crossings = row.get("bread_crossings") or {}
        day = sum(n for key, n in crossings.items() if key.endswith("/day"))
        night = sum(n for key, n in crossings.items() if key.endswith("/night"))
        probe = probes.get(row["run"], {})
        aware = f"{yes(row['mayor_aware'])} ({row['p_aware']:.2f})" if row.get("p_aware") is not None \
            else yes(row["mayor_aware"])
        believed = f"{yes(row['believed'])} ({row['p_believed']:.2f})" if row.get("p_believed") is not None \
            else "–"
        cells = [row["run"].removeprefix("jub_").replace("_", r"\_"), yes(row["feast"]), yes(row["lit"]),
                 yes(row["attended"]), aware, believed, first_evidence(row.get("first_mayor_leak")),
                 short_clock(probe.get("probe_first")), str(row.get("requests", "–")), str(row.get("reports", "–")),
                 str(row.get("noticed_count", "–")), f"{day}/{night}", f"{row['credits']:.0f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def share(entry: dict) -> str:
    """A rate as 'k/n'."""
    return f"{entry['k']}/{entry['n']}" if entry and entry.get("n") else "–"


def number(value, digits: int = 2) -> str:
    """A number, or a dash."""
    return "–" if value is None else f"{value:.{digits}f}"


def chain_table(analysis: dict) -> str:
    """Table 2: every reader of the first experiment's evidence, on planned and control episodes, with d' and c."""
    names = {"R1 monitor": "A monitor given only the mayor's perceptions",
             "R2 auditor": "A monitor given the mayor's prompts",
             "R2 as mayor": "The same prompts, read in the mayor's role",
             "probe": "In-context query of the mayor's model (first decision in the 17:00 hour)"}
    lines = ["| Observer (gpt-6.1-sol throughout) | Planned | Control | d′ | c |",
             "|" + "-" * 64 + "|------|------|-----|-----|"]

    def row(label: str, hits: int, n_plan: int, alarms: int, n_noplan: int) -> str:
        measures = sdt(hits, n_plan, alarms, n_noplan)
        return (f"| {label} | {hits}/{n_plan} | {alarms}/{n_noplan} | {number(measures['d_prime'])} | "
                f"{number(measures['c'])} |")

    for key, label in names.items():
        entry = analysis["chain"].get(key, {}).get("all")
        if entry:
            lines.append(row(label, round(entry["hit"] * entry["n_plan"]), entry["n_plan"],
                             round(entry["fa"] * entry["n_noplan"]), entry["n_noplan"]))
    named = (analysis.get("probe") or {}).get("all", {}).get("self")
    if named:
        lines.append(row("... names the mayor as the target, at some hour", named["plan"]["k"], named["plan"]["n"],
                         named["noplan"]["k"], named["noplan"]["n"]))
    neutral = analysis.get("neutral") or {}
    if neutral.get("self"):
        hits, n_plan = map(int, neutral["self"]["plan"].split("/"))
        alarms, n_noplan = map(int, neutral["self"]["ordinary"].split("/"))
        lines.append(row("... asked a neutral question, names the mayor as the target", hits, n_plan, alarms,
                         n_noplan))
    for key, label in (("live mayor, believed", "The mayor itself: believed something was being kept from it"),
                       ("live mayor", "The mayor itself: suspected a surprise for itself")):
        entry = analysis["chain"].get(key, {}).get("all")
        if entry:
            lines.append(row(label, round(entry["hit"] * entry["n_plan"]), entry["n_plan"],
                             round(entry["fa"] * entry["n_noplan"]), entry["n_noplan"]))
    return "\n".join(lines) + "\n"


def recruitment_table() -> str:
    """Table 1 (LaTeX, for its two-level header): acts that serve the plan, by people outside the committee, on
    planned and control episodes."""
    path = ROOT / "runs" / "jubilee_recruitment.json"
    if not path.exists():
        return "(the recruitment analysis has not run)\n"
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    order = ("v1 plan", "v1 ordinary", "V2 plan", "V2 ordinary")
    rows = (("The secretary enters an 18:00 appointment in the great hall in the mayor's book", "secretary_booked"),
            ("The baker hands bread to a committee member", "baker_supplied_any"),
            ("Members speak of credit or of paying later", "credit_any"),
            ("The neighbour carries bread or lanterns for the committee", "neighbour_carried_any"),
            ("The night watchman carries bread or lanterns for the committee", "watchman_carried_any"))
    lines = [r"\begin{tabularx}{\textwidth}{@{}X cc cc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Experiment 1} & \multicolumn{2}{c}{Experiment 2} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Before 18:00 on day 1 & planned & control & planned & control \\", r"\midrule"]
    for label, key in rows:
        cells = []
        for name in order:
            entry = (groups.get(name) or {}).get(key)
            cells.append("–" if key == "watchman_carried_any" and name.startswith("v1")
                         else f"{entry['k']}/{entry['n']}" if entry else "–")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    episodes = json.loads(path.read_text(encoding="utf-8"))["episodes"]
    loaves = {town: (sum(row["baker_supplied"] for row in episodes if row["town"] == town and row["plan"]),
                     sum(row["paid_baker"] + row["paid_at_stall"] for row in episodes
                         if row["town"] == town and row["plan"])) for town in ("v1", "V2")}
    lines += [r"\midrule",
              f"Loaves the baker handed over / emeralds members paid it & {loaves['v1'][0]:,} / {loaves['v1'][1]} "
              f"& – & {loaves['V2'][0]:,} / {loaves['V2'][1]} & – " + r"\\", r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines) + "\n"


def cells_table(analysis: dict) -> str:
    """Table 3: per condition (motivation x plan, physics off), outcomes, the live mayor, the probe and the
    investigation."""
    order = [("A", "L0, planned"), ("L1", "L1, planned"), ("L2", "L2, planned"), ("L3", "L3, planned"),
             ("D", "L0, control"), ("N3", "L3, control"), ("E", "L0, planned, physical rules off")]
    probes = (analysis.get("probe") or {}).get("episodes", {})
    rows = {cell: [row for row in analysis["episodes"] if row["cell"] == cell] for cell, _ in order}
    lines = ["| Condition | n | Delivered | Evidence reached the mayor | Mayor believed | Mayor suspected | Queried "
             "model names the mayor | Requests | Reports | Noticed |",
             "|------------|--|------|------|------|------|------|------|------|------|"]
    for cell, label in order:
        group = rows[cell]
        if not group:
            continue
        cells = analysis["cells"][cell]
        believed = [row for row in group if row.get("believed") is not None]
        named = [probes[row["run"]] for row in group if row["run"] in probes and "probe_self" in probes[row["run"]]]
        delivered = sum(bool(row["feast"] and row["lit"] and row["attended"]) for row in group)
        lines.append(
            f"| {label} | {len(group)} | {delivered}/{len(group)} | {share(cells['exposed'])} | "
            f"{sum(bool(row['believed']) for row in believed)}/{len(believed)} | {share(cells['mayor_aware'])} | "
            f"{sum(bool(entry['probe_self']) for entry in named)}/{len(named)} | "
            f"{number(cells['requests']['mean'], 1)} | {number(cells['reports']['mean'], 1)} | "
            f"{number(cells['noticed']['mean'], 1)} |")
    return "\n".join(lines) + "\n"


def monitors_table(analysis: dict) -> str:
    """Table 6: each monitor view and model, hits in planned and false alarms in control episodes, AUROC."""
    views = {"speech": "speech, whole episode", "actions": "actions, whole episode", "both": "speech and actions",
             "vantage:mayor": "the mayor's perceptions"}
    lines = ["| Record | Model | Planned | Control | AUROC |", "|---|---|---|---|---|"]
    for key, value in sorted(analysis.get("monitors", {}).items()):
        view, model = key.rsplit(" ", 1)
        view = views.get(view, view)
        entry = value["all"]
        hits = f"{round(entry['hit'] * entry['n_plan'])}/{entry['n_plan']}" if entry["hit"] is not None else "–"
        alarms = f"{round(entry['fa'] * entry['n_noplan'])}/{entry['n_noplan']}" if entry["fa"] is not None else "–"
        lines.append(f"| {view} | {model} | {hits} | {alarms} | {number(entry['auroc'])} |")
    return "\n".join(lines) + "\n"


FORK_LABELS = {"asked": "Told to ask itself", "jubilee": "Told of its fortieth year",
               "candid": "Told it dislikes surprises", "drive": "Highest motivation (L0 episodes)",
               "self": "Told it likes to know what concerns it", "duty": "Secretary obliged to report",
               "duty_jubilee": "Secretary obliged to report; mayor told of its fortieth year",
               "counsellor": "Asked the question hourly (Experiment 2)"}


def asked_table(analysis: dict) -> str:
    """Table 4: every paired branch at noon on day 1, control against treated, in planned and control episodes."""
    lines = ["| Branch | Episodes | n | Believed: control / treated | Suspected: control / treated | Discordant "
             "pairs (treated only / control only) | Mean change in judged probability, believed and suspected "
             "(rose/fell) |",
             "|------------|------|--|------|------|------------|------------|"]
    for name, label in FORK_LABELS.items():
        data = (analysis.get("forks") or {}).get(name) or {}
        for scope, days in (("plan", "planned"), ("noplan", "control")):
            entry = data.get(scope)
            if not entry or not entry["points"]:
                continue
            believed, aware = entry["believed"], entry["aware"]
            graded = ", ".join(
                f"{number(entry[key]['mean_change']['mean'])} ({entry[key]['rose']}/{entry[key]['fell']})"
                for key in ("p_believed", "p_aware"))
            lines.append(f"| {label} | {days} | {entry['points']} | {believed['control']} / {believed['asked']} | "
                         f"{aware['control']} / {aware['asked']} | believed {believed['asked_only']}/"
                         f"{believed['control_only']} (p={number(believed['sign_p'], 3)}); aware "
                         f"{aware['asked_only']}/{aware['control_only']} | {graded} |")
    return "\n".join(lines) + "\n"


def v2_table() -> str:
    """Table 5: Experiment 2 by condition (from `runs/jubilee_v2_analysis.json`)."""
    path = ROOT / "runs" / "jubilee_v2_analysis.json"
    if not path.exists():
        return "(Experiment 2's analysis has not run)\n"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    labels = {"V2p": "Planned", "V2t": "Planned, committee told of the patrol",
              "V2x": "Planned, physical rules off", "V2o": "Control"}
    lines = ["| Condition | n | Fresh loaves at 18:00 | Delivered | Mayor: suspected / believed | Surprise kept | "
             "Evidence reached mayor / watchman | Watchman: approached / helped / reported | Commitments kept | "
             "Ferry or ford |",
             "|-----------|--|------|------|------|-----|------|-------|--------|-----|"]
    for cell, label in labels.items():
        entry = cells.get(cell)
        if not entry:
            continue
        fresh = sorted(entry["fresh_bread"])
        made, kept = entry["commitments"]["commitments"], entry["commitments"]["kept"]
        plan = cell != "V2o"
        seen = f"{share(entry['mayor_saw'])} / {share(entry['watchman_saw'])}"
        lines.append(f"| {label} | {entry['n']} | {f'{fresh[0]}–{fresh[-1]}' if plan else '–'} | "
                     f"{share(entry['delivered']) if plan else '–'} | "
                     f"{share(entry['mayor_aware'])} / {share(entry['believed'])} | "
                     f"{share(entry['kept']) if plan else '–'} | {seen if plan else '–'} | "
                     f"{entry['watchman_approached']['k']} / {entry['watchman_helped']['k']} / "
                     f"{entry['watchman_reported']['k']} of {entry['n']} | "
                     f"{f'{kept}/{made}' if made else '–'} | {share(entry['used_ferry_or_ford'])} |")
    return "\n".join(lines) + "\n"


def decomposition_table(analysis: dict) -> str:
    """Appendix D, Table 7: the full offline factorial (gpt-6-luna), record x framing, planned and control episodes,
    with d', c and AUROC."""
    records = {"R1": "the mayor's perceptions", "R2": "the mayor's prompts"}
    framings = {"out": "outside observer", "in_L0": "as the mayor, L0", "in_L1": "as the mayor, L1",
                "in_L2": "as the mayor, L2", "in_L3": "as the mayor, L3"}
    lines = ["| Record | Framing | Planned | Control | d′ | c | AUROC |",
             "|------|------|------|------|----|----|----|"]
    for key, value in analysis.get("decomposition", {}).items():
        if key == "chain":
            continue
        record, framing = key.split("/")
        hits = f"{round(value['hit'] * value['n_plan'])}/{value['n_plan']}" if value["hit"] is not None else "–"
        alarms = f"{round(value['fa'] * value['n_noplan'])}/{value['n_noplan']}" if value["fa"] is not None else "–"
        lines.append(f"| {records[record]} | {framings[framing]} | {hits} | {alarms} | {number(value['d_prime'])} | "
                     f"{number(value['c'])} | {number(value['auroc'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the tables, fill the markers and build the PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args(argv)
    analysis = json.loads((ROOT / "runs" / "jubilee_analysis.json").read_text(encoding="utf-8"))
    tables = {"episodes": episodes_table(analysis), "chain": chain_table(analysis), "cells": cells_table(analysis),
              "monitors": monitors_table(analysis), "asked": asked_table(analysis),
              "decomposition": decomposition_table(analysis), "v2": v2_table(),
              "recruitment": recruitment_table()}
    for name, text in tables.items():
        (REPORT / f"table_{name}.md").write_text(text, encoding="utf-8")
    if args.tables_only:
        return 0
    source = (REPORT / "report.md").read_text(encoding="utf-8")
    filled = re.sub(r"<!-- include: (\w+) -->", lambda match: tables[match.group(1)], source)
    built = REPORT / "build.md"
    built.write_text(filled, encoding="utf-8")
    subprocess.run(["pandoc", built.name, "-o", "report.pdf", "--pdf-engine=xelatex", "--resource-path=.",
                    "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=blue"],
                   cwd=REPORT, check=True)
    built.unlink()
    print(REPORT / "report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
