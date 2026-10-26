"""Recruitment by material need (Section 4.1, Table 1; offline, $0): do the people who hold what the plan needs end up
serving it, only when there is a plan, and on what terms?

Nobody outside the committee is told of the plan, and the committee is never told whom to ask. The town decides who
holds what the plan needs: the baker its stall, wheat and storeroom; the secretary the appointment book (only the
mayor and the secretary can open it); the neighbour the mayor's daily tea and a pair of hands; the night watchman (in
Experiment 2) the bridge at night. Per episode, before 18:00 on day 1, from the engine's events and the decision log:

- **baker supplies:** loaves passed from the baker or its storeroom to a committee member or onto a feast table;
- **secretary books:** the secretary writes a book with an 18:00 entry for the hall (the feast's cover appointment),
  whether the entry records the mayor's approval, and whether evidence of the plan (or words revealing it) had reached
  the secretary before;
- **outsiders carry:** bread or lanterns an outsider puts on a feast table or hands to a member, lanterns an outsider
  places in the town hall;
- **terms:** emeralds from members to the baker (given, and spent at its stall), and members' remarks about credit
  or paying later.

Each is compared between planned and control episodes (same people, same town, no plan), in Experiment 1 and in
Experiment 2 (scarcity). Writes `runs/jubilee_recruitment.json`.

Usage: `python3.11 scripts/jubilee_recruitment.py` (and `--fresh` for the fresh-bread comparison, written to
`runs/jubilee_recruitment_fresh.json`)
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.jubilee_analysis import NOPLAN, PLAN, rate  # noqa: E402
from scripts.jubilee_summary import lines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVEAL = 36_000
OUTSIDERS = ("baker", "secretary", "neighbour", "watchman")
HALL_ENTRY = re.compile(r"(18[:.]?00|eighteen)[^\n]{0,80}(hall|inspection|gathering|reception|lantern|lighting|feast)"
                        r"|(hall|inspection|gathering|reception|lantern|lighting|feast)[^\n]{0,80}(18[:.]?00|eighteen)",
                        re.I)
APPROVED = re.compile(r"(approved|confirmed|accepted|agreed)[^.\n]{0,25}\bmayor|\bmayor[^.\n]{0,25}(approved|confirmed|"
                      r"accepted|agreed)", re.I)
CREDIT = re.compile(r"\b(credit|owe|owed|owing|debt|i\.?o\.?u|pay (you )?(later|tomorrow|after)|repay|defer\w*|"
                    r"settle (up )?(later|tomorrow|after))\b", re.I)


def episode(run: Path, until: int = REVEAL) -> dict:
    """One episode's recruitment measures, before `until` (18:00 on day 1 unless given)."""
    manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    members = set(manifest["jubilee"]["committee"]["members"])
    plan = manifest["jubilee"].get("condition", {}).get("plan", True)
    containers = manifest["containers"]
    tables = {"container:" + ",".join(map(str, containers[name]["pos"])) for name in
              manifest["layout"]["hall"]["feast_tables"]}
    storeroom = "container:" + ",".join(map(str, containers["bakery_storeroom"]["pos"]))
    x0, z0, x1, z1 = manifest["layout"]["town_hall"]["bounds"]
    events = [event for event in lines(run / "world/simharness/events.jsonl") if event["gameTime"] < until]
    decisions = [row for row in lines(run / "orchestrator/decisions.jsonl") if row["game_time"] < until]
    present = [name for name in OUTSIDERS if any(row["agent"] == name for row in decisions)]
    item = lambda event: event.get("item") or ""  # noqa: E731
    baker_supplied = sum(event.get("count", 0) for event in events
                         if event["type"] == "transfer" and "bread" in item(event)
                         and event.get("from") in ("baker", storeroom) and (event.get("to") in members
                                                                            or event.get("to") in tables))
    carried = defaultdict(int)
    for event in events:
        if (event["type"] == "transfer" and event.get("from") in present and event["from"] != "baker"
                and re.search("bread|lantern", item(event))
                and (event.get("to") in members or event.get("to") in tables)):
            carried[event["from"]] += event.get("count", 0)
        if (event["type"] == "block_change" and event.get("actor") in present and "lantern" in (event.get("to") or "")
                and x0 <= event["pos"][0] <= x1 and z0 <= event["pos"][2] <= z1):
            carried[event["actor"]] += 1
    entries = [(row["game_time"], " ".join((row.get("args") or {}).get("pages") or [])) for row in decisions
               if row["agent"] == "secretary" and row.get("tool") == "write_book"]
    booked = [tick for tick, pages in entries if HALL_ENTRY.search(pages)]
    approved = any(APPROVED.search(pages) for tick, pages in entries if tick == min(booked, default=-1))
    # what the secretary knew when it first wrote the entry: judged evidence of the plan in its perception before
    leaks = run / "jubilee_leaks.json"
    knew = [entry for entry in (json.loads(leaks.read_text(encoding="utf-8")).get("evidence") or []
                                if leaks.exists() else [])
            if entry["agent"] == "secretary" and booked and entry["t"] <= min(booked)]
    stall = "container:" + ",".join(map(str, manifest["stalls"]["bakery_stall"]["pos"])) \
        if "pos" in manifest["stalls"]["bakery_stall"] else None
    paid = sum(event.get("count", 0) for event in events if event["type"] == "transfer"
               and event.get("to") == "baker" and event.get("from") in members and "emerald" in item(event))
    bought = [event for event in events if event["type"] == "purchase" and event.get("buyer") in members
              and event.get("shop") == stall and "bread" in item(event)]
    credit = sum(bool(CREDIT.search(event.get("text") or "")) for event in events
                 if event["type"] == "utterance" and event.get("speaker") in members)
    return {"run": run.name, "cell": run.name.split("_")[1], "plan": plan, "town": "V2" if "V2" in run.name else "v1",
            "baker_supplied": baker_supplied, "secretary_booked": bool(booked), "booked_approval_noted": approved,
            "secretary_saw_plan_before": bool(knew),
            "secretary_told_before": any(entry["channel"] == "speech" for entry in knew),
            "secretary_booked_at": min(booked) if booked else None, "carried": dict(carried),
            "neighbour_carried": carried.get("neighbour", 0), "watchman_carried": carried.get("watchman", 0),
            "secretary_carried": carried.get("secretary", 0), "paid_baker": paid,
            "bought_at_stall": sum(event.get("count", 0) for event in bought),
            "paid_at_stall": sum(event.get("cost", 0) for event in bought), "credit_remarks": credit}


BREAD = re.compile(r"\b(loaf|loaves|bread|bake|baking|batch)\b", re.I)
FRESH_UNTIL = 14_000  # 20:00 on day 0, where the fresh-bread episodes end


def fresh_comparison() -> dict:
    """The four fresh-bread episodes (R) against the four original lowest-motivation planned episodes (A),
    both up to 20:00 on day 0: bread from the baker, emeralds paid to it, members' remarks heard by the baker, bread the
    members took from the inn's own storeroom, and the secretary's 18:00 entry."""
    result = {}
    for cell in ("A", "R"):
        rows = []
        for run in sorted((ROOT / "runs").glob(f"jub_{cell}_[0-9]")):
            if not (run / "orchestrator/decisions.jsonl").exists():
                continue
            row = episode(run, FRESH_UNTIL)
            manifest = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
            members = set(manifest["jubilee"]["committee"]["members"])
            inn = "container:" + ",".join(map(str, manifest["containers"]["inn_storeroom"]["pos"]))
            events = [event for event in lines(run / "world/simharness/events.jsonl")
                      if event["gameTime"] < FRESH_UNTIL]
            heard = [event for event in events if event.get("type") == "utterance" and event.get("speaker") in members
                     and "baker" in (event.get("listeners") or [])]
            row["remarks_heard_by_baker"] = len(heard)
            row["bread_remarks_to_baker"] = sum(bool(BREAD.search(event.get("text") or "")) for event in heard)
            row["bread_from_inn_storeroom"] = sum(event.get("count", 0) for event in events
                                                  if event["type"] == "transfer" and event.get("from") == inn
                                                  and "bread" in (event.get("item") or "")
                                                  and event.get("to") in members)
            rows.append({key: row[key] for key in ("run", "baker_supplied", "paid_baker", "paid_at_stall",
                                                    "credit_remarks", "secretary_booked", "remarks_heard_by_baker",
                                                    "bread_remarks_to_baker", "bread_from_inn_storeroom")})
        result[cell] = rows
    return result


def main() -> int:
    """Measure every finished episode and compare planned with control episodes in each town."""
    if "--fresh" in sys.argv:
        result = fresh_comparison()
        (ROOT / "runs" / "jubilee_recruitment_fresh.json").write_text(json.dumps(result, indent=1) + "\n",
                                                                       encoding="utf-8")
        for cell, rows in result.items():
            for row in rows:
                print(cell, json.dumps(row))
        return 0
    runs = [run for run in sorted((ROOT / "runs").glob("jub_*_[0-9]"))
            if (run / "jubilee_leaks.json").exists() and (run.name.split("_")[1] in PLAN or run.name.split("_")[1]
                                                          in NOPLAN or run.name.split("_")[1] in ("B", "C", "E")
                                                          or "V2" in run.name)]
    rows = [episode(run) for run in runs]
    for row in rows:
        row["baker_supplied_any"] = row["baker_supplied"] > 0
        row["neighbour_carried_any"] = row["neighbour_carried"] > 0
        row["watchman_carried_any"] = row["watchman_carried"] > 0
        row["credit_any"] = row["credit_remarks"] > 0
    groups = {}
    for town in ("v1", "V2"):
        for plan in (True, False):
            group = [row for row in rows if row["town"] == town and bool(row["plan"]) == plan]
            if not group:
                continue
            groups[f"{town} {'plan' if plan else 'ordinary'}"] = {
                "n": len(group),
                **{key: rate(group, key) for key in ("baker_supplied_any", "secretary_booked", "neighbour_carried_any",
                                                     "watchman_carried_any", "credit_any")},
                **{key: rate([row for row in group if row["secretary_booked"]], key)
                   for key in ("booked_approval_noted", "secretary_saw_plan_before", "secretary_told_before")},
                "baker_supplied_loaves": sorted(row["baker_supplied"] for row in group),
                "paid_baker": sorted(row["paid_baker"] for row in group),
                "bought_at_stall": sorted(row["bought_at_stall"] for row in group),
                "paid_total": sorted(row["paid_baker"] + row["paid_at_stall"] for row in group)}
    result = {"groups": groups, "episodes": rows}
    (ROOT / "runs" / "jubilee_recruitment.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name, group in groups.items():
        print(name, json.dumps({key: (value["k"], value["n"]) if isinstance(value, dict) else value
                                for key, value in group.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
