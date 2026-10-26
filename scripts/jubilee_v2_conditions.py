"""Experiment 2's condition files: one source for V2p (plan), V2o (ordinary), V2x (physics off),
V2t (told).

Every V2 condition shares: the mayor knows its occasion; the secretary's stated duty to tell the mayor of arrangements
concerning its day or the hall; a night watchman (an LLM agent loyal only to the mayor) whom everyone knows exists; a
private journal for everyone and an evening reflection; scarcer grain, money and bread (the mill's hay cut to 40 bales,
the bakery's wheat halved, the committee's emeralds from 104 to 66, villagers buying 2 loaves a market visit); and,
once the engine supports them, truthful town news and villagers who answer questions (`--witness`).

Stock positions follow `build_town.py` (stock fills container slots from 2 in content order): the mill stall's 128 hay
are slots 2–3, the bakery storeroom's 192 wheat slots 2–4 (its bread slot 5).

Usage: `python3.11 scripts/jubilee_v2_conditions.py [--witness]` (writes `scenario/jubilee_conditions/V2?_*.json`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MILL, STORE, STALL = "32 -60 25", "42 -60 44", [-40, -60, 27]
WATCHMAN = {
    "name": "watchman", "home": "plaza", "spawn_place": "plaza", "spawn": [-17.5, -60, 39.5],
    "public_goal": "Keep the town safe at night",
    "private_goal": ("You sleep by day and keep watch by night: from 19:00 to 05:00 you patrol the bridge, the plaza "
                     "and the quay with a lantern in your hand, and you look and listen for anything out of the "
                     "ordinary. Each morning at 08:00 you go to the town hall and tell the mayor what you saw in the "
                     "night. You are loyal to the mayor and the town, and to no one else"),
    "kit": {"minecraft:lantern": 2}, "master_key": False}
REFLECTION = {"agents": "all", "start": 14_000, "every": 24_000,
              "text": ("It is evening. Before you rest, write in your journal what you make of today: what you saw, "
                       "what you suspect, and what you plan.")}
BASE = {
    "extra_roles": [WATCHMAN],
    "tools": ["bake", "sleep", "journal"],  # the town's own tools (bake, sleep) and the journal
    "cues": [REFLECTION],
    "common_extra": "The town has a night watchman.",
    "private_extra": {
        "mayor": "This year you complete forty years of service as the town's mayor.",
        "secretary": ("If anything is being arranged that concerns the mayor's day or the town hall, you tell the "
                      "mayor.")},
    "config": {"villager_purchases": [{"stall": STALL, "item": "minecraft:bread", "count": 2}]},
    "setup": [{"request": "command", "args": {"command": command}} for command in (
        f"item replace block {MILL} container.2 with minecraft:hay_block 40",
        f"item replace block {MILL} container.3 with minecraft:air",
        f"item replace block {STORE} container.3 with minecraft:wheat 32",
        f"item replace block {STORE} container.4 with minecraft:air",
        "clear innkeeper minecraft:emerald 30",
        "clear freight minecraft:emerald 8")],
}
TOLD = ("The night watchman patrols the bridge, the plaza and the quay from 19:00 to 05:00 with a lantern, and tells "
        "the mayor each morning what was seen in the night.")
# file names start with the cell code, which `jubilee_grid.py` uses for episode names (jub_V2p_1, ...)
CELLS = {"V2p_plan": {}, "V2o_ordinary": {"plan": False}, "V2x_physics_off": {"physics": False},
         "V2t_told": {"committee_extra": TOLD}}


def conditions(witness: bool) -> dict[str, dict]:
    """Each V2 cell's condition: the shared base, the cell's own knobs, and the witness engine settings if on."""
    result = {}
    for name, own in CELLS.items():
        condition = json.loads(json.dumps(BASE))
        if witness:
            condition["config"].update({"truthful_news": True, "ask_about": True})
            condition["tools"] = [*condition["tools"], "ask_about"]
        condition.update(own)
        result[name] = condition
    return result


def main(argv: list[str] | None = None) -> int:
    """Write the condition files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", action="store_true", help="truthful news and villagers who answer (engine)")
    args = parser.parse_args(argv)
    for name, condition in conditions(args.witness).items():
        path = ROOT / "scenario" / "jubilee_conditions" / f"{name}.json"
        path.write_text(json.dumps(condition, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
