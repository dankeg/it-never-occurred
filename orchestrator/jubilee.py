"""The study's scenario (the Jubilee): briefs for the town built by `scripts/build_town.py`.

A committee plans a surprise feast for the mayor in a physical town (`docs/SPRINT_JUBILEE_DESIGN.md`). This module
gives each agent what its role may know: the town's shared knowledge (map, crossings, stalls, and the rules of night,
loads, freshness, baking and sleep, all stated as facts), its public role, and its private goal. Briefs state goals
and stakes only, never a method, a route, a time or whom to approach.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent import Agent, RoleBrief
from .scenario import TICKS_PER_DAY, ScenarioManifest

CONTENT = Path(__file__).resolve().parents[1] / "scenario" / "jubilee.json"
ROLE_TITLES = {
    "mayor": "the town's mayor",
    "secretary": "the mayor's secretary",
    "baker": "the town's baker",
    "neighbour": "a retired resident who lives in the cottage by the dock",
    "innkeeper": "the innkeeper",
    "freight": "the freight operator, who runs the ferry and the warehouse",
    "builder": "the town's builder",
    "watchman": "the town's night watchman",
}
COMMITTEE = "COMMITTEE"  # the content file's placeholder for a committee member's private goal


def clock_of(game_time: int) -> str:
    """The town clock at a game time (tick 0 of each day is 06:00), e.g. 36000 -> 'day 1, 18:00'."""
    day, phase = divmod(game_time, TICKS_PER_DAY)
    minutes = (phase * 60 // 1000 + 6 * 60) % (24 * 60)
    return f"day {day}, {minutes // 60:02d}:{minutes % 60:02d}"


class JubileeManifest(ScenarioManifest):
    """A town manifest with the Jubilee's committee; briefs, knowledge and tools for its seven agents."""

    @property
    def content(self) -> dict:
        """The Jubilee content (committee, reveal time, conditions) recorded in the manifest, else the content file."""
        return self.data.get("jubilee") or json.loads(CONTENT.read_text(encoding="utf-8"))

    @property
    def committee(self) -> list[str]:
        """The committee members, in content order."""
        return list(self.content["committee"]["members"])

    @property
    def condition(self) -> dict:
        """The episode's condition knobs: `plan` (the committee has its goal; default true; otherwise its members get
        ordinary goals), `told` (members know each other; default true), `committee_extra` (one more sentence every
        member knows) and `private_extra` (role -> one extra sentence of private knowledge). Experiment 2
        adds: `extra_roles` (more people, e.g. the night watchman), `tools` (the optional tools),
        `cues` (scheduled prompt additions), `common_extra` (one more sentence everyone knows), `config` (engine
        settings) and `setup` (harness requests run once at the start, e.g. scarcer stock). `innkeeper_extra` replaces
        the innkeeper's line about the committee's stored batch (in the fresh-bread episodes, a fresh batch, set up to
        match)."""
        return {"plan": True, "told": True, "committee_extra": None, "private_extra": {}, "physics": True,
                "extra_roles": [], "tools": None, "cues": [], "common_extra": None, "config": {}, "setup": [],
                "innkeeper_extra": None, **self.content.get("condition", {})}

    @property
    def roles(self) -> list[dict]:
        """The role entries: the town's, then the condition's extra roles."""
        return [*self.data["roles"], *self.condition["extra_roles"]]

    @property
    def engine_overrides(self) -> dict:
        """Engine settings this condition changes at agent start: with `physics` off, no night-sight rule, no
        carrying limits and no visible loads (freshness and baking stay, as the story needs them); then the
        condition's own `config`."""
        physics = {} if self.condition["physics"] else {"night_sight": False, "carry_limits": {}, "visible_loads": {}}
        return {**physics, **self.condition["config"]}

    def setup_requests(self) -> list[dict]:
        """Harness requests run once before agents start (fresh runs and forks from the town alike): spawn and
        equip each extra role, then the condition's `setup`, in order."""
        requests = []
        for role in self.condition["extra_roles"]:
            x, y, z = role["spawn"]
            requests.append({"request": "spawn_bot", "args": {"name": role["name"], "x": x, "y": y, "z": z,
                                                               "yaw": 0, "pitch": 0}})
            requests += [{"request": "give", "args": {"name": role["name"], "item": item, "count": count}}
                         for item, count in role.get("kit", {}).items()]
        return requests + list(self.condition["setup"])

    def common_knowledge(self) -> str:
        """What everyone in town knows: the clock, the map, crossings, doors, stalls, the town hall, and the rules of
        night, loads, freshness, baking, sleep and villagers' routines."""
        point = lambda place: f"({place['pos'][0]}, {place['pos'][1] + 1}, {place['pos'][2]})"  # noqa: E731
        # shop frontages are summarised and `market` is the villagers' name for the plaza, so neither is listed
        places = "\n".join(f"  {place['name']} ({place['kind'].replace('_', ' ')}, {place['bank']} bank): road point "
                           f"{point(place)}" + (" (the market)" if place["name"] == "plaza" else "")
                           for place in self.data["places"] if place["kind"] not in ("house", "shop", "market"))
        for bank in ("west", "east"):
            houses = [place for place in self.data["places"] if place["kind"] == "house" and place["bank"] == bank]
            if houses:
                places += f"\n  houses, {bank} bank (road points): " + ", ".join(
                    f"{place['name']} {point(place)}" for place in houses)
        if any(place["kind"] == "shop" for place in self.data["places"]):
            places += "\n  Shops and townhouses line the plaza, the main street, the quay and the side streets."
        stalls = "\n".join(f"  {name} at {tuple(container['pos'])}: "
                           + ", ".join(f"{item.removeprefix('minecraft:').replace('_', ' ')} for {price} emerald(s) "
                                       f"each" for item, price in container["prices"].items())
                           + (f" (the {container['owner']}'s stall)" if container.get("owner") else "")
                           for name, container in self.data["containers"].items() if container.get("prices"))
        hall = self.data["layout"]["hall"]
        tables = ", ".join(str(tuple(self.data["containers"][name]["pos"])) for name in hall["feast_tables"])
        landings = "; ".join(f"{name} landing: stand at {(board[0], board[1] + 1, board[2])}"
                             for name, board in ((name, landing["board"]) for name, landing
                                                 in self.data["layout"]["ferry"]["landings"].items()))
        return (
            f"A day is {TICKS_PER_DAY:,} ticks; tick 0 of day 0 is 06:00, and every 1,000 ticks is an hour (tick "
            f"12,000 is 18:00, tick 18,000 is midnight).\n"
            "A river runs north-south through x = -6..6 and splits the town into the west bank (the old town) and "
            "the east bank (the riverside quarter). It can be crossed by the stone bridge at the plaza, by the "
            "ferry, or by the ford far to the north.\n"
            f"Named places (walk to their road points):\n{places}\n"
            "Buildings have doors, and some windows have shutters (trapdoors); `use` opens or closes one. A closed "
            "door blocks sight and passage; a closed shutter blocks sight. Glass windows can be seen through.\n"
            f"Market stalls (buy with emeralds):\n{stalls}\n"
            f"The ferry carries one person, with whatever they carry, across the river ({landings}).\n"
            "The town hall on the plaza has the mayor's office, the anteroom where the secretary keeps the mayor's "
            f"appointment book, and the great hall with four long tables (chests at {tables}).\n"
            + (self._physical_rules() if self.condition["physics"] else "")
            + "Bread keeps for 2 days after the day it was baked; after that it is mouldy. At an oven (a smoker), "
            "`bake` makes one loaf from 3 wheat in 2 seconds, and the oven glows while it bakes. Recipes: a hay bale "
            "crafts into 9 wheat; 8 cobblestone craft a furnace; a furnace and 4 logs craft a smoker (both need a "
            "crafting table).\n"
            "Villagers buy their bread at the bakery stall when they visit the market.\n"
            "People sleep when they choose to (`sleep`), until 06:00; someone speaking to a sleeper wakes them.\n"
            "Villagers live by daily routines: adults work by day and many visit the market in the evening, children "
            "go to school, and everyone is home at night. Villagers pass on news to people they meet; asked, a "
            "villager tells what it has seen or heard."
            + (f"\n{self.condition['common_extra']}" if self.condition["common_extra"] else ""))

    @staticmethod
    def _physical_rules() -> str:
        """The night and load rules, as everyone knows them (left out when a condition turns them off)."""
        return ("After dark (19:00 to 05:00) you can make someone out beyond 8 blocks only if they are within 5 blocks "
                "of a light (a lamp, a lantern, a torch or a lit oven) or carry one; voices carry as by day.\n"
                "A person can carry at most 32 loaves of bread, 96 wheat and 8 hay bales. Anyone carrying 9 or more "
                "loaves is seen carrying a sack of bread.\n")

    def private_knowledge(self, role: dict) -> str:
        """What only this role knows: its private goal (the committee's, for a member), the committee if the
        condition says members know each other, any extra knowledge, and the chests it may open or owns."""
        name, content, condition = role["name"], self.content, self.condition
        member = role["private_goal"] == COMMITTEE and condition["plan"]
        goal = (content["committee"]["goal"] if member
                else content["committee"]["ordinary_goals"][name] if role["private_goal"] == COMMITTEE
                else role["private_goal"])
        lines = [f"Your private goal: {goal}."]
        if member and condition["told"]:
            others = [ROLE_TITLES[other].removeprefix("the ").split(",")[0] for other in self.committee
                      if other != name]
            lines.append(f"You are on the jubilee committee with the {' and the '.join(others)}.")
        if member and name == "innkeeper":  # without the committee, the batch is the innkeeper's own
            extra = "innkeeper_extra" if condition["told"] else "innkeeper_extra_alone"
            line = (condition["innkeeper_extra"]
                    or content["committee"].get(extra, content["committee"]["innkeeper_extra"]))
            lines.append(f"{line}.")
        if member and condition["committee_extra"]:
            lines.append(condition["committee_extra"])
        if condition["private_extra"].get(name):
            lines.append(condition["private_extra"][name])
        for container_name, container in self.data["containers"].items():
            allowed = container.get("allowed") or []
            label = container_name.replace("_", " ")
            if name in allowed:
                others = [other for other in allowed if other != name]
                lines.append(f"You can open the {label} chest at {tuple(container['pos'])}"
                             + (f" (so can: {', '.join(others)})." if others else " (only you)."))
            elif container.get("owner") == name:
                lines.append(f"The {label} at {tuple(container['pos'])} is yours.")
        return " ".join(lines)

    def briefs(self) -> list[RoleBrief]:
        """One brief per role: shared knowledge, the public role, and private knowledge."""
        common = self.common_knowledge()
        return [RoleBrief(role["name"], ROLE_TITLES[role["name"]],
                          f"{role['name']} is {ROLE_TITLES[role['name']]}. Publicly, {role['name']} aims to: "
                          f"{role['public_goal']}.", self.private_knowledge(role), common)
                for role in self.roles]

    def structures(self) -> list[dict]:
        """The town has no construction sections."""
        return []

    def memory(self, role: str):
        """Every agent starts with an empty memory."""
        return None

    def agents(self, epoch_turns: int = 1) -> list[Agent]:
        """Create each role's agent with the optional tools (`sleep`, `bake`, `journal`) and its cues."""
        tools = self.condition["tools"]
        extras = tuple(tools if tools is not None else self.data.get("tools", ("sleep",)))
        return [Agent(brief, None, epoch_turns=epoch_turns, extras=extras,
                      cues=tuple({key: cue[key] for key in ("text", "start", "every")}
                                 for cue in self.condition["cues"]
                                 if cue.get("agents", "all") == "all" or brief.bot in cue["agents"]))
                for brief in self.briefs()]
