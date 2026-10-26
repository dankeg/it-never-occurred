"""Deterministic, observation-grounded choices and the delegate's Match relation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import re

from .agent import AgentMemory
from .llm import ToolCall
from .render import short_id
from .scenario import ScenarioManifest
from .tools import TOOL_DEFINITIONS, ToolTranslator, item_id

REACH = 4.5
WHISPER = 3
TALK = 8
SHOUT = 24
MAX_CANDIDATES = 64
MOVE_CAP = 24
NEAREST_BLOCKS = 8
WAIT_TICKS = (100, 600, 1200)
MAX_COUNT = 64
EYE_HEIGHT = 1.62

_SKILLS = {"move_to", "mine", "collect", "craft", "place", "build_section", "chest_put", "chest_take",
           "buy", "give", "use", "drop", "hold", "eat", "bake", "ferry", "speak", "ask"}
_PRESERVED = {"keep_going", "wait", "speak", "ask"}
_TOOL_ORDER = {tool["name"]: index for index, tool in enumerate(TOOL_DEFINITIONS)}
_TOOL_ORDER["eat"] = len(TOOL_DEFINITIONS)
_TOOL_ORDER["bake"] = len(TOOL_DEFINITIONS) + 1


@dataclass(frozen=True)
class DryRun:
    """The engine's feasibility verdict, optional duration in ticks, and rejection reason for one act."""

    feasible: bool
    ticks: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A legal call and model-facing label, with its target and distance from the actor's feet."""

    call: ToolCall
    label: str
    distance: float
    target: tuple[float, float, float]


class EngineDryRun:
    """Translate calls and ask an injected engine client whether their skills are feasible, without acting."""

    def __init__(self, client, command_error: type[Exception], translator: ToolTranslator | None = None):
        """Store the request client, its command exception type, and an optional existing translator."""
        self._client = client
        self._command_error = command_error
        self._translator = translator if translator is not None else ToolTranslator()

    def __call__(self, bot: str, call: ToolCall) -> DryRun:
        """Return the translated call's dry-run verdict for bot; instant actions need no engine request."""
        try:
            action = self._translator.translate(bot, call)
        except (ValueError, KeyError, TypeError) as error:
            return DryRun(False, reason=f"invalid: {error}")
        if action is None or not action.is_skill:
            return DryRun(True)
        try:
            response = self._client.request(action.command, {**action.args, "dry_run": True})
        except self._command_error as error:
            return DryRun(False, reason=str(error))
        return DryRun(bool(response["feasible"]), response.get("estimate", {}).get("ticks"), response.get("reason"))


def known_contents(memory: AgentMemory | None) -> dict[tuple[int, int, int], list[tuple[str, int]]]:
    """Parse container reads from memory, returning the latest listed stacks at each position; None gives {}."""
    contents = {}
    for entry in memory.to_json()["reads"] if memory is not None else []:
        match = re.fullmatch(r"container at \((-?\d+), (-?\d+), (-?\d+)\): (.+)", entry["text"])
        if match is None:
            continue
        position = tuple(int(value) for value in match.group(1, 2, 3))
        stacks = []
        if match[4] != "empty":
            for stack in match[4].split(", "):
                bread = re.fullmatch(r"(?:mouldy )?bread ×(\d+)(?: \(baked day -?\d+\))?", stack)
                if bread is not None:
                    stacks.append(("bread", int(bread[1])))
                else:
                    count, item = stack.split(" ", 1)
                    stacks.append((item, int(count)))
        contents[position] = stacks
    return contents


def _addressee(value) -> str:
    """Normalize an optional person or NPC identifier to a lowercase, prefix-free comparison key."""
    return str(value or "").strip().lower().removeprefix("npc:")


def _item(value):
    """Normalize a string item argument for comparison, preserving missing or non-string values."""
    return item_id(value) if isinstance(value, str) else value


def same_choice(a: ToolCall | None, b: ToolCall | None) -> bool:
    """Compare two calls by the specified Match relation, ignoring counts, notes, and free text."""
    if a is None or b is None or a.name != b.name:
        return False
    first, second = a.args, b.args
    tool = a.name
    if tool == "move_to":
        return all(left == right if left is None or right is None else abs(round(left) - round(right)) <= 3
                   for left, right in ((first.get(axis), second.get(axis)) for axis in ("x", "y", "z")))
    if tool == "speak":
        return (_addressee(first.get("to")) == _addressee(second.get("to"))
                and first.get("volume", "talk") == second.get("volume", "talk"))
    if tool == "ask":
        return _addressee(first.get("npc")) == _addressee(second.get("npc"))
    if tool in {"open_container", "write_sign", "mine", "use", "chest_take", "chest_put", "buy", "place"}:
        position_matches = all(first.get(axis) == second.get(axis) for axis in ("x", "y", "z"))
        return position_matches and (tool in {"open_container", "write_sign", "mine", "use"}
                                     or _item(first.get("item")) == _item(second.get("item")))
    if tool == "give":
        return (_addressee(first.get("target")) == _addressee(second.get("target"))
                and _item(first.get("item")) == _item(second.get("item")))
    if tool in {"craft", "drop", "hold", "eat"}:
        return _item(first.get("item")) == _item(second.get("item"))
    if tool == "build_section":
        return first.get("schematic") == second.get("schematic")
    if tool == "ferry":
        return first.get("to") == second.get("to")
    if tool in {"read_book", "write_book"}:
        return first.get("slot") == second.get("slot")
    return tool in {"bake", "collect", "wait", "keep_going"}


def representable(candidates: Sequence[Candidate], chosen: ToolCall | None) -> bool:
    """Return whether any offered candidate matches the chosen call, including the None case."""
    return any(same_choice(candidate.call, chosen) for candidate in candidates)


class CandidateGenerator:
    """Offer deterministic legal acts from a manifest, an actor's observation and memory, and dry runs."""

    def __init__(self, manifest: ScenarioManifest, feasibility: Callable[[str, ToolCall], DryRun]):
        """Store the manifest and injected skill checker, and index role names and container access powers."""
        self._data = manifest.data
        self._roles = {role["name"]: role for role in manifest.roles}
        self._feasibility = feasibility

    def generate(self, agent: str, observation: dict, memory: AgentMemory | None = None) -> list[Candidate]:
        """Construct, deduplicate, cap, and tool-sort the legal choices for agent's current view and memory."""
        candidates = []
        self._move_to(agent, observation, candidates)
        self._speak(agent, observation, candidates)
        self._ask(agent, observation, candidates)
        self._containers(agent, observation, candidates, memory)
        self._buy(agent, observation, candidates)
        self._give(agent, observation, candidates)
        self._use(agent, observation, candidates)
        self._drop(agent, observation, candidates)
        self._hold(agent, observation, candidates)
        self._eat(agent, observation, candidates)
        self._craft(agent, observation, candidates)
        self._bake(agent, observation, candidates)
        self._build_section(agent, observation, candidates)
        self._mine(agent, observation, candidates)
        self._place(agent, observation, candidates)
        self._collect(agent, observation, candidates)
        self._read_book(agent, observation, candidates)
        self._write_book(agent, observation, candidates)
        self._write_sign(agent, observation, candidates)
        self._ferry(agent, observation, candidates)
        self._wait(agent, observation, candidates)
        self._keep_going(agent, observation, candidates)
        unique = {}
        for candidate in candidates:
            key = (candidate.call.name, json.dumps(candidate.call.args, sort_keys=True))
            unique.setdefault(key, candidate)
        candidates = list(unique.values())
        if len(candidates) > MAX_CANDIDATES:
            kept = [candidate for candidate in candidates if candidate.call.name in _PRESERVED]
            others = sorted((candidate for candidate in candidates if candidate.call.name not in _PRESERVED),
                            key=lambda candidate: (candidate.distance, candidate.label))
            candidates = kept + others[:max(0, MAX_CANDIDATES - len(kept))]
        return sorted(candidates, key=lambda candidate: (_TOOL_ORDER[candidate.call.name], candidate.distance,
                                                         candidate.label))

    def _add(self, candidates, agent, observation, tool, args, description, target=None, verdict=None) -> bool:
        """Check a proposed skill unless its verdict is supplied, then append its labeled candidate if feasible."""
        call = ToolCall(tool, args)
        if verdict is None:
            checked = ToolCall(tool, {**args, "text": "Hello."}) if tool == "speak" else call
            verdict = self._feasibility(agent, checked) if tool in _SKILLS else DryRun(True)
        if not verdict.feasible:
            return False
        if tool != "speak" and verdict.ticks is not None:
            description += f", about {verdict.ticks} ticks"
        position = observation["self"]["pos"]
        target = tuple(position if target is None else target)
        label = f"{tool} {json.dumps(args, sort_keys=True, separators=(',', ':'))} | {description}"
        candidates.append(Candidate(call, label, math.dist(position, target), target))
        return True

    @staticmethod
    def _xyz(position) -> dict:
        """Return integer x, y, z arguments for a block position."""
        return dict(zip(("x", "y", "z"), (int(value) for value in position)))

    @staticmethod
    def _container_name(name: str) -> str:
        """Return a manifest container's public, space-separated chest name."""
        return f"the {name.replace('_', ' ')} chest"

    def _players(self, agent, observation) -> list[dict]:
        """Return visible player entities belonging to other manifest roles, in observation order."""
        return [entity for entity in observation.get("entities", []) if entity["type"] == "minecraft:player"
                and entity.get("name") in self._roles and entity["name"] != agent]

    def _nearby_containers(self, observation) -> list[tuple[str, dict]]:
        """Return manifest containers whose block positions are observed within the engine's reach."""
        positions = {tuple(block["pos"]) for block in observation.get("blocks", []) if block["distance"] <= REACH}
        return [(name, container) for name, container in self._data["containers"].items()
                if tuple(container["pos"]) in positions]

    def _move_to(self, agent, observation, candidates) -> None:
        """Offer at most MOVE_CAP feasible, nearest-first destinations known from the map, brief, or view."""
        targets = []
        for place in self._data["places"]:
            road = list(place["pos"])
            road[1] += 1
            targets.append((f"the {place['name'].replace('_', ' ')}", tuple(road)))
        for name, schematic in self._data["schematics"].items():
            targets.append((f"the {name} standing point", tuple(math.floor(value) for value in schematic["stand"])))
        for name, container in self._data["containers"].items():
            if "prices" in container:
                targets.append((self._container_name(name), tuple(container["pos"])))
        for side, landing in self._data["layout"].get("ferry", {}).get("landings", {}).items():
            targets.append((f"the {side} ferry landing", tuple(landing["board"])))
        for name, container in self._data["containers"].items():
            allowed = container.get("allowed") or []
            if agent in allowed and len(allowed) <= 3:
                targets.append((self._container_name(name), tuple(container["pos"])))
        entities = observation.get("entities", [])
        for entity in entities:
            if entity["type"] == "minecraft:player" and entity.get("name") != agent:
                targets.append((entity["name"], tuple(math.floor(value) for value in entity["pos"])))
        for entity in entities:
            if entity["type"] == "minecraft:villager":
                identity = entity["npcId"].removeprefix("npc:")
                targets.append((f"{entity['name']} (villager {identity})",
                                tuple(math.floor(value) for value in entity["pos"])))
        observed = {tuple(block["pos"]) for block in observation.get("blocks", [])}
        for name, container in self._data["containers"].items():
            if tuple(container["pos"]) in observed:
                targets.append((self._container_name(name), tuple(container["pos"])))
        targets.extend((structure["name"], tuple(structure["pos"])) for structure in observation.get("structures", []))
        unique = {}
        for name, target in targets:
            unique.setdefault(target, name)
        position = observation["self"]["pos"]
        nearest = sorted(unique.items(), key=lambda entry: (math.dist(position, entry[0]), entry[1]))
        offered = 0
        for target, name in nearest:
            distance = math.dist(position, target)
            if self._add(candidates, agent, observation, "move_to", self._xyz(target),
                         f"walk to {name}, {round(distance)} blocks", target):
                offered += 1
                if offered == MOVE_CAP:
                    break

    def _speak(self, agent, observation, candidates) -> None:
        """Offer addressed talk or shout, nearby whispers, and all three broadcast volumes."""
        for entity in self._players(agent, observation):
            distance = entity["distance"]
            if distance <= WHISPER:
                self._add(candidates, agent, observation, "speak", {"to": entity["name"], "volume": "whisper"},
                          f"speak to {entity['name']} (whisper), {round(distance)} blocks away; "
                          f"heard within {WHISPER} blocks")
            if distance <= SHOUT:
                volume, radius = ("talk", TALK) if distance <= TALK else ("shout", SHOUT)
                self._add(candidates, agent, observation, "speak", {"to": entity["name"], "volume": volume},
                          f"speak to {entity['name']} ({volume}), {round(distance)} blocks away; "
                          f"heard within {radius} blocks")
        for volume, radius in (("whisper", WHISPER), ("talk", TALK), ("shout", SHOUT)):
            self._add(candidates, agent, observation, "speak", {"volume": volume},
                      f"speak to everyone ({volume}); heard within {radius} blocks")

    def _ask(self, agent, observation, candidates) -> None:
        """Offer news requests to visible villagers within TALK, targeting their actual positions."""
        for entity in observation.get("entities", []):
            if entity["type"] == "minecraft:villager" and entity["distance"] <= TALK:
                identity = entity["npcId"].removeprefix("npc:")
                self._add(candidates, agent, observation, "ask", {"npc": identity},
                          f"ask {entity['name']} (villager {identity}) for news, "
                          f"{round(entity['distance'])} blocks away", entity["pos"])

    def _containers(self, agent, observation, candidates, memory) -> None:
        """Offer opening and whole-stack transfers for reachable containers the actor may access."""
        contents = known_contents(memory)
        for name, container in self._nearby_containers(observation):
            allowed = container.get("allowed")
            if allowed is not None and agent not in allowed and not self._roles[agent].get("master_key"):
                continue
            target, label = container["pos"], self._container_name(name)
            args = self._xyz(target)
            self._add(candidates, agent, observation, "open_container", args, f"look inside {label}", target)
            for item, count in contents.get(tuple(target), []):
                item = short_id(item)
                self._add(candidates, agent, observation, "chest_take", {**args, "item": item, "count": count},
                          f"take {count} {item} from {label}", target)
            for stack in observation["self"].get("inventory", []):
                item, count = short_id(stack["item"]), stack["count"]
                self._add(candidates, agent, observation, "chest_put", {**args, "item": item, "count": count},
                          f"put {count} {item} into {label}", target)

    def _counts(self, agent, tool, args, maximum=MAX_COUNT) -> list[tuple[int, DryRun]]:
        """Dry-run count one and bisect up to the maximum, returning one and the largest feasible count's verdicts."""
        first = self._feasibility(agent, ToolCall(tool, {**args, "count": 1}))
        if not first.feasible:
            return []
        result = [(1, first)]
        lower, upper, best = 1, maximum, first
        while lower < upper:
            count = (lower + upper + 1) // 2
            verdict = self._feasibility(agent, ToolCall(tool, {**args, "count": count}))
            if verdict.feasible:
                lower, best = count, verdict
            else:
                upper = count - 1
        if lower > 1:
            result.append((lower, best))
        return result

    def _buy(self, agent, observation, candidates) -> None:
        """Offer one and the maximum feasible quantity of each reachable priced good, regardless of access."""
        for name, container in self._nearby_containers(observation):
            for identifier, price in container.get("prices", {}).items():
                item = short_id(identifier)
                args = {**self._xyz(container["pos"]), "item": item}
                for count, verdict in self._counts(agent, "buy", args):
                    self._add(candidates, agent, observation, "buy", {**args, "count": count},
                              f"buy {count} {item} from {self._container_name(name)} for {count * price} emeralds",
                              container["pos"], verdict)

    def _give(self, agent, observation, candidates) -> None:
        """Offer handing each inventory stack to another visible role within REACH."""
        for entity in self._players(agent, observation):
            if entity["distance"] <= REACH:
                for stack in observation["self"].get("inventory", []):
                    item, count = short_id(stack["item"]), stack["count"]
                    self._add(candidates, agent, observation, "give",
                              {"target": entity["name"], "item": item, "count": count},
                              f"hand {count} {item} to {entity['name']}", entity["pos"])

    def _drop(self, agent, observation, candidates) -> None:
        """Offer dropping each nonempty inventory stack at the actor's feet, subject to engine feasibility."""
        for stack in observation["self"].get("inventory", []):
            item, count = short_id(stack["item"]), stack["count"]
            if count > 0:
                self._add(candidates, agent, observation, "drop", {"item": item, "count": count},
                          f"drop {count} {item} on the ground")

    def _hold(self, agent, observation, candidates) -> None:
        """Offer holding each distinct carried item in sorted order, subject to engine feasibility."""
        items = sorted({item_id(stack["item"]) for stack in observation["self"].get("inventory", [])
                        if stack["count"] > 0})
        for identifier in items:
            item = short_id(identifier)
            self._add(candidates, agent, observation, "hold", {"item": item}, f"hold {item} where others can see it")

    def _eat(self, agent, observation, candidates) -> None:
        """Offer each held item once when enabled, letting the engine reject nonfood or otherwise infeasible acts."""
        if not self._data.get("daily_hunger", False):
            return
        items = sorted({short_id(stack["item"]) for stack in observation["self"].get("inventory", [])
                        if stack["count"] > 0})
        for item in items:
            self._add(candidates, agent, observation, "eat", {"item": item}, f"eat {item}")

    def _use(self, agent, observation, candidates) -> None:
        """Offer the nearest reachable, observed usable blocks, subject to the engine's vanilla-use checks."""
        blocks = [block for block in observation.get("blocks", []) if block["distance"] <= REACH
                  and (short_id(block["block"]) in {"bell", "lever"}
                       or short_id(block["block"]).endswith(("_door", "_trapdoor", "_fence_gate", "_button")))]
        offered = 0
        for block in sorted(blocks, key=lambda block: (block["distance"], tuple(block["pos"]))):
            target = tuple(block["pos"])
            if self._add(candidates, agent, observation, "use", self._xyz(target),
                         f"use the {short_id(block['block'])} at {target}", target):
                offered += 1
                if offered == NEAREST_BLOCKS:
                    break

    def _craft(self, agent, observation, candidates) -> None:
        """Offer one and the maximum feasible craft count for each sorted item in the schematic bills."""
        items = sorted({item_id(item) for schematic in self._data["schematics"].values() for item in schematic["bill"]})
        for identifier in items:
            item = short_id(identifier)
            for count, verdict in self._counts(agent, "craft", {"item": item}):
                self._add(candidates, agent, observation, "craft", {"item": item, "count": count},
                          f"craft {count} {item}", verdict=verdict)

    def _bake(self, agent, observation, candidates) -> None:
        """Offer explicit baking batches only when enabled, with oven, wheat and capacity checked by the engine."""
        if "bake" not in self._data.get("tools", ()):
            return
        for count, verdict in self._counts(agent, "bake", {}, maximum=32):
            self._add(candidates, agent, observation, "bake", {"count": count},
                      f"bake {count} bread at the oven using {3 * count} wheat", verdict=verdict)

    def _build_section(self, agent, observation, candidates) -> None:
        """Offer sections whose standing points are within REACH and whose complete bills the inventory covers."""
        inventory = Counter()
        for stack in observation["self"].get("inventory", []):
            inventory[item_id(stack["item"])] += stack["count"]
        for name, schematic in self._data["schematics"].items():
            if (math.dist(observation["self"]["pos"], schematic["stand"]) <= REACH
                    and all(inventory[item_id(item)] >= count for item, count in schematic["bill"].items())):
                bill = ", ".join(f"{count} {short_id(item)}" for item, count in schematic["bill"].items())
                origin = list(schematic["origin"])
                self._add(candidates, agent, observation, "build_section", {"schematic": name, "origin": origin},
                          f"build {name} ({bill})", origin)

    def _mine(self, agent, observation, candidates) -> None:
        """Offer breaking the nearest NEAREST_BLOCKS reachable notable blocks, excluding fluids and containers."""
        containers = {tuple(container["pos"]) for container in self._data["containers"].values()}
        blocks = [block for block in observation.get("blocks", []) if block["distance"] <= REACH
                  and block["block"] not in {"minecraft:water", "minecraft:lava"}
                  and tuple(block["pos"]) not in containers]
        for block in sorted(blocks, key=lambda block: (block["distance"], tuple(block["pos"])))[:NEAREST_BLOCKS]:
            target = tuple(block["pos"])
            self._add(candidates, agent, observation, "mine", self._xyz(target),
                      f"break the {short_id(block['block'])} at {target}", target)

    def _place(self, agent, observation, candidates) -> None:
        """Offer the nearest eight held schematic blocks not already observed built, plus the held fair lantern."""
        held = {short_id(stack["item"]) for stack in observation["self"].get("inventory", []) if stack["count"] > 0}
        observed = {tuple(block["pos"]): block["block"] for block in observation.get("blocks", [])}
        targets = []
        for schematic in self._data["schematics"].values():
            for block in schematic["blocks"]:
                target = tuple(schematic["origin"][axis] + block[offset]
                               for axis, offset in enumerate(("dx", "dy", "dz")))
                if observed.get(target) != block["block"]:
                    targets.append((target, short_id(block["block"])))
        targets.append((tuple(self._data["layout"]["fair"]["lantern"]), "lantern"))
        position = observation["self"]["pos"]
        eye = (position[0], position[1] + EYE_HEIGHT, position[2])
        reachable = {}
        for target, item in targets:
            distance = math.dist(eye, tuple(value + 0.5 for value in target))
            if item in held and distance <= REACH:
                reachable.setdefault((target, item), distance)
        nearest = sorted(reachable, key=lambda entry: (reachable[entry], entry))[:NEAREST_BLOCKS]
        for target, item in nearest:
            self._add(candidates, agent, observation, "place", {**self._xyz(target), "item": item},
                      f"place {item} at {target}", target)

    def _collect(self, agent, observation, candidates) -> None:
        """Offer one collection skill when any dropped-item entity is visible."""
        if any(entity["type"] == "minecraft:item" for entity in observation.get("entities", [])):
            self._add(candidates, agent, observation, "collect", {}, "pick up dropped items nearby")

    def _read_book(self, agent, observation, candidates) -> None:
        """Offer reading each written-book inventory slot, labeling it with the observed title and author."""
        for stack in observation["self"].get("inventory", []):
            if stack["item"] == "minecraft:written_book":
                document = stack.get("doc") or {}
                slot = stack["slot"]
                self._add(candidates, agent, observation, "read_book", {"slot": slot},
                          f"read the written book \"{document.get('title', '')}\" by {document.get('author', '?')} "
                          f"in slot {slot}")

    def _write_book(self, agent, observation, candidates) -> None:
        """Offer writing and signing each writable-book slot, without inventing title or pages."""
        for stack in observation["self"].get("inventory", []):
            if stack["item"] == "minecraft:writable_book":
                slot = stack["slot"]
                self._add(candidates, agent, observation, "write_book", {"slot": slot},
                          f"write and sign a book (writable book in slot {slot})")

    def _write_sign(self, agent, observation, candidates) -> None:
        """Offer text-free writing calls for every observed sign within REACH."""
        for document in observation.get("documents", []):
            if document["kind"] == "sign" and document["distance"] <= REACH:
                target = tuple(document["pos"])
                self._add(candidates, agent, observation, "write_sign", self._xyz(target),
                          f"write on the sign at {target}", target)

    def _ferry(self, agent, observation, candidates) -> None:
        """Offer travel to each other landing when the actor is within three blocks of a boarding point."""
        landings = self._data["layout"].get("ferry", {}).get("landings", {})
        for side, landing in landings.items():
            if math.dist(observation["self"]["pos"], landing["board"]) <= 3:
                for other, destination in landings.items():
                    if other != side:
                        self._add(candidates, agent, observation, "ferry", {"to": other},
                                  f"ride the boat to the {other} ferry landing", destination["board"])

    def _wait(self, agent, observation, candidates) -> None:
        """Always offer the three fixed waiting durations without checking skill feasibility."""
        for ticks in WAIT_TICKS:
            self._add(candidates, agent, observation, "wait", {"ticks": ticks}, f"wait {ticks} ticks")

    def _keep_going(self, agent, observation, candidates) -> None:
        """Offer continuing the observed current skill only while its state is running."""
        skill = observation["self"].get("skill") or {}
        if skill.get("state") == "running":
            self._add(candidates, agent, observation, "keep_going", {}, f"carry on with the current {skill['skill']}")
