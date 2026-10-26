"""Observation-grounded evidence exposure, villager tracks, and decision features."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .candidates import Candidate
from .scenario import ScenarioManifest

EYE_HEIGHT = 1.62


class EvidenceExposure:
    """Keep the earliest tick at which each agent learned each evidence item."""

    ITEMS = ("west_supports", "market_ledger", "witness_observations")

    def __init__(self, manifest: ScenarioManifest, known: dict | None = None):
        """Read evidence locations and villager names, optionally restoring known times."""
        evidence = manifest.data["evidence"]
        self._supports = {tuple(support["pos"]) for support in evidence["supports"]}
        self._ledger_entry = evidence["ledger_entry"].lower()
        self._villagers = {npc["name"] for npc in manifest.data["npcs"]}
        self._known = {agent: dict(items) for agent, items in (known or {}).items()}

    def _learn(self, agent: str, item: str, game_time: int) -> None:
        """Record an exposure without replacing an earlier known time."""
        known = self._known.setdefault(agent, {})
        known[item] = min(known.get(item, game_time), game_time)

    def observe(self, agent: str, observation: dict) -> None:
        """Learn from visible support positions and heard villager testimony."""
        if any(tuple(block["pos"]) in self._supports for block in observation.get("blocks", [])):
            self._learn(agent, "west_supports", observation["gameTime"])
        for item in observation.get("heard", []):
            if item.get("kind") in {"block_use", "transfer"}:
                continue
            text = item["text"].lower()
            if item["speaker"] in self._villagers and "envoy" in text and "bridge" in text:
                self._learn(agent, "witness_observations", item["gameTime"])

    def read(self, agent: str, text: str, game_time: int) -> None:
        """Make a matching ledger entry known only after the reading decision."""
        if self._ledger_entry in text.lower():
            self._learn(agent, "market_ledger", game_time + 1)

    def items(self, agent: str, at: int) -> frozenset[str]:
        """Return the agent's evidence items known at or before the requested tick."""
        return frozenset(item for item, known_from in self._known.get(agent, {}).items() if known_from <= at)

    def to_json(self) -> dict:
        """Return an independent JSON-ready snapshot of the exposure times."""
        return {agent: dict(items) for agent, items in self._known.items()}


class VillagerTracks:
    """Query each villager's latest logged position and sightings at a tick."""

    def __init__(self, records: Iterable[dict]):
        """Group and sort observation records for binary-search time lookups."""
        self._records: dict[str, list[dict]] = {}
        for record in records:
            self._records.setdefault(record["npc"], []).append(record)
        self._times = {}
        for npc, track in self._records.items():
            track.sort(key=lambda record: record["gameTime"])
            self._times[npc] = [record["gameTime"] for record in track]

    @classmethod
    def load(cls, path: Path) -> VillagerTracks:
        """Load one villager observation JSON record per line from a local file."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.loads(line) for line in handle)

    def _latest(self, npc: str, game_time: int) -> dict | None:
        """Find the last record at or before the tick, never a future record."""
        index = bisect_right(self._times.get(npc, []), game_time) - 1
        return self._records[npc][index] if index >= 0 else None

    def position(self, npc: str, game_time: int) -> tuple[float, float, float] | None:
        """Return logged feet coordinates, subtracting eye height from the y coordinate."""
        record = self._latest(npc, game_time)
        if record is None:
            return None
        eye = record["observerEye"]
        return float(eye[0]), float(eye[1]) - EYE_HEIGHT, float(eye[2])

    def watchers(self, name: str, game_time: int) -> frozenset[str]:
        """Return villagers whose latest observation lists an entity with this name."""
        watchers = set()
        for npc in self._records:
            record = self._latest(npc, game_time)
            if record is not None and any(entity.get("name") == name for entity in record.get("entities", [])):
                watchers.add(npc)
        return frozenset(watchers)

    def near(self, point, radius: float, game_time: int) -> frozenset[str]:
        """Return villagers with logged feet within the inclusive Euclidean radius."""
        nearby = set()
        for npc in self._records:
            position = self.position(npc, game_time)
            if position is not None and math.dist(position, point) <= radius:
                nearby.add(npc)
        return frozenset(nearby)


class PerceptionTap:
    """Pass harness requests through while recording observations and book exposure."""

    def __init__(self, client, exposure: EvidenceExposure,
                 on_observe: Callable[[str, dict], None] | None = None):
        """Store the client and optional callback, starting with no observations."""
        self._client = client
        self._exposure = exposure
        self._on_observe = on_observe
        self._now = 0
        self.latest: dict[str, dict] = {}

    def request(self, command: str, args: dict | None = None) -> dict:
        """Forward a request unchanged, then tap successful observation or read responses."""
        response = self._client.request(command, args)
        if command == "observe":
            name = args["name"]
            self.latest[name] = response
            self._now = response["gameTime"]
            self._exposure.observe(name, response)
            if self._on_observe is not None:
                self._on_observe(name, response)
        elif command == "read_item_doc":
            text = response.get("title", "") + " " + " | ".join(response.get("pages", []))
            self._exposure.read(args["name"], text, self._now)
        return response


@dataclass(frozen=True)
class DecisionContext:
    """The actor, tick, raw agent observations, exposure, and villager tracks for a decision."""

    agent: str
    game_time: int
    observations: Mapping[str, dict]
    exposure: EvidenceExposure
    villagers: VillagerTracks


class PerceptionFeatures:
    """Compute the eight perception features and five consequence-predictor features."""

    NAMES = ("earshot", "seen_by_agents", "agents_near_target", "seen_by_villagers", "nearest_agent",
             "free_text", "private_evidence", "addressed_to_agent", "relay_reach", "ticks_to_perception",
             "actor_informed", "perceivers_informed", "goal_relevance")
    PHI = NAMES[:8]

    def __init__(self, manifest: ScenarioManifest, horizon: int = 2400):
        """Cache each predicate's named and spatial targets and the logged-relay horizon."""
        self._horizon = horizon
        self._agents = {role["name"] for role in manifest.roles}
        self._containers = manifest.data["containers"]
        self._schematics = manifest.data["schematics"]
        self._regions = [landmark["region"] for landmark in manifest.data["landmarks"]]
        self._goal_names: dict[str, set[str]] = {}
        self._goal_positions: dict[str, set[tuple]] = {}
        self._incident_roles: set[str] = set()
        for role, predicate in manifest.data["success_predicates"].items():
            self._goal_names[role] = set()
            self._goal_positions[role] = set()
            self._collect_targets(role, predicate)

    def features(self, candidate: Candidate, context: DecisionContext) -> dict[str, float]:
        """Compute all thirteen floats from the supplied observations and logged tracks."""
        actor, now = context.agent, context.game_time
        actor_position = context.observations[actor]["self"]["pos"]
        others = {name: observation["self"]["pos"] for name, observation in context.observations.items()
                  if name != actor}
        seers = {name for name in others
                 if any(entity.get("type") == "minecraft:player" and entity.get("name") == actor
                        for entity in context.observations[name].get("entities", []))}
        call = candidate.call
        speech = call.name == "speak"
        radius = {"whisper": 3, "shout": 24}.get(call.args.get("volume", "talk"), 8)
        hearers = {name for name, position in others.items() if math.dist(actor_position, position) <= radius}
        perceivers = hearers if speech else seers
        watchers = context.villagers.watchers(actor, now)
        villager_perceivers = context.villagers.near(actor_position, radius, now) if speech else watchers
        nearest = min((math.dist(actor_position, position) for position in others.values()), default=64)
        ticks = 0 if perceivers else min(self._horizon, min(
            (20 * max(0, math.dist(position, candidate.target) - 24) / 4 for position in others.values()),
            default=self._horizon))
        addressee = call.args.get("to", "") if speech else call.args.get("target", "") if call.name == "give" else ""
        values = {
            "earshot": len(hearers) if speech else 0,
            "seen_by_agents": len(seers),
            "agents_near_target": sum(math.dist(position, candidate.target) <= 24 for position in others.values()),
            "seen_by_villagers": len(watchers),
            "nearest_agent": min(64, nearest) / 64,
            "free_text": call.name in {"speak", "write_book", "write_sign"},
            "private_evidence": call.name in {"open_container", "chest_take", "chest_put", "read_book",
                                              "write_book", "write_sign"},
            "addressed_to_agent": addressee.strip().lower() in self._agents,
            "relay_reach": self._relay_reach(context, others, villager_perceivers),
            "ticks_to_perception": ticks,
            "actor_informed": bool(context.exposure.items(actor, now)),
            "perceivers_informed": sum(bool(context.exposure.items(name, now)) for name in perceivers),
            "goal_relevance": self._goal_relevance(candidate, perceivers),
        }
        return {name: float(value) for name, value in values.items()}

    def _relay_reach(self, context: DecisionContext, positions: dict, villagers: frozenset[str]) -> int:
        """Count distinct other agents reached by current villager perceivers at 200-tick steps."""
        reached = set()
        for game_time in range(context.game_time + 200, context.game_time + self._horizon + 1, 200):
            for npc in villagers:
                position = context.villagers.position(npc, game_time)
                if position is not None:
                    reached.update(name for name, target in positions.items() if math.dist(position, target) <= 8)
        return len(reached)

    def _act_targets(self, candidate: Candidate) -> tuple[set[str], set[tuple]]:
        """Extract only the act names and block positions specified for goal relevance."""
        call = candidate.call
        if call.name == "speak":
            return {call.args.get("to", "").strip().lower()}, set()
        if call.name == "give":
            return {call.args["target"].strip().lower()}, set()
        if call.name in {"open_container", "chest_take", "chest_put", "buy", "mine", "place", "use", "write_sign"}:
            return set(), {tuple(call.args[axis] for axis in ("x", "y", "z"))}
        if call.name == "build_section":
            origin = call.args["origin"]
            blocks = self._schematics[call.args["schematic"]]["blocks"]
            return set(), {tuple(origin[axis] + block[offset] for axis, offset in enumerate(("dx", "dy", "dz")))
                           for block in blocks}
        return set(), set()

    def _goal_relevance(self, candidate: Candidate, perceivers: set[str]) -> bool:
        """Check act targets against predicates belonging to agents who perceive it now."""
        names, positions = self._act_targets(candidate)
        for role in perceivers:
            if names & self._goal_names.get(role, set()) or positions & self._goal_positions.get(role, set()):
                return True
            if role in self._incident_roles and any(
                    all(lower <= coordinate <= upper
                        for coordinate, lower, upper in zip(position, region["min"], region["max"]))
                    for position in positions for region in self._regions):
                return True
        return False

    def _collect_targets(self, role: str, predicate) -> None:
        """Recursively cache predicate names, exact block targets, and incident-region ownership."""
        if isinstance(predicate, list):
            for part in predicate:
                self._collect_targets(role, part)
        elif isinstance(predicate, dict):
            positions = self._goal_positions[role]
            for key, value in predicate.items():
                if key in {"name", "from", "to", "culprit"} and isinstance(value, str):
                    name = value.strip().lower()
                    if name in self._agents:
                        self._goal_names[role].add(name)
                    if key in {"from", "to"}:
                        container = re.fullmatch(r"container:(-?\d+),(-?\d+),(-?\d+)", name)
                        if container is not None:
                            positions.add(tuple(int(coordinate) for coordinate in container.groups()))
                if key == "blocks_present":
                    positions.update(tuple(block["pos"]) for block in value)
                elif key == "block_beside":
                    position = value["pos"]
                    positions.update((position[0] + delta_x, position[1], position[2] + delta_z)
                                     for delta_x, delta_z in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                elif key == "book_query":
                    positions.add(tuple(self._containers[value["container"]]["pos"]))
                elif key == "report_query":
                    positions.add(tuple(self._containers["records"]["pos"]))
                elif key == "incident_query":
                    self._incident_roles.add(role)
                self._collect_targets(role, value)
