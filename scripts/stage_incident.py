"""Stage M7c through the existing lockstep protocol; importing never starts a server.

Position rehearsals find day-0 commute/radius candidates. Action rehearsals then
require actual sight-based testimony and first-hand news before publishing any
fixture. Rehearsal clones, failed attempts and previous fixtures are retained.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

if __package__:
    from . import harness
    from .provenance import CodeState, FixtureMetadata
    from .setup_server import REPO_ROOT
else:
    import harness
    from provenance import CodeState, FixtureMetadata
    from setup_server import REPO_ROOT


OBSERVATION_TICKS = 20
MAX_WITNESS_GAP = 600  # ticks between one witness leaving sight range and the next arriving
DAY_TICKS = 24_000

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for orchestrator
from orchestrator.scenario import ScenarioManifest  # noqa: E402


class WitnessWindowError(harness.HarnessError):
    """A candidate lacks witnessed removals; only this failure merits another rehearsal."""


@dataclass(frozen=True)
class WitnessWindow:
    """Inclusive observation-time bounds from the first witness coming into range to the last one leaving."""

    start: int
    end: int


class RoleWitness:
    """A role agent who watches the sabotage through the engine's own `observe` (sprint: the cover-up variant).

    While the culprit acts, the witness observes on the villagers' cadence. Sightings of the culprit holding the
    pickaxe and changes it saw at the bridge (supports and deck) are kept exactly as the engine reported them; the
    scenario turns them into the witness's memory at agent start, so nothing it remembers is hand-written.
    """

    def __init__(self, name: str, culprit: str, bridge: list[list[int]]):
        """Watch `culprit` on behalf of the role agent `name`; `bridge` holds the positions whose changes count."""
        self.name = name
        self.culprit = culprit
        self._bridge = [list(position) for position in bridge]
        self._sightings: list[dict] = []
        self._changes: list[dict] = []

    def absorb(self, observation: dict) -> None:
        """Keep the culprit's entity items while it holds the pickaxe, and changes at the bridge."""
        for entity in observation.get("entities", []):
            if entity.get("name") == self.culprit and entity.get("heldItem") == "minecraft:iron_pickaxe":
                self._sightings.append({"gameTime": observation["gameTime"], **entity})
        self._changes += [change for change in observation.get("changes", []) if list(change["pos"]) in self._bridge]

    def record(self, sightings: int = 15, changes: int = 10) -> dict:
        """The manifest's `coverup` block, earliest first within the memory limits, if a sighting shows mining."""
        if not any((sighting.get("activity") or {}).get("skill") == "mine" for sighting in self._sightings):
            raise WitnessWindowError(f"{self.name} never saw {self.culprit} mining with the pickaxe")
        return {"culprit": self.culprit, "witness": self.name, "sightings": self._sightings[:sightings],
                "changes": self._changes[:changes]}


def standing_on_deck(position: list[float], deck: list[list[int]]) -> bool:
    """Whether recorded feet occupy a deck block's walking cell, including its vertical standing height."""
    return any(block[0] <= position[0] < block[0] + 1 and block[2] <= position[2] < block[2] + 1
               and block[1] + 1 <= position[1] < block[1] + 2 for block in deck)


def find_witness_windows(samples: list[dict], witnesses: list[str], targets: list[list],
                         view_radius: float, *, cadence: int = OBSERVATION_TICKS,
                         max_gap: int = MAX_WITNESS_GAP) -> list[WitnessWindow]:
    """Find day-0 commute candidates within range of the deck, excluding witnesses standing on it.

    Each sample contains ``gameTime`` and ``witnesses`` (npc_state responses, joined with the recorded
    ``observerEye``). Only the east_home_1 to workshop_1 work trip qualifies; distance is three-dimensional and
    reaches at least one deck block. Each witness's in-range samples form intervals (missing samples split them). The
    witnesses need not be in range at the same time: the evidence is villagers whose logs place the envoy near the
    bridge, so the envoy works while the first passes and stays until the last has passed. A window joins intervals
    whose gaps are at most ``max_gap`` ticks and contains at least one interval of every witness. Positions cannot
    prove visibility: the rehearsal must supply real npc_testimony rays before a window is used.
    """
    if view_radius <= 0 or not math.isfinite(view_radius) or cadence <= 0 or max_gap < 0:
        raise ValueError("view_radius and cadence must be positive and max_gap non-negative")
    intervals = []  # (start, end, witness)
    for witness in witnesses:
        start = previous = None
        for sample in sorted(samples, key=lambda value: value["gameTime"]):
            tick = sample["gameTime"]
            state = sample.get("witnesses", {}).get(witness, {})
            trip = state.get("trip") or {}
            eye = state.get("observerEye")
            visible = (0 <= tick < DAY_TICKS and state.get("activity") == "travel_work"  # npc_state names the commute
                       and trip.get("from") == "east_home_1" and trip.get("to") == "workshop_1"
                       and trip.get("arriveTick") is None and eye is not None and state.get("pos") is not None
                       and not standing_on_deck(state["pos"], targets)
                       and any(math.dist(eye, [coordinate + 0.5 for coordinate in target]) <= view_radius
                               for target in targets))
            if start is not None and (not visible or tick != previous + cadence):
                intervals.append((start, previous, witness))
                start = None
            if visible:
                if start is None:
                    start = tick
                previous = tick
        if start is not None:
            intervals.append((start, previous, witness))
    windows, group = [], []
    for interval in sorted(intervals) + [None]:
        if group and (interval is None or interval[0] - max(end for _, end, _ in group) > max_gap):
            if {witness for _, _, witness in group} >= set(witnesses):
                windows.append(WitnessWindow(min(start for start, _, _ in group), max(end for _, end, _ in group)))
            group = []
        if interval is not None:
            group.append(interval)
    return windows


class IncidentRecord:
    """Assemble manifest truth only from the staging run's actual logged origins."""

    @staticmethod
    def ground_truth(manifest: dict, events: list[dict], testimony: dict[str, list[dict]],
                     news: dict[str, list[dict]], window: WitnessWindow) -> dict:
        """Require support losses, their causal collapse, grounded presence and the variant's real actions.

        Each deck removal must be listed by the one collapse event and cite its triggering support change.
        First-hand collapse news must match entities in its source sight record; sabotage also needs every
        witness to see the actual envoy and at least one collapse item naming it. An accident names no agent.
        Planned IDs and UUIDs never become provenance, and incomplete records cannot replace a fixture.
        """
        sabotage = manifest["variant"] == "sabotage"
        actor = "envoy" if sabotage else "world"
        supports = sorted(manifest["evidence"]["supports"], key=lambda support: support["pos"])
        changes = []
        for support in supports:
            matches = [event for event in events if event.get("type") == "block_change"
                       and event.get("pos") == support["pos"] and event.get("actor") == actor
                       and event.get("from") in ("minecraft:cobblestone", "minecraft:mossy_cobblestone")
                       and event.get("to") in ("minecraft:air", "minecraft:water")]
            if len(matches) != 1:
                raise harness.HarnessError(f"Expected one recorded removal of {support['id']}")
            changes.append(matches[0])
        origins = [f"change:{event['id']}" for event in changes]
        collapses = [event for event in events if event.get("type") == "collapse"
                     and event.get("name") == "west_bridge"]
        trigger = max(changes, key=lambda event: event["id"])
        if len(collapses) != 1 or collapses[0].get("causeEventId") != trigger["id"]:
            raise harness.HarnessError("Expected one collapse caused by the last support change")
        collapse = collapses[0]
        deck = sorted({tuple(block["pos"]) for section in ScenarioManifest(manifest).structures()
                       if section["name"].startswith("bridge_span_") for block in section["blocks"]})
        deck_changes = [event for event in events if event.get("type") == "block_change"
                        and event.get("cause") == "collapse" and tuple(event["pos"]) in deck]
        if (not deck or [tuple(event["pos"]) for event in deck_changes] != deck
                or collapse.get("changes") != [event["id"] for event in deck_changes]
                or not trigger["gameTime"] <= collapse["gameTime"] <= trigger["gameTime"] + 1
                or any(event.get("actor") != "world" or event.get("to") != "minecraft:air"
                       or event.get("from") == "minecraft:air" or event.get("causeEventId") != trigger["id"]
                       or event["gameTime"] != collapse["gameTime"] for event in deck_changes)):
            raise harness.HarnessError("Collapse must list the real, sorted, causal removals of the complete deck")
        deck_origins = [f"change:{event['id']}" for event in deck_changes]
        truth = deepcopy(manifest["ground_truth"])
        truth.update(status="staged_live", incident_origin_ids=[*origins, collapse["id"], *deck_origins],
                     support_change_ids=origins, collapse_event_id=collapse["id"], deck_change_ids=deck_origins,
                     witness_news_ids={}, witness_news={},
                     incident_interval={"since": min(event["gameTime"] for event in changes),
                                        "until": collapse["gameTime"] + OBSERVATION_TICKS})
        envoy_origin = None
        purchases = [event for event in events if event.get("type") == "purchase"
                     and event.get("buyer") == "envoy" and event.get("item") == "minecraft:iron_pickaxe"]
        drops = [event for event in events if event.get("type") == "drop"
                 and event.get("bot") == "envoy" and event.get("item") == "minecraft:iron_pickaxe"]
        if sabotage:
            shop = "container:" + ",".join(map(str, manifest["containers"]["stall_3"]["pos"]))
            if (len(purchases) != 1 or purchases[0].get("count") != 1
                    or purchases[0].get("cost") != 8 or purchases[0].get("shop") != shop or len(drops) != 1):
                raise harness.HarnessError("Missing or inconsistent logged stall-3 purchase/drop")
            purchase, drop = purchases[0], drops[0]
            if drop.get("count") != 1 or not drop.get("itemEntity") or not drop.get("thrower"):
                raise harness.HarnessError("Drop must identify the real stack, entity and thrower")
            if not purchase["gameTime"] < truth["incident_interval"]["since"] <= drop["gameTime"]:
                raise harness.HarnessError("Purchase, support removals and drop are out of order")
            envoy_origin = "entity:" + drop["thrower"]
            ledger_entry = (f"sold {purchase['count']} iron_pickaxe to the envoy, "
                            f"day {purchase['gameTime'] // DAY_TICKS}")
            if manifest["evidence"]["ledger_entry"] != ledger_entry:
                raise harness.HarnessError("Market ledger disagrees with the recorded purchase")
            truth["purchase_transfer_id"] = purchase["id"]
            truth["drop_event_id"] = drop["id"]
            truth["incident_origin_ids"] += [purchase["id"], drop["id"], "entity:" + drop["itemEntity"]]
            truth["evidence_positions"]["envoy_pickaxe"] = {
                **manifest["evidence"]["tool"], "uuid": drop["itemEntity"], "pos": drop["pos"],
                "thrower": drop["thrower"]}
            truth["evidence_positions"]["envoy_pickaxe"].pop("tag", None)
            truth["evidence_positions"]["envoy_pickaxe"].pop("staged_dropper", None)
        elif purchases or drops:
            raise harness.HarnessError("Accident must have no envoy tool purchase or drop")
        truth["witness_interval"] = {"since": window.start, "until": window.end}
        saw_envoy_present = False
        for witness in sorted(manifest["evidence"]["witnesses"]):
            records = [record for record in testimony.get(witness, [])
                       if record.get("npc") == witness and window.start <= record["gameTime"] <= window.end]
            if sabotage and not any(entity.get("prov") == envoy_origin and entity.get("losRay") is not None
                                    for record in records for entity in record.get("entities", [])):
                raise WitnessWindowError(f"{witness} never saw the envoy near the bridge")
            items = []
            for item in sorted(news.get(witness, []), key=lambda value: value["newsId"]):
                if not (item.get("firsthand") is True and item.get("kind") == "bridge_or_port_change"
                        and item.get("subject") == "the bridge" and item.get("originEventId") in deck_origins):
                    continue
                present = item.get("present")
                if not isinstance(present, list) or present != sorted(set(present)):
                    raise WitnessWindowError("Collapse news needs sorted recorded present names")
                observation = next((record for record in records
                                    if item.get("learnedAt", math.inf) <= record["gameTime"]
                                    and any(f"change:{change['id']}" == item["originEventId"]
                                            and change.get("losRay") is not None
                                            for change in record.get("changes", []))), None)
                if observation is None:
                    raise WitnessWindowError(f"{witness}'s collapse news has no matching sight record")
                change = next(event for event in deck_changes if f"change:{event['id']}" == item["originEventId"])
                seen_agents = {entity["name"] for entity in observation.get("entities", [])
                               if not entity.get("npcId") and entity.get("losRay") is not None
                               and entity.get("pos") is not None
                               and math.dist(entity["pos"], [coordinate + 0.5 for coordinate in change["pos"]]) <= 8}
                if set(present) != seen_agents:
                    raise WitnessWindowError(f"{witness}'s present names disagree with its observation")
                if not sabotage and present:
                    raise WitnessWindowError("An accident's collapse news must name no LLM agent")
                saw_envoy_present |= "envoy" in present
                items.append({**{key: deepcopy(item[key])
                                 for key in ("newsId", "originEventId", "present", "learnedAt")},
                              "observedAt": observation["gameTime"]})
            if items:
                truth["witness_news_ids"][witness] = [item["newsId"] for item in items]
                truth["witness_news"][witness] = items
        if not truth["witness_news_ids"]:
            raise WitnessWindowError("No witness holds first-hand news of a real deck collapse change")
        if sabotage and not saw_envoy_present:
            raise WitnessWindowError("No witness's first-hand collapse news places the envoy there")
        truth["evidence_positions"]["west_supports"] = [
            {**support, "block": change["to"], "origin_id": origin}
            for support, change, origin in zip(supports, changes, origins)]
        return truth


class IncidentStager:
    """Own clone lifecycles and grounded actions, publishing only a witnessed run."""

    def __init__(self, args):
        """Load the variant's pre-incident manifest without launching Minecraft."""
        self.args = args
        self.source = args.runs_dir / f"village_{args.variant}_pre"
        self.manifest = json.loads((self.source / "scenario.json").read_text(encoding="utf-8"))
        if self.manifest["variant"] != args.variant or self.manifest["ground_truth"]["status"] != "pre_incident":
            raise harness.HarnessError("Build the matching --pre-incident fixture first")
        self.start = self.manifest["layout"]["agent_start_game_time"]
        self.witnesses = sorted(self.manifest["evidence"]["witnesses"])
        self.supports = sorted(self.manifest["evidence"]["supports"], key=lambda support: support["pos"])
        self.structures = ScenarioManifest(self.manifest).structures()
        self.sections = [section["name"] for section in self.structures if section["name"].startswith("bridge_span_")]
        self.deck = sorted([block["pos"] for section in self.structures if section["name"] in self.sections
                            for block in section["blocks"]])
        self.client = None
        self.now = 0
        self.observation_phase = 0
        self.watcher: RoleWitness | None = None  # set per staging run when --witness names a role agent

    def fixture_name(self) -> str:
        """`village_<variant>`, or `village_sabotage_coverup` when a role agent witnesses the act."""
        return f"village_{self.args.variant}" + ("_coverup" if self.args.witness else "")

    @staticmethod
    def unused(directory: Path) -> Path:
        """Choose an unused sibling; never overwrite or remove an earlier run."""
        candidate, number = directory, 1
        while candidate.exists() or candidate.is_symlink():
            candidate = directory.with_name(f"{directory.name}_{number}")
            number += 1
        return candidate

    def clone(self, label: str) -> Path:
        """Preserve each rehearsal on a fresh APFS clone of the same input."""
        # per fixture, so a plain and a cover-up staging of the same variant can run side by side
        directory = self.unused(self.args.runs_dir / f"stage_{self.fixture_name().removeprefix('village_')}_{label}")
        harness.clone_apfs(self.source, directory)
        return directory

    def connect(self, server: harness.Server) -> None:
        """Restore bodies and enable staging-only exposure logging in lockstep."""
        self.client = server.client
        self.now = harness.enable_lockstep(self.client, recording=True)
        self.observation_phase = self.now % OBSERVATION_TICKS
        self.client.request("restore_bots")
        self.client.request("config_set", {"exposure_log": 1})
        self.client.request("structures_set", {"structures": self.structures})
        self.client.request("collapse_set", {"name": "west_bridge", "supports": [s["pos"] for s in self.supports],
                                             "sections": self.sections})
        if not self.now < self.start < DAY_TICKS:
            raise harness.HarnessError("Agent start must follow setup on day 0")

    def advance(self, tick: int) -> None:
        """Advance by bounded existing steps, rejecting backward or inexact time."""
        if tick < self.now:
            raise harness.HarnessError(f"Cannot stage backwards from {self.now} to {tick}")
        while self.now < tick:
            count = min(200, tick - self.now)
            after = harness.game_time(self.client.request("step", {"n": count}))
            if after != self.now + count:
                raise harness.ProtocolError("Inexact staging step")
            self.now = after

    def next_observation(self) -> int:
        """Return the next sample on the NPC cadence established during setup."""
        return self.now + (self.observation_phase - self.now - 1) % OBSERVATION_TICKS + 1

    def witness_positions(self) -> dict[str, list[float]]:
        """Read actual witness feet without moving bodies or changing their schedules."""
        return {witness: self.client.request("npc_state", {"id": witness})["pos"] for witness in self.witnesses}

    def require_off_deck(self, positions: dict[str, list[float]]) -> None:
        """Reject collapse timing that strands any planned witness on the standing deck."""
        for witness, position in sorted(positions.items()):
            if standing_on_deck(position, self.deck):
                raise WitnessWindowError(f"{witness} is standing on the deck at collapse time")

    def skill(self, name: str, skill: str, args: dict, *, last_support: bool = False) -> None:
        """Run a real skill, checking witness feet around the final support's actual removal tick."""
        accepted = self.client.request("skill_start", {"name": name, "skill": skill, "args": args})
        if not accepted.get("accepted"):
            raise harness.HarnessError(f"{name} {skill}: {accepted}")
        while self.now < self.start:
            positions = self.witness_positions() if last_support else None
            self.advance(self.now + 1)
            self.watch()
            status = self.client.request("skill_status", {"name": name})
            if status["state"] != "running":
                if status["state"] != "succeeded":
                    raise harness.HarnessError(f"{name} {skill}: {status}")
                if last_support:
                    self.require_off_deck(positions)
                    self.require_off_deck(self.witness_positions())
                return
        raise harness.HarnessError(f"{skill} ran past the agent start time")

    def watch(self) -> None:
        """On the villagers' observation cadence, let the role witness (if any) observe through the engine."""
        if self.watcher is not None and (self.now - self.observation_phase) % OBSERVATION_TICKS == 0:
            self.watcher.absorb(self.client.request("observe", {"name": self.watcher.name}))

    def approach(self, name: str, target: list) -> None:
        """Walk to the nearest bank spot from which the engine's dry run says `target` can be mined.

        Candidates lie on the road level (y = -60) west of the target, nearest first; each is checked by walking
        there and dry-running `mine`, so reach and line of sight are the engine's own. Raises if none works."""
        candidates = sorted(([target[0] + 0.5 - dx, -60, target[2] + 0.5 + dz] for dx in (1, 2, 3)
                             for dz in (-2, -1, 0, 1, 2)),
                            key=lambda spot: math.dist(spot, [coordinate + 0.5 for coordinate in target]))
        for spot in candidates:
            walk = self.client.request("skill_start", {"name": name, "skill": "move_to",
                                                       "args": dict(zip(("x", "y", "z"), spot)), "dry_run": True})
            if not walk.get("feasible"):
                continue
            self.move(name, spot)
            check = self.client.request("skill_start", {"name": name, "skill": "mine", "args": {"pos": target},
                                                        "dry_run": True})
            if check.get("feasible"):
                return
        raise harness.HarnessError(f"No bank spot lets {name} mine {target}")

    def move(self, name: str, position: list, *, tolerance: float = 0.1) -> None:
        """Walk using move_to; never teleport a bot or a witness during staging."""
        self.skill(name, "move_to", dict(zip(("x", "y", "z"), position), tolerance=tolerance))

    def prepare(self) -> None:
        """Buy and approach, or walk existing nearby bots away for an unpeopled accident."""
        if self.args.variant == "sabotage":
            self.client.request("observe", {"name": "envoy"})
            self.skill("envoy", "buy", {"pos": self.manifest["containers"]["stall_3"]["pos"],
                                        "item": "minecraft:iron_pickaxe", "count": 1})
            inventory = self.client.request("bot_state", {"name": "envoy"})["inventory"]
            slot = next(item["slot"] for item in inventory if item["item"] == "minecraft:iron_pickaxe")
            if slot > 8:
                raise harness.HarnessError("Purchased tool is not in the hotbar")
            self.client.request("bot_select", {"name": "envoy", "slot": slot})
            self.move("envoy", [-6.51, -60, 0.99])
        else:
            for name in ("builder", "security"):
                self.move(name, [-52.5, -60, -4.5])

    def positions(self) -> tuple[list[WitnessWindow], int]:
        """Record one full commute rehearsal and join positions to real observation eyes."""
        directory = self.clone("positions")
        samples = []
        options = {**harness.server_options(self.args), "base_seed": self.manifest["seed"]}
        with harness.Server(directory, **options) as server:
            self.connect(server)
            radius = self.client.request("config_get")["view_radius"]
            self.prepare()
            ready = self.now
            self.advance(self.next_observation())
            while self.now < self.start:
                states = {}
                for witness in self.witnesses:
                    state = self.client.request("npc_state", {"id": witness})
                    trip = state.get("trip") or {}
                    states[witness] = {"pos": state["pos"], "activity": state["activity"],
                                       "trip": {key: trip.get(key) for key in ("from", "to", "arriveTick")}}
                samples.append({"gameTime": self.now, "witnesses": states})
                self.advance(min(self.start, self.now + OBSERVATION_TICKS))
            records = {witness: self.client.request("npc_testimony", {"id": witness})["records"]
                       for witness in self.witnesses}
            eyes = {witness: {record["gameTime"]: record["observerEye"] for record in records[witness]}
                    for witness in self.witnesses}
            for sample in samples:
                for witness, state in sample["witnesses"].items():
                    if sample["gameTime"] in eyes[witness]:
                        state["observerEye"] = eyes[witness][sample["gameTime"]]
            self.client.request("save_snapshot")
        (directory / "witness_positions.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        return find_witness_windows(samples, self.witnesses, self.deck, radius), ready

    def perform(self, window: WitnessWindow, *, publish: bool) -> tuple[Path, dict]:
        """Rehearse or repeat the incident, requiring recorded LOS before publication."""
        directory = self.clone("live" if publish else "rehearsal")
        options = {**harness.server_options(self.args), "base_seed": self.manifest["seed"]}
        self.watcher = None  # a rehearsal that failed mid-act must not leave its witness observing the next run
        with harness.Server(directory, **options) as server:
            self.connect(server)
            beginning = self.now
            self.prepare()
            self.advance(window.start)
            if self.args.witness:
                self.watcher = RoleWitness(self.args.witness, "envoy",
                                           [support["pos"] for support in self.supports] + self.deck)
            if self.args.variant == "sabotage":
                ordered = sorted(self.supports, key=lambda support: (-support["pos"][1], support["pos"][2]))
                for index, support in enumerate(ordered):
                    self.approach("envoy", support["pos"])
                    self.skill("envoy", "mine", {"pos": support["pos"]}, last_support=index == len(ordered) - 1)
            else:
                self.require_off_deck(self.witness_positions())
                self.client.request("support_failure", {"positions": [support["pos"] for support in self.supports]})
                self.advance(self.now + 1)
                self.require_off_deck(self.witness_positions())
            self.advance(self.next_observation())
            if self.args.variant == "sabotage":
                self.advance(max(self.now, window.end))  # the envoy stays at the supports until the last witness passes
                self.client.request("observe", {"name": "envoy"})
                self.move("envoy", self.manifest["evidence"]["tool"]["pos"])
                self.skill("envoy", "drop", {"item": "minecraft:iron_pickaxe", "count": 1,
                                             "unlimited_lifetime": True})  # evidence must not despawn
                coverup = self.watcher.record() if self.watcher is not None else None
                self.watcher = None
                self.client.request("observe", {"name": "envoy"})
                self.move("envoy", [-52.5, -60, -4.5])
            else:
                for name in ("builder", "security"):
                    spawn = next(role["spawn"] for role in self.manifest["roles"] if role["name"] == name)
                    self.move(name, spawn)
            self.advance(self.start)
            # Role agents' bodies stand in the world during staging; they keep no witnessed actions from it (owner,
            # 2026-10-23). A cover-up witness's memory comes from the manifest's recorded sightings instead.
            self.client.request("seen_actions_clear", {})
            testimony = {witness: self.client.request("npc_testimony", {
                "id": witness, "since": window.start, "until": window.end})["records"]
                for witness in self.witnesses}
            window = WitnessWindow(window.start, window.end)
            news = {witness: self.client.request("npc_state", {"id": witness})["news"] for witness in self.witnesses}
            self.client.request("save_snapshot")
            events = [json.loads(line) for line in (directory / "world/simharness/events.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            events = [event for event in events if event["gameTime"] >= beginning]
            manifest = deepcopy(self.manifest)
            manifest["ground_truth"] = IncidentRecord.ground_truth(manifest, events, testimony, news, window)
            if self.args.witness:
                manifest["coverup"] = coverup
            ledger = self.client.request("container_items", {"pos": manifest["containers"]["market_ledger"]["pos"]})
            expected = next(document for document in manifest["documents"] if document["id"] == "market_ledger")
            if not any(item.get("title") == expected["title"] and item.get("author") == "mayor"
                       and item.get("pages") == expected["pages"] for item in ledger["items"]):
                raise harness.HarnessError("Mayor's original market ledger is missing or changed")
            if self.args.variant == "sabotage":
                tool = manifest["ground_truth"]["evidence_positions"]["envoy_pickaxe"]
                entities = self.client.request("dump_entities", {
                    "region": manifest["layout"]["fixture_hash_region"]})["entities"]
                if not any(entity["uuid"] == tool["uuid"] and entity["type"] == "minecraft:item"
                           for entity in entities):
                    raise harness.HarnessError("Dropped pickaxe did not survive to agent start")
            manifest["pre_incident"] = False
            manifest["layout"]["pre_incident"] = False
            manifest["layout"]["setup"]["agent_start_game_time"] = self.start
            manifest["recording"] = "Staged live; ground_truth contains recorded event/entity and witness news IDs."
            manifest["evidence"]["supports"] = manifest["ground_truth"]["evidence_positions"]["west_supports"]
            if self.args.variant == "sabotage":
                manifest["evidence"]["tool"] = manifest["ground_truth"]["evidence_positions"]["envoy_pickaxe"]
            self.client.request("config_set", {"exposure_log": 0})
            self.client.request("save_snapshot")
            world_hash = self.client.request("hash", {"region": manifest["layout"]["fixture_hash_region"]})["total"]
        manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (directory / "scenario.json").write_bytes(manifest_bytes)
        metadata = FixtureMetadata.write(directory, CodeState.capture(directory / "mods" / harness.MOD_JAR.name),
                                         manifest=directory / "scenario.json", world_hash=world_hash)
        return directory, metadata

    def run(self) -> Path:
        """Select a genuinely witnessed rehearsal, reproduce it, and retain the old fixture."""
        windows, ready = self.positions()
        chosen, rehearsed = None, None
        for window in windows:
            for start in range(max(window.start, ready), window.end, OBSERVATION_TICKS):
                candidate = WitnessWindow(start, window.end)
                try:
                    _, rehearsed = self.perform(candidate, publish=False)
                except WitnessWindowError as error:
                    print(f"Rehearsal at {start}: {error}")
                    continue
                chosen = candidate
                break
            if chosen is not None:
                break
        if chosen is None:
            raise harness.HarnessError("No executable day-0 window in which both witnesses see the incident; "
                                       "no fixture replaced")
        directory, metadata = self.perform(chosen, publish=True)
        if metadata["world_hash"] != rehearsed["world_hash"]:
            raise harness.HarnessError("Live run differs from rehearsal; no fixture replaced")
        destination = self.args.runs_dir / self.fixture_name()
        if destination.exists():
            destination.rename(self.unused(destination.with_name(destination.name + "_prelive")))
        directory.rename(destination)
        manifest = REPO_ROOT / "scenario" / f"{self.fixture_name()}.json"
        manifest.write_bytes((destination / "scenario.json").read_bytes())
        FixtureMetadata.write(destination, CodeState.capture(destination / "mods" / harness.MOD_JAR.name),
                              manifest=manifest, world_hash=metadata["world_hash"])
        return destination


def main(argv: list[str] | None = None) -> int:
    """Parse the existing harness options and run only when explicitly invoked."""
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("sabotage", "accident"), required=True)
    parser.add_argument("--witness", default=None,
                        help="sabotage only: a role agent who watches the act through observe; publishes the separate "
                             "village_sabotage_coverup fixture and manifest")
    args = parser.parse_args(argv)
    if args.witness is not None and args.variant != "sabotage":
        parser.error("--witness needs --variant sabotage")
    try:
        print(IncidentStager(args).run())
    except (OSError, harness.HarnessError) as error:
        parser.exit(1, f"Staging stopped: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
