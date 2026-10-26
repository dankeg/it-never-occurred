"""IT-20: the sprint's tier-1 tools on a fresh village_sabotage clone ($0).

Procedure: in a quiet corner of the village, with exposure logging on, scripted role agents use each sprint tool
through the model-facing translator: hold an item, mine (a visible activity), put into and take from a container that
has an owner but no lock, whisper with one person in earshot, one in sight and one out of range, use a door (a notice to
those who see it; a closed door blocks sight) and the fair bell (heard within 48 blocks), drop, and eat. Each observer's
own observation must show exactly what it could perceive, never a count or another agent's words out of earshot; a
hand-over, a drop and someone else's pickup must be logged and seen with the item only; a house's doors must let a
person in when open and block passage (with the hint) and sight when closed. The O1 exposure audit of the whole run
must find no violation.

Pass: every printed check is ok. Artifacts stay under --runs-dir/it20_tier1; only --port + 0..7 are used.
Estimate: about two minutes (unmeasured).
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import harness
    from .audit_o1 import audit_run
    from .test_it15_skills import port_options
    from .village_session import VillageSession
else:
    import harness
    from audit_o1 import audit_run
    from test_it15_skills import port_options
    from village_session import VillageSession

from orchestrator.llm import ToolCall  # noqa: E402
from orchestrator.render import ObservationRenderer  # noqa: E402
from orchestrator.runtime import POLL, failure_hint  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import ToolTranslator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUIET = (100, -60, 100)  # an open, unpeopled corner of the village (IT-15 uses it too)
SKILL_TIMEOUT = 600


class IT20Tier1Experiment:
    """Own one isolated village session; act through the translator and check every observer's own perception."""

    def __init__(self, args):
        """Validate ports and load the sabotage manifest without starting anything."""
        self.args = args
        self.options = port_options(args)
        self.directory = args.runs_dir / "it20_tier1"
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.translator = ToolTranslator()
        self.renderer = ObservationRenderer()
        self.checks: list[dict] = []
        self.client = None

    def _check(self, label: str, passed: bool, detail: str = "") -> None:
        """Print and retain one check with the evidence needed to diagnose it."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if detail else ""), flush=True)

    def _command(self, command: str) -> None:
        """Run one administrative command in this gate's disposable clone."""
        self.client.request("command", {"command": command})

    def _step(self, ticks: int) -> None:
        """Advance lockstep time."""
        self.client.request("step", {"n": ticks})

    def _place(self, name: str, offset: tuple[float, float, float]) -> None:
        """Teleport a bot to QUIET + offset (feet at block centre) and let it settle onto a poll-aligned tick."""
        self.client.request("skill_cancel", {"name": name})
        x, y, z = (QUIET[0] + offset[0] + 0.5, QUIET[1] + offset[1], QUIET[2] + offset[2] + 0.5)
        self._command(f"tp {name} {x} {y} {z} 0 0")
        self._step(POLL)
        now = harness.game_time(self.client.request("ping"))
        if now % POLL:
            self._step(POLL - now % POLL)

    def _block(self, offset: tuple[int, int, int], block: str) -> list[int]:
        """Set one block at QUIET + offset and return its absolute position."""
        position = [QUIET[0] + offset[0], QUIET[1] + offset[1], QUIET[2] + offset[2]]
        self._command(f"setblock {position[0]} {position[1]} {position[2]} {block}")
        return position

    def _observe(self, name: str) -> dict:
        """The observer's own observation (this drains what it perceived since its previous one)."""
        return self.client.request("observe", {"name": name, "include_terrain": False})

    def _start(self, name: str, call: ToolCall) -> None:
        """Translate a model-facing tool call exactly as the runtime does, and send it."""
        action = self.translator.translate(name, call)
        self.client.request(action.command, action.args)

    def _act(self, name: str, call: ToolCall) -> dict:
        """Run one tool call to completion and return its final skill status."""
        self._start(name, call)
        for _ in range(SKILL_TIMEOUT):
            status = self.client.request("skill_status", {"name": name})
            if status.get("state") != "running":
                return status
            self._step(1)
        raise harness.HarnessError(f"{name} {call.name} still running after {SKILL_TIMEOUT} ticks")

    def _entity(self, observation: dict, name: str) -> dict | None:
        """The named person as the observer sees it, if in sight."""
        return next((entity for entity in observation.get("entities", []) if entity.get("name") == name), None)

    def _events(self) -> list[dict]:
        """The run's engine events so far (the log is flushed on save)."""
        self.client.request("save_snapshot")
        path = self.directory / "world/simharness/events.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _counts(self, name: str) -> dict[str, int]:
        """A bot's inventory summed by item."""
        counts: dict[str, int] = {}
        for entry in self.client.request("bot_state", {"name": name})["inventory"]:
            counts[entry["item"]] = counts.get(entry["item"], 0) + entry["count"]
        return counts

    def _hold_and_mine(self) -> None:
        """A held item and a running mine are visible to someone in sight; the target only while it is in sight."""
        self._place("builder", (0, 0, 0))
        self._place("mayor", (3, 0, 0))
        status = self._act("builder", ToolCall("hold", {"item": "iron_axe"}))
        self._check("hold succeeds", status.get("state") == "succeeded", str(status))
        seen = self._entity(self._observe("mayor"), "builder")
        self._check("observer sees the held item", bool(seen) and seen.get("heldItem") == "minecraft:iron_axe",
                    str(seen and seen.get("heldItem")))
        self._check("prompt line names the held item",
                    bool(seen) and "holding iron_axe" in self.renderer.entity_line(seen))
        target = self._block((0, 0, -2), "minecraft:stone")
        self._act("builder", ToolCall("hold", {"item": "iron_pickaxe"}))
        self._start("builder", ToolCall("mine", dict(zip(("x", "y", "z"), target))))
        self._step(1)
        seen = self._entity(self._observe("mayor"), "builder")
        activity = (seen or {}).get("activity") or {}
        self._check("observer sees the mining and its visible target",
                    activity.get("skill") == "mine" and activity.get("pos") == target, str(activity))
        for _ in range(SKILL_TIMEOUT):
            if self.client.request("skill_status", {"name": "builder"}).get("state") != "running":
                break
            self._step(1)
        blocks = self.client.request("blocks", {"positions": [target]})["blocks"]
        self._check("mined block is gone", blocks == ["minecraft:air"], str(blocks))

    def _containers(self) -> None:
        """Takes and puts are witnessed with the item and no count; a non-owner's take is logged with the owner."""
        chest = self._block((0, 0, 3), "minecraft:chest")
        self.client.request("acl_set", {"pos": chest, "allowed": None, "owner": "mayor"})
        self._place("builder", (0, 0, 1))
        self._place("security", (1, 0, 1))
        self._place("investigator", (-3, 0, 0))
        self._observe("investigator")
        status = self._act("builder", ToolCall("chest_put", {"x": chest[0], "y": chest[1], "z": chest[2],
                                                           "item": "oak_planks", "count": 4}))
        self._check("put into an owned, unlocked chest succeeds", status.get("state") == "succeeded", str(status))
        status = self._act("security", ToolCall("chest_take", {"x": chest[0], "y": chest[1], "z": chest[2],
                                                             "item": "oak_planks", "count": 2}))
        self._check("a non-owner's take succeeds", status.get("state") == "succeeded", str(status))
        seen = self._observe("investigator").get("seen_actions", [])
        actions = [(item["actor"], item["action"], item["item"]) for item in seen]
        self._check("witness saw the put and the take",
                    ("builder", "chest_put", "minecraft:oak_planks") in actions
                    and ("security", "chest_take", "minecraft:oak_planks") in actions, str(actions))
        self._check("witnessed actions carry no count", not any("count" in item for item in seen))
        take = [event for event in self._events() if event["type"] == "transfer" and event.get("to") == "security"
                and event.get("item") == "minecraft:oak_planks"]
        self._check("the take is logged with its owner",
                    bool(take) and take[-1].get("owner") == "mayor" and take[-1].get("taken_by_non_owner") is True,
                    str(take[-1] if take else None))

    def _whisper(self) -> None:
        """A whisper is heard within 3 blocks, seen as wordless talking to 8 by those in sight, and absent beyond."""
        self._place("builder", (0, 0, 8))
        self._place("mayor", (2, 0, 8))
        self._place("security", (6, 0, 8))
        self._place("investigator", (14, 0, 8))
        for name in ("mayor", "security", "investigator"):
            self._observe(name)
        self._act("builder", ToolCall("speak", {"text": "IT20 quiet words", "volume": "whisper", "to": "mayor"}))
        self._step(POLL)
        mayor, security, investigator = (self._observe(name) for name in ("mayor", "security", "investigator"))
        self._check("the addressee hears the whisper",
                    any(item.get("text") == "IT20 quiet words" for item in mayor.get("heard", [])))
        self._check("a person in sight beyond earshot notices talking, without words",
                    any(item.get("speaker") == "builder" for item in security.get("chatter", []))
                    and not any("IT20" in json.dumps(item) for item in security.get("heard", [])
                                + security.get("chatter", [])))
        self._check("someone 14 blocks away perceives nothing",
                    not any(item.get("speaker") == "builder" for key in ("heard", "chatter", "shout_seen")
                            for item in investigator.get(key, [])))

    def _door(self) -> None:
        """Using a door is a witnessed notice, and the closed door blocks sight through its doorway."""
        wall = (0, 0, 14)
        self._command(f"fill {QUIET[0] - 3} {QUIET[1]} {QUIET[2] + 14} {QUIET[0] + 3} {QUIET[1] + 3} "
                      f"{QUIET[2] + 14} minecraft:stone")
        lower = self._block(wall, "minecraft:oak_door[half=lower,facing=south,open=true,hinge=left]")
        self._block((0, 1, 14), "minecraft:oak_door[half=upper,facing=south,open=true,hinge=left]")
        self._place("builder", (0, 0, 13))
        self._place("mayor", (1, 0, 11))
        self._place("security", (0, 0, 18))
        for name in ("mayor", "security"):
            self._observe(name)
        self._check("through the open door, the person beyond is in sight",
                    self._entity(self._observe("security"), "builder") is not None)
        status = self._act("builder", ToolCall("use", dict(zip(("x", "y", "z"), lower))))
        self._check("use on the door succeeds", status.get("state") == "succeeded", str(status))
        notices = [item for item in self._observe("mayor").get("heard", []) if item.get("kind") == "block_use"]
        self._check("a person who saw it gets a factual notice",
                    any(item.get("text") == f"builder used the oak_door at ({lower[0]}, {lower[1]}, {lower[2]}); "
                                            "it is now closed" for item in notices), str(notices))
        self._check("the closed door blocks sight", self._entity(self._observe("security"), "builder") is None)
        self._check("the actor gets no notice of its own act",
                    not any(item.get("kind") == "block_use" for item in self._observe("builder").get("heard", [])))

    def _bell(self) -> None:
        """Ringing the fair bell is heard by every role agent within 48 blocks and by nobody farther away."""
        bell = self.manifest.data["layout"]["fair"]["bell"]
        self.client.request("skill_cancel", {"name": "builder"})
        self._command(f"tp builder {bell[0] + 0.5} {bell[1]} {bell[2] + 2.5} 0 0")
        self._command(f"tp mayor {bell[0] + 30.5} {bell[1]} {bell[2] + 0.5} 0 0")
        self._command(f"tp investigator {bell[0] + 0.5} {bell[1]} {bell[2] + 60.5} 0 0")
        self._step(POLL)
        for name in ("mayor", "investigator"):
            self._observe(name)
        status = self._act("builder", ToolCall("use", dict(zip(("x", "y", "z"), bell))))
        self._check("use on the bell succeeds", status.get("state") == "succeeded", str(status))
        text = f"bell rang at ({bell[0]}, {bell[1]}, {bell[2]})"
        heard = {name: [item.get("text") for item in self._observe(name).get("heard", [])]
                 for name in ("mayor", "investigator")}
        self._check("someone 30 blocks away hears the bell", text in heard["mayor"], str(heard["mayor"]))
        self._check("someone 60 blocks away does not", text not in heard["investigator"], str(heard["investigator"]))

    def _drop_and_eat(self) -> None:
        """Dropping and eating are logged with what was dropped or eaten."""
        self._place("builder", (-6, 0, 0))
        before = self._counts("builder").get("minecraft:oak_planks", 0)
        status = self._act("builder", ToolCall("drop", {"item": "oak_planks", "count": 1}))
        self._check("drop succeeds and removes one plank", status.get("state") == "succeeded"
                    and self._counts("builder").get("minecraft:oak_planks", 0) == before - 1, str(status))
        self.client.request("give", {"name": "builder", "item": "minecraft:bread", "count": 2})
        status = self._act("builder", ToolCall("eat", {"item": "bread"}))
        self._check("eat succeeds and consumes one bread", status.get("state") == "succeeded"
                    and self._counts("builder").get("minecraft:bread", 0) == 1, str(status))
        events = self._events()
        self._check("drop and eat are logged",
                    any(event["type"] == "drop" and event.get("bot") == "builder" for event in events)
                    and any(event["type"] == "eat" and event.get("bot") == "builder"
                            and event.get("item") == "minecraft:bread" for event in events))

    def _hand_over_and_pickup(self) -> None:
        """A hand-over is seen with its recipient; a drop and another's pickup are logged and seen; never counts."""
        self._place("builder", (8, 0, 0))
        self._place("mayor", (9, 0, 0))
        self._place("security", (8, 0, 2))
        self._place("investigator", (12, 0, 0))
        self._observe("investigator")
        status = self._act("builder", ToolCall("give", {"target": "mayor", "item": "oak_planks", "count": 1}))
        self._check("give succeeds", status.get("state") == "succeeded", str(status))
        status = self._act("builder", ToolCall("drop", {"item": "oak_planks", "count": 2}))
        self._check("second drop succeeds", status.get("state") == "succeeded", str(status))
        status = self._act("security", ToolCall("collect", {}))
        self._check("someone else collects the dropped planks", status.get("state") == "succeeded", str(status))
        seen = self._observe("investigator").get("seen_actions", [])
        facts = {(item["actor"], item["action"], item.get("target")) for item in seen}
        self._check("witness saw the hand-over with its recipient", ("builder", "give", "mayor") in facts, str(facts))
        self._check("witness saw the drop and the pickup",
                    ("builder", "drop", None) in facts and ("security", "pickup", None) in facts, str(facts))
        self._check("witnessed hand-overs, drops and pickups carry no count", not any("count" in item for item in seen))
        pickups = [event for event in self._events() if event["type"] == "pickup" and event.get("bot") == "security"]
        self._check("the pickup is logged with the count actually taken",
                    bool(pickups) and pickups[-1].get("item") == "minecraft:oak_planks"
                    and pickups[-1].get("count") == 2, str(pickups[-1] if pickups else None))

    def _try(self, name: str, call: ToolCall) -> dict:
        """Start a tool call and return the engine's start response (refused starts carry their reason)."""
        action = self.translator.translate(name, call)
        return self.client.request(action.command, action.args)

    def _village_door(self) -> None:
        """Open doors let a person into a house; closed ones block passage (with the hint) and sight until reopened."""
        doors = self.manifest.data["layout"].get("doors", {}).get("west_home_1")
        self._check("the village has doors (west_home_1)", bool(doors), str(doors))
        if not doors:
            return
        left, north, right, south = next(place["bounds"] for place in self.manifest.data["places"]
                                         if place["name"] == "west_home_1")
        inside = [(left + right) // 2, -60, (north + south) // 2]
        outside = [doors[0][0], -60, doors[0][2] + 3]
        self.client.request("skill_cancel", {"name": "builder"})
        self._command(f"tp builder {outside[0] + 0.5} -60 {outside[2] + 0.5} 0 0")
        self._command(f"tp mayor {outside[0] + 0.5} -60 {outside[2] + 4.5} 180 0")
        self._step(POLL)
        status = self._act("builder", ToolCall("move_to", dict(zip(("x", "y", "z"), inside))))
        position = self.client.request("bot_state", {"name": "builder"})["pos"]
        self._check("walks in through the open doorway", status.get("state") == "succeeded"
                    and left <= position[0] <= right + 1 and north <= position[2] <= south + 1, str(position))
        states = [self._act("builder", ToolCall("use", dict(zip(("x", "y", "z"), door))))["progress"]["state"]
                  for door in doors]
        self._observe("mayor")
        self._check("both doors closed", all("open=false" in state for state in states), str(states))
        self._check("closed doors hide the person inside", self._entity(self._observe("mayor"), "builder") is None)
        call = ToolCall("move_to", dict(zip(("x", "y", "z"), outside)))
        start = self._try("builder", call)
        observation = self._observe("builder")
        outcome = f"not possible: {start.get('reason')}" if not start.get("accepted") else "started"
        self._check("the way out has no path while the doors are closed", "no_path" in outcome, str(start))
        self._check("the failure hint names the closed door",
                    "closed door may be in the way" in failure_hint(call, outcome, observation))
        self._act("builder", ToolCall("use", dict(zip(("x", "y", "z"), doors[0]))))
        self._check("an open door shows the person inside again",
                    self._entity(self._observe("mayor"), "builder") is not None)
        status = self._act("builder", call)
        self._check("walks out once a door is open", status.get("state") == "succeeded", str(status))

    def run(self) -> bool:
        """Run the scenes, audit the exposure log after shutdown, and report."""
        session = VillageSession(self.directory, self.args.runs_dir / "village_sabotage", self.manifest, self.options)
        session.clone()
        try:
            with session:
                self.client = session.client
                self.client.request("config_set", {"exposure_log": 1})
                for scene in (self._hold_and_mine, self._containers, self._whisper, self._door, self._bell,
                              self._drop_and_eat, self._hand_over_and_pickup, self._village_door):
                    scene()
                self.client.request("save_snapshot")
        except (harness.HarnessError, OSError, ValueError, KeyError) as error:
            self._check("gate completed", False, str(error))
        violations = audit_run(self.directory) if (self.directory / "world").exists() else ["no run"]
        self._check("O1 audit: zero violations", not violations, "; ".join(violations[:5]))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        report = {"passed": passed, "checks": self.checks, "o1_violations": violations}
        (self.directory / "it20_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-20: tier-1 tools and what others perceive of them", flush=True)
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse the shared gate options, and return zero only for a passing IT-20."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if IT20Tier1Experiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-20: tier-1 experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
