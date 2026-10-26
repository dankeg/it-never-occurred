"""IT-15: skills, feedback and night rest on a fresh village_sabotage clone ($0).

Procedure: settle the builder on the village's existing 15/16-high west-spine dirt path, walk two road legs,
approach an actual chest, mine a forest log then collect, ask absent villagers, attempt 650-character speech,
give to an adjacent agent, and build a real bridge section without and with its material bill. Check successful
speech with and without a listener, then keep an idle agent through a whole night with one addressed wake-up.
Every action runs through SimulationRuntime and has its outcome checked in the same agent's very next prompt;
every rejection must include failure_hint there. Physical effects and thinking/rest timing are checked too.

Pass: every printed check is ok. Artifacts stay under --runs-dir/it15_skills; only --port + 0..7 are permitted.
Estimate: about two minutes, unmeasured until the owner runs this server gate (suggested timeout: six minutes).
Negative control: the owner will revert the feet-position fix (e714311) on a scratch branch and rerun. The road
case requires actual dirt_path blocks and settled feet at surface_y + 15/16 before walking, not gravel collars
or a teleport at integer height. That revert check has NOT been run here; this gate never reverts the fix.
Revert check (2026-10-18): fails at e714311^ (feet position on 15/16-high paths).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

if __package__:
    from . import harness
    from .village_session import VillageSession
else:
    import harness
    from village_session import VillageSession

from orchestrator.decisions import Decision  # noqa: E402
from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.runtime import NIGHT_REST, POLL, THINKING_TICKS, SimulationRuntime, failure_hint  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROAD_CELLS = ((-12, 16), (-12, 24), (-12, 32))
QUIET = (100.5, -60, 100.5)
WAKE_TEXT = "IT15 night wake"
OVERLONG_SPEECH = "word " * 130
ACTION_TIMEOUT = 4_000


def port_options(args) -> dict:
    """Keep both listeners inside the gate's eight-port allocation, never using the harness's default 25565."""
    options = harness.server_options(args)
    if not 1 <= args.port <= 65528:
        raise ValueError("--port must leave room for its eight-port allocation (1..65528)")
    game_port = args.port + 1 if args.game_port is None else args.game_port
    if not args.port <= game_port <= args.port + 7 or game_port == args.port:
        raise ValueError("--game-port must differ from --port and lie inside --port + 0..7")
    options["game_port"] = game_port
    return options


def on_dirt_path(state: dict, cell: tuple[int, int], surface_y: int) -> bool:
    """Require grounded feet at a dirt path's real 15/16 collision height (incident R1), within the 1.5-block arrival
    radius of `move_to` around the cell's centre, not at integer-height air."""
    position = state.get("pos", [])
    return (len(position) == 3 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in position)
            and state.get("onGround") is True
            and math.hypot(position[0] - cell[0] - 0.5, position[2] - cell[1] - 0.5) <= 1.5
            and abs(position[1] - (surface_y + 15 / 16)) < 0.0001)


def feedback_checks(first: Decision, following: Decision, resolved: str, observation: dict) -> dict[str, bool]:
    """Check the immediate next decision's actual user prompt, including final skill results and failure hints."""
    call = first.tool_call
    args = {key: value for key, value in call.args.items() if key != "note"}
    action = f"{call.name} {json.dumps(args, sort_keys=True)} -> {resolved}"
    next_turn = following.agent == first.agent and following.game_time > first.game_time
    checks = {"outcome in next prompt": next_turn and action in following.user_prompt}
    if first.outcome.startswith(("not possible", "failed", "invalid")):
        hint = failure_hint(call, first.outcome, observation)
        checks["failure_hint in next prompt"] = (next_turn and bool(hint) and first.outcome.endswith(hint)
                                                  and first.outcome in following.user_prompt)
        delay = following.game_time - first.game_time
        checks["failure thinking time"] = THINKING_TICKS <= delay < THINKING_TICKS + POLL
    return checks


def inventory_delta(prompt: str, item: str) -> int:
    """Sum actual inventory-change messages across polls; building need not spend its whole bill in one tick."""
    pattern = rf"([+-]\d+) {re.escape(item.removeprefix('minecraft:'))}(?=,|$)"
    return sum(int(delta) for line in prompt.splitlines() if "your inventory:" in line
               for delta in re.findall(pattern, line))


def rest_checks(decisions: list[Decision], night_start: int, wake_tick: int) -> dict[str, bool]:
    """Reject missing coverage or any night decision besides the single measured speech-delivery poll."""
    night_end = night_start + NIGHT_REST[1] - NIGHT_REST[0]
    night = [decision for decision in decisions if night_start <= decision.game_time < night_end]
    return {
        "idle before rest": any(decision.game_time == night_start - POLL and decision.tool is None
                                for decision in decisions),
        "no night decisions unless spoken to": [decision.game_time for decision in night] == [wake_tick],
        "wake speech in next prompt": len(night) == 1 and any(
            "] mayor (" in line and line.endswith(f': "{WAKE_TEXT}"') for line in night[0].user_prompt.splitlines()),
        "decisions resume at dawn": any(decision.game_time == night_end and decision.tool is None
                                        for decision in decisions),
    }


class IT15SkillsExperiment:
    """Own one isolated village session and report each grounded action's outcome and next-prompt feedback."""

    def __init__(self, args):
        """Validate ports before cloning and retain only this gate's paths, manifest and check receipts."""
        self.args = args
        self.options = port_options(args)
        self.directory = args.runs_dir / "it15_skills"
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.checks: list[dict] = []
        self.last_status: dict = {}
        self.mining_probe: dict = {}
        self.client = None

    def _check(self, label: str, passed: bool, detail: str = "") -> None:
        """Print and retain a check, including enough actual evidence to diagnose a failure."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if detail else ""), flush=True)

    def _command(self, command: str) -> None:
        """Run one checked administrative setup command in this gate's disposable clone only."""
        self.client.request("command", {"command": command})

    def _state(self, name: str = "builder") -> dict:
        """Read a bot's actual position and inventory without consuming its pending observations."""
        return self.client.request("bot_state", {"name": name})

    def _counts(self, name: str = "builder") -> dict[str, int]:
        """Sum stacks by item so transfer and construction checks do not depend on inventory slots."""
        counts: dict[str, int] = {}
        for entry in self._state(name)["inventory"]:
            counts[entry["item"]] = counts.get(entry["item"], 0) + entry["count"]
        return counts

    def _blocks(self, positions: list[list[int]]) -> list[str]:
        """Read existing block identifiers for precise terrain and physical-effect checks."""
        return self.client.request("blocks", {"positions": positions})["blocks"]

    def _teleport(self, name: str, position) -> None:
        """Cancel leftover waits, place a body, then let real gravity settle it on a poll-aligned tick."""
        self.client.request("skill_cancel", {"name": name})
        self._command(f"tp {name} {' '.join(map(str, position))} 0 0")
        self.client.request("step", {"n": POLL})
        now = harness.game_time(self.client.request("ping"))
        if now % POLL:
            self.client.request("step", {"n": POLL - now % POLL})

    def _runtime(self, label: str, decide, *, epoch_turns: int = 1) -> SimulationRuntime:
        """Use real village briefs/timing; full turns retain failures, incremental turns exercise pickup updates."""
        agent = next(agent for agent in self.manifest.agents(epoch_turns=epoch_turns) if agent.name == "builder")
        return SimulationRuntime(self.client, [agent], ScriptedClient(decide), self.directory / "orchestrator" / label,
                                 command_error=harness.CommandError, model_concurrency=1,
                                 min_gap=THINKING_TICKS, rest=NIGHT_REST)

    def _exercise(self, label: str, call: ToolCall, expected: str = "succeeded") -> str:
        """Execute exactly one planned action, then obtain and check its very next real agent prompt."""
        calls = 0

        def decide(system: str, user: str) -> ToolCall | None:
            """Try the planned action once; the idle follow-up supplies its next prompt without retrying it."""
            nonlocal calls
            calls += 1
            if calls == 2:
                self.last_status = self.client.request("skill_status", {"name": "builder"})
            return call if calls == 1 else None

        observation = self.client.request("observe", {"name": "builder", "include_terrain": False})
        runtime = self._runtime(label, decide, epoch_turns=2 if expected == "succeeded" else 1)
        deadline = runtime.game_time() + ACTION_TIMEOUT
        while len(runtime.log.decisions()) < 2 and runtime.game_time() < deadline:
            runtime.run(runtime.game_time() + POLL)
        decisions = runtime.log.decisions()
        self._check(f"{label}: next decision reached", len(decisions) == 2)
        if len(decisions) != 2:
            raise harness.HarnessError(f"{label}: no next decision within {ACTION_TIMEOUT} ticks")
        first, following = decisions
        resolved = runtime.agents[0].memory.to_json()["actions"][0]["outcome"]
        if expected == "succeeded":
            passed = first.outcome == "started" and resolved.split(" (", 1)[0] == expected
        else:
            hint = failure_hint(call, first.outcome, observation)
            passed = first.outcome == expected + hint and resolved == first.outcome
        self._check(f"{label}: expected outcome", passed, f"{first.outcome}; final={resolved}")
        for check, passed in feedback_checks(first, following, resolved, observation).items():
            self._check(f"{label}: {check}", passed)
        return following.user_prompt

    def _roads(self) -> None:
        """Walk consecutive legs starting and finishing on real, settled dirt paths, never paving a substitute."""
        surface_y = self.manifest.data["layout"]["surface_y"]
        road = next(road for road in self.manifest.data["roads"] if road["name"] == "west_spine")
        left, north, right, south = road["bounds"]
        positions = [[ROAD_CELLS[0][0], surface_y, depth] for depth in range(ROAD_CELLS[0][1], ROAD_CELLS[-1][1] + 1)]
        self._check("road corridor is inside the village west_spine",
                    all(left <= horizontal <= right and north <= depth <= south for horizontal, _, depth in positions))
        self._check("road corridor is actual dirt_path, not the gravel collar",
                    self._blocks(positions) == ["minecraft:dirt_path"] * len(positions))
        start_x, start_z = ROAD_CELLS[0]
        self._teleport("builder", (start_x + 0.5, surface_y + 1, start_z + 0.5))
        for index, cell in enumerate(ROAD_CELLS[1:], 1):
            before = self._state()
            self._check(f"road_{index}: settled 15/16-high starting feet",
                        on_dirt_path(before, ROAD_CELLS[index - 1], surface_y), str(before["pos"]))
            self._exercise(f"road_{index}", ToolCall("move_to", {"x": cell[0], "y": surface_y + 1, "z": cell[1]}))
            after = self._state()
            self._check(f"road_{index}: arrived on dirt path at actual collision height",
                        on_dirt_path(after, cell, surface_y), str(after["pos"]))
            self._check(f"road_{index}: body actually moved", math.dist(before["pos"], after["pos"]) >= 7)

    def _chest(self) -> None:
        """Target the village's occupied fair-delivery chest, not a preselected adjacent air block."""
        chest = self.manifest.data["containers"]["fair_delivery"]["pos"]
        horizontal, height, depth = chest
        self._check("chest target is an actual chest", self._blocks([chest]) == ["minecraft:chest"])
        self._teleport("builder", (horizontal + 0.5, height, depth + 5.5))
        self._exercise("chest_approach", ToolCall("move_to", dict(zip(("x", "y", "z"), chest))))
        state = self._state()
        position = state["pos"]
        distance = math.hypot(position[0] - horizontal - 0.5, position[2] - depth - 0.5)
        self._check("chest approach ends beside, not inside, the chest",
                    state["onGround"] and 0.5 < distance <= 2.5 and abs(position[1] - height) < 0.1,
                    str(position))

    def _drop_paths(self, position: list[float]) -> list[dict]:
        """Probe legal standing cells beside a drop under a low ceiling, without walking or forcing a pickup."""
        paths = []
        horizontal, height, depth = map(math.floor, position)
        for delta_x in (-1, 0, 1):
            for delta_z in (-1, 0, 1):
                for elevation in (0, 1, -1, -2, -3):
                    target = [horizontal + delta_x + 0.5, height + elevation, depth + delta_z + 0.5]
                    if abs(target[0] - position[0]) > 1.3 or abs(target[2] - position[2]) > 1.3:
                        continue
                    receipt = self.client.request("skill_start", {"name": "builder", "skill": "move_to",
                                                  "args": dict(zip(("x", "y", "z"), target)), "dry_run": True})
                    paths.append({"pos": target, **receipt})
        return paths

    def _mining(self) -> None:
        """Probe a bottom-log drop under an intact log, then collect it using only the miner's recent knowledge."""
        tree = [-109, -60, 8]
        upper = [tree[0], tree[1] + 1, tree[2]]
        self._check("mine targets are actual village oak logs",
                    self._blocks([tree, upper]) == ["minecraft:oak_log", "minecraft:oak_log"])
        self._command("fill -111 -60 7 -111 -58 9 minecraft:stone")
        self._teleport("mayor", (-112.5, -60, 8.5))
        self._teleport("builder", (-106.5, -60, 8.5))
        region = {"x0": -115, "x1": -103, "z0": 3, "z1": 14, "y0": -62, "y1": -55}
        old_items = {entity["uuid"] for entity in self.client.request("dump_entities", {"region": region})["entities"]}
        before = self._counts()
        prompt = self._exercise("mine_bottom", ToolCall("mine", dict(zip(("x", "y", "z"), tree))))
        drops = [entity for entity in self.client.request("dump_entities", {"region": region})["entities"]
                 if entity["type"] == "minecraft:item" and entity["uuid"] not in old_items]
        seen = self.client.request("observe", {"name": "builder", "include_terrain": False})["entities"]
        collect = self.client.request("skill_start", {"name": "builder", "skill": "collect", "args": {},
                                                     "dry_run": True})
        paths = []
        for drop in drops:
            position = list(map(float, drop["pos"].split(",")))
            routes = self._drop_paths(position)
            paths.append({"uuid": drop["uuid"], "pos": position,
                          "visible": any(entity["uuid"] == drop["uuid"] for entity in seen),
                          "distance": math.dist(self._state()["pos"], position),
                          "path": {"feasible": any(route["feasible"] for route in routes), "candidates": routes}})
        stranger = self.client.request("skill_start", {"name": "mayor", "skill": "collect", "args": {},
                                                       "dry_run": True})
        self.mining_probe = {"picked_up": self._counts().get("minecraft:oak_log", 0)
                            > before.get("minecraft:oak_log", 0), "drops": paths, "collect": collect,
                            "mine_progress": self.last_status.get("progress", {}), "other_bot_collect": stranger}
        print("  bottom-log probe: " + json.dumps(self.mining_probe, sort_keys=True), flush=True)
        self._check("bottom-log mine leaves the upper log intact",
                    self._blocks([tree, upper]) == ["minecraft:air", "minecraft:oak_log"])
        self._check("probe reproduced an unpicked, reachable drop",
                    not self.mining_probe["picked_up"] and bool(paths)
                    and all(drop["distance"] <= 6 and drop["path"]["feasible"] for drop in paths))
        if any(drop["visible"] for drop in paths):
            self._command("setblock -108 -59 8 minecraft:stone")
        hidden_view = self.client.request("observe", {"name": "builder", "include_terrain": False})["entities"]
        hidden_collect = self.client.request("skill_start", {"name": "builder", "skill": "collect", "args": {},
                                                            "dry_run": True})
        self.mining_probe["occluded_collect"] = hidden_collect
        self._check("the same mined UUID is now out of sight",
                    bool(paths)
                    and all(drop["uuid"] not in {entity["uuid"] for entity in hidden_view} for drop in paths))
        reported = {drop["uuid"] for drop in self.mining_probe["mine_progress"].get("drops", [])}
        self._check("mine progress identifies the actual unpicked drop", bool(reported)
                    and reported == {drop["uuid"] for drop in paths})
        self._check("miner can collect its hidden drop", hidden_collect.get("feasible") is True)
        self._check("mine's next prompt locates the unpicked drop", "unpicked drop:" in prompt)
        self._check("another bot cannot collect the unseen drop",
                    stranger.get("feasible") is False and stranger.get("reason") == "target_not_found")
        self.client.request("skill_cancel", {"name": "builder"})
        prompt = self._exercise("collect_bottom", ToolCall("collect", {}))
        after = self._counts()
        self._check("bottom log collected without mining the upper log",
                    after.get("minecraft:oak_log", 0) - before.get("minecraft:oak_log", 0) == 1
                    and self._blocks([upper]) == ["minecraft:oak_log"])
        self._check("pickup reported in the next prompt", inventory_delta(prompt, "minecraft:oak_log") == 1)
        self._exercise("collect_after_mine", ToolCall("collect", {}), "not possible: target_not_found")
        self._check("empty collect leaves inventory unchanged", self._counts() == after)
        self._raised_mining()

    def _raised_mining(self) -> None:
        """Collect a plinth log's drop from a lower adjacent pickup cell, keeping the upper log (the hidden-drop case is
        the bottom-log probe's)."""
        tree, upper = [100, -59, 100], [100, -58, 100]
        self._command("setblock 100 -60 100 minecraft:stone")
        self._command("setblock 102 -60 100 minecraft:stone")
        self._command("setblock 100 -59 100 minecraft:oak_log")
        self._command("setblock 100 -58 100 minecraft:oak_log")
        self._teleport("builder", (102.5, -59, 100.5))
        before = self._counts()
        self._exercise("mine_plinth", ToolCall("mine", dict(zip(("x", "y", "z"), tree))))
        drops = self.last_status.get("progress", {}).get("drops", [])
        self.client.request("skill_cancel", {"name": "builder"})
        collect = self.client.request("skill_start", {"name": "builder", "skill": "collect", "args": {},
                                                     "dry_run": True})
        self.mining_probe["plinth"] = {"drops": drops, "collect": collect, "start": self._state()["pos"]}
        self._check("plinth mine records its drop", bool(drops))
        self._check("plinth drop is not picked up by the mine", self._counts() == before)
        self._check("plinth drop has a lower adjacent pickup route", collect.get("feasible") is True)
        self._exercise("collect_plinth", ToolCall("collect", {}))
        self._check("descent collected the plinth log without breaking the upper log",
                    self._counts().get("minecraft:oak_log", 0) - before.get("minecraft:oak_log", 0) == 1
                    and self._blocks([upper]) == ["minecraft:oak_log"]
                    and self._state()["onGround"] and self._state()["pos"][1] < -59.5)

    def _social(self) -> None:
        """Check both absent-villager hints, overlong speech, adjacent transfer and speech delivery feedback."""
        self._teleport("builder", QUIET)
        self._teleport("mayor", (QUIET[0] + 2, QUIET[1], QUIET[2]))
        self._exercise("ask_missing", ToolCall("ask", {"npc": "it15_absent"}), "not possible: target_not_found")
        self._exercise("ask_out_of_sight", ToolCall("ask", {"npc": "adult_01"}), "not possible: out_of_reach")
        self._check("overlong speech is exactly 650 characters", len(OVERLONG_SPEECH) == 650)
        self._exercise("overlong_speech", ToolCall("speak", {"text": OVERLONG_SPEECH}),
                       "failed: Speech needs nonempty text of at most 600 characters and whisper/talk/shout volume")
        self.client.request("give", {"name": "builder", "item": "minecraft:emerald", "count": 3})
        before, recipient = self._counts(), self._counts("mayor")
        prompt = self._exercise("give_adjacent", ToolCall("give", {"target": "mayor", "item": "minecraft:emerald",
                                                                  "count": 3}))
        after = self._counts()
        self._check("adjacent give conserves exactly three emeralds",
                    before.get("minecraft:emerald", 0) - after.get("minecraft:emerald", 0) == 3
                    and self._counts("mayor").get("minecraft:emerald", 0) - recipient.get("minecraft:emerald", 0) == 3)
        self._check("give inventory feedback in next prompt", inventory_delta(prompt, "minecraft:emerald") == -3)
        prompt = self._exercise("speech_heard", ToolCall("speak", {"text": "IT15 hello", "to": "mayor"}))
        heard = self.client.request("observe", {"name": "mayor", "include_terrain": False})["heard"]
        self._check("adjacent agent actually heard speech",
                    any(item["speaker"] == "builder" and item["text"] == "IT15 hello" for item in heard))
        self._check("speaker's next prompt names its listener", "heard by: mayor" in prompt)
        self._teleport("mayor", (QUIET[0] - 16, QUIET[1], QUIET[2]))
        prompt = self._exercise("speech_unheard", ToolCall("speak", {"text": "IT15 alone"}))
        self._check("speaker's next prompt reports nobody heard it", "nobody was close enough to hear" in prompt)

    def _building(self) -> None:
        """Reject a missing real bridge bill without effects, then consume that exact bill to build the section."""
        section = self.manifest.data["schematics"]["bridge_span_1"]
        origin = section["origin"]
        positions = [[origin[0] + block["dx"], origin[1] + block["dy"], origin[2] + block["dz"]]
                     for block in section["blocks"]]
        wanted = [block["block"] for block in section["blocks"]]
        self._teleport("builder", section["stand"])
        self._command("clear builder")
        before = self._blocks(positions)
        self._check("bridge section needs its full bill",
                    all(actual != target for actual, target in zip(before, wanted)))
        call = ToolCall("build_section", {"schematic": "bridge_span_1", "origin": origin})
        self._exercise("build_without_bill", call, "not possible: missing_items")
        self._check("rejected construction changed neither blocks nor inventory",
                    self._blocks(positions) == before and self._counts() == {})
        for item, count in section["bill"].items():
            self.client.request("give", {"name": "builder", "item": item, "count": count})
        prompt = self._exercise("build_with_bill", call)
        self._check("the actual bridge section is complete", self._blocks(positions) == wanted)
        self._check("construction consumed the exact bill", self._counts() == {})
        self._check("construction inventory feedback in next prompt",
                    all(inventory_delta(prompt, item) == -count for item, count in section["bill"].items()))

    def _night(self) -> None:
        """Observe a full rest window with no waiting skill, one real utterance, and prompt-driven dawn decisions."""
        self._teleport("builder", QUIET)
        self._teleport("mayor", (QUIET[0] + 2, QUIET[1], QUIET[2]))

        def idle(system: str, user: str) -> None:
            """Choose no action, leaving the body idle so a long wait cannot masquerade as night rest."""
            return None

        now = harness.game_time(self.client.request("ping"))
        night_start = now // 24_000 * 24_000 + NIGHT_REST[0]
        if now >= night_start - POLL:
            night_start += 24_000
        self.client.request("step", {"n": night_start - POLL - now})
        runtime = self._runtime("night", idle)
        speech_start = night_start + 2_400
        runtime.run(speech_start)
        self._check("night body is idle, not running a wait skill",
                    self.client.request("skill_status", {"name": "builder"})["state"]
                    in {"idle", "succeeded", "failed", "cancelled"})
        self._check("quiet night has no decisions before speech", len(runtime.log.decisions()) == 1)
        speech = self.client.request("speak", {"name": "mayor", "text": WAKE_TEXT, "to": "builder"})
        self._check("night wake speech accepted", speech.get("accepted") is True)
        runtime.run(speech_start + 4 * POLL)
        status = self.client.request("skill_status", {"name": "mayor"})
        self._check("night speech actually reached the resting agent", status["state"] == "succeeded"
                    and "builder" in status.get("progress", {}).get("heardBy", []))
        wake_tick = math.ceil(status["endedAt"] / POLL) * POLL
        night_end = night_start + NIGHT_REST[1] - NIGHT_REST[0]
        runtime.run(night_end + POLL)
        for label, passed in rest_checks(runtime.log.decisions(), night_start, wake_tick).items():
            self._check(label, passed)

    def run(self) -> bool:
        """Run only this gate, close its server before reporting, and preserve artifacts on success or failure."""
        session = VillageSession(self.directory, self.args.runs_dir / "village_sabotage", self.manifest, self.options)
        session.clone()
        try:
            with session:
                self.client = session.client
                self._roads()
                self._chest()
                self._mining()
                self._social()
                self._building()
                self._night()
        except (harness.HarnessError, OSError, ValueError, KeyError) as error:
            self._check("gate completed", False, str(error))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        report = {"passed": passed, "checks": self.checks, "mining_probe": self.mining_probe,
                  "negative_control": "Owner's future feet-position revert check; not run by this gate.",
                  "estimated_seconds": 120}
        (self.directory / "it15_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-15: village skills, next-prompt feedback and night rest", flush=True)
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse the shared gate options, and return zero only for a passing IT-15."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if IT15SkillsExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-15: skills experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
