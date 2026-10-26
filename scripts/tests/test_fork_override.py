"""Unit tests for decision forks (DELEGATE_SPEC §8): the runtime's override (swap, placebo, swap-hold, forced replay)
and the parts of `scripts/fork_decision.py` that need no server."""

from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import Decision, DecisionLog
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.perception import EvidenceExposure
from orchestrator.runtime import DecisionOverride, SimulationRuntime
from orchestrator.scenario import OutcomeTracker, ScenarioManifest
from scripts.fork_decision import ForkPoint, OutcomeInputs, held_rejections, parse_call


class CommandRejected(Exception):
    """The stand-in world's rejection of a command (like the harness's CommandError)."""


class StandInWorld:
    """The few harness requests the runtime sends, with a world that never changes: every agent idles at a fixed
    position and every skill finishes at once, so prompts depend only on the decisions taken."""

    def __init__(self):
        """Start at game time 0 with no commands seen."""
        self.time = 0
        self.actions: list[tuple[int, str, dict]] = []

    def request(self, command: str, args: dict | None = None) -> dict:
        """Answer one request: time, skill status, observation, stepping, or an action (recorded and accepted)."""
        args = args or {}
        if command == "ping":
            return {"gameTime": self.time}
        if command == "skill_status":
            return {"skillId": None, "skill": None, "state": "idle"}
        if command == "observe":
            return {"gameTime": self.time, "self": {"name": args["name"], "pos": [0.5, -60.0, 0.5], "health": 20.0,
                                                    "food": 20, "inventory": [], "skill": {"state": "idle"}},
                    "entities": [], "blocks": [], "documents": [], "heard": [], "changes": []}
        if command == "step":
            self.time += args["n"]
            return {"gameTime": self.time}
        self.actions.append((self.time, args["name"], dict(args.get("args", args))))
        return {"accepted": True, "skillId": f"{args['name']}:{len(self.actions)}"}


def agents() -> list[Agent]:
    """Two agents with fixed briefs."""
    return [Agent(RoleBrief(name, "a villager", f"{name} lives here.", "Nothing secret.")) for name in ("ann", "bob")]


class Policy:
    """A scripted model: ann waits a number of ticks that depends on how many decisions it has seen; bob always
    waits 40. Every call is counted per agent."""

    def __init__(self, ann_ticks: int = 40):
        """`ann_ticks` is what ann waits, so two policies can disagree."""
        self.ann_ticks = ann_ticks
        self.calls: list[str] = []

    def decide(self, system: str, user: str) -> ToolCall:
        """The policy's call for the agent named in the prompt."""
        name = "ann" if "You are ann" in user else "bob"
        self.calls.append(name)
        return ToolCall("wait", {"ticks": self.ann_ticks if name == "ann" else 40})


class OverrideTest(unittest.TestCase):
    """A fork arm changes exactly the decisions its override names and reuses the factual run everywhere else."""

    def setUp(self):
        """A factual run of 160 ticks (polls at 0, 40, 80, 120) whose log the branches use as their source."""
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        factual = SimulationRuntime(StandInWorld(), agents(), ScriptedClient(Policy().decide, model="sol"),
                                    self.root / "factual", command_error=CommandRejected)
        factual.run(160)
        self.source = DecisionLog(self.root / "factual" / "decisions.jsonl")

    def tearDown(self):
        """Remove the run directories."""
        self.folder.cleanup()

    def branch(self, name: str, override: DecisionOverride | None, model=None, policy: Policy | None = None,
               world: StandInWorld | None = None) -> SimulationRuntime:
        """Run a branch from the start with the factual log as its source."""
        policy = policy or Policy()
        client = model or ScriptedClient(policy.decide, model="sol")
        runtime = SimulationRuntime(world or StandInWorld(), agents(), client, self.root / name,
                                    command_error=CommandRejected, source=self.source, override=override)
        runtime.run(160)
        return runtime

    def test_swap_forces_the_call_at_its_point_only(self):
        policy, world = Policy(), StandInWorld()
        runtime = self.branch("swap", DecisionOverride("ann", 40, call=ToolCall("move_to", {"x": 3, "y": -60, "z": 1})),
                              policy=policy, world=world)
        swapped = runtime.log.get("ann", 40)
        self.assertEqual((swapped.source, swapped.tool), ("forced", "move_to"))
        self.assertTrue(runtime.override_applied)
        self.assertIn((40, "ann", {"x": 3.5, "y": -60, "z": 1.5}), world.actions)
        self.assertEqual(runtime.log.get("bob", 40).source, "replay")  # bob's prompt is unchanged
        self.assertEqual(runtime.log.get("ann", 0).source, "replay")
        self.assertEqual(policy.calls, ["ann", "ann"])  # ann's later prompts changed (her own last turn differs)

    def test_placebo_resamples_its_point_even_when_the_log_matches(self):
        policy = Policy()
        runtime = self.branch("placebo", DecisionOverride("ann", 40, resample=True), policy=policy)
        self.assertEqual(runtime.log.get("ann", 40).source, "model")
        self.assertTrue(runtime.override_applied)
        self.assertEqual(policy.calls, ["ann"])  # same policy, so everything after it matches the log again
        self.assertEqual(runtime.log.get("ann", 80).source, "replay")

    def test_unreached_point_is_reported(self):
        runtime = self.branch("missed", DecisionOverride("ann", 60, resample=True))  # 60 is not a poll
        self.assertFalse(runtime.override_applied)

    def test_held_agents_take_logged_actions_without_the_model(self):
        policy = Policy(ann_ticks=80)  # a different model: ann's own decisions no longer match the log
        runtime = self.branch("hold", DecisionOverride("ann", 40, call=ToolCall("wait", {"ticks": 60}),
                                                       hold=frozenset({"bob"}), hold_from=40, hold_until=120),
                              policy=policy)
        self.assertEqual([runtime.log.get("bob", t).source for t in (40, 80, 120)], ["held"] * 3)
        self.assertEqual(runtime.log.get("bob", 0).source, "replay")  # before the window: the normal path
        self.assertNotIn("bob", policy.calls)

    def test_held_agent_decides_only_at_logged_polls(self):
        sparse = DecisionLog(self.root / "sparse.jsonl")
        for decision in self.source.decisions():
            if decision.agent == "ann" or decision.game_time in (0, 120):
                sparse.append(decision)
        runtime = SimulationRuntime(StandInWorld(), agents(), None, self.root / "forced", command_error=CommandRejected,
                                    source=sparse, override=DecisionOverride(hold=frozenset({"ann", "bob"})))
        runtime.run(160)
        self.assertEqual([(d.game_time, d.agent) for d in runtime.log.decisions() if d.agent == "bob"],
                         [(0, "bob"), (120, "bob")])
        self.assertEqual({d.source for d in runtime.log.decisions()}, {"held"})
        self.assertEqual(runtime.stats.model_calls, 0)

    def test_override_reuses_decisions_logged_by_another_model(self):
        policy = Policy()
        runtime = self.branch("local", DecisionOverride("ann", 40, resample=True), policy=policy,
                              model=ScriptedClient(policy.decide, model="qwen"))
        self.assertEqual(runtime.log.get("bob", 80).source, "replay")
        self.assertEqual(policy.calls, ["ann"])

    def test_without_override_another_model_reuses_nothing(self):
        policy = Policy()
        runtime = self.branch("plain", None, model=ScriptedClient(policy.decide, model="qwen"))
        self.assertEqual(runtime.stats.replayed, 0)
        self.assertEqual(len(policy.calls), 8)

    def test_override_needs_a_source_log(self):
        with self.assertRaises(ValueError):
            SimulationRuntime(StandInWorld(), agents(), None, self.root / "none", command_error=CommandRejected,
                              override=DecisionOverride(hold=frozenset({"ann"})))


class ForkPointTest(unittest.TestCase):
    """Each arm of a decision point gets the override §8 defines for it."""

    def setUp(self):
        """A point: the builder at tick 2400, with a walk to the market swapped in."""
        self.point = ForkPoint("sol", "builder", 2400, parse_call('{"name": "move_to", "args": {"x": -46, "y": -60, '
                                                                   '"z": -13}}'))
        self.agents = ["builder", "envoy", "mayor"]

    def test_swap_argument_parses_into_a_tool_call(self):
        self.assertEqual(self.point.swap, ToolCall("move_to", {"x": -46, "y": -60, "z": -13}))

    def test_swap_and_placebo_change_only_the_point(self):
        self.assertEqual(self.point.override("swap", self.agents, 2400),
                         DecisionOverride("builder", 2400, call=self.point.swap))
        self.assertEqual(self.point.override("placebo", self.agents, 2400),
                         DecisionOverride("builder", 2400, resample=True))
        self.assertIsNone(self.point.override("control", self.agents, 2400))

    def test_swap_hold_holds_every_other_agent_over_the_horizon(self):
        override = self.point.override("swap_hold", self.agents, 2400)
        self.assertEqual((override.call, override.hold, override.hold_from, override.hold_until),
                         (self.point.swap, frozenset({"envoy", "mayor"}), 2400, 4800))
        with self.assertRaises(ValueError):
            self.point.override("twin", self.agents, 2400)


class HeldRejectionTest(unittest.TestCase):
    """A held action the engine refuses, where the factual run's same action went through, is reported."""

    def test_only_newly_refused_held_actions_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            factual, arm = DecisionLog(Path(folder) / "factual.jsonl"), DecisionLog(Path(folder) / "arm.jsonl")
            for log, source, outcomes in ((factual, "model", ("started", "not possible: busy", "started")),
                                          (arm, "held", ("not possible: no_path", "not possible: busy", "started"))):
                for tick, outcome in zip((40, 80, 120), outcomes):
                    log.append(Decision(tick, "mayor", "d", False, source, "wait", {"ticks": 40}, "sol", 0, 0, "",
                                        outcome=outcome))
            self.assertEqual(held_rejections(arm, factual), [{"agent": "mayor", "game_time": 40,
                                                              "outcome": "not possible: no_path",
                                                              "logged_outcome": "started"}])


class OutcomeWorld:
    """The read-only queries an arm's outcome needs: blocks, inventories, the records chest, and the time."""

    def __init__(self, blocks: dict, inventories: dict):
        """`blocks` maps positions to block ids (water elsewhere); `inventories` maps agents to inventory lists."""
        self.blocks, self.inventories = blocks, inventories

    def request(self, command: str, args: dict | None = None) -> dict:
        """Answer one read-only query."""
        if command == "blocks":
            return {"blocks": [self.blocks.get(tuple(pos), "minecraft:water") for pos in args["positions"]]}
        if command == "bot_state":
            return {"inventory": self.inventories.get(args["name"], [])}
        if command == "container_items":
            return {"items": []}
        if command == "ping":
            return {"gameTime": 4800}
        if command == "save_snapshot":
            return {}
        raise AssertionError(f"unexpected request {command}")


class OutcomeInputsTest(unittest.TestCase):
    """An arm's outcome inputs read sections, emeralds and evidence from the world and the exposure tracker."""

    def test_sections_emeralds_and_information(self):
        manifest = ScenarioManifest.load(Path(__file__).resolve().parents[2] / "scenario/village_sabotage.json")
        span = next(structure for structure in manifest.structures() if structure["name"] == "bridge_span_1")
        world = OutcomeWorld({tuple(block["pos"]): block["block"] for block in span["blocks"]},
                             {"mayor": [{"item": "minecraft:emerald", "count": 64},
                                        {"item": "minecraft:emerald", "count": 6},
                                        {"item": "minecraft:lantern", "count": 1}]})
        exposure = EvidenceExposure(manifest)
        support = manifest.data["evidence"]["supports"][0]["pos"]
        exposure.observe("security", {"gameTime": 3000, "blocks": [{"pos": support, "block": "minecraft:water",
                                                                     "distance": 9.0}], "heard": []})
        tracker = OutcomeTracker({})
        with tempfile.TemporaryDirectory() as folder:
            outcome = OutcomeInputs(manifest).collect(world, Path(folder) / "events.jsonl", tracker, exposure, 4800)
        self.assertTrue(outcome["sections"]["bridge_span_1"])
        self.assertFalse(outcome["sections"]["bridge_span_2"])
        self.assertFalse(outcome["fair_open"])
        self.assertEqual((outcome["emeralds"]["mayor"], outcome["emeralds"]["builder"]), (70, 0))
        self.assertEqual(outcome["information"]["security"], ["west_supports"])
        self.assertEqual(outcome["information"]["mayor"], [])
        self.assertFalse(outcome["roles"]["builder"])  # not all spans built, and no payment logged


if __name__ == "__main__":
    unittest.main()
