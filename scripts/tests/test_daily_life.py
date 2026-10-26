"""Daily-life tools, own hunger narration, and byte-stable default-off orchestration."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.candidates import CandidateGenerator, DryRun, EngineDryRun, representable, same_choice
from orchestrator.decisions import prompt_digest
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import GameMaster, NarratedRuntime, ShadowEngine, gm_system_prompt
from orchestrator.render import ObservationRenderer
from orchestrator.runtime import DecisionOverride, ReplayDiverged, SimulationRuntime, failure_hint
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, TOOL_NAMES, ToolTranslator, tool_definitions
from scripts.harness import CommandError
from scripts.tests import test_golden_prompts
from scripts.tests.fake_world import FakeWorld

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("golden")


def actor(*, daily_hunger=False):
    """Build a role with a multi-turn epoch, optionally opting into daily-life tools."""
    return Agent(RoleBrief("alice", "worker", "Works.", "Help."), epoch_turns=8, daily_hunger=daily_hunger)


class RecordingClient(ScriptedClient):
    """Record the exact model inputs while answering with an inexpensive deterministic policy."""

    def __init__(self, policy, model="scripted"):
        """Start with a scripted policy and no requests."""
        super().__init__(policy, model=model)
        self.requests = []

    def complete(self, system, user, tools, seed=None, history=()):
        """Save prompt, schema, and conversation before returning the scripted response."""
        self.requests.append({"system": system, "user": user, "tools": deepcopy(tools), "history": history})
        return super().complete(system, user, tools, seed=seed, history=history)


class EatTranslationTest(unittest.TestCase):
    """Eat uses the same normalized, tick-based command and dry-run paths as drop."""

    def test_eat_translation_normalizes_item_and_keeps_only_the_item(self):
        """Notes and engine-only extras cannot change the single-item eating skill."""
        for item in ("Bread", " bread ", "minecraft:bread"):
            with self.subTest(item=item):
                action = ToolTranslator().translate("alice", ToolCall("eat", {
                    "item": item, "note": "Have lunch.", "count": 64, "eat_ticks": 1}))
                self.assertEqual(action.command, "skill_start")
                self.assertEqual(action.args, {"name": "alice", "skill": "eat", "args": {"item": "minecraft:bread"}})
                self.assertTrue(action.is_skill)
        self.assertIn("eat", TOOL_NAMES)

    def test_eat_requires_a_string_item(self):
        """Missing and malformed items are rejected locally, like other invalid tool calls."""
        with self.assertRaises(KeyError):
            ToolTranslator().translate("alice", ToolCall("eat", {}))
        for item in (None, 1, [], {}):
            with self.subTest(item=item), self.assertRaises(TypeError):
                ToolTranslator().translate("alice", ToolCall("eat", {"item": item}))

    def test_engine_and_shadow_share_normalized_dry_run_and_execution(self):
        """Both feasibility adapters estimate eating, and the shadow polls it until it completes."""
        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"eat": 32})
        call = ToolCall("eat", {"item": "Bread"})
        self.assertEqual(EngineDryRun(world, CommandError)("alice", call), DryRun(True, 32))
        shadow = ShadowEngine(world, CommandError)
        self.assertEqual(shadow.check("alice", call), {"feasible": True, "reason": None, "ticks": 32})
        expected = {"name": "alice", "skill": "eat", "args": {"item": "minecraft:bread"}, "dry_run": True}
        self.assertEqual(world.requests, [("skill_start", expected)] * 2)
        self.assertEqual(shadow.execute("alice", call), "started")
        self.assertEqual(shadow.skill_state("alice"), "running")
        shadow.advance(32)
        self.assertEqual(shadow.skill_state("alice"), "succeeded")

    def test_eat_failure_hints_explain_missing_food_and_nonfood(self):
        """Engine reason codes remain visible with grounded inventory guidance."""
        call = ToolCall("eat", {"item": "bread"})
        self.assertIn("food you carry", failure_hint(call, "not possible: missing_items", {}))
        self.assertIn("not food", failure_hint(call, "not possible: not_food", {}))


class DailyLifeSchemaTest(unittest.TestCase):
    """Scenario gating adds one schema while retaining every byte of the original tools and prompts."""

    def test_default_off_tools_and_all_role_prompts_match_unmodified_goldens(self):
        """Absent and false flags preserve the original schema and the first/fifth-turn role snapshots."""
        self.assertIs(tool_definitions(), TOOL_DEFINITIONS)
        self.assertIs(tool_definitions(daily_hunger=False), TOOL_DEFINITIONS)
        self.assertIs(actor().tools, TOOL_DEFINITIONS)
        self.assertEqual((json.dumps(actor().tools, indent=1, ensure_ascii=False) + "\n").encode(),
                         (GOLDEN / "tools.json").read_bytes())
        for relative, actual in test_golden_prompts.GoldenPromptsTest().snapshots().items():
            with self.subTest(file=relative):
                self.assertEqual(actual, (GOLDEN / relative).read_bytes())

    def test_enabled_schema_appends_only_eat_and_does_not_mutate_default_tools(self):
        """Daily-life requests retain the original order and give eat its exact description and item schema."""
        enabled = actor(daily_hunger=True).tools
        self.assertEqual(enabled[:-1], TOOL_DEFINITIONS)
        self.assertEqual(enabled[-1], {
            "name": "eat", "description": "Eat food from your inventory (it takes a few seconds).",
            "parameters": {"type": "object", "properties": {
                "item": {"type": "string"}, "note": {"type": "string", "maxLength": 100}},
                "required": ["item"], "additionalProperties": False}})
        self.assertNotIn("eat", {tool["name"] for tool in TOOL_DEFINITIONS})

    def test_agent_snapshot_preserves_flag_and_eat_epoch_without_changing_old_snapshots(self):
        """Forks restore an enabled schema and prior eat turn, while legacy states still default to off."""
        enabled = actor(daily_hunger=True)
        enabled.finish_turn("Lunch time.", ToolCall("eat", {"item": "bread"}), "started")
        restored = Agent.from_json(json.loads(json.dumps(enabled.to_json())))
        self.assertTrue(restored.daily_hunger)
        self.assertEqual(restored.tools, enabled.tools)
        self.assertEqual(restored.history(), enabled.history())
        legacy = actor().to_json()
        self.assertNotIn("dailyHunger", legacy)
        self.assertFalse(Agent.from_json(legacy).daily_hunger)
        self.assertEqual(Agent.from_json(legacy).to_json(), legacy)


class EatCandidatesTest(unittest.TestCase):
    """Only enabled scenarios offer observed held foods, with engine-grounded feasibility and Match."""

    def generate(self, daily_hunger):
        """Run a minimal scenario and retain every feasibility request, rejecting nonfood via the engine stub."""
        data = {"roles": [{"name": "alice"}], "places": [], "containers": {}, "schematics": {},
                "layout": {"fair": {"lantern": [0, 64, 0]}}}
        if daily_hunger is not None:
            data["daily_hunger"] = daily_hunger
        checked = []

        def feasible(agent, call):
            """Reject nonfood and an otherwise edible item, as the authoritative engine might."""
            checked.append((agent, call))
            return DryRun(call.name != "eat" or call.args["item"] == "bread", 32)

        world = FakeWorld({"alice": [0, 64, 0]}, inventories={"alice": [
            {"slot": 0, "item": "minecraft:bread", "count": 2},
            {"slot": 1, "item": "minecraft:bread", "count": 1},
            {"slot": 2, "item": "minecraft:oak_planks", "count": 4},
            {"slot": 3, "item": "minecraft:carrot", "count": 0},
            {"slot": 4, "item": "minecraft:apple", "count": 1}]})
        observed = world.request("observe", {"name": "alice"})
        generator = CandidateGenerator(ScenarioManifest(data), feasible)
        return generator.generate("alice", observed), checked

    def test_only_enabled_scenario_probes_and_offers_each_feasible_held_food_once(self):
        """Duplicate slots are deduplicated, empty slots ignored, and nonfood/failed skills are never offered."""
        candidates, checked = self.generate(True)
        meals = [candidate for candidate in candidates if candidate.call.name == "eat"]
        self.assertEqual([call.args for agent, call in checked if call.name == "eat"],
                         [{"item": "apple"}, {"item": "bread"}, {"item": "oak_planks"}])
        self.assertEqual([candidate.call for candidate in meals], [ToolCall("eat", {"item": "bread"})])
        self.assertEqual(meals[0].label, 'eat {"item":"bread"} | eat bread, about 32 ticks')
        self.assertEqual((meals[0].target, meals[0].distance), ((0, 64, 0), 0))
        self.assertTrue(representable(candidates, ToolCall("eat", {"item": "Bread", "note": "Lunch."})))

    def test_absent_or_false_flag_never_probes_or_offers_eat(self):
        """Missing and false top-level flags retain exactly the original candidate behavior."""
        for daily_hunger in (None, False):
            with self.subTest(daily_hunger=daily_hunger):
                candidates, checked = self.generate(daily_hunger)
                self.assertFalse(any(candidate.call.name == "eat" for candidate in candidates))
                self.assertFalse(any(call.name == "eat" for agent, call in checked))

    def test_match_uses_normalized_food_identity_not_notes(self):
        """Eat matches its own food, not another food, a missing item, or a drop action."""
        chosen = ToolCall("eat", {"item": "Bread", "note": "Hungry."})
        self.assertTrue(same_choice(chosen, ToolCall("eat", {"item": "minecraft:bread"})))
        for other in (ToolCall("eat", {"item": "apple"}), ToolCall("eat", {}),
                      ToolCall("drop", {"item": "bread", "count": 1})):
            with self.subTest(other=other):
                self.assertFalse(same_choice(chosen, other))


class DailyLifeManifestTest(unittest.TestCase):
    """Builder manifests select the same tools and candidates through their top-level feature flag."""

    @staticmethod
    def feasible(agent, call):
        """Model the engine's food check while allowing other observed skills in these integration fixtures."""
        return DryRun(call.name != "eat" or call.args.get("item") == "bread", 32)

    @staticmethod
    def world(manifest):
        """Stage every builder-defined role at its spawn with the inventory supplied by its kit."""
        return FakeWorld({role["name"]: role["spawn"] for role in manifest.roles}, inventories={
            role["name"]: [{"slot": slot, "item": item, "count": count}
                           for slot, (item, count) in enumerate(role["kit"].items())] for role in manifest.roles})

    def test_enabled_builder_manifest_produces_eat_tools_and_candidates_for_every_role(self):
        """The real daily-life variant gives each role bread, enabled agent schemas, and an eat choice."""
        manifest = ScenarioManifest.load(ROOT / "scenario" / "village_sabotage_daily_life.json")
        self.assertIs(manifest.data["daily_hunger"], True)
        self.assertNotIn("defaults", manifest.data)
        world = self.world(manifest)
        generator = CandidateGenerator(manifest, self.feasible)
        for agent in manifest.agents(epoch_turns=8):
            with self.subTest(agent=agent.name):
                self.assertTrue(agent.daily_hunger)
                self.assertEqual(agent.epoch_turns, 8)
                self.assertIn("eat", {tool["name"] for tool in agent.tools})
                observed = world.request("observe", {"name": agent.name})
                candidates = generator.generate(agent.name, observed)
                self.assertIn(ToolCall("eat", {"item": "bread"}), [candidate.call for candidate in candidates])

    def test_missing_and_false_manifest_flags_preserve_prompt_schema_and_candidate_bytes(self):
        """Legacy and explicit-off manifests produce identical five-turn prompts, tools, states and choices."""
        explicit = ScenarioManifest.load(ROOT / "scenario" / "village_sabotage.json")
        explicit.data["daily_hunger"] = False
        legacy = ScenarioManifest(deepcopy(explicit.data))
        legacy.data.pop("daily_hunger")
        world = self.world(explicit)
        legacy_candidates = CandidateGenerator(legacy, self.feasible)
        explicit_candidates = CandidateGenerator(explicit, self.feasible)
        expected_tools = (GOLDEN / "tools.json").read_bytes()
        for old, off in zip(legacy.agents(epoch_turns=8), explicit.agents(epoch_turns=8)):
            with self.subTest(agent=old.name):
                self.assertFalse(old.daily_hunger)
                self.assertFalse(off.daily_hunger)
                observed = world.request("observe", {"name": old.name})
                choices = legacy_candidates.generate(old.name, observed)
                self.assertEqual(choices, explicit_candidates.generate(off.name, observed))
                self.assertFalse(any(candidate.call.name == "eat" for candidate in choices))
                for turn in range(5):
                    observed["gameTime"] = turn * 40
                    prompts = [agent.next_prompt(observed, interrupted=turn == 4) for agent in (old, off)]
                    self.assertEqual(prompts[0].encode(), prompts[1].encode())
                    self.assertEqual(old.system_prompt().encode(), off.system_prompt().encode())
                    self.assertEqual(json.dumps(old.to_json(), sort_keys=True).encode(),
                                     json.dumps(off.to_json(), sort_keys=True).encode())
                    for agent, user in zip((old, off), prompts):
                        self.assertEqual((json.dumps(agent.tools, indent=1, ensure_ascii=False) + "\n").encode(),
                                         expected_tools)
                        agent.finish_turn(user, ToolCall("wait", {"ticks": 40}), "started")


class OwnHungerTest(unittest.TestCase):
    """Food totals alone never imply hunger; only the feature-gated own observation narrates it."""

    def test_hunger_is_narrated_only_when_engine_marks_self_hungry(self):
        """Below six is narrated when enabled, but low-food legacy observations and false flags stay unchanged."""
        observed = FakeWorld({"alice": [0, 64, 0]}).request("observe", {"name": "alice"})
        renderer = ObservationRenderer()
        for food in (0, 5, 6, 20):
            with self.subTest(food=food):
                observed["self"]["food"] = food
                legacy = renderer.render(observed)
                self.assertNotIn("You are hungry.", legacy)
                observed["self"]["hungry"] = food < 6
                rendered = renderer.render(observed)
                self.assertEqual("You are hungry." in rendered, food < 6)
                if food >= 6:
                    self.assertEqual(rendered, legacy)
                del observed["self"]["hungry"]


class DailyLifeRuntimeTest(unittest.TestCase):
    """Actual model schemas and every replay digest follow the agent's saved scenario flag."""

    def test_eat_polls_completion_updates_epoch_and_replays_exactly(self):
        """A running eat is not instant, adds no thinking gap, and replays without another model request."""
        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"eat": 80})
        agent = actor(daily_hunger=True)
        model = RecordingClient(lambda system, user: ToolCall("eat", {"item": "Bread"}))
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [agent], model, Path(directory) / "factual",
                                        command_error=CommandError, min_gap=100)
            runtime.run(200)
            decisions = runtime.log.decisions()
            self.assertEqual([decision.game_time for decision in decisions], [0, 80, 160])
            self.assertEqual([decision.outcome for decision in decisions], ["started"] * 3)
            self.assertIn("your eat", model.requests[1]["user"])
            self.assertEqual(model.requests[1]["history"][0].call.name, "eat")
            for request, decision in zip(model.requests, decisions):
                self.assertEqual(request["tools"], agent.tools)
                self.assertEqual(decision.digest, prompt_digest(model.model, **request))
            replay_world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"eat": 80})
            replay = SimulationRuntime(replay_world, [actor(daily_hunger=True)], None, Path(directory) / "replay",
                                       command_error=CommandError, source=runtime.log, min_gap=100)
            replay.run(200)
            self.assertEqual(replay.stats.replayed, 3)
            self.assertEqual(replay.stats.model_calls, 0)
            self.assertEqual(replay_world.requests, world.requests)
            runtime.save_state(Path(directory) / "state.json")
            unused, restored = SimulationRuntime.load_agents(Path(directory) / "state.json")
            self.assertTrue(restored[0].daily_hunger)
            self.assertEqual(restored[0].history(), agent.history())

    def test_off_schema_replays_but_enabling_eat_changes_digest(self):
        """Old tools remain byte-identical in actual requests and prevent cross-feature prompt-cache reuse."""
        model = RecordingClient(lambda system, user: ToolCall("wait", {"ticks": 40}))
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [actor()], model,
                                        Path(directory) / "factual", command_error=CommandError)
            runtime.run(80)
            self.assertTrue(all(request["tools"] == TOOL_DEFINITIONS for request in model.requests))
            replay = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [actor()], None,
                                       Path(directory) / "replay", command_error=CommandError, source=runtime.log)
            replay.run(80)
            self.assertEqual(replay.stats.replayed, 2)
            enabled = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [actor(daily_hunger=True)], None,
                                        Path(directory) / "enabled", command_error=CommandError, source=runtime.log)
            with self.assertRaises(ReplayDiverged):
                enabled.run(40)

    def test_enabled_forced_held_and_cross_model_reuse_digest_the_enabled_schema(self):
        """Fork-specific digest paths use eat-enabled tools even when they do not ask a model."""
        model = RecordingClient(lambda system, user: ToolCall("eat", {"item": "bread"}))
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [actor(daily_hunger=True)], model,
                                        Path(directory) / "factual", command_error=CommandError)
            runtime.run(40)
            for mode in ("forced", "held", "cross_model"):
                with self.subTest(mode=mode):
                    override = (DecisionOverride(agent="alice", game_time=0, call=ToolCall("eat", {"item": "apple"}))
                                if mode == "forced" else DecisionOverride(hold=frozenset({"alice"}))
                                if mode == "held" else DecisionOverride())
                    branch_model = RecordingClient(lambda system, user: None,
                                                   model="other" if mode == "cross_model" else model.model)
                    branch = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [actor(daily_hunger=True)],
                                               branch_model, Path(directory) / mode, command_error=CommandError,
                                               source=runtime.log, override=override)
                    branch.run(40)
                    self.assertEqual(branch.stats.model_calls, 0)
                    self.assertEqual(branch.log.decisions()[0].digest, runtime.log.decisions()[0].digest)

    def test_narrated_runtime_and_gm_gate_eat_and_track_its_duration(self):
        """Narrated requests, GM instructions, digesting and shadow execution all agree on the scenario tools."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                agent = actor(daily_hunger=enabled)
                call = ToolCall("eat", {"item": "bread"}) if enabled else ToolCall("wait", {"ticks": 80})
                model = RecordingClient(lambda system, user: call)
                gm = GameMaster(ScriptedClient(lambda system, user: ToolCall("resolve", {
                    "accepted": True, "reason": "", "duration_ticks": 80, "effects": [],
                    "actor_view": "Beside the bell.", "observations": []})), "A town.", {"alice": "Beside the bell."})
                shadow_world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"eat": 80})
                runtime = NarratedRuntime([agent], model, gm, Path(directory), min_gap=100,
                                          shadow=ShadowEngine(shadow_world, CommandError))
                runtime.run(0, 200)
                self.assertEqual([decision.game_time for decision in runtime.log.decisions()], [0, 80, 160])
                self.assertEqual("- eat: Eat food from your inventory" in gm._system, enabled)
                if not enabled:
                    self.assertEqual(gm._system, gm_system_prompt(TOOL_DEFINITIONS))
                for request, decision in zip(model.requests, runtime.log.decisions()):
                    self.assertEqual(request["tools"], agent.tools)
                    self.assertEqual(decision.digest, prompt_digest(model.model, **request))
                self.assertTrue(all(act["engine"] == "started" for act in runtime.acts))


if __name__ == "__main__":
    unittest.main()
