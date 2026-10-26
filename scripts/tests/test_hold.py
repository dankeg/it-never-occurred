"""Offline tests for the hold tool, grounded choices, visibility, and engine routing."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.candidates import Candidate, CandidateGenerator, DryRun, EngineDryRun, representable, same_choice
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import ShadowEngine, gm_system_prompt
from orchestrator.perception import DecisionContext, EvidenceExposure, PerceptionFeatures, VillagerTracks
from orchestrator.runtime import SimulationRuntime, failure_hint
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, TOOL_NAMES, ToolTranslator
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld

ROOT = Path(__file__).resolve().parents[2]


class SynchronousHoldWorld(FakeWorld):
    """Model the engine's hold completion at the same game time as skill_start."""

    def _start(self, name, skill, args, *, dry_run=False, ticks=None):
        """Finish accepted hold skills before returning, while leaving other scripted skills unchanged."""
        result = super()._start(name, skill, args, dry_run=dry_run, ticks=0 if skill == "hold" else ticks)
        if skill == "hold" and result.get("accepted"):
            self._finish_tick()
        return result


class HoldTranslationTest(unittest.TestCase):
    """Holding exposes only an item and optional note, and starts a sanitized engine skill."""

    def test_schema_matches_the_spec(self):
        """The model sees the exact description, a required string item, and the shared optional note."""
        definitions = [tool for tool in TOOL_DEFINITIONS if tool["name"] == "hold"]
        self.assertEqual(len(definitions), 1)
        self.assertIn("hold", TOOL_NAMES)
        self.assertEqual(definitions[0]["description"],
                         "Hold an item from your inventory in your hand, where others can see it.")
        self.assertEqual(definitions[0]["parameters"], {
            "type": "object", "properties": {"item": {"type": "string"},
                                              "note": {"type": "string", "maxLength": 100}},
            "required": ["item"], "additionalProperties": False})

    def test_translation_normalizes_items(self):
        """Human names, registry ids, namespaces, and copied book labels share existing item normalization."""
        for supplied, expected in ((" Iron Pickaxe ", "minecraft:iron_pickaxe"),
                                   ("minecraft:iron_pickaxe", "minecraft:iron_pickaxe"),
                                   ("custom:Iron Pickaxe", "custom:iron_pickaxe"),
                                   ('written_book "Report" by mayor', "minecraft:written_book")):
            with self.subTest(item=supplied):
                action = ToolTranslator().translate("envoy", ToolCall("hold", {"item": supplied}))
                self.assertEqual(action.command, "skill_start")
                self.assertEqual(action.args, {"name": "envoy", "skill": "hold", "args": {"item": expected}})
                self.assertTrue(action.is_skill)

    def test_extra_arguments_are_filtered_without_mutating_the_call(self):
        """Counts, slots, targets, notes, and engine-only flags cannot leak into hold skill arguments."""
        call = ToolCall("hold", {"item": "Iron Pickaxe", "note": "show this", "count": 64, "slot": 8,
                                 "target": "builder", "dry_run": True, "unlimited_lifetime": True})
        original = deepcopy(call.args)
        action = ToolTranslator().translate("envoy", call)
        self.assertEqual(action.args, {"name": "envoy", "skill": "hold", "args": {"item": "minecraft:iron_pickaxe"}})
        self.assertEqual(call.args, original)

    def test_item_is_required(self):
        """A note alone cannot start holding an unspecified item."""
        with self.assertRaises(KeyError):
            ToolTranslator().translate("envoy", ToolCall("hold", {"note": "show it"}))

    def test_match_uses_only_the_normalized_item(self):
        """Hold matches the same item despite notes or stray counts, but never another item or action."""
        call = ToolCall("hold", {"item": " Iron Pickaxe ", "note": "show this"})
        self.assertTrue(same_choice(call, ToolCall("hold", {"item": "minecraft:iron_pickaxe", "count": 2})))
        self.assertFalse(same_choice(call, ToolCall("hold", {"item": "bread"})))
        self.assertFalse(same_choice(call, ToolCall("drop", {"item": "iron_pickaxe", "count": 1})))
        self.assertFalse(same_choice(call, None))


class HoldCandidatesTest(unittest.TestCase):
    """Candidates come only from distinct nonempty inventory items and engine feasibility."""

    def setUp(self):
        """Load the real manifest and record checks while rejecting unrelated skills."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.checks = []
        self.rejected = set()
        self.generator = CandidateGenerator(self.manifest, self._feasible)

    def _feasible(self, agent, call):
        """Record each check and allow only non-rejected hold calls with zero duration."""
        self.checks.append((agent, call))
        feasible = call.name == "hold" and call.args["item"] not in self.rejected
        return DryRun(feasible, 0 if feasible else None, None if feasible else "missing_items")

    def _generate(self, inventory):
        """Generate the actor's hold choices with no other people or items visible."""
        observation = {"self": {"pos": [0, 64, 0], "inventory": inventory}}
        return [candidate for candidate in self.generator.generate("envoy", observation)
                if candidate.call.name == "hold"]

    def test_inventory_candidates_are_unique_sorted_and_grounded(self):
        """Duplicate stacks normalize to one item; zero stacks add nothing and quantities stay private."""
        inventory = [{"slot": 8, "item": "minecraft:iron_pickaxe", "count": 1},
                     {"slot": 10, "item": "bread", "count": 3},
                     {"slot": 0, "item": "minecraft:bread", "count": 2},
                     {"slot": 12, "item": "minecraft:emerald", "count": 0}]
        original = deepcopy(inventory)
        candidates = self._generate(inventory)
        self.assertEqual([candidate.call for candidate in candidates],
                         [ToolCall("hold", {"item": "bread"}), ToolCall("hold", {"item": "iron_pickaxe"})])
        self.assertEqual([call.args for agent, call in self.checks if call.name == "hold"],
                         [{"item": "bread"}, {"item": "iron_pickaxe"}])
        self.assertTrue(all(candidate.distance == 0 and candidate.target == (0, 64, 0) for candidate in candidates))
        self.assertIn("hold bread where others can see it, about 0 ticks", candidates[0].label)
        self.assertEqual(inventory, original)
        self.assertTrue(representable(candidates, ToolCall("hold", {"item": "minecraft:bread", "note": "show it"})))
        self.assertFalse(representable(candidates, ToolCall("hold", {"item": "emerald"})))

    def test_inventory_order_does_not_change_candidates_or_checks(self):
        """Shuffling inventory slots leaves both choice order and hold feasibility request order unchanged."""
        inventory = [{"slot": 3, "item": "minecraft:iron_pickaxe", "count": 1},
                     {"slot": 1, "item": "minecraft:bread", "count": 2}]
        candidates = self._generate(inventory)
        checks = [(agent, call) for agent, call in self.checks if call.name == "hold"]
        self.checks.clear()
        self.assertEqual(self._generate(list(reversed(inventory))), candidates)
        self.assertEqual([(agent, call) for agent, call in self.checks if call.name == "hold"], checks)

    def test_empty_inventory_offers_no_hold(self):
        """Holding cannot invent items from the manifest or a remembered container."""
        self.assertEqual(self._generate([]), [])
        self.assertFalse(any(call.name == "hold" for agent, call in self.checks))

    def test_engine_rejection_removes_the_candidate(self):
        """A carried item is offered only when the engine permits starting the hold skill."""
        self.rejected.add("bread")
        self.assertEqual(self._generate([{"item": "minecraft:bread", "count": 1}]), [])
        self.assertIn(("envoy", ToolCall("hold", {"item": "bread"})), self.checks)


class HoldRuntimeTest(unittest.TestCase):
    """Zero-duration holding keeps normal skill lifecycle, dry-run, and missing-item feedback."""

    def test_dry_run_and_narrated_shadow_use_the_same_zero_tick_skill(self):
        """Both feasibility paths and shadow execution send only the normalized item through skill_start."""
        world = SynchronousHoldWorld({"envoy": [0, 64, 0]})
        call = ToolCall("hold", {"item": "Iron Pickaxe", "count": 8, "note": "show this"})
        self.assertEqual(EngineDryRun(world, CommandError)("envoy", call), DryRun(True, 0))
        shadow = ShadowEngine(world, CommandError)
        self.assertEqual(shadow.check("envoy", call), {"feasible": True, "reason": None, "ticks": 0})
        self.assertEqual(shadow.execute("envoy", call), "started")
        args = {"name": "envoy", "skill": "hold", "args": {"item": "minecraft:iron_pickaxe"}}
        self.assertEqual(world.requests, [("skill_start", {**args, "dry_run": True})] * 2 + [("skill_start", args)])
        status = world.request("skill_status", {"name": "envoy"})
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(status["startedAt"], world.game_time)
        self.assertEqual(status["endTick"], world.game_time)

    def test_runtime_records_and_polls_the_instant_skill(self):
        """Holding uses skill_start despite zero duration and permits a decision at the next normal poll."""
        call = ToolCall("hold", {"item": "Iron Pickaxe", "note": "show it", "slot": 8})

        def policy(system, user):
            """Repeat the same hold call without a model or server."""
            return call

        inventory = [{"slot": 10, "item": "minecraft:iron_pickaxe", "count": 1}]
        world = SynchronousHoldWorld({"envoy": [0, 64, 0]}, inventories={"envoy": inventory})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("envoy", "envoy", "Works.", "Help."))],
                                        ScriptedClient(policy), Path(directory), command_error=CommandError,
                                        min_gap=100)
            runtime.run(80)
            self.assertEqual([(decision.game_time, decision.outcome) for decision in runtime.log.decisions()],
                             [(0, "started"), (40, "started")])
        expected = {"name": "envoy", "skill": "hold", "args": {"item": "minecraft:iron_pickaxe"}}
        self.assertEqual([args for command, args in world.requests if command == "skill_start"], [expected] * 2)
        self.assertTrue(all(skill["endTick"] == skill["startedAt"] and skill["state"] == "succeeded"
                            for skill in world.skills["envoy"]))

    def test_missing_item_dry_run_is_rejected(self):
        """An engine missing_items verdict remains a rejected zero-side-effect feasibility check."""
        world = FakeWorld({"envoy": [0, 64, 0]}, rejects={"hold": "missing_items"})
        verdict = EngineDryRun(world, CommandError)("envoy", ToolCall("hold", {"item": "bread"}))
        self.assertEqual(verdict, DryRun(False, reason="missing_items"))
        self.assertEqual(world.skills["envoy"], [])

    def test_missing_items_hint_is_about_holding_not_materials_or_counts(self):
        """Rejected holding points to a carried item, while success and unrelated failures add no hint."""
        call = ToolCall("hold", {"item": "bread"})
        hint = failure_hint(call, "not possible: missing_items", {})
        self.assertIn("hold an item you carry", hint)
        self.assertIn("check the item in your inventory", hint)
        self.assertNotIn("materials", hint)
        self.assertNotIn("count", hint)
        for outcome in ("started", "done", "succeeded", "not possible: busy"):
            with self.subTest(outcome=outcome):
                self.assertEqual(failure_hint(call, outcome, {}), "")

    def test_gm_catalogue_includes_the_exact_hold_description(self):
        """The existing GM catalogue automatically picks up hold and its grounded engine rules."""
        self.assertIn("- hold: Hold an item from your inventory in your hand, where others can see it.",
                      gm_system_prompt(TOOL_DEFINITIONS))

    def test_gm_rules_enforce_instant_inventory_only_holding(self):
        """The narrator must not invent items, time costs, transfers or unseen inventory information."""
        prompt = gm_system_prompt(TOOL_DEFINITIONS)
        for rule in ("Holding an item takes 0 ticks", "otherwise missing_items", "Select its hotbar slot",
                     "without consuming or transferring items", "reveals no inventory contents or counts"):
            with self.subTest(rule=rule):
                self.assertIn(rule, prompt)


class HoldPerceptionTest(unittest.TestCase):
    """Holding is a visible, non-addressed inventory action, not speech or a transfer."""

    def test_only_actual_seers_perceive_hold(self):
        """Nearby non-seers do not hear a hold, and holding invents no addressee or goal target."""
        manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        observations = {
            "envoy": {"self": {"pos": [0, 64, 0]}},
            "builder": {"self": {"pos": [3, 64, 0]},
                        "entities": [{"type": "minecraft:player", "name": "envoy"}]},
            "mayor": {"self": {"pos": [2, 64, 0]}, "entities": []},
        }
        exposure = EvidenceExposure(manifest, {"builder": {"west_supports": 90}, "mayor": {"west_supports": 90}})
        context = DecisionContext("envoy", 100, observations, exposure, VillagerTracks([]))
        candidate = Candidate(ToolCall("hold", {"item": "iron_pickaxe"}), "hold the pickaxe", 0, (0, 64, 0))
        features = PerceptionFeatures(manifest).features(candidate, context)
        self.assertEqual(features["seen_by_agents"], 1.0)
        self.assertEqual(features["perceivers_informed"], 1.0)
        self.assertEqual(features["ticks_to_perception"], 0.0)
        for feature in ("earshot", "free_text", "private_evidence", "addressed_to_agent", "goal_relevance"):
            with self.subTest(feature=feature):
                self.assertEqual(features[feature], 0.0)


if __name__ == "__main__":
    unittest.main()
