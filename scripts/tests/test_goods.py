"""Goods tools, engine-provided freshness and failure hints, with byte-stable default-off prompts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.agent import Agent, RoleBrief
from orchestrator.candidates import CandidateGenerator, DryRun, EngineDryRun, known_contents, same_choice
from orchestrator.candidates import _TOOL_ORDER
from orchestrator.decisions import prompt_digest
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import ENGINE_RULES, GM_RULES, GameMaster, NarratedRuntime, ShadowEngine, gm_system_prompt
from orchestrator.render import ObservationRenderer, stack_text
from orchestrator.runtime import ReplayDiverged, SimulationRuntime, failure_hint
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, TOOL_NAMES, ToolTranslator, tool_definitions
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld

GOLDEN = Path(__file__).with_name("golden")
CAPACITY = {"reason": "too_heavy", "item": "minecraft:bread", "limit": 32, "have": 30, "remaining": 2,
            "hint": "you can carry 32 bread at most; you have 30; you can take 2 more"}


def observation(inventory=(), *, game_time=0):
    """Build an actor's view with only supplied inventory facts and no visible people or blocks."""
    return {"gameTime": game_time, "self": {"name": "alice", "pos": [0, 64, 0], "health": 20, "food": 20,
                                         "inventory": list(inventory), "skill": {"state": "idle"}},
            "entities": [], "blocks": [], "documents": []}


def manifest(**config):
    """Build a minimal candidate manifest without implying that goods are enabled."""
    return ScenarioManifest({"roles": [{"name": "alice"}], "places": [], "containers": {}, "schematics": {},
                             "layout": {"fair": {"lantern": [0, 64, 0]}}, **config})


class GoodsWorld(FakeWorld):
    """Return scripted goods rejection details immediately or when a timed skill finishes."""

    def __init__(self, *, rejection=None, delayed_failure=None, **kwargs):
        """Keep failure responses separate from the fake world's unrelated command behavior."""
        super().__init__({"alice": [0, 64, 0]}, **kwargs)
        self.rejection = deepcopy(rejection)
        self.delayed_failure = deepcopy(delayed_failure)

    def request(self, command, args=None):
        """Record a rejected baking start with exactly the engine's top-level fields."""
        if command == "skill_start" and args.get("skill") == "bake" and self.rejection is not None:
            self.requests.append((command, deepcopy(args)))
            return {"accepted": False, **deepcopy(self.rejection)}
        return super().request(command, args)

    def _finish_tick(self):
        """Put the same rejection fields in progress after an otherwise timed completion."""
        super()._finish_tick()
        for skills in self.skills.values():
            if (self.delayed_failure is not None and skills and skills[-1]["state"] == "succeeded"
                    and skills[-1]["skill"] == "bake"):
                skills[-1].update(state="failed", reason=self.delayed_failure["reason"],
                                  progress=deepcopy(self.delayed_failure))


class BakeToolTest(unittest.TestCase):
    """Baking is a count-only opt-in tool translated to the engine's timed skill."""

    def test_schema_is_explicitly_enabled_and_legacy_tools_are_unchanged(self):
        """Absent baking configuration preserves the exact tool snapshot, including daily-life selection."""
        self.assertIs(tool_definitions(), TOOL_DEFINITIONS)
        self.assertIs(tool_definitions(extras=()), TOOL_DEFINITIONS)
        self.assertEqual((json.dumps(TOOL_DEFINITIONS, indent=1, ensure_ascii=False) + "\n").encode(),
                         (GOLDEN / "tools.json").read_bytes())
        self.assertNotIn("bake", {tool["name"] for tool in TOOL_DEFINITIONS})
        self.assertIn("bake", TOOL_NAMES)
        for hunger in (False, True):
            with self.subTest(daily_hunger=hunger):
                enabled = tool_definitions(daily_hunger=hunger, extras=("bake",))
                self.assertEqual(enabled[:-1], tool_definitions(daily_hunger=hunger))
                parameters = enabled[-1]["parameters"]
                self.assertEqual(enabled[-1]["name"], "bake")
                self.assertEqual(parameters["required"], ["count"])
                self.assertEqual(parameters["properties"], {
                    "count": {"type": "integer", "minimum": 1, "maximum": 32},
                    "note": {"type": "string", "maxLength": 100}})
                self.assertFalse(parameters["additionalProperties"])

    def test_every_valid_count_translates_without_engine_only_arguments(self):
        """A batch never supplies its oven, timing, stamp or output directly."""
        for count in range(1, 33):
            with self.subTest(count=count):
                action = ToolTranslator().translate("alice", ToolCall("bake", {
                    "count": count, "note": "Bake.", "pos": [1, 64, 0], "bake_ticks": 0, "baked_day": -3}))
                self.assertEqual(action.command, "skill_start")
                self.assertEqual(action.args, {"name": "alice", "skill": "bake", "args": {"count": count}})
                self.assertTrue(action.is_skill)

    def test_missing_noninteger_and_out_of_range_counts_are_rejected(self):
        """Booleans and integral floats are not accepted as JSON integer counts, and counts never get clamped."""
        with self.assertRaises(KeyError):
            ToolTranslator().translate("alice", ToolCall("bake", {}))
        for count in (None, True, False, "4", 4.0, [], {}):
            with self.subTest(count=count), self.assertRaises(TypeError):
                ToolTranslator().translate("alice", ToolCall("bake", {"count": count}))
        for count in (-1, 0, 33, 100):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ToolTranslator().translate("alice", ToolCall("bake", {"count": count}))

    def test_dry_run_and_shadow_use_the_same_count_only_translation(self):
        """Feasibility and narrated shadow execution keep the authoritative engine duration."""
        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"bake": 160})
        call = ToolCall("bake", {"count": 4})
        self.assertEqual(EngineDryRun(world, CommandError)("alice", call), DryRun(True, 160))
        shadow = ShadowEngine(world, CommandError)
        self.assertEqual(shadow.check("alice", call), {"feasible": True, "reason": None, "ticks": 160})
        expected = {"name": "alice", "skill": "bake", "args": {"count": 4}, "dry_run": True}
        self.assertEqual(world.requests, [("skill_start", expected)] * 2)
        self.assertEqual(shadow.execute("alice", call), "started")
        shadow.advance(120)
        self.assertEqual(shadow.skill_state("alice"), "running")
        shadow.advance(160)
        self.assertEqual(shadow.skill_state("alice"), "succeeded")


class GoodsActivationTest(unittest.TestCase):
    """Actual engine configuration reaches tool schemas, replay digests, agent snapshots and narrated runs."""

    def test_agent_snapshot_restores_enabled_tools_without_changing_disabled_snapshots(self):
        """Only enabled optional tools add a snapshot field; legacy snapshots restore with baking off."""
        agent = Agent(RoleBrief("alice", "baker", "Bakes.", "Work."), epoch_turns=8, extras=("bake",))
        agent.finish_turn("Beside the oven.", ToolCall("bake", {"count": 4}), "started")
        restored = Agent.from_json(json.loads(json.dumps(agent.to_json())))
        self.assertEqual(restored.extras, ("bake",))
        self.assertEqual(restored.tools, agent.tools)
        self.assertEqual(restored.history(), agent.history())
        legacy = Agent(agent.brief).to_json()
        self.assertNotIn("tools", legacy)
        self.assertEqual(Agent.from_json(legacy).extras, ())
        self.assertEqual(Agent.from_json(legacy).to_json(), legacy)

    def test_scenario_agents_take_their_optional_tools_from_the_manifest(self):
        """A manifest's `tools` list enables baking; without one the village keeps its default tools."""
        root = Path(__file__).resolve().parents[2]
        data = ScenarioManifest.load(root / "scenario" / "village_sabotage.json")
        self.assertTrue(all(agent.tools is TOOL_DEFINITIONS for agent in data.agents()))
        data.data["tools"] = ["bake"]
        self.assertTrue(all(agent.extras == ("bake",) for agent in data.agents()))

    def test_engine_config_enables_real_model_tools_and_round_trips_through_replay(self):
        """An explicit config enables baking without extra protocol calls and replays the engine duration exactly."""
        world = FakeWorld({"alice": [0, 64, 0]}, config={"bake_ticks": 40}, skill_ticks={"bake": 160})
        agent = Agent(RoleBrief("alice", "baker", "Bakes.", "Work."), extras=("bake",))
        model = ScriptedClient(lambda system, user: ToolCall("bake", {"count": 4}))
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "complete", wraps=model.complete) as ask:
            runtime = SimulationRuntime(world, [agent], model, Path(directory) / "factual", command_error=CommandError)
            runtime.run(80)
            runtime.run(160)
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(ask.call_args.args[2], tool_definitions(extras=("bake",)))
            decision = runtime.log.decisions()[0]
            system, user, tools = ask.call_args.args
            self.assertEqual(decision.digest, prompt_digest(model.model, system, user, tools,
                                                           ask.call_args.kwargs["history"]))
            self.assertNotIn("config_get", [command for command, args in world.requests])
            self.assertEqual([args for command, args in world.requests if command == "skill_start"],
                             [{"name": "alice", "skill": "bake", "args": {"count": 4}}])
            runtime.save_state(Path(directory) / "state.json")
            unused, restored = SimulationRuntime.load_agents(Path(directory) / "state.json")
            self.assertEqual(restored[0].tools, agent.tools)
            replay = SimulationRuntime(
                FakeWorld({"alice": [0, 64, 0]}, config={"bake_ticks": 40}, skill_ticks={"bake": 160}),
                [Agent(agent.brief, extras=("bake",))], None, Path(directory) / "replay", command_error=CommandError,
                source=runtime.log)
            replay.run(160)
            self.assertEqual(replay.stats.replayed, 1)
            self.assertEqual(replay.log.decisions()[0].digest, decision.digest)

    def test_engine_config_keeps_default_model_tools_and_prevents_cross_feature_reuse(self):
        """Default tools stay the legacy object, and enabling baking changes the replay digest."""
        agent = Agent(RoleBrief("alice", "baker", "Bakes.", "Work."))
        model = ScriptedClient(lambda system, user: ToolCall("wait", {"ticks": 40}))
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "complete", wraps=model.complete) as ask:
            runtime = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}), [agent], model,
                                        Path(directory) / "off", command_error=CommandError)
            runtime.run(40)
            self.assertIs(ask.call_args.args[2], TOOL_DEFINITIONS)
            self.assertNotIn("tools", agent.to_json())
            replay = SimulationRuntime(FakeWorld({"alice": [0, 64, 0]}, config={"bake_ticks": 40}),
                                       [Agent(agent.brief, extras=("bake",))], None, Path(directory) / "on",
                                       command_error=CommandError, source=runtime.log)
            with self.assertRaises(ReplayDiverged):
                replay.run(40)

    def test_narrated_runtime_enables_bake_for_the_agent_and_game_master(self):
        """A configured actor's real request and GM instruction both include baking on the first turn."""
        agent = Agent(RoleBrief("alice", "baker", "Bakes.", "Work."), extras=("bake",))
        model = ScriptedClient(lambda system, user: ToolCall("bake", {"count": 4}))
        gm = GameMaster(ScriptedClient(lambda system, user: ToolCall("resolve", {
            "accepted": True, "reason": "", "duration_ticks": 160, "effects": [],
            "actor_view": "Beside the oven.", "observations": []})), "A bakery.", {"alice": "Beside the oven."},
            bake_ticks=40)
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "complete", wraps=model.complete) as ask:
            runtime = NarratedRuntime([agent], model, gm, Path(directory))
            runtime.run(0, 80)
            self.assertEqual(ask.call_args.args[2], tool_definitions(extras=("bake",)))
            self.assertEqual(runtime.log.decisions()[0].tool, "bake")
            self.assertIn("- bake:", gm._system)
            self.assertIn("40 ticks per loaf", gm._system)


class FreshnessRenderingTest(unittest.TestCase):
    """Freshness comes only from per-stack engine fields, never from a second Python age rule."""

    def test_inventory_keeps_each_observed_bread_stack_separate(self):
        """Fresh, mouldy, negative-day and unstamped-day-zero stacks retain their observed labels and counts."""
        stacks = [{"slot": 0, "item": "minecraft:bread", "count": 32, "baked_day": 1, "mouldy": False},
                  {"slot": 1, "item": "minecraft:bread", "count": 64, "baked_day": -3, "mouldy": True},
                  {"slot": 2, "item": "minecraft:bread", "count": 2, "baked_day": -1, "mouldy": False},
                  {"slot": 3, "item": "minecraft:bread", "count": 1, "baked_day": 0, "mouldy": False}]
        rendered = ObservationRenderer().render(observation(stacks, game_time=24_000))
        for line in ("slot 0: bread ×32 (baked day 1)", "slot 1: mouldy bread ×64",
                     "slot 2: bread ×2 (baked day -1)", "slot 3: bread ×1 (baked day 0)"):
            self.assertIn(line, rendered)
        self.assertEqual(rendered.count("slot "), 4)

    def test_renderer_does_not_recompute_the_engine_mouldy_boolean(self):
        """Even at another observation day the Python renderer trusts the supplied freshness verdict."""
        stack = {"slot": 0, "item": "minecraft:bread", "count": 32, "baked_day": -3, "mouldy": False}
        rendered = ObservationRenderer().render(observation([stack], game_time=240_000))
        self.assertIn("bread ×32 (baked day -3)", rendered)
        self.assertNotIn("mouldy bread", rendered)
        self.assertEqual(stack_text({**stack, "mouldy": True}), "mouldy bread ×32")

    def test_absent_freshness_fields_preserve_legacy_items_and_documents(self):
        """A hidden custom-data stamp or fields on nonbread cannot opt into bread freshness rendering."""
        stacks = [{"slot": 0, "item": "minecraft:bread", "count": 32,
                   "custom_data": {"simharness_baked_day": -3}},
                  {"slot": 1, "item": "minecraft:wheat", "count": 9, "baked_day": -3, "mouldy": True},
                  {"slot": 2, "item": "minecraft:written_book", "count": 1,
                   "doc": {"title": "Appointments", "author": "secretary"}}]
        rendered = ObservationRenderer().render(observation(stacks))
        self.assertIn("slot 0: 32 bread", rendered)
        self.assertIn("slot 1: 9 wheat", rendered)
        self.assertIn('slot 2: 1 written_book "Appointments" by secretary', rendered)
        self.assertNotIn("baked day", rendered)
        self.assertNotIn("mouldy", rendered)

    def test_public_baking_activity_names_the_oven_and_only_its_visible_position(self):
        """An activity's private batch count, duration and ingredients are not rendered."""
        renderer = ObservationRenderer()
        entity = {"type": "minecraft:player", "name": "baker", "pos": [0, 64, 0], "distance": 2,
                  "activity": {"skill": "bake", "pos": [1, 64, 2], "count": 32, "args": {"count": 32}}}
        self.assertEqual(renderer.entity_line(entity),
                         "baker at (0, 64, 0), 2 blocks away, baking at the oven at (1, 64, 2)")
        entity["activity"] = {"skill": "bake"}
        self.assertTrue(renderer.entity_line(entity).endswith(", baking at the oven"))


class GoodsRuntimeTest(unittest.TestCase):
    """Container memory and both immediate and delayed failures preserve the engine's contract."""

    def run_action(self, world, call, until=40):
        """Run one scripted action through the normal runtime without starting a real server or model."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        agent = Agent(RoleBrief("alice", "baker", "Bakes bread.", "Run the bakery."))
        runtime = SimulationRuntime(world, [agent], ScriptedClient(lambda system, user: call),
                                    Path(directory.name), command_error=CommandError)
        runtime.run(until)
        return runtime, agent

    def test_container_read_uses_the_same_stack_labels_and_remains_parseable(self):
        """New stack labels survive memory parsing and candidate generation without changing registry ids."""
        stacks = [{"slot": 0, "item": "minecraft:bread", "count": 32, "baked_day": 1, "mouldy": False},
                  {"slot": 1, "item": "minecraft:bread", "count": 64, "baked_day": -3, "mouldy": True},
                  {"slot": 2, "item": "minecraft:emerald", "count": 7}]
        world = FakeWorld({"alice": [0, 64, 0]}, containers={(1, 64, 0): stacks})
        unused, agent = self.run_action(world, ToolCall("open_container", {"x": 1, "y": 64, "z": 0}))
        self.assertEqual(agent.memory.to_json()["reads"][0]["text"],
                         "container at (1, 64, 0): bread ×32 (baked day 1), mouldy bread ×64, 7 emerald")
        self.assertEqual(known_contents(agent.memory), {(1, 64, 0): [("bread", 32), ("bread", 64), ("emerald", 7)]})
        data = manifest()
        data.data["containers"] = {"store": {"pos": [1, 64, 0], "allowed": None}}
        view = observation()
        view["blocks"] = [{"block": "minecraft:chest", "pos": [1, 64, 0], "distance": 1}]
        candidates = CandidateGenerator(data, lambda agent, call: DryRun(True)).generate("alice", view, agent.memory)
        takes = [choice.call.args for choice in candidates if choice.call.name == "chest_take"]
        self.assertEqual({(args["item"], args["count"]) for args in takes},
                         {("bread", 32), ("bread", 64), ("emerald", 7)})

    def test_legacy_container_reads_and_latest_negative_day_reads_still_parse(self):
        """Count-first old reads stay unchanged, and the latest read still wins per container."""
        world = FakeWorld({"alice": [0, 64, 0]}, containers={(1, 64, 0): [
            {"slot": 0, "item": "minecraft:bread", "count": 32}]})
        unused, agent = self.run_action(world, ToolCall("open_container", {"x": 1, "y": 64, "z": 0}))
        self.assertEqual(agent.memory.to_json()["reads"][0]["text"], "container at (1, 64, 0): 32 bread")
        agent.memory.record_read(40, "container at (1, 64, 0): bread ×2 (baked day -1)")
        self.assertEqual(known_contents(agent.memory), {(1, 64, 0): [("bread", 2)]})
        agent.memory.record_read(80, "container at (1, 64, 0): empty")
        self.assertEqual(known_contents(agent.memory), {(1, 64, 0): []})

    def test_immediate_capacity_failure_preserves_the_engine_hint_and_count(self):
        """The engine's optional top-level details reach the outcome verbatim, with no retry or partial request."""
        runtime, agent = self.run_action(GoodsWorld(rejection=CAPACITY), ToolCall("bake", {"count": 4}))
        expected = f"not possible: too_heavy ({CAPACITY['hint']})"
        self.assertEqual(runtime.log.decisions()[0].outcome, expected)
        self.assertEqual(agent.memory.to_json()["actions"][0]["outcome"], expected)
        starts = [args for command, args in runtime._client.requests if command == "skill_start"]
        self.assertEqual(starts, [{"name": "alice", "skill": "bake", "args": {"count": 4}}])

    def test_delayed_capacity_failure_uses_progress_and_enters_the_next_prompt(self):
        """A started skill's progress hint survives resolution into action memory before the next decision."""
        runtime, agent = self.run_action(GoodsWorld(delayed_failure=CAPACITY), ToolCall("bake", {"count": 4}), 80)
        self.assertEqual(runtime.log.decisions()[0].outcome, "started")
        self.assertIn(CAPACITY["hint"], agent.memory.to_json()["actions"][0]["outcome"])
        self.assertIn(CAPACITY["hint"], runtime.log.decisions()[1].user_prompt)
        self.assertIsNone(agent.memory.to_json()["actions"][0]["skillId"])

    def test_other_delayed_goods_reasons_receive_their_grounded_hints(self):
        """Delayed reasons are explained even when the engine supplies no prose hint."""
        for reason, words in (("no_oven", "in sight"), ("not_enough_wheat", "3 wheat"), ("mouldy", "cannot be eaten")):
            with self.subTest(reason=reason):
                unused, agent = self.run_action(GoodsWorld(delayed_failure={"reason": reason}),
                                                ToolCall("bake", {"count": 4}), 80)
                self.assertIn(words, agent.memory.to_json()["actions"][0]["outcome"])


class GoodsHintsTest(unittest.TestCase):
    """Reason-only fallback hints remain useful without assuming a hidden capacity or recipient inventory."""

    def test_structured_capacity_fields_are_used_when_the_prose_hint_is_absent(self):
        """Limit, have and remaining are authoritative even if the observation contains another count."""
        view = observation([{"slot": 0, "item": "minecraft:bread", "count": 1}])
        details = {key: value for key, value in CAPACITY.items() if key != "hint"}
        hint = failure_hint(ToolCall("bake", {"count": 4}), "not possible: too_heavy", view, details)
        for words in ("32 bread at most", "you have 30 bread", "you can take 2 more", "partial"):
            self.assertIn(words, hint)
        self.assertNotIn("you have 1 bread", hint)

    def test_zero_capacity_fields_are_not_discarded(self):
        """Zero limit, carried count and remaining space are meaningful engine values."""
        hint = failure_hint(ToolCall("bake", {"count": 1}), "failed: too_heavy", observation(),
                            {"item": "minecraft:bread", "limit": 0, "have": 0, "remaining": 0})
        for words in ("carry 0 bread", "have 0 bread", "take 0 more"):
            self.assertIn(words, hint)

    def test_observed_capacity_and_split_inventory_counts_are_a_fallback(self):
        """A known self capacity can be combined with all observed stacks, but no default limit is invented."""
        view = observation([{"slot": 0, "item": "minecraft:bread", "count": 20},
                            {"slot": 1, "item": "minecraft:bread", "count": 10}])
        view["self"]["carry_limits"] = {"minecraft:bread": 32}
        call = ToolCall("chest_take", {"item": "Bread", "count": 4})
        self.assertIn("you have 30 bread", failure_hint(call, "not possible: too_heavy", view))
        self.assertIn("carry 32 bread", failure_hint(call, "not possible: too_heavy", view))
        self.assertNotIn("carry 32", failure_hint(call, "not possible: too_heavy", observation()))

    def test_give_uses_recipient_details_not_the_givers_inventory(self):
        """Capacity is about the person receiving items, not the person holding them now."""
        call = ToolCall("give", {"target": "baker", "item": "bread", "count": 4})
        details = {key: value for key, value in CAPACITY.items() if key != "hint"}
        hint = failure_hint(call, "failed: too_heavy", observation(), details)
        self.assertIn("baker can carry 32 bread", hint)
        self.assertIn("baker has 30 bread", hint)
        self.assertNotIn("you have", hint)
        self.assertNotIn("has 0", failure_hint(call, "failed: too_heavy", observation()))

    def test_reason_only_hints_explain_oven_wheat_mould_and_count_selection(self):
        """Older reason-only responses need neither new observation fields nor response metadata."""
        bake = ToolCall("bake", {"count": 4})
        view = observation([{"slot": 0, "item": "minecraft:wheat", "count": 5},
                            {"slot": 1, "item": "minecraft:wheat", "count": 4}])
        self.assertIn("smoker within reach and in sight", failure_hint(bake, "not possible: no_oven", view))
        wheat_hint = failure_hint(bake, "not possible: not_enough_wheat", view)
        for words in ("3 wheat", "you have 9"):
            self.assertIn(words, wheat_hint)
        self.assertIn("cannot be eaten", failure_hint(ToolCall("eat", {"item": "bread"}), "failed (mouldy)", view))
        self.assertIn("carrying limit", failure_hint(ToolCall("collect", {}), "failed: too_heavy", {}))
        self.assertEqual(failure_hint(None, "failed: too_heavy", view, CAPACITY), "")
        self.assertEqual(failure_hint(bake, "started", view, CAPACITY), "")


class BakeCandidatesTest(unittest.TestCase):
    """Only explicit baking scenarios dry-run and offer count-bounded baking candidates."""

    def choices(self, data, maximum=32):
        """Record checks and let the fake engine choose the largest legal batch, without inferring ingredients."""
        checked = []

        def feasible(agent, call):
            """Reject baking counts over the supplied capacity and accept unrelated acts."""
            checked.append(call)
            return DryRun(call.name != "bake" or call.args["count"] <= maximum,
                          40 * call.args.get("count", 1))

        choices = CandidateGenerator(data, feasible).generate("alice", observation())
        return [choice for choice in choices if choice.call.name == "bake"], checked

    def test_absent_and_null_baking_do_not_even_dry_run_the_new_tool(self):
        """Carry limits or freshness alone do not implicitly enable baking."""
        for data in (manifest(), manifest(bake_ticks=None), manifest(carry_limits={"minecraft:bread": 32}),
                     manifest(fresh_days=2)):
            with self.subTest(config=data.data):
                choices, checked = self.choices(data)
                self.assertEqual(choices, [])
                self.assertNotIn("bake", {call.name for call in checked})

    def test_explicit_baking_offers_one_and_the_engine_maximum_up_to_32(self):
        """Wheat, oven access and capacity come from dry runs; no count outside 1–32 is requested."""
        for maximum, expected in ((0, []), (1, [1]), (4, [1, 4]), (64, [1, 32])):
            with self.subTest(maximum=maximum):
                choices, checked = self.choices(manifest(tools=["bake"]), maximum)
                self.assertEqual([choice.call.args["count"] for choice in choices], expected)
                self.assertTrue(all(1 <= call.args["count"] <= 32 for call in checked if call.name == "bake"))
                for choice in choices:
                    count = choice.call.args["count"]
                    self.assertIn(f"using {3 * count} wheat", choice.label)
                    self.assertIn(f"about {40 * count} ticks", choice.label)

    def test_baking_match_ignores_batch_count_like_other_counted_tools(self):
        """The delegate's Match relation recognizes two batch sizes as the same count-free choice."""
        self.assertTrue(same_choice(ToolCall("bake", {"count": 1}), ToolCall("bake", {"count": 32})))
        self.assertFalse(same_choice(ToolCall("bake", {"count": 1}), ToolCall("craft", {"item": "bread"})))

    def test_every_legacy_rank_including_eat_stays_exactly_where_it_was(self):
        """Baking is appended after the original table and eat, without moving ferry or any other old tool."""
        legacy = json.loads((GOLDEN / "tools.json").read_text(encoding="utf-8"))
        for index, tool in enumerate(legacy):
            with self.subTest(tool=tool["name"]):
                self.assertEqual(_TOOL_ORDER[tool["name"]], index)
        self.assertEqual(_TOOL_ORDER["eat"], len(legacy))
        self.assertEqual(_TOOL_ORDER["bake"], len(legacy) + 1)


class GoodsNarrationTest(unittest.TestCase):
    """Each enabled goods mechanic adds one plain sentence while legacy GM instructions stay exact."""

    def test_disabled_goods_leave_the_original_prompt_byte_identical(self):
        """Empty collections and null freshness or baking append no rules, even for an old owned shop world."""
        actions = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in TOOL_DEFINITIONS)
        expected = f"{GM_RULES}\n\nActs people can attempt:\n{actions}\n\n{ENGINE_RULES}"
        self.assertEqual(gm_system_prompt(TOOL_DEFINITIONS), expected)
        self.assertEqual(gm_system_prompt(TOOL_DEFINITIONS, carry_limits={}, fresh_days=None, bake_ticks=None,
                                          villager_purchases=[]), expected)
        gm = GameMaster(ScriptedClient(lambda system, user: None), "An owned market stall.", {"alice": "Here."})
        self.assertEqual(gm._system, expected)

    def test_each_opt_in_adds_one_mechanic_sentence_and_enables_owned_stalls(self):
        """Any active goods mechanic enables owner trading, without adding another configuration key."""
        base = gm_system_prompt(TOOL_DEFINITIONS)
        settings = [({"carry_limits": {"minecraft:wheat": 96, "minecraft:bread": 32}}, "32 bread, 96 wheat"),
                    ({"fresh_days": 0}, "greater than 0"), ({"bake_ticks": 80}, "80 ticks per loaf"),
                    ({"villager_purchases": [{"stall": [1, 64, 0], "item": "minecraft:bread", "count": 2}]},
                     "freshest non-mouldy stock")]
        for config, words in settings:
            with self.subTest(config=config):
                prompt = gm_system_prompt(TOOL_DEFINITIONS, **config)
                self.assertTrue(prompt.startswith(base + "\n"))
                sentences = prompt[len(base) + 1:].splitlines()
                self.assertEqual(len(sentences), 2)
                self.assertTrue(all(sentence.startswith("- ") and sentence.count(".") == 1 for sentence in sentences))
                self.assertIn(words, prompt)
                self.assertIn("A stall's owner may stock its chest and take its takings", prompt)

    def test_gm_keeps_goods_rules_when_daily_hunger_reconfigures_tools(self):
        """The narrated runtime's existing eat configuration cannot erase preconfigured goods mechanics."""
        gm = GameMaster(ScriptedClient(lambda system, user: None), "A bakery.", {"alice": "Beside an oven."},
                        carry_limits={"minecraft:bread": 32}, fresh_days=2, bake_ticks=40,
                        villager_purchases=[{"stall": [1, 64, 0], "item": "minecraft:bread", "count": 2}])
        gm.configure_tools(daily_hunger=True, extras=("bake",))
        self.assertEqual(gm._system, gm_system_prompt(tool_definitions(daily_hunger=True, extras=("bake",)),
                         carry_limits={"minecraft:bread": 32}, fresh_days=2, bake_ticks=40,
                         villager_purchases=[{"stall": [1, 64, 0], "item": "minecraft:bread", "count": 2}]))
        for words in ("- eat:", "- bake:", "greater than 2", "40 ticks per loaf", "sold out", "partial amount"):
            self.assertIn(words, gm._system)


if __name__ == "__main__":
    unittest.main()
