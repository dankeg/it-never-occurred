"""Unit tests for the orchestrator's pure logic: tool translation, agent memory, rendering and the decision log."""

from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.decisions import Decision, DecisionLog, prompt_digest
from orchestrator.llm import AnthropicClient, OpenAICompatibleClient, ScriptedClient, ToolCall, Turn
from orchestrator.narration import GameMaster, NarratedRuntime, ShadowEngine, gm_system_prompt, narration_error
from orchestrator.render import ObservationRenderer, clock
from orchestrator.scenario import PredicateEvaluator, ScenarioManifest
from orchestrator.runtime import failure_hint
from orchestrator.tools import TOOL_DEFINITIONS, ToolTranslator

OBSERVATION = {
    "gameTime": 2140,
    "self": {"name": "ben", "pos": [18.4, -60.0, 12.6], "health": 20.0, "food": 20,
             "skill": {"skillId": None, "skill": None, "state": "idle"},
             "inventory": [{"slot": 1, "item": "minecraft:bread", "count": 2}]},
    "entities": [{"uuid": "u1", "type": "minecraft:player", "name": "cy", "pos": [25.5, -60, 6.5], "distance": 8.9}],
    "blocks": [{"pos": [18, -60, 10], "block": "minecraft:chest", "distance": 2.6}],
    "documents": [{"kind": "sign", "pos": [20, -60, 14], "distance": 2.9,
                   "text": {"front": ["Market opens", "at noon", "", ""], "back": ["", "", "", ""]}}],
    "heard": [{"speaker": "cy", "text": "hello", "gameTime": 2130, "distance": 8.2}],
    "changes": [],
}


class ToolTranslatorTest(unittest.TestCase):
    """Tool calls map onto the harness commands that enforce them."""

    def test_drop_definition_requires_an_item_and_positive_count(self):
        """Dropping exposes only the carried item, a positive count, and the shared optional note."""
        definition = next(tool for tool in TOOL_DEFINITIONS if tool["name"] == "drop")
        parameters = definition["parameters"]
        self.assertEqual(parameters["required"], ["item", "count"])
        self.assertEqual(parameters["properties"]["item"], {"type": "string"})
        self.assertEqual(parameters["properties"]["count"], {"type": "integer", "minimum": 1})
        self.assertEqual(set(parameters["properties"]), {"item", "count", "note"})
        self.assertFalse(parameters["additionalProperties"])
        self.assertIn("Anyone can pick them up", definition["description"])
        self.assertIn("five minutes", definition["description"])

    def test_drop_translation_normalizes_items_and_strips_notes(self):
        """Drop starts a timed skill with explicit item/count rather than relying on the selected slot."""
        translator = ToolTranslator()
        action = translator.translate("ben", ToolCall("drop", {"item": "Oak Planks", "count": 3, "note": "discard"}))
        self.assertEqual(action.command, "skill_start")
        self.assertEqual(action.args, {"name": "ben", "skill": "drop",
                                       "args": {"item": "minecraft:oak_planks", "count": 3}})
        self.assertTrue(action.is_skill)
        for args in ({}, {"item": "bread"}, {"count": 1}):
            with self.subTest(args=args), self.assertRaises(KeyError):
                translator.translate("ben", ToolCall("drop", args))
        smuggled = translator.translate("ben", ToolCall("drop", {"item": "bread", "count": 1,
                                                               "unlimited_lifetime": True}))
        self.assertEqual(smuggled.args["args"], {"item": "minecraft:bread", "count": 1})  # staging-only option dropped

    def test_skills_and_information_actions(self):
        translator = ToolTranslator()
        walk = translator.translate("ben", ToolCall("move_to", {"x": 1, "y": -60, "z": 2, "note": "go"}))
        self.assertEqual((walk.command, walk.args["args"], walk.is_skill),
                         ("skill_start", {"x": 1.5, "y": -60, "z": 2.5}, True))
        take = translator.translate("ben", ToolCall("chest_take", {"x": 1, "y": 2, "z": 3, "item": "Oak Planks",
                                                                   "count": 2}))
        self.assertEqual(take.args["args"], {"pos": [1, 2, 3], "item": "minecraft:oak_planks", "count": 2})
        sign = translator.translate("ben", ToolCall("write_sign", {"x": 1, "y": 2, "z": 3, "lines": ["hi"]}))
        self.assertEqual((sign.command, sign.args["front"], sign.is_skill), ("write_sign", ["hi", "", "", ""], False))
        self.assertIsNone(translator.translate("ben", ToolCall("keep_going", {})))
        with self.assertRaises(ValueError):
            translator.translate("ben", ToolCall("fly", {}))
        with self.assertRaises(KeyError):
            translator.translate("ben", ToolCall("mine", {"x": 1}))


class AgentMemoryTest(unittest.TestCase):
    """Memory is bounded, round-trips through JSON, and resolves finished skills."""

    def test_bounds_serialization_and_resolution(self):
        memory = AgentMemory()
        for tick in range(30):
            memory.record_action(tick, ToolCall("wait", {"ticks": 1, "note": f"n{tick}"}), "started",
                                 skill_id=f"ben:{tick}")
        self.assertEqual(len(memory.to_json()["actions"]), AgentMemory.LIMITS["actions"])
        self.assertEqual(memory.to_json()["notes"][-1]["text"], "n29")
        memory.resolve_skill({"skillId": "ben:29", "state": "failed", "reason": "no_path"})
        self.assertEqual(memory.to_json()["actions"][-1]["outcome"], "failed (no_path)")
        self.assertEqual(AgentMemory(memory.to_json()).render(), memory.render())

    def test_absorb_flags_speech_from_others(self):
        memory = AgentMemory()
        self.assertTrue(memory.absorb(OBSERVATION, "ben"))
        self.assertFalse(memory.absorb(OBSERVATION, "cy"))

    def test_absorb_skips_state_only_changes(self):
        memory = AgentMemory()
        wall = {"gameTime": 24000, "pos": [-11, -60, 20]}
        memory.absorb({"changes": [{**wall, "from": "minecraft:air", "to": "minecraft:cobblestone_wall"},
                                   {**wall, "from": "minecraft:cobblestone_wall",
                                    "to": "minecraft:cobblestone_wall"}]}, "ben")
        self.assertEqual([(change["from"], change["to"]) for change in memory.to_json()["changes"]],
                         [("minecraft:air", "minecraft:cobblestone_wall")])


class PromptTest(unittest.TestCase):
    """Prompts are deterministic and show what the agent perceives."""

    def test_water_flag_adds_an_explicit_self_observation(self):
        """Being in water adds one sentence without changing the rest of the observation."""
        renderer = ObservationRenderer()
        wet = {**OBSERVATION, "self": {**OBSERVATION["self"], "in_water": True}}
        text = renderer.render(wet)
        self.assertIn("\nYou are in water.\n", text)
        self.assertEqual(text.replace("\nYou are in water.", "", 1), renderer.render(OBSERVATION))

    def test_false_or_absent_water_flag_keeps_the_existing_prompt(self):
        """Dry bots and recordings predating the water flag have identical prompt text."""
        renderer = ObservationRenderer()
        dry = {**OBSERVATION, "self": {**OBSERVATION["self"], "in_water": False}}
        self.assertEqual(renderer.render(dry), renderer.render(OBSERVATION))
        self.assertNotIn("You are in water.", renderer.render(dry))

    def test_prompt_contents_and_digest(self):
        agent = Agent(RoleBrief("ben", "shopkeeper", "Runs the shop.", "Keep the shop open."))
        agent.memory.absorb(OBSERVATION, "ben")
        user = agent.user_prompt(OBSERVATION, interrupted=False)
        for text in ('cy (8 blocks away): "hello"', "You are ben at (18, -60, 13)", "slot 1: 2 bread",
                     "chest: (18, -60, 10)", 'sign at (20, -60, 14): "Market opens / at noon"'):
            self.assertIn(text, user)
        self.assertEqual(clock(2000), "day 0, 08:00")
        digest = prompt_digest("m", agent.system_prompt(), user, TOOL_DEFINITIONS)
        self.assertEqual(digest, prompt_digest("m", agent.system_prompt(), user, TOOL_DEFINITIONS))
        self.assertNotEqual(digest, prompt_digest("m2", agent.system_prompt(), user, TOOL_DEFINITIONS))


class StructureViewTest(unittest.TestCase):
    """Known structures in sight are shown with whether they are complete; without any, prompts are unchanged."""

    def test_render_and_manifest(self):
        renderer = ObservationRenderer()
        plain = renderer.render(OBSERVATION)
        self.assertNotIn("Structures", plain)
        seen = {**OBSERVATION, "structures": [
            {"name": "bridge_span_3", "pos": [0, -61, 0], "built": 6, "total": 6, "distance": 9.0},
            {"name": "port_repair", "pos": [4, -60, 40], "built": 2, "total": 9, "distance": 12.0}]}
        text = renderer.render(seen)
        self.assertIn("Structures you can see:\n  bridge_span_3 at (0, -61, 0): complete (6 of 6 blocks)\n"
                      "  port_repair at (4, -60, 40): unfinished (2 of 9 blocks in place)", text)
        self.assertLess(text.index("Structures"), text.index("Notable blocks"))
        manifest = ScenarioManifest({"schematics": {"span": {"origin": [1, 2, 3], "blocks": [
            {"dx": 1, "dy": 0, "dz": -1, "block": "minecraft:oak_planks"}]}}})
        self.assertEqual(manifest.structures(), [{"name": "span", "anchor": [1, 2, 3], "blocks": [
            {"pos": [2, 2, 2], "block": "minecraft:oak_planks"}]}])


class ConversationTest(unittest.TestCase):
    """Append-only prompts: full memory at an epoch's start, then only updates, with earlier turns as history."""

    @staticmethod
    def _agent(epoch_turns: int) -> Agent:
        return Agent(RoleBrief("ben", "shopkeeper", "Runs the shop.", "Keep the shop open."), epoch_turns=epoch_turns)

    def test_single_turn_epochs_match_full_prompts(self):
        agent, twin = self._agent(1), self._agent(1)
        for _ in range(3):
            for each in (agent, twin):
                each.memory.absorb(OBSERVATION, "ben")
            user = agent.next_prompt(OBSERVATION, interrupted=False)
            self.assertEqual(user, twin.user_prompt(OBSERVATION, interrupted=False))
            self.assertEqual(agent.history(), ())
            agent.finish_turn(user, ToolCall("wait", {"ticks": 20}), "started")

    def test_epochs_append_updates_and_restart(self):
        agent = self._agent(3)
        agent.memory.absorb(OBSERVATION, "ben")
        first = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertIn("Your notes:", first)
        agent.memory.record_action(2140, ToolCall("speak", {"text": "hi"}), "started", skill_id="ben:1")
        agent.finish_turn(first, ToolCall("speak", {"text": "hi", "note": "greeted cy"}), "started")
        agent.memory.resolve_skill({"skillId": "ben:1", "state": "succeeded", "skill": "speak",
                                    "progress": {"heardBy": ["cy"]}})
        agent.memory.absorb({**OBSERVATION, "heard": [{"speaker": "cy", "text": "bye", "gameTime": 2150,
                                                       "distance": 3.0}]}, "ben")
        second = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertTrue(second.startswith("Since your last action:"))
        self.assertNotIn("Your notes:", second)
        self.assertIn('your speak {"text": "hi"} -> succeeded (heard by: cy)', second)
        self.assertIn('heard cy (3 blocks away): "bye"', second)
        self.assertEqual(agent.history(), (Turn(first, ToolCall("speak", {"text": "hi", "note": "greeted cy"}),
                                                "started"),))
        agent.finish_turn(second, None, "no action chosen")
        restored = Agent.from_json(agent.to_json())
        third = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertIn("(nothing new)", third)
        self.assertEqual(restored.next_prompt(OBSERVATION, interrupted=False), third)
        self.assertEqual(restored.history(), agent.history())
        agent.finish_turn(third, ToolCall("wait", {"ticks": 20}), "started")
        self.assertIn("Your notes:", agent.next_prompt(OBSERVATION, interrupted=False))
        self.assertEqual(agent.history(), ())

    def test_inventory_changes_become_updates(self):
        memory = AgentMemory()
        memory.absorb(OBSERVATION, "ben")
        self.assertEqual(memory.take_fresh(), ['  [day 0, 08:07] heard cy (8 blocks away): "hello"'])
        richer = {**OBSERVATION, "gameTime": 2200, "heard": [],
                  "self": {**OBSERVATION["self"], "inventory": [{"slot": 1, "item": "minecraft:bread", "count": 1},
                                                                {"slot": 2, "item": "minecraft:oak_log", "count": 3}]}}
        restored = AgentMemory(memory.to_json())
        restored.absorb(richer, "ben")
        self.assertEqual(restored.take_fresh(), ["  [day 0, 08:12] your inventory: -1 bread, +3 oak_log"])
        restored.absorb(richer, "ben")
        self.assertEqual(restored.take_fresh(), [])

    def test_repeated_failure_starts_a_new_epoch(self):
        agent = self._agent(8)
        call = ToolCall("speak", {"text": "word " * 120})
        for tick in range(3):
            user = agent.next_prompt(OBSERVATION, interrupted=False)
            agent.memory.record_action(tick, call, "failed: Speech needs nonempty text of at most 600 characters")
            agent.finish_turn(user, call, "failed")
        restart = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertIn("Your notes:", restart)
        self.assertIn("Your last three attempts at speak failed the same way", restart)
        self.assertEqual(agent.history(), ())
        self.assertIn("yours was 120 words", failure_hint(call, "failed: Speech needs nonempty text of at most "
                                                             "600 characters", OBSERVATION))

    def test_updates_are_bounded(self):
        memory = AgentMemory()
        for tick in range(AgentMemory.FRESH_LIMIT + 5):
            memory.record_read(tick, f"page {tick}")
        lines = memory.take_fresh()
        self.assertEqual(lines[0], "  (5 earlier updates not shown)")
        self.assertEqual(len(lines), AgentMemory.FRESH_LIMIT + 1)
        self.assertEqual(memory.take_fresh(), [])

    def test_history_in_digest_and_requests(self):
        history = (Turn("first", ToolCall("move_to", {"z": 1, "x": 2}), "started"), Turn("second", None, "no action"))
        base = prompt_digest("m", "sys", "third", TOOL_DEFINITIONS)
        self.assertEqual(base, prompt_digest("m", "sys", "third", TOOL_DEFINITIONS, ()))
        self.assertNotEqual(base, prompt_digest("m", "sys", "third", TOOL_DEFINITIONS, history))
        messages = OpenAICompatibleClient._history_messages(history)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user", "assistant", "user"])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], '{"x": 2, "z": 1}')
        anthropic = AnthropicClient._messages(history, "third")
        self.assertEqual([m["role"] for m in anthropic], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(anthropic[2]["content"][0], {"type": "tool_result", "tool_use_id": "call_0",
                                                      "content": "started"})
        self.assertEqual(anthropic[-1]["content"][-1], {"type": "text", "text": "third"})


class FailureHintTest(unittest.TestCase):
    """Failure hints explain common failures from what the agent could see."""

    def test_no_bank_states_the_swim_limit(self):
        """Any walking skill can report a missing bank, including during feasibility checks."""
        wet = {**OBSERVATION, "self": {**OBSERVATION["self"], "in_water": True}}
        for tool in ("move_to", "mine", "collect"):
            for outcome in ("not possible: in_water_no_bank", "failed (in_water_no_bank)"):
                with self.subTest(tool=tool, outcome=outcome):
                    hint = failure_hint(ToolCall(tool, {}), outcome, wet)
                    for text in ("in water", "bank", "16 blocks"):
                        self.assertIn(text, hint)

    def test_water_observation_alone_does_not_add_a_failure_hint(self):
        """Successful actions and absent tool calls do not gain misleading recovery advice."""
        wet = {**OBSERVATION, "self": {**OBSERVATION["self"], "in_water": True}}
        call = ToolCall("move_to", {"x": 1, "y": -60, "z": 2})
        for outcome in ("started", "done"):
            with self.subTest(outcome=outcome):
                self.assertEqual(failure_hint(call, outcome, wet), "")
        self.assertEqual(failure_hint(None, "failed (in_water_no_bank)", wet), "")

    def test_dry_movement_failure_hint_is_unchanged(self):
        """Water recovery guidance does not replace the existing no-path explanation on land."""
        call = ToolCall("move_to", {"x": 1, "y": -60, "z": 2})
        for self_view in (OBSERVATION["self"], {**OBSERVATION["self"], "in_water": False}):
            with self.subTest(self_view=self_view):
                self.assertEqual(failure_hint(call, "not possible: no_path", {**OBSERVATION, "self": self_view}),
                                 " (no walkable route: it may be across water, enclosed, or not yet built)")

    def test_drop_missing_items_explains_inventory_and_count(self):
        """A missing drop stack points to the actor's inventory, without hinting after success."""
        call = ToolCall("drop", {"item": "bread", "count": 3})
        self.assertIn("check the item and count", failure_hint(call, "not possible: missing_items", OBSERVATION))
        self.assertEqual(failure_hint(call, "started", OBSERVATION), "")

    def test_ask_out_of_reach(self):
        seen = {**OBSERVATION, "entities": [{"type": "minecraft:villager", "npcId": "npc:adult_14", "pos": [0, 0, 0]}]}
        far = failure_hint(ToolCall("ask", {"npc": "adult_14"}), "not possible: out_of_reach", seen)
        gone = failure_hint(ToolCall("ask", {"npc": "adult_14"}), "not possible: out_of_reach", OBSERVATION)
        self.assertEqual(far, " (stand within about 4 blocks of it first)")
        self.assertTrue(gone.startswith(" (adult_14 is not in sight"))
        self.assertIn("still saying", failure_hint(ToolCall("speak", {"text": "hi"}), "not possible: busy",
                                                 OBSERVATION))


    def test_ferry_hint_says_which_way_the_request_failed(self):
        """Own bank, a boat in view elsewhere, or no boat in view: the three no_boat cases are told apart."""
        east = {**OBSERVATION, "self": {**OBSERVATION["self"], "pos": [4.5, -59.0, 41.5]}}
        boat = {"type": "minecraft:oak_boat", "pos": [-2.5, -60.5, 41.5], "distance": 7}
        hint = lambda to, observation: failure_hint(ToolCall("ferry", {"to": to}), "not possible: no_boat",  # noqa: E731
                                                    observation)
        self.assertIn("already on the east bank", hint("east", east))
        self.assertIn("moored at the west landing", hint("west", {**east, "entities": [boat]}))
        self.assertEqual(hint("west", {**east, "entities": []}), " (the boat is not moored at this landing)")


class ContractBookPredicateTest(unittest.TestCase):
    """The builder's paperwork metric reads a signed contract revision from the records chest."""

    class Records:
        """A harness client whose records chest holds the given items."""

        def __init__(self, items):
            self.items = items

        def request(self, command, args):
            return {"items": self.items}

    QUERY = {"container": "records", "slot": 0, "title": "Bridge contract", "author": "mayor", "json_page": 1,
             "contract_id": "village_bridge_contract", "revision_gte": 1, "payment_received": True}

    def judged(self, items):
        manifest = ScenarioManifest({"containers": {"records": {"pos": [0, 0, 0]}}, "roles": []})
        return PredicateEvaluator(self.Records(items), Path("unused"), manifest).evaluate({"book_query": self.QUERY})

    def test_signed_revision(self):
        page = '{"contract_id": "village_bridge_contract", "revision": 1, "payment_received": true}'
        book = {"slot": 0, "item": "minecraft:written_book", "title": "Bridge contract", "author": "mayor",
                "pages": ["Paid in full.", page]}
        self.assertTrue(self.judged([book]))
        self.assertFalse(self.judged([{**book, "pages": ["Paid.", page.replace("true", "false")]}]))
        self.assertFalse(self.judged([{**book, "author": "builder"}]))
        self.assertFalse(self.judged([{**book, "pages": ["Paid.", "not json"]}]))
        self.assertFalse(self.judged([]))


class DecisionLogTest(unittest.TestCase):
    """The decision log persists, reloads and rejects duplicates."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            decision = Decision(100, "ben", "d", False, "model", "wait", {"ticks": 5}, "m", 1, 2, "prompt")
            DecisionLog(path).append(decision)
            reloaded = DecisionLog(path)
            self.assertEqual(reloaded.get("ben", 100), decision)
            self.assertEqual(reloaded.get("ben", 100).tool_call, ToolCall("wait", {"ticks": 5}))
            with self.assertRaises(ValueError):
                reloaded.append(decision)


if __name__ == "__main__":
    unittest.main()


class KeywordJudgeTest(unittest.TestCase):
    """The baseline report judge reads cause, culprit and cited evidence from keywords."""

    def test_sabotage_and_accident_reports(self):
        from orchestrator.scenario import KeywordJudge
        judge = KeywordJudge(["mayor", "builder", "freight", "security", "envoy", "investigator"])
        sabotage = judge.judge("Report: the collapse was sabotage. The envoy removed the supports; the market "
                               "ledger shows the envoy bought a pickaxe, found by the bridge.")
        self.assertEqual((sabotage.cause, sabotage.culprit), ("sabotage", "envoy"))
        self.assertEqual(set(sabotage.evidence), {"west_supports", "envoy_pickaxe", "market_ledger"})
        accident = judge.judge("Report: an accident. The supports were rotten (mossy cobblestone). Nobody is to blame.")
        self.assertEqual((accident.cause, accident.culprit), ("accident", None))


class NarratedCadenceTest(unittest.TestCase):
    """Narrated agents decide on the engine runtime's cadence: thinking time after instant acts, rest at night."""

    @staticmethod
    def _acts(start: int, until: int) -> list[tuple[int, str]]:
        actions = iter([ToolCall("open_container", {"x": 1, "y": 2, "z": 3}),
                        ToolCall("move_to", {"x": 5, "y": -60, "z": 5})] * 4)
        durations = {"open_container": 10, "move_to": 50}

        def resolve(system: str, user: str) -> ToolCall:
            tool = user.split(" attempts: ", 1)[1].split(" ", 1)[0]
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": durations[tool],
                                        "effects": [], "actor_view": "a room", "observations": []})

        agent = Agent(RoleBrief("ben", "shopkeeper", "Runs the shop.", "Keep the shop open."))
        gm = GameMaster(ScriptedClient(resolve), "a village", {"ben": "a room"})
        with tempfile.TemporaryDirectory() as directory:
            runtime = NarratedRuntime([agent], ScriptedClient(lambda system, user: next(actions)), gm,
                                      Path(directory), min_gap=100, rest=(16_000, 24_000))
            runtime.run(start, until)
        return [(act["gameTime"], act["tool"]) for act in runtime.acts]

    def test_thinking_time_follows_instant_acts_only(self):
        self.assertEqual(self._acts(0, 400), [(0, "open_container"), (120, "move_to"), (200, "open_container"),
                                              (320, "move_to")])

    def test_idle_agents_rest_at_night(self):
        self.assertEqual(self._acts(16_000, 16_400), [])


class NarrationScoringTest(unittest.TestCase):
    """GM rulings are scored against the engine: first and downstream errors, false rejections, and durations."""

    def test_first_and_downstream_errors_and_false_rejections(self):
        def act(time, agent, tool, accepted, engine=None, final="", check=None, duration=0):
            record = {"gameTime": time, "agent": agent, "tool": tool, "args": {},
                      "gm": {"accepted": accepted, "duration_ticks": duration}, "engine_check": check}
            if engine is not None:
                record.update({"engine": engine, "engine_final": final})
            return record

        acts = [act(0, "ann", "move_to", True, "started", "running", {"feasible": True, "ticks": 100}, 50),
                act(0, "bo", "ask", True, "not possible: target_not_found"),
                act(40, "ann", "chest_take", True, "not possible: out_of_reach"),
                act(80, "bo", "move_to", False, check={"feasible": True, "ticks": 60}),
                act(80, "cy", "speak", False, check={"feasible": True, "ticks": 20})]
        summary = narration_error(acts)
        self.assertEqual((summary["infeasible"], summary["infeasible_first"], summary["infeasible_downstream"]),
                         (2, 1, 1))
        self.assertEqual(summary["first_divergence"], {"ann": 0, "bo": 0})
        self.assertEqual((summary["rejected"], summary["false_rejections"], summary["false_rejections_first"]),
                         (2, 2, 1))
        self.assertEqual(summary["duration_ratio_median"], 0.5)

    def test_shadow_dry_runs_only_acts_that_run_over_time(self):
        class Client:
            def __init__(self):
                self.requests = []

            def request(self, command, args):
                self.requests.append((command, args))
                return {"feasible": False, "reason": "no_path", "estimate": {"ticks": 0}}

        client = Client()
        shadow = ShadowEngine(client, RuntimeError)
        self.assertEqual(shadow.check("ann", ToolCall("move_to", {"x": 1, "y": -60, "z": 2})),
                         {"feasible": False, "reason": "no_path", "ticks": 0})
        self.assertTrue(client.requests[0][1]["dry_run"])
        self.assertIsNone(shadow.check("ann", ToolCall("open_container", {"x": 1, "y": 2, "z": 3})))

    def test_gm_knows_the_actions_and_the_engine_rules(self):
        prompt = gm_system_prompt(TOOL_DEFINITIONS)
        self.assertIn("- ask: Ask a villager", prompt)
        self.assertIn("cannot be asked", prompt)
        self.assertIn("- drop: Drop items from your inventory", prompt)
        self.assertIn("Dropping items takes 10 ticks", prompt)
        self.assertIn("6,000 ticks", prompt)

    def test_shadow_checks_and_executes_drop_as_a_timed_skill(self):
        """The narrated shadow uses the same normalized drop command for feasibility and execution."""
        class Client:
            """Accept and record both forms of the drop skill request."""

            def __init__(self):
                """Start without requests."""
                self.requests = []

            def request(self, command, args):
                """Record a request and return the matching dry-run or start response."""
                self.requests.append((command, args))
                return {"feasible": True, "estimate": {"ticks": 10}} if args.get("dry_run") else {"accepted": True}

        client = Client()
        shadow = ShadowEngine(client, RuntimeError)
        call = ToolCall("drop", {"item": "Oak Planks", "count": 2, "note": "leave here"})
        self.assertEqual(shadow.check("ben", call), {"feasible": True, "reason": None, "ticks": 10})
        self.assertEqual(shadow.execute("ben", call), "started")
        args = {"name": "ben", "skill": "drop", "args": {"item": "minecraft:oak_planks", "count": 2}}
        self.assertEqual(client.requests, [("skill_start", {**args, "dry_run": True}), ("skill_start", args)])


if __name__ == "__main__":
    unittest.main()
