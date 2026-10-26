"""Offline contracts for whispering, block use, and factual engine action notices."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.candidates import Candidate, CandidateGenerator, DryRun, EngineDryRun, same_choice
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import ShadowEngine, gm_system_prompt
from orchestrator.perception import DecisionContext, EvidenceExposure, PerceptionFeatures, VillagerTracks
from orchestrator.runtime import DecisionOverride, SimulationRuntime, failure_hint
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, ToolTranslator
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld
from scripts.tests.test_candidates import RecordingClient, RecordingFeasibility, calls_for, observation, player

ROOT = Path(__file__).resolve().parents[2]


class InteractionToolsTest(unittest.TestCase):
    """New model-facing calls retain the engine's skill, timing, and rejection contracts."""

    def test_whisper_schema_description_and_translation(self):
        """Whisper is optional like talk, describes limited hearing, and passes through without a note."""
        definition = next(tool for tool in TOOL_DEFINITIONS if tool["name"] == "speak")
        self.assertEqual(definition["parameters"]["properties"]["volume"]["enum"], ["whisper", "talk", "shout"])
        self.assertEqual(definition["parameters"]["required"], ["text"])
        for phrase in ("3 blocks", "same rate", "no shout cooldown", "within 8 blocks who can see you",
                       "without hearing the words", "about 3 words per second"):
            self.assertIn(phrase, definition["description"])
        action = ToolTranslator().translate("envoy", ToolCall("speak", {
            "text": "Quietly now.", "volume": "whisper", "to": "builder", "note": "private"}))
        self.assertEqual(action.command, "speak")
        self.assertEqual(action.args, {"name": "envoy", "text": "Quietly now.", "volume": "whisper", "to": "builder"})
        self.assertTrue(action.is_skill)
        default = ToolTranslator().translate("envoy", ToolCall("speak", {"text": "Hello."}))
        self.assertNotIn("volume", default.args)

    def test_use_schema_and_translation(self):
        """Using a block requires all three integer coordinates and starts a timed skill without centering them."""
        definition = next(tool for tool in TOOL_DEFINITIONS if tool["name"] == "use")
        parameters = definition["parameters"]
        self.assertEqual(parameters["required"], ["x", "y", "z"])
        self.assertEqual(set(parameters["properties"]), {"x", "y", "z", "note"})
        self.assertFalse(parameters["additionalProperties"])
        for axis in ("x", "y", "z"):
            self.assertEqual(parameters["properties"][axis], {"type": "integer"})
        action = ToolTranslator().translate("envoy", ToolCall("use", {"x": -6, "y": -60, "z": 0, "note": "open"}))
        self.assertEqual((action.command, action.is_skill), ("skill_start", True))
        self.assertEqual(action.args, {"name": "envoy", "skill": "use", "args": {"x": -6, "y": -60, "z": 0}})
        for missing in ("x", "y", "z"):
            with self.subTest(missing=missing), self.assertRaises(KeyError):
                ToolTranslator().translate("envoy", ToolCall("use", {
                    axis: value for axis, value in action.args["args"].items() if axis != missing}))

    def test_use_dry_run_shadow_and_rejections(self):
        """Candidates and the narrated shadow send the same skill and preserve engine rejection codes."""
        call = ToolCall("use", {"x": -6, "y": -60, "z": 0, "note": "open"})
        args = {"name": "envoy", "skill": "use", "args": {"x": -6, "y": -60, "z": 0}}
        client = RecordingClient({"feasible": True, "accepted": True, "estimate": {"ticks": 5}})
        self.assertEqual(EngineDryRun(client, CommandError)("envoy", call), DryRun(True, 5))
        shadow = ShadowEngine(client, CommandError)
        self.assertEqual(shadow.check("envoy", call), {"feasible": True, "reason": None, "ticks": 5})
        self.assertEqual(shadow.execute("envoy", call), "started")
        self.assertEqual(client.requests, [("skill_start", {**args, "dry_run": True})] * 2 + [("skill_start", args)])
        for reason in ("not_usable", "out_of_reach"):
            with self.subTest(reason=reason):
                client = RecordingClient({"feasible": False, "accepted": False, "reason": reason})
                self.assertEqual(EngineDryRun(client, CommandError)("envoy", call), DryRun(False, reason=reason))
                shadow = ShadowEngine(client, CommandError)
                self.assertEqual(shadow.check("envoy", call), {"feasible": False, "reason": reason, "ticks": None})
                self.assertEqual(shadow.execute("envoy", call), f"not possible: {reason}")

    def test_gm_rules_cover_whisper_use_and_unlocked_ownership(self):
        """The narrator knows audibility, visible chatter, use timing, bell hearing, and ownership versus locks."""
        prompt = gm_system_prompt(TOOL_DEFINITIONS)
        for phrase in ("Whispering carries 3 blocks", "same speaking rate as talk", "no shout cooldown",
                       "Beyond 3 but within 8 blocks", "without sight they perceive nothing",
                       "- use: Use the block", "5 ticks", "use_ticks", "not_usable", "out_of_reach",
                       "within 48 blocks", "bell rang at (x, y, z)", "allowed is null", "not spoken words"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, prompt)


class InteractionCandidatesTest(unittest.TestCase):
    """Only observed, reachable, engine-feasible interactions become deterministic delegate choices."""

    def setUp(self):
        """Load village facts and record all candidate feasibility checks."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.checker = RecordingFeasibility(ticks=5)
        self.generator = CandidateGenerator(self.manifest, self.checker)

    def test_use_candidates_cover_usable_blocks_and_exclude_other_targets(self):
        """Doors, trapdoors, gates, bells, levers and buttons qualify; unseen, far or unrelated blocks do not."""
        kinds = ("oak_door", "copper_trapdoor", "spruce_fence_gate", "bell", "lever", "stone_button")
        blocks = [{"pos": [-56, -60, 62 + index], "block": f"minecraft:{kind}", "distance": 2}
                  for index, kind in enumerate(kinds)]
        blocks += [{"pos": [-56, -60, 70 + index], "block": f"minecraft:{kind}", "distance": 2}
                   for index, kind in enumerate(("red_bed", "stone", "chest", "water"))]
        blocks.append({"pos": [-60, -60, 62], "block": "minecraft:bell", "distance": 4.5001})
        view = observation(blocks=blocks)
        before = deepcopy(view)
        choices = [candidate for candidate in self.generator.generate("investigator", view)
                   if candidate.call.name == "use"]
        self.assertEqual({tuple(choice.call.args.values()) for choice in choices},
                         {(-56, -60, 62 + index) for index in range(len(kinds))})
        for choice in choices:
            self.assertIn(("investigator", choice.call), self.checker.calls)
            self.assertEqual(choice.target, tuple(choice.call.args.values()))
            self.assertIn("about 5 ticks", choice.label)
        self.assertEqual(view, before)
        self.assertFalse(calls_for(self.generator.generate("investigator", observation()), "use"))
        self.assertEqual(choices, [candidate for candidate in self.generator.generate(
            "investigator", observation(blocks=list(reversed(blocks)))) if candidate.call.name == "use"])

    def test_use_candidates_include_reach_boundary_and_skip_rejected_targets(self):
        """Observed eye distance gates use, and engine rejection removes even a nearby usable-looking block."""
        boundary = {"pos": [-56, -60, 62], "block": "minecraft:bell", "distance": 4.5}
        rejected = ToolCall("use", {"x": -55, "y": -60, "z": 62})
        self.checker.rejected_calls.append(rejected)
        view = observation(blocks=[boundary, {"pos": [-55, -60, 62], "block": "minecraft:oak_door", "distance": 1}])
        self.assertEqual(calls_for(self.generator.generate("investigator", view), "use"),
                         [ToolCall("use", {"x": -56, "y": -60, "z": 62})])
        self.checker.rejected_tools.add("use")
        self.assertFalse(calls_for(self.generator.generate("investigator", view), "use"))

    def test_use_matches_only_the_exact_block_position(self):
        """Block use ignores notes, not coordinates, and never matches a different action."""
        call = ToolCall("use", {"x": -56, "y": -60, "z": 62})
        self.assertTrue(same_choice(call, ToolCall("use", {**call.args, "note": "close"})))
        for axis in ("x", "y", "z"):
            self.assertFalse(same_choice(call, ToolCall("use", {**call.args, axis: call.args[axis] + 1})))
        self.assertFalse(same_choice(call, ToolCall("mine", call.args)))

    def test_whisper_choices_require_nearby_addressees_and_survive_shout_rejection(self):
        """Whispers include the three-block boundary and broadcast, without replacing existing talk choices."""
        self.checker.rejected_calls.append(ToolCall("speak", {"volume": "shout", "text": "Hello."}))
        view = observation(entities=[player("mayor", (-53, -60, 62), 3),
                                     player("builder", (-52.999, -60, 62), 3.001)])
        choices = calls_for(self.generator.generate("investigator", view), "speak")
        self.assertIn(ToolCall("speak", {"to": "mayor", "volume": "whisper"}), choices)
        self.assertNotIn(ToolCall("speak", {"to": "builder", "volume": "whisper"}), choices)
        self.assertIn(ToolCall("speak", {"volume": "whisper"}), choices)
        self.assertIn(ToolCall("speak", {"to": "mayor", "volume": "talk"}), choices)
        self.assertNotIn(ToolCall("speak", {"volume": "shout"}), choices)

    def test_owned_unlocked_container_remains_accessible_to_non_owner(self):
        """Ownership is narrated, not treated as an implicit lock or a grant overriding an explicit allowed list."""
        container = self.manifest.data["containers"]["records"]
        container.update(owner="builder", allowed=None)
        position = container["pos"]
        memory = AgentMemory()
        memory.record_read(0, "container at (-56, -60, 62): 2 emerald")
        view = observation(blocks=[{"pos": position, "block": "minecraft:chest", "distance": 2}])
        candidates = self.generator.generate("investigator", view, memory)
        self.assertTrue(calls_for(candidates, "open_container"))
        self.assertTrue(calls_for(candidates, "chest_take"))
        self.assertIn("; owner builder; opened by anyone", self.manifest.gm_world())
        container["allowed"] = ["builder"]
        self.assertFalse(calls_for(self.generator.generate("investigator", view, memory), "chest_take"))


class InteractionNoticesTest(unittest.TestCase):
    """Engine action notices stay factual, preserve provenance, and never masquerade as speech."""

    def test_notices_are_remembered_rendered_and_restored_without_spoken_words(self):
        """Each sense is rendered explicitly, with no speech quotes or speech-triggered interruption."""
        for kind, sense, text in (("block_use", "sight", "envoy opened oak_door at (1, 64, 0)"),
                                  ("block_use", "hearing", "bell rang at (1, 64, 0)"),
                                  ("transfer", "sight", "envoy took 2 emerald from builder's container")):
            with self.subTest(kind=kind, sense=sense):
                item = {"gameTime": 20, "kind": kind, "sense": sense, "speaker": "bell" if sense == "hearing" else
                        "envoy", "text": text, "prov": "event:7", "pos": [1, 64, 0]}
                memory = AgentMemory()
                self.assertFalse(memory.absorb({"heard": [item]}, "builder"))
                data = memory.to_json()
                self.assertEqual(data["heard"], [])
                self.assertEqual(data["noticed"], [item])
                rendered = f"{'saw' if sense == 'sight' else 'heard'} {text}"
                self.assertIn(rendered, memory.render())
                self.assertNotIn(f'"{text}"', memory.render())
                self.assertEqual(AgentMemory(data).render(), memory.render())
                self.assertIn(f"noticed: {rendered}", memory.take_fresh()[0])

    def test_only_actual_speech_interrupts_and_chatter_does_not_reveal_words(self):
        """Mixing notices with speech retains speech behavior; visual whisper chatter supplies no quoted words."""
        memory = AgentMemory()
        speech = {"gameTime": 20, "speaker": "envoy", "text": "Hello.", "distance": 2}
        notice = {**speech, "kind": "block_use", "sense": "sight", "text": "envoy flipped a lever"}
        self.assertTrue(memory.absorb({"heard": [notice, speech]}, "builder"))
        self.assertEqual(len(memory.to_json()["heard"]), 1)
        self.assertIn('envoy (2 blocks away): "Hello."', memory.render())
        chatter = {"gameTime": 30, "speaker": "mayor", "reason": "out_of_earshot", "text": "secret"}
        self.assertFalse(memory.absorb({"chatter": [chatter]}, "builder"))
        self.assertIn("talking; you could not make out the words", memory.render())
        self.assertNotIn("secret", memory.render())

    def test_muting_speech_does_not_hide_action_notices(self):
        """The speech intervention leaves block use and transfers perceivable even when the actor is muted."""
        speech = {"speaker": "envoy", "text": "Hello."}
        notices = [{**speech, "kind": kind} for kind in ("block_use", "transfer")]
        view = {"heard": [speech, *notices], "chatter": [speech], "shout_seen": [speech]}
        before = deepcopy(view)
        override = DecisionOverride(game_time=20, mute=frozenset({("builder", "envoy")}))
        perceived = override.perceived("builder", view, 20)
        self.assertEqual(perceived, {"heard": notices, "chatter": [], "shout_seen": []})
        self.assertEqual(view, before)


class InteractionRuntimeTest(unittest.TestCase):
    """Engine rejection reaches the decision log and memory with the appropriate recovery hint."""

    def test_use_rejections_are_logged_with_hints_and_thinking_delay(self):
        """Unsupported blocks and distant targets remain rejections rather than successful narrated interactions."""
        call = ToolCall("use", {"x": 1, "y": 64, "z": 0})
        for reason, hint in (("not_usable", "use works on doors"), ("out_of_reach", "stand within about 4 blocks")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                world = FakeWorld({"alice": [0, 64, 0]}, rejects={"use": reason})
                agent = Agent(RoleBrief("alice", "worker", "Works.", "Help."))
                runtime = SimulationRuntime(world, [agent], ScriptedClient(lambda system, user: call),
                                            Path(directory), command_error=CommandError, min_gap=100)
                runtime.run(160)
                decisions = runtime.log.decisions()
                self.assertEqual([decision.game_time for decision in decisions], [0, 120])
                for decision in decisions:
                    self.assertTrue(decision.outcome.startswith(f"not possible: {reason}"))
                    self.assertIn(hint, decision.outcome)
                self.assertEqual(agent.memory.to_json()["actions"][-1]["outcome"], decisions[-1].outcome)
                self.assertEqual(failure_hint(call, "started", {}), "")


class InteractionPerceptionTest(unittest.TestCase):
    """Whisper hearing and positional block-use relevance feed the existing perception features."""

    def setUp(self):
        """Create a real manifest with empty evidence and logged villager tracks."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.exposure = EvidenceExposure(self.manifest)
        self.features = PerceptionFeatures(self.manifest)

    def test_whisper_earshot_is_three_blocks_inclusively_not_talk_range(self):
        """A visible but out-of-earshot agent does not become a hearer or receive the whispered evidence."""
        observations = {"envoy": observation((0, 64, 0)), "builder": observation((3, 64, 0)),
                        "mayor": observation((3.001, 64, 0), entities=[{"type": "minecraft:player", "name": "envoy"}])}
        ledger = self.manifest.data["evidence"]["ledger_entry"]
        self.exposure.read("mayor", ledger, 0)
        context = DecisionContext("envoy", 20, observations, self.exposure, VillagerTracks([]))
        whisper = Candidate(ToolCall("speak", {"volume": "whisper", "text": "Hello."}), "whisper", 0, (0, 64, 0))
        result = self.features.features(whisper, context)
        self.assertEqual(result["earshot"], 1.0)
        self.assertEqual(result["seen_by_agents"], 1.0)
        self.assertEqual(result["perceivers_informed"], 0.0)

    def test_use_goal_relevance_uses_the_target_position(self):
        """A watched block-use target participates in spatial predicate relevance just like other block actions."""
        position = (-4, -61, 0)
        candidate = Candidate(ToolCall("use", dict(zip(("x", "y", "z"), position))), "use", 0, position)
        observations = {"envoy": observation(), "builder": observation(entities=[
            {"type": "minecraft:player", "name": "envoy"}])}
        context = DecisionContext("envoy", 20, observations, self.exposure, VillagerTracks([]))
        self.assertEqual(self.features.features(candidate, context)["goal_relevance"], 1.0)
        observations["builder"]["entities"] = []
        self.assertEqual(self.features.features(candidate, context)["goal_relevance"], 0.0)

    def test_action_notices_are_not_villager_testimony(self):
        """Factual action text mentioning the incident cannot be mistaken for a villager's spoken testimony."""
        speaker = self.manifest.data["npcs"][0]["name"]
        self.exposure.observe("investigator", {"gameTime": 20, "heard": [{
            "gameTime": 20, "speaker": speaker, "text": "envoy used the bridge gate", "kind": "block_use"}]})
        self.assertEqual(self.exposure.items("investigator", 20), frozenset())


if __name__ == "__main__":
    unittest.main()
