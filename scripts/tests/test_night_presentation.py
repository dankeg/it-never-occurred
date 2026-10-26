"""Offline presentation tests for opt-in night sight, visible loads and delivered villager news."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import ENGINE_RULES, GM_RULES, GameMaster, NarratedRuntime, gm_system_prompt
from orchestrator.render import ObservationRenderer, clock
from orchestrator.tools import TOOL_DEFINITIONS, tool_definitions


class VisibleLoadRenderingTest(unittest.TestCase):
    """Only an admitted public sack band changes the visible entity's description."""

    def setUp(self):
        """Use one visible person without a load as the legacy baseline."""
        self.renderer = ObservationRenderer()
        self.entity = {"type": "minecraft:player", "name": "Rowan", "pos": [12, 64, 3], "distance": 7}
        self.legacy = "Rowan at (12, 64, 3), 7 blocks away"

    def test_sack_of_bread_has_no_exact_count(self):
        """Render only the public band, even if private fields are supplied alongside it."""
        self.entity["load"] = {"item": "minecraft:bread", "count_band": "sack", "count": 37}
        self.entity["inventory"] = [{"item": "minecraft:diamond", "count": 53}]
        original = deepcopy(self.entity)
        self.assertEqual(self.renderer.entity_line(self.entity), self.legacy + ", carrying a sack of bread")
        self.assertEqual(self.entity, original)

    def test_load_coexists_with_held_item_and_public_activity(self):
        """Holding a lantern and walking do not hide a separate inventory load."""
        self.entity.update(load={"item": "minecraft:bread", "count_band": "sack"},
                           heldItem="minecraft:lantern", activity={"skill": "move_to", "pos": [20, 64, 3]})
        self.assertEqual(self.renderer.entity_line(self.entity), self.legacy +
                         ", holding lantern, carrying a sack of bread, walking towards (20, 64, 3)")

    def test_item_names_are_readable_without_restricting_to_bread(self):
        """A configured item with a multiword registry name receives the same sack phrasing."""
        self.entity["load"] = {"item": "minecraft:sweet_berries", "count_band": "sack"}
        self.assertEqual(self.renderer.entity_line(self.entity), self.legacy + ", carrying a sack of sweet berries")

    def test_missing_empty_or_unsupported_loads_leave_legacy_output_identical(self):
        """Never infer a sack from a private count, an empty load, or an unknown band."""
        self.assertEqual(self.renderer.entity_line(self.entity), self.legacy)
        for load in (None, {}, {"item": "minecraft:bread", "count": 9},
                     {"item": "minecraft:bread", "count_band": "9"}, {"count_band": "sack"}):
            with self.subTest(load=load):
                self.entity["load"] = load
                self.assertEqual(self.renderer.entity_line(self.entity), self.legacy)

    def test_full_observation_uses_the_public_load_description(self):
        """The normal observation path, not just the entity helper, exposes the sack."""
        self.entity["load"] = {"item": "minecraft:bread", "count_band": "sack"}
        observation = {"gameTime": 14000, "self": {"name": "Ben", "pos": [5, 64, 3], "health": 20,
                                                 "food": 20, "inventory": []}, "entities": [self.entity]}
        self.assertIn("  " + self.legacy + ", carrying a sack of bread\n", self.renderer.render(observation))


class NightNewsPresentationTest(unittest.TestCase):
    """Java formats timed ask answers; Python must retain the delivered speech, not reword its news metadata."""

    @staticmethod
    def reports():
        """Use the Java schema: person subjects for sightings, place subjects for stall and building news."""
        return [({"kind": "load", "subject": "Rowan", "item": "minecraft:bread", "count_band": "sack"},
                 "Rowan was carrying a sack of bread at the dock."),
                ({"kind": "night", "subject": "Rowan"}, "Rowan was outside at night near the dock."),
                ({"kind": "sold_out", "subject": "bakery stall", "item": "minecraft:bread"},
                 "The bakery stall had sold out of bread."),
                ({"kind": "decoration", "subject": "town hall"}, "Lanterns were being placed in the town hall.")]

    @staticmethod
    def heard(report, answer):
        """Deliver an engine-shaped utterance with detached relayed news and its first report's common fields."""
        news = {"newsId": "sight:7", "originEventId": "npc_observation:7", "originatedAt": 13000,
                "learnedAt": 13020, "firsthand": False, "chain": ["Ada", "Bramble"], "salience": 0.5,
                "pos": [12, 64, 3], "place": "dock", **report}
        return {"gameTime": 13040, "speaker": "Bramble", "distance": 3.2, "text": answer, "news": [news],
                **{key: news[key] for key in ("newsId", "kind", "subject", "firsthand", "chain", "originEventId")}}

    def test_each_new_news_kind_keeps_the_natural_ask_answer(self):
        """Optional summary metadata never replaces the complete words actually heard from the villager."""
        for report, answer in self.reports():
            for has_summary in (False, True):
                with self.subTest(kind=report["kind"], has_summary=has_summary):
                    heard = self.heard({**report, **({"text": answer} if has_summary else {})}, answer)
                    original = deepcopy(heard)
                    memory = AgentMemory()
                    self.assertTrue(memory.absorb({"heard": [heard]}, "Ben"))
                    expected = f'  [{clock(13040)}] Bramble (3 blocks away): "{answer}"'
                    self.assertIn(expected, memory.render())
                    self.assertEqual(memory.take_fresh(), [expected.replace("] Bramble", "] heard Bramble")])
                    self.assertNotIn("npc_observation", memory.render())
                    self.assertNotIn("minecraft:", memory.render())
                    self.assertEqual(AgentMemory(memory.to_json()).render(), memory.render())
                    self.assertEqual(heard, original)

    def test_mixed_new_and_legacy_news_preserves_the_entire_utterance(self):
        """News summaries may contain semicolons; Python must not split or reconstruct a timed utterance."""
        report, summary = self.reports()[0]
        legacy = "bridge_or_port_change at [4,64,8]. Rowan was there."
        answer = f"{legacy}; {summary} There were lights; the dock was busy."
        heard = self.heard(report, answer)
        heard["news"].insert(0, {"kind": "bridge_or_port_change", "subject": [4, 64, 8], "present": ["Rowan"]})
        memory = AgentMemory()
        memory.absorb({"heard": [heard]}, "Ben")
        self.assertEqual(memory.to_json()["heard"][0]["text"], answer)
        self.assertIn(f'Bramble (3 blocks away): "{answer}"', memory.render())

    def test_legacy_answer_is_unchanged(self):
        """Default-off speech and memory updates retain their exact original wording."""
        answer = "bridge_or_port_change at [4,64,8]. Rowan was there."
        heard = self.heard({"kind": "bridge_or_port_change", "subject": [4, 64, 8]}, answer)
        memory = AgentMemory()
        memory.absorb({"heard": [heard]}, "Ben")
        self.assertEqual(memory.to_json()["heard"], [{"gameTime": 13040, "speaker": "Bramble",
                                                     "text": answer, "distance": 3}])
        self.assertEqual(memory.take_fresh(), [f'  [{clock(13040)}] heard Bramble (3 blocks away): "{answer}"'])

    def test_unheard_news_never_leaks_from_chatter_metadata(self):
        """Seeing a villager talk is not permission to read its structured news or summary."""
        report, answer = self.reports()[0]
        memory = AgentMemory()
        self.assertFalse(memory.absorb({"chatter": [self.heard({**report, "text": answer}, answer)]}, "Ben"))
        self.assertIn("talking; you could not make out the words", memory.render())
        self.assertNotIn("Rowan", memory.render())
        self.assertNotIn("sack", memory.render())


class NightNarrationRulesTest(unittest.TestCase):
    """GM sight and load instructions are explicit opt-ins independent of tool configuration."""

    def test_default_and_explicitly_disabled_prompts_are_byte_identical_to_legacy(self):
        """No unconfigured mechanic may change the old prompt or its cache digest."""
        actions = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in TOOL_DEFINITIONS)
        legacy = f"{GM_RULES}\n\nActs people can attempt:\n{actions}\n\n{ENGINE_RULES}"
        self.assertEqual(gm_system_prompt(TOOL_DEFINITIONS), legacy)
        self.assertEqual(gm_system_prompt(TOOL_DEFINITIONS, night_sight=False, visible_loads={}), legacy)
        gm = GameMaster(ScriptedClient(lambda system, user: None), "A town.", {"Ben": "At home."})
        gm.configure_tools()
        self.assertEqual(gm._system, legacy)

    def test_night_rule_has_the_shared_sight_boundaries_and_unchanged_hearing(self):
        """Two plain sentences cover both observers, both light sources, near sight and occlusion."""
        prompt = gm_system_prompt(TOOL_DEFINITIONS, night_sight=True)
        rule = prompt.split(ENGINE_RULES + "\n- ", 1)[1]
        self.assertEqual(rule.count("."), 2)
        for phrase in ("19:00 to 05:00", "people and villagers", "beyond 8 blocks", "within 5 blocks",
                       "light-emitting block", "main hand", "within 8 blocks", "as by day", "line of sight",
                       "whispering", "hearing is unchanged"):
            self.assertIn(phrase, rule)
        self.assertNotIn("sack", prompt)

    def test_load_rule_uses_only_configured_thresholds_and_no_exact_observed_counts(self):
        """The GM learns the thresholds without being told to expose private counts or every carried item."""
        loads = {"minecraft:wheat": 12, "minecraft:bread": 9}
        prompt = gm_system_prompt(TOOL_DEFINITIONS, visible_loads=loads)
        rule = prompt.split(ENGINE_RULES + "\n- ", 1)[1]
        self.assertEqual(rule.count("."), 2)
        self.assertIn("at least 9 bread or 12 wheat", rule)
        self.assertIn("visible person's inventory totals", rule)
        self.assertIn("people and villagers", rule)
        self.assertIn("carrying a sack", rule)
        self.assertIn("never exact inventory counts or unconfigured items", rule)
        self.assertNotIn("At night", prompt)
        self.assertEqual(prompt, gm_system_prompt(TOOL_DEFINITIONS, visible_loads=dict(reversed(list(loads.items())))))

    def test_configured_rules_reach_the_model_and_survive_tool_reconfiguration(self):
        """Changing daily-hunger tools neither disables selected mechanics nor follows later caller mutations."""
        prompts = []

        def policy(system, user):
            """Record the actual model instructions and accept one idle act."""
            prompts.append(system)
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 40})

        loads = {"minecraft:bread": 9}
        gm = GameMaster(ScriptedClient(policy), "A town.", {"Ben": "At home."},
                        night_sight=True, visible_loads=loads)
        loads["minecraft:bread"] = 99
        agent = Agent(RoleBrief("Ben", "baker", "Bakes bread.", "Help the town."), daily_hunger=True)
        with tempfile.TemporaryDirectory() as directory:
            NarratedRuntime([agent], ScriptedClient(lambda system, user: None), gm, Path(directory))
            for daily_hunger in (True, False):
                gm.configure_tools(daily_hunger=daily_hunger)
                gm.resolve(13000, "Ben", ToolCall("wait", {"ticks": 40}))
                self.assertEqual(prompts[-1], gm_system_prompt(tool_definitions(daily_hunger=daily_hunger),
                                                              night_sight=True, visible_loads={"minecraft:bread": 9}))


if __name__ == "__main__":
    unittest.main()
