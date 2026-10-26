"""Jubilee briefs, the scenario-gated `sleep` tool and its world action (no server)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent
from orchestrator.jubilee import JubileeManifest, clock_of
from orchestrator.llm import ToolCall
from orchestrator.runtime import carried, perceived, sleep_as_wait
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, tool_definitions

ROOT = Path(__file__).resolve().parents[2]
CONTENT = json.loads((ROOT / "scenario" / "jubilee.json").read_text(encoding="utf-8"))
SURPRISE_WORDS = ("jubilee", "feast", "surprise", "committee", "mouldy batch")


def town_manifest(**condition) -> dict:
    """A minimal town manifest with the content file's roles, containers and stalls at made-up positions."""
    places = [{"name": name, "kind": kind, "bank": bank, "pos": pos} for name, kind, bank, pos in (
        ("plaza", "plaza", "west", [-30, -61, 0]), ("town_hall", "town_hall", "west", [-50, -61, 0]),
        ("bakery", "bakery", "east", [40, -61, 10]), ("west_home_1", "house", "west", [-80, -61, 30]))]
    containers = {}
    for index, (name, container) in enumerate({**CONTENT["containers"], **CONTENT["stalls"]}.items()):
        containers[name] = {**container, "pos": [index, -60, 100]}
    content = {**CONTENT, "condition": condition} if condition else CONTENT
    return {"variant": "jubilee", "places": places, "roles": CONTENT["roles"], "containers": containers,
            "layout": {"hall": {"feast_tables": [f"feast_table_{n}" for n in range(1, 5)]},
                       "ferry": {"landings": {"west": {"board": [-7, -61, 60]}, "east": {"board": [7, -61, 60]}}}},
            "jubilee": content, "tools": ["sleep"]}


class SleepToolTest(unittest.TestCase):
    """`sleep` exists only where a scenario enables it, and becomes a wait until the next 06:00."""

    def test_default_tools_are_unchanged_and_sleep_is_opt_in(self):
        self.assertIs(tool_definitions(), TOOL_DEFINITIONS)
        self.assertEqual([tool["name"] for tool in tool_definitions(extras=("sleep",))][-1], "sleep")
        self.assertEqual([tool["name"] for tool in tool_definitions(daily_hunger=True, extras=("sleep",))][-2:],
                         ["eat", "sleep"])
        with self.assertRaises(ValueError):
            tool_definitions(extras=("fly",))

    def test_sleep_waits_until_six_and_other_calls_pass_through(self):
        self.assertEqual(sleep_as_wait(ToolCall("sleep", {}), 16_040), ToolCall("wait", {"ticks": 7_960}))
        self.assertEqual(sleep_as_wait(ToolCall("sleep", {"note": "tired"}), 47_000).args, {"ticks": 1_000})
        call = ToolCall("wait", {"ticks": 40})
        self.assertIs(sleep_as_wait(call, 100), call)
        self.assertIsNone(sleep_as_wait(None, 100))


class PerceptionLogTest(unittest.TestCase):
    """The perception record keeps who was seen and what was perceived, and drops rays, terrain and empties."""

    def test_record_is_compact(self):
        observation = {"self": {"pos": [0, -60, 0]}, "terrain": [{"x": 1}], "blocks": [{"pos": [1, 2, 3]}],
                       "entities": [{"name": "freight", "type": "player", "pos": [3, -60, 4], "losRay": 2,
                                     "load": {"item": "minecraft:bread", "count_band": "sack"}, "heldItem": None}],
                       "heard": [{"speaker": "baker", "text": "Morning!"}], "chatter": [], "changes": []}
        self.assertEqual(perceived(4_000, "mayor", observation), {
            "t": 4_000, "agent": "mayor",
            "entities": [{"name": "freight", "type": "player", "pos": [3, -60, 4],
                          "load": {"item": "minecraft:bread", "count_band": "sack"}}],
            "heard": [{"speaker": "baker", "text": "Morning!"}]})
        self.assertEqual(perceived(40, "baker", {"entities": []}), {"t": 40, "agent": "baker"})

    def test_carried_goods(self):
        inventory = [{"item": "minecraft:bread", "count": 20, "baked_day": 0}, {"item": "minecraft:bread", "count": 5},
                     {"item": "minecraft:emerald", "count": 9}, {"item": "minecraft:wheat", "count": 3}]
        self.assertEqual(carried({"self": {"inventory": inventory}}), {"bread": 25, "wheat": 3})
        self.assertEqual(carried({"self": {"inventory": []}}), {})


class JubileeBriefTest(unittest.TestCase):
    """Only committee members learn of the plan; everyone shares the town's facts; tools survive a fork."""

    def setUp(self):
        self.manifest = JubileeManifest(town_manifest())
        self.briefs = {brief.bot: brief for brief in self.manifest.briefs()}

    def test_load_dispatches_to_the_jubilee_class(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "town.json"
            path.write_text(json.dumps(town_manifest()), encoding="utf-8")
            self.assertIsInstance(ScenarioManifest.load(path), JubileeManifest)

    def test_committee_members_get_the_goal_and_each_other(self):
        for member in CONTENT["committee"]["members"]:
            private = self.briefs[member].private
            self.assertIn(CONTENT["committee"]["goal"], private)
            self.assertIn("You are on the jubilee committee with the ", private)
            self.assertNotIn("COMMITTEE", private)
        self.assertIn("freight operator", self.briefs["innkeeper"].private)
        self.assertIn("has gone mouldy", self.briefs["innkeeper"].private)
        self.assertNotIn("has gone mouldy", self.briefs["builder"].private)

    def test_nobody_else_and_no_shared_text_mentions_the_plan(self):
        outsiders = set(self.briefs) - set(CONTENT["committee"]["members"])
        self.assertEqual(outsiders, {"mayor", "secretary", "baker", "neighbour"})
        for name in outsiders:
            for word in SURPRISE_WORDS:
                self.assertNotIn(word, self.briefs[name].private.lower(), (name, word))
        for word in SURPRISE_WORDS:
            self.assertNotIn(word, self.manifest.common_knowledge().lower(), word)

    def test_untold_members_keep_the_goal_without_the_committee(self):
        briefs = {brief.bot: brief for brief in JubileeManifest(town_manifest(told=False)).briefs()}
        self.assertIn(CONTENT["committee"]["goal"], briefs["freight"].private)
        self.assertNotIn("committee with", briefs["freight"].private)
        for member in CONTENT["committee"]["members"]:
            self.assertNotIn("committee", briefs[member].private.lower(), member)

    def test_ordinary_day_and_watched_knobs(self):
        ordinary = {brief.bot: brief for brief in JubileeManifest(town_manifest(plan=False)).briefs()}
        for name, brief in ordinary.items():
            for word in SURPRISE_WORDS:
                self.assertNotIn(word, brief.private.lower(), (name, word))
        self.assertIn(CONTENT["committee"]["ordinary_goals"]["builder"], ordinary["builder"].private)
        watched = {brief.bot: brief for brief in JubileeManifest(town_manifest(committee_extra="Note X.")).briefs()}
        self.assertTrue(all(watched[name].private.endswith("Note X.") or "Note X." in watched[name].private
                            for name in CONTENT["committee"]["members"]))
        self.assertNotIn("Note X.", watched["mayor"].private)

    def test_physics_off_drops_the_rules_from_knowledge_and_the_engine(self):
        self.assertEqual(self.manifest.engine_overrides, {})
        off = JubileeManifest(town_manifest(physics=False))
        self.assertEqual(off.engine_overrides, {"night_sight": False, "carry_limits": {}, "visible_loads": {}})
        for fact in ("After dark", "at most 32 loaves", "sack of bread"):
            self.assertNotIn(fact, off.common_knowledge())
        self.assertIn("mouldy", off.common_knowledge())

    def test_access_and_ownership_lines(self):
        self.assertIn("You can open the appointments chest", self.briefs["secretary"].private)
        self.assertIn("(so can: mayor)", self.briefs["secretary"].private)
        self.assertIn("The bakery stall at", self.briefs["baker"].private)
        self.assertIn("is yours", self.briefs["baker"].private)

    def test_shared_knowledge_states_the_rules_as_facts(self):
        common = self.manifest.common_knowledge()
        for fact in ("tick 0 of day 0 is 06:00", "After dark (19:00 to 05:00)", "at most 32 loaves",
                     "mouldy", "`bake` makes one loaf from 3 wheat", "`sleep`", "four long tables"):
            self.assertIn(fact, common)

    def test_agents_carry_their_tools_through_a_fork(self):
        agent = self.manifest.agents()[0]
        self.assertEqual(agent.tools[-1]["name"], "sleep")
        self.assertEqual(Agent.from_json(agent.to_json()).extras, ("sleep",))

    def test_clock(self):
        self.assertEqual(clock_of(0), "day 0, 06:00")
        self.assertEqual(clock_of(36_000), "day 1, 18:00")
        self.assertEqual(clock_of(18_500), "day 0, 00:30")


if __name__ == "__main__":
    unittest.main()
