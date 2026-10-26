"""Pure unit tests for public entity rendering and bounded witnessed-container memory."""

import copy
import json
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.render import ObservationRenderer


OBSERVATION = {
    "gameTime": 0,
    "self": {"name": "ben", "pos": [0, -60, 0], "health": 20, "food": 20, "inventory": []},
    "entities": [{"uuid": "builder-id", "type": "minecraft:player", "name": "builder",
                  "pos": [-60, -60, 75], "distance": 12.4}],
}
ENTITY_LINE = "  builder at (-60, -60, 75), 12 blocks away"
LEGACY_VIEW = (
    "Time: day 0, 06:00 (tick 0)\n\n"
    "You are ben at (0, -60, 0). Health 20/20, food 20/20.\nInventory: empty\n\n"
    "People and things you can see:\n" + ENTITY_LINE + "\n\n"
    "Notable blocks you can see:\n  (none)\n\nWriting you can read:\n  (none)"
)
LEGACY_MEMORY_TEXT = (
    "Your notes:\n  (none)\n\nYour recent actions:\n  (none)\n\nWhat you heard:\n  (none)\n\n"
    "Changes you saw:\n  (none)\n\nThings you read:\n  (none)"
)
LEGACY_MEMORY = {"notes": [], "actions": [], "heard": [], "changes": [], "reads": [], "noticed": [],
                 "fresh": [], "freshDropped": 0, "inventory": None}
TAKE = {"gameTime": 0, "actor": "builder", "action": "chest_take", "pos": [-8, -61, 0],
        "item": "minecraft:iron_pickaxe"}
PUT = {"gameTime": 20, "actor": "farmer", "action": "chest_put", "pos": [-9, -61, 0],
       "item": "minecraft:bread"}
TAKE_LINE = "  [day 0, 06:00] saw builder take iron_pickaxe from the chest at (-8, -61, 0)"
PUT_LINE = "  [day 0, 06:01] saw farmer put bread into the chest at (-9, -61, 0)"


class PerceptionRenderingTest(unittest.TestCase):
    """Only held items, public skill names and visible target positions extend entity lines."""

    def render_entity(self, **fields) -> str:
        """Render a fresh observation with the supplied visible entity fields."""
        observation = copy.deepcopy(OBSERVATION)
        observation["entities"][0].update(fields)
        return ObservationRenderer().render(observation)

    def assert_entity_suffix(self, suffix: str, **fields) -> None:
        """Pin the whole prompt, not just the newly added words."""
        expected = LEGACY_VIEW.replace(ENTITY_LINE, ENTITY_LINE + suffix)
        self.assertEqual(self.render_entity(**fields).encode("utf-8"), expected.encode("utf-8"))

    def test_absent_fields_preserve_legacy_bytes(self):
        """Older observations produce exactly the original compact view."""
        self.assertEqual(self.render_entity().encode("utf-8"), LEGACY_VIEW.encode("utf-8"))

    def test_null_and_empty_fields_preserve_legacy_bytes(self):
        """Idle, empty-handed entities add no commas, placeholders or sections."""
        for fields in ({"heldItem": None}, {"activity": None}, {"heldItem": None, "activity": None},
                       {"activity": {}}, {"activity": {"skill": None}}, {"heldItem": "", "activity": {}}):
            with self.subTest(fields=fields):
                self.assert_entity_suffix("", **fields)

    def test_held_items_shorten_only_the_minecraft_namespace(self):
        """The registry item remains recognizable without leaking inventory quantities."""
        for item, label in (("minecraft:iron_pickaxe", "iron_pickaxe"), ("mod:hammer", "mod:hammer"),
                            ("bread", "bread")):
            for activity in (None, {"skill": None}):
                with self.subTest(item=item, activity=activity):
                    self.assert_entity_suffix(f", holding {label}", heldItem=item, activity=activity)

    def test_held_item_and_visible_mining_target_match_the_spec(self):
        """Holding and mining extend the existing line in that order, with whole-block rounding."""
        self.assert_entity_suffix(
            ", holding iron_pickaxe, mining the block at (-8, -61, 0)",
            heldItem="minecraft:iron_pickaxe", activity={"skill": "mine", "pos": [-8.2, -60.9, 0.1]})

    def test_visible_targets_use_natural_descriptions(self):
        """A public position is rendered even when it is the origin, without consulting arguments."""
        descriptions = {
            "move_to": "walking towards", "mine": "mining the block at", "place": "placing a block at",
            "build_section": "building at", "place_section": "building at",
            "chest_take": "taking items from the chest at", "chest_put": "putting items into the chest at",
            "buy": "buying items at", "ferry": "riding a ferry towards", "future_skill": "doing future_skill at",
        }
        for skill, description in descriptions.items():
            with self.subTest(skill=skill):
                self.assert_entity_suffix(f", {description} (0, 0, 0)",
                                          activity={"skill": skill, "pos": [0, 0, 0]})

    def test_running_skill_names_without_targets_never_reveal_arguments(self):
        """All running names work with absent or null public targets, including future names."""
        descriptions = {
            "move_to": "walking", "mine": "mining", "collect": "collecting dropped items", "craft": "crafting",
            "place": "placing a block", "build_section": "building", "place_section": "building",
            "chest_take": "taking items from a chest", "chest_put": "putting items into a chest",
            "buy": "buying items", "give": "handing over items", "drop": "dropping items", "hold": "holding an item",
            "wait": "waiting", "ferry": "riding a ferry", "speak": "speaking", "ask": "asking a villager",
            "future_skill": "doing future_skill",
        }
        for skill, description in descriptions.items():
            for target in ({}, {"pos": None}):
                with self.subTest(skill=skill, target=target):
                    activity = {"skill": skill, **target, "args": {
                        "pos": [991, 992, 993], "item": "secret_item", "count": 917, "note": "secret_note"},
                        "progress": {"pos": [994, 995, 996]}, "target": "secret_target"}
                    self.assert_entity_suffix(f", {description}", activity=activity)

    def test_extra_activity_fields_do_not_leak_or_mutate_the_observation(self):
        """Only the explicit public position matters, even if richer engine metadata is present."""
        observation = copy.deepcopy(OBSERVATION)
        observation["entities"][0].update(heldItem="minecraft:iron_pickaxe", activity={
            "skill": "chest_take", "pos": [-8, -61, 0], "item": "secret_item", "count": 917,
            "args": {"pos": [991, 992, 993], "note": "secret_note"}, "prov": {"id": "secret_receipt"}})
        original = copy.deepcopy(observation)
        expected = LEGACY_VIEW.replace(ENTITY_LINE, ENTITY_LINE +
                                       ", holding iron_pickaxe, taking items from the chest at (-8, -61, 0)")
        self.assertEqual(ObservationRenderer().render(observation), expected)
        self.assertEqual(observation, original)

    def test_missing_skill_does_not_expose_a_target_or_extra_fields(self):
        """An incomplete activity is not a running skill and cannot grant a target by itself."""
        for skill in ({}, {"skill": None}, {"skill": ""}):
            with self.subTest(skill=skill):
                self.assert_entity_suffix("", activity={**skill, "pos": [991, 992, 993], "note": "secret"})

    def test_entity_order_and_bounds_are_unchanged(self):
        """Entity distance and UUID, not new fields, still determine the bounded visible list."""
        observation = copy.deepcopy(OBSERVATION)
        template = observation["entities"][0]
        observation["entities"] = [
            {**template, "uuid": "z", "name": "hidden", "distance": 20, "heldItem": "minecraft:diamond"},
            {**template, "uuid": "b", "name": "second", "distance": 5, "activity": {"skill": "craft"}},
            {**template, "uuid": "a", "name": "first", "distance": 5, "heldItem": "minecraft:bread"},
        ]
        rendered = ObservationRenderer(max_entities=2).render(observation)
        self.assertIn("  first at (-60, -60, 75), 5 blocks away, holding bread\n"
                      "  second at (-60, -60, 75), 5 blocks away, crafting", rendered)
        self.assertNotIn("hidden", rendered)
        self.assertNotIn("diamond", rendered)
        self.assertEqual(rendered, ObservationRenderer(max_entities=2).render(observation))


class WitnessedMemoryTest(unittest.TestCase):
    """Seen transfers follow the existing memory and fresh-update lifecycle, without private data."""

    def test_absent_null_and_empty_events_preserve_legacy_memory_bytes(self):
        """Old snapshots restore without a new empty section or serialized key."""
        for observation in ({}, {"seen_actions": None}, {"seen_actions": []}):
            with self.subTest(observation=observation):
                memory = AgentMemory(copy.deepcopy(LEGACY_MEMORY))
                self.assertFalse(memory.absorb(observation, "ben"))
                self.assertEqual(memory.render().encode("utf-8"), LEGACY_MEMORY_TEXT.encode("utf-8"))
                self.assertEqual(json.dumps(memory.to_json()).encode("utf-8"),
                                 json.dumps(LEGACY_MEMORY).encode("utf-8"))
                self.assertEqual(memory.take_fresh(), [])

    def test_take_and_put_are_stored_and_rendered_in_receipt_order(self):
        """Full memory and fresh updates use the same count-free, timestamped descriptions."""
        memory = AgentMemory()
        self.assertFalse(memory.absorb({"seen_actions": [copy.deepcopy(TAKE), copy.deepcopy(PUT)]}, "ben"))
        self.assertEqual(memory.to_json()["seen_actions"], [TAKE, PUT])
        expected = LEGACY_MEMORY_TEXT.replace("Things you read:",
                                             f"Actions you saw:\n{TAKE_LINE}\n{PUT_LINE}\n\nThings you read:")
        self.assertEqual(memory.render(), expected)
        self.assertEqual(memory.take_fresh(), [TAKE_LINE, PUT_LINE])
        self.assertEqual(memory.take_fresh(), [])
        self.assertEqual(memory.to_json()["actions"], [])

    def test_optional_provenance_and_private_extras_do_not_leak(self):
        """Store only witnessed facts; optional provenance, quantities and notes never enter prompts."""
        for provenance in (None, "transfer:17", {"id": "secret_receipt"}):
            with self.subTest(provenance=provenance):
                memory = AgentMemory()
                events = [{**TAKE, "prov": provenance, "count": 917, "note": "secret_note"},
                          {**PUT, "args": {"item": "secret_item", "count": 918}, "prov": None}]
                original = copy.deepcopy(events)
                memory.absorb({"seen_actions": events}, "ben")
                self.assertEqual(events, original)
                self.assertEqual(memory.to_json()["seen_actions"], [TAKE, PUT])
                self.assertEqual(memory.take_fresh(), [TAKE_LINE, PUT_LINE])
                self.assertNotIn("secret", json.dumps(memory.to_json()))
                self.assertNotIn("transfer:17", json.dumps(memory.to_json()))

    def test_other_item_namespaces_survive(self):
        """Mod items retain their namespace in both retained memory and fresh updates."""
        memory = AgentMemory()
        memory.absorb({"seen_actions": [{**TAKE, "item": "mod:hammer"}]}, "ben")
        expected = TAKE_LINE.replace("iron_pickaxe", "mod:hammer")
        self.assertIn(expected, memory.render())
        self.assertEqual(memory.take_fresh(), [expected])

    def test_unsupported_events_are_not_misreported_as_container_transfers(self):
        """Only the two recognized completion receipts can add witnessed transfer memory."""
        memory = AgentMemory()
        memory.absorb({"seen_actions": [{"action": "give"}, {"action": "mine"}, {"action": None}]}, "ben")
        self.assertEqual(memory.to_json(), LEGACY_MEMORY)
        self.assertEqual(memory.take_fresh(), [])

    def test_running_activity_is_not_inferred_to_be_a_completed_transfer(self):
        """Rendering a visible activity is separate from remembering witnessed completion receipts."""
        observation = copy.deepcopy(OBSERVATION)
        observation["entities"][0]["activity"] = {"skill": "chest_take", "pos": TAKE["pos"]}
        memory = AgentMemory()
        memory.absorb(observation, "ben")
        self.assertNotIn("seen_actions", memory.to_json())
        self.assertEqual(memory.take_fresh(), [])

    def test_seen_actions_do_not_change_the_speech_interruption_flag(self):
        """Witnessed transfers are updates, but only hearing someone else's words interrupts."""
        for speaker, expected in ((None, False), ("ben", False), ("builder", True)):
            with self.subTest(speaker=speaker):
                heard = [] if speaker is None else [{"gameTime": 0, "speaker": speaker, "text": "hi", "distance": 2}]
                memory = AgentMemory()
                self.assertEqual(memory.absorb({"heard": heard, "seen_actions": [TAKE]}, "ben"), expected)
                self.assertEqual(memory.to_json()["seen_actions"], [TAKE])

    def test_block_changes_reads_and_notices_still_coexist(self):
        """Transfers get their own conditional section without replacing existing memory sections."""
        memory = AgentMemory()
        change = {"gameTime": 0, "pos": [1, -60, 2], "from": "minecraft:air", "to": "minecraft:oak_planks"}
        memory.absorb({"changes": [change], "seen_actions": [TAKE]}, "ben")
        memory.record_read(0, "public sign")
        memory.record_noticed(0, "a dropped log")
        rendered = memory.render()
        self.assertIn("block at (1, -60, 2): air -> oak_planks", rendered)
        self.assertLess(rendered.index("Changes you saw:"), rendered.index("Actions you saw:"))
        self.assertLess(rendered.index("Actions you saw:"), rendered.index("Things you read:"))
        self.assertIn("public sign", rendered)
        self.assertIn("a dropped log", rendered)
        self.assertEqual(memory.to_json()["changes"], [change])
        self.assertEqual(memory.take_fresh()[1], TAKE_LINE)

    def test_retained_events_are_bounded_oldest_first(self):
        """Repeated observation batches evict only the oldest transfers, like block changes."""
        memory = AgentMemory()
        limit = AgentMemory.LIMITS["seen_actions"]
        events = [{**TAKE, "gameTime": index * 20, "actor": f"builder_{index}"} for index in range(limit + 3)]
        for event in events:
            memory.absorb({"seen_actions": [event]}, "ben")
        self.assertEqual(memory.to_json()["seen_actions"], events[-limit:])
        rendered = memory.render()
        self.assertEqual(rendered.count(" take iron_pickaxe from the chest"), limit)
        self.assertNotIn("saw builder_0 take", rendered)
        self.assertLess(rendered.index("saw builder_3 take"), rendered.index(f"saw builder_{limit + 2} take"))

    def test_fresh_bounds_and_overflow_survive_json_restore(self):
        """Fresh overflow is counted independently of the bounded retained transfer section."""
        memory = AgentMemory()
        events = [{**TAKE, "gameTime": index * 20, "actor": f"builder_{index}"}
                  for index in range(AgentMemory.FRESH_LIMIT + 5)]
        memory.absorb({"seen_actions": events}, "ben")
        snapshot = json.loads(json.dumps(memory.to_json()))
        self.assertEqual(len(snapshot["seen_actions"]), AgentMemory.LIMITS["seen_actions"])
        self.assertEqual(len(snapshot["fresh"]), AgentMemory.FRESH_LIMIT)
        self.assertEqual(snapshot["freshDropped"], 5)
        restored = AgentMemory(snapshot)
        self.assertEqual(restored.to_json(), snapshot)
        self.assertEqual(restored.render(), memory.render())
        updates = restored.take_fresh()
        self.assertEqual(updates, memory.take_fresh())
        self.assertEqual(updates[0], "  (5 earlier updates not shown)")
        self.assertIn("saw builder_5 take", updates[1])
        self.assertIn(f"saw builder_{AgentMemory.FRESH_LIMIT + 4} take", updates[-1])
        self.assertEqual(restored.take_fresh(), [])
        self.assertEqual(restored.to_json()["freshDropped"], 0)

    def test_consuming_fresh_does_not_remove_retained_events(self):
        """An epoch restart or restored branch can still see transfers after updates were consumed."""
        memory = AgentMemory()
        memory.absorb({"seen_actions": [TAKE, PUT]}, "ben")
        memory.take_fresh()
        restored = AgentMemory(json.loads(json.dumps(memory.to_json())))
        self.assertEqual(restored.render(), memory.render())
        self.assertIn(TAKE_LINE, restored.render())
        self.assertIn(PUT_LINE, restored.render())
        self.assertEqual(restored.take_fresh(), [])


class PerceptionPromptTest(unittest.TestCase):
    """Witnessed transfers reach full prompts, incremental updates and restored epochs exactly once."""

    @staticmethod
    def agent(epoch_turns: int) -> Agent:
        """Create an observer with no unrelated actions or memory."""
        return Agent(RoleBrief("ben", "observer", "Watches.", "Remember."), epoch_turns=epoch_turns)

    def test_legacy_full_and_incremental_prompts_are_byte_stable(self):
        """Null additions cannot change either epoch shape, including after restoring an old snapshot."""
        expected_full = f"{LEGACY_MEMORY_TEXT}\n\n{LEGACY_VIEW}\n\nChoose your next action."
        expected_incremental = (f"Since your last action:\n  (nothing new)\n\n{LEGACY_VIEW}\n\n"
                                "Choose your next action.")
        for fields in ({}, {"seen_actions": None}, {"seen_actions": []}):
            with self.subTest(fields=fields):
                agent = self.agent(8)
                agent.memory = AgentMemory(copy.deepcopy(LEGACY_MEMORY))
                observation = {**copy.deepcopy(OBSERVATION), **fields}
                observation["entities"][0].update(heldItem=None, activity=None)
                agent.memory.absorb(observation, agent.name)
                first = agent.next_prompt(observation, False)
                self.assertEqual(first.encode("utf-8"), expected_full.encode("utf-8"))
                agent.finish_turn(first, None, "no action chosen")
                snapshot = json.dumps(agent.to_json())
                restored = Agent.from_json(json.loads(snapshot))
                self.assertEqual(json.dumps(restored.to_json()), snapshot)
                self.assertEqual(restored.next_prompt(observation, False).encode("utf-8"),
                                 expected_incremental.encode("utf-8"))

    def test_full_incremental_and_new_epoch_prompts_include_transfers_once(self):
        """Full memory retains witnessed events while incremental prompts show each drained batch once."""
        for epoch_turns in (1, 2):
            with self.subTest(epoch_turns=epoch_turns):
                agent = self.agent(epoch_turns)
                first = agent.next_prompt(OBSERVATION, False)
                agent.finish_turn(first, None, "no action chosen")
                agent.memory.absorb({"seen_actions": [TAKE, PUT]}, agent.name)
                second = agent.next_prompt(OBSERVATION, False)
                self.assertEqual(second.count(TAKE_LINE), 1)
                self.assertEqual(second.count(PUT_LINE), 1)
                if epoch_turns == 2:
                    self.assertTrue(second.startswith(f"Since your last action:\n{TAKE_LINE}\n{PUT_LINE}\n\n"))
                else:
                    self.assertIn("Actions you saw:", second)
                agent.finish_turn(second, None, "no action chosen")
                agent.memory.absorb({"seen_actions": []}, agent.name)
                third = agent.next_prompt(OBSERVATION, False)
                self.assertIn("Actions you saw:", third)
                self.assertEqual(third.count(TAKE_LINE), 1)
                self.assertEqual(third.count(PUT_LINE), 1)
                self.assertEqual(agent.history(), ())

    def test_pending_updates_and_history_round_trip_through_agent_snapshot(self):
        """A fork before a decision receives the same updates and no repeat on its following decision."""
        agent = self.agent(8)
        first = agent.next_prompt(OBSERVATION, False)
        agent.finish_turn(first, None, "no action chosen")
        agent.memory.absorb({"seen_actions": [TAKE, PUT]}, agent.name)
        snapshot = json.loads(json.dumps(agent.to_json()))
        restored = Agent.from_json(snapshot)
        self.assertEqual(restored.to_json(), snapshot)
        self.assertEqual(restored.history(), agent.history())
        second = agent.next_prompt(OBSERVATION, True)
        self.assertEqual(restored.next_prompt(OBSERVATION, True), second)
        self.assertEqual(second.count(TAKE_LINE), 1)
        self.assertEqual(second.count(PUT_LINE), 1)
        restored.finish_turn(second, None, "no action chosen")
        third = restored.next_prompt(OBSERVATION, False)
        self.assertIn("  (nothing new)", third)
        self.assertNotIn("saw builder take", third)
        self.assertIn(TAKE_LINE, restored.user_prompt(OBSERVATION, False))


if __name__ == "__main__":
    unittest.main()
