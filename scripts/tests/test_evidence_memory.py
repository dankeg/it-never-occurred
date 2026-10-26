"""Pure unit tests for bounded, count-free memories of witnessed item movements."""

import copy
import json
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief


GIVE = {"gameTime": 0, "actor": "envoy", "action": "give", "target": "freight",
        "item": "minecraft:iron_pickaxe"}
DROP = {"gameTime": 20, "actor": "envoy", "action": "drop", "pos": [-8, -61, 0],
        "item": "minecraft:iron_pickaxe"}
PICKUP = {"gameTime": 40, "actor": "security", "action": "pickup", "pos": [-8, -61, 0],
          "item": "minecraft:iron_pickaxe"}
FACTS = (GIVE, DROP, PICKUP)
LINES = (
    "  [day 0, 06:00] saw envoy give iron_pickaxe to freight",
    "  [day 0, 06:01] saw envoy drop iron_pickaxe at (-8, -61, 0)",
    "  [day 0, 06:02] saw security pick up iron_pickaxe at (-8, -61, 0)",
)
EMPTY_MEMORY = {"notes": [], "actions": [], "heard": [], "changes": [], "reads": [], "noticed": [],
                "fresh": [], "freshDropped": 0, "inventory": None}
EMPTY_MEMORY_TEXT = (
    "Your notes:\n  (none)\n\nYour recent actions:\n  (none)\n\nWhat you heard:\n  (none)\n\n"
    "Changes you saw:\n  (none)\n\nThings you read:\n  (none)"
)
CHEST_FACT = {"gameTime": 0, "actor": "builder", "action": "chest_take", "pos": [-8, -61, 0],
              "item": "minecraft:iron_pickaxe"}
CHEST_LINE = "  [day 0, 06:00] saw builder take iron_pickaxe from the chest at (-8, -61, 0)"
CHEST_MEMORY = {"notes": [], "actions": [], "heard": [], "changes": [], "reads": [], "noticed": [],
                "seen_actions": [CHEST_FACT], "fresh": [], "freshDropped": 0, "inventory": None}
OBSERVATION = {"gameTime": 0, "self": {"name": "ben", "pos": [0, -60, 0], "health": 20,
                                     "food": 20, "inventory": []}}
VIEW = (
    "Time: day 0, 06:00 (tick 0)\n\n"
    "You are ben at (0, -60, 0). Health 20/20, food 20/20.\nInventory: empty\n\n"
    "People and things you can see:\n  (none)\n\nNotable blocks you can see:\n  (none)\n\n"
    "Writing you can read:\n  (none)"
)


class EvidenceMemoryTest(unittest.TestCase):
    """Completed item movements share the bounded witnessed-action and fresh-update lifecycle."""

    def test_give_drop_and_pickup_render_in_receipt_order(self):
        """Each action renders its public item and destination, never implying a speech interruption."""
        memory = AgentMemory()
        self.assertFalse(memory.absorb({"seen_actions": copy.deepcopy(FACTS)}, "ben"))
        self.assertEqual(memory.to_json()["seen_actions"], list(FACTS))
        expected = EMPTY_MEMORY_TEXT.replace("Things you read:",
                                             "Actions you saw:\n" + "\n".join(LINES) + "\n\nThings you read:")
        self.assertEqual(memory.render(), expected)
        self.assertEqual(memory.to_json()["actions"], [])
        self.assertEqual(memory.take_fresh(), list(LINES))
        self.assertEqual(memory.take_fresh(), [])

    def test_only_action_specific_public_facts_are_retained(self):
        """Provenance, sight proofs, quantities, notes and raw arguments cannot enter memory or prompts."""
        for fact, line in zip(FACTS, LINES):
            for provenance in (None, "transfer:secret", {"id": "secret_receipt"}):
                with self.subTest(action=fact["action"], provenance=provenance):
                    receipt = {**copy.deepcopy(fact), "prov": provenance, "count": 917, "note": "secret_note",
                               "args": {"item": "secret_item", "target": "secret_target", "count": 918},
                               "actorSight": {"inventory": ["secret_inventory"]}, "thrower": "secret_thrower",
                               "itemEntity": "secret_entity"}
                    if fact["action"] == "give":
                        receipt["pos"] = [991, 992, 993]
                    else:
                        receipt["target"] = "secret_target"
                    original = copy.deepcopy(receipt)
                    agent = Agent(RoleBrief("ben", "observer", "Watches.", "Remember."))
                    agent.memory.absorb({"seen_actions": [receipt]}, agent.name)
                    self.assertEqual(receipt, original)
                    self.assertEqual(agent.memory.to_json()["seen_actions"], [fact])
                    prompt = agent.next_prompt(OBSERVATION, False)
                    self.assertEqual(prompt.count(line), 1)
                    self.assertNotIn("secret", json.dumps(agent.to_json()) + prompt)
                    self.assertNotIn("917", json.dumps(agent.to_json()) + prompt)
                    self.assertNotIn("991", json.dumps(agent.to_json()) + prompt)

    def test_positions_are_copied_from_observations(self):
        """Changing an already-absorbed observation cannot rewrite a remembered drop or pickup."""
        receipts = copy.deepcopy([DROP, PICKUP])
        memory = AgentMemory()
        memory.absorb({"seen_actions": receipts}, "ben")
        for receipt in receipts:
            receipt["pos"][0] = 999
        self.assertEqual(memory.to_json()["seen_actions"], [DROP, PICKUP])
        self.assertEqual(memory.take_fresh(), list(LINES[1:]))

    def test_item_namespaces_and_block_coordinates_use_existing_formatting(self):
        """Only Minecraft item names shorten, and public positions keep whole-block rounding."""
        for fact, line in zip(FACTS, LINES):
            with self.subTest(action=fact["action"]):
                receipt = {**fact, "item": "mod:hammer"}
                expected = line.replace("iron_pickaxe", "mod:hammer")
                if "pos" in fact:
                    receipt["pos"] = [-8.2, -60.9, 0.1]
                memory = AgentMemory()
                memory.absorb({"seen_actions": [receipt]}, "ben")
                self.assertIn(expected, memory.render())
                self.assertEqual(memory.take_fresh(), [expected])

    def test_unknown_and_incomplete_receipts_add_nothing(self):
        """Unrecognized actions and unfinished item-movement facts are not completion evidence."""
        receipts = [{"action": "mine"}, {"action": "future_action", "note": "secret"}, {"action": None}]
        for fact in FACTS:
            receipts.append({"action": fact["action"]})
            receipts.extend({key: value for key, value in fact.items() if key != missing}
                            for missing in fact if missing != "action")
        memory = AgentMemory()
        self.assertFalse(memory.absorb({"seen_actions": receipts}, "ben"))
        self.assertEqual(memory.to_json(), EMPTY_MEMORY)
        self.assertEqual(memory.render(), EMPTY_MEMORY_TEXT)
        self.assertEqual(memory.take_fresh(), [])

    def test_visible_running_activities_do_not_create_completion_memories(self):
        """A visible give, drop or collect skill is not itself a witnessed completed movement."""
        for skill in ("give", "drop", "collect"):
            with self.subTest(skill=skill):
                memory = AgentMemory()
                memory.absorb({"entities": [{"name": "envoy", "activity": {"skill": skill}}]}, "ben")
                self.assertEqual(memory.to_json(), EMPTY_MEMORY)

    def test_retained_and_fresh_bounds_survive_json_restore(self):
        """All three actions evict oldest receipts and count fresh overflow independently."""
        for fact, line in zip(FACTS, LINES):
            with self.subTest(action=fact["action"]):
                count = AgentMemory.FRESH_LIMIT + 5
                receipts = [{**copy.deepcopy(fact), "actor": f"actor_{index}"} for index in range(count)]
                lines = [line.replace(f"saw {fact['actor']} ", f"saw actor_{index} ") for index in range(count)]
                memory = AgentMemory()
                for receipt in receipts:
                    memory.absorb({"seen_actions": [receipt]}, "ben")
                retained_limit = AgentMemory.LIMITS["seen_actions"]
                snapshot = json.dumps(memory.to_json())
                restored = AgentMemory(json.loads(snapshot))
                self.assertEqual(json.dumps(restored.to_json()), snapshot)
                self.assertEqual(restored.to_json()["seen_actions"], receipts[-retained_limit:])
                self.assertEqual(restored.to_json()["fresh"], lines[-AgentMemory.FRESH_LIMIT:])
                self.assertEqual(restored.to_json()["freshDropped"], 5)
                expected = EMPTY_MEMORY_TEXT.replace(
                    "Things you read:",
                    "Actions you saw:\n" + "\n".join(lines[-retained_limit:]) + "\n\nThings you read:")
                self.assertEqual(restored.render(), expected)
                self.assertEqual(restored.render(), memory.render())
                self.assertEqual(restored.take_fresh(), ["  (5 earlier updates not shown)"]
                                 + lines[-AgentMemory.FRESH_LIMIT:])
                self.assertEqual(restored.take_fresh(), [])
                self.assertEqual(restored.to_json()["freshDropped"], 0)

    def test_consumed_updates_leave_retained_memories_after_restore(self):
        """Draining updates does not discard evidence needed at the start of the next epoch."""
        memory = AgentMemory()
        memory.absorb({"seen_actions": copy.deepcopy(FACTS)}, "ben")
        memory.take_fresh()
        restored = AgentMemory(json.loads(json.dumps(memory.to_json())))
        self.assertEqual(restored.to_json(), memory.to_json())
        self.assertEqual(restored.render(), memory.render())
        for line in LINES:
            self.assertIn(line, restored.render())
        self.assertEqual(restored.take_fresh(), [])


class EvidencePromptTest(unittest.TestCase):
    """Item evidence reaches full epochs and incremental prompts without changing legacy prompt bytes."""

    @staticmethod
    def agent(epoch_turns: int) -> Agent:
        """Create an observer with no unrelated state or chosen actions."""
        return Agent(RoleBrief("ben", "observer", "Watches.", "Remember."), epoch_turns=epoch_turns)

    def test_full_incremental_and_new_epoch_prompts_include_each_action_once(self):
        """Epoch length changes prompt shape, not the retained facts or their receipt order."""
        for epoch_turns in (1, 2):
            with self.subTest(epoch_turns=epoch_turns):
                agent = self.agent(epoch_turns)
                first = agent.next_prompt(OBSERVATION, False)
                agent.finish_turn(first, None, "no action chosen")
                agent.memory.absorb({"seen_actions": copy.deepcopy(FACTS)}, agent.name)
                second = agent.next_prompt(OBSERVATION, False)
                for line in LINES:
                    self.assertEqual(second.count(line), 1)
                if epoch_turns == 2:
                    self.assertEqual(second, "Since your last action:\n" + "\n".join(LINES)
                                     + f"\n\n{VIEW}\n\nChoose your next action.")
                else:
                    self.assertIn("Actions you saw:", second)
                agent.finish_turn(second, None, "no action chosen")
                agent.memory.absorb({"seen_actions": []}, agent.name)
                third = agent.next_prompt(OBSERVATION, False)
                self.assertIn("Actions you saw:", third)
                for line in LINES:
                    self.assertEqual(third.count(line), 1)
                self.assertEqual(agent.history(), ())

    def test_pending_evidence_and_epoch_history_round_trip(self):
        """Forked pending updates appear once, while retained evidence remains in later full prompts."""
        agent = self.agent(8)
        first = agent.next_prompt(OBSERVATION, False)
        agent.finish_turn(first, None, "no action chosen")
        agent.memory.absorb({"seen_actions": copy.deepcopy(FACTS)}, agent.name)
        snapshot = json.dumps(agent.to_json())
        restored = Agent.from_json(json.loads(snapshot))
        self.assertEqual(json.dumps(restored.to_json()), snapshot)
        self.assertEqual(restored.history(), agent.history())
        second = agent.next_prompt(OBSERVATION, True)
        self.assertEqual(restored.next_prompt(OBSERVATION, True), second)
        self.assertTrue(second.startswith("Since your last action:\n" + "\n".join(LINES) + "\n\n"))
        for line in LINES:
            self.assertEqual(second.count(line), 1)
        restored.finish_turn(second, None, "no action chosen")
        third = restored.next_prompt(OBSERVATION, False)
        self.assertEqual(third, f"Since your last action:\n  (nothing new)\n\n{VIEW}\n\nChoose your next action.")
        for line in LINES:
            self.assertIn(line, restored.user_prompt(OBSERVATION, False))

    def test_no_new_actions_preserve_serialization_render_and_prompt_bytes(self):
        """Empty and chest-only snapshots keep their exact keys, order, sections and epoch text."""
        chest_text = EMPTY_MEMORY_TEXT.replace("Things you read:",
                                               f"Actions you saw:\n{CHEST_LINE}\n\nThings you read:")
        for state, text in ((EMPTY_MEMORY, EMPTY_MEMORY_TEXT), (CHEST_MEMORY, chest_text)):
            for fields in ({}, {"seen_actions": None}, {"seen_actions": []},
                           {"seen_actions": [{"action": "future_action"}]}):
                with self.subTest(chest=bool(state.get("seen_actions")), fields=fields):
                    agent = self.agent(8)
                    agent.memory = AgentMemory(copy.deepcopy(state))
                    self.assertFalse(agent.memory.absorb(fields, agent.name))
                    self.assertEqual(json.dumps(agent.memory.to_json()).encode("utf-8"),
                                     json.dumps(state).encode("utf-8"))
                    self.assertEqual(agent.memory.render().encode("utf-8"), text.encode("utf-8"))
                    first = agent.next_prompt(OBSERVATION, False)
                    expected = f"{text}\n\n{VIEW}\n\nChoose your next action."
                    self.assertEqual(first.encode("utf-8"), expected.encode("utf-8"))
                    agent.finish_turn(first, None, "no action chosen")
                    snapshot = json.dumps(agent.to_json())
                    restored = Agent.from_json(json.loads(snapshot))
                    self.assertEqual(json.dumps(restored.to_json()).encode("utf-8"), snapshot.encode("utf-8"))
                    incremental = f"Since your last action:\n  (nothing new)\n\n{VIEW}\n\nChoose your next action."
                    self.assertEqual(restored.next_prompt(OBSERVATION, False).encode("utf-8"),
                                     incremental.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
