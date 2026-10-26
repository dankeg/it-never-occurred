"""Unit tests for IT-01's checkers: response fields and types, and rejection texts and reasons (no server)."""

import unittest

from scripts import harness
from scripts.test_it01_protocol import (
    COMMANDS, INVALID_REQUESTS, KNOWN_REASONS, LIBRARY_MESSAGE, NO_ARGUMENTS, RESPONSE_FIELDS, rejection_problems,
    response_problems,
)


def bot_state() -> dict:
    """Return a concrete bot response with a nonempty inventory, not a fake client or server."""
    return {"name": "it_a", "pos": [0.5, -59, 0.5], "deltaMovement": [0, 0, 0], "yRot": 0, "xRot": 0,
            "onGround": True, "health": 20.0, "food": 20, "selectedSlot": 0,
            "inventory": [{"slot": 0, "item": "minecraft:written_book", "count": 1}]}


def observation() -> dict:
    """Return populated caller-visible nested fields, including book metadata and a heard utterance."""
    return {
        "gameTime": 20,
        "self": {"name": "it_a", "pos": [0.5, -59, 0.5], "yaw": 0, "pitch": 0, "health": 20, "food": 20,
                 "selectedSlot": 0, "skill": {"skillId": None, "skill": None, "args": {}, "state": "idle",
                                            "startedAt": None, "progress": {}},
                 "inventory": [{"slot": 0, "item": "minecraft:written_book", "count": 1,
                                "doc": {"prov": "doc:book:1@v1", "title": "IT-01", "author": "it_a"}}]},
        "entities": [{"prov": "entity:one", "uuid": "one", "type": "minecraft:player", "name": "it_b",
                      "pos": [1.5, -59, 0.5], "distance": 1.0}],
        "blocks": [{"prov": "block:sign", "pos": [-1, -59, 1], "block": "minecraft:oak_sign", "distance": 2.0}],
        "structures": [{"prov": "structure:sign", "name": "sign", "pos": [-1, -59, 1], "built": 1,
                        "total": 1, "distance": 2.0}],
        "terrain": [{"x": 0, "y": -60, "z": 0, "block": "minecraft:stone"}],
        "documents": [{"prov": "doc:sign@v1", "pos": [-1, -59, 1], "distance": 2.0,
                       "front": ["IT-01", "", "", ""], "back": ["", "", "", ""]}],
        "changes": [{"prov": "change:1", "pos": [-1, -59, -1], "from": "minecraft:oak_planks",
                     "to": "minecraft:air", "gameTime": 0}],
        "heard": [{"prov": "utterance:1", "speaker": "it_b", "gameTime": 14, "text": "Hi", "volume": "talk",
                   "distance": 1.0, "to": "it_a"}],
        "chatter": [], "shout_seen": [],
    }


class CommandTablesTest(unittest.TestCase):
    """Every command has a response contract, and every command with arguments an invalid request."""

    def test_tables_cover_every_command(self):
        """No command can drop out of the response checks or, unless it takes no arguments, the rejection checks."""
        self.assertEqual(COMMANDS, tuple(sorted(COMMANDS)))
        self.assertEqual(set(RESPONSE_FIELDS), set(COMMANDS))
        self.assertEqual(set(INVALID_REQUESTS), set(COMMANDS) - set(NO_ARGUMENTS))

    def test_expected_reasons_are_declared(self):
        """Each invalid request's expected reason is in the declared set, except the JSON library's own messages."""
        self.assertEqual(KNOWN_REASONS, tuple(sorted(set(KNOWN_REASONS))))
        for command, (_, reason) in INVALID_REQUESTS.items():
            if reason != LIBRARY_MESSAGE:
                self.assertIn(reason, KNOWN_REASONS, command)


class ResponseFieldsTest(unittest.TestCase):
    """Responses must carry the fields the Python callers read, with their JSON types."""

    def test_valid_responses_pass(self):
        """A complete bot state and a populated observation have no problems."""
        self.assertEqual(response_problems("bot_state", bot_state()), [])
        self.assertEqual(response_problems("observe", observation()), [])

    def test_missing_nested_field_is_named_by_path(self):
        """A missing `self.skill.state` is reported with its full path."""
        response = observation()
        del response["self"]["skill"]["state"]
        self.assertTrue(any("result.self.skill.state" in problem for problem in response_problems("observe", response)))

    def test_boolean_is_not_a_number(self):
        """A count of `True` is reported, though Python treats bools as ints."""
        response = bot_state()
        response["inventory"][0]["count"] = True
        self.assertTrue(any("inventory[0].count" in problem for problem in response_problems("bot_state", response)))


class RejectionTest(unittest.TestCase):
    """Rejections are `CommandError('<command>: <reason>')` with a declared reason."""

    def test_declared_reason_passes(self):
        """A declared reason with the command's prefix has no problems."""
        error = harness.CommandError("bot_state: Unknown bot: it_missing")
        self.assertEqual(rejection_problems("bot_state", error, "Unknown bot: it_missing"), [])

    def test_unknown_reason_is_reported(self):
        """The gate's negative control: a fabricated reason is reported as unknown."""
        problems = rejection_problems("ping", harness.CommandError("ping: it_unknown_reason"))
        self.assertTrue(any("Unknown rejection reason" in problem for problem in problems))

    def test_wrong_prefix_and_accepted_probe_are_reported(self):
        """Another command's prefix, or no error at all, is a problem."""
        self.assertTrue(rejection_problems("blocks", harness.CommandError("hash: region is required")))
        self.assertTrue(rejection_problems("blocks", None))

    def test_library_message_needs_only_the_prefix(self):
        """Where the harness passes the JSON library's message through, any reason with the right prefix passes."""
        error = harness.CommandError("lockstep: Expected a boolean but was BEGIN_OBJECT")
        self.assertEqual(rejection_problems("lockstep", error, LIBRARY_MESSAGE), [])


if __name__ == "__main__":
    unittest.main()
