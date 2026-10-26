"""Offline field validation for successful pickup events without changing legacy event admission."""

import unittest

from scripts import audit_o1
from scripts.tests.test_observation import exposure


ITEM_ENTITY = "a1234567-89ab-4cde-8f01-23456789abcd"
THROWER = "b1234567-89ab-4cde-8f01-23456789abcd"


class PickupEventAuditTests(unittest.TestCase):
    """Pickup records require a positive actual count, canonical UUIDs and the entity provenance link."""

    def setUp(self):
        """Build a partial pickup whose fractional position and thrower come from the item entity."""
        self.event = {"id": 7, "type": "pickup", "gameTime": 10, "bot": "builder",
                      "item": "minecraft:iron_pickaxe", "count": 1, "itemEntity": ITEM_ENTITY,
                      "pos": [1.25, -59.875, 2.625], "prov": f"entity:{ITEM_ENTITY}", "thrower": THROWER}

    def audit(self):
        """Validate the event with an empty but otherwise valid offline observation."""
        return audit_o1.audit_logs([self.event], [exposure([])])

    def test_partial_and_multi_item_pickups_with_optional_thrower(self):
        """The actual positive count may be partial or multiple, and natural item drops have no thrower."""
        for count in (1, 3, 64):
            with self.subTest(count=count):
                self.event["count"] = count
                self.assertEqual([], self.audit())
        del self.event["thrower"]
        self.assertEqual([], self.audit())
        self.event["pos"] = [1, -60, 2]
        self.assertEqual([], self.audit())

    def test_every_required_event_field_is_required(self):
        """Missing pickup payload or common event fields fail before indexing the event."""
        for field in set(self.event) - {"thrower"}:
            with self.subTest(field=field):
                original = self.event.pop(field)
                self.assertTrue(self.audit())
                self.event[field] = original

    def test_count_is_a_strict_positive_integer(self):
        """Zero, negative, boolean, fractional and nonnumeric counts cannot describe a successful pickup."""
        for count in (0, -1, True, False, 1.0, 1.5, "1", None, float("nan"), float("inf")):
            with self.subTest(count=count):
                self.event["count"] = count
                self.assertTrue(self.audit())

    def test_bot_and_item_are_nonempty_strings(self):
        """An actor and item identity are mandatory, not numbers, arrays or nulls."""
        for field in ("bot", "item"):
            original = self.event[field]
            for value in ("", None, True, 1, [], {}):
                with self.subTest(field=field, value=value):
                    self.event[field] = value
                    self.assertTrue(self.audit())
            self.event[field] = original

    def test_entity_and_thrower_uuids_must_be_canonical(self):
        """Both UUID fields reject alternate spellings and invalid data, even when provenance is forged to match."""
        for field, canonical in (("itemEntity", ITEM_ENTITY), ("thrower", THROWER)):
            for value in (canonical.upper(), canonical.replace("-", ""), "{" + canonical + "}",
                          "urn:uuid:" + canonical, "not-a-uuid", "", None, 7, True):
                with self.subTest(field=field, value=value):
                    self.event[field] = value
                    if field == "itemEntity":
                        self.event["prov"] = f"entity:{value}"
                    self.assertTrue(self.audit())
            self.event[field] = canonical
            self.event["prov"] = f"entity:{ITEM_ENTITY}"

    def test_provenance_names_the_picked_up_entity_not_the_event(self):
        """The raw event uses entity provenance, unlike its pickup:<id> public action source reference."""
        for provenance in (f"entity:{THROWER}", "pickup:7", "drop:7", "transfer:7", ITEM_ENTITY, "", None):
            with self.subTest(provenance=provenance):
                self.event["prov"] = provenance
                self.assertTrue(self.audit())

    def test_position_requires_three_finite_numeric_coordinates(self):
        """Item positions accept fractions, but not booleans, infinities or malformed vectors."""
        for position in (None, [], [0, 0], [0, 0, 0, 0], [True, 0, 0], [0, "0", 0],
                         [0, float("nan"), 0], [0, 0, float("inf")], [float("-inf"), 0, 0]):
            with self.subTest(position=position):
                self.event["pos"] = position
                self.assertTrue(self.audit())

    def test_invalid_pickup_is_not_indexed(self):
        """Malformed event payloads cannot subsequently serve as resolved source events."""
        audit = audit_o1.ExposureAudit()
        self.event["count"] = 0
        with self.assertRaises(audit_o1.AuditError):
            audit.event(self.event)
        self.assertNotIn(7, audit.events)

    def test_legacy_drop_transfer_and_other_event_behavior_is_unchanged(self):
        """Pickup-only validation does not retroactively require fields on unrelated legacy event kinds."""
        for kind in ("drop", "transfer", "skill_end"):
            with self.subTest(kind=kind):
                legacy = {"id": 1, "type": kind, "gameTime": 0, "count": 0, "itemEntity": "legacy"}
                self.assertEqual([], audit_o1.audit_logs([legacy], [exposure([])]))


if __name__ == "__main__":
    unittest.main()
