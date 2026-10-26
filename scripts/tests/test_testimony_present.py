"""Pin v2 same-observation presence and relay preservation without running a server."""

from copy import deepcopy
import math
import unittest

from scripts.audit_testimony import TestimonyAudit


CHANGE = {"type": "block_change", "id": 7, "pos": [1, 2, 3], "gameTime": 100}


def observation(entities: list[dict] | None = None) -> dict:
    """Build one Villagers-format sight record containing a block ray and an agent ray at tick 110."""
    return {"id": 8, "type": "npc_observation", "prov": "npc_observation:8", "observer": "npc:a", "npc": "a",
            "gameTime": 110, "observerEye": [0, 4, 0], "changes": [{**deepcopy(CHANGE), "losRay": 0}],
            "entities": deepcopy(entities if entities is not None else
                                 [{"prov": "entity:envoy", "name": "envoy", "pos": [1.5, 2.5, 3.5], "losRay": 0}])}


def news(present: list[str] | None = None) -> dict:
    """Build first-hand block news using only the existing fields plus present."""
    return {"newsId": "change:7", "originEventId": "change:7", "kind": "bridge_or_port_change",
            "subject": "the bridge", "chain": ["A"], "firsthand": True, "learnedAt": 110, "originatedAt": 100,
            "present": list(present if present is not None else ["envoy"])}


def speech(item: dict, speaker: str = "A", key: str = "npc:a", start: int = 200,
           listeners: tuple[str, ...] = ()) -> dict:
    """Carry one news item in a completed utterance whose listeners fully heard its content."""
    return {"type": "utterance", "uttId": start, "speaker": speaker, "speakerKey": key,
            "gameTime": start + 10, "startedAt": start, "listeners": list(listeners), "news": [deepcopy(item)]}


class TestimonyPresenceTest(unittest.TestCase):
    """Presence is a complete sorted list from one sight record, not a guess from nearby times or speakers."""

    def check_item(self, item: dict, records: list[dict]) -> TestimonyAudit:
        """Audit one first-hand item against a real-format event and the supplied in-memory sight records."""
        return TestimonyAudit([deepcopy(CHANGE), speech(item)], records).run()

    def test_sorted_unique_agents_with_rays_pass(self):
        """NPCs, occluded entities and out-of-sphere entities never become present names."""
        entities = [{"name": "envoy", "pos": [1.5, 2.5, 3.5], "losRay": 0},
                    {"name": "alice", "pos": [9.5, 2.5, 3.5], "losRay": 1},
                    {"name": "alice", "pos": [1.5, 2.5, 3.5], "losRay": 2},
                    {"name": "villager", "npcId": "adult_01", "pos": [1.5, 2.5, 3.5], "losRay": 0},
                    {"name": "hidden", "pos": [1.5, 2.5, 3.5]},
                    {"name": "distant", "pos": [9.5, 10.5, 3.5], "losRay": 0}]
        audit = self.check_item(news(["alice", "envoy"]), [observation(entities)])
        self.assertEqual((audit.checked, audit.violations), (1, []))

    def test_inclusive_eight_block_center_distance_in_all_three_axes(self):
        """Feet-to-center distance includes exactly eight blocks and excludes the next representable radius."""
        center = [1.5, 2.5, 3.5]
        for axis in range(3):
            for direction in (-1, 1):
                with self.subTest(axis=axis, direction=direction):
                    position = center.copy()
                    position[axis] += direction * 8
                    record = observation([{"name": "envoy", "pos": position, "losRay": 0}])
                    self.assertEqual(self.check_item(news(), [record]).violations, [])
                    record["entities"][0]["pos"][axis] = center[axis] + direction * math.nextafter(8, math.inf)
                    self.assertTrue(self.check_item(news(), [record]).violations)
                    self.assertEqual(self.check_item(news([]), [record]).violations, [])

    def test_names_must_have_entity_los_in_the_same_record(self):
        """An absent/null/invalid entity ray or a missing name cannot justify attributed presence."""
        for ray in (None, -1, False, "0"):
            with self.subTest(ray=ray):
                record = observation()
                record["entities"][0]["losRay"] = ray
                self.assertTrue(self.check_item(news(), [record]).violations)
        record = observation()
        record["entities"][0].pop("losRay")
        self.assertTrue(self.check_item(news(), [record]).violations)
        record = observation()
        record["entities"][0].pop("name")
        self.assertTrue(self.check_item(news(), [record]).violations)

    def test_npc_named_envoy_is_not_an_agent(self):
        """The current Villagers npcId distinction excludes even a colliding display name."""
        record = observation()
        record["entities"][0]["npcId"] = "adult_01"
        self.assertTrue(self.check_item(news(), [record]).violations)
        self.assertEqual(self.check_item(news([]), [record]).violations, [])

    def test_different_observations_cannot_supply_presence(self):
        """Neither earlier, same-tick nor later entity sightings can be joined to another record's change."""
        for tick in (105, 110, 115):
            with self.subTest(tick=tick):
                sighting = observation()
                sighting.update(id=9, prov="npc_observation:9", gameTime=tick)
                sighting["changes"] = []
                self.assertTrue(self.check_item(news(), [observation([]), sighting]).violations)

    def test_learning_time_may_predate_but_never_follow_the_first_hand_observation(self):
        """A relay-to-sighting upgrade retains minimum learnedAt without borrowing another observation's entities."""
        for tick in (90, 100, 110):
            with self.subTest(learnedAt=tick):
                item = news()
                item["learnedAt"] = tick
                self.assertEqual(self.check_item(item, [observation()]).violations, [])
        for tick in (111, 201):
            with self.subTest(learnedAt=tick):
                item = news()
                item["learnedAt"] = tick
                self.assertTrue(self.check_item(item, [observation()]).violations)

    def test_presence_is_resolved_by_origin_and_observer_not_a_new_metadata_field(self):
        """Present needs no extra engine field, and another observer's entities cannot substantiate it."""
        item = news()
        self.assertNotIn("observation", item)
        self.assertEqual(self.check_item(item, [observation()]).violations, [])
        wrong_observer = observation()
        wrong_observer["observer"] = "npc:b"
        self.assertTrue(self.check_item(item, [observation([]), wrong_observer]).violations)

    def test_repeated_origin_changes_cannot_choose_a_more_convenient_observation(self):
        """Observation cursors admit each change once; ambiguous duplicate records cannot justify presence."""
        duplicate = observation()
        duplicate.update(id=9, prov="npc_observation:9", gameTime=120)
        self.assertTrue(self.check_item(news(), [observation([]), duplicate]).violations)

    def test_future_observation_does_not_ground_already_started_speech(self):
        """An observation acquired while speaking is not a source for the item selected at speech start."""
        item, record = news(), observation()
        record["gameTime"] = 201
        item["learnedAt"] = 201
        self.assertTrue(self.check_item(item, [record]).violations)

    def test_block_change_needs_its_own_ray_position_and_event_time(self):
        """Agent LOS alone cannot ground an unseen, mistimed or differently positioned block change."""
        for field, wrong in (("losRay", None), ("losRay", -1), ("gameTime", 99), ("pos", [30, 2, 3])):
            with self.subTest(field=field, wrong=wrong):
                record = observation()
                record["changes"][0][field] = wrong
                self.assertTrue(self.check_item(news(), [record]).violations)
        record = observation()
        record["changes"][0].pop("losRay")
        self.assertTrue(self.check_item(news(), [record]).violations)

    def test_observation_cannot_predate_its_block_event(self):
        """Matching item and record ticks still cannot precede the event supposedly witnessed."""
        item, record = news(), observation()
        record["gameTime"] = 99
        item["learnedAt"] = 99
        self.assertTrue(self.check_item(item, [record]).violations)

    def test_presence_is_complete_sorted_and_unique(self):
        """Omitting an admitted bot, inventing one or corrupting the list is a violation."""
        for present in ([], ["fabricated"], ["envoy", "envoy"], ["envoy", "alice"], [None], "envoy", None):
            with self.subTest(present=present):
                item = news()
                item["present"] = present
                self.assertTrue(self.check_item(item, [observation()]).violations)

    def test_accident_scene_has_explicit_empty_presence(self):
        """No admitted nearby agent is valid empty first-hand presence, resolved to the change observation."""
        self.assertEqual(self.check_item(news([]), [observation([])]).violations, [])

    def test_old_first_hand_item_without_presence_stays_compatible(self):
        """Old logs need no new entity/ray fields when present is absent."""
        item = news()
        item.pop("present")
        legacy = {"observer": "npc:a", "gameTime": 110, "changes": [{"id": 7}]}
        self.assertEqual(self.check_item(item, [legacy]).violations, [])

    def test_three_hop_relay_preserves_original_presence(self):
        """Learned times advance and chains grow without replacing the original presence."""
        source = news()
        second = deepcopy(source)
        second.update(chain=["A", "B"], firsthand=False, learnedAt=210)
        third = deepcopy(second)
        third.update(chain=["A", "B", "C"], learnedAt=310)
        events = [CHANGE, speech(source, listeners=("npc:b",)),
                  speech(second, "B", "npc:b", 300, ("npc:c",)), speech(third, "C", "npc:c", 400)]
        audit = TestimonyAudit(events, [observation()]).run()
        self.assertEqual((audit.checked, audit.violations), (3, []))

    def test_relay_cannot_add_remove_or_replace_presence(self):
        """A valid delivery of a news ID is not permission to alter its original presence."""
        for present in ([], ["alice"], ["alice", "envoy"]):
            with self.subTest(present=present):
                relay = news(present)
                relay.update(chain=["A", "B"], firsthand=False, learnedAt=210)
                audit = TestimonyAudit([CHANGE, speech(news(), listeners=("npc:b",)),
                                       speech(relay, "B", "npc:b", 300)], [observation()]).run()
                self.assertEqual(len(audit.violations), 1)
                self.assertIn("relay changed present", audit.violations[0])

    def test_relay_cannot_drop_nonempty_presence(self):
        """Missing presence means [], so removing a nonempty source list is still a changed relay."""
        relay = news()
        relay.pop("present")
        relay.update(chain=["A", "B"], firsthand=False, learnedAt=210)
        audit = TestimonyAudit([CHANGE, speech(news(), listeners=("npc:b",)),
                               speech(relay, "B", "npc:b", 300)], [observation()]).run()
        self.assertEqual(len(audit.violations), 1)

    def test_future_matching_delivery_cannot_repair_a_changed_relay(self):
        """Only completed deliveries before speech start may justify a relayed presence value."""
        relay = news([])
        relay.update(chain=["A", "B"], firsthand=False, learnedAt=210)
        future = speech(news([]), start=400, listeners=("npc:b",))
        audit = TestimonyAudit([CHANGE, speech(news(), listeners=("npc:b",)),
                               speech(relay, "B", "npc:b", 300), future], [observation()]).run()
        self.assertTrue(any("utt 300" in violation and "relay changed" in violation for violation in audit.violations))

    def test_legacy_absent_and_explicit_empty_presence_are_equivalent_on_relays(self):
        """Missing presence defaults to [] for comparison without inventing metadata in old runs."""
        source = news([])
        for source_missing, relay_missing in ((True, True), (True, False), (False, True)):
            with self.subTest(source_missing=source_missing, relay_missing=relay_missing):
                old_source, relay = deepcopy(source), deepcopy(source)
                if source_missing:
                    old_source.pop("present")
                if relay_missing:
                    relay.pop("present")
                relay.update(chain=["A", "B"], firsthand=False, learnedAt=210)
                events = [CHANGE, speech(old_source, "Bot", "bot", listeners=("npc:b",)),
                          speech(relay, "B", "npc:b", 300)]
                events[-1]["news"][0]["chain"] = ["Bot", "B"]
                self.assertEqual(TestimonyAudit(events, []).run().violations, [])

    def test_shout_news_cannot_invent_block_presence(self):
        """A heard shout remains a claim, not a source of first-hand block-presence evidence."""
        item = news()
        item.update(originEventId="utt:1", newsId="utt:1", kind="shout")
        origin = speech({}, "Bot", "bot", start=1)
        origin["news"] = []
        audit = TestimonyAudit([origin, speech(item)], []).run()
        self.assertEqual(len(audit.violations), 1)
        item["present"] = []
        self.assertEqual(TestimonyAudit([origin, speech(item)], []).run().violations, [])


if __name__ == "__main__":
    unittest.main()
