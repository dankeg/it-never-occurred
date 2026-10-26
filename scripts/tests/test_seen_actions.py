"""Offline provenance checks for completion-time, count-free action witnesses."""

import copy
import math
import unittest

from scripts import audit_o1
from scripts.tests.test_observation import EYE, exposure, spatial


NEW_ACTIONS = ("give", "drop", "pickup")
ITEM_ENTITY = "a1234567-89ab-4cde-8f01-23456789abcd"
THROWER = "b1234567-89ab-4cde-8f01-23456789abcd"


def completed_action(action):
    """Build a count-free witness with entity sight captured before its later observation."""
    event = {"id": 7, "gameTime": 10, "type": "transfer" if action == "give" else action,
             "item": "minecraft:iron_pickaxe", "count": 3}
    item = {"category": "seen_action", "prov": f"{event['type']}:7", "gameTime": 10,
            "actor": "builder", "action": action, "item": event["item"], "radius": 24.0,
            "inRadius": True, "observerEye": list(EYE),
            "actorSight": {**spatial("entity", [1.5, -60, 2.5]), "emptyTarget": False}}
    if action == "give":
        event.update({"from": "builder", "to": "freight", "note": "private handover"})
        item.update(target="freight", targetSight={**spatial("entity", [2.5, -60, 1.5]), "emptyTarget": False})
    else:
        event.update(bot="builder", itemEntity=ITEM_ENTITY, pos=[1.25, -59.875, 2.625],
                     prov=f"entity:{ITEM_ENTITY}", thrower=THROWER)
        item["pos"] = list(event["pos"])
    return event, item, exposure([item], observerEye=[100, 70, 100])


class SeenActionsAuditTests(unittest.TestCase):
    """Witnesses must resolve to real transfers and retain actor and container sight at completion."""

    def setUp(self):
        """Build a witnessed take; the observer moves away before polling its pending queue."""
        self.event = {"id": 7, "gameTime": 10, "type": "transfer", "from": "container:0,-60,2",
                      "to": "builder", "item": "minecraft:iron_pickaxe", "count": 3}
        self.item = spatial("seen_action", [0, -60, 2], "transfer:7")
        self.item.update(gameTime=10, actor="builder", action="chest_take", item="minecraft:iron_pickaxe",
                         observerEye=list(EYE), actorSight=spatial("entity", [1.5, -60, 2.5]))
        self.observation = exposure([self.item], observerEye=[100, 70, 100])

    def audit(self):
        """Return violations for this test's event and captured sight evidence."""
        return audit_o1.audit_logs([self.event], [self.observation])

    def test_take_and_put_resolve_without_revealing_count(self):
        """Both directions use completion-time evidence, not the observer's later location."""
        self.assertEqual([], self.audit())
        self.item["action"] = "chest_put"
        self.event.update({"from": "builder", "to": "container:0,-60,2"})
        self.assertEqual([], self.audit())

    def test_forged_transfer_facts_fail_closed(self):
        """Changing who, what, where, when or which action breaks the logged transfer link."""
        for field, value in (("actor", "mayor"), ("item", "minecraft:diamond"), ("pos", [0, -60, 3]),
                             ("gameTime", 11), ("action", "give"), ("prov", "transfer:8")):
            with self.subTest(field=field):
                original = self.item[field]
                self.item[field] = value
                self.assertTrue(self.audit())
                self.item[field] = original

    def test_private_arguments_and_self_witnesses_are_rejected(self):
        """Count, notes and raw skill arguments cannot enter a bystander's evidence."""
        for field, value in (("count", 3), ("note", "secret"), ("notes", ["secret"]),
                             ("args", {"count": 3}), ("target", "mayor"), ("extra", "secret")):
            with self.subTest(field=field):
                self.item[field] = value
                self.assertTrue(self.audit())
                del self.item[field]
        self.observation["observer"] = "builder"
        self.assertTrue(self.audit())

    def test_actor_and_container_need_completion_time_sight(self):
        """A missing actor proof, false radius or blocked ray cannot be admitted."""
        for target in (self.item, self.item["actorSight"]):
            for field, value in (("inRadius", False), ("distance", 100), ("hitLocation", [99, 99, 99]),
                                 ("rayFrom", [100, 100, 100])):
                with self.subTest(actor=target is not self.item, field=field):
                    original = target[field]
                    target[field] = value
                    self.assertTrue(self.audit())
                    target[field] = original
        del self.item["actorSight"]
        self.assertTrue(self.audit())

    def test_container_and_actor_sight_radii_agree(self):
        """The unchanged chest layout still uses one view radius for both completion-time proofs."""
        self.item["actorSight"]["radius"] = 25
        self.assertTrue(self.audit())

    def test_future_and_duplicate_delivery_fail_closed(self):
        """A transfer must have completed, and each witness receives it only once."""
        self.observation["gameTime"] = 9
        self.assertTrue(self.audit())
        self.observation["gameTime"] = 20
        repeated = copy.deepcopy(self.observation)
        self.assertTrue(audit_o1.audit_logs([self.event], [self.observation, repeated]))


class CompletedActionsAuditTests(unittest.TestCase):
    """Give, drop and pickup witnesses must match their sources and both required sight proofs."""

    def assert_rejected(self, event, observation):
        """Require an observation failure, rather than relying on an unrelated event violation."""
        violations = audit_o1.audit_logs([event], [observation])
        self.assertTrue(any(error.startswith("observation 1 item 1:") for error in violations), violations)

    def test_completed_actions_use_count_free_facts_and_captured_eyes(self):
        """Every new action admits its exact envelope even after the observer has moved away."""
        for action in NEW_ACTIONS:
            with self.subTest(action=action):
                event, item, observation = completed_action(action)
                self.assertNotIn("count", item)
                self.assertNotIn("note", item)
                self.assertNotIn("distance", item)
                self.assertFalse(audit_o1.RAY_FIELDS.intersection(item))
                self.assertEqual([], audit_o1.audit_logs([event], [observation]))

    def test_both_entity_rays_and_unobstructed_endpoint_hits_are_admitted(self):
        """Actor and recipient may be seen by either entity ray, including a hit at the endpoint."""
        for action in NEW_ACTIONS:
            for ray in (0, 1):
                for hit_type in ("MISS", "BLOCK"):
                    with self.subTest(action=action, ray=ray, hit_type=hit_type):
                        event, item, observation = completed_action(action)
                        fields = ("actorSight", "targetSight") if action == "give" else ("actorSight",)
                        for field in fields:
                            sight = item[field]
                            ray_to = sight["targetEye" if ray == 0 else "targetCenter"]
                            sight.update(losRay=ray, rayTo=list(ray_to), hitType=hit_type,
                                         hitPos=[math.floor(value) for value in ray_to], hitLocation=list(ray_to))
                        self.assertEqual([], audit_o1.audit_logs([event], [observation]))

    def test_forged_or_missing_public_facts_fail(self):
        """Actor, item, completion, recipient and item position must exactly match the source."""
        for action in NEW_ACTIONS:
            mutations = [("actor", "mayor"), ("item", "minecraft:diamond"), ("gameTime", 11),
                         ("gameTime", 10.0), ("gameTime", True), ("action", "collect")]
            mutations.append(("target", "mayor") if action == "give" else ("pos", [1.25, -59.875, 2.75]))
            for field, value in mutations:
                with self.subTest(action=action, field=field, value=value):
                    event, item, observation = completed_action(action)
                    item[field] = value
                    self.assert_rejected(event, observation)
            fields = audit_o1.SEEN_ACTION_FACTS | ({"target"} if action == "give" else {"pos"})
            for field in fields:
                with self.subTest(action=action, missing=field):
                    event, item, observation = completed_action(action)
                    del item[field]
                    self.assert_rejected(event, observation)

    def test_source_participants_and_payload_cannot_be_forged(self):
        """Changing a valid source's sender, recipient, bot, time, item or position invalidates the witness."""
        for action in NEW_ACTIONS:
            mutations = [("gameTime", 9), ("item", "minecraft:diamond")]
            mutations += ([("from", "mayor"), ("to", "mayor")] if action == "give" else
                          [("bot", "mayor"), ("pos", [1.25, -59.875, 2.75])])
            for field, value in mutations:
                with self.subTest(action=action, field=field):
                    event, _, observation = completed_action(action)
                    event[field] = value
                    self.assert_rejected(event, observation)

    def test_unknown_wrong_kind_and_malformed_source_provenance_fail(self):
        """An event ID must exist, have the action's source kind and use its canonical source prefix."""
        for action in NEW_ACTIONS:
            for provenance in ("transfer:999", "drop:999", "pickup:999", "event:7", f"entity:{ITEM_ENTITY}",
                               "transfer:07", "drop:0", "pickup:-7", "pickup:7:extra", None):
                with self.subTest(action=action, provenance=provenance):
                    event, item, observation = completed_action(action)
                    item["prov"] = provenance
                    self.assert_rejected(event, observation)
            with self.subTest(action=action, wrong_kind=True):
                event, _, observation = completed_action(action)
                event["type"] = "block_change"
                event.update(pos=[0, -60, 0], actor="builder", cause="test", **{"from": "stone", "to": "air"})
                self.assert_rejected(event, observation)

    def test_cross_action_source_prefixes_fail(self):
        """A transfer cannot stand in for a drop, or a drop for a pickup, even with a matching ID."""
        for action in NEW_ACTIONS:
            for prefix in {"transfer", "drop", "pickup"} - {"transfer" if action == "give" else action}:
                with self.subTest(action=action, prefix=prefix):
                    event, item, observation = completed_action(action)
                    item["prov"] = f"{prefix}:7"
                    self.assert_rejected(event, observation)

    def test_private_and_invented_envelope_fields_are_rejected(self):
        """Counts, raw source metadata, skill arguments and all other public embellishments are forbidden."""
        extras = {"count": 3, "note": "private", "notes": ["private"], "args": {"count": 3},
                  "itemEntity": ITEM_ENTITY, "thrower": THROWER, "text": "secret", "facts": {},
                  "distance": 2, **{field: None for field in audit_o1.RAY_FIELDS}}
        for action in NEW_ACTIONS:
            action_extras = {"pos": [1, -60, 2]} if action == "give" else {"target": "freight", "targetSight": {}}
            for field, value in (extras | action_extras).items():
                with self.subTest(action=action, field=field):
                    event, item, observation = completed_action(action)
                    item[field] = value
                    self.assert_rejected(event, observation)

    def test_missing_envelope_admission_fields_fail(self):
        """Completion-time eyes, positive admission and the envelope radius cannot be omitted or malformed."""
        for action in NEW_ACTIONS:
            for field in ("category", "radius", "inRadius", "observerEye", "actorSight"):
                with self.subTest(action=action, missing=field):
                    event, item, observation = completed_action(action)
                    del item[field]
                    self.assert_rejected(event, observation)
            for field, value in (("radius", -1), ("radius", True), ("radius", float("inf")),
                                 ("radius", "24"), ("inRadius", False), ("inRadius", 1),
                                 ("observerEye", [100, 70, 100]), ("observerEye", [0, 0])):
                with self.subTest(action=action, field=field, value=value):
                    event, item, observation = completed_action(action)
                    item[field] = value
                    self.assert_rejected(event, observation)

    def test_every_entity_sight_field_is_required(self):
        """Neither actor nor recipient can be admitted without its complete entity spatial proof."""
        for action in NEW_ACTIONS:
            for target in (("actorSight", "targetSight") if action == "give" else ("actorSight",)):
                for field in audit_o1.ENTITY_SIGHT_FIELDS:
                    with self.subTest(action=action, target=target, missing=field):
                        event, item, observation = completed_action(action)
                        del item[target][field]
                        self.assert_rejected(event, observation)
                for value in (None, [], "visible"):
                    with self.subTest(action=action, target=target, invalid=value):
                        event, item, observation = completed_action(action)
                        item[target] = value
                        self.assert_rejected(event, observation)
                with self.subTest(action=action, missing=target):
                    event, item, observation = completed_action(action)
                    del item[target]
                    self.assert_rejected(event, observation)

    def test_actor_and_recipient_sight_geometry_fail_closed(self):
        """False flags, malformed rays, obstructions and out-of-range actors or recipients fail independently."""
        mutations = (("category", "block"), ("inRadius", False), ("inRadius", 1), ("distance", 100),
                     ("distance", True), ("distance", float("nan")), ("radius", 1), ("losRay", 2),
                     ("losRay", True), ("rayFrom", [100, 70, 100]), ("rayTo", [99, 99, 99]),
                     ("hitLocation", [99, 99, 99]), ("hitType", "ENTITY"), ("hitPos", [True, 0, 0]),
                     ("targetEye", [99, 99, 99]), ("targetCenter", [0, 0]), ("emptyTarget", True),
                     ("count", 3), ("note", "private"), ("args", {}))
        for action in NEW_ACTIONS:
            for target in (("actorSight", "targetSight") if action == "give" else ("actorSight",)):
                for field, value in mutations:
                    with self.subTest(action=action, target=target, field=field):
                        event, item, observation = completed_action(action)
                        item[target][field] = value
                        self.assert_rejected(event, observation)
                with self.subTest(action=action, target=target, blocked=True):
                    event, item, observation = completed_action(action)
                    midpoint = [(start + end) / 2 for start, end in zip(EYE, item[target]["rayTo"])]
                    item[target].update(hitType="BLOCK", hitLocation=midpoint,
                                        hitPos=[math.floor(value) for value in midpoint])
                    self.assert_rejected(event, observation)

    def test_envelope_actor_and_recipient_radii_must_agree(self):
        """A proof cannot silently substitute a wider radius than the completion envelope or its sibling."""
        for action in NEW_ACTIONS:
            for target in ((None, "actorSight", "targetSight") if action == "give" else (None, "actorSight")):
                with self.subTest(action=action, target=target):
                    event, item, observation = completed_action(action)
                    (item if target is None else item[target])["radius"] = 25
                    self.assert_rejected(event, observation)

    def test_internally_consistent_out_of_range_entity_proofs_fail(self):
        """A truthful far-away position and distance cannot be admitted merely by setting inRadius true."""
        for action in NEW_ACTIONS:
            for target in (("actorSight", "targetSight") if action == "give" else ("actorSight",)):
                with self.subTest(action=action, target=target):
                    event, item, observation = completed_action(action)
                    item[target] = {**spatial("entity", [100.5, -60, 2.5]), "emptyTarget": False}
                    self.assert_rejected(event, observation)

    def test_item_positions_are_finite_vectors_not_block_coordinates(self):
        """Drop and pickup positions preserve fractions while rejecting malformed source and fact coordinates."""
        for action in ("drop", "pickup"):
            for position in ([1.25, -59.875, 2.625], [1, -60, 2]):
                with self.subTest(action=action, valid=position):
                    event, item, observation = completed_action(action)
                    event["pos"] = item["pos"] = position
                    self.assertEqual([], audit_o1.audit_logs([event], [observation]))
            for position in (None, [0, 1], [True, 0, 0], [0, float("nan"), 0], [0, 0, "2"]):
                with self.subTest(action=action, invalid=position):
                    event, item, observation = completed_action(action)
                    event["pos"] = item["pos"] = position
                    self.assert_rejected(event, observation)

    def test_actor_cannot_observe_own_action_but_recipient_can(self):
        """Self-witnessing is prohibited; being the give recipient alone does not prevent witnessing."""
        for action in NEW_ACTIONS:
            with self.subTest(action=action):
                event, _, observation = completed_action(action)
                observation["observer"] = "builder"
                self.assert_rejected(event, observation)
        event, _, observation = completed_action("give")
        observation["observer"] = "freight"
        self.assertEqual([], audit_o1.audit_logs([event], [observation]))

    def test_completion_cannot_follow_observation(self):
        """Matching source and fact times still fail if delivery predates completion, but equal ticks are valid."""
        for action in NEW_ACTIONS:
            with self.subTest(action=action):
                event, _, observation = completed_action(action)
                observation["gameTime"] = 9
                self.assert_rejected(event, observation)
                observation["gameTime"] = 10
                self.assertEqual([], audit_o1.audit_logs([event], [observation]))

    def test_delivery_is_once_per_observer_and_source_event(self):
        """Duplicates fail within or across polls, while a different observer can receive the same source."""
        for action in NEW_ACTIONS:
            with self.subTest(action=action):
                event, item, observation = completed_action(action)
                repeated = copy.deepcopy(observation)
                violations = audit_o1.audit_logs([event], [observation, repeated])
                self.assertTrue(any("twice" in error for error in violations), violations)
                observation["items"].append(copy.deepcopy(item))
                violations = audit_o1.audit_logs([event], [observation])
                self.assertTrue(any("twice" in error for error in violations), violations)
                observation["items"].pop()
                repeated["observer"] = "mayor"
                self.assertEqual([], audit_o1.audit_logs([event], [observation, repeated]))

    def test_same_item_entity_can_have_distinct_drop_and_pickup_sources(self):
        """Reusing an entity UUID never confuses drop/pickup IDs or two successful partial pickups."""
        events, items = [], []
        for event_id, action in enumerate(("drop", "pickup", "pickup"), 7):
            event, item, _ = completed_action(action)
            event.update(id=event_id, count=1)
            item["prov"] = f"{action}:{event_id}"
            events.append(event)
            items.append(item)
        self.assertEqual([], audit_o1.audit_logs(events, [exposure(items)]))

    def test_nested_source_facts_resolve_all_action_prefixes(self):
        """Recursive provenance checks accept only count-free facts resolving to the correct source event."""
        for action in NEW_ACTIONS:
            with self.subTest(action=action):
                event, item, _ = completed_action(action)
                fields = audit_o1.SEEN_ACTION_FACTS | ({"target"} if action == "give" else {"pos"})
                fact = {field: value for field, value in item.items() if field in fields}
                audit = audit_o1.ExposureAudit()
                audit.event(event)
                audit.nested_documents({"pending": [fact]}, 20)
                for field, value in (("prov", f"{event['type']}:99"), ("count", 1), ("item", "forged")):
                    with self.subTest(field=field):
                        with self.assertRaises(audit_o1.AuditError):
                            audit.nested_documents({"pending": [{**fact, field: value}]}, 20)
                with self.assertRaises(audit_o1.AuditError):
                    audit.nested_documents({"pending": [fact]}, 9)


if __name__ == "__main__":
    unittest.main()
