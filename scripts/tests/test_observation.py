"""Unit tests for exposure auditing, O1 orchestration, and atomic JAR installation."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from scripts import audit_o1, harness, test_o1
from scripts.tests import test_experiments


EYE = [0.5, -58.38, 0.5]
ENTITY_UUID = "00000000-0000-0000-0000-000000000001"
BOOK_PROV = "doc:book:4@v1"


def spatial(category, position, provenance=None, *, empty=False):
    """Build a spatial exposure fixture with geometry and optional provenance."""
    target = list(position) if category == "entity" else [coordinate + 0.5 for coordinate in position]
    ray_to = [position[0], EYE[1], position[2]] if category == "entity" else target
    item = {"category": category, "pos": list(position), "distance": math.dist(EYE, target),
            "radius": 24.0, "inRadius": True, "losRay": 0, "rayFrom": list(EYE), "rayTo": ray_to,
            "hitType": "MISS" if empty or category == "entity" else "BLOCK",
            "hitPos": [math.floor(value) for value in ray_to], "hitLocation": list(ray_to)}
    if provenance is not None:
        item["prov"] = provenance
    if category == "entity":
        item.update(targetEye=list(ray_to), targetCenter=[position[0] + 0.25, position[1] + 0.9, position[2] - 0.125])
    if empty:
        item["emptyTarget"] = True
    if category in {"document", "container"}:
        item.update(radius=4.5, reachOk=True)
    if category == "terrain":
        item.update(x=position[0], y=position[1], z=position[2], block="minecraft:grass_block", radius=12,
                    distance=math.dist([EYE[0], EYE[2]], [target[0], target[2]]))
    return item


def intrinsic(category="inventory_document"):
    """Build an owned or self exposure fixture without a sight ray."""
    item = {"category": category, "distance": 0, "inRadius": True, "reachOk": True,
            "losRay": None, "ownedBy": "alice"}
    if category == "self":
        item.update(name="alice", pos=[0.5, -60, 0.5], inventory=[
            {"slot": 0, "item": "minecraft:written_book", "count": 1, "doc": {"prov": BOOK_PROV}}])
    else:
        item.update(prov=BOOK_PROV, slot=0, title="notes", author="bob")
    return item


def exposure(items, kind="observe", **extra):
    """Wrap exposure items in an observation-log fixture."""
    return {"gameTime": 20, "observer": "alice", "observerEye": list(EYE), "kind": kind,
            "items": items, **extra}


class AuditTests(unittest.TestCase):
    """Exercise exposure geometry, provenance, and document content validation."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        self.events = [
            {"id": 1, "gameTime": 10, "type": "origin", "kind": "entity", "uuid": ENTITY_UUID,
             "prov": f"entity:{ENTITY_UUID}"},
            {"id": 2, "gameTime": 10, "type": "origin", "kind": "block", "pos": [0, -60, 2],
             "prov": "block:0,-60,2@base"},
            {"id": 3, "gameTime": 11, "type": "block_change", "pos": [2, -60, 1],
             "from": "minecraft:stone", "to": "minecraft:air", "actor": "bob", "cause": "bot_break"},
            {"id": 4, "gameTime": 10, "type": "doc_write", "author": "bob", "kind": "book",
             "item": "minecraft:written_book", "title": "notes", "contentSha256": "a" * 64,
             "docId": "book:4", "version": 1, "prov": BOOK_PROV},
            {"id": 5, "gameTime": 10, "type": "origin", "kind": "document", "pos": [1, -60, 2],
             "prov": "doc:sign:1,-60,2@v1"},
            {"id": 6, "gameTime": 10, "type": "origin", "kind": "container", "pos": [0, -60, 1],
             "prov": "container:0,-60,1@1"},
            {"id": 7, "gameTime": 12, "type": "utterance", "uttId": 41, "speaker": "bob",
             "pos": [10.5, EYE[1], 0.5], "text": "hello", "radius": 16.0, "listeners": ["alice"],
             "listenerEyes": {"alice": list(EYE)}},
            {"id": 8, "gameTime": 13, "type": "block_change", "pos": [3, -60, 1],
             "from": "minecraft:air", "to": "minecraft:chest", "actor": "bob", "cause": "bot_place"},
            {"id": 9, "gameTime": 10, "type": "origin", "kind": "structure", "name": "bridge_span_1",
             "prov": "structure:bridge_span_1"},
        ]
        entity = spatial("entity", [2.5, -60, 0.5], f"entity:{ENTITY_UUID}")
        entity["uuid"] = ENTITY_UUID
        baseline = spatial("block", [0, -60, 2], "block:0,-60,2@base")
        baseline["block"] = "minecraft:chest"
        document = spatial("document", [1, -60, 2], "doc:sign:1,-60,2@v1", empty=True)
        document.update(kind="sign", text={"front": ["SECRET", "", "", ""], "back": ["", "", "", ""]})
        change = spatial("change", [2, -60, 1], "change:3", empty=True)
        change.update(gameTime=11, **{"from": "minecraft:stone", "to": "minecraft:air"})
        heard = {"category": "heard", "prov": "utt:41", "uttId": 41, "speaker": "bob", "text": "hello",
                 "gameTime": 12, "distance": 10.0, "radius": 16.0, "inRadius": True, "losRay": None,
                 "speakerEye": [10.5, EYE[1], 0.5], "listenerEye": list(EYE)}
        changed = spatial("block", [3, -60, 1], "block:3,-60,1@8")
        changed["block"] = "minecraft:chest"
        container = spatial("container", [0, -60, 1], "container:0,-60,1@1")
        container["items"] = [{"slot": 0, "doc": {"prov": BOOK_PROV}}]
        structure = spatial("structure", [4, -61, 0], "structure:bridge_span_1", empty=True)
        structure.update(name="bridge_span_1", built=2, total=6)
        self.observations = [exposure([intrinsic("self"), entity, baseline, document, change, heard,
                                      intrinsic(), spatial("terrain", [2, -61, 2]), structure, changed]),
                             exposure([intrinsic()], "read_item_doc"), exposure([container], "open_container")]

    def item(self, category):
        """Return the first exposure fixture with the requested category."""
        return next(item for observation in self.observations for item in observation["items"]
                    if item["category"] == category)

    def violations(self):
        """Audit the current log fixtures and return their violations."""
        return audit_o1.audit_logs(self.events, self.observations)

    def test_all_categories_and_origin_types_are_accepted(self):
        """Verify all categories and origin types are accepted."""
        self.assertEqual(self.violations(), [])

    def test_baseline_and_external_document_version_zero_is_valid_but_not_a_write(self):
        """Verify baseline and external document version zero is valid but not a write."""
        document = self.item("document")
        self.events[4].update(prov="doc:sign:1,-60,2@v0", version=0, docId="sign:1,-60,2")
        document["prov"] = self.events[4]["prov"]
        self.assertEqual(self.violations(), [])
        self.events[3].update(type="origin", kind="document", prov="doc:book:4@v0", version=0)
        for observation in self.observations:
            for item in observation["items"]:
                if item["category"] == "inventory_document":
                    item["prov"] = self.events[3]["prov"]
        self.item("self")["inventory"][0]["doc"]["prov"] = self.events[3]["prov"]
        self.item("container")["items"][0]["doc"]["prov"] = self.events[3]["prov"]
        self.assertEqual(self.violations(), [])
        self.events[3].update(type="doc_write", kind="book")
        self.assertTrue(self.violations())

    def test_container_document_metadata_must_match_its_parent_slot(self):
        """Verify container document metadata must match its parent slot."""
        document = spatial("document", [0, -60, 1], BOOK_PROV)
        document.update(slot=0, containerProv="container:0,-60,1@1", title="notes", author="bob")
        self.observations[-1]["items"].append(document)
        self.assertEqual(self.violations(), [])
        for field, value in (("containerProv", "container:0,-60,1@2"), ("slot", 1),
                             ("slot", True), ("category", "inventory_document"), ("pos", [1, -60, 1])):
            with self.subTest(field=field):
                original = copy.deepcopy(document)
                document[field] = value
                self.assertTrue(self.violations())
                document.clear()
                document.update(original)
        parent = self.observations[-1]["items"].pop(0)
        self.assertTrue(self.violations())
        self.observations[-1]["items"].insert(0, parent)
        parent["items"][0]["doc"]["prov"] = "doc:sign:1,-60,2@v1"
        self.assertTrue(self.violations())

    def test_all_fixed_block_face_rays_and_both_entity_rays_are_valid(self):
        """Verify all fixed block face rays and both entity rays are valid."""
        block = self.item("block")
        center = [coordinate + 0.5 for coordinate in block["pos"]]
        for ray in range(7):
            with self.subTest(ray=ray):
                target = list(center)
                if ray:
                    target[(ray - 1) // 2] += -0.49 if ray % 2 else 0.49
                block.update(losRay=ray, rayTo=target, hitLocation=list(target))
                self.assertEqual(self.violations(), [])
        entity = self.item("entity")
        center = entity["targetCenter"]
        entity.update(losRay=1, rayTo=list(center), hitLocation=list(center),
                      hitPos=[math.floor(coordinate) for coordinate in center])
        self.assertEqual(self.violations(), [])

    def test_entity_proof_requires_both_exact_api_targets(self):
        """Verify entity proof requires both exact api targets."""
        for field in ("targetEye", "targetCenter"):
            for value in (None, [0, 0], [True, 0, 0], [float("nan"), 0, 0]):
                with self.subTest(field=field, value=value):
                    original = copy.deepcopy(self.observations)
                    self.item("entity")[field] = value
                    self.assertTrue(self.violations())
                    self.observations = original
        entity = self.item("entity")
        entity["targetEye"] = [2.75, -58.38, 0.75]
        self.assertTrue(self.violations())
        entity.update(rayTo=list(entity["targetEye"]), hitLocation=list(entity["targetEye"]))
        self.assertEqual(self.violations(), [])

    def test_nested_references_cannot_hide_behind_an_unrecognized_field(self):
        """Verify nested references cannot hide behind an unrecognized field."""
        self.item("self")["extra"] = {"prov": "doc:missing@v0"}
        self.assertTrue(self.violations())

    def test_nested_document_requires_provenance_even_without_a_separate_exposure(self):
        """Verify nested document requires provenance even without a separate exposure."""
        self.item("self")["inventory"][0]["doc"] = {}
        self.assertTrue(self.violations())

    def test_empty_target_exception_cannot_bypass_a_real_wall_hit(self):
        """Verify empty target exception cannot bypass a real wall hit."""
        for category in ("document", "change"):
            with self.subTest(category=category):
                original = copy.deepcopy(self.observations)
                item = self.item(category)
                midpoint = [(start + end) / 2 for start, end in zip(item["rayFrom"], item["rayTo"])]
                item.update(emptyTarget=True, hitType="BLOCK", hitLocation=midpoint,
                            hitPos=[math.floor(coordinate) for coordinate in midpoint])
                self.assertTrue(self.violations())
                self.observations = original

    def test_speech_uses_captured_eyes_not_listener_current_position(self):
        """Verify speech uses captured eyes not listener current position."""
        self.observations = [exposure([self.item("heard")], observerEye=[100, 100, 100])]
        self.assertEqual(self.violations(), [])

    def test_empty_visual_targets_require_explicit_label(self):
        """Verify empty visual targets require explicit label."""
        for category in ("block", "document", "change", "container", "terrain"):
            with self.subTest(category=category):
                original = copy.deepcopy(self.observations)
                item = self.item(category)
                item["hitType"] = "MISS"
                item.pop("emptyTarget", None)
                self.assertTrue(self.violations())
                item["emptyTarget"] = True
                self.assertEqual(self.violations(), [])
                self.observations = original

    def test_each_spatial_category_rejects_false_flags_and_forged_geometry(self):
        """Verify each spatial category rejects false flags and forged geometry."""
        for category in audit_o1.SPATIAL:
            for field, value in (("inRadius", False), ("inRadius", 1), ("distance", 1000),
                                 ("distance", -1), ("distance", float("nan")), ("radius", 0.1),
                                 ("radius", True), ("losRay", True), ("losRay", 99),
                                 ("rayFrom", [99, 99, 99]), ("rayTo", [99, -58, 99]),
                                 ("hitLocation", [99, 99, 99]), ("hitType", "ENTITY"),
                                 ("hitPos", [True, 0, 0]), ("emptyTarget", "true")):
                with self.subTest(category=category, field=field, value=value):
                    original = copy.deepcopy(self.observations)
                    self.item(category)[field] = value
                    self.assertTrue(self.violations())
                    self.observations = original

    def test_strict_blocks_cannot_claim_another_blocks_hit(self):
        """Verify strict blocks cannot claim another blocks hit."""
        self.item("block")["hitPos"] = [1, -60, 2]
        self.assertTrue(self.violations())

    def test_entity_block_obstruction_is_rejected(self):
        """Verify entity block obstruction is rejected."""
        item = self.item("entity")
        item.update(hitType="BLOCK", hitPos=[1, -59, 0], hitLocation=[1.5, EYE[1], 0.5])
        self.assertTrue(self.violations())

    def test_documents_and_containers_need_reach_and_geometric_distance(self):
        """Verify documents and containers need reach and geometric distance."""
        for category in ("document", "container"):
            for field, value in (("reachOk", False), ("reachOk", 1), ("radius", 1), ("distance", 0)):
                with self.subTest(category=category, field=field):
                    original = copy.deepcopy(self.observations)
                    self.item(category)[field] = value
                    self.assertTrue(self.violations())
                    self.observations = original

    def test_terrain_is_horizontal_ambient_but_still_requires_los(self):
        """Verify terrain is horizontal ambient but still requires los."""
        terrain = self.item("terrain")
        self.assertNotEqual(terrain["distance"], math.dist(EYE, terrain["rayTo"]))
        for field, value in (("y", 999), ("prov", "block:2,-61,2@base"),
                             ("distance", math.dist(EYE, terrain["rayTo"]))):
            with self.subTest(field=field):
                original = copy.deepcopy(self.observations)
                self.item("terrain")[field] = value
                self.assertTrue(self.violations())
                self.observations = original

    def test_self_identity_and_nested_inventory_provenance(self):
        """Verify self identity and nested inventory provenance."""
        for field, value in (("name", "bob"), ("ownedBy", "bob"), ("distance", 1),
                             ("reachOk", False), ("losRay", 0), ("prov", "entity:invented")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.observations)
                self.item("self")[field] = value
                self.assertTrue(self.violations())
                self.observations = original
        self.item("self")["inventory"][0]["doc"]["prov"] = "doc:missing@v1"
        self.assertTrue(self.violations())

    def test_drop_skill_progress_names_its_item_entity_itementity(self):
        """Verify a drop skill's nested progress resolves by itemEntity, and a different entity is still rejected."""
        progress = {"count": 1, "eventId": 1, "item": "minecraft:iron_pickaxe", "itemEntity": ENTITY_UUID,
                    "prov": f"entity:{ENTITY_UUID}"}
        self.item("self")["skill"] = {"skill": "drop", "state": "succeeded", "progress": progress}
        self.assertEqual(self.violations(), [])
        progress["itemEntity"] = "other"
        self.assertTrue(self.violations())

    def test_inventory_documents_need_ownership_and_zero_distance(self):
        """Verify inventory documents need ownership and zero distance."""
        for field, value in (("ownedBy", "bob"), ("slot", True), ("distance", 1), ("reachOk", False), ("losRay", 0)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.observations)
                self.item("inventory_document")[field] = value
                self.assertTrue(self.violations())
                self.observations = original

    def test_all_provenance_categories_reject_missing_or_wrong_origin(self):
        """Verify all provenance categories reject missing or wrong origin."""
        for category in ("entity", "block", "document", "change", "heard", "container", "inventory_document"):
            wrong_kind = "block:0,-60,2@base" if category in {"document", "inventory_document"} else BOOK_PROV
            for provenance in (None, "made-up", wrong_kind):
                with self.subTest(category=category, provenance=provenance):
                    original = copy.deepcopy(self.observations)
                    self.item(category)["prov"] = provenance
                    self.assertTrue(self.violations())
                    self.observations = original

    def test_origin_metadata_and_changed_block_payload_must_match(self):
        """Verify origin metadata and changed block payload must match."""
        for category, field, value in (("entity", "uuid", "other"), ("block", "pos", [0, -60, 3]),
                                       ("document", "pos", [1, -60, 3]), ("change", "to", "minecraft:stone"),
                                       ("change", "gameTime", 12), ("container", "pos", [1, -60, 1]),
                                       ("structure", "name", "bridge_span_2")):
            with self.subTest(category=category, field=field):
                original = copy.deepcopy(self.observations)
                self.item(category)[field] = value
                self.assertTrue(self.violations())
                self.observations = original
        self.observations[0]["items"][-1]["block"] = "minecraft:air"
        self.assertTrue(self.violations())

    def test_future_provenance_and_duplicate_registrations_fail(self):
        """Verify future provenance and duplicate registrations fail."""
        self.events[0]["gameTime"] = 21
        self.assertTrue(self.violations())
        self.events[0]["gameTime"] = 10
        for extra in (copy.deepcopy(self.events[0]), {**self.events[0], "id": 99}):
            with self.subTest(extra=extra):
                self.events.append(extra)
                self.assertTrue(self.violations())
                self.events.pop()

    def test_malformed_events_fail_closed(self):
        """Verify malformed events fail closed."""
        for index, field, value in ((0, "id", True), (0, "uuid", "invalid"), (0, "kind", "container"),
                                    (1, "pos", [1, -60, 2]), (3, "contentSha256", "x" * 64),
                                    (3, "version", 2), (5, "prov", "container:0,-60,1@base"),
                                    (6, "uttId", True), (6, "listeners", ["alice", "alice"]),
                                    (6, "listeners", ["bob"]), (6, "listenerEyes", {}),
                                    (6, "listenerEyes", {"alice": [100, EYE[1], 0.5]})):
            with self.subTest(index=index, field=field):
                original = copy.deepcopy(self.events)
                self.events[index][field] = value
                self.assertTrue(self.violations())
                self.events = original

    def test_speech_cannot_change_payload_radius_eyes_or_event_counter(self):
        """Verify speech cannot change payload radius eyes or event counter."""
        for field, value in (("uttId", 7), ("speaker", "carol"), ("text", "forged"), ("gameTime", 13),
                             ("radius", 99), ("distance", 0), ("distance", 20), ("distance", True),
                             ("speakerEye", [0, 0, 0]), ("listenerEye", [0, 0, 0]), ("losRay", 0)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.observations)
                self.item("heard")[field] = value
                self.assertTrue(self.violations())
                self.observations = original

    def test_speech_listener_membership_and_queue_drain(self):
        """Verify speech listener membership and queue drain."""
        heard = self.item("heard")
        for observer in ("bob", "carol"):
            self.assertTrue(audit_o1.audit_logs(self.events, [exposure([heard], observer=observer)]))
        self.observations.append(exposure([copy.deepcopy(heard)]))
        self.assertTrue(any("twice" in error for error in self.violations()))

    def test_bad_exposure_envelopes_unknown_categories_and_missing_proofs(self):
        """Verify bad exposure envelopes unknown categories and missing proofs."""
        for field, value in (("gameTime", True), ("observer", ""), ("observerEye", [0, 0]),
                             ("kind", "unknown"), ("items", {})):
            with self.subTest(field=field):
                original = copy.deepcopy(self.observations)
                self.observations[0][field] = value
                self.assertTrue(self.violations())
                self.observations = original
        for field in ("category", "prov", "distance", "inRadius", "losRay", "rayFrom", "rayTo", "hitPos",
                      "hitLocation"):
            with self.subTest(missing=field):
                original = copy.deepcopy(self.observations)
                self.item("block").pop(field)
                self.assertTrue(self.violations())
                self.observations = original
        self.observations[1]["items"] = [self.item("block")]
        self.assertTrue(self.violations())

    def test_empty_missing_malformed_and_duplicate_key_logs_are_not_zero_violations(self):
        """Verify empty missing malformed and duplicate key logs are not zero violations."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.assertTrue(audit_o1.audit_run(run))
            logs = run / "world/simharness"
            logs.mkdir(parents=True)
            events_path = logs / "events.jsonl"
            observations_path = logs / "observations.jsonl"
            observations_path.write_text("\n".join(json.dumps(value) for value in self.observations) + "\n")
            for payload in ("", "{}\n", "[]\n", "{bad\n", '{"id":1,"id":2}\n', '{"distance":NaN}\n', "\n"):
                with self.subTest(payload=payload):
                    events_path.write_text(payload)
                    self.assertTrue(audit_o1.audit_run(run))
            events_path.write_text("\n".join(json.dumps(value) for value in self.events) + "\n")
            self.assertEqual(audit_o1.audit_run(run), [])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit_o1.main([str(run)]), 0)
            self.assertIn("PASS O1 audit: 0 violations", output.getvalue())
            observations_path.write_text("")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit_o1.main([str(run)]), 1)
            self.assertIn("FAIL O1 audit:", output.getvalue())


class ObservationScheduleTests(unittest.TestCase):
    """Exercise observation timing at absolute routine boundaries."""
    setUp = test_experiments.ExperimentTests.setUp
    fake_setup = test_experiments.ExperimentTests.fake_setup
    fake_clone = test_experiments.ExperimentTests.fake_clone
    base = test_experiments.ExperimentTests.base
    invoke = test_experiments.ExperimentTests.invoke
    def test_e1_observes_once_at_existing_action_boundary_and_preserves_all_steps(self):
        """Verify e1 observes once at existing action boundary and preserves all steps."""
        self.base()
        from scripts import test_e1
        self.invoke(test_e1)
        for name in ("e1_a", "e1_b"):
            events = [event for event in self.events if event[1] == name]
            observations = [event for event in events if event[0] == "observe"]
            self.assertEqual([event[2] for event in observations], list(range(300, 24100, 400)))
            self.assertTrue(all(event[3] == {"name": "alice"} for event in observations))
            self.assertEqual([event[0] for event in events if event[0] in ("lockstep", "await_entities", "recording")],
                             ["lockstep", "await_entities", "recording"])
            self.assertEqual(next(event[3] for event in events if event[0] == "recording"), {"enabled": True})
            current = 100
            expected_steps = []
            while current < 24100:
                after = min(24100, harness.routine.next_action_time(current))
                expected_steps.append((current, after - current))
                current = after
            self.assertEqual([(event[2], event[3]["n"]) for event in events if event[0] == "step"], expected_steps)

    def test_e2_observes_the_identical_absolute_times_after_restore(self):
        """Verify e2 observes the identical absolute times after restore."""
        from scripts import test_e2
        self.base()
        with patch.object(test_e2, "setup_server", side_effect=self.fake_setup):
            self.invoke(test_e2)
        observed_a = [event[2] for event in self.events if event[:2] == ("observe", "e2_a")]
        observed_c = [event[2] for event in self.events if event[:2] == ("observe", "e2_c")]
        self.assertEqual(observed_a, list(range(300, 24100, 400)))
        self.assertEqual(observed_c, [game_time for game_time in observed_a if game_time >= 12100])

    def test_harness_hash_component_is_required(self):
        """Verify harness hash component is required."""
        value = {"gameTime": 20, **{key: "a" * 64 for key in harness.HASH_KEYS}}
        self.assertIn("harness", value)
        value.pop("harness")
        with self.assertRaises(harness.ProtocolError):
            harness.hash_components(value, 20)

    def test_recording_can_be_deferred_for_o1_without_ticking(self):
        """Verify recording can be deferred for o1 without ticking."""
        client = Mock()
        client.request.return_value = {"gameTime": 100}
        self.assertEqual(harness.enable_lockstep(client, recording=False), 100)
        self.assertEqual(client.request.call_args_list, [call("lockstep", {"enabled": True}),
            call("await_entities", harness.FORCELOADED_CHUNKS), call("recording", {"enabled": False}), call("ping")])


class O1Control(harness.Client):
    """Simulate visibility, documents, speech, and ACL responses for O1 tests."""
    def __init__(self, directory):
        """Initialize this test fixture without launching a real server."""
        self.directory = directory
        self.calls = []
        self.current = 100
        self.positions = {}
        self.blocks = {}
        self.walled = False
        self.recording = False
        self.log = []
        self.changes = []
        self.cursors = {}
        self.queues = {}
        self.pending_speech = None
        self.sign_text = None
        self.book = None
        self.allowed = "*"
        self.master_keys = set()

    def change(self, position, block, actor):
        """Record a fake block change with its position, block, and actor."""
        previous = self.blocks.get(tuple(position), "minecraft:air")
        self.blocks[tuple(position)] = block
        if self.recording:
            event = {"type": "block_change", "id": len(self.log) + 1, "gameTime": self.current,
                     "pos": list(position), "from": previous, "to": block, "actor": actor, "cause": "test"}
            self.log.append(event)
            self.changes.append({"prov": f"change:{event['id']}", "pos": list(position),
                                 "from": previous, "to": block, "gameTime": self.current})

    def visible(self, observer, position):
        """Return whether fixture geometry exposes a position to the observer."""
        return not (self.walled and self.positions[observer][2] < 3 < position[2])

    def request(self, command, args=None):
        """Handle a scripted control request using the enclosing test fixture."""
        args = args or {}
        self.calls.append((command, copy.deepcopy(args), self.current))
        if command in {"ping", "await_entities"}:
            return {"gameTime": self.current}
        if command in {"lockstep", "config_set"}:
            return {}
        if command == "recording":
            self.recording = args["enabled"]
            return {}
        if command == "command":
            parts = args["command"].split()
            if parts[0] == "fill" and parts[1:7] == ["-3", "-61", "3", "3", "-57", "3"]:
                self.walled = parts[7] == "minecraft:stone"
            elif parts[0] == "setblock":
                self.change([int(value) for value in parts[1:4]], parts[4], "system")
            elif parts[0] == "tp":
                self.positions[parts[1]] = [float(value) for value in parts[2:5]]
            return {"value": 1}
        if command == "spawn_bot":
            self.positions[args["name"]] = [args[key] for key in ("x", "y", "z")]
            return {}
        if command == "bot_state":
            return {"pos": self.positions[args["name"]], "uuid": f"uuid-{args['name']}"}
        if command == "step":
            self.current += args["n"]
            if self.pending_speech is not None and self.current >= self.pending_speech["end"]:
                speech = self.pending_speech
                for name, position in self.positions.items():
                    if name != speech["name"] and math.dist(speech["pos"], position) <= 8:
                        self.queues.setdefault(name, []).append({"prov": "utt:1", "speaker": speech["name"],
                                                                 "text": speech["text"]})
                self.pending_speech = None
            return {"gameTime": self.current}
        if command == "write_sign":
            self.sign_text = {"front": args["front"], "back": args.get("back", ["", "", "", ""])}
            return {"prov": "doc:sign:0,-60,7@v1"}
        if command == "observe":
            name = args["name"]
            blocks = [{"pos": list(position), "block": block} for position, block in self.blocks.items()
                      if block in {"minecraft:chest", "minecraft:oak_sign"} and self.visible(name, position)]
            documents = []
            observer_eye = list(self.positions[name])
            observer_eye[1] += 1.62
            if self.sign_text is not None and math.dist(observer_eye, [0.5, -59.5, 7.5]) <= 4.5:
                documents.append({"pos": test_o1.SIGN, "text": self.sign_text})
            heard = self.queues.pop(name, [])
            changes = [change for change in self.changes[self.cursors.get(name, 0):]
                       if self.visible(name, change["pos"])]
            self.cursors[name] = len(self.changes)
            return {"blocks": blocks, "documents": documents, "heard": heard, "changes": changes,
                    "entities": [{"name": other, "uuid": f"uuid-{other}"}
                                 for other, position in self.positions.items()
                                 if other != name and self.visible(name, position)]}
        if command == "skill_start" and args.get("dry_run"):
            return {"feasible": True, "estimate": {"ticks": 7, "materials": []}}
        if command == "speak":
            self.pending_speech = {**args, "pos": list(self.positions[args["name"]]), "end": self.current + 7}
            return {"accepted": True}
        if command == "bot_break":
            self.change([args[key] for key in ("x", "y", "z")], "minecraft:air", args["name"])
            return {}
        if command == "give":
            return {}
        if command == "write_book":
            document_id = f"book:{len(self.log) + 1}"
            self.book = {"prov": f"doc:{document_id}@v1", "title": args["title"],
                         "author": args["name"], "pages": list(args["pages"])}
            self.log.append({"type": "doc_write", "id": len(self.log) + 1, "gameTime": self.current,
                             "author": args["name"], "prov": self.book["prov"]})
            return {"prov": self.book["prov"]}
        if command == "read_item_doc":
            return copy.deepcopy(self.book)
        if command == "acl_set":
            self.allowed = args["allowed"]
            return {}
        if command == "bot_flag":
            if args["master_key"]:
                self.master_keys.add(args["name"])
            return {}
        if command == "open_container":
            if self.allowed != "*" and args["name"] not in self.allowed and args["name"] not in self.master_keys:
                raise harness.CommandError("ACL denied")
            return {"prov": "container:0,-60,6@1", "items": []}
        if command == "save_snapshot":
            logs = self.directory / "world/simharness"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in self.log))
            return {}
        raise AssertionError(f"unexpected O1 command: {command}")


class InstallCurrentModTests(unittest.TestCase):
    """Exercise atomic JAR refresh and preservation of run files."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        (self.base / "mods").mkdir(parents=True)
        (self.base / "world/simharness").mkdir(parents=True)
        (self.base / "world/level.dat").write_bytes(b"saved world")
        (self.base / "world/simharness/events.jsonl").write_bytes(b"saved provenance log\n")
        (self.base / "mods/simharness-0.1.0.jar").write_bytes(b"old M0 harness")
        (self.base / "mods/fabric-api.jar").write_bytes(b"fabric api")
        self.run = self.root / "o1"
        harness.shutil.copytree(self.base, self.run)
        self.target = self.run / "mods/simharness-0.1.0.jar"
        self.built_mod = self.root / "build/simharness-0.1.0.jar"
        self.built_mod.parent.mkdir()
        self.built_mod.write_bytes(b"current M1a harness")
        mod_patch = patch.object(harness, "MOD_JAR", self.built_mod)
        mod_patch.start()
        self.addCleanup(mod_patch.stop)

    def assert_preserved(self):
        """Assert that the original JAR and unrelated files remain unchanged."""
        self.assertEqual((self.base / "mods/simharness-0.1.0.jar").read_bytes(), b"old M0 harness")
        self.assertEqual((self.base / "world/level.dat").read_bytes(), b"saved world")
        self.assertEqual((self.run / "world/level.dat").read_bytes(), b"saved world")
        self.assertEqual((self.run / "world/simharness/events.jsonl").read_bytes(), b"saved provenance log\n")
        self.assertEqual((self.run / "mods/fabric-api.jar").read_bytes(), b"fabric api")
        self.assertEqual(sorted(path.name for path in (self.run / "mods").iterdir()),
                         ["fabric-api.jar", "simharness-0.1.0.jar"])

    def prepare_server(self):
        """Create the local server files required by the lifecycle fixture."""
        java_home = self.root / "jdk"
        (java_home / "bin").mkdir(parents=True)
        (java_home / "bin/java").touch()
        (self.run / "fabric-server-launch.jar").touch()
        return java_home

    def test_current_build_replaces_only_cloned_mod_not_world_base_or_other_mods(self):
        """Verify current build replaces only cloned mod not world base or other mods."""
        harness.install_current_mod(self.run)
        self.assertEqual(self.target.read_bytes(), b"current M1a harness")
        self.assert_preserved()

    def test_install_is_safe_even_if_the_old_mod_is_a_hardlink_or_symlink(self):
        """Verify install is safe even if the old mod is a hardlink or symlink."""
        for link_type in ("hard", "symbolic"):
            with self.subTest(link_type=link_type):
                self.target.unlink()
                source = self.base / "mods/simharness-0.1.0.jar"
                if link_type == "hard":
                    harness.os.link(source, self.target)
                else:
                    self.target.symlink_to(source)
                harness.install_current_mod(self.run)
                self.assertFalse(self.target.is_symlink())
                self.assertEqual(self.target.read_bytes(), b"current M1a harness")
                self.assert_preserved()

    def test_copy_or_replace_failure_preserves_the_previous_jar_and_cleans_staging(self):
        """Verify copy or replace failure preserves the previous jar and cleans staging."""
        def failed_copy(source, target):
            """Raise the scripted copy failure during atomic-installation validation."""
            Path(target).write_bytes(b"partial JAR")
            raise OSError("copy failed")

        for target, replacement in (("copy2", failed_copy), ("replace", OSError("replace failed"))):
            with self.subTest(target=target):
                owner = harness.shutil if target == "copy2" else harness.os
                with patch.object(owner, target, side_effect=replacement):
                    with self.assertRaises(OSError):
                        harness.install_current_mod(self.run)
                self.assertEqual(self.target.read_bytes(), b"old M0 harness")
                self.assert_preserved()

    def test_missing_build_fails_before_socket_log_or_jvm_start(self):
        """Verify missing build fails before socket log or jvm start."""
        java_home = self.prepare_server()
        with patch.object(harness, "JAVA_HOME", java_home), \
                patch.object(harness, "MOD_JAR", self.root / "absent.jar"), \
                patch.object(harness.socket, "socket") as socket_factory, \
                patch.object(harness.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(FileNotFoundError, "Build the harness mod first"):
                harness.Server(self.run).start()
        socket_factory.assert_not_called()
        popen.assert_not_called()
        self.assertFalse((self.run / "server.log").exists())
        self.assertEqual(self.target.read_bytes(), b"old M0 harness")
        self.assert_preserved()

    def test_server_refreshes_mod_before_attempting_process_creation(self):
        """Verify server refreshes mod before attempting process creation."""
        java_home = self.prepare_server()

        def launch(*args, **kwargs):
            """Check startup ordering and apply the scripted launch behavior."""
            self.assertEqual(self.target.read_bytes(), b"current M1a harness")
            raise OSError("mock launch; no server started")

        with patch.object(harness, "JAVA_HOME", java_home), patch.object(harness.socket, "socket"), \
                patch.object(harness.subprocess, "Popen", side_effect=launch) as popen:
            server = harness.Server(self.run)
            with self.assertRaisesRegex(OSError, "mock launch"):
                server.start()
        popen.assert_called_once()
        self.assertIsNone(server.process)
        self.assertIsNone(server._log)
        self.assert_preserved()

    def test_missing_or_shared_mods_directory_is_rejected(self):
        """Verify missing or shared mods directory is rejected."""
        directory = self.root / "bad-run"
        directory.mkdir()
        with self.assertRaisesRegex(harness.HarnessError, "own mods directory"):
            harness.install_current_mod(directory)
        (directory / "mods").symlink_to(self.base / "mods", target_is_directory=True)
        with self.assertRaisesRegex(harness.HarnessError, "own mods directory"):
            harness.install_current_mod(directory)
        self.assert_preserved()


class O1GateTests(unittest.TestCase):
    """Exercise O1 reporting and offline audit failures without a server."""
    def setUp(self):
        """Reset the fixtures used by each test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runs = Path(temporary.name)
        (self.runs / "base").mkdir()
        self.control = O1Control(self.runs / "o1")

    def invoke(self, audit_status=0):
        """Run the gate with patched dependencies and capture its output."""
        server = contextlib.nullcontext(Mock(client=self.control))
        with patch.object(harness, "clone_apfs", side_effect=lambda source, target: target.mkdir()) as clone, \
                patch.object(harness, "Server", return_value=server) as factory, \
                patch.object(test_o1.audit_o1, "main", return_value=audit_status) as audit, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            status = test_o1.main(["--runs-dir", str(self.runs), "--port", "25601", "--request-timeout", "3"])
        clone.assert_called_once_with(self.runs / "base", self.runs / "o1")
        factory.assert_called_once_with(self.runs / "o1", port=25601, base_seed=harness.BASE_SEED,
                                       game_port=None, heap=None, startup_timeout=180.0, request_timeout=3.0)
        audit.assert_called_once_with([str(self.runs / "o1")])
        return status, output.getvalue()

    def test_t1_through_t7_and_recording_setup_order_without_a_server(self):
        """Verify t1 through t7 and recording setup order without a server."""
        status, output = self.invoke()
        self.assertEqual(status, 0, output)
        for label, _test in test_o1.TESTS:
            self.assertIn(f"PASS {label}\n", output)
        self.assertTrue(output.endswith("PASS O1\n"))
        calls = self.control.calls
        self.assertEqual([command for command, _args, _time in calls[:4]],
                         ["lockstep", "await_entities", "recording", "ping"])
        recording_on = next(index for index, (command, args, _time) in enumerate(calls)
                            if command == "recording" and args["enabled"])
        spawns = [index for index, (command, _args, _time) in enumerate(calls) if command == "spawn_bot"]
        self.assertEqual(len(spawns), 3)
        self.assertLess(max(spawns), recording_on)
        self.assertEqual([args["n"] for command, args, _time in calls if command == "step"], [1, 6, 1, 1])

    def test_audit_violation_fails_the_overall_gate(self):
        """Verify audit violation fails the overall gate."""
        status, output = self.invoke(audit_status=1)
        self.assertEqual(status, 1)
        self.assertTrue(output.endswith("FAIL O1\n"))

    def test_test_failure_does_not_skip_other_tests_or_the_audit(self):
        """Verify test failure does not skip other tests or the audit."""
        original = test_o1.TESTS
        failed = Mock(side_effect=harness.HarnessError("negative fixture"))
        tests = ((original[0][0], failed), *original[1:])
        with patch.object(test_o1, "TESTS", tests):
            status, output = self.invoke()
        self.assertEqual(status, 1)
        self.assertIn("FAIL T1 hidden chest: negative fixture", output)
        self.assertIn("PASS T7 ACL", output)
        self.assertTrue(output.endswith("FAIL O1\n"))

    def test_existing_o1_run_is_never_overwritten_or_started(self):
        """Verify existing o1 run is never overwritten or started."""
        directory = self.runs / "o1"
        directory.mkdir()
        (directory / "keep").write_text("untouched")
        with patch.object(harness, "clone_apfs") as clone, patch.object(harness, "Server") as server:
            with self.assertRaises(FileExistsError):
                test_o1.main(["--runs-dir", str(self.runs)])
        clone.assert_not_called()
        server.assert_not_called()
        self.assertEqual((directory / "keep").read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
