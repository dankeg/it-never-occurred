"""Server-free M7c position search, recorded provenance and fixture metadata tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import harness
from scripts.build_village import ScenarioConfig
from scripts.provenance import CodeState, FixtureMetadata
from scripts.stage_incident import (IncidentRecord, WitnessWindow, WitnessWindowError,
                                    find_witness_windows, standing_on_deck)
from orchestrator.scenario import ScenarioManifest

WINDOW = WitnessWindow(3000, 3200)


WITNESSES = ["adult_01", "adult_02"]


def position_sample(tick: int, eye: list[float]) -> dict:
    """Build a small recorded-position row without emulating a server or client."""
    return {"gameTime": tick, "witnesses": {
        witness: {"pos": [eye[0], eye[1] - 1.62, eye[2]], "observerEye": list(eye), "activity": "travel_work",
                  "trip": {"from": "east_home_1", "to": "workshop_1", "arriveTick": None}}
        for witness in WITNESSES}}


def incident_records(variant: str = "sabotage") -> tuple[dict, list[dict], dict, dict]:
    """Return literal protocol-shaped events, testimony and news, not a mock world."""
    manifest = ScenarioConfig.create(variant, 17, pre_incident=True).to_dict()
    events = [{"id": 400 + index, "type": "block_change", "gameTime": 3000 + index,
               "pos": support["pos"], "from": support["block"], "to": "minecraft:water",
               "actor": "envoy" if variant == "sabotage" else "world",
               "cause": "bot_action" if variant == "sabotage" else "world"}
              for index, support in enumerate(manifest["evidence"]["supports"])]
    last = 400 + len(manifest["evidence"]["supports"]) - 1  # the last support change triggers the collapse
    collapse_time = 3000 + len(manifest["evidence"]["supports"]) - 1  # in the tick of the last removal
    deck = sorted([block["pos"] for section in ScenarioManifest(manifest).structures()
                   if section["name"].startswith("bridge_span_") for block in section["blocks"]])
    deck_changes = [{"id": 410 + index, "type": "block_change", "gameTime": collapse_time,
                     "pos": position, "from": "minecraft:oak_planks", "to": "minecraft:air",
                     "actor": "world", "cause": "collapse", "causeEventId": last}
                    for index, position in enumerate(deck)]
    events += deck_changes + [{"id": 450, "type": "collapse", "gameTime": collapse_time, "name": "west_bridge",
                               "actor": "world", "causeEventId": last,
                               "changes": [change["id"] for change in deck_changes]}]
    testimony, news = {}, {}
    for witness in WITNESSES:
        testimony[witness] = [{"id": 700, "type": "npc_observation", "gameTime": 3020, "npc": witness,
                               "observer": "npc:" + witness, "observerEye": [-10, -58, 0],
                               "changes": [{**event, "losRay": 0} for event in deck_changes],
                               "entities": ([{"prov": "entity:actual-envoy", "name": "envoy", "losRay": 0,
                                              "pos": [-6.5, -60, 0.99]}] if variant == "sabotage" else [])}]
        news[witness] = [{"newsId": "change:410", "originEventId": "change:410", "firsthand": True,
                          "kind": "bridge_or_port_change", "subject": "the bridge", "learnedAt": 3020,
                          "present": ["envoy"] if variant == "sabotage" else []}]
    if variant == "sabotage":
        events += [{"id": 310, "type": "purchase", "gameTime": 500, "buyer": "envoy",
                    "shop": "container:-46,-60,-13", "item": "minecraft:iron_pickaxe", "count": 1, "cost": 8},
                   {"id": 500, "type": "drop", "gameTime": 3100, "bot": "envoy",
                    "item": "minecraft:iron_pickaxe", "count": 1, "itemEntity": "actual-tool",
                    "thrower": "actual-envoy", "prov": "entity:actual-tool", "pos": [-8.5, -60, -3.5]}]
    return manifest, events, testimony, news


class WitnessWindowTests(unittest.TestCase):
    """Search real-position-shaped samples without predicting NPC motion or LOS."""

    def test_inclusive_three_dimensional_radius_and_day_zero(self):
        """Exact radius qualifies; any deck block suffices, but excessive height and day 1 do not."""
        targets = [[0, 0, 0]]
        samples = [position_sample(1000, [24.5, 0.5, 0.5]),
                   position_sample(1020, [0.5, 24.5, 0.5]),
                   position_sample(1040, [0.5, 24.51, 0.5]),
                   position_sample(24_000, [0.5, 0.5, 0.5])]
        self.assertEqual(find_witness_windows(samples, WITNESSES, targets, 24), [WitnessWindow(1000, 1020)])
        self.assertEqual(find_witness_windows(samples, WITNESSES, targets + [[100, 0, 0]], 24),
                         [WitnessWindow(1000, 1020)])
        self.assertEqual(find_witness_windows(samples, WITNESSES, [], 24), [])

    def test_window_excludes_witnesses_standing_on_deck(self):
        """A visible commute over the deck is not a safe collapse window; off-deck approaches still qualify."""
        samples = [position_sample(tick, [0.5, 2.62, 0.5]) for tick in (1000, 1020)]
        self.assertEqual(find_witness_windows(samples, WITNESSES, [[0, 0, 0]], 24), [])
        samples.append(position_sample(1040, [-0.1, 2.62, 0.5]))
        self.assertEqual(find_witness_windows(samples, WITNESSES, [[0, 0, 0]], 24), [WitnessWindow(1040, 1040)])
        self.assertTrue(standing_on_deck([0.5, 1, 0.5], [[0, 0, 0]]))
        self.assertFalse(standing_on_deck([1, 1, 0.5], [[0, 0, 0]]))
        self.assertFalse(standing_on_deck([0.5, -1, 0.5], [[0, 0, 0]]))

    def test_witnesses_need_not_be_in_range_together(self):
        """Each witness's own intervals count; nearby intervals of different witnesses join into one window."""
        samples = [position_sample(tick, [0.5, 2, 0.5]) for tick in (1000, 1020, 1040, 1080, 1100, 1120)]
        samples[1]["witnesses"]["adult_02"]["observerEye"] = [100, 2, 0]
        samples[4]["witnesses"]["adult_01"]["trip"]["from"] = "east_home_2"
        samples[5]["witnesses"]["adult_02"].pop("observerEye")
        self.assertEqual(find_witness_windows(list(reversed(samples)), WITNESSES, [[0, 0, 0]], 24),
                         [WitnessWindow(1000, 1120)])
        self.assertEqual(find_witness_windows(samples, WITNESSES, [[0, 0, 0]], 24, max_gap=0),
                         [WitnessWindow(1000, 1040), WitnessWindow(1080, 1100)])

    def test_arrived_and_wrong_activity_are_not_the_commute(self):
        """A body at the workshop or at home cannot stand in for sight on the approach."""
        samples = [position_sample(1000, [0, 0, 0]), position_sample(1020, [0, 0, 0])]
        samples[0]["witnesses"]["adult_01"]["trip"]["arriveTick"] = 999
        samples[1]["witnesses"]["adult_02"]["activity"] = "home"
        self.assertEqual(find_witness_windows(samples, WITNESSES, [[0, 0, 0]], 24), [WitnessWindow(1000, 1020)])
        samples[0]["witnesses"]["adult_02"]["activity"] = "home"
        self.assertEqual(find_witness_windows(samples, WITNESSES, [[0, 0, 0]], 24), [])


class IncidentRecordTests(unittest.TestCase):
    """Require all actual events and first-hand bridge evidence, including ray zero."""

    def test_sabotage_uses_actual_origins_and_does_not_mutate_manifest(self):
        """Event/entity IDs come from the log, not the fixed UUID in the plan."""
        manifest, events, testimony, news = incident_records()
        original = deepcopy(manifest)
        truth = IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)
        self.assertEqual(truth["status"], "staged_live")
        self.assertEqual(truth["incident_origin_ids"], ["change:400", "change:401",
                                                       450, *[f"change:{index}" for index in range(410, 436)],
                                                       310, 500, "entity:actual-tool"])
        self.assertEqual(truth["collapse_event_id"], 450)
        self.assertEqual(truth["deck_change_ids"], [f"change:{index}" for index in range(410, 436)])
        self.assertEqual(truth["purchase_transfer_id"], 310)
        self.assertEqual(truth["evidence_positions"]["envoy_pickaxe"]["thrower"], "actual-envoy")
        self.assertNotIn("tag", truth["evidence_positions"]["envoy_pickaxe"])
        self.assertEqual(truth["witness_news_ids"], {witness: ["change:410"] for witness in WITNESSES})
        self.assertEqual(truth["witness_news"]["adult_01"][0]["present"], ["envoy"])
        self.assertEqual(manifest, original)

    def test_accident_needs_neither_purchase_nor_actor_testimony(self):
        """World-caused removals alone ground an accident; no tool is invented."""
        manifest, events, testimony, news = incident_records("accident")
        for records in testimony.values():
            records[0]["entities"] = []
        truth = IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)
        self.assertIsNone(truth["culprit"])
        self.assertIsNone(truth["evidence_positions"]["envoy_pickaxe"])
        supports = len(manifest["evidence"]["supports"])
        deck = sum(event.get("cause") == "collapse" for event in events)
        self.assertEqual(len(truth["incident_origin_ids"]), supports + 1 + deck)  # supports, the collapse, the deck
        self.assertEqual(truth["witness_news"]["adult_01"][0]["present"], [])

    def test_missing_removal_purchase_or_real_drop_is_rejected(self):
        """A partial incident never inherits pre-written ground truth as evidence."""
        for missing in (400, 310, 500, 410, 450):
            manifest, events, testimony, news = incident_records()
            with self.subTest(missing=missing), self.assertRaises(harness.HarnessError):
                IncidentRecord.ground_truth(manifest, [event for event in events if event["id"] != missing],
                                            testimony, news, WINDOW)

    def test_wrong_actor_ledger_or_purchase_is_rejected(self):
        """Attribution and the mayor's book must agree with the actual purchase."""
        for mismatch in ("actor", "cost", "shop", "ledger"):
            manifest, events, testimony, news = incident_records()
            if mismatch == "actor":
                events[0]["actor"] = "system"
            elif mismatch == "ledger":
                manifest["evidence"]["ledger_entry"] = "no sales"
            else:
                next(event for event in events if event["type"] == "purchase")[mismatch] = 99
            with self.subTest(mismatch=mismatch), self.assertRaises(harness.HarnessError):
                IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)

    def test_every_witness_must_see_the_envoy_and_one_must_hold_first_hand_news(self):
        """Geometric radius, occluded or out-of-window sightings, or only relayed news cannot certify witnesses."""
        for missing in ("actor", "ray", "time", "firsthand", "origin"):
            manifest, events, testimony, news = incident_records()
            record = testimony["adult_02"][0]
            if missing == "actor":
                record["entities"] = []
            elif missing == "ray":
                record["entities"][0].pop("losRay")
            elif missing == "time":
                record["gameTime"] = WINDOW.start - 1
            elif missing == "firsthand":
                for witness in WITNESSES:
                    news[witness][0]["firsthand"] = False
            else:
                for witness in WITNESSES:
                    news[witness][0]["originEventId"] = "change:99999"
            with self.subTest(missing=missing), self.assertRaises(WitnessWindowError):
                IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)

    def test_one_witness_with_first_hand_news_suffices(self):
        """The second witness may only have seen the envoy; news IDs are kept for those who hold them."""
        manifest, events, testimony, news = incident_records()
        news["adult_02"][0]["firsthand"] = False
        testimony["adult_02"][0]["changes"] = []
        truth = IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)
        self.assertEqual(truth["witness_news_ids"], {"adult_01": ["change:410"]})
        self.assertEqual(truth["witness_interval"], {"since": 3000, "until": 3200})

    def test_accident_needs_a_witness_to_see_the_failure(self):
        """An actorless accident with no witness sight of any support change is not staged."""
        manifest, events, testimony, news = incident_records("accident")
        for records in testimony.values():
            records[0]["entities"] = []
            for change in records[0]["changes"]:
                change.pop("losRay")
        with self.assertRaises(WitnessWindowError):
            IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)

    def test_collapse_must_have_real_sorted_changes_and_support_causality(self):
        """Wrong attribution, cause, ordering or an invented collapse member never becomes manifest truth."""
        for mismatch in ("causeEventId", "actor", "cause", "to", "changes", "order", "gameTime"):
            manifest, events, testimony, news = incident_records()
            change = next(event for event in events if event["id"] == 410)
            collapse = next(event for event in events if event["type"] == "collapse")
            if mismatch == "changes":
                collapse["changes"].append(9999)
            elif mismatch == "order":
                events[4], events[5] = events[5], events[4]
            else:
                change[mismatch] = {"causeEventId": 400, "actor": "envoy", "cause": "bot_action",
                                    "to": "minecraft:water", "gameTime": 3004}[mismatch]
            with self.subTest(mismatch=mismatch), self.assertRaises(harness.HarnessError):
                IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)

    def test_present_requires_same_observation_sight_and_a_nearby_agent(self):
        """Names cannot be inferred from another observation, a blocked ray, an NPC, or a distant entity."""
        for mismatch in ("learnedAt", "present", "ray", "position", "npc", "support_news"):
            manifest, events, testimony, news = incident_records()
            for witness in WITNESSES:
                item, entity = news[witness][0], testimony[witness][0]["entities"][0]
                if mismatch == "learnedAt":
                    item["learnedAt"] = 3040
                elif mismatch == "present":
                    item["present"] = []
                elif mismatch == "ray":
                    entity.pop("losRay")
                elif mismatch == "position":
                    entity["pos"] = [100, -60, 0]
                elif mismatch == "npc":
                    entity["npcId"] = "impostor"
                else:
                    item["originEventId"] = "change:400"
            with self.subTest(mismatch=mismatch), self.assertRaises(WitnessWindowError):
                IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)

    def test_accident_rejects_collapse_news_naming_any_agent(self):
        """Actorless support removal is insufficient when collapse news actually puts an agent there."""
        manifest, events, testimony, news = incident_records("accident")
        testimony["adult_01"][0]["entities"] = [{"name": "builder", "pos": [-6.5, -60, 0.99], "losRay": 0}]
        news["adult_01"][0]["present"] = ["builder"]
        with self.assertRaises(WitnessWindowError):
            IncidentRecord.ground_truth(manifest, events, testimony, news, WINDOW)


class FixtureTests(unittest.TestCase):
    """Pin pre-incident setup and the exact fixture.json byte identities."""

    def test_pre_incident_has_supports_stock_and_funded_envoy(self):
        """The pure pre-incident config changes no witness itinerary or body."""
        for variant in ("sabotage", "accident"):
            config = ScenarioConfig.create(variant, 17, pre_incident=True)
            manifest = config.to_dict()
            self.assertEqual(manifest["ground_truth"]["status"], "pre_incident")
            self.assertEqual(manifest["ground_truth"]["incident_origin_ids"], [])
            self.assertEqual(config.evidence.supports, ScenarioConfig.create("accident", 17).evidence.supports)
            self.assertEqual(config.containers["stall_3"]["stock"]["minecraft:iron_pickaxe"], 1)
            self.assertGreaterEqual(next(role.kit["minecraft:emerald"] for role in config.roles
                                         if role.name == "envoy"), 8)
            self.assertEqual(config.npcs, ScenarioConfig.create(variant, 17).npcs)
            self.assertEqual(manifest["layout"]["agent_start_game_time"], 6000)
            region = manifest["layout"]["fixture_hash_region"]
            self.assertLessEqual(region["y0"], -63)
            self.assertGreaterEqual(region["y1"], -57)

    def test_metadata_uses_exact_bytes_and_existing_world_hash(self):
        """Only JAR/manifest fixture inputs are hashed; world hash is not rehashed."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(FixtureMetadata.read(directory), {"status": "legacy_unversioned"})
            manifest = directory / "scenario.json"
            manifest.write_text(json.dumps({"evidence": {"witnesses": WITNESSES}}), encoding="utf-8")
            state = CodeState("abc123", False, "", (), hashlib.sha256(b"jar bytes").hexdigest())
            expected = FixtureMetadata.write(directory, state, manifest=manifest, world_hash="a" * 64)
            self.assertEqual(FixtureMetadata.read(directory), json.loads(json.dumps(expected)))
            self.assertEqual(expected["code"]["commit"], "abc123")
            self.assertEqual(expected["world_hash"], "a" * 64)
            self.assertEqual(expected["manifest_fields"], FixtureMetadata.field_hashes(manifest))


if __name__ == "__main__":
    unittest.main()
