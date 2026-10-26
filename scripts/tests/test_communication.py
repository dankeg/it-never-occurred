"""Server-free checks for communication measurements and grounded interaction evidence."""

import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import audit_o1, probe_communication


def utterance(identifier=1, *, distance=5, end_distance=None, volume="talk", heard=True):
    """Create a completed utterance with independent start/end eye geometry."""
    end_distance = distance if end_distance is None else end_distance
    event = {"id": identifier, "type": "utterance", "uttId": identifier, "speaker": f"speaker_{identifier}",
            "speakerKey": f"speaker_{identifier}", "text": "hello", "volume": volume,
            "radius": {"talk": 8, "shout": 24, "whisper": 3}[volume],
            "startedAt": 10, "endsAt": 20, "gameTime": 20,
            "speakerStartEye": [0, 0, 0], "pos": [0, 0, 0], "listenerStartEyes": {"alice": [distance, 0, 0]},
            "listenerEyes": {"alice": [end_distance, 0, 0]}, "listeners": ["alice"] if heard else [],
            "attentionSlots": 2, "attentionLost": {}, "news": [], "interrupted": False}
    if volume == "whisper":
        event["chatterRadius"] = 8
    return event


def speech_item(event, category="heard", **extra):
    """Build one audit item from the Java completion/evidence field contract."""
    radius = event["radius"] * (2 if category == "shout_seen" else 1)
    if category == "chatter" and extra.get("reason") == "out_of_earshot":
        radius = event["chatterRadius"]
    item = {field: copy.deepcopy(event[field]) for field in
            ("uttId", "speaker", "volume", "startedAt", "gameTime", "attentionSlots", "speakerStartEye")}
    start = event["listenerStartEyes"].get("alice")
    end = event["listenerEyes"]["alice"]
    item.update(category=category, prov=f"utt:{event['uttId']}", radius=radius, speakerEye=event["pos"],
                listenerStartEye=start, listenerEye=end,
                startDistance=math.dist(event["speakerStartEye"], start) if start is not None else None,
                distance=math.dist(event["pos"], end), inRadius=True, losRay=None)
    item.update(inStart=start is not None and item["startDistance"] <= radius, inEnd=item["distance"] <= radius)
    if category == "heard":
        item["text"] = event["text"]
    return {**item, **extra}


def ray_evidence(eye, target, *, index=0, hit_type="MISS", empty=False):
    """Build deterministic Sight.Ray.audit fields reaching a supplied probe endpoint."""
    return {"losRay": index, "rayFrom": list(eye), "rayTo": list(target), "hitType": hit_type,
            "hitPos": [math.floor(coordinate) for coordinate in target], "hitLocation": list(target),
            "emptyTarget": empty}


def whisper_chatter(*, start=5, end=5, ray_index=0):
    """Attest one visual whisper, optionally with a newly arrived listener or center ray."""
    event = utterance(distance=start, end_distance=end, volume="whisper", heard=False)
    if start is None:
        event["listenerStartEyes"] = {}
    target = event["pos"] if ray_index == 0 else [0, -0.7, 0]
    ray = ray_evidence(event["listenerEyes"]["alice"], target, index=ray_index)
    event["whisperSeen"] = {"alice": copy.deepcopy(ray)}
    return event, speech_item(event, "chatter", reason="out_of_earshot", **ray)


def action_notice(kind="block_use", *, sense="sight", identifier=1):
    """Pair a source action with its exact stored perception and later heard exposure."""
    event = {"id": identifier, "type": kind, "actor": "builder", "pos": [0, 0, 0],
             "prov": f"event:{identifier}", "gameTime": 20}
    event.update(block="minecraft:bell" if sense == "hearing" else "minecraft:oak_door",
                 state="minecraft:bell" if sense == "hearing" else "Block{minecraft:oak_door}[facing=north,open=true]")
    notice = {field: copy.deepcopy(event[field]) for field in ("prov", "gameTime", "pos")}
    if sense == "hearing":
        words = "bell rang at (0, 0, 0)"
    else:
        words = "builder used the oak_door at (0, 0, 0); it is now open"
    notice.update(kind=kind, sense=sense, speaker="bell" if sense == "hearing" else event["actor"],
                  text=words,
                  radius=48 if sense == "hearing" else 24, listenerEye=[4.5, 0.5, 0.5], distance=4,
                  inStart=True, inEnd=True)
    if sense == "hearing":
        notice["losRay"] = None
    else:
        notice.update(ray_evidence(notice["listenerEye"], [0.5, 0.5, 0.5], hit_type="BLOCK"))
    event["perceptions"] = {"alice": copy.deepcopy(notice)}
    return event, {**notice, "category": "heard", "inRadius": True}


def audit(events, items):
    """Audit a small in-memory exposure, never a server or fabricated world."""
    return audit_o1.audit_logs(events, [{"gameTime": 30, "observer": "alice", "observerEye": [100, 0, 0],
                                        "kind": "observe", "items": items}])


class SpeechEvidenceTests(unittest.TestCase):
    """Check endpoint admission, reason-aware chatter, and second-hand origins."""

    def test_npc_display_name_can_equal_another_listeners_canonical_key(self):
        """Exclude self by speakerKey, never by a potentially colliding display name."""
        event = utterance()
        event.update(speaker="alice", speakerKey="npc:adult_00")
        self.assertEqual(audit([event], [speech_item(event)]), [])

    def test_endpoint_motion_partial_presence_and_duplicate_delivery(self):
        """Require both endpoints for content, but exactly one for partial chatter."""
        event = utterance(end_distance=12, heard=False)
        item = speech_item(event, "chatter", reason="partial_presence")
        self.assertEqual(audit([event], [item]), [])
        self.assertTrue(audit([event], [speech_item(event)]))
        self.assertTrue(audit([event], [item, item]))
        event["listenerEyes"]["alice"] = [6, 0, 0]
        self.assertTrue(audit([event], [speech_item(event, "chatter", reason="partial_presence")]))

    def test_shout_seen_is_text_free_and_bounded_at_either_endpoint(self):
        """Allow seen-plus-heard separately but never conspicuity outside twice range."""
        event = utterance(volume="shout")
        self.assertEqual(audit([event], [speech_item(event), speech_item(event, "shout_seen")]), [])
        event = utterance(distance=48, end_distance=60, volume="shout", heard=False)
        self.assertEqual(audit([event], [speech_item(event, "shout_seen")]), [])
        self.assertTrue(audit([event], [speech_item(event, "shout_seen", text="leaked")]))
        event["listenerStartEyes"]["alice"] = [49, 0, 0]
        self.assertTrue(audit([event], [speech_item(event, "shout_seen")]))

    def test_endpoint_flags_tolerate_roundoff_without_changing_presence(self):
        """Preserve strict Java endpoint flags while tolerating boundary rounding."""
        for start, end in ((8 + 1e-8, 7), (7, 8 + 1e-8)):
            with self.subTest(start=start, end=end):
                event = utterance(distance=start, end_distance=end, heard=False)
                item = speech_item(event, "chatter", reason="partial_presence")
                self.assertEqual(audit([event], [item]), [])
                self.assertTrue(audit([event], [{**item, "inStart": True, "inEnd": True}]))
        event = utterance(distance=8 + 1e-8)
        self.assertEqual(audit([event], [speech_item(event, inStart=True, inEnd=True)]), [])
        event = utterance(distance=7, heard=False)
        self.assertTrue(audit([event], [speech_item(event, "chatter", reason="partial_presence", inStart=False)]))
        event = utterance(distance=9)
        self.assertTrue(audit([event], [speech_item(event, inStart=True, inEnd=True)]))

    def test_attention_chatter_checks_rank_overlap_and_geometry(self):
        """Full-presence attention chatter needs two nearer overlapping speakers."""
        events = [utterance(index, distance=index, heard=index < 3) for index in (1, 2, 3)]
        evidence = [{"uttId": event["uttId"], "speaker": event["speaker"], "startedAt": 10,
                     "endsAt": 20, "distance": index} for index, event in enumerate(events, 1)]
        events[-1]["attentionLost"] = {"alice": evidence}
        item = speech_item(events[-1], "chatter", reason="attention", attention=copy.deepcopy(evidence))
        self.assertEqual(audit(events, [item]), [])
        events[-1]["attentionLost"]["alice"][0]["distance"] = 7
        self.assertTrue(audit(events, [item]))
        events[-1]["attentionLost"]["alice"] = evidence
        events[-1]["listeners"] = ["alice"]
        self.assertTrue(audit(events, [speech_item(events[-1])]))

    def test_relay_preserves_origin_chain_and_secondhand_status(self):
        """Compare delivered relay metadata with the source news and block origin."""
        origin = {"id": 1, "type": "block_change", "gameTime": 2, "pos": [0, 0, 1],
                  "from": "minecraft:air", "to": "minecraft:barrier", "actor": "system", "cause": "test"}
        event = utterance(2)
        source = {"newsId": "news_1", "kind": "road_blocked", "subject": origin["pos"],
                  "originEventId": "change:1", "chain": ["witness"], "firsthand": True, "learnedAt": 2}
        event["news"] = [source]
        delivered = {**source, "chain": ["witness", event["speaker"]], "firsthand": False, "learnedAt": 20}
        item = speech_item(event, news=[delivered], **{field: delivered[field] for field in
                           ("newsId", "kind", "subject", "originEventId", "chain", "firsthand")})
        self.assertEqual(audit([origin, event], [item]), [])
        for field, value in (("firsthand", True), ("chain", ["invented"]), ("originEventId", "change:999")):
            with self.subTest(field=field):
                self.assertTrue(audit([origin, event], [{**item, field: value}]))

    def test_interrupted_speech_and_npc_communication_need_no_self_item(self):
        """NPC completion records carry standalone auditory geometry without self."""
        event = utterance(heard=False)
        event.update(interrupted=True, gameTime=15)
        item = speech_item(event, "chatter", reason="interrupted")
        item.pop("inRadius")
        item.pop("losRay")
        record = {"gameTime": 15, "observer": "alice", "observerEye": [5, 0, 0],
                  "kind": "communication", "items": [item]}
        self.assertEqual(audit_o1.audit_logs([event], [record]), [])
        event["listeners"] = ["alice"]
        self.assertTrue(audit([event], [speech_item(event)]))


class WhisperEvidenceTests(unittest.TestCase):
    """Require three-block hearing or independently attested, text-free visual chatter."""

    def test_whisper_hearing_and_partial_presence_use_only_three_blocks(self):
        """Keep ordinary speech admission at the whisper radius, without requiring visual metadata."""
        event = utterance(distance=3, volume="whisper")
        event.pop("chatterRadius")
        self.assertEqual(audit([event], [speech_item(event)]), [])
        event = utterance(distance=2, end_distance=4, volume="whisper", heard=False)
        self.assertEqual(audit([event], [speech_item(event, "chatter", reason="partial_presence")]), [])
        self.assertTrue(audit([event], [speech_item(event)]))
        event = utterance(distance=4, volume="whisper")
        self.assertTrue(audit([event], [speech_item(event)]))

    def test_whisper_chatter_radius_is_fixed_at_eight_and_separate_from_hearing(self):
        """Pin visual chatter to eight while retaining three-block hearing and requiring sight metadata."""
        event = utterance(distance=3, volume="whisper")
        self.assertEqual(event["chatterRadius"], 8)
        self.assertEqual(speech_item(event)["radius"], 3)
        self.assertEqual(audit([event], [speech_item(event)]), [])
        for distance in (3 + 1e-8, 8):
            with self.subTest(distance=distance):
                event, item = whisper_chatter(end=distance)
                self.assertEqual(item["radius"], 8)
                self.assertEqual(audit([event], [item]), [])
        event.pop("chatterRadius")
        self.assertTrue(audit([event], [item]))

    def test_whisper_chatter_radius_rejects_non_eight_overrides(self):
        """Reject configurable or malformed visual ranges in both heard and sight-attested records."""
        for radius in (-1, 0, 2, 3, 9, 100, math.nan, math.inf, True, "8", None):
            with self.subTest(chatterRadius=radius):
                event = utterance(distance=2, volume="whisper")
                event["chatterRadius"] = radius
                self.assertTrue(audit([event], [speech_item(event)]))
                event, item = whisper_chatter(start=4, end=4)
                event["chatterRadius"] = item["radius"] = radius
                self.assertTrue(audit([event], [item]))

    def test_whisper_attention_loss_keeps_existing_competition_rules(self):
        """Near whisper listeners still need ranked overlapping competitors for attention chatter."""
        events = [utterance(index, distance=distance, volume="whisper", heard=index < 3)
                  for index, distance in enumerate((1, 2, 2.5), 1)]
        candidates = [{"uttId": event["uttId"], "speaker": event["speaker"], "startedAt": 10,
                       "endsAt": 20, "distance": event["listenerStartEyes"]["alice"][0]} for event in events]
        events[-1]["attentionLost"] = {"alice": candidates}
        item = speech_item(events[-1], "chatter", reason="attention", attention=copy.deepcopy(candidates))
        self.assertEqual(audit(events, [item]), [])
        item["attention"] = list(reversed(candidates))
        self.assertTrue(audit(events, [item]))

    def test_visual_whisper_allows_outer_motion_new_listeners_and_both_entity_rays(self):
        """Visual endpoint flags use chatter range, including absent start geometry and center rays."""
        for start, end in ((5, 5), (9, 5), (5, 8), (None, 5)):
            for ray_index in (0, 1):
                with self.subTest(start=start, end=end, ray=ray_index):
                    event, item = whisper_chatter(start=start, end=end, ray_index=ray_index)
                    self.assertEqual(audit([event], [item]), [])
                    self.assertEqual(item["radius"], 8)
                    self.assertEqual(item["inStart"], start is not None and start <= 8)
        event, item = whisper_chatter()
        self.assertTrue(audit([event], [item, copy.deepcopy(item)]))

    def test_visual_whisper_cannot_leak_words_or_relay_metadata(self):
        """Reject every speech-content and relay field even when the visual proof remains valid."""
        event, item = whisper_chatter()
        for field, value in (("text", "secret"), ("news", []), ("originEventId", "change:99"),
                             ("chain", []), ("firsthand", False), ("subject", [0, 0, 0]),
                             ("newsId", "invented"), ("kind", "rumor"), ("to", "alice"), ("present", True),
                             ("learnedAt", 20), ("extra", {"words": "secret"})):
            with self.subTest(field=field):
                self.assertTrue(audit([event], [{**item, field: value}]))

    def test_visual_whisper_requires_inaudible_endpoints_and_in_range_completion(self):
        """A visible endpoint never justifies this branch if either audible endpoint was present."""
        for start, end in ((2, 5), (3, 5), (5, 0), (5, 2), (5, 3), (5, 8 + 1e-8), (9, 9)):
            with self.subTest(start=start, end=end):
                event, item = whisper_chatter(start=start, end=end)
                self.assertTrue(audit([event], [item]))
                self.assertTrue(audit([event], [{**item, "inEnd": True}]))
        event, item = whisper_chatter(start=None)
        self.assertTrue(audit([event], [{**item, "inStart": True}]))
        self.assertTrue(audit([event], [{**item, "listenerStartEye": [5, 0, 0], "startDistance": 5}]))
        self.assertTrue(audit([event], [{**item, "radius": 9}]))
        event["volume"] = item["volume"] = "talk"
        self.assertTrue(audit([event], [item]))

    def test_visual_whisper_requires_exact_complete_ray_attestation(self):
        """Missing source evidence and independent exposure mutations cannot invent visibility."""
        event, item = whisper_chatter()
        for field, value in (("losRay", 1), ("rayFrom", [6, 0, 0]), ("rayTo", [1, 0, 0]),
                             ("hitType", "BLOCK"), ("hitPos", [1, 0, 0]),
                             ("hitLocation", [1, 0, 0]), ("emptyTarget", True)):
            with self.subTest(field=field):
                self.assertTrue(audit([event], [{**item, field: value}]))
        for field in sorted(audit_o1.RAY_FIELDS):
            with self.subTest(missing=field):
                missing_item = {key: value for key, value in item.items() if key != field}
                self.assertTrue(audit([event], [missing_item]))
                missing_event = copy.deepcopy(event)
                missing_event["whisperSeen"]["alice"].pop(field)
                self.assertTrue(audit([missing_event], [item]))
        for seen in ({}, [], {"bob": event["whisperSeen"]["alice"]},
                     {"alice": {**event["whisperSeen"]["alice"], "text": "secret"}}):
            with self.subTest(seen=seen):
                self.assertTrue(audit([{**event, "whisperSeen": seen}], [item]))

    def test_attested_whisper_rays_still_need_finite_unobstructed_geometry(self):
        """Matching source and exposure cannot excuse impossible ray geometry or a false whisper radius."""
        for field, value in (("losRay", 2), ("losRay", True), ("rayFrom", [math.nan, 0, 0]),
                             ("rayFrom", [6, 0, 0]), ("rayTo", [1, 0, 0]), ("hitType", "ENTITY"),
                             ("hitPos", [0.5, 0, 0]), ("hitLocation", [2, 0, 0]), ("emptyTarget", True)):
            with self.subTest(field=field, value=value):
                event, item = whisper_chatter()
                event["whisperSeen"]["alice"][field] = value
                item[field] = value
                self.assertTrue(audit([event], [item]))
        for field, value in (("radius", 4), ("chatterRadius", 2), ("chatterRadius", math.inf)):
            with self.subTest(field=field):
                event, item = whisper_chatter()
                event[field] = value
                self.assertTrue(audit([event], [item]))


class ActionNoticeEvidenceTests(unittest.TestCase):
    """Validate exact action perceptions rather than treating notices as invented utterances."""

    def test_own_skill_progress_may_cite_its_block_use_event(self):
        """An actor's self view citing the event its own use caused is a reference, not a heard notice (IT-20)."""
        event, _ = action_notice()
        self_item = {"category": "self", "name": "alice", "ownedBy": "alice", "pos": [100, 0, 0], "distance": 0,
                     "reachOk": True, "losRay": None, "inRadius": True,
                     "skill": {"skill": "use", "state": "succeeded", "progress": {"prov": event["prov"]}}}
        self.assertEqual(audit([event], [self_item]), [])
        self_item["skill"]["progress"]["prov"] = "event:99"
        self.assertTrue(audit([event], [self_item]))

    def test_bell_and_visual_use_accept_occurrence_time_evidence(self):
        """Both notice shapes resolve without utterance fields or the observer's later eye position."""
        for sense in ("hearing", "sight"):
            with self.subTest(sense=sense):
                event, item = action_notice(sense=sense)
                self.assertEqual(audit([event], [item]), [])
                self.assertFalse({"uttId", "startedAt", "volume"}.intersection(item))

    def test_action_and_speech_ids_are_independent_and_notices_drain_once(self):
        """Distinct provenance namespaces coexist, but no second exposure may redeliver an action."""
        speech = utterance(2)
        speech["id"] = 1
        event, item = action_notice(sense="hearing", identifier=2)
        self.assertEqual(audit([speech, event], [speech_item(speech), item]), [])
        self.assertTrue(audit([event], [item, copy.deepcopy(item)]))
        records = [{"gameTime": tick, "observer": "alice", "observerEye": [100, 0, 0],
                    "kind": "observe", "items": [copy.deepcopy(item)]} for tick in (30, 31)]
        self.assertTrue(audit_o1.audit_logs([event], records))

    def test_legacy_transfers_without_action_notices_remain_valid(self):
        """Ordinary historical transfer events do not acquire new owner or perception requirements."""
        transfer = {"id": 1, "type": "transfer", "gameTime": 5, "from": "builder", "to": "alice",
                    "item": "minecraft:bread", "count": 1}
        speech = utterance(2)
        self.assertEqual(audit([transfer, speech], [speech_item(speech)]), [])

    def test_exposure_fields_must_match_the_complete_observer_perception(self):
        """Reject altered provenance, factual words, geometry, category, and invented extra fields."""
        event, item = action_notice()
        for field, value in (("prov", "event:99"), ("prov", "event:01"), ("prov", "utt:1"),
                             ("kind", "transfer"), ("pos", [1, 0, 0]), ("gameTime", 21),
                             ("speaker", "envoy"), ("text", "invented words"), ("radius", 99),
                             ("distance", 0), ("listenerEye", [1, 1, 1]), ("sense", "hearing"),
                             ("inStart", False), ("inEnd", False), ("losRay", None),
                             ("category", "chatter"), ("inRadius", False), ("extra", "secret")):
            with self.subTest(field=field, value=value):
                self.assertTrue(audit([event], [{**item, field: value}]))

    def test_notice_provenance_requires_the_matching_source_event_and_observer(self):
        """Source kinds, identities, times and perception-map keys are independently checked."""
        event, item = action_notice()
        for field, value in (("type", "origin"), ("type", "transfer"), ("prov", "event:2"), ("pos", [1, 0, 0]),
                             ("gameTime", 21), ("actor", "envoy"),
                             ("perceptions", {}), ("perceptions", []),
                             ("perceptions", {"bob": event["perceptions"]["alice"]})):
            with self.subTest(field=field):
                self.assertTrue(audit([{**event, field: value}], [item]))
        event, item = action_notice()
        event["gameTime"] = event["perceptions"]["alice"]["gameTime"] = item["gameTime"] = 31
        self.assertTrue(audit([event], [item]))

    def test_bell_hearing_needs_exact_words_source_radius_and_no_ray(self):
        """Only an attested bell may produce the fixed text at the fixed 48-block hearing radius."""
        for field, value in (("text", "bell said secret words"), ("radius", 47), ("radius", 49),
                             ("speaker", "builder"), ("losRay", 0)):
            with self.subTest(field=field, value=value):
                event, item = action_notice(sense="hearing")
                event["perceptions"]["alice"][field] = item[field] = value
                self.assertTrue(audit([event], [item]))
        event, item = action_notice(sense="hearing")
        self.assertTrue(audit([{**event, "block": "minecraft:lever"}], [item]))
        event, item = action_notice(sense="hearing")
        self.assertTrue(audit([{**event, "block": "minecraft:oak_door"}], [item]))

    def test_attested_notices_cannot_invent_speech_or_relay_fields(self):
        """Even source-attested payloads must remain action notices, not spoken or relayed content."""
        for field, value in (("uttId", 1), ("startedAt", 10), ("volume", "talk"), ("news", []),
                             ("originEventId", "change:99"), ("chain", ["builder"]),
                             ("firsthand", False), ("to", "alice")):
            with self.subTest(field=field):
                event, item = action_notice()
                event["perceptions"]["alice"][field] = item[field] = value
                self.assertTrue(audit([event], [item]))

    def test_attested_visual_words_must_describe_the_actual_source_action(self):
        """Reconstruct Java's factual visual notice text instead of trusting arbitrary attested words."""
        event, item = action_notice()
        event["perceptions"]["alice"]["text"] = item["text"] = "builder said secret words"
        self.assertTrue(audit([event], [item]))
        for field, value in (("block", "minecraft:lever"), ("state", "Block{minecraft:oak_door}[open=false]"),
                             ("actor", "envoy")):
            with self.subTest(field=field):
                event, item = action_notice()
                event[field] = value
                self.assertTrue(audit([event], [item]))

    def test_action_ranges_use_block_center_and_preserve_boundary_admission(self):
        """Check actual sensory distance, including the exact boundary and one block outside it."""
        for sense in ("sight", "hearing"):
            for outside in (False, True):
                with self.subTest(sense=sense, outside=outside):
                    event, item = action_notice(sense=sense)
                    distance = item["radius"] + int(outside)
                    item.update(listenerEye=[distance + 0.5, 0.5, 0.5], distance=distance)
                    if sense == "sight":
                        item["rayFrom"] = list(item["listenerEye"])
                    event["perceptions"]["alice"] = {field: copy.deepcopy(value) for field, value in item.items()
                                                     if field not in {"category", "inRadius"}}
                    self.assertEqual(bool(audit([event], [item])), outside)

    def test_attested_action_geometry_and_flags_cannot_be_forged(self):
        """A matching perception cannot excuse nonfinite distances, obstructed rays, or nonboolean flags."""
        for field, value in (("distance", math.nan), ("distance", 3.5), ("radius", math.inf),
                             ("radius", True), ("listenerEye", [math.nan, 0, 0]),
                             ("inStart", 1), ("inEnd", False), ("losRay", True),
                             ("rayFrom", [9, 0.5, 0.5]), ("rayTo", [1.5, 0.5, 0.5]),
                             ("hitLocation", [2.5, 0.5, 0.5]), ("hitType", "MISS"),
                             ("hitPos", [2, 0, 0])):
            with self.subTest(field=field, value=value):
                event, item = action_notice()
                event["perceptions"]["alice"][field] = item[field] = value
                self.assertTrue(audit([event], [item]))


class MeasurementTests(unittest.TestCase):
    """Check measured threshold, population, and knowledge-matching logic."""

    def test_village_events_and_six_bot_loop_starts(self):
        """Use village event sites, a road loop starting at the market, and six distinct bots spread over its stops."""
        self.assertEqual(probe_communication.MAJOR_POS, [-12, -60, 55])
        self.assertEqual(probe_communication.MINOR_POS, [112, -60, -48])
        self.assertEqual(probe_communication.LOOP[0], (-52.5, -60, -3.5))
        bots = [probe_communication._RoutineBot(index, 600) for index in range(6)]
        self.assertEqual([bot.name for bot in bots], [f"comm_{index}" for index in range(6)])
        self.assertEqual([bot._waypoint for bot in bots], [index % len(probe_communication.LOOP) for index in range(6)])

    def test_setup_reads_selected_village_roster_without_a_server(self):
        """Keep the built village's one-based IDs rather than V1's zero-based roster."""
        expected = ({f"adult_{index:02d}" for index in range(1, 23)}
                    | {f"child_{index:02d}" for index in range(1, 9)})
        for variant in ("sabotage", "accident"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(runs_dir=Path(temporary), variant=variant)
                probe = probe_communication.CommunicationProbe(args)
                probe.setup()
                self.assertEqual(probe._source, args.runs_dir / f"village_{variant}")
                self.assertEqual(set(probe._npc_ids), expected)
                self.assertEqual(len(probe._npc_ids), 30)
                self.assertTrue((args.runs_dir / "probe_comm").is_dir())
                with self.assertRaises(FileExistsError):
                    probe.setup()

    def test_curves_count_both_populations_and_keep_missing_thresholds_null(self):
        """Retain zero-time thresholds and distinguish absent time-to-75 data."""
        curve = probe_communication.SpreadCurve(100)
        curve.sample(100, {str(index) for index in range(9)}, set())
        curve.sample(300, {str(index) for index in range(12)}, {str(index) for index in range(6)})
        report = curve.report()
        self.assertEqual(report["timeToPercentTicks"], {"25": 0, "50": 200, "75": None})
        self.assertEqual(report["finalReach"], 0.5)
        self.assertEqual(report["finalBotReach"], 1)
        report["samples"][-1]["fraction"] = 0
        self.assertEqual(curve.report()["samples"][-1]["fraction"], 0.5)
        with self.assertRaises(ValueError):
            curve.sample(300, set(), set())

    def test_subject_matching_includes_nested_ask_news_and_shout_text(self):
        """Count actual closure evidence, not mere speaker location or chatter."""
        event = {"pos": [-18, -60, 0], "originEventId": 42}
        matches = probe_communication.CommunicationProbe.matches
        self.assertTrue(matches({"news": [{"originEventId": "change:42"}]}, event))
        self.assertTrue(matches({"kind": "shout", "subject": [3, 4, 5],
                                 "text": "road_blocked at [-18, -60, 0]"}, event))
        self.assertFalse(matches({"speaker": "alice", "kind": "shout", "subject": [3, 4, 5]}, event))
        self.assertFalse(matches({"kind": "shout", "subject": event["pos"], "text": "hello"}, event))
        self.assertFalse(matches({"text": "road_blocked elsewhere"}, event))
        rumor = {"kind": "rumor", "text": probe_communication.RUMOR_TEXT}
        self.assertTrue(matches({"text": rumor["text"], "volume": "shout"}, rumor))
        self.assertTrue(matches({"news": [{"kind": "shout", "text": rumor["text"], "firsthand": False}]}, rumor))
        self.assertTrue(matches({"kind": "shout", "text": rumor["text"], "firsthand": True}, rumor))
        self.assertFalse(matches({"text": rumor["text"] + " Maybe."}, rumor))
        self.assertFalse(matches({"kind": "construction", "text": rumor["text"]}, rumor))
        self.assertFalse(matches({"speaker": "comm_0", "volume": "shout"}, rumor))

    def test_knowledge_partitions_witnesses_and_uses_non_witness_denominators(self):
        """Keep sight disjoint from speech, including later sightings and empty populations."""
        curve = probe_communication.SpreadCurve(100)
        curve.sample(100, {str(index) for index in range(10)}, {"heard", "seen"},
                     npc_witnesses={str(index) for index in range(6)}, bot_witnesses={"seen"})
        curve.sample(300, {str(index) for index in range(8, 14)}, {"heard"},
                     npc_witnesses={str(index) for index in range(8)}, bot_witnesses={"seen"})
        curve.sample(500, set(), set(), npc_witnesses={str(index) for index in range(30)},
                     bot_witnesses={str(index) for index in range(6)})
        report = curve.report()
        first, second, last = report["samples"]
        self.assertEqual(first["witnesses"]["count"], 7)
        self.assertEqual(first["informed_by_speech"]["count"], 5)
        self.assertEqual(first["informed_by_speech"]["fraction"], 5 / 29)
        self.assertEqual(first["total"]["count"], 12)
        self.assertEqual(second["informed_by_speech"]["npcFraction"], 6 / 22)
        self.assertEqual(second["total"]["count"], 16)
        self.assertEqual(report["timeToPercentTicks"], {"25": 0, "50": 400, "75": 400})
        self.assertEqual(report["speechTimeToPercentTicks"], {"25": 200, "50": None, "75": None})
        self.assertEqual(last["informed_by_speech"]["population"], 0)
        self.assertIsNone(report["finalSpeechReach"])
        last["witnesses"]["count"] = 0
        self.assertEqual(curve.report()["samples"][-1]["witnesses"]["count"], 36)

    def test_ratios_never_invent_times_or_divide_by_zero(self):
        """Report unavailable comparisons as null instead of infinities or zero."""
        ratio = probe_communication.CommunicationProbe.ratio
        self.assertEqual(ratio(600, 200), 3)
        self.assertEqual(ratio(0, 200), 0)
        self.assertIsNone(ratio(None, 200))
        self.assertIsNone(ratio(200, 0))

    def test_loop_steering_uses_minecraft_cardinal_yaws(self):
        """Drive ordinary physical inputs rather than teleporting or replanning."""
        heading = probe_communication._RoutineBot.heading
        for target, yaw in (((0, 0, 5), 0), ((-5, 0, 0), 90), ((5, 0, 0), -90)):
            with self.subTest(target=target):
                self.assertEqual(heading((0, 0, 0), target), (5, yaw))


if __name__ == "__main__":
    unittest.main()
