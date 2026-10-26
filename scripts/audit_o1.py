"""Offline O1 exposure/provenance audit; never starts or contacts a server.

M3 chatter is reason-aware: R2 attention loss permits full endpoint presence,
clarifying R4's partial-presence-only wording. Interrupted utterances also
expose no content. Visual whisper chatter and action notices require source-event
attestations. Legacy instantaneous O1 logs remain independently auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import uuid

if __package__:
    from .setup_server import RUNS_DIR
else:
    from setup_server import RUNS_DIR


EPSILON = 1e-6
WHISPER_CHATTER_RADIUS = 8.0
SPATIAL = {"entity", "block", "document", "change", "container", "terrain", "structure"}
SPEECH = {"heard", "chatter", "shout_seen"}
RAY_FIELDS = {"losRay", "rayFrom", "rayTo", "hitType", "hitPos", "hitLocation", "emptyTarget"}
SEEN_ACTION_FACTS = {"prov", "gameTime", "actor", "action", "item"}
ENTITY_SIGHT_FIELDS = {"category", "pos", "distance", "radius", "inRadius", "targetEye", "targetCenter", *RAY_FIELDS}
CATEGORIES = {
    "observe": {"self", "terrain", "entity", "block", "document", "change", "structure", *SPEECH,
                "inventory_document", "seen_action"},
    "read_item_doc": {"inventory_document"},
    "open_container": {"container", "document"},
    "communication": SPEECH,
}
POSITION_PROV = re.compile(r"(block|container):(-?\d+),(-?\d+),(-?\d+)@(base|[1-9]\d*)\Z")
DOCUMENT_PROV = re.compile(r"doc:(.+)@v(0|[1-9]\d*)\Z")


class AuditError(ValueError):
    """Signal malformed evidence or an exposure/provenance invariant violation."""

    pass


def require(condition: bool, message: str) -> None:
    """Raise an audit error when a recorded invariant does not hold."""
    if not condition:
        raise AuditError(message)


def number(value, label: str) -> float:
    """Return finite numeric evidence, rejecting booleans and malformed values."""
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric data")
    return value


def integer(value, label: str, minimum: int = 0) -> int:
    """Return integer evidence at or above the required minimum."""
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def vector(value, label: str, *, block: bool = False) -> list:
    """Validate three finite coordinates, optionally requiring block integers."""
    require(isinstance(value, list) and len(value) == 3, f"{label} must have three coordinates")
    for coordinate in value:
        number(coordinate, label)
        require(not block or type(coordinate) is int, f"{label} must contain block coordinates")
    return value


def text(value, label: str) -> str:
    """Return nonempty text evidence or raise an audit error."""
    require(isinstance(value, str) and bool(value), f"{label} must be a nonempty string")
    return value


def close(left, right) -> bool:
    """Test whether positions agree within the existing spatial tolerance."""
    return math.dist(left, right) <= EPSILON


def sight_ray(item: dict, eye: list, maximum: int) -> None:
    """Validate finite, ordered sight-ray evidence without choosing a target shape."""
    require(integer(item.get("losRay"), "losRay") <= maximum, "invalid LOS ray index")
    ray_from = vector(item.get("rayFrom"), "rayFrom")
    ray_to = vector(item.get("rayTo"), "rayTo")
    require(close(ray_from, eye), "rayFrom differs from observerEye")
    hit_location = vector(item.get("hitLocation"), "hitLocation")
    hit_pos = vector(item.get("hitPos"), "hitPos", block=True)
    require(item.get("hitType") in {"MISS", "BLOCK"}, "invalid hitType")
    if item["hitType"] == "BLOCK":
        require(all(hit_pos[axis] - EPSILON <= hit_location[axis] <= hit_pos[axis] + 1 + EPSILON
                    for axis in range(3)), "hitLocation outside hitPos")
    require(abs(math.dist(ray_from, hit_location) + math.dist(hit_location, ray_to) -
                math.dist(ray_from, ray_to)) <= EPSILON, "hitLocation is not on ray segment")
    require(type(item.get("emptyTarget", False)) is bool, "emptyTarget must be boolean")


def _unique_object(pairs):
    """Decode one JSON object while rejecting duplicate keys."""
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    """Reject nonfinite constants accepted by Python's default JSON decoder."""
    raise AuditError(f"non-finite JSON constant {value}")


def read_jsonl(path: Path) -> list[dict]:
    """Read strict JSON objects, rejecting duplicate keys and nonfinite data."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                require(isinstance(record, dict), "record must be an object")
            except ValueError as error:
                raise AuditError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    return records


class SpeechAudit:
    """Validate timed speech geometry, attention loss, and news provenance."""

    def __init__(self, events: dict, utterances: dict):
        """Retain the containing audit's indexes, including later completions."""
        self._events = events
        self._utterances = utterances

    @staticmethod
    def event(event: dict) -> None:
        """Require complete endpoint evidence for a completed M3 utterance."""
        started = integer(event.get("startedAt"), "speech startedAt")
        require(started <= event["gameTime"], "speech completed before it started")
        require(event.get("volume") in {"talk", "shout", "whisper"}, "invalid speech volume")
        if event["volume"] == "whisper":
            require(event["radius"] == 3, "whisper radius must be 3")
        if "chatterRadius" in event:
            require(event["volume"] == "whisper"
                    and number(event["chatterRadius"], "chatterRadius") == WHISPER_CHATTER_RADIUS,
                    "whisper chatter radius must be 8")
        key = text(event.get("speakerKey"), "speakerKey")
        integer(event.get("attentionSlots"), "attentionSlots")
        planned_end = integer(event.get("endsAt"), "speech endsAt")
        require(planned_end >= event["gameTime"]
                and (event.get("interrupted") is True or planned_end == event["gameTime"]),
                "speech completion differs from planned end without interruption")
        require(type(event.get("interrupted", False)) is bool, "interrupted must be boolean")
        require(isinstance(event.get("attentionLost", {}), dict), "attentionLost must be an object")
        require(isinstance(event.get("news", []), list), "utterance news must be a list")
        start_eye = vector(event.get("speakerStartEye"), "speakerStartEye")
        starts, ends = event.get("listenerStartEyes"), event.get("listenerEyes")
        require(isinstance(starts, dict) and isinstance(ends, dict), "listener endpoint maps must be objects")
        require(set(event["listeners"]) <= starts.keys() & ends.keys(), "content listener lacks endpoint evidence")
        require(key not in event["listeners"], "speakerKey listed as own listener")
        for listener in starts:
            text(listener, "listener key")
            vector(starts[listener], "listener start eye")
        for listener in ends:
            text(listener, "listener key")
            vector(ends[listener], "listener end eye")
            if listener in event["listeners"]:
                require(not event.get("interrupted", False) and listener not in event.get("attentionLost", {}),
                        "interrupted or attention-lost utterance admitted content")
                require(math.dist(start_eye, starts[listener]) <= event["radius"] + EPSILON
                        and math.dist(event["pos"], ends[listener]) <= event["radius"] + EPSILON,
                        "content listener outside speech radius at an endpoint")
        seen = event.get("whisperSeen", {})
        require(isinstance(seen, dict), "whisperSeen must be an object")
        for listener, ray in seen.items():
            text(listener, "whisper listener")
            SpeechAudit.whisper_sight(event, listener, ray)

    @staticmethod
    def whisper_sight(event: dict, observer: str, ray: dict) -> None:
        """Attest text-free visibility strictly beyond three and within eight blocks."""
        require(event["volume"] == "whisper", "visual chatter requires a whisper")
        radius = number(event.get("chatterRadius"), "chatterRadius")
        require(radius == WHISPER_CHATTER_RADIUS, "visual whisper chatter radius must be 8")
        starts, ends = event["listenerStartEyes"], event["listenerEyes"]
        require(observer in ends and observer != event["speakerKey"] and observer not in event["listeners"],
                "invalid visual whisper listener")
        start_distance = math.dist(event["speakerStartEye"], starts[observer]) if observer in starts else math.inf
        end_distance = math.dist(event["pos"], ends[observer])
        require(start_distance > event["radius"] and event["radius"] < end_distance <= radius,
                "visual whisper requires inaudible endpoints and an in-range completion")
        require(isinstance(ray, dict) and set(ray) == RAY_FIELDS, "whisper sight must contain only ray evidence")
        sight_ray(ray, ends[observer], 1)
        require(ray["emptyTarget"] is False, "whisper sight must target the speaker entity")
        if ray["losRay"] == 0:
            require(close(ray["rayTo"], event["pos"]), "whisper eye ray differs from speaker endpoint")
        require(math.dist(ray["hitLocation"], ray["rayTo"]) <= 0.05 + EPSILON, "whisper LOS endpoint obstructed")

    def _origin(self, value, when: int) -> dict:
        if type(value) is int:
            origin = self._events.get(integer(value, "news originEventId", 1))
        else:
            require(isinstance(value, str) and re.fullmatch(r"(?:change|utt):[1-9]\d*", value) is not None,
                    "invalid news originEventId")
            prefix, identifier = value.split(":")
            origin = (self._events if prefix == "change" else self._utterances).get(int(identifier))
            require(origin is not None and origin["type"] == ("block_change" if prefix == "change" else "utterance"),
                    "news origin has the wrong event kind")
        require(origin is not None and origin["gameTime"] <= when, "unresolved or future news origin")
        return origin

    def _news(self, news: dict, event: dict) -> None:
        require(isinstance(news, dict), "news item must be an object")
        text(news.get("newsId"), "newsId")
        text(news.get("kind"), "news kind")
        origin = self._origin(news.get("originEventId"), event["startedAt"])
        require(news.get("firsthand") is False, "received news must be second-hand")
        chain = news.get("chain")
        require(isinstance(chain, list) and bool(chain), "relayed news requires a provenance chain")
        for relayer in chain:
            text(relayer, "news relayer")
        require(chain[-1] in {event["speaker"], event["speakerKey"]}, "news chain does not end at speaker")
        if "learnedAt" in news:
            require(integer(news["learnedAt"], "news learnedAt") <= event["gameTime"], "news learned in the future")
        subject = news.get("subject")
        if isinstance(subject, dict):
            subject = subject.get("pos")
        if isinstance(subject, list) and origin["type"] == "block_change":
            require(vector(subject, "news subject", block=True) == origin["pos"], "news subject differs from origin")

    def _attention(self, event: dict, observer: str) -> None:
        slots = event["attentionSlots"]
        candidates = event.get("attentionLost", {}).get(observer)
        require(slots > 0 and isinstance(candidates, list) and len(candidates) > slots,
                "attention chatter lacks competing-speaker loss evidence")
        ranked, identifiers, starts, ends = [], [], [], []
        for candidate in candidates:
            require(isinstance(candidate, dict), "attention candidate must be an object")
            identifier = integer(candidate.get("uttId"), "attention candidate uttId", 1)
            speaker = text(candidate.get("speaker"), "attention speaker")
            starts.append(integer(candidate.get("startedAt"), "attention startedAt"))
            ends.append(integer(candidate.get("endsAt"), "attention endsAt"))
            distance = number(candidate.get("distance"), "attention distance")
            require(distance >= 0, "negative attention distance")
            other = self._utterances.get(identifier)
            if other is not None:
                for field in ("speaker", "startedAt", "endsAt"):
                    require(candidate[field] == other.get(field), f"attention candidate {field} mismatch")
                require(observer in other["listenerStartEyes"], "attention listener absent at candidate start")
                expected = math.dist(other["speakerStartEye"], other["listenerStartEyes"][observer])
                require(abs(distance - expected) <= EPSILON and distance <= other["radius"] + EPSILON,
                        "attention candidate distance mismatch")
            identifiers.append(identifier)
            ranked.append((distance, speaker, identifier))
        require(max(starts) < min(ends) and max(starts) <= event["gameTime"], "attention candidates did not overlap")
        require(len(identifiers) == len(set(identifiers)), "duplicate attention candidate")
        require(event["uttId"] in identifiers, "attention candidates omit the rejected utterance")
        require(ranked == sorted(ranked), "attention evidence is not nearest-speaker ordered")
        winners = {identifier for _distance, _speaker, identifier in ranked[:slots]}
        require(event["uttId"] not in winners, "attention rejected a nearest-speaker winner")

    def item(self, item: dict, event: dict, observer: str) -> None:
        """Validate one heard/chatter/shout_seen item against its completion."""
        category = item["category"]
        require(observer != event["speakerKey"], "speaker observed own speech")
        integer(item.get("startedAt"), "speech startedAt")
        integer(item.get("attentionSlots"), "speech attentionSlots")
        for field in ("speaker", "volume", "startedAt", "gameTime", "uttId", "attentionSlots"):
            require(item.get(field) == event.get(field), f"speech {field} mismatch")
        visual_whisper = category == "chatter" and item.get("reason") == "out_of_earshot"
        if visual_whisper:
            fields = {"category", "prov", "speaker", "volume", "startedAt", "gameTime", "uttId", "attentionSlots",
                      "speakerStartEye", "speakerPos", "speakerEye", "listenerStartEye", "listenerEye",
                      "startDistance", "distance", "radius", "inStart", "inEnd", "inRadius", "reason"}
            require(set(item) <= fields | RAY_FIELDS, "visual whisper leaks content or invented fields")
            ray = {field: item[field] for field in RAY_FIELDS if field in item}
            require(ray == event.get("whisperSeen", {}).get(observer), "whisper sight differs from event attestation")
            self.whisper_sight(event, observer, ray)
            if "speakerPos" in item:
                require(close(vector(item["speakerPos"], "speakerPos"), vector(event.get("speakerPos"), "speakerPos")),
                        "whisper speakerPos mismatch")
        else:
            require(item.get("losRay") is None, "speech must not claim a LOS ray")
        radius = event["chatterRadius"] if visual_whisper else event["radius"] * (2 if category == "shout_seen" else 1)
        require(number(item.get("radius"), "speech radius") == radius, "speech radius differs from event")
        starts, ends = event["listenerStartEyes"], event["listenerEyes"]
        require(observer in ends, "observer lacks speech completion evidence")
        for field, expected in (("speakerStartEye", event["speakerStartEye"]), ("speakerEye", event["pos"]),
                                ("listenerEye", ends[observer])):
            require(close(vector(item.get(field), field), expected), f"speech {field} mismatch")
        if observer in starts:
            require(close(vector(item.get("listenerStartEye"), "listenerStartEye"), starts[observer]),
                    "speech listenerStartEye mismatch")
            start_distance = math.dist(event["speakerStartEye"], starts[observer])
            require(abs(number(item.get("startDistance"), "startDistance") - start_distance) <= EPSILON,
                    "speech start geometric distance mismatch")
        else:
            require(item.get("listenerStartEye") is None and item.get("startDistance") is None,
                    "new listener must not invent start geometry")
            start_distance = math.inf
        end_distance = math.dist(event["pos"], ends[observer])
        require(abs(number(item.get("distance"), "distance") - end_distance) <= EPSILON,
                "speech end geometric distance mismatch")
        in_start, in_end = item.get("inStart"), item.get("inEnd")
        require(type(in_start) is bool and type(in_end) is bool, "speech endpoint flags must be boolean")
        for inside, distance in ((in_start, start_distance), (in_end, end_distance)):
            require(distance <= radius + EPSILON if inside else distance >= radius - EPSILON,
                    "speech endpoint flags mismatch")
        if category == "heard":
            require(in_start and in_end and observer in event["listeners"],
                    "heard content lacks two-endpoint admission")
            require(item.get("text") == event["text"], "heard text mismatch")
            require(item.get("to") == event.get("to"), "directed speech addressee mismatch")
            require(not event.get("attentionLost", {}).get(observer), "attention-lost content was delivered")
            expected_news = []
            for source in event.get("news", []):
                require(isinstance(source, dict) and isinstance(source.get("chain"), list)
                        and type(source.get("firsthand")) is bool, "invalid source news metadata")
                require(integer(source.get("learnedAt"), "source news learnedAt") <= event["startedAt"],
                        "speaker supplied news learned after speech started")
                news = dict(source)
                chain = list(news["chain"])
                if not chain or chain[-1] != event["speaker"]:
                    chain.append(event["speaker"])
                news.update(chain=chain, firsthand=False, learnedAt=event["gameTime"])
                expected_news.append(news)
            require(item.get("news", []) == expected_news, "heard news mismatch")
            for news in expected_news:
                self._news(news, event)
            if expected_news:
                for field in ("originEventId", "chain", "firsthand", "subject", "kind", "newsId"):
                    require(item.get(field) == expected_news[0].get(field), f"relayed {field} mismatch")
                self._news(item, event)
            else:
                require(not {"originEventId", "chain", "firsthand", "subject"}.intersection(item),
                        "invented relay provenance")
        else:
            require(not {"text", "news", "originEventId", "chain", "firsthand", "subject",
                         "newsId", "kind"}.intersection(item),
                    "non-content speech leaks content or news")
            if category == "shout_seen":
                require(event["volume"] == "shout" and (in_start or in_end), "shout_seen outside conspicuity radius")
            else:
                require(observer not in event["listeners"], "chatter listener also admitted content")
                reason = item.get("reason")
                if reason == "partial_presence":
                    require(in_start != in_end, "partial-presence chatter needs exactly one endpoint")
                elif reason == "attention":
                    require(in_start and in_end, "attention chatter requires full presence")
                    require(item.get("attention") == event.get("attentionLost", {}).get(observer),
                            "attention loss evidence differs from utterance")
                    self._attention(event, observer)
                elif reason == "out_of_earshot":
                    require(visual_whisper and in_end, "visual whisper requires completion within chatter radius")
                else:
                    require(reason == "interrupted" and (in_start or in_end), "invalid chatter reason or geometry")
                    require(event.get("interrupted") is True, "interrupted chatter lacks event evidence")

    def attention_limits(self) -> None:
        """Reject content admission exceeding the slots during any shared interval."""
        listeners = {}
        for event in self._utterances.values():
            if "startedAt" in event:
                for observer in event["listeners"]:
                    listeners.setdefault(observer, []).append(event)
        for observer, events in listeners.items():
            for tick in sorted({event["startedAt"] for event in events}):
                active = [event for event in events if event["startedAt"] <= tick < event["gameTime"]]
                limits = [event["attentionSlots"] for event in active if event["attentionSlots"] > 0]
                require(not limits or len(active) <= min(limits),
                        f"{observer}: overlapping content exceeds attention slots")


class ExposureAudit:
    """Audit logged evidence offline, preserving event and delivery ordering."""

    def __init__(self):
        """Create fresh provenance indexes and per-audit delivery state."""
        self.events = {}
        self.origins = {}
        self.utterances = {}
        self._speech = SpeechAudit(self.events, self.utterances)
        self.delivered = set()
        self.violations = []

    def event(self, event: dict) -> None:
        """Validate and index one event, rejecting duplicate IDs or provenance."""
        require(isinstance(event, dict), "event must be an object")
        event_id = integer(event.get("id"), "event id", 1)
        integer(event.get("gameTime"), "event gameTime")
        kind = text(event.get("type"), "event type")
        require(event_id not in self.events, f"duplicate event id {event_id}")
        if kind == "block_change":
            vector(event.get("pos"), "change pos", block=True)
            for field in ("from", "to", "actor", "cause"):
                text(event.get(field), field)
        elif kind == "pickup":
            text(event.get("bot"), "pickup bot")
            text(event.get("item"), "pickup item")
            integer(event.get("count"), "pickup count", 1)
            entity_uuid = text(event.get("itemEntity"), "pickup itemEntity")
            require(str(uuid.UUID(entity_uuid)) == entity_uuid, "noncanonical pickup itemEntity UUID")
            vector(event.get("pos"), "pickup pos")
            require(event.get("prov") == f"entity:{entity_uuid}", "pickup entity provenance mismatch")
            if "thrower" in event:
                thrower = text(event["thrower"], "pickup thrower")
                require(str(uuid.UUID(thrower)) == thrower, "noncanonical pickup thrower UUID")
        elif kind == "block_use":
            text(event.get("actor"), "action actor")
            vector(event.get("pos"), "action pos", block=True)
            require(event.get("prov") == f"event:{event_id}", "action event provenance mismatch")
            text(event.get("block"), "used block")
            text(event.get("state"), "resulting block state")
            perceptions = event.get("perceptions")
            require(isinstance(perceptions, dict), "action perceptions must be an object")
            for observer, notice in perceptions.items():
                text(observer, "action observer")
                self.action_notice(notice, event, observer)
        elif kind == "utterance":
            utterance_id = integer(event.get("uttId"), "uttId", 1)
            require(utterance_id not in self.utterances, f"duplicate uttId {utterance_id}")
            speaker = text(event.get("speaker"), "speaker")
            require(isinstance(event.get("text"), str) and ("startedAt" in event or len(event["text"]) <= 256),
                    "invalid utterance text")
            vector(event.get("pos"), "speaker eye")
            require(number(event.get("radius"), "speech radius") >= 0, "negative speech radius")
            listeners = event.get("listeners")
            require(isinstance(listeners, list), "listeners must be a list")
            for listener in listeners:
                text(listener, "listener")
            require(len(listeners) == len(set(listeners)), "duplicate listener")
            if "startedAt" not in event and "volume" not in event:
                require(speaker not in listeners, "speaker listed as own listener")
            listener_eyes = event.get("listenerEyes")
            if "startedAt" in event or "volume" in event:
                self._speech.event(event)
            else:
                require(isinstance(listener_eyes, dict) and set(listener_eyes) == set(listeners),
                        "listenerEyes must match listeners")
                for listener, listener_eye in listener_eyes.items():
                    vector(listener_eye, f"listenerEyes {listener}")
                    require(math.dist(event["pos"], listener_eye) <= event["radius"] + EPSILON,
                            "utterance listener outside radius")
            self.utterances[utterance_id] = event
        elif kind in ("origin", "doc_write"):
            provenance = text(event.get("prov"), "origin/write prov")
            require(provenance not in self.origins, f"duplicate provenance registration {provenance}")
            if kind == "origin":
                origin_kind = event.get("kind")
                require(origin_kind in {"entity", "block", "document", "container", "structure"}, "invalid origin kind")
                if origin_kind == "entity":
                    entity_uuid = text(event.get("uuid"), "entity UUID")
                    require(str(uuid.UUID(entity_uuid)) == entity_uuid, "noncanonical entity UUID")
                    require(provenance == f"entity:{entity_uuid}", "entity origin UUID mismatch")
                elif origin_kind in {"block", "container"}:
                    match = POSITION_PROV.fullmatch(provenance)
                    require(match is not None and match[1] == origin_kind, "invalid position origin prov")
                    require([int(match[index]) for index in (2, 3, 4)] ==
                            vector(event.get("pos"), "origin pos", block=True), "origin position mismatch")
                    require(origin_kind != "container" or match[5] != "base", "container requires a version")
                elif origin_kind == "structure":
                    name = text(event.get("name"), "structure name")
                    require(provenance == f"structure:{name}", "structure origin name mismatch")
                else:
                    match = DOCUMENT_PROV.fullmatch(provenance)
                    require(match is not None, "invalid document origin prov")
                    if "version" in event:
                        require(integer(event["version"], "origin document version") == int(match[2]),
                                "origin document version mismatch")
                    if "docId" in event:
                        require(event["docId"] == match[1], "origin docId mismatch")
                    if "pos" in event:
                        vector(event["pos"], "document origin pos", block=True)
            else:
                doc_id = text(event.get("docId"), "docId")
                version = integer(event.get("version"), "document version", 1)
                require(provenance == f"doc:{doc_id}@v{version}", "doc_write provenance mismatch")
                text(event.get("author"), "document author")
                text(event.get("kind"), "document kind")
                digest = event.get("contentSha256")
                require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                        "invalid document contentSha256")
                if "pos" in event:
                    vector(event["pos"], "document write pos", block=True)
            self.origins[provenance] = event
        self.events[event_id] = event

    def resolve(self, item: dict, category: str, when: int) -> dict:
        """Resolve item provenance to compatible evidence available by this tick."""
        provenance = text(item.get("prov"), "item prov")
        if category in {"entity", "document", "inventory_document", "container"} or (
                category == "block" and provenance.endswith("@base")):
            event = self.origins.get(provenance)
            require(event is not None, f"unresolved provenance {provenance}")
            expected_kind = "document" if category == "inventory_document" else category
            require((event["type"] == "origin" and event["kind"] == expected_kind) or
                    (expected_kind == "document" and event["type"] == "doc_write"),
                    f"wrong origin kind for {provenance}")
            if category == "entity":
                uuid = item["uuid"] if "uuid" in item else item.get("itemEntity")  # drop progress: itemEntity
                require(uuid == event["uuid"], "entity UUID differs from origin")
            if category in {"block", "container"} or (category == "document" and "pos" in event):
                require(item.get("pos") == event["pos"], "item position differs from origin")
            if category == "block" and "block" in event:
                require(item.get("block") == event["block"], "baseline block differs from origin")
        elif category == "block":
            match = POSITION_PROV.fullmatch(provenance)
            require(match is not None and match[1] == "block" and match[5] != "base", "invalid block prov")
            require(item.get("pos") == [int(match[index]) for index in (2, 3, 4)], "block prov position mismatch")
            event = self.events.get(int(match[5]))
            require(event is not None and event["type"] == "block_change", "block prov is not a block_change")
            require(item["pos"] == event["pos"] and item.get("block") == event["to"], "block change mismatch")
        elif category == "structure":
            event = self.origins.get(provenance)
            require(event is not None and event["type"] == "origin" and event["kind"] == "structure",
                    f"unresolved structure provenance {provenance}")
            require(item.get("name") == event.get("name"), "structure name differs from origin")
        elif category == "change":
            integer(item.get("gameTime"), "change gameTime")
            require(re.fullmatch(r"change:[1-9]\d*", provenance) is not None, "invalid change prov")
            event = self.events.get(int(provenance.split(":")[1]))
            require(event is not None and event["type"] == "block_change", "unresolved change provenance")
            for field in ("pos", "from", "to", "gameTime"):
                require(item.get(field) == event.get(field), f"change {field} mismatch")
        elif category == "seen_action":
            action = item.get("action")
            require(action in {"chest_take", "chest_put", "give", "drop", "pickup"}, "invalid seen action")
            source_kind = action if action in {"drop", "pickup"} else "transfer"
            require(re.fullmatch(rf"{source_kind}:[1-9]\d*", provenance) is not None,
                    "invalid seen action source prov")
            event = self.events.get(int(provenance.split(":")[1]))
            require(event is not None and event["type"] == source_kind, "unresolved seen action source provenance")
            actor = text(item.get("actor"), "seen action actor")
            integer(item.get("gameTime"), "seen action gameTime")
            text(item.get("item"), "seen action item")
            fields = SEEN_ACTION_FACTS | {"category", "radius", "inRadius", "observerEye", "actorSight"}
            if action in {"chest_take", "chest_put"}:
                position = vector(item.get("pos"), "container action pos", block=True)
                container = "container:" + ",".join(str(coordinate) for coordinate in position)
                source, target = (container, actor) if action == "chest_take" else (actor, container)
                require(event.get("from") == source and event.get("to") == target, "transfer participants mismatch")
                fields |= {"pos", "distance", *RAY_FIELDS}
            elif action == "give":
                target = text(item.get("target"), "give target")
                require(event.get("from") == actor and event.get("to") == target, "transfer participants mismatch")
                fields |= {"target", "targetSight"}
            else:
                position = vector(item.get("pos"), "item action pos")
                require(event.get("bot") == actor, "item action actor mismatch")
                require(position == event.get("pos"), "item action position mismatch")
                fields.add("pos")
            for field in ("gameTime", "item"):
                require(item[field] == event.get(field), f"seen action {field} mismatch")
            require(set(item) <= fields, "private or invented seen action details exposed")
        elif category == "heard" and provenance.startswith("event:"):
            require(re.fullmatch(r"event:[1-9]\d*", provenance) is not None, "invalid action event prov")
            event = self.events.get(int(provenance.split(":")[1]))
            require(event is not None and event["type"] == "block_use", "unresolved action event provenance")
            require(event.get("prov") == provenance, "action event provenance mismatch")
            integer(item.get("gameTime"), "action gameTime")
            vector(item.get("pos"), "action notice pos", block=True)
            for field, expected in (("kind", event["type"]), ("pos", event.get("pos")),
                                    ("gameTime", event["gameTime"])):
                require(item.get(field) == expected, f"action {field} mismatch")
        elif category in SPEECH:
            integer(item.get("gameTime"), "heard gameTime")
            integer(item.get("uttId"), "heard uttId", 1)
            number(item.get("radius"), "heard radius")
            require(re.fullmatch(r"utt:[1-9]\d*", provenance) is not None, "invalid utterance prov")
            utterance_id = int(provenance.split(":")[1])
            event = self.utterances.get(utterance_id)
            require(event is not None, "unresolved utterance provenance")
            fields = ("uttId", "speaker", "gameTime") if "startedAt" in event else (
                "uttId", "speaker", "text", "gameTime", "radius")
            for field in fields:
                require(item.get(field) == event.get(field), f"heard {field} mismatch")
        else:
            raise AuditError(f"unhandled provenance category {category}")
        require(event["gameTime"] <= when, "provenance refers to a future event")
        return event

    def action_notice(self, item: dict, event: dict, observer: str) -> None:
        """Require an exact attested notice and its occurrence-time sensory geometry."""
        require(isinstance(item, dict), "action notice must be an object")
        notice = {field: value for field, value in item.items() if field not in {"category", "inRadius"}}
        perceptions = event.get("perceptions")
        require(isinstance(perceptions, dict) and observer in perceptions,
                "action notice lacks observer perception attestation")
        require(notice == perceptions[observer], "action notice differs from event perception")
        sense = notice.get("sense")
        require(sense in {"sight", "hearing"}, "invalid action sense")
        fields = {"prov", "gameTime", "kind", "sense", "speaker", "text", "pos", "radius",
                  "listenerEye", "distance", "inStart", "inEnd"}
        require(set(notice) == fields | (RAY_FIELDS if sense == "sight" else {"losRay"}),
                "action notice has missing or invented fields")
        require(notice["prov"] == event.get("prov") == f"event:{event['id']}", "action notice provenance mismatch")
        require(integer(notice["gameTime"], "action gameTime") == event["gameTime"], "action gameTime mismatch")
        require(notice["kind"] == event["type"] == "block_use", "action kind mismatch")
        position = vector(notice["pos"], "action pos", block=True)
        require(position == event.get("pos"), "action pos mismatch")
        listener_eye = vector(notice["listenerEye"], "action listenerEye")
        distance = number(notice["distance"], "action distance")
        target = [coordinate + 0.5 for coordinate in position]
        require(abs(distance - math.dist(listener_eye, target)) <= EPSILON, "action geometric distance mismatch")
        radius = number(notice["radius"], "action radius")
        require(radius >= 0 and 0 <= distance <= radius + EPSILON, "action outside sensory radius")
        require(notice["inStart"] is True and notice["inEnd"] is True, "action endpoint flags must be true")
        text(notice["text"], "action text")
        position_text = "(" + ", ".join(map(str, position)) + ")"
        if sense == "hearing":
            require(event["type"] == "block_use" and event.get("block") == "minecraft:bell",
                    "only bell actions may be heard without LOS")
            require(radius == 48 and notice["speaker"] == "bell" and notice["losRay"] is None,
                    "invalid bell hearing evidence")
            expected_text = "bell rang at " + position_text
            require(notice["text"] == expected_text, "bell notice invents words")
        else:
            actor = text(event.get("actor"), "action actor")
            require(notice["speaker"] == actor, "action speaker mismatch")
            block = text(event.get("block"), "used block").removeprefix("minecraft:")
            opened = re.search(r"[\[,]open=(true|false)[,\]]", text(event.get("state"), "resulting block state"))
            now = f"; it is now {'open' if opened.group(1) == 'true' else 'closed'}" if opened else ""
            expected_text = f"{actor} used the {block} at {position_text}{now}"
            require(notice["text"] == expected_text, "visual action notice invents words")
            self.spatial(notice, "block", listener_eye)

    def spatial(self, item: dict, category: str, eye: list) -> None:
        """Validate recorded distance, radius, reach, and line-of-sight evidence."""
        position = vector(item.get("pos"), "target pos", block=category != "entity")
        target = position if category == "entity" else [coordinate + 0.5 for coordinate in position]
        distance = number(item.get("distance"), "distance")
        geometric = (math.dist([eye[0], eye[2]], [target[0], target[2]])
                     if category == "terrain" else math.dist(eye, target))
        require(abs(distance - geometric) <= EPSILON, "geometric distance mismatch")
        if category == "terrain":
            require(all(item.get(field) == position[index] for index, field in enumerate(("x", "y", "z"))),
                    "terrain coordinates differ from pos")
        radius = number(item.get("radius"), "radius")
        require(radius >= 0 and 0 <= distance <= radius + EPSILON, "target outside radius")
        sight_ray(item, eye, 1 if category == "entity" else 6)
        ray, ray_to = item["losRay"], item["rayTo"]
        if category != "entity":
            expected = list(target)
            if ray:
                axis = (ray - 1) // 2
                expected[axis] += -0.49 if ray % 2 else 0.49
            require(close(ray_to, expected), "rayTo differs from the specified block ray")
        else:
            target_eye = vector(item.get("targetEye"), "targetEye")
            target_center = vector(item.get("targetCenter"), "targetCenter")
            require(close(ray_to, (target_eye, target_center)[ray]), "entity ray differs from API target")
        hit_location, hit_pos, hit_type = item["hitLocation"], item["hitPos"], item["hitType"]
        if category == "entity" or item.get("emptyTarget") is True:
            require(math.dist(hit_location, ray_to) <= 0.05 + EPSILON, "LOS endpoint obstructed")
        else:
            require(hit_type == "BLOCK" and hit_pos == position, "LOS did not hit target block")
            require(all(position[axis] - EPSILON <= hit_location[axis] <= position[axis] + 1 + EPSILON
                        for axis in range(3)), "block hitLocation outside target")
        if category in {"document", "container"}:
            require(item.get("reachOk") is True, "reachOk must be true")

    def item(self, item: dict, exposure: dict) -> None:
        """Validate one exposed item against its category and provenance."""
        require(isinstance(item, dict), "item must be an object")
        category = item.get("category")
        require(category in CATEGORIES[exposure["kind"]], "invalid item category for exposure kind")
        require(item.get("inRadius", exposure["kind"] == "communication") is True, "inRadius must be true")
        if category in {"self", "terrain"}:
            require("prov" not in item, "ambient/self item must not have provenance")
            event = None
        else:
            event = self.resolve(item, category, exposure["gameTime"])
        if category == "seen_action":
            require(item["actor"] != exposure["observer"], "actor cannot witness its own action")
            eye = vector(item.get("observerEye"), "action-time observerEye")
            radius = number(item.get("radius"), "seen action radius")
            require(radius >= 0, "negative seen action radius")
            container_action = item["action"] in {"chest_take", "chest_put"}
            if container_action:
                self.spatial(item, category, eye)
            else:
                fields = SEEN_ACTION_FACTS | {"category", "radius", "inRadius", "observerEye", "actorSight"}
                fields |= {"target", "targetSight"} if item["action"] == "give" else {"pos"}
                require(set(item) == fields, "seen action has missing or invented fields")
            for field in (("actorSight", "targetSight") if item["action"] == "give" else ("actorSight",)):
                sight = item.get(field)
                require(isinstance(sight, dict), f"missing {field} evidence")
                if not container_action:
                    require(set(sight) == ENTITY_SIGHT_FIELDS, f"invalid {field} fields")
                    require(sight["emptyTarget"] is False, f"{field} must target an entity")
                require(sight.get("category") == "entity", f"{field} must be entity sight")
                require(sight.get("inRadius") is True, f"{field} must be in sight radius")
                self.spatial(sight, "entity", eye)
                require(sight["radius"] == radius, f"{field} radius differs from envelope")
            delivery = (exposure["observer"], item["prov"])
            require(delivery not in self.delivered, "seen action delivered twice to observer")
            self.delivered.add(delivery)
        elif category in SPATIAL:
            self.spatial(item, category, exposure["observerEye"])
            if category == "document" and exposure["kind"] == "open_container":
                self.container_document(item, exposure)
        elif category == "heard" and event["type"] == "block_use":
            self.action_notice(item, event, exposure["observer"])
            delivery = (exposure["observer"], item["prov"])
            require(delivery not in self.delivered, "action notice delivered twice to observer")
            self.delivered.add(delivery)
        elif category in SPEECH and "startedAt" in event:
            self._speech.item(item, event, exposure["observer"])
            channel = "shout_seen" if category == "shout_seen" else "content"
            delivery = (exposure["observer"], event["uttId"], channel)
            require(delivery not in self.delivered, "speech item delivered twice to observer")
            self.delivered.add(delivery)
        elif category == "heard":
            observer = exposure["observer"]
            require(observer != event["speaker"] and observer in event["listeners"],
                    "observer not an utterance listener")
            distance = number(item.get("distance"), "speech distance")
            require(0 <= distance <= event["radius"] + EPSILON, "speech outside radius")
            speaker_eye = vector(item.get("speakerEye"), "speakerEye")
            listener_eye = vector(item.get("listenerEye"), "listenerEye")
            require(close(speaker_eye, event["pos"]) and close(listener_eye, event["listenerEyes"][observer]),
                    "speech-time eye mismatch")
            require(abs(distance - math.dist(speaker_eye, listener_eye)) <= EPSILON,
                    "speech geometric distance mismatch")
            require("losRay" in item and item["losRay"] is None, "speech must not claim a LOS ray")
            delivery = (observer, event["uttId"])
            require(delivery not in self.delivered, "utterance delivered twice to observer")
            self.delivered.add(delivery)
        elif category == "inventory_document":
            require(item.get("ownedBy") == exposure["observer"], "inventory document not owned by observer")
            integer(item.get("slot"), "inventory slot")
            require(number(item.get("distance"), "inventory distance") == 0, "inventory distance must be zero")
            require(item.get("reachOk") is True and "losRay" in item and item["losRay"] is None,
                    "invalid inventory admission")
        elif category == "self":
            require(item.get("name") == exposure["observer"] and item.get("ownedBy") == exposure["observer"],
                    "self identity differs from observer")
            vector(item.get("pos"), "self pos")
            require(number(item.get("distance"), "self distance") == 0 and item.get("reachOk") is True
                    and "losRay" in item and item["losRay"] is None, "invalid intrinsic self admission")
        self.nested_documents(item, exposure["gameTime"])

    def nested_documents(self, value, when: int) -> None:
        """Recursively resolve provenance embedded in exposed data."""
        if isinstance(value, dict):
            if "prov" in value and str(value["prov"]).startswith("event:") and value.get("category") != "heard":
                # e.g. an actor's own skill progress citing the block_use event it caused: a reference, not a notice
                provenance = text(value["prov"], "nested prov")
                event = (self.events.get(int(provenance.split(":")[1]))
                         if re.fullmatch(r"event:[1-9]\d*", provenance) else None)
                require(event is not None and event.get("prov") == provenance, "unresolved nested event provenance")
            elif "prov" in value:
                provenance = text(value["prov"], "nested prov")
                category = {"entity": "entity", "block": "block", "doc": "inventory_document",
                            "change": "change", "utt": value.get("category", "heard"),
                            "event": "heard", "container": "container", "structure": "structure",
                            "transfer": "seen_action", "drop": "seen_action", "pickup": "seen_action"}.get(
                                provenance.split(":")[0])
                require(category is not None, "unknown nested provenance")
                self.resolve(value, category, when)
            for key, child in value.items():
                if key == "doc":
                    require(isinstance(child, dict), "nested doc must be an object")
                    text(child.get("prov"), "nested doc prov")
                if isinstance(child, (dict, list)):
                    self.nested_documents(child, when)
        elif isinstance(value, list):
            for child in value:
                self.nested_documents(child, when)

    def container_document(self, item: dict, exposure: dict) -> None:
        """Require exposed container documents to belong to the logged container."""
        provenance = text(item.get("containerProv"), "containerProv")
        slot = integer(item.get("slot"), "container document slot")
        parents = [candidate for candidate in exposure["items"] if isinstance(candidate, dict)
                   and candidate.get("category") == "container" and candidate.get("prov") == provenance]
        require(len(parents) == 1, "container document lacks an admitted parent container")
        parent = parents[0]
        require(item.get("pos") == parent.get("pos"), "container document position differs from parent")
        require(isinstance(parent.get("items"), list), "container items must be a list")
        matching = [entry for entry in parent["items"] if isinstance(entry, dict) and entry.get("slot") == slot]
        require(len(matching) == 1 and isinstance(matching[0].get("doc"), dict)
                and matching[0]["doc"].get("prov") == item.get("prov"),
                "container document does not match parent slot doc")

    def exposure(self, exposure: dict, index: int) -> None:
        """Validate an observation and retain individually labeled item failures."""
        require(isinstance(exposure, dict), "exposure must be an object")
        integer(exposure.get("gameTime"), "exposure gameTime")
        text(exposure.get("observer"), "observer")
        vector(exposure.get("observerEye"), "observerEye")
        require(exposure.get("kind") in CATEGORIES, "unknown exposure kind")
        require(isinstance(exposure.get("items"), list), "items must be a list")
        for item_index, item in enumerate(exposure["items"], 1):
            try:
                self.item(item, exposure)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.violations.append(f"observation {index} item {item_index}: {error}")


    def audit_logs(self, events: list[dict], observations: list[dict]) -> list[str]:
        """Index events before checking observations and return ordered violations."""
        if not events:
            self.violations.append("events log is empty")
        if not observations:
            self.violations.append("observations log is empty")
        for index, event in enumerate(events, 1):
            try:
                self.event(event)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.violations.append(f"event {index}: {error}")
        for index, exposure in enumerate(observations, 1):
            try:
                self.exposure(exposure, index)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.violations.append(f"observation {index}: {error}")
        try:
            self._speech.attention_limits()
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.violations.append(f"attention: {error}")
        return self.violations

    def run(self, directory: Path) -> list[str]:
        """Read a run's logs and retain read or parse failures as violations."""
        logs = Path(directory) / "world" / "simharness"
        try:
            return self.audit_logs(read_jsonl(logs / "events.jsonl"), read_jsonl(logs / "observations.jsonl"))
        except (OSError, ValueError) as error:
            self.violations = [str(error)]
            return self.violations

    def report(self) -> int:
        """Print ordered violations and the original audit verdict and exit code."""
        for violation in self.violations:
            print(f"FAIL audit: {violation}")
        print(f"{'FAIL' if self.violations else 'PASS'} O1 audit: {len(self.violations)} violations")
        return int(bool(self.violations))


Auditor = ExposureAudit


def audit_logs(events: list[dict], observations: list[dict]) -> list[str]:
    """Audit in-memory records through the compatible fresh-auditor API."""
    return ExposureAudit().audit_logs(events, observations)


def audit_run(directory: Path) -> list[str]:
    """Audit a run directory through the compatible fresh-auditor API."""
    return ExposureAudit().run(directory)


def main(argv: list[str] | None = None) -> int:
    """Parse the original audit CLI and report one offline exposure audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=RUNS_DIR / "o1")
    args = parser.parse_args(argv)
    audit = ExposureAudit()
    audit.run(args.run)
    return audit.report()


if __name__ == "__main__":
    raise SystemExit(main())
