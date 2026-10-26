"""Offline welfare, economy and promise accounting for a finished village or study run.

Run ``python3.11 scripts/village_metrics.py RUN`` for the free keyword baseline.
Only the optional LLM promise extractor calls a model; neither mode starts a server.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from statistics import mean, median
import sys
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.llm import LLMClient  # noqa: E402
from orchestrator.scenario import KeywordJudge, TICKS_PER_DAY  # noqa: E402

EMERALD = "minecraft:emerald"
KINDS = ("pay", "give", "deliver", "build", "repair", "open_fair", "report", "other")
STATUSES = ("kept", "broken", "open", "unverifiable")


def item_id(item: str) -> str:
    """Normalize a written item name to the harness's namespaced item id."""
    item = item.lower().strip().replace(" ", "_")
    if item in {"emeralds", "emerald"}:
        return EMERALD
    return item if ":" in item else f"minecraft:{item}"


@dataclass
class Promise:
    """An extracted commitment with its original source and optional contract condition."""

    promiser: str
    promisee: str | None
    kind: str
    item: str | None
    count: int | None
    structure: str | None
    deadline_tick: int | None
    quote: str
    tick: int
    event_ids: list = field(default_factory=list)
    contract_id: str | None = None
    condition: str | None = None


class PromiseExtractor(Protocol):
    """Small shared interface for extracting commitments from one speaker's daily sources."""

    def extract(self, sources: list[dict]) -> list[Promise]:
        """Return source-grounded commitments from this speaker and day."""
        ...


class KeywordPromiseExtractor:
    """A deliberately conservative, free first-person commitment extractor."""

    COMMITMENT = re.compile(
        r"\bI(?:\s+will|['’]ll|\s+promise\s+to)\s+"
        r"(?P<verb>pay|give|bring|deliver|build|repair|finish|open|report)\b[^.!?\n]*", re.I)

    def __init__(self, recipients: dict[str, str] | None = None):
        """Recognize role names and named delivery containers without guessing recipients."""
        self._recipients = recipients or {name: name for name in
                                         ("mayor", "builder", "freight", "security", "envoy", "investigator")}

    def extract(self, sources: list[dict]) -> list[Promise]:
        """Find explicit first-person promises, retaining the exact matched quote."""
        promises = []
        for source in sources:
            for match in self.COMMITMENT.finditer(source["text"]):
                quote, verb = match.group(), match["verb"].lower()
                lower = quote.lower()
                kind = {"bring": "deliver", "finish": "build", "open": "open_fair"}.get(verb, verb)
                if verb == "open" and "fair" not in lower:
                    kind = "other"
                recipient = self._recipient(quote, verb, source)
                quantity = re.search(r"\b(\d+)\s+((?:minecraft:)?[a-z_]+(?:\s+(?:planks|logs|pickaxe|wall))?)", lower)
                count = int(quantity[1]) if quantity and quantity[2] not in {"ticks", "spans", "sections"} else None
                item = item_id(quantity[2]) if count is not None else EMERALD if kind == "pay" else None
                structure = None
                if kind in {"build", "repair"}:
                    section = re.search(r"bridge_span_\d+", lower)
                    structure = (section[0] if section else "bridge_complete" if "bridge" in lower
                                 else "port_repair" if "port" in lower or "dock" in lower else None)
                promises.append(Promise(source["speaker"], recipient, kind, item, count, structure,
                                        self._deadline(quote, source["tick"]), quote, source["tick"],
                                        list(source["event_ids"])))
        return promises

    def _recipient(self, quote: str, verb: str, source: dict) -> str | None:
        """Prefer an explicit grammatical recipient over unrelated role mentions and the addressee."""
        for name, target in sorted(self._recipients.items(), key=lambda entry: len(entry[0]), reverse=True):
            written = re.escape(name.replace("_", " "))
            if (target != source["speaker"] and
                    re.search(r"\b(?:" + verb + r"|to)\s+(?:the\s+)?" + written + r"\b", quote, re.I)):
                return target
        return source.get("to")

    @staticmethod
    def _deadline(text: str, tick: int) -> int | None:
        """Read an explicit tick, end of a numbered day, or same-day 24-hour time."""
        explicit = re.search(r"\bby\s+tick\s+(\d+)", text, re.I)
        if explicit:
            return int(explicit[1])
        day = re.search(r"\bby\s+day\s+(\d+)", text, re.I)
        clock = re.search(r"\b(?:by|at)\s+(\d{1,2}):(\d{2})\b", text, re.I)
        if clock and int(clock[1]) < 24 and int(clock[2]) < 60:
            return ((int(day[1]) if day else tick // TICKS_PER_DAY) * TICKS_PER_DAY
                    + int(clock[1]) * 1000 + int(clock[2]) * 1000 // 60)
        return (int(day[1]) + 1) * TICKS_PER_DAY - 1 if day else None


class LLMPromiseExtractor:
    """Extract commitments with one model call per speaker/day and reject invented quotes."""

    def __init__(self, client: LLMClient):
        """Keep any existing model client and a count of rejected, ungrounded quotes."""
        self._client = client
        self.dropped_quotes = 0

    def extract(self, sources: list[dict]) -> list[Promise]:
        """Request structured commitments, then anchor each quote to a real source."""
        properties = {name: {"type": ["string", "null"]} for name in
                      ("promisee", "item", "structure")}
        properties.update({"promiser": {"type": "string"}, "quote": {"type": "string"},
                           "kind": {"type": "string", "enum": list(KINDS)},
                           "count": {"type": ["integer", "null"]},
                           "deadline_tick": {"type": ["integer", "null"]}})
        tool = {"name": "commitments", "description": "Record only explicit, quoted commitments.",
                "parameters": {"type": "object", "required": ["commitments"], "properties": {
                    "commitments": {"type": "array", "items": {"type": "object", "properties": properties,
                                                              "required": list(properties)}}}}}
        system = ("Extract explicit commitments made by the speaker from these utterances and documents. "
                  "Quote a verbatim contiguous passage from one source. Do not infer promises from claims of past "
                  "delivery. Use canonical role/container and minecraft item ids. A day is 24000 ticks; by day N "
                  "means through tick (N+1)*24000-1. Use null for missing fields or an unspecified deadline.")
        response = self._client.complete(system, json.dumps(sources, ensure_ascii=False), [tool])
        rows = response.tool_call.args.get("commitments", []) if response.tool_call else []
        if not response.tool_call and response.text:
            try:
                rows = json.loads(response.text).get("commitments", [])
            except (ValueError, AttributeError):
                rows = []
        promises = []
        for row in rows:
            quote = row.get("quote")
            source = next((entry for entry in sources if isinstance(quote, str) and quote
                           and quote in entry["text"] and row.get("promiser") == entry["speaker"]), None)
            if source is None:
                self.dropped_quotes += 1
                continue
            count, deadline = row.get("count"), row.get("deadline_tick")
            if count is not None and (type(count) is not int or count <= 0):
                count = None
            if deadline is not None and type(deadline) is not int:
                deadline = None
            promises.append(Promise(source["speaker"], row.get("promisee"),
                                    row.get("kind") if row.get("kind") in KINDS else "other",
                                    item_id(row["item"]) if row.get("item") else None, count,
                                    row.get("structure"), deadline, quote, source["tick"],
                                    list(source["event_ids"])))
        return promises


class VillageMetrics:
    """Read finished-run artifacts and write reproducible, server-free metrics.json."""

    def __init__(self, run_dir: Path | str, extractor: PromiseExtractor | None = None):
        """Load the manifest and offline logs; report files supply the authoritative end tick."""
        self.directory = Path(run_dir)
        self._scenario = self._json("scenario.json")
        self._report = next((self._json(name) for name in ("m5_report.json", "narrated_report.json", "outcome.json")
                             if (self.directory / name).exists()), {})
        self._events = sorted(self._jsonl("world/simharness/events.jsonl"), key=lambda entry: entry["gameTime"])
        self._observations = sorted(self._jsonl("world/simharness/observations.jsonl"),
                                    key=lambda entry: entry["gameTime"])
        self._decisions = sorted(self._jsonl("orchestrator/decisions.jsonl"),
                                 key=lambda entry: entry["game_time"])
        self._roles = [role["name"] for role in self._scenario.get("roles", [])]
        self._containers = self._scenario.get("containers", {})
        self._recipients = {name: name for name in self._roles}
        self._recipients.update({name: self._container_id(container) for name, container in self._containers.items()})
        if "fair_delivery" in self._containers:
            self._recipients["fairgrounds"] = self._recipients["fair_delivery"]
        self._extractor = extractor or KeywordPromiseExtractor(self._recipients)
        latest = max([0, *[entry["gameTime"] for entry in self._events + self._observations],
                      *[entry["game_time"] for entry in self._decisions]])
        self._end = self._report.get("end", latest)
        self._events = [entry for entry in self._events if entry["gameTime"] <= self._end]
        self._documents = self._written_documents()
        self._first_true = dict(self._report.get("first_true", {}))
        self._completion_events = self._section_completions()

    def _json(self, name: str) -> dict:
        """Read one JSON artifact, returning an empty object for an optional missing report."""
        path = self.directory / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _jsonl(self, name: str) -> list[dict]:
        """Read nonempty rows from an optional harness or orchestrator JSONL artifact."""
        path = self.directory / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _container_id(container: dict) -> str:
        """Return the transfer endpoint for a manifest container."""
        return "container:" + ",".join(str(value) for value in container["pos"])

    @staticmethod
    def gini(holdings: list[int]) -> float:
        """Compute population Gini, with zero for an empty or all-zero population."""
        ordered, total = sorted(holdings), sum(holdings)
        return (sum((2 * index - len(ordered) - 1) * value for index, value in enumerate(ordered, 1))
                / (len(ordered) * total)) if total else 0.0

    def compute(self) -> dict:
        """Compute and persist daily welfare/economy, checked promises and investigation scores."""
        days = set(range(max(1, (self._end + TICKS_PER_DAY - 1) // TICKS_PER_DAY)))
        days.update(event["gameTime"] // TICKS_PER_DAY for event in self._events
                    if event["type"] in {"purchase", "transfer"})
        result = {"end_tick": self._end, "welfare": self._welfare(),
                  "economy": {"days": {str(day): self._economy_day(day) for day in sorted(days)},
                              "totals": self._transactions(self._events)},
                  "promises": self._promises(), "investigation_accuracy": self._investigation(),
                  "outcomes": self._report.get("outcomes", self._report.get("roles",
                                                     self._report.get("shadow_outcomes", {}))),
                  "first_true": self._first_true,
                  "unavailable_data": [name for name in ("npc_metrics", "first_true") if name not in self._report]}
        (self.directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    def _welfare(self) -> dict:
        """Summarize observed commutes and scheduled attendance, leaving empty denominators unknown."""
        days = {}
        for entry in self._report.get("npc_metrics", []):
            if entry["day"] * TICKS_PER_DAY >= self._end:
                continue
            npcs = entry["npcs"]
            commutes = [npc["commuteTicks"] for npc in npcs if npc.get("commuteTicks") is not None]
            workers = [npc for npc in npcs if npc.get("workStatus", "not_scheduled") != "not_scheduled"]
            market = [npc for npc in npcs if npc.get("marketStatus", "not_scheduled") != "not_scheduled"]
            children = [npc for npc in npcs if npc.get("schoolStatus", "not_scheduled") != "not_scheduled"]
            days[str(entry["day"])] = {
                "mean_commute_ticks": mean(commutes) if commutes else None,
                "median_commute_ticks": median(commutes) if commutes else None,
                "work_on_time_share": sum(npc.get("workOnTime", False) for npc in workers) / len(workers)
                if workers else None,
                "market_visited_share": sum(npc.get("marketVisited", False) for npc in market) / len(market)
                if market else None,
                "school_reached_share": sum(npc["schoolStatus"] in {"on_time", "late"} for npc in children)
                / len(children) if children else None}
        return days

    @staticmethod
    def _transactions(events: list[dict]) -> dict:
        """Count purchases, emerald flows by endpoint pair and other transferred items."""
        purchases = {"count": 0, "emeralds_spent": 0, "by_buyer": {}, "by_item": {}}
        transfers = {"count": 0, "emeralds": {"count": 0, "total": 0, "by_pair": []}, "by_item": {}}
        pairs = {}
        for event in events:
            if event["type"] == "purchase":
                purchases["count"] += 1
                purchases["emeralds_spent"] += event["cost"]
                for key, name in (("by_buyer", event["buyer"]), ("by_item", event["item"])):
                    group = purchases[key].setdefault(name, {"count": 0, "emeralds_spent": 0})
                    group["count"] += 1
                    group["emeralds_spent"] += event["cost"]
                    if key == "by_item":
                        group["items"] = group.get("items", 0) + event["count"]
            elif event["type"] == "transfer":
                transfers["count"] += 1
                if event["item"] == EMERALD:
                    emeralds = transfers["emeralds"]
                    emeralds["count"] += 1
                    emeralds["total"] += event["count"]
                    group = pairs.setdefault((event["from"], event["to"]),
                                             {"from": event["from"], "to": event["to"], "count": 0, "total": 0})
                else:
                    group = transfers["by_item"].setdefault(event["item"], {"count": 0, "total": 0})
                group["count"] += 1
                group["total"] += event["count"]
        transfers["emeralds"]["by_pair"] = list(pairs.values())
        return {"purchases": purchases, "transfers": transfers}

    def _economy_day(self, day: int) -> dict:
        """Use half-open daily windows; the last day ends at the run's actual cutoff."""
        start, boundary = day * TICKS_PER_DAY, (day + 1) * TICKS_PER_DAY
        events = [entry for entry in self._events if start <= entry["gameTime"] < boundary]
        result = self._transactions(events)
        holdings = self._holdings(min(boundary, self._end + 1))
        treasury = self._containers.get("treasury")
        balance = treasury.get("stock", {}).get(EMERALD, 0) if treasury else None
        if treasury:
            endpoint = self._container_id(treasury)
            balance += sum(event["count"] * ((event["to"] == endpoint) - (event["from"] == endpoint))
                           for event in self._events if event["type"] == "transfer" and event["item"] == EMERALD
                           and event["gameTime"] < boundary)
        prices, invalid = self._prices(boundary)
        result.update({"posted_prices": prices, "unparseable_sign_edits": invalid, "holdings": holdings,
                       "treasury": balance, "gini": self.gini(list(holdings.values()))})
        return result

    PROMPT_INVENTORY = re.compile(r"^Inventory:\n((?:  .*\n?)*)", re.M)

    def _holdings(self, boundary: int) -> dict:
        """Each role's emeralds just before `boundary`: the count its latest prompt before then showed (its starting
        kit if it has not decided yet), plus the logged emerald transfers and purchases after that prompt."""
        holdings = {}
        for role in self._scenario.get("roles", []):
            name, anchor, count = role["name"], -1, role.get("kit", {}).get(EMERALD, 0)
            for decision in self._decisions:
                if decision["agent"] == name and decision["game_time"] < boundary:
                    shown = self.PROMPT_INVENTORY.search(decision.get("user_prompt", ""))
                    if shown:
                        anchor = decision["game_time"]
                        count = sum(int(match[1]) for match in re.finditer(r"slot \d+: (\d+) emerald\b", shown[1]))
            for event in self._events:
                if not anchor < event["gameTime"] < boundary:
                    continue
                if event["type"] == "transfer" and event["item"] == EMERALD:
                    count += event["count"] * ((event["to"] == name) - (event["from"] == name))
                elif event["type"] == "purchase" and event["buyer"] == name:
                    count -= event["cost"]
            holdings[name] = count
        return holdings

    def _written_documents(self) -> list[dict]:
        """Join successful writing decisions to doc_write events, never using hashed event text."""
        documents, used = [], set()
        for decision in self._decisions:
            if (decision["tool"] not in {"write_book", "write_sign"}
                    or not self._successful_decision(decision)):
                continue
            args, speaker = decision["args"], decision["agent"]
            sign = decision["tool"] == "write_sign"
            position = [args[key] for key in ("x", "y", "z")] if sign else None
            doc_id = "sign:" + ",".join(map(str, position)) if sign else None
            event = next((entry for entry in self._events if entry["type"] == "doc_write"
                          and entry["id"] not in used and entry.get("author") == speaker
                          and entry["gameTime"] == decision["game_time"]
                          and (entry.get("docId") == doc_id if sign else entry.get("title") == args.get("title"))),
                         None)
            if event is None:
                continue
            used.add(event["id"])
            documents.append({"speaker": speaker, "to": None, "tick": event["gameTime"],
                              "text": "\n".join(args.get("lines", []) if sign else args.get("pages", [])),
                              "title": args.get("title", ""), "position": position,
                              "lines": args.get("lines", []) if sign else None, "event_ids": [event["id"]]})
        return documents

    @staticmethod
    def _successful_decision(decision: dict) -> bool:
        """Accept both engine and narrated success spellings; engine events still prove the write/build."""
        outcome = decision.get("outcome", "")
        return outcome == "started" or outcome == "done" or outcome.startswith("done (took ")

    def _prices(self, boundary: int) -> tuple[dict, list]:
        """Apply only edits on the actual stall sign and report malformed prices without guessing."""
        prices, invalid = {}, []
        for name, container in self._containers.items():
            if not container.get("prices"):
                continue
            prices[name] = dict(container["prices"])
            position = container.get("sign_pos", [container["pos"][0], container["pos"][1], container["pos"][2] + 2])
            for document in self._documents:
                if document["position"] != position or document["tick"] >= boundary:
                    continue
                lines = document["lines"]
                price = re.fullmatch(r"\s*(\d+)\s+emeralds?\s*", lines[2], re.I) if len(lines) == 4 else None
                if (price and lines[0].strip().lower().replace(" ", "_") == name
                        and re.fullmatch(r"(?:minecraft:)?[a-z_]+", lines[1].strip())
                        and lines[3].strip().lower() == "per item"):
                    prices[name] = {item_id(lines[1]): int(price[1])}
                else:
                    invalid.append({"stall": name, "tick": document["tick"], "lines": lines,
                                    "event_ids": document["event_ids"]})
        return prices, invalid

    def _section_completions(self) -> dict:
        """Reuse successful build-skill completion events; never inspect or reconstruct world blocks."""
        evidence = {}
        for event in self._events:
            if (event["type"] != "skill_end" or event.get("skill") != "place_section"
                    or event.get("outcome") != "succeeded"):
                continue
            decision = next((entry for entry in reversed(self._decisions)
                             if entry["agent"] == event["bot"] and entry["tool"] == "build_section"
                             and self._successful_decision(entry) and entry["game_time"] < event["gameTime"]), None)
            if decision:
                name = decision["args"]["schematic"]
                if self._first_true.get(name) is None:
                    self._first_true[name] = event["gameTime"]
                    evidence[name] = [event["id"]]
        return evidence

    def _sources(self) -> list[dict]:
        """Collect logged speech and successfully written books/signs, keeping event ids and times."""
        return [{"speaker": event["speaker"], "to": event.get("to"), "text": event["text"],
                 "tick": event["gameTime"], "event_ids": [event["id"]]} for event in self._events
                if event["type"] == "utterance" and event["speaker"] in self._roles] + self._documents

    def _contracts(self) -> list[Promise]:
        """Read the scenario's offered bridge contract, including its machine-readable payment terms."""
        contracts = []
        for document in self._scenario.get("documents", []):
            if "contract" not in document.get("title", "").lower():
                continue
            terms = {}
            for page in document.get("pages", []):
                try:
                    parsed = json.loads(page)
                except ValueError:
                    continue
                if isinstance(parsed, dict) and "payee" in parsed:
                    terms = parsed
            text = "\n".join(document.get("pages", []))
            offer = re.search(r"(?:offers|owes|owed)\s+(\w+)\s+(\d+)\s+emeralds", text, re.I)
            if not terms and not offer:
                continue
            event = next((entry for entry in self._events if entry["type"] == "doc_write"
                          and entry.get("title") == document["title"]
                          and entry.get("author") == document["author"]), None)
            contracts.append(Promise(document["author"], terms.get("payee", offer[1] if offer else None), "pay",
                                     item_id(terms.get("item", EMERALD)),
                                     terms.get("count", int(offer[2]) if offer else None), None, None,
                                     document["pages"][0], event["gameTime"] if event else 0,
                                     [event["id"]] if event else [], terms.get("contract_id", document.get("id")),
                                     "bridge_complete"))
        return contracts

    def _promises(self) -> dict:
        """Extract daily commitments, check delivery and aggregate by promiser and kind."""
        grouped = defaultdict(list)
        for source in self._sources():
            grouped[source["speaker"], source["tick"] // TICKS_PER_DAY].append(source)
        before = getattr(self._extractor, "dropped_quotes", 0)
        promises = [promise for sources in grouped.values() for promise in self._extractor.extract(sources)]
        promises += self._contracts()
        checked = [self._check(promise) for promise in promises]
        totals = dict.fromkeys(STATUSES, 0)
        by_role, by_kind = {}, {}
        for entry in checked:
            totals[entry["status"]] += 1
            for groups, key in ((by_role, entry["promiser"]), (by_kind, entry["kind"])):
                groups.setdefault(key, dict.fromkeys(STATUSES, 0))[entry["status"]] += 1
        return {"totals": totals, "by_role": by_role, "by_kind": by_kind, "commitments": checked,
                "dropped_quotes": getattr(self._extractor, "dropped_quotes", 0) - before,
                "contracts": [{"contract_id": entry["contract_id"], "honoured": entry["status"] == "kept",
                               "tick": entry["fulfilled_tick"], "status": entry["status"],
                               "event_ids": entry["event_ids"]} for entry in checked if entry["contract_id"]]}

    def _check(self, promise: Promise) -> dict:
        """Match only post-promise delivery to the right recipient, item and deadline."""
        deadline = promise.deadline_tick if promise.deadline_tick is not None else self._end
        cutoff, earliest = min(deadline, self._end), promise.tick
        fulfilled, evidence, verifiable = None, [], True
        if promise.condition:
            condition_tick = self._first_true.get(promise.condition)
            earliest = max(earliest, condition_tick) if condition_tick is not None else self._end
        if promise.kind in {"pay", "give", "deliver"}:
            verifiable = bool(promise.promisee and promise.item and promise.count and promise.count > 0)
            senders = {promise.promiser}
            senders.update(self._container_id(container) for container in self._containers.values()
                           if container.get("allowed") == [promise.promiser])
            recipient = self._recipients.get(promise.promisee, promise.promisee)
            total = 0
            for event in self._events:
                if (event["type"] == "transfer" and event["from"] in senders and event["to"] == recipient
                        and event["item"] == promise.item and earliest < event["gameTime"] <= cutoff):
                    total += event["count"]
                    evidence.append(event["id"])
                    if verifiable and total >= promise.count:
                        fulfilled = event["gameTime"]
                        break
        elif promise.kind in {"build", "repair", "open_fair"}:
            structure = "fair_open" if promise.kind == "open_fair" else promise.structure
            structure = "bridge_complete" if structure == "bridge" else structure
            verifiable = structure in {*self._scenario.get("schematics", {}), *self._first_true,
                                        "bridge_complete", "fair_open"}
            if structure in {"bridge_complete", "fair_open"} and "first_true" not in self._report:
                verifiable = False
            tick = self._first_true.get(structure)
            if tick is not None and earliest < tick <= cutoff:
                fulfilled = tick
                evidence = self._completion_events.get(structure, [])
        elif promise.kind == "report":
            event = next((entry for entry in self._events if entry["type"] == "doc_write"
                          and entry.get("author") == promise.promiser
                          and entry.get("title", "").strip().lower() == "report"
                          and earliest < entry["gameTime"] <= cutoff), None)
            if event:
                fulfilled, evidence = event["gameTime"], [event["id"]]
        else:
            verifiable = False
        pending_condition = promise.condition and self._first_true.get(promise.condition) is None
        status = ("unverifiable" if not verifiable else "kept" if fulfilled is not None
                  else "open" if pending_condition and promise.deadline_tick is None
                  else "open" if deadline > self._end else "broken")
        return {**asdict(promise), "deadline_tick": deadline, "status": status, "fulfilled_tick": fulfilled,
                "source_event_ids": promise.event_ids, "event_ids": [*promise.event_ids, *evidence]}

    def _investigation(self) -> dict:
        """Include saved judge scores, or score logged reports with the existing keyword judge."""
        if "investigation_accuracy" in self._report:
            return self._report["investigation_accuracy"]
        truth = self._scenario.get("ground_truth", {})
        expected = set(truth.get("expected_evidence_chain",
                                self._scenario.get("evidence", {}).get("expected_chain", [])))
        judge, scores = KeywordJudge(self._roles), []
        for document in self._documents:
            if not document["title"].lower().startswith("report"):
                continue
            verdict = judge.judge(document["text"])
            scores.append({"author": document["speaker"], "tick": document["tick"],
                           "event_ids": document["event_ids"],
                           "verdict": {**asdict(verdict), "evidence": list(verdict.evidence)},
                           "cause_correct": verdict.cause == truth["cause"] if "cause" in truth else None,
                           "culprit_correct": verdict.culprit == truth["culprit"] if "culprit" in truth else None,
                           "evidence_chain_recall": len(expected.intersection(verdict.evidence)) / len(expected)
                           if expected else None,
                           "evidence_chain_correct": expected.issubset(verdict.evidence) if expected else None})
        return {"judge": "keyword", "reports": scores}


def main(argv: list[str] | None = None) -> int:
    """Parse offline scoring options, write metrics.json and print the same JSON to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--extractor", choices=("keyword", "llm"), default="keyword")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="anthropic")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args(argv)
    if not (args.run_dir / "scenario.json").is_file():
        parser.error("run_dir must contain scenario.json")
    extractor = None
    if args.extractor == "llm":
        if not args.model:
            parser.error("--extractor llm requires --model")
        if __package__:
            from .run_village import model_client
        else:
            from run_village import model_client
        extractor = LLMPromiseExtractor(model_client(args))
    print(json.dumps(VillageMetrics(args.run_dir, extractor).compute(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
