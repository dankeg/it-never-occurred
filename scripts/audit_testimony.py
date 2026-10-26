"""Testimony audit (gate V-G3): villager speech never contradicts the event log.

Every news item a villager says aloud (a gossip relay or an answer to `ask`) must be grounded:
- **Origin:** its `originEventId` names a logged event: a block change, or an utterance for shout news.
- **Subject:** for block news, the subject is the changed block's position, or a named landmark.
- **Provenance:**
  - A first-hand item (the speaker is the chain's first link) must be in the speaker's own sight-based observation
    log before it speaks.
  - A relayed item must have reached the speaker earlier, in an utterance it fully heard from the previous link in
    the chain, carrying the same news ID.
- **Presence:** block news records the sorted bot names with LOS within eight blocks of the changed block's
  center in the observer's unique sight record containing that origin change, before speech starts. The record's
  gameTime may follow learnedAt when a first-hand sighting upgrades a relay. Relays preserve `present` unchanged.
  Older news without `present` is treated as an empty list.

Run it on a finished run directory. It prints the violations and returns nonzero if there are any.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import sys
from pathlib import Path


@dataclass
class TestimonyAudit:
    """Checks every news-bearing villager utterance in one run against that run's logs."""

    events: list[dict]
    observations: list[dict]
    violations: list[str] = field(default_factory=list)
    checked: int = 0

    @classmethod
    def load(cls, run: Path) -> TestimonyAudit:
        """Read a run's event log and villager observation log."""
        logs = Path(run) / "world" / "simharness"

        def read(name: str) -> list[dict]:
            path = logs / name
            return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
        return cls(read("events.jsonl"), read("npc_observations.jsonl"))

    def run(self) -> TestimonyAudit:
        """Audit every news item in every villager utterance; return self with violations filled in."""
        changes = {f"change:{e['id']}": e for e in self.events if e["type"] == "block_change"}
        utterances = sorted((e for e in self.events if e["type"] == "utterance"), key=lambda e: e["uttId"])
        seen = {}
        for record in self.observations:
            for change in record.get("changes", []):
                seen.setdefault((record["observer"], f"change:{change['id']}"), []).append(record)
        heard = {}
        for utterance in utterances:
            for item in utterance.get("news", []):
                for listener in utterance.get("listeners", []):
                    heard.setdefault((listener, item["newsId"], utterance["speaker"]), []).append((utterance, item))
        for utterance in utterances:
            if not utterance["speakerKey"].startswith("npc:"):
                continue
            for item in utterance.get("news", []):
                self.checked += 1
                self._check(utterance, item, changes, utterances, seen, heard)
        return self

    def _check(self, utterance: dict, item: dict, changes: dict, utterances: list[dict],
               seen: dict, heard: dict) -> None:
        """Check one news item's origin, subject and provenance; record any violation."""
        label = f"utt {utterance['uttId']} by {utterance['speaker']}: {item['newsId']}"
        present = item.get("present", [])
        if (not isinstance(present, list)
                or any(not isinstance(name, str) or not name.strip() for name in present)
                or present != sorted(set(present))):
            self.violations.append(f"{label}: present must be sorted unique agent names")
            return
        origin = item["originEventId"]
        if origin.startswith("change:"):
            change = changes.get(origin)
            if change is None:
                self.violations.append(f"{label}: origin {origin} is not a logged block change")
                return
            if isinstance(item["subject"], list) and item["subject"] != change["pos"]:
                self.violations.append(f"{label}: subject {item['subject']} != changed block {change['pos']}")
        elif origin.startswith("utt:"):
            if not any(f"utt:{u['uttId']}" == origin for u in utterances):
                self.violations.append(f"{label}: origin {origin} is not a logged utterance")
                return
            if present:
                self.violations.append(f"{label}: present requires an observed block change, not a shout")
        else:
            self.violations.append(f"{label}: unknown origin {origin}")
            return
        chain, speaker = item["chain"], utterance["speakerKey"]
        if chain and chain[-1] == utterance["speaker"] and len(chain) == 1:
            if origin.startswith("change:"):
                records = [record for record in seen.get((speaker, origin), [])
                           if record["gameTime"] <= utterance["startedAt"]]
                if not records:
                    self.violations.append(f"{label}: first-hand claim not in the speaker's observation log beforehand")
                elif "present" in item:
                    self._check_present(label, item, changes[origin], records)
        else:
            source = (chain[-1] if chain[-1] != utterance["speaker"] else
                      (chain[-2] if len(chain) > 1 else None)) if chain else None
            previous = [news for event, news in heard.get((speaker, item["newsId"], source), [])
                        if event["gameTime"] <= utterance["startedAt"]]
            if not previous:
                self.violations.append(f"{label}: relayed claim not heard from {source} beforehand")
            elif not any(news.get("present", []) == present for news in previous):
                self.violations.append(f"{label}: relay changed present")

    def _check_present(self, label: str, item: dict, change: dict, records: list[dict]) -> None:
        """Use the unique observer/origin sight record, never entities merged from a different observation tick."""
        if len(records) != 1:
            self.violations.append(f"{label}: present needs a unique observation of the origin change")
            return
        record = records[0]
        if type(item.get("learnedAt")) is not int or item["learnedAt"] > record["gameTime"]:
            self.violations.append(f"{label}: learnedAt follows the first-hand observation gameTime")
            return
        witnessed = next(observed for observed in record["changes"] if observed["id"] == change["id"])
        if (not self._has_ray(witnessed) or witnessed.get("pos") != change["pos"]
                or witnessed.get("gameTime") != change.get("gameTime")
                or change["gameTime"] > record["gameTime"]):
            self.violations.append(f"{label}: changed block lacks matching time and LOS in source observation")
            return
        if item["present"] != self._present(record, change):
            self.violations.append(f"{label}: present does not match same-observation agents with LOS within 8 blocks")

    @staticmethod
    def _has_ray(item: dict) -> bool:
        """Recognize the nonnegative LOS ray index emitted by the existing sight schema, including index zero."""
        return type(item.get("losRay")) is int and item["losRay"] >= 0

    @classmethod
    def _present(cls, record: dict, change: dict) -> list[str]:
        """Reproduce the inclusive feet-to-block-center presence rule using only this observation's bot entities."""
        center = [coordinate + 0.5 for coordinate in change["pos"]]
        return sorted({entity["name"] for entity in record.get("entities", [])
                       if "npcId" not in entity and isinstance(entity.get("name"), str) and entity["name"].strip()
                       and cls._has_ray(entity) and len(entity.get("pos", [])) == 3
                       and math.dist(entity["pos"], center) <= 8})


def main(argv: list[str] | None = None) -> int:
    """Audit the run directories given on the command line."""
    status = 0
    for run in (argv if argv is not None else sys.argv[1:]):
        audit = TestimonyAudit.load(Path(run)).run()
        print(f"{run}: {audit.checked} news items checked, {len(audit.violations)} violations")
        for violation in audit.violations[:20]:
            print("  " + violation)
        status |= bool(audit.violations)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
