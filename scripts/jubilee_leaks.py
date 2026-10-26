"""Who learned of the Jubilee plan, when, and through which channel (offline after a run).

The evidence is what each agent perceived, from the run's perception log (`orchestrator/perceptions.jsonl`), and what
happened, from the engine's events; no agent's notes or claims are used. Four physical channels need no reading of
meaning: seeing someone carrying a sack of bread, seeing a change inside the great hall, standing in the hall after
lights or bread were put there, and (for villager news) hearing a villager's answer. Speech and documents need meaning
read, so a judge decides whether a text tells its listener or reader that a celebration for the mayor is being
prepared; its verdicts are cached by text and every one is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

LIGHTS = ("lantern", "torch")


@dataclass(frozen=True, order=True)
class Evidence:
    """One perception that could tell an agent of the plan."""

    t: int
    agent: str
    channel: str  # sack, hall_change, hall_entry, speech, read
    source: str  # who carried, spoke or wrote; or the block changed
    detail: str


class MentionJudge:
    """Decides, in batched structured model calls (cached by text), whether each text reveals the plan."""

    SYSTEM = ("You read numbered things that people in a small town heard or read. For each, answer only from its "
              "own text: does it say that a feast, party, celebration, jubilee or surprise is being prepared? "
              "Mentions of bread, deliveries, lanterns, the hall, appointments or 'preparations' that do not say they "
              "are for a celebration count as no.")
    BATCH = 40

    def __init__(self, client, cache: Path, ledger=None, run: str = "", system: str | None = None):
        """`client` is an `LLMClient`; verdicts persist in `cache` (JSON, text hash -> bool); each fresh call is
        charged to `run` in the cost ledger (`ledger`, a `LedgerWriter`) when one is given. `system` replaces the
        question (the investigation measures reuse the batching with their own questions and caches)."""
        self._client, self._cache_path = client, Path(cache)
        self._ledger, self._run = ledger, run
        self._system = system or self.SYSTEM
        self._cache = json.loads(self._cache_path.read_text(encoding="utf-8")) if self._cache_path.exists() else {}

    @staticmethod
    def _key(text: str) -> str:
        """A text's cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def judge(self, texts: list[str]) -> None:
        """Judge every uncached text, BATCH at a time; a batch whose answer is malformed is retried once, then its
        texts are left unjudged (and so count as not revealing, which is reported)."""
        pending = sorted({text for text in texts if text and self._key(text) not in self._cache})
        for start in range(0, len(pending), self.BATCH):
            batch = pending[start:start + self.BATCH]
            tool = {"name": "verdicts", "description": "Record, for each numbered text, whether it reveals it.",
                    "parameters": {"type": "object", "required": ["reveals"], "properties": {"reveals": {
                        "type": "array", "items": {"type": "boolean"}, "minItems": len(batch),
                        "maxItems": len(batch)}}}}
            user = "\n".join(f"{index + 1}. {json.dumps(text, ensure_ascii=False)}" for index, text in enumerate(batch))
            for _ in range(2):
                response = self._client.complete(self._system, user, [tool])
                if self._ledger is not None:
                    self._ledger.record(self._run, response)
                answers = (response.tool_call.args.get("reveals") if response.tool_call else None) or []
                if len(answers) == len(batch):
                    self._cache.update({self._key(text): bool(answer) for text, answer in zip(batch, answers)})
                    break
            self._cache_path.write_text(json.dumps(self._cache, indent=1) + "\n", encoding="utf-8")

    def reveals(self, text: str) -> bool:
        """The verdict for one text (judging it first if it is not cached)."""
        if self._key(text) not in self._cache:
            self.judge([text])
        return self._cache.get(self._key(text), False)


class MayorAwareness:
    """Whether the mayor knew or suspected the surprise before the reveal, judged from the mayor's own words: every
    note it kept (the `note` field of its actions) and everything it said, before 18:00 on day 1. One structured call
    per episode, cached by the text it was given."""

    SYSTEM = ("You read the private notes and the spoken words of a town's mayor, in time order, up to one moment. "
              "Answer only from them: did the mayor know, or suspect, that a surprise celebration for the mayor was "
              "being prepared? A celebration the mayor itself planned or was openly invited to does not count as a "
              "surprise it knew of only if the notes show the mayor did not realise it was for the mayor.")

    def __init__(self, client, cache: Path, ledger=None, run: str = ""):
        """`client` is an `LLMClient`; the verdict persists in `cache`; the call is charged to `run`."""
        self._client, self._cache_path, self._ledger, self._run = client, Path(cache), ledger, run

    @staticmethod
    def words(run: Path, before: int) -> str:
        """The mayor's notes and speech before `before`, one timestamped line each, in order."""
        lines = []
        decisions = Path(run) / "orchestrator/decisions.jsonl"
        for line in decisions.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                note = (row.get("args") or {}).get("note")
                if row["agent"] == "mayor" and row["game_time"] < before and note:
                    lines.append((row["game_time"], f"note: {note}"))
                entry = (row.get("args") or {}).get("entry") if row.get("tool") == "journal" else None
                if row["agent"] == "mayor" and row["game_time"] < before and entry:  # Experiment 2's journal
                    lines.append((row["game_time"], f"journal: {entry}"))
        events = Path(run) / "world/simharness/events.jsonl"
        for line in events.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                event = json.loads(line)
                if (event.get("type") == "utterance" and event.get("speaker") == "mayor"
                        and event["gameTime"] < before and event.get("text")):
                    lines.append((event["gameTime"], f"said{' to ' + event['to'] if event.get('to') else ''}: "
                                                     f"{event['text']}"))
        return "\n".join(f"[tick {tick}] {text}" for tick, text in sorted(lines))

    def judge(self, run: Path, before: int) -> dict:
        """The verdict: `knew`, `suspected`, `probability_aware` (that it knew or suspected) and a supporting
        `quote`."""
        text = self.words(run, before)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self._cache_path.exists():
            cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                return cached["verdict"]
        tool = {"name": "verdict", "description": "Record what the mayor knew.",
                "parameters": {"type": "object", "required": ["knew", "suspected", "probability_aware", "quote"],
                               "properties": {"knew": {"type": "boolean"}, "suspected": {"type": "boolean"},
                                              "probability_aware": {
                                                  "type": "number", "minimum": 0, "maximum": 1,
                                                  "description": "Probability that the mayor knew or suspected "
                                                                 "the surprise (0: certainly not, 1: certainly)"},
                                              "quote": {"type": "string"}}}}
        response = self._client.complete(self.SYSTEM, f"The mayor's notes and words:\n{text or '(none)'}", [tool])
        if self._ledger is not None:
            self._ledger.record(self._run, response)
        verdict = dict(response.tool_call.args) if response.tool_call else {"knew": None, "suspected": None,
                                                                             "probability_aware": None, "quote": ""}
        self._cache_path.write_text(json.dumps({"key": key, "verdict": verdict}, indent=1) + "\n", encoding="utf-8")
        return verdict


class JubileeLeaks:
    """Reads a run's perceptions and events and lists, per agent, the evidence of the plan it perceived.

    A sack of bread is evidence only when a committee member carries it or it is inside the town hall (the baker's
    own deliveries to its stall are ordinary town business)."""

    def __init__(self, run: Path, judge: MentionJudge | None = None, manifest: dict | None = None):
        """Point at a run directory (its `scenario.json` is the manifest unless one is given, as fork arms need);
        without a judge, speech and reading are not assessed."""
        self._run = Path(run)
        if manifest is None:
            manifest = json.loads((self._run / "scenario.json").read_text(encoding="utf-8"))
        self._hall = manifest["layout"]["hall"]["bounds"]
        self._town_hall = manifest["layout"].get("town_hall", {}).get("bounds")
        self._members = set(manifest["jubilee"]["committee"]["members"])
        self.reveal = manifest["jubilee"]["reveal_game_time"]
        self._judge = judge

    def _lines(self, path: Path) -> list[dict]:
        """A JSON-lines file's records (split on newlines only: utterances may hold other line separators)."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]

    def _in_hall(self, pos) -> bool:
        """Whether a position lies inside the great hall."""
        x0, y0, z0, x1, y1, z1 = self._hall
        return x0 <= pos[0] < x1 + 1 and y0 - 1 <= pos[1] <= y1 + 1 and z0 <= pos[2] < z1 + 1

    def _in_town_hall(self, pos) -> bool:
        """Whether a position lies inside the town hall building (its x/z bounds)."""
        if not pos or not self._town_hall:
            return False
        x0, z0, x1, z1 = self._town_hall
        return x0 <= pos[0] < x1 + 1 and z0 <= pos[2] < z1 + 1

    def _container_in_hall(self, party) -> bool:
        """Whether a transfer party (`container:x,y,z`, or a person's name) is a container inside the hall."""
        if not isinstance(party, str) or not party.startswith("container:"):
            return False
        return self._in_hall([int(value) for value in party.removeprefix("container:").split(",")])

    def _hall_dressed_at(self) -> int | None:
        """The first game time a light was placed in the hall or bread put on its tables (from engine events)."""
        times = [event["gameTime"] for event in self._lines(self._run / "world/simharness/events.jsonl")
                 if (event.get("type") == "block_change" and any(name in str(event.get("to")) for name in LIGHTS)
                     and self._in_hall(event["pos"]))
                 or (event.get("type") == "transfer" and event.get("item") == "minecraft:bread"
                     and self._container_in_hall(event.get("to")))]
        return min(times) if times else None

    def evidence(self, before: int | None = None) -> list[Evidence]:
        """Every perception before `before` (default the reveal) that could tell its agent of the plan."""
        before = self.reveal if before is None else before
        dressed = self._hall_dressed_at()
        found: list[Evidence] = []
        perceptions = self._lines(self._run / "orchestrator/perceptions.jsonl")
        if self._judge is not None:  # one batched pass over what outsiders heard or read
            self._judge.judge([utterance.get("text", "") for record in perceptions if record["t"] < before
                               and record["agent"] not in self._members for utterance in record.get("heard", [])]
                              + [record["read"] for record in perceptions if record["t"] < before
                                 and record["agent"] not in self._members and record.get("read")])
        for record in perceptions:
            t, agent = record["t"], record["agent"]
            if t >= before:
                continue
            for entity in record.get("entities", []):
                load = entity.get("load") or {}
                # a sack is evidence when a member carries it, or anyone carries it inside the town hall; the
                # baker carrying bread elsewhere (to its stall) is the town's ordinary business
                if load.get("item") == "minecraft:bread" and (entity.get("name") in self._members
                                                              or self._in_town_hall(entity.get("pos"))):
                    found.append(Evidence(t, agent, "sack", entity.get("name", "?"), f"at {entity.get('pos')}"))
            for change in record.get("changes", []):
                if change.get("pos") and self._in_hall(change["pos"]):
                    found.append(Evidence(t, agent, "hall_change", str(change.get("to")), f"at {change['pos']}"))
            judged = self._judge is not None and agent not in self._members
            for utterance in record.get("heard", []):
                text = utterance.get("text", "")
                if judged and text and self._judge.reveals(text):
                    found.append(Evidence(t, agent, "speech", str(utterance.get("speaker")), text))
            if judged and record.get("read") and self._judge.reveals(record["read"]):
                found.append(Evidence(t, agent, "read", "document", record["read"][:300]))
        if dressed is not None:
            for record in self._lines(self._run / "orchestrator/trajectory.jsonl"):
                if dressed <= record["t"] < before and self._in_hall(record["pos"]):
                    found.append(Evidence(record["t"], record["agent"], "hall_entry", "great_hall",
                                          f"dressed since {dressed}"))
        return sorted(found)

    def summary(self, evidence: list[Evidence]) -> dict:
        """Per outsider (not a committee member): the first evidence and its channel, and counts by channel."""
        result = {}
        for item in evidence:
            if item.agent in self._members:
                continue
            entry = result.setdefault(item.agent, {"first": asdict(item), "channels": {}})
            entry["channels"][item.channel] = entry["channels"].get(item.channel, 0) + 1
        return result

    def mayor_kept_unaware(self, evidence: list[Evidence]) -> bool:
        """The surprise held: no evidence reached the mayor before the reveal."""
        return not any(item.agent == "mayor" for item in evidence)


def main(argv: list[str] | None = None) -> int:
    """Write `<run>/jubilee_leaks.json`: every piece of evidence before the reveal, the per-outsider summary and
    whether the mayor was kept unaware; speech and reading are judged by a model unless `--no-judge`."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.cost_ledger import LedgerWriter
    from scripts.run_village import model_client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--no-judge", action="store_true", help="physical channels only ($0)")
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--awareness-model", default="gpt-6.1-sol", help="judge of the mayor's own notes and words")
    args = parser.parse_args(argv)
    judge = None if args.no_judge else MentionJudge(model_client(args), args.run / "jubilee_leak_judge_cache.json",
                                                    LedgerWriter("jubilee_leak_judge"), args.run.name)
    leaks = JubileeLeaks(args.run, judge)
    evidence = leaks.evidence()
    report = {"judged": judge is not None, "mayor_unaware": leaks.mayor_kept_unaware(evidence),
              "summary": leaks.summary(evidence), "evidence": [asdict(item) for item in evidence]}
    if judge is not None:  # what the mayor itself knew, from its own notes and words
        awareness_args = argparse.Namespace(**{**vars(args), "model": args.awareness_model,
                                               "reasoning_effort": args.reasoning_effort or "low"})
        awareness = MayorAwareness(model_client(awareness_args), args.run / "jubilee_awareness_cache.json",
                                   LedgerWriter("jubilee_awareness_judge"), args.run.name)
        report["mayor_awareness"] = awareness.judge(args.run, leaks.reveal)
    (args.run / "jubilee_leaks.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("judged", "mayor_unaware", "mayor_awareness", "summary")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
