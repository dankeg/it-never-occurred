"""Offline, public-evidence views of a run's speech, actions, exposure and sampled meetings.

Run ``python3.11 scripts/run_views.py RUN --kind both --agents builder,envoy --until 12000``.
Only ``--out`` writes a file; reading a run never changes it. Speech uses the event's
gameTime (when the utterance was logged), not a reconstructed hearing estimate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from math import dist, isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.render import TICKS_PER_DAY, clock  # noqa: E402


class RunViews:
    """Read saved run artifacts without starting the engine or loading a model.

    Action joins use the agent, translated skill name and successful-start window:
    decision logs do not normally record skill IDs. Failed attempts and carried-on
    polls never take ownership of a running skill. Unmatched engine evidence stays
    visible as a row without a decision, rather than being assigned speculatively.
    """

    _SPEECH = {"speak", "ask"}
    _SKILLS = {"build_section": "place_section", "keep_going": "wait"}
    _EFFECT_TOOLS = {
        "block_change": {"mine", "place", "build_section"},
        "transfer": {"give", "chest_take", "chest_put", "collect"},
        "purchase": {"buy"},
        "doc_write": {"write_book", "write_sign"},
        "ferry_trip": {"ferry"},
        "drop": {"drop"},
        "block_use": {"use"},
        "eat": {"eat"},
        "pickup": {"collect"},
    }
    _PRIVATE_FIELDS = {"note", "notes", "prompt", "user_prompt", "system_prompt", "brief",
                       "goal", "goals", "private_goal", "private_goals"}

    def __init__(self, run_dir: Path | str):
        """Load the scenario and optional logs, preserving file order at equal ticks."""
        self.directory = Path(run_dir)
        if not (self.directory / "scenario.json").is_file():
            raise ValueError(f"run directory must contain scenario.json: {self.directory}")
        self.scenario = self.read_json("scenario.json")
        self.decisions = sorted(self._jsonl("orchestrator/decisions.jsonl"), key=lambda row: row["game_time"])
        self.events = sorted(self._jsonl("world/simharness/events.jsonl"), key=lambda row: row["gameTime"])
        self.trajectory = sorted(self._jsonl("orchestrator/trajectory.jsonl"), key=lambda row: row["t"])
        self.observations = sorted(self._jsonl("world/simharness/observations.jsonl"), key=lambda row: row["gameTime"])
        self.perceptions = sorted(self._jsonl("orchestrator/perceptions.jsonl"), key=lambda row: row["t"])
        self._roles = {role["name"] for role in self.scenario.get("roles", [])}

    def read_json(self, name: str) -> dict:
        """Read a JSON artifact, returning an empty object for an absent optional file."""
        path = self.directory / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _jsonl(self, name: str) -> list[dict]:
        """Read nonempty JSONL lines; malformed evidence raises rather than disappearing."""
        path = self.directory / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def speech(self) -> list[dict]:
        """Return every utterance with the recorded addressee and actual listener keys.

        NPC display names and canonical speaker keys are both retained. An intended
        addressee is not necessarily a listener, and listener positions are not proof
        of hearing. Chatter lists the recorded whisperSeen and attentionLost keys,
        excluding listeners who heard words; missing chatter evidence is not proof
        of privacy. An ask's exchange records its NPC addressee's canonical key.
        The clock matches the orchestrator: day 0 starts at 06:00.
        """
        rows = []
        for event in self.events:
            if event["type"] != "utterance":
                continue
            speaker = event.get("speakerKey", event["speaker"])
            listeners = list(event.get("listeners", []))
            chatter = set(event.get("whisperSeen", {})) | set(event.get("attentionLost", {}))
            addressee = event.get("to")
            exchange = event.get("exchange", {})
            if addressee is not None and exchange.get("requester") == speaker:
                addressee = exchange.get("npc", addressee)
            rows.append({"tick": event["gameTime"], "day": event["gameTime"] // TICKS_PER_DAY,
                         "clock": clock(event["gameTime"]), "speaker": event["speaker"], "speaker_key": speaker,
                         "volume": event.get("volume"), "to": event.get("to"), "to_key": addressee,
                         "text": event.get("text", ""), "listeners": listeners,
                         "chatter": sorted(chatter - set(listeners) - {speaker}), "event_id": event.get("id")})
        return rows

    def exposure(self) -> list[dict]:
        """Return role utterances' recorded audiences, without words or position inference.

        Listener and chatter lists retain canonical role names and npc: keys. The
        addressee_heard value is null for unaddressed speech, otherwise a boolean
        based only on listeners. NPC speech never becomes a role exchange merely
        because the NPC's display name happens to match a role.
        """
        rows = []
        for utterance in self.speech():
            if utterance["speaker_key"] not in self._roles:
                continue
            row = {key: value for key, value in utterance.items() if key != "text"}
            row["addressee_heard"] = (None if row["to_key"] is None else row["to_key"] in row["listeners"])
            rows.append(row)
        return rows

    def exposure_pairs(self, rows: list[dict] | None = None) -> list[dict]:
        """Count addressed utterances per unordered role pair, optionally within supplied rows.

        Each utterance is one exchange, even when the addressee did not hear it.
        Heard/chatter counts are exchanges with any recorded third-party role or
        villager, not audience sizes; perceived counts their union once. Tick is
        the pair's last exchange. Unaddressed and self-addressed speech is omitted.
        """
        pairs = {}
        for row in self.exposure() if rows is None else rows:
            speaker, addressee = row["speaker_key"], row["to_key"]
            if speaker not in self._roles or addressee not in self._roles or speaker == addressee:
                continue
            pair = tuple(sorted((speaker, addressee)))
            if pair not in pairs:
                pairs[pair] = {"agents": list(pair), "tick": row["tick"], "exchanges": 0,
                               "heard_by_third_party": 0, "chatter_by_third_party": 0, "perceived_by_third_party": 0}
            summary = pairs[pair]
            heard, chatter = [any(listener not in pair and (listener in self._roles or listener.startswith("npc:"))
                                  for listener in row[field]) for field in ("listeners", "chatter")]
            summary["tick"] = max(summary["tick"], row["tick"])
            summary["exchanges"] += 1
            summary["heard_by_third_party"] += int(heard)
            summary["chatter_by_third_party"] += int(chatter)
            summary["perceived_by_third_party"] += int(heard or chatter)
        return [pairs[pair] for pair in sorted(pairs)]

    @classmethod
    def _public_evidence(cls, value: object) -> object:
        """Copy public JSON without private fields, including nested future event payloads."""
        if isinstance(value, dict):
            return {key: cls._public_evidence(item) for key, item in value.items() if key not in cls._PRIVATE_FIELDS}
        if isinstance(value, list):
            return [cls._public_evidence(item) for item in value]
        return value

    @staticmethod
    def _effect_agents(event: dict) -> set[str]:
        """Return explicitly attributed actors or transfer participants, never bystanders."""
        fields = {"block_change": ("actor",), "transfer": ("from", "to"),
                  "purchase": ("buyer",), "doc_write": ("author",), "ferry_trip": ("rider",),
                  "drop": ("bot",), "block_use": ("actor",), "eat": ("bot",), "pickup": ("bot", "actor"),
                  "death": ("bot",)}
        keys = fields.get(event["type"], ("actor", "bot", "agent", "buyer", "author", "rider"))
        return {event[key] for key in keys if isinstance(event.get(key), str) and event[key]}

    def actions(self) -> list[dict]:
        """Join non-speech decisions to skill endings and attributed engine effects.

        Each row has tick, agent, tool, args, immediate outcome, skill_end and effects.
        Engine-only rows have a null tool/outcome. Transfers can appear for both role
        participants; participation alone does not establish who initiated a transfer.
        Deaths stay standalone evidence attributed to the bot, never its last action.
        Private prompts and notes are never copied. Hunger is not an action; other
        unrecognized event types remain standalone evidence without a causal guess.
        """
        decisions = defaultdict(list)
        for decision in self.decisions:
            decisions[decision["agent"]].append(decision)
        endings = defaultdict(list)
        for event_index, event in enumerate(self.events):
            if event["type"] == "skill_end":
                endings[event["bot"]].append((event_index, event))
        rows, windows, used_endings = [], {}, set()
        by_agent = defaultdict(list)
        for agent, entries in decisions.items():
            next_start = None
            limits = {}
            for entry_index in reversed(range(len(entries))):
                limits[entry_index] = next_start
                if entries[entry_index].get("outcome") == "started":
                    next_start = entries[entry_index]["game_time"]
            for entry_index, decision in enumerate(entries):
                tool, tick = decision["tool"], decision["game_time"]
                if tool in self._SPEECH:
                    continue
                row = {"tick": tick, "agent": agent, "tool": tool,
                       "args": self._public_evidence(decision.get("args", {})),
                       "outcome": decision.get("outcome"), "skill_end": None, "effects": []}
                limit = limits[entry_index]
                if decision.get("outcome") == "started":
                    skill = self._SKILLS.get(tool, tool)
                    for event_index, event in endings[agent]:
                        if (event_index not in used_endings and event["skill"] == skill
                                and event["gameTime"] >= tick and (limit is None or event["gameTime"] <= limit)):
                            row["skill_end"] = self._public_evidence(event)
                            used_endings.add(event_index)
                            break
                next_tick = entries[entry_index + 1]["game_time"] if entry_index + 1 < len(entries) else None
                windows[id(row)] = limit if row["outcome"] == "started" else next_tick
                rows.append(row)
                by_agent[agent].append(row)
        agents = (self._roles | set(decisions)) - {"world"}
        for event_index, event in enumerate(self.events):
            if (event["type"] == "skill_end" and event_index not in used_endings
                    and event["bot"] in agents and event["skill"] not in self._SPEECH):
                rows.append({"tick": event["gameTime"], "agent": event["bot"], "tool": None, "args": {},
                             "outcome": None, "skill_end": self._public_evidence(event), "effects": []})
            if event["type"] not in self._EFFECT_TOOLS:
                # Other events are evidence only when they name a role agent; engine bookkeeping (origin records,
                # villager logs, the staged collapse) is not an action and must never reach an auditor.
                if event["type"] not in {"utterance", "skill_end", "hunger"}:
                    for agent in sorted(self._effect_agents(event) & agents):
                        rows.append({"tick": event["gameTime"], "agent": agent, "tool": None, "args": {},
                                     "outcome": None, "skill_end": None, "effects": [self._public_evidence(event)]})
                continue
            for agent in sorted(self._effect_agents(event) & agents):
                match = None
                for row in reversed(by_agent[agent]):
                    if (row["tool"] not in self._EFFECT_TOOLS[event["type"]] or row["tick"] > event["gameTime"]
                            or row["outcome"] not in {"started", "done"}):
                        continue
                    end = row["skill_end"]["gameTime"] if row["skill_end"] else windows[id(row)]
                    if end is None or event["gameTime"] <= end:
                        match = row
                        break
                if match is None:
                    rows.append({"tick": event["gameTime"], "agent": agent, "tool": None, "args": {},
                                 "outcome": None, "skill_end": None, "effects": [self._public_evidence(event)]})
                else:
                    match["effects"].append(self._public_evidence(event))
        return sorted(rows, key=lambda row: (row["tick"], row["agent"]))

    def meetings(self, radius: float = 8) -> list[dict]:
        """Return pairwise co-location intervals at synchronous trajectory polls.

        Endpoints are inclusive first/last close samples, not interpolated crossing
        times. A missing agent closes the interval at its last close sample; stale
        positions are never carried forward. Three close roles produce three pairs.
        """
        if not isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        snapshots = defaultdict(dict)
        for entry in self.trajectory:
            if entry["agent"] in self._roles and entry.get("pos") is not None:
                snapshots[entry["t"]][entry["agent"]] = entry["pos"]
            else:
                snapshots[entry["t"]]
        active, intervals = {}, []
        for tick, positions in sorted(snapshots.items()):
            close = {pair for pair in combinations(sorted(positions), 2)
                     if dist(positions[pair[0]], positions[pair[1]]) <= radius}
            for pair in sorted(active.keys() - close):
                intervals.append(active.pop(pair))
            for pair in sorted(close):
                if pair not in active:
                    active[pair] = {"start_tick": tick, "end_tick": tick, "agents": list(pair)}
                active[pair]["end_tick"] = tick
        intervals.extend(active.values())
        return sorted(intervals, key=lambda row: (row["start_tick"], row["agents"], row["end_tick"]))

    def vantage(self, observer: str) -> list[tuple[int, str, dict]]:
        """What one person perceived, from the perception log: (tick, label, record) rows.

        A person in sight is listed when first seen and again whenever what they hold, do or carry changes, or
        when they come back into sight; speech, changes, seen actions and reading are listed as perceived."""
        rows, last = [], {}
        for record in self.perceptions:
            if record["agent"] != observer:
                continue
            tick, present = record["t"], {}
            for entity in record.get("entities", []):
                name = entity.get("name") or entity.get("type")
                state = {key: entity[key] for key in ("heldItem", "activity", "load") if key in entity}
                present[name] = state
                if last.get(name) != state:
                    rows.append((tick, "SAW", {"who": name, "pos": entity.get("pos"), **state}))
            last = present
            for key, label in (("heard", "HEARD"), ("chatter", "CHATTER"), ("shout_seen", "SHOUT_SEEN"),
                               ("changes", "CHANGE"), ("seen_actions", "ACTION_SEEN")):
                rows.extend((tick, label, entry) for entry in record.get(key, []))
            if record.get("read"):
                rows.append((tick, "READ", {"text": record["read"]}))
        return rows

    @staticmethod
    def _where(pos) -> str:
        """A position as whole blocks."""
        return "(" + ", ".join(str(int(value // 1)) for value in pos) + ")" if pos else "?"

    def compact(self, label: str, value: dict) -> str:
        """One record as a short readable line for monitors: the same evidence without ids, hashes or provenance."""
        name = lambda item: str(item).removeprefix("minecraft:").replace("_", " ")  # noqa: E731
        if label == "SPEECH":
            heard = ", ".join(value["listeners"]) or "no one"
            seen = f"; seen talking by {', '.join(value['chatter'])}" if value.get("chatter") else ""
            to = f" to {value['to']}" if value.get("to") else ""
            return f"{value['speaker']}{to} ({value.get('volume') or 'talk'}; heard by {heard}{seen}): {value['text']}"
        if label == "ACTION":
            args = value.get("args") or {}
            where = self._where([args[key] for key in ("x", "y", "z")]) if all(key in args for key in "xyz") else ""
            rest = " ".join(f"{key}={name(item)}" for key, item in args.items() if key not in ("x", "y", "z"))
            line = f"{value['agent']} {value['tool']} {where} {rest}".replace("  ", " ").strip()
            return f"{line} -> {value['outcome']}"
        if label in ("EFFECT", "SKILL_END"):
            event = value["event"]
            kind = event["type"]
            if kind == "transfer":
                parties = f"{name(event.get('from'))} -> {name(event.get('to'))}"
                return f"{parties}: {event.get('count')} {name(event.get('item'))}"
            if kind == "purchase":
                cost = f"for {event.get('cost')} emeralds"
                return f"{event['buyer']} bought {event['count']} {name(event['item'])} {cost}"
            if kind == "block_change":
                change = f"{name(event['from'])} -> {name(event['to'])}"
                return f"{event.get('actor')} changed {self._where(event['pos'])}: {change}"
            if kind == "bake":
                where = self._where(event["pos"])
                return f"{event['bot']} baked {event['count']} bread at {where} ({event['outcome']})"
            if kind == "ferry_trip":
                return f"{event['rider']} crossed by ferry {event['from']} -> {event['to']} ({event['outcome']})"
            if kind in ("drop", "pickup"):
                return f"{event['bot']} {kind} {event['count']} {name(event['item'])} at {self._where(event['pos'])}"
            if kind == "skill_end":
                reason = f" ({event['reason']})" if event.get("reason") else ""
                return f"{event['bot']} {event['skill']} {event['outcome']}{reason}"
            return f"{kind} " + json.dumps({key: item for key, item in event.items()
                                            if key not in ("id", "gameTime", "prov", "contentSha256", "itemEntity",
                                                           "thrower", "uuid")}, ensure_ascii=False, sort_keys=True)
        if label == "SAW":
            extra = "".join(f", {text}" for text in (
                f"holding {name(value['heldItem'])}" if value.get("heldItem") else "",
                f"{(value.get('activity') or {}).get('skill', '')}ing" if value.get("activity") else "",
                f"carrying a sack of {name(value['load']['item'])}" if value.get("load") else "") if text)
            return f"saw {value['who']} at {self._where(value.get('pos'))}{extra}"
        if label in ("HEARD", "CHATTER", "SHOUT_SEEN"):
            text = f": {value['text']}" if value.get("text") else ""
            return f"{label.lower().replace('_', ' ')} {value.get('speaker', '?')}{text}"
        if label == "CHANGE":
            return f"saw {self._where(value.get('pos'))} change: {name(value.get('from'))} -> {name(value.get('to'))}"
        if label == "READ":
            return f"read {value['text']}"
        return f"{label} " + json.dumps(value, ensure_ascii=False, sort_keys=True)

    def render(self, kind: str, agents: list[str] | None = None, until: int | None = None, *,
               compact: bool = False, since: int | None = None) -> str:
        """Render whitelisted public evidence in deterministic, global tick order.

        Agent filters select speakers/actors, not everyone who heard an utterance.
        Joined endings/effects are rendered at their own ticks, so ``until`` cannot
        reveal a later outcome on an earlier decision. Equal-tick speech precedes
        decisions, effects and skill endings, with stable order within each kind.
        """
        if kind not in {"speech", "actions", "both", "exposure"} and not kind.startswith("vantage:"):
            raise ValueError("kind must be speech, actions, both, exposure or vantage:<observer>")
        selected = set(agents) if agents is not None else None
        records = []
        if kind.startswith("vantage:"):
            records.extend((tick, 0, label, value) for tick, label, value in self.vantage(kind.split(":", 1)[1]))
        if kind == "exposure":
            rows = [row for row in self.exposure()
                    if (selected is None or selected.intersection((row["speaker"], row["speaker_key"])))
                    and (until is None or row["tick"] <= until)]
            records.extend((row["tick"], 0, "EXPOSURE", row) for row in rows)
            records.extend((row["tick"], 1, "PAIR", row) for row in self.exposure_pairs(rows))
        if kind in {"speech", "both"}:
            for row in self.speech():
                if selected is None or selected.intersection((row["speaker"], row["speaker_key"])):
                    records.append((row["tick"], 0, "SPEECH", row))
        if kind in {"actions", "both"}:
            for row in self.actions():
                if selected is not None and row["agent"] not in selected:
                    continue
                if row["tool"] is not None:
                    public = {key: row[key] for key in ("agent", "tool", "args", "outcome")}
                    records.append((row["tick"], 1, "ACTION", public))
                for event in row["effects"]:
                    records.append((event["gameTime"], 2, "EFFECT",
                                    {"agent": row["agent"], "action_tick": row["tick"], "event": event}))
                if row["skill_end"]:
                    event = row["skill_end"]
                    records.append((event["gameTime"], 3, "SKILL_END",
                                    {"agent": row["agent"], "action_tick": row["tick"], "event": event}))
        records = [record for record in sorted(records, key=lambda row: row[:2])
                   if (until is None or record[0] <= until) and (since is None or record[0] >= since)]
        if compact:
            return "".join(f"[{clock(tick)}] {self.compact(label, value)}\n" for tick, _, label, value in records)
        return "".join(f"[tick {tick}; {clock(tick)}] {label} "
                       + json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                       for tick, priority, label, value in records)


def main(argv: list[str] | None = None) -> int:
    """Print an offline auditor view or write it only to the requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--kind", default="both", help="speech, actions, both, exposure or vantage:<observer>")
    parser.add_argument("--agents", help="comma-separated speaker/actor names")
    parser.add_argument("--until", type=int, help="inclusive final tick")
    parser.add_argument("--since", type=int, help="inclusive first tick")
    parser.add_argument("--compact", action="store_true", help="short readable lines (for monitors)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        views = RunViews(args.run_dir)
        text = views.render(args.kind, [name.strip() for name in args.agents.split(",") if name.strip()]
                            if args.agents is not None else None, args.until, compact=args.compact, since=args.since)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
