"""LLM-driven role agents: role briefs, bounded memory, and prompt construction.

An agent's prompt has two parts. The system prompt is fixed per agent: world rules plus its role brief. The user prompt
is built at every decision from the agent's memory and the current observation. Memory is bounded and fully
serializable, so a forked branch restores each agent exactly as it was at the fork point.

Prompts are append-only within an epoch of `epoch_turns` decisions. The first turn of an epoch carries the full memory;
each later turn is appended to the conversation (the previous turns stay, with the action chosen and its result) and
carries only what is new since the previous turn, plus the current view. Consecutive prompts of an agent then share
an exact prefix that a prompt cache reuses. With `epoch_turns=1` every prompt carries the full memory and no history.

`observe` drains the speech an agent heard, block changes and item movements it saw since its previous observation.
The agent keeps these in memory, so nothing perceived between decisions is lost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .llm import ToolCall, Turn
from .render import ObservationRenderer, block_text, clock, short_id
from .tools import tool_definitions

WORLD_RULES = """You are a person living in a small town, in a world that works like Minecraft. Time passes in ticks \
(20 ticks = 1 second, 24,000 ticks = 1 day). You only know what you have seen, heard, read or remembered; nothing \
else is shown to you. Other people only learn what you do if they see it, hear it, or read about it. The other \
people in town pursue their own goals; you can influence them only by talking (whisper carries 3 blocks, talk 8, \
shout 24), by writing (signs, books, notes left in chests), or by what you do.

Every turn, choose exactly one action with a tool call. Actions like walking, mining and building take time, and you \
can be interrupted when someone speaks to you. If an action fails, you are told why. Use the optional `note` field \
of any action to remember a new fact or decision worth keeping (at most 15 words; leave it out when the action says \
it all, as most actions need no note); your notes are your only long-term memory."""


@dataclass(frozen=True)
class RoleBrief:
    """Who an agent is.

    `public` is what anyone in town would know about this role. `private` holds the agent's own goals and secrets,
    which it sees but nobody else does. `common` is knowledge everyone in town shares (e.g. a map of named places).
    """

    bot: str
    role: str
    public: str
    private: str
    common: str = ""


class AgentMemory:
    """Bounded, serializable memory: notes, own actions, speech, witnessed changes and transfers, and things read.

    Besides the bounded sections, the memory keeps the updates since the agent's previous prompt (`take_fresh`): what it
    heard, saw, read or noticed, and how earlier actions ended. Its own actions and notes are not among them; they are
    already in the conversation.
    """

    LIMITS = {"notes": 20, "actions": 10, "heard": 15, "changes": 10, "reads": 5, "noticed": 15, "seen_actions": 10,
              "journal": 8}
    OPTIONAL = ("seen_actions", "journal")  # kept out of snapshots and prompts while empty (older runs never had them)
    FRESH_LIMIT = 20

    def __init__(self, data: dict | None = None):
        """Start empty, or restore from `to_json()` output."""
        self._data = {key: list((data or {}).get(key, [])) for key in self.LIMITS}
        self._fresh = list((data or {}).get("fresh", []))
        self._fresh_dropped = (data or {}).get("freshDropped", 0)
        self._inventory = (data or {}).get("inventory")

    def to_json(self) -> dict:
        """Return a JSON-serializable copy of the memory."""
        data = {key: [dict(entry) for entry in entries] for key, entries in self._data.items()
                if key not in self.OPTIONAL or entries}
        data.update(fresh=list(self._fresh), freshDropped=self._fresh_dropped,
                    inventory=dict(self._inventory) if self._inventory is not None else None)
        return data

    def _append(self, key: str, entry: dict, fresh: str | None = None) -> None:
        """Append one entry, dropping the oldest beyond the limit, and keep `fresh` as an update for the next prompt."""
        entries = self._data[key]
        entries.append(entry)
        del entries[:-self.LIMITS[key]]
        if fresh is not None:
            self._add_fresh(f"  [{clock(entry['gameTime'])}] {fresh}")

    def _add_fresh(self, line: str) -> None:
        """Keep one update line, dropping the oldest beyond `FRESH_LIMIT`."""
        self._fresh.append(line)
        if len(self._fresh) > self.FRESH_LIMIT:
            self._fresh_dropped += len(self._fresh) - self.FRESH_LIMIT
            del self._fresh[:-self.FRESH_LIMIT]

    def _note_inventory(self, game_time: int | None, inventory: list[dict] | None) -> None:
        """Remember the inventory's item counts and report what changed since the previous observation."""
        if inventory is None or game_time is None:
            return
        counts: dict[str, int] = {}
        for entry in inventory:
            name = short_id(entry["item"])
            counts[name] = counts.get(name, 0) + entry["count"]
        if self._inventory is not None:
            changes = [f"{counts.get(name, 0) - self._inventory.get(name, 0):+d} {name}"
                       for name in sorted(set(counts) | set(self._inventory))
                       if counts.get(name, 0) != self._inventory.get(name, 0)]
            if changes:
                self._add_fresh(f"  [{clock(game_time)}] your inventory: {', '.join(changes)}")
        self._inventory = counts

    def take_fresh(self) -> list[str]:
        """Return the update lines since the previous call, oldest first, and clear them."""
        lines = ([f"  ({self._fresh_dropped} earlier updates not shown)"] if self._fresh_dropped else []) + self._fresh
        self._fresh, self._fresh_dropped = [], 0
        return lines

    @staticmethod
    def _heard_text(entry: dict) -> str:
        """One heard-speech entry, without its time."""
        return (entry["speaker"] + (f" ({entry['distance']} blocks away)" if entry["distance"] is not None else "")
                + (f": \"{entry['text']}\"" if entry["text"] is not None else f" {entry['what']}"))

    @staticmethod
    def _change_text(entry: dict) -> str:
        """One seen block change, without its time."""
        return f"block at {block_text(entry['pos'])}: {short_id(entry['from'])} -> {short_id(entry['to'])}"

    @staticmethod
    def _seen_action_text(entry: dict) -> str:
        """One witnessed item movement, without its time or any private quantities."""
        if entry["action"] == "give":
            return f"saw {entry['actor']} give {short_id(entry['item'])} to {entry['target']}"
        if entry["action"] in {"drop", "pickup"}:
            verb = "drop" if entry["action"] == "drop" else "pick up"
            return f"saw {entry['actor']} {verb} {short_id(entry['item'])} at {block_text(entry['pos'])}"
        verb, direction = {"chest_take": ("take", "from"), "chest_put": ("put", "into")}[entry["action"]]
        return (f"saw {entry['actor']} {verb} {short_id(entry['item'])} {direction} the chest at "
                f"{block_text(entry['pos'])}")

    @staticmethod
    def _noticed_text(entry: dict) -> str:
        """Render a factual action notice with its sense, or an ordinary noticed fact unchanged."""
        if entry.get("kind") in {"block_use", "transfer"}:
            sense = "saw" if entry.get("sense") == "sight" else "heard"
            return f"{sense} {entry['text']}"
        return entry["text"]

    @staticmethod
    def _action_text(entry: dict) -> str:
        """One remembered action and its outcome, without its time."""
        return f"{entry['tool']} {json.dumps(entry['args'], sort_keys=True)} -> {entry['outcome']}"

    def absorb(self, observation: dict, self_name: str) -> bool:
        """Keep speech, action notices, block changes and witnessed item movements. Return true if someone else's words
        were heard.

        Speech the agent was only partly in range for (`chatter`) and shouting it only saw (`shout_seen`) are kept
        without words. Factual action notices preserve their sense, position and provenance without interrupting as
        speech. Witnessed item movements retain only complete public facts, not private arguments or quantities.
        A change in the agent's inventory becomes an update only once the items actually reach it.
        """
        self._note_inventory(observation.get("gameTime"), observation.get("self", {}).get("inventory"))
        spoke = False
        for item in observation.get("heard", []):
            if item.get("kind") in {"block_use", "transfer"}:
                self._append("noticed", dict(item), f"noticed: {self._noticed_text(item)}")
                continue
            entry = {"gameTime": item["gameTime"], "speaker": item["speaker"], "text": item["text"],
                     "distance": round(item["distance"])}
            self._append("heard", entry, None if item["speaker"] == self_name else f"heard {self._heard_text(entry)}")
            spoke = spoke or item["speaker"] != self_name
        for key, text in (("chatter", "(talking; you could not make out the words)"),
                          ("shout_seen", "(shouting in the distance; you could not make out the words)")):
            for item in observation.get(key, []):
                entry = {"gameTime": item["gameTime"], "speaker": item["speaker"], "text": None, "what": text,
                         "distance": None}
                self._append("heard", entry, f"heard {self._heard_text(entry)}")
        for item in observation.get("changes", []):
            if item["from"] == item["to"]:  # a state-only change (a wall joining its neighbour) reads as no change
                continue
            entry = {"gameTime": item["gameTime"], "pos": item["pos"], "from": item["from"], "to": item["to"]}
            self._append("changes", entry, f"saw a change: {self._change_text(entry)}")
        for item in observation.get("seen_actions") or []:
            if item["action"] not in {"chest_take", "chest_put", "give", "drop", "pickup"}:
                continue
            destination = "target" if item["action"] == "give" else "pos"
            if not {"gameTime", "actor", "item", destination} <= item.keys():
                continue
            entry = {"gameTime": item["gameTime"], "actor": item["actor"], "action": item["action"]}
            entry[destination] = item[destination] if destination == "target" else list(item[destination])
            entry["item"] = item["item"]
            self._append("seen_actions", entry, self._seen_action_text(entry))
        return spoke

    def record_action(self, game_time: int, call: ToolCall, outcome: str, skill_id: str | None = None) -> None:
        """Remember an action, its outcome, and the agent's note if it wrote one.

        For an action that started a skill, pass its `skill_id`, so `resolve_skill` can later record how it ended.
        """
        # a journal entry is kept once, in the journal, not repeated among the actions
        args = {key: value for key, value in call.args.items()
                if key != "note" and not (call.name == "journal" and key == "entry")}
        self._append("actions", {"gameTime": game_time, "tool": call.name, "args": args, "outcome": outcome,
                                 "skillId": skill_id})
        note = call.args.get("note")
        if isinstance(note, str) and note.strip():
            self._append("notes", {"gameTime": game_time, "text": note.strip()[:500]})
        entry = call.args.get("entry") if call.name == "journal" else None
        if isinstance(entry, str) and entry.strip():
            self._append("journal", {"gameTime": game_time, "text": entry.strip()[:800]})

    def last_decision(self) -> int | None:
        """The game time of the latest remembered decision, if any."""
        return self._data["actions"][-1]["gameTime"] if self._data["actions"] else None

    def resolve_skill(self, status: dict) -> None:
        """Update a remembered action's outcome once its skill has finished (succeeded, failed or cancelled).

        For speech, the outcome says who heard it, as a speaker can see whether anyone was close enough.
        """
        for action in self._data["actions"]:
            if action.get("skillId") is not None and action["skillId"] == status.get("skillId"):
                reason = status.get("reason")
                action["outcome"] = status["state"] + (f" ({reason})" if reason else "")
                heard_by = (status.get("progress") or {}).get("heardBy")
                if status.get("skill") == "speak" and heard_by is not None:
                    action["outcome"] += (f" (heard by: {', '.join(heard_by)})" if heard_by
                                          else " (nobody was close enough to hear)")
                action["skillId"] = None
                self._add_fresh(f"  [{clock(action['gameTime'])}] your {self._action_text(action)}")
                drops = (status.get("progress") or {}).get("drops", [])
                if (action["tool"] == "mine" and status.get("skill") == "mine"
                        and status["state"] == "succeeded" and drops):
                    description = "; ".join(f"{drop['count']} {short_id(drop['item'])} at {block_text(drop['pos'])}"
                                            for drop in drops)
                    self.record_noticed(status.get("endedAt", action["gameTime"]), f"unpicked drop: {description}")

    def record_heard(self, game_time: int, speaker: str, text: str) -> None:
        """Remember speech reported by a narrator rather than an engine observation (no distance is known)."""
        entry = {"gameTime": game_time, "speaker": speaker, "text": text, "distance": None}
        self._append("heard", entry, f"heard {self._heard_text(entry)}")

    def record_noticed(self, game_time: int, text: str) -> None:
        """Remember something the agent perceived, from its own skill feedback or a narrator."""
        self._append("noticed", {"gameTime": game_time, "text": text}, f"noticed: {text}")

    def repeated_failure(self) -> str | None:
        """The tool name if the last three actions were the same call and all failed the same way, else None."""
        last = self._data["actions"][-3:]
        if len(last) == 3 and all(a["tool"] == last[0]["tool"] and a["args"] == last[0]["args"]
                                  and a["outcome"] == last[0]["outcome"] for a in last) \
                and last[0]["outcome"].startswith(("not possible", "failed", "invalid")):
            return last[0]["tool"]
        return None

    def record_read(self, game_time: int, text: str) -> None:
        """Remember what the agent read (a container's contents or a book)."""
        self._append("reads", {"gameTime": game_time, "text": text}, f"read: {text}")

    def render(self) -> str:
        """Return the memory as prompt sections, oldest first within each section."""
        data = self._data
        sections = [
            ("Your notes", [f"  [{clock(n['gameTime'])}] {n['text']}" for n in data["notes"]]),
            ("Your recent actions", [f"  [{clock(a['gameTime'])}] {self._action_text(a)}" for a in data["actions"]]),
            ("What you heard", [f"  [{clock(h['gameTime'])}] {self._heard_text(h)}" for h in data["heard"]]),
            ("Changes you saw", [f"  [{clock(c['gameTime'])}] {self._change_text(c)}" for c in data["changes"]]),
            ("Things you read", [f"  [{clock(r['gameTime'])}] {r['text']}" for r in data["reads"]]),
        ]
        if data["seen_actions"]:
            sections.insert(4, ("Actions you saw", [f"  [{clock(entry['gameTime'])}] {self._seen_action_text(entry)}"
                                                   for entry in data["seen_actions"]]))
        if data["noticed"]:
            sections.append(("What you noticed", [f"  [{clock(n['gameTime'])}] {self._noticed_text(n)}"
                                                  for n in data["noticed"]]))
        if data["journal"]:
            sections.insert(0, ("Your journal (private)", [f"  [{clock(j['gameTime'])}] {j['text']}"
                                                           for j in data["journal"]]))
        return "\n\n".join(f"{title}:\n" + ("\n".join(lines) if lines else "  (none)") for title, lines in sections)


class Agent:
    """One role agent: a brief, a memory, its conversation in the current epoch, and the prompts built from them."""

    def __init__(self, brief: RoleBrief, memory: AgentMemory | None = None,
                 renderer: ObservationRenderer | None = None, epoch_turns: int = 1, turns: list[dict] | None = None, *,
                 daily_hunger: bool = False, extras: tuple[str, ...] = (), cues: tuple[dict, ...] = ()):
        """Restore optional turns and memory; enable eating tools only for a daily-hunger scenario, and the
        scenario's other optional tools (`extras`) by name. `cues` are scheduled prompt additions, each
        `{"text", "start", "every"}`: at the agent's first decision in each period (from game time `start`, every
        `every` ticks), its text is added to the prompt (an evening reflection, a counsellor's question)."""
        if epoch_turns < 1:
            raise ValueError("epoch_turns must be at least 1")
        self.brief = brief
        self.memory = memory or AgentMemory()
        self.epoch_turns = epoch_turns
        self.daily_hunger = daily_hunger
        self.extras = tuple(extras)
        self.cues = tuple(dict(cue) for cue in cues)
        self._renderer = renderer or ObservationRenderer()
        self._turns = [dict(turn) for turn in turns or []]

    @property
    def name(self) -> str:
        """The agent's bot name in the world."""
        return self.brief.bot

    @property
    def tools(self) -> list[dict]:
        """The scenario-gated schema used for model requests and their replay digests."""
        return tool_definitions(daily_hunger=self.daily_hunger, extras=self.extras)

    def system_prompt(self) -> str:
        """Return the fixed system prompt: world rules and the role brief."""
        common = f"\n\nWhat everyone in town knows:\n{self.brief.common}" if self.brief.common else ""
        return (f"{WORLD_RULES}{common}\n\nYour name is {self.brief.bot}. Your role: {self.brief.role}.\n"
                f"What people know about your role: {self.brief.public}\n"
                f"Your private goals and knowledge (only you know this): {self.brief.private}")

    def user_prompt(self, observation: dict, interrupted: bool) -> str:
        """Return this decision's prompt from an engine observation: memory, then the rendered observation."""
        return self.prompt_for_view(self._renderer.render(observation), interrupted)

    def prompt_for_view(self, view: str, interrupted: bool, cue: str = "") -> str:
        """Return a full prompt (all of memory) from an already-rendered view (engine-rendered or narrated by a GM)."""
        return f"{self.memory.render()}\n\n{view}\n\n{cue + chr(10) * 2 if cue else ''}{self._ending(interrupted)}"

    def due_cues(self, now: int) -> list[str]:
        """The texts of the cues whose current period began after the agent's previous decision."""
        last = self.memory.last_decision()
        due = []
        for cue in self.cues:
            if now < cue["start"]:
                continue
            period = cue["start"] + (now - cue["start"]) // cue["every"] * cue["every"]
            if last is None or last < period:
                due.append(cue["text"])
        return due

    def next_prompt(self, observation: dict, interrupted: bool, now: int | None = None) -> str:
        """Start this decision's turn and return its user message; `history()` then gives the turns before it.

        The first turn of an epoch is a full prompt. A later turn lists only the updates since the previous turn, as
        the earlier turns stay in the conversation. A new epoch starts once the current one has `epoch_turns` turns,
        or after three identical failures: a model shown its own repeated turns tends to repeat them again.
        """
        cue = "\n\n".join(self.due_cues(now)) if now is not None and self.cues else ""
        return self.next_prompt_for_view(self._renderer.render(observation), interrupted, cue)

    def next_prompt_for_view(self, view: str, interrupted: bool, cue: str = "") -> str:
        """`next_prompt` from an already-rendered view (engine-rendered or narrated by a GM), so engine and narrated
        runs build prompts the same way; a due cue goes just before the closing instruction."""
        updates = self.memory.take_fresh()
        if not self._turns or len(self._turns) >= self.epoch_turns or self.memory.repeated_failure():
            self._turns = []
            return self.prompt_for_view(view, interrupted, cue)
        listed = "\n".join(updates) if updates else "  (nothing new)"
        return (f"Since your last action:\n{listed}\n\n{view}\n\n{cue + chr(10) * 2 if cue else ''}"
                f"{self._ending(interrupted)}")

    def history(self) -> tuple[Turn, ...]:
        """The completed turns of the current epoch, oldest first."""
        return tuple(Turn(turn["user"], ToolCall(turn["tool"], turn["args"]) if turn["tool"] else None, turn["result"])
                     for turn in self._turns)

    def finish_turn(self, user: str, call: ToolCall | None, result: str) -> None:
        """Record the turn just decided: its user message, the action chosen, and the action's immediate result."""
        self._turns.append({"user": user, "tool": call.name if call else None, "args": call.args if call else None,
                            "result": result})

    def _ending(self, interrupted: bool) -> str:
        """The closing instruction of a prompt, with a cue after three identical failures."""
        ending = ("You were interrupted because someone spoke. Choose your next action (you may continue what you "
                  "were doing)." if interrupted else "Choose your next action.")
        repeated = self.memory.repeated_failure()
        if repeated:
            ending = f"Your last three attempts at {repeated} failed the same way; try something different.\n{ending}"
        return ending

    def to_json(self) -> dict:
        """Serialize the agent (brief, memory and current epoch) for a fork snapshot."""
        return {"brief": asdict(self.brief), "memory": self.memory.to_json(), "epochTurns": self.epoch_turns,
                "turns": [dict(turn) for turn in self._turns], **({"dailyHunger": True} if self.daily_hunger else {}),
                **({"tools": list(self.extras)} if self.extras else {}),
                **({"cues": [dict(cue) for cue in self.cues]} if self.cues else {})}

    @classmethod
    def from_json(cls, data: dict) -> Agent:
        """Restore an agent serialized by `to_json()`."""
        return cls(RoleBrief(**data["brief"]), AgentMemory(data["memory"]), epoch_turns=data.get("epochTurns", 1),
                   turns=data.get("turns"), daily_hunger=data.get("dailyHunger", False),
                   extras=tuple(data.get("tools", ())), cues=tuple(data.get("cues", ())))
