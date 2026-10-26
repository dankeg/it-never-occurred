"""A scripted, deepcopy-safe control client for the runtime's unit tests."""

import copy
import json
import math

from scripts.harness import CommandError


class FakeWorld:
    """Answer only the runtime and predicate commands with fixed data and timed skills."""

    def __init__(self, bots: dict[str, list[float]], *, game_time: int = 0,
                 skill_ticks: dict[str, int] | None = None, rejects: dict[str, str] | None = None,
                 listeners: dict[str, list[str]] | None = None, inventories: dict[str, list[dict]] | None = None,
                 containers: dict[tuple, list[dict]] | None = None, books: dict[tuple[str, int], dict] | None = None,
                 blocks: dict[tuple, str] | None = None, npc_metrics: list[dict] | None = None,
                 config: dict | None = None):
        """Copy the scripted data so neither callers nor a fork share mutable world state."""
        self.bots = copy.deepcopy(bots)
        self.game_time = game_time
        self.skill_ticks = dict(skill_ticks or {})
        self.rejects = dict(rejects or {})
        self.listeners = copy.deepcopy(listeners or {})
        self.inventories = copy.deepcopy(inventories or {})
        self.containers = copy.deepcopy(containers or {})
        self.books = copy.deepcopy(books or {})
        self.blocks = dict(blocks or {})
        self.npc_metrics = copy.deepcopy(npc_metrics or [])
        self.config = copy.deepcopy(config or {})
        self.skills = {name: [] for name in bots}
        self.heard = {name: [] for name in bots}
        self.requests = []

    def hear(self, listener: str, speaker: str, text: str) -> None:
        """Queue one complete line of speech at the current game time."""
        self.heard[listener].append({"speaker": speaker, "text": text, "gameTime": self.game_time, "distance": 3.0})

    def _status(self, name: str) -> dict:
        """Return an independent snapshot of the current or last skill, or idle."""
        return copy.deepcopy(self.skills[name][-1] if self.skills[name] else
                             {"skillId": None, "skill": None, "state": "idle", "progress": {}})

    def _start(self, name: str, skill: str, args: dict, *, dry_run: bool = False, ticks: int | None = None) -> dict:
        """Reject, estimate, or start one scripted skill and interrupt its predecessor."""
        reason = self.rejects.get(skill)
        duration = args["ticks"] if skill == "wait" else self.skill_ticks.get(skill, 40)
        if ticks is not None:
            duration = ticks
        if dry_run:
            return {"feasible": reason is None, "estimate": {"ticks": duration if reason is None else None},
                    **({"reason": reason} if reason is not None else {})}
        if reason is not None:
            return {"accepted": False, "reason": reason}
        if self.skills[name] and self.skills[name][-1]["state"] == "running":
            self.skills[name][-1].update(state="cancelled", reason="interrupted")
        skill_id = f"{name}:{len(self.skills[name]) + 1}"
        self.skills[name].append({"skillId": skill_id, "skill": skill, "state": "running", "progress": {},
                                  "args": copy.deepcopy(args), "startedAt": self.game_time,
                                  "endTick": self.game_time + duration})
        return {"accepted": True, "skillId": skill_id}

    def _finish_tick(self) -> None:
        """Complete due skills, moving bots or delivering speech only when they finish."""
        for name, skills in self.skills.items():
            if not skills or skills[-1]["state"] != "running" or skills[-1]["endTick"] > self.game_time:
                continue
            status = skills[-1]
            status["state"] = "succeeded"
            if status["skill"] == "move_to":
                self.bots[name] = [status["args"][axis] for axis in ("x", "y", "z")]
            elif status["skill"] == "speak":
                listeners = self.listeners.get(name, [other for other in self.bots if other != name])
                for listener in listeners:
                    self.hear(listener, name, status["args"]["text"])
                status["progress"]["heardBy"] = list(listeners)

    def request(self, command: str, args: dict | None = None) -> dict:
        """Record and answer one supported harness command, rejecting anything else."""
        args = copy.deepcopy(args or {})
        self.requests.append((command, args))
        if command == "ping":
            return {"gameTime": self.game_time}
        if command == "config_get":
            return copy.deepcopy(self.config)
        if command == "step":
            hashes = []
            for unused in range(args["n"]):
                self.game_time += 1
                self._finish_tick()
                if args.get("region") is not None and self.game_time % args.get("hash_every", 20) == 0:
                    summary = json.dumps({"bots": self.bots, "skills": self.skills}, sort_keys=True)
                    hashes.append({"gameTime": self.game_time, "hash": {"world": summary}})
            return {"gameTime": self.game_time, "hashes": hashes}
        if command == "skill_start":
            return self._start(args["name"], args["skill"], args["args"], dry_run=args.get("dry_run", False))
        if command == "speak":
            status = self._status(args["name"])
            if status["state"] == "running" and status["skill"] == "speak":
                return {"accepted": False, "reason": "busy"}
            return self._start(args["name"], "speak", {key: value for key, value in args.items() if key != "name"},
                               ticks=math.ceil(len(args["text"].split()) / 0.15))
        if command == "ask":
            return self._start(args["name"], "ask", {"npc": args["npc"]})
        if command == "skill_status":
            return self._status(args["name"])
        if command == "observe":
            name = args["name"]
            heard, self.heard[name] = self.heard[name], []
            return {"gameTime": self.game_time,
                    "self": {"name": name, "pos": list(self.bots[name]), "health": 20.0, "food": 20,
                             "inventory": copy.deepcopy(self.inventories.get(name, [])), "skill": self._status(name)},
                    "entities": [], "blocks": [], "documents": [], "heard": heard, "changes": []}
        if command in {"open_container", "container_items"}:
            position = tuple(args["pos"])
            if position not in self.containers:
                raise CommandError(f"{command}: no_container")
            return {"items": copy.deepcopy(self.containers[position])}
        if command == "read_item_doc":
            book = self.books.get((args["name"], args["slot"]))
            if book is None:
                raise CommandError("read_item_doc: no_document")
            return copy.deepcopy(book)
        if command == "blocks":
            return {"blocks": [self.blocks.get(tuple(position), "minecraft:air") for position in args["positions"]]}
        if command == "bot_state":
            return {"inventory": copy.deepcopy(self.inventories.get(args["name"], []))}
        if command == "npc_metrics":
            return {"days": copy.deepcopy(self.npc_metrics)}
        raise CommandError(f"{command}: unknown_command")
