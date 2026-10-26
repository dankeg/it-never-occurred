"""IT-01: the protocol contract between the Python side and the harness (plan §4).

Procedure: on a fresh `base` clone, enable lockstep, save and restore an empty bot snapshot, then spawn two bots and
prepare a stone floor with a sign, a book, a chest, a fixed-UUID boat and one NPC. Send every control command the
Python side uses (about forty) once with valid arguments and, for every command that takes arguments, once with
invalid ones. `ping`, `config_get`, `save_snapshot` and `stop` take no arguments and have no invalid form.

Pass: every valid response has the fields and JSON types the Python callers read; every invalid request raises
`CommandError` with text `<command>: <reason>` and a reason from the Java sources' declared set (`KNOWN_REASONS`);
for a malformed JSON type where the harness passes the JSON library's own message through (`lockstep`,
`await_entities`, a fractional `npc_metrics` day), only the `<command>: ` form is required; read-only commands and a
dry-run `skill_start` leave the region's world hash unchanged (`observe` is not read-only: it drains cursors).

Negative control: the reason checker must report a fabricated `CommandError('ping: it_unknown_reason')`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess

if __package__:
    from . import harness
else:
    import harness


COMMANDS = (
    "acl_set", "ask", "await_entities", "blocks", "bot_break", "bot_flag", "bot_input", "bot_select", "bot_state",
    "command", "config_get", "config_set", "container_items", "dump_entities", "ferry_setup", "give", "hash",
    "lockstep", "npc_metrics", "npc_setup", "npc_state", "observe", "open_container", "ping", "read_item_doc",
    "recording", "reseed", "restore_bots", "save_snapshot", "shop_set", "skill_cancel", "skill_start", "skill_status",
    "spawn_bot", "speak", "step", "stop", "structures_set", "write_book", "write_sign",
)
REGION = {"x0": -8, "z0": -8, "x1": 8, "z1": 8, "y0": -61, "y1": -53}
CHEST = [2, -59, 1]
SIGN = [-1, -59, 1]
BOAT_UUID = "00000000-0000-0000-0000-000000000107"
NO_ARGUMENTS = ("config_get", "ping", "save_snapshot", "stop")
LIBRARY_MESSAGE = "(the JSON library's message)"  # expected reason where the harness passes it through
READ_ONLY = frozenset((
    "ping", "hash", "blocks", "container_items", "bot_state", "skill_status", "config_get", "npc_metrics",
    "dump_entities", "npc_state",
))

# Literal exceptions from mod/src/main/java/org/minecraftsim/harness/{ControlCommands,ControlSocket,Skills,
# Documents,Ferry,Structures,Villagers,BotManager,Readiness,Region,SimConfig}.java. Dynamic strings are materialized
# only for the fixed negative arguments below. Gson/BigDecimal exception messages are not declared reason codes.
KNOWN_REASONS = (
    "ACL entries must be bot names",
    "At least two landings are required",
    "Block query outside loaded chunks: 1000000,-59,1000000",
    "Boat must be within two blocks of a landing moor",
    "Bot coordinates must be finite",
    "Bot is not managed by this harness",
    "Bot names must be 1-16 ASCII letters, digits or underscores",
    "Bot snapshot game time does not match the world",
    "Commands must run on the server main thread",
    "Container has unresolved loot; opening it would generate loot and consume world RNG",
    "Documents and containers are limited to the overworld",
    "Documents must be accessed on the server main thread",
    "Ferry already configured",
    "Ferry duration exceeds the skill tick range",
    "Game time advanced during await_entities",
    "Hotbar slot must be between 0 and 8",
    "Invalid bot entry",
    "Invalid moor coordinate",
    "Invalid saved skill",
    "Invalid schematic name",
    "Item count must be between 1 and 2304",
    "Item has no book content",
    "Landing names must not be blank",
    "NPC IDs must be unique and nonempty; role must be adult or child",
    "Prices must be nonnegative",
    "Region minimum bounds must not exceed maximum bounds",
    "Reserved ferry rider has no running ferry skill",
    "Schedule times must lie within a day",
    "Schematic must be inside simharness/schematics",
    "Skills are limited to the overworld",
    "Skills require a bot without infinite materials",
    "Structures need unique nonblank names and at least one block",
    "Target chunk is not loaded",
    "Target is not visible",
    "Target is outside document/container reach",
    "Unknown bot: it_missing",
    "Unknown configuration key: it_invalid",
    "Unknown skill: it_invalid",
    "Vanilla command failed: it_invalid",
    "air is not an item",
    "allowed must be null, '*' or an array of bot names",
    "args must be an object",
    "base_seed must be an integer",
    "boat must identify a loaded, unoccupied boat",
    "enabled must be a boolean",
    "hash_every must be positive",
    "id must be a number",
    "market_days entries must be 0..6",
    "master_key must be a boolean",
    "moor must contain three coordinates",
    "npc_setup is required",
    "pos requires three coordinates",
    "region is required",
    "restore_bots requires idle lockstep and no existing bots",
    "slot must be an integer inventory slot",
    "speed must be finite and positive",
    "terrain_step must be positive",
)

INVALID_REQUESTS = {
    "acl_set": ({"pos": CHEST, "allowed": 0}, "allowed must be null, '*' or an array of bot names"),
    "ask": ({"name": "it_missing", "npc": "it_npc"}, "Unknown bot: it_missing"),
    "await_entities": ({"cx0": {}, "cz0": 0, "cx1": 0, "cz1": 0}, LIBRARY_MESSAGE),
    "blocks": ({"positions": [[1000000, -59, 1000000]]},
               "Block query outside loaded chunks: 1000000,-59,1000000"),
    "bot_break": ({"name": "it_a", "pos": []}, "pos requires three coordinates"),
    "bot_flag": ({"name": "it_a", "master_key": 0}, "master_key must be a boolean"),
    "bot_input": ({"name": "it_missing"}, "Unknown bot: it_missing"),
    "bot_select": ({"name": "it_a", "slot": -1}, "Hotbar slot must be between 0 and 8"),
    "bot_state": ({"name": "it_missing"}, "Unknown bot: it_missing"),
    "command": ({"command": "it_invalid"}, "Vanilla command failed: it_invalid"),
    "config_set": ({"it_invalid": 0}, "Unknown configuration key: it_invalid"),
    "container_items": ({"pos": []}, "pos requires three coordinates"),
    "dump_entities": ({}, "region is required"),
    "ferry_setup": ({"boat": BOAT_UUID, "speed": 0}, "speed must be finite and positive"),
    "give": ({"name": "it_a", "item": "minecraft:bread", "count": 0}, "Item count must be between 1 and 2304"),
    "hash": ({}, "region is required"),
    "lockstep": ({"enabled": {}}, LIBRARY_MESSAGE),
    "npc_metrics": ({}, "npc_setup is required"),
    "npc_setup": ({"speed": 0}, "speed must be finite and positive"),
    "npc_state": ({"id": "it_npc"}, "npc_setup is required"),
    "observe": ({"name": "it_missing"}, "Unknown bot: it_missing"),
    "open_container": ({"name": "it_a", "pos": []}, "pos requires three coordinates"),
    "read_item_doc": ({"name": "it_missing", "slot": 0}, "Unknown bot: it_missing"),
    "recording": ({"enabled": 0}, "enabled must be a boolean"),
    "reseed": ({"base_seed": "it_invalid"}, "base_seed must be an integer"),
    "restore_bots": ({}, "restore_bots requires idle lockstep and no existing bots"),
    "shop_set": ({"pos": CHEST, "prices": {"minecraft:bread": -1}}, "Prices must be nonnegative"),
    "skill_cancel": ({"name": "it_missing"}, "Unknown bot: it_missing"),
    "skill_start": ({"name": "it_a", "skill": "it_invalid", "args": {}}, "Unknown skill: it_invalid"),
    "skill_status": ({"name": "it_missing"}, "Unknown bot: it_missing"),
    "spawn_bot": ({"name": "!"}, "Bot names must be 1-16 ASCII letters, digits or underscores"),
    "speak": ({"name": "it_missing", "text": "Hi"}, "Unknown bot: it_missing"),
    "step": ({"n": 1, "hash_every": 0}, "hash_every must be positive"),
    "structures_set": ({"structures": [{"name": "", "blocks": []}]},
                       "Structures need unique nonblank names and at least one block"),
    "write_book": ({"name": "it_missing", "slot": 0}, "Unknown bot: it_missing"),
    "write_sign": ({"name": "it_a", "pos": []}, "pos requires three coordinates"),
}

NUMBER = (int, float)
TIME_FIELDS = {"gameTime": int}
HASH_FIELDS = {**TIME_FIELDS, **dict.fromkeys(harness.HASH_KEYS, str)}
BOOK_METADATA = {"prov": str, "title": str, "author": str}
ITEM_FIELDS = {"slot": int, "item": str, "count": int, "doc?": BOOK_METADATA}
SKILL_FIELDS = {
    "skillId": (str, type(None)), "skill": (str, type(None)), "args": dict, "state": str,
    "startedAt": (int, type(None)), "endedAt?": int, "reason?": str,
    "progress": {"placed?": int, "total?": int, "ticks?": int, "phase?": str, "uttId?": int},
}
BOT_FIELDS = {
    "name": str, "pos": [NUMBER], "deltaMovement": [NUMBER], "yRot": NUMBER, "xRot": NUMBER,
    "onGround": bool, "health": NUMBER, "food": int, "selectedSlot": int, "inventory": [ITEM_FIELDS],
}
CONFIG_FIELDS = {
    **dict.fromkeys(("view_radius", "reach", "speech_radius", "speech_rate", "talk_radius", "shout_radius",
                     "relay_rate", "exposure_log"), NUMBER),
    **dict.fromkeys(("terrain_radius", "terrain_step", "shout_cooldown_ticks", "attention_slots", "news_ttl_days",
                     "craft_ticks", "place_ticks", "container_ticks", "give_ticks"), int),
    "notable_blocks": [str],
}
NEWS_FIELDS = {
    "newsId": int, "kind": str, "subject": (str, list, dict), "originEventId": int,
    "chain": [str], "firsthand": bool, "learnedAt": int, "originatedAt": int, "salience": int, "text?": str,
}
TRIP_FIELDS = {
    "day": int, "npc": str, "from": str, "to": str, "activity": str, "departTick": int,
    "arriveTick": (int, type(None)), "ticks": int, "pathLength": int, "onTime": (bool, type(None)),
    "outcome": str, "travelled": NUMBER, "path": [[int]],
    "segments": [{"tick": int, "position": [NUMBER], "travelled": NUMBER, "movingTicks": int,
                  "snapDistance": NUMBER, "path": [[int]]}],
}
DAILY_FIELDS = {
    "day": int, "npcs": [{"npc": str, "commuteTicks": int, "workOnTime?": bool,  # null fields are omitted
                           "workStatus": str, "marketVisited?": bool, "marketStatus": str,
                           "schoolStatus": str}],
    "trips": [TRIP_FIELDS], "meanCommute": NUMBER, "maxCommute": int, "lateShare": NUMBER,
    "missedMarketShare": NUMBER, "absentShare": NUMBER, "affectedNpcs": [str],
}
HEARD_FIELDS = {
    "prov": str, "speaker": str, "gameTime": int, "text?": str, "volume?": str, "distance?": NUMBER,
    "to?": str, "firsthand?": bool, "chain?": [str], "originEventId?": int, "newsId?": int,
    "kind?": str, "subject?": (str, list, dict), "news?": [NEWS_FIELDS],
}
OBSERVATION_FIELDS = {
    **TIME_FIELDS,
    "self": {"name": str, "pos": [NUMBER], "yaw": NUMBER, "pitch": NUMBER, "health": NUMBER, "food": int,
             "selectedSlot": int, "skill": SKILL_FIELDS, "inventory": [ITEM_FIELDS]},
    "entities": [{"prov": str, "uuid": str, "type": str, "pos": [NUMBER], "distance": NUMBER,
                  "name?": str, "npcId?": str}],
    "blocks": [{"prov": str, "pos": [int], "block": str, "distance": NUMBER}],
    "structures": [{"prov": str, "name": str, "pos": [int], "built": int, "total": int, "distance": NUMBER}],
    "terrain": [{"x": int, "y": int, "z": int, "block": str}],
    "documents": [{"prov": str, "pos": [int], "distance": NUMBER, "front?": [str], "back?": [str],
                   "title?": str, "author?": str, "pages?": [str]}],
    "changes": [{"prov": str, "pos": [int], "from": str, "to": str, "gameTime": int}],
    "heard": [HEARD_FIELDS], "chatter": [HEARD_FIELDS], "shout_seen": [HEARD_FIELDS],
}
RESPONSE_FIELDS = {
    **dict.fromkeys(("acl_set", "ping", "reseed", "save_snapshot", "shop_set", "stop"), TIME_FIELDS),
    **dict.fromkeys(("spawn_bot", "bot_state", "bot_input", "bot_select", "give"), BOT_FIELDS),
    **dict.fromkeys(("config_get", "config_set"), CONFIG_FIELDS),
    **dict.fromkeys(("skill_cancel", "skill_status"), SKILL_FIELDS),
    "ask": {"accepted": bool, "skillId": str, "uttId": int},
    "await_entities": {**TIME_FIELDS, "chunks": int, "waitedMs": NUMBER},
    "blocks": {**TIME_FIELDS, "blocks": [str]},
    "bot_break": {**TIME_FIELDS, "destroyed": bool},
    "bot_flag": {"master_key": bool},
    "command": {**TIME_FIELDS, "value": int},
    "container_items": {**TIME_FIELDS, "items": [{"slot": int, "item": str, "count": int,
                                                  "title?": str, "author?": str, "pages?": [str]}]},
    "dump_entities": {"entities": [{"uuid": str, "type": str, "pos": str, "motion": str, "rot": str, "nbt": str}]},
    "ferry_setup": {"boat": str, "speed": NUMBER, "landing": str,
                   "landings": {"*": {"board": [int], "moor": [NUMBER]}}},
    "hash": HASH_FIELDS,
    "lockstep": {**TIME_FIELDS, "enabled": bool},
    "npc_metrics": {"days": [DAILY_FIELDS]},
    "npc_setup": {"ignored": bool},
    "npc_state": {"id": str, "npcId": str, "name": str, "uuid": str, "pos": [NUMBER], "activity": str,
                  "path": [[int]], "index": int, "offset": NUMBER, "graphVersion": int,
                  "trip?": dict, "news": [NEWS_FIELDS]},  # no trip: the field is omitted
    "observe": OBSERVATION_FIELDS,
    "open_container": {"prov": str, "items": [ITEM_FIELDS]},
    "read_item_doc": {**BOOK_METADATA, "pages": [str]},
    "recording": {**TIME_FIELDS, "enabled": bool},
    "restore_bots": {**TIME_FIELDS, "bots": [BOT_FIELDS]},
    "skill_start": {"accepted": bool, "skillId": str},
    "speak": {"accepted": bool, "skillId": str, "uttId": int},
    "step": {**TIME_FIELDS, "hashes": [{**TIME_FIELDS, "hash": HASH_FIELDS}]},
    "structures_set": {"structures": int},
    "write_book": {"prov": str},
    "write_sign": {"prov": str},
}
DRY_RUN_FIELDS = {"feasible": bool, "estimate": {"ticks": int, "materials": {"*": int}}, "reason?": str}


def field_problems(value, schema, path: str = "result") -> list[str]:
    """Check nested required/optional keys and every list/map value, rejecting bool wherever numbers are required."""
    if isinstance(schema, dict):
        if type(value) is not dict:
            return [f"{path}: expected object, got {type(value).__name__}"]
        problems = []
        for field, nested in schema.items():
            if field == "*":
                for key, item in value.items():
                    problems.extend(field_problems(item, nested, f"{path}.{key}"))
                continue
            key = field.removesuffix("?")
            if key not in value:
                if not field.endswith("?"):
                    problems.append(f"{path}.{key}: missing required field")
            else:
                problems.extend(field_problems(value[key], nested, f"{path}.{key}"))
        return problems
    if isinstance(schema, list):
        if type(value) is not list:
            return [f"{path}: expected array, got {type(value).__name__}"]
        return [problem for index, item in enumerate(value)
                for problem in field_problems(item, schema[0], f"{path}[{index}]" )]
    types = schema if isinstance(schema, tuple) else (schema,)
    if type(value) not in types:
        names = "/".join(kind.__name__ for kind in types)
        return [f"{path}: expected {names}, got {type(value).__name__}"]
    if type(value) is float and not math.isfinite(value):
        return [f"{path}: expected a finite number"]
    return []


def response_problems(command: str, response, *, dry_run: bool = False) -> list[str]:
    """Validate the actual caller-consumed response shape and require successful valid skill/setup requests."""
    if command not in RESPONSE_FIELDS:
        return [f"No response contract for {command}"]
    schema = DRY_RUN_FIELDS if dry_run else RESPONSE_FIELDS[command]
    problems = field_problems(response, schema)
    if type(response) is not dict:
        return problems
    if command in ("skill_start", "speak", "ask"):
        field = "feasible" if dry_run else "accepted"
        if response.get(field) is not True:
            problems.append(f"result.{field}: valid request was not successful")
    if command == "npc_setup" and response.get("ignored") is not False:
        problems.append("result.ignored: valid NPC setup did not initialize the fixture")
    if command == "npc_state" and type(response.get("trip")) is dict:
        problems.extend(field_problems(response["trip"], TRIP_FIELDS, "result.trip"))
    if "gameTime" in response:
        try:
            harness.game_time(response)
        except harness.ProtocolError as error:
            problems.append(str(error))
    return problems


def rejection_problems(command: str, error: Exception | None, expected_reason: str | None = None) -> list[str]:
    """Require CommandError, an exact command prefix and a known, optionally case-specific, Java rejection reason
    (only the prefix where the expected reason is the JSON library's own message)."""
    problems = []
    if not isinstance(error, harness.CommandError):
        problems.append(f"{command}: expected CommandError, got {type(error).__name__ if error else 'accepted probe'}")
        return problems
    prefix = f"{command}: "
    if not str(error).startswith(prefix):
        return [*problems, f"Expected exact {prefix!r} prefix, got {str(error)!r}"]
    reason = str(error)[len(prefix):]
    if expected_reason == LIBRARY_MESSAGE:
        return problems
    if reason not in KNOWN_REASONS:
        problems.append(f"Unknown rejection reason: {reason!r}")
    if expected_reason is not None and reason != expected_reason:
        problems.append(f"Expected reason {expected_reason!r}, got {reason!r}")
    return problems


class ProtocolExperiment:
    """Own two isolated server lifecycles and report the complete, deliberately fail-closed command contract."""

    def __init__(self, args: argparse.Namespace):
        """Retain options and initialize evidence without creating directories or starting servers."""
        self._args = args
        self._directory = args.runs_dir / "it01_protocol"
        self._checks: list[dict] = []
        self._valid_commands: set[str] = set()
        self._invalid_commands: set[str] = set()

    def setup(self) -> None:
        """Clone only the supplied base fixture into a fresh run directory under --runs-dir."""
        harness.clone_apfs(self._args.runs_dir / "base", self._directory)

    def _record(self, label: str, problems: list[str]) -> None:
        """Retain each individual result and its failure explanations without suppressing negative cases."""
        self._checks.append({"label": label, "passed": not problems, "problems": problems})

    def _valid(self, client, command: str, args: dict, *, label: str = "") -> dict | None:
        """Validate one successful response and bracket every declared read-only query with harness world hashes."""
        self._valid_commands.add(command)
        readonly = command in READ_ONLY or (command == "skill_start" and args.get("dry_run") is True)
        suffix = f" ({label})" if label else ""
        try:
            before = client.request("hash", {"region": REGION}) if readonly else None
            response = client.request(command, args)
            self._record(f"valid {command}{suffix}", response_problems(
                command, response, dry_run=args.get("dry_run") is True))
            if readonly:
                after = client.request("hash", {"region": REGION})
                problems = response_problems("hash", before) + response_problems("hash", after)
                if before != after:
                    problems.append("regional harness hash changed across a read-only request")
                self._record(f"read-only {command}{suffix}", problems)
            return response
        except (OSError, ValueError, harness.HarnessError) as error:
            self._record(f"valid {command}{suffix}", [f"{type(error).__name__}: {error}"])
            if readonly:
                self._record(f"read-only {command}{suffix}", ["could not establish unchanged before/after hashes"])
            return None

    def _invalid(self, client, command: str, *, args: dict | None = None, reason: str | None = None) -> None:
        """Send an explicit negative request, reporting an accepted probe or a wrong rejection as BAD."""
        self._invalid_commands.add(command)
        arguments, expected = INVALID_REQUESTS[command] if args is None else (args, reason)
        error = None
        try:
            client.request(command, arguments)
        except (OSError, ValueError, harness.HarnessError) as failure:
            error = failure
        self._record(f"invalid {command}: {arguments!r}", rejection_problems(command, error, expected))

    def _spawn(self, client) -> None:
        """Spawn exactly two survival bots on the stone platform, keeping their inventories initially empty."""
        for name, horizontal in (("it_a", 0.5), ("it_b", 1.5)):
            self._valid(client, "spawn_bot", {"name": name, "x": horizontal, "y": -59, "z": 0.5,
                                               "yaw": 0, "pitch": 0}, label=name)

    def _prepare_world(self, client) -> None:
        """Build a clear stone floor, reachable nonempty chest, sign, break target and fixed-UUID boat."""
        for command in (
            "setblock -8 -59 -8 minecraft:barrier",
            "fill -8 -59 -8 8 -53 8 minecraft:air",
            "setblock -8 -60 -8 minecraft:barrier",
            "fill -8 -60 -8 8 -60 8 minecraft:stone",
            "setblock 2 -59 1 minecraft:chest",
            "item replace block 2 -59 1 container.0 with minecraft:bread 2",
            "setblock -1 -59 1 minecraft:oak_sign",
            "setblock -1 -59 -1 minecraft:oak_planks",
            "summon minecraft:oak_boat 4.5 -59 4.5 {UUID:[I;0,0,0,263]}",
        ):
            self._valid(client, "command", {"command": command}, label=command)

    def _exercise(self, client) -> None:
        """Exercise setup-sensitive cases in dependency order, then query populated nested response structures."""
        self._valid(client, "lockstep", {"enabled": True})
        self._invalid(client, "lockstep")
        self._valid(client, "await_entities", harness.FORCELOADED_CHUNKS)
        self._invalid(client, "await_entities")
        self._valid(client, "recording", {"enabled": False})
        self._invalid(client, "recording")
        self._valid(client, "save_snapshot", {})
        self._valid(client, "restore_bots", {})
        self._invalid(client, "npc_metrics")
        self._invalid(client, "npc_state")
        self._invalid(client, "npc_setup")
        self._prepare_world(client)
        self._spawn(client)
        for command in COMMANDS:
            if command not in self._invalid_commands and command not in ("ferry_setup", *NO_ARGUMENTS):
                self._invalid(client, command)
        self._valid(client, "config_set", {"view_radius": 6, "terrain_radius": 2, "terrain_step": 1,
                                             "exposure_log": 0})
        self._valid(client, "recording", {"enabled": True})
        self._valid(client, "reseed", {"base_seed": self._args.base_seed})
        self._valid(client, "bot_flag", {"name": "it_a", "master_key": False})
        self._valid(client, "bot_input", {"name": "it_a", "forward": 0, "strafe": 0, "jump": False,
                                            "sneak": False, "sprint": False, "yaw": 0, "pitch": 0})
        self._valid(client, "bot_select", {"name": "it_a", "slot": 0})
        self._valid(client, "give", {"name": "it_a", "item": "minecraft:writable_book", "count": 1})
        self._valid(client, "write_book", {"name": "it_a", "slot": 0, "title": "IT-01", "pages": ["protocol"]})
        self._valid(client, "read_item_doc", {"name": "it_a", "slot": 0})
        self._valid(client, "command", {"command": "item replace block 2 -59 1 container.1 from entity it_a hotbar.0"})
        self._valid(client, "write_sign", {"name": "it_a", "pos": SIGN, "front": ["IT-01", "protocol", "", ""],
                                             "back": ["", "", "", ""]})
        self._valid(client, "acl_set", {"pos": CHEST, "allowed": "*"})
        self._valid(client, "shop_set", {"pos": CHEST, "prices": {"minecraft:bread": 1}})
        self._valid(client, "container_items", {"pos": CHEST})
        self._valid(client, "open_container", {"name": "it_a", "pos": CHEST})
        self._valid(client, "bot_break", {"name": "it_a", "pos": [-1, -59, -1]})
        self._valid(client, "structures_set", {"structures": [{"name": "protocol_sign", "anchor": SIGN,
                                                                "blocks": [{"pos": SIGN,
                                                                            "block": "minecraft:oak_sign"}]}]})
        self._invalid(client, "ferry_setup")
        self._valid(client, "ferry_setup", {"boat": BOAT_UUID, "speed": 0.2, "landings": {
            "home": {"board": [4, -60, 3], "moor": [4.5, -59, 4.5]},
            "away": {"board": [6, -60, 3], "moor": [6.5, -59, 4.5]},
        }})
        self._valid(client, "npc_setup", {"region": REGION, "surfaces": ["minecraft:stone"],
                                            "places": {"home": [-2, -60, 0]},
                                            "npcs": [{"id": "it_npc", "name": "Protocol NPC", "home": "home",
                                                      "role": "adult"}]})
        self._invalid(client, "npc_metrics", args={"day": 0.5}, reason=LIBRARY_MESSAGE)
        self._valid(client, "npc_metrics", {})
        self._valid(client, "npc_state", {"id": "it_npc"})
        self._valid(client, "config_get", {})
        self._valid(client, "ping", {})
        self._valid(client, "blocks", {"positions": [CHEST, SIGN, [-2, -60, 0]]})
        self._valid(client, "hash", {"region": REGION})
        self._valid(client, "dump_entities", {"region": REGION})
        self._valid(client, "bot_state", {"name": "it_a"})
        self._valid(client, "skill_start", {"name": "it_a", "skill": "wait", "args": {"ticks": 20}})
        self._valid(client, "skill_status", {"name": "it_a"})
        self._valid(client, "skill_start", {"name": "it_a", "skill": "wait", "args": {"ticks": 10},
                                              "dry_run": True}, label="dry run")
        self._valid(client, "skill_cancel", {"name": "it_a"})
        self._valid(client, "speak", {"name": "it_b", "text": "Hi", "volume": "talk", "to": "it_a"})
        self._valid(client, "step", {"n": 20, "hash_every": 20, "region": REGION})
        self._valid(client, "observe", {"name": "it_a", "include_terrain": True})
        self._valid(client, "ask", {"name": "it_a", "npc": "it_npc"})
        self._valid(client, "stop", {})

    def run(self) -> None:
        """Run the whole contract on one server; `stop` comes last."""
        with harness.Server(self._directory, **harness.server_options(self._args)) as server:
            self._exercise(server.client)

    def check(self) -> bool:
        """Report all results, explicit coverage failures and the fabricated-unknown-reason negative control."""
        for command in COMMANDS:
            self._record(f"coverage {command}", [] if (
                command in self._valid_commands and (command in self._invalid_commands or command in NO_ARGUMENTS))
                else ["missing a valid or invalid request attempt"])
        fabricated = harness.CommandError("ping: it_unknown_reason")
        detected = rejection_problems("ping", fabricated)
        self._record("negative control: fabricated unknown rejection reason detected", [] if any(
            "Unknown rejection reason" in problem for problem in detected) else ["unknown reason escaped the checker"])
        for result in self._checks:
            detail = "; ".join(result["problems"])
            print(f"  {'ok ' if result['passed'] else 'BAD'} {result['label']}" + (f": {detail}" if detail else ""))
        passed = all(result["passed"] for result in self._checks)
        report = {"passed": passed, "valid_commands": sorted(self._valid_commands),
                  "invalid_commands": sorted(self._invalid_commands), "checks": self._checks}
        (self._directory / "protocol_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-01: forty-command protocol contract, including rejection coverage")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse shared options, and run setup/run/check without hiding contract failures."""
    harness.stop_on_sigterm()
    experiment = ProtocolExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    try:
        experiment.setup()
        experiment.run()
    except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
        experiment._record("experiment completed", [f"{type(error).__name__}: {error}"])
        if not experiment._directory.is_dir():
            print(f"  BAD {error}\nFAIL IT-01: {error}")
            return 1
    return 0 if experiment.check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
