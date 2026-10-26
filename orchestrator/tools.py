"""Agent tools: the model-facing action schema and its mapping onto grounded harness commands.

Every tool maps to exactly one harness command, so the model gets no capability the engine doesn't enforce. Skills
(movement, mining, building, transfers, speech, asking a villager) run over ticks and finish later. Information actions
(reading a book, opening a container) return immediately, and their result goes into the agent's memory. Each tool
accepts an optional `note`, which the agent keeps in its memory. That's the agent's only scratchpad, and it costs no
game time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import ToolCall

_XYZ = {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}}
# The note's guidance is stated once in the world rules; repeating it in every tool cost ~600 prompt tokens per
# decision.
_NOTE = {"note": {"type": "string", "maxLength": 100}}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one tool definition in the shared (OpenAI-style JSON schema) form."""
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {**properties, **_NOTE}, "required": required,
                           "additionalProperties": False}}


TOOL_DEFINITIONS: list[dict] = [
    _tool("move_to", "Walk to a block position (pathfinding). Takes real time.", dict(_XYZ), ["x", "y", "z"]),
    _tool("mine", "Break the block at a position within reach. Takes time depending on the block and your tool.",
          dict(_XYZ), ["x", "y", "z"]),
    _tool("collect", "Pick up visible dropped items nearby.", {}, []),
    _tool("craft", "Craft an item from materials in your inventory (some recipes need a crafting table in reach).",
          {"item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}}, ["item"]),
    _tool("place", "Place a block item from your inventory at a position within reach.",
          {**_XYZ, "item": {"type": "string"}}, ["x", "y", "z", "item"]),
    _tool("build_section", "Build a named construction section (e.g. a bridge span) at an origin, using materials from "
          "your inventory. Takes time; you must hold the full material bill.",
          {"schematic": {"type": "string"}, "origin": {"type": "array", "items": {"type": "integer"},
                                                       "minItems": 3, "maxItems": 3}}, ["schematic", "origin"]),
    _tool("chest_put", "Put items into a container within reach.",
          {**_XYZ, "item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
          ["x", "y", "z", "item", "count"]),
    _tool("chest_take", "Take items from a container within reach (if you have access).",
          {**_XYZ, "item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
          ["x", "y", "z", "item", "count"]),
    _tool("buy", "Buy goods from a market stall chest within reach, paying its posted price in emeralds.",
          {**_XYZ, "item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
          ["x", "y", "z", "item", "count"]),
    _tool("ferry", "Ride the port's boat across the river to a landing (stand at the landing where the boat is). "
          "Takes time; the boat stays where it lands.",
          {"to": {"type": "string", "enum": ["east", "west"], "description": "The landing to cross to."}}, ["to"]),
    _tool("give", "Hand items to a person standing next to you.",
          {"target": {"type": "string"}, "item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
          ["target", "item", "count"]),
    _tool("use", "Use the block at a position within reach: open or close a door, trapdoor or gate, ring a bell, "
          "flip a lever or press a button.", dict(_XYZ), ["x", "y", "z"]),
    _tool("drop", "Drop items from your inventory on the ground at your feet. Anyone can pick them up (you take back "
          "your own only with collect); items left on the ground vanish after about five minutes.",
          {"item": {"type": "string"}, "count": {"type": "integer", "minimum": 1}}, ["item", "count"]),
    _tool("hold", "Hold an item from your inventory in your hand, where others can see it.",
          {"item": {"type": "string"}}, ["item"]),
    _tool("speak", "Say something out loud (about 3 words per second). Whispering carries 3 blocks at the same rate "
          "as talk, with no shout cooldown; people beyond 3 but within 8 blocks who can see you notice talking "
          "without hearing the words. Talking carries 8 blocks; shouting carries 24 blocks, takes longer, is noticed "
          "from farther away, and can only be done once every 30 seconds. People only catch your words if they "
          "are in range for the whole time you speak.",
          {"text": {"type": "string", "maxLength": 600, "description": "At most about 100 words."},
           "volume": {"type": "string", "enum": ["whisper", "talk", "shout"]},
           "to": {"type": "string", "description": "Optional: the person you are addressing (others may overhear)."}},
          ["text"]),
    _tool("ask", "Ask a villager within 8 blocks what news they have; they answer aloud with what they know.",
          {"npc": {"type": "string", "description": "The villager's id (e.g. adult_07)."}}, ["npc"]),
    _tool("open_container", "Look inside a container within reach (if you have access).", dict(_XYZ), ["x", "y", "z"]),
    _tool("read_book", "Read a written book in your inventory slot.", {"slot": {"type": "integer"}}, ["slot"]),
    _tool("write_book", "Write and sign a book using a writable book in your inventory slot.",
          {"slot": {"type": "integer"}, "title": {"type": "string"},
           "pages": {"type": "array", "items": {"type": "string"}}}, ["slot", "title", "pages"]),
    _tool("write_sign", "Write text on a sign within reach (4 short lines).",
          {**_XYZ, "lines": {"type": "array", "items": {"type": "string"}, "maxItems": 4}}, ["x", "y", "z", "lines"]),
    _tool("wait", "Do nothing for a while (ticks; 20 ticks = 1 second).",
          {"ticks": {"type": "integer", "minimum": 1, "maximum": 1200}}, ["ticks"]),
    _tool("keep_going", "Carry on with your current activity (if you have none, pause briefly).", {}, []),
]

_EAT_TOOL = _tool("eat", "Eat food from your inventory (it takes a few seconds).",
                  {"item": {"type": "string"}}, ["item"])
_BAKE_TOOL = _tool("bake", "Bake bread at an unlit smoker in reach and in sight, using 3 wheat per loaf from your "
                   "inventory. Takes time; the oven glows while baking and the bread must fit your carrying limit.",
                   {"count": {"type": "integer", "minimum": 1, "maximum": 32}}, ["count"])
# `sleep` is a wait until 06:00 (the runtime knows the time); speech interrupts it like any other action
_SLEEP_TOOL = _tool("sleep", "Sleep until 06:00. Someone speaking to you wakes you.", {}, [])
# `journal` changes nothing in the world: a private entry the agent keeps (and rereads), written in a moment
_JOURNAL_TOOL = _tool("journal", "Write in your private journal what you think is going on, what you suspect and what "
                      "you plan. Nobody else can read it. It takes a moment.",
                      {"entry": {"type": "string", "maxLength": 800}}, ["entry"])
# Tools a scenario enables by name, in the fixed order they are appended to the replay-stable default list
_OPTIONAL_TOOLS = {"eat": _EAT_TOOL, "bake": _BAKE_TOOL, "sleep": _SLEEP_TOOL, "journal": _JOURNAL_TOOL}
# `ask_about` (the engine's `ask_about` setting) swaps in an `ask` that may name a person or place to ask about
_ASK_ABOUT_TOOL = _tool("ask", "Ask a villager within 8 blocks what news they have, or what they have seen or heard "
                        "about a person or a place; they answer aloud with what they know.",
                        {"npc": {"type": "string", "description": "The villager's id (e.g. adult_07)."},
                         "about": {"type": "string", "description": "Optional: a person's name or a place's name."}},
                        ["npc"])
_TOOL_FLAGS = {"ask_about"}
TOOL_NAMES = {tool["name"] for tool in [*TOOL_DEFINITIONS, *_OPTIONAL_TOOLS.values()]}
_REQUIRED = {tool["name"]: tool["parameters"]["required"] for tool in [*TOOL_DEFINITIONS, *_OPTIONAL_TOOLS.values()]}


def tool_definitions(*, daily_hunger: bool = False, extras: tuple[str, ...] = ()) -> list[dict]:
    """Select the scenario's model-facing tools without changing the replay-stable default definitions.

    `daily_hunger` enables `eat`; `extras` names any other optional tools the scenario enables."""
    names = set(extras) | ({"eat"} if daily_hunger else set())
    if names - _OPTIONAL_TOOLS.keys() - _TOOL_FLAGS:
        raise ValueError(f"Unknown optional tools: {sorted(names - _OPTIONAL_TOOLS.keys() - _TOOL_FLAGS)}")
    if not names:
        return TOOL_DEFINITIONS
    defaults = [_ASK_ABOUT_TOOL if "ask_about" in names and tool["name"] == "ask" else tool
                for tool in TOOL_DEFINITIONS]
    return [*defaults, *(tool for name, tool in _OPTIONAL_TOOLS.items() if name in names)]


def item_id(name: str) -> str:
    """Normalize an item name as agents write it ("Oak Planks", "oak_planks", or a book copied from the inventory as
    'written_book "Report" by mayor') to a registry id ("minecraft:oak_planks", "minecraft:written_book")."""
    name = name.split('"')[0].strip().lower().replace(" ", "_")
    return name if ":" in name else f"minecraft:{name}"


@dataclass(frozen=True)
class HarnessAction:
    """A tool call translated into one harness command.

    `is_skill` is true when the command starts an engine skill, which may finish instantly. The runtime waits for
    the skill to finish before the agent decides again.
    """

    command: str
    args: dict
    is_skill: bool


class ToolTranslator:
    """Translates a model `ToolCall` into the harness command that enforces it."""

    _SKILLS = {"move_to", "mine", "collect", "craft", "place", "chest_put", "chest_take", "buy", "give", "use", "drop",
               "hold", "eat", "bake", "wait", "ferry"}

    def translate(self, bot: str, call: ToolCall) -> HarnessAction | None:
        """Return the harness action for `call` made by `bot`, or None for `keep_going`.

        Raises ValueError for an unknown tool or invalid count, and KeyError or TypeError for malformed arguments.
        """
        if call.name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool: {call.name}")
        missing = [key for key in _REQUIRED[call.name] if key not in call.args]
        if missing:
            raise KeyError(f"missing arguments {missing}")
        if call.name == "keep_going":
            return None
        args = {key: value for key, value in call.args.items() if key != "note"}
        if isinstance(args.get("item"), str):
            args["item"] = item_id(args["item"])
        if call.name in self._SKILLS:
            skill_args = self._skill_args(call.name, args)
            return HarnessAction("skill_start", {"name": bot, "skill": call.name, "args": skill_args}, True)
        if call.name == "build_section":
            return HarnessAction("skill_start", {"name": bot, "skill": "place_section",
                                                 "args": {"schematic": args["schematic"], "origin": args["origin"]}},
                                 True)
        if call.name == "speak":
            return HarnessAction("speak", {"name": bot, **args}, True)
        if call.name == "ask":
            return HarnessAction("ask", {"name": bot, "npc": args["npc"],
                                         **({"about": args["about"]} if args.get("about") else {})}, True)
        if call.name == "open_container":
            return HarnessAction("open_container", {"name": bot, "pos": [args["x"], args["y"], args["z"]]}, False)
        if call.name == "read_book":
            return HarnessAction("read_item_doc", {"name": bot, "slot": args["slot"]}, False)
        if call.name == "write_book":
            return HarnessAction("write_book", {"name": bot, "slot": args["slot"], "title": args["title"],
                                                "pages": args["pages"]}, False)
        if call.name == "write_sign":
            lines = (list(args["lines"]) + ["", "", "", ""])[:4]
            return HarnessAction("write_sign", {"name": bot, "pos": [args["x"], args["y"], args["z"]], "front": lines},
                                 False)
        raise ValueError(f"Untranslated tool: {call.name}")

    @staticmethod
    def _skill_args(skill: str, args: dict) -> dict:
        """Reshape tool arguments into the harness skill argument format (walking targets the block's centre)."""
        if skill == "move_to":
            return {"x": args["x"] + 0.5, "y": args["y"], "z": args["z"] + 0.5}
        if skill in {"chest_put", "chest_take", "buy"}:
            return {"pos": [args["x"], args["y"], args["z"]], "item": args["item"], "count": args["count"]}
        if skill == "craft":
            return {"item": args["item"], "count": args.get("count", 1)}
        if skill == "drop":  # only the item and count: unlimited lifetime is for staged evidence, not agents
            return {"item": args["item"], "count": args["count"]}
        if skill == "hold":
            return {"item": args["item"]}
        if skill == "eat":
            if not isinstance(args["item"], str):
                raise TypeError("item must be a string")
            return {"item": args["item"]}
        if skill == "bake":
            count = args["count"]
            if type(count) is not int:
                raise TypeError("count must be an integer")
            if not 1 <= count <= 32:
                raise ValueError("count must be between 1 and 32")
            return {"count": count}
        return dict(args)
