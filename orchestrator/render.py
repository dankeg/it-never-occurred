"""Render a harness observation into the compact text an agent model reads.

The rendered text is the agent's whole view of the world at a decision point. It's built only from the `observe`
response (already filtered by sight, reach and hearing) plus the agent's own memory. That keeps the observation
function the single source of what an agent can know.

Rendering is deterministic: sections and items come in a fixed order, and positions and distances are rounded to whole
blocks. Rounding makes the prompt insensitive to sub-block motion noise. Since the decision cache is keyed by the
rendered prompt, that's also what decides when a counterfactual branch can reuse a factual decision.
"""

from __future__ import annotations

TICKS_PER_DAY = 24_000


def block_text(position) -> str:
    """Format a position as whole-block coordinates `(x, y, z)`."""
    return "(" + ", ".join(str(int(round(value)) if isinstance(value, float) else int(value))
                          for value in position) + ")"


def short_id(identifier: str) -> str:
    """Drop the `minecraft:` namespace from a registry id."""
    return identifier.split(":", 1)[1] if identifier.startswith("minecraft:") else identifier


def stack_text(stack: dict) -> str:
    """Describe an observed stack, using engine-provided bread freshness only when those fields are present."""
    item, count = short_id(stack["item"]), stack["count"]
    if item == "bread" and ("baked_day" in stack or "mouldy" in stack):
        if stack.get("mouldy"):
            return f"mouldy bread ×{count}"
        baked = f" (baked day {stack['baked_day']})" if "baked_day" in stack else ""
        return f"bread ×{count}{baked}"
    return f"{count} {item}"


def door_state(block: dict) -> str | None:
    """Return a door, trapdoor or fence gate's observed open/closed state, or None when it is unknown."""
    if not short_id(block.get("block", "")).endswith(("_door", "_trapdoor", "_fence_gate")):
        return None
    opened = block.get("properties", {}).get("open")
    if opened is True or opened == "true":
        return "open"
    if opened is False or opened == "false":
        return "closed"
    return None


def clock(game_time: int) -> str:
    """Format a game time as `day D, HH:MM` (day 0 starts at 06:00, as in Minecraft)."""
    day, tick = divmod(game_time, TICKS_PER_DAY)
    minutes = (tick * 1440 // TICKS_PER_DAY + 360) % 1440
    return f"day {day}, {minutes // 60:02d}:{minutes % 60:02d}"


class ObservationRenderer:
    """Turns one `observe` response into prompt text, with bounded section sizes."""

    def __init__(self, max_entities: int = 20, max_blocks: int = 30, max_documents: int = 8, max_per_block: int = 6):
        """Bound the number of items shown per section, nearest first, and the positions listed per block type."""
        self._max_entities = max_entities
        self._max_blocks = max_blocks
        self._max_documents = max_documents
        self._max_per_block = max_per_block

    def render(self, observation: dict) -> str:
        """Return the observation as text sections: time, self, people, notable blocks, documents."""
        sections = [f"Time: {clock(observation['gameTime'])} (tick {observation['gameTime']})",
                    self._self(observation["self"])]
        for title, lines in (("People and things you can see", self._entities(observation.get("entities", []))),
                             ("Notable blocks you can see", self._blocks(observation.get("blocks", []))),
                             ("Writing you can read", self._documents(observation.get("documents", [])))):
            sections.append(f"{title}:\n" + ("\n".join(lines) if lines else "  (none)"))
        structures = observation.get("structures") or []
        if structures:  # omitted when the scenario defines none, so other prompts are unchanged
            sections.insert(3, "Structures you can see:\n" + "\n".join(self._structure(s) for s in structures))
        return "\n\n".join(sections)

    @staticmethod
    def _structure(structure: dict) -> str:
        """One known structure in sight and whether it is complete."""
        built, total = structure["built"], structure["total"]
        state = (f"complete ({total} of {total} blocks)" if built == total
                 else f"unfinished ({built} of {total} blocks in place)")
        return f"  {structure['name']} at {block_text(structure['pos'])}: {state}"

    @staticmethod
    def _self(self_view: dict) -> str:
        """Describe the agent's own position, health, inventory and current activity."""
        lines = [f"You are {self_view['name']} at {block_text(self_view['pos'])}. Health {self_view['health']:.0f}/20, "
                 f"food {self_view['food']}/20."]
        if self_view.get("in_water"):
            lines.append("You are in water.")
        if self_view.get("hungry"):
            lines.append("You are hungry.")
        inventory = self_view.get("inventory", [])
        if inventory:
            lines.append("Inventory:")
            for item in inventory:
                entry = f"  slot {item['slot']}: {stack_text(item)}"
                document = item.get("doc")
                if document:
                    entry += f" \"{document.get('title', '')}\" by {document.get('author', '?')}"
                lines.append(entry)
        else:
            lines.append("Inventory: empty")
        skill = self_view.get("skill") or {}
        state = skill.get("state", "idle")
        if state == "running":
            lines.append(f"Current activity: {skill.get('skill')} (started tick {skill.get('startedAt')}).")
        elif state != "idle":
            reason = skill.get("reason")
            lines.append(f"Last activity: {skill.get('skill')} {state}" + (f" ({reason})." if reason else "."))
        return "\n".join(lines)

    def _entities(self, entities: list[dict]) -> list[str]:
        """List visible entities, held items and public activity, nearest first."""
        return [f"  {self.entity_line(entity)}" for entity in
                sorted(entities, key=lambda item: (item["distance"], item.get("uuid", "")))[:self._max_entities]]

    def entity_line(self, entity: dict) -> str:
        """Describe a visible entity, its held item, public load band and activity, never private item counts."""
        kind = short_id(entity.get("type", "?"))
        label = entity.get("name") or kind
        if kind == "item":
            label = "a dropped item"
        elif kind == "villager" and entity.get("npcId"):
            label = f"{entity.get('name', 'a villager')} (villager, id {entity['npcId'].removeprefix('npc:')})"
        elif kind not in {"player", "villager"} and entity.get("name"):
            label = f"{entity['name']} ({kind})"
        line = f"{label} at {block_text(entity['pos'])}, {entity['distance']:.0f} blocks away"
        if entity.get("heldItem"):
            line += f", holding {short_id(entity['heldItem'])}"
        load = entity.get("load") or {}
        if load.get("count_band") == "sack" and load.get("item"):
            line += f", carrying a sack of {short_id(load['item']).replace('_', ' ')}"
        activity = self._activity(entity.get("activity"))
        return f"{line}, {activity}" if activity else line

    @staticmethod
    def _activity(activity: dict | None) -> str:
        """Describe only the public skill name and optional visible position, never its arguments."""
        if not activity or not activity.get("skill"):
            return ""
        skill = activity["skill"]
        untargeted, targeted = {
            "move_to": ("walking", "walking towards"),
            "mine": ("mining", "mining the block at"),
            "collect": ("collecting dropped items", "collecting dropped items at"),
            "craft": ("crafting", "crafting at"),
            "bake": ("baking at the oven", "baking at the oven at"),
            "place": ("placing a block", "placing a block at"),
            "build_section": ("building", "building at"),
            "place_section": ("building", "building at"),
            "chest_take": ("taking items from a chest", "taking items from the chest at"),
            "chest_put": ("putting items into a chest", "putting items into the chest at"),
            "buy": ("buying items", "buying items at"),
            "give": ("handing over items", "handing over items at"),
            "drop": ("dropping items", "dropping items at"),
            "eat": ("eating", "eating at"),
            "use": ("using a block", "using the block at"),
            "hold": ("holding an item", "holding an item at"),
            "wait": ("waiting", "waiting at"),
            "ferry": ("riding a ferry", "riding a ferry towards"),
            "speak": ("speaking", "speaking at"),
            "ask": ("asking a villager", "asking a villager at"),
        }.get(skill, (f"doing {skill}", f"doing {skill} at"))
        position = activity.get("pos")
        return f"{targeted} {block_text(position)}" if position is not None else untargeted

    def _blocks(self, blocks: list[dict]) -> list[str]:
        """List visible notable blocks, grouped by block type, nearest first; a type with many blocks (e.g. a gravel
        floor) lists its nearest few and counts the rest. Door states belong to individual positions."""
        groups: dict[str, list[dict]] = {}
        for block in sorted(blocks, key=lambda item: (item["distance"], item["pos"]))[:self._max_blocks]:
            groups.setdefault(short_id(block["block"]), []).append(block)
        lines = []
        for name, group in groups.items():
            positions = []
            for block in group[:self._max_per_block]:
                state = door_state(block)
                positions.append(block_text(block["pos"]) + (f" ({state})" if state else ""))
            shown = ", ".join(positions)
            more = len(group) - self._max_per_block
            lines.append(f"  {name}: {shown}" + (f", and {more} more nearby" if more > 0 else ""))
        return lines

    def _documents(self, documents: list[dict]) -> list[str]:
        """Quote readable signs and lecterns within reach."""
        lines = []
        for document in sorted(documents, key=lambda item: (item["distance"], item["pos"]))[:self._max_documents]:
            if document.get("kind") == "sign":
                text = document.get("text", {})
                words = " / ".join(line for line in text.get("front", []) + text.get("back", []) if line)
                lines.append(f"  sign at {block_text(document['pos'])}: \"{words}\"")
            else:
                pages = " | ".join(document.get("pages", []))
                lines.append(f"  book \"{document.get('title', '')}\" on a lectern at "
                             f"{block_text(document['pos'])}: \"{pages}\"")
        return lines
