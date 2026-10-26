"""A small three-agent workshop scenario with a scripted, prompt-driven policy, for $0 runtime gates (E3, L1).

The policy stands in for an LLM. It's a pure function of the prompt text, so decisions depend only on what the agent
perceives and remembers, as an LLM's do. It tracks its plan through the agent's own notes (`step N`), and uses many
tools: walking, mining, collecting, crafting, placing, containers, signs, books, giving and speech.

Layout (flat stone floor, y = -61):
- Ana works alone in the west (x around -26), out of sight and earshot of the others.
- Ben, in the east, takes bread from a chest, reads the market sign, repeats it aloud, writes it in a book, and gives
  Cy some bread.
- Cy waits near the market sign and answers anything it hears with a reply that quotes it.

Changing the market sign changes what Ben says, and so what Cy hears and replies. Ana is unaffected: the intervention's
causal cone is {Ben, Cy}.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.agent import Agent, RoleBrief  # noqa: E402
from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402

REGION = {"x0": -32, "z0": -32, "x1": 31, "z1": 31, "y0": -64, "y1": -50}
SIGN = (20, -60, 14)
CHEST = (18, -60, 10)
SIGN_TEXT = ("Market opens", "at noon", "", "")
NOTABLE_EXTRA = ["minecraft:oak_log", "minecraft:oak_planks"]


def _vanilla_sign(lines: tuple[str, ...]) -> str:
    """Return the SNBT front-text payload for a sign's four lines."""
    return "{front_text:{messages:[" + ",".join(f'"{line}"' for line in lines) + "]}}"


class WorkshopScenario:
    """Builds the workshop world, its three agents, and the scripted policy that drives them."""

    BOTS = {"ana": (-24.5, -60, -24.5), "ben": (16.5, -60, 16.5), "cy": (24.5, -60, 6.5)}

    def setup(self, client) -> None:
        """Clear the arena, place the fixtures, spawn and supply the bots, then turn recording on.

        Call it with recording off (`enable_lockstep(client, recording=False)`), so agents don't perceive the setup
        itself as block changes.
        """
        command = lambda text: client.request("command", {"command": text})  # noqa: E731
        # A vanilla fill that changes nothing reports failure, so seed one barrier block before each fill.
        command("setblock -32 -60 -32 minecraft:barrier")
        command("fill -32 -60 -32 31 -55 31 minecraft:air")
        command("setblock -32 -61 -32 minecraft:barrier")
        command("fill -32 -61 -32 31 -61 31 minecraft:stone")
        notable = client.request("config_get", {})["notable_blocks"]
        client.request("config_set", {"notable_blocks": sorted(set(notable) | set(NOTABLE_EXTRA))})
        for x, z in ((-26, -20), (-26, -18)):
            command(f"setblock {x} -60 {z} minecraft:oak_log")
        command("setblock -28 -60 -28 minecraft:oak_sign")
        command(f"setblock {SIGN[0]} {SIGN[1]} {SIGN[2]} minecraft:oak_sign{_vanilla_sign(SIGN_TEXT)}")
        command(f"setblock {CHEST[0]} {CHEST[1]} {CHEST[2]} minecraft:chest")
        command(f"item replace block {CHEST[0]} {CHEST[1]} {CHEST[2]} container.0 with minecraft:bread 4")
        client.request("acl_set", {"pos": list(CHEST), "allowed": "*"})
        for name, (x, y, z) in self.BOTS.items():
            client.request("spawn_bot", {"name": name, "x": x, "y": y, "z": z, "yaw": 0.0, "pitch": 0.0})
        client.request("give", {"name": "ben", "item": "minecraft:writable_book", "count": 1})
        client.request("recording", {"enabled": True})

    @staticmethod
    def change_sign(client, lines: tuple[str, ...]) -> None:
        """Intervention: rewrite the market sign's text with a world command."""
        client.request("command", {"command": f"data merge block {SIGN[0]} {SIGN[1]} {SIGN[2]} {_vanilla_sign(lines)}"})

    def agents(self) -> list[Agent]:
        """Return the three agents with short role briefs."""
        return [Agent(RoleBrief(name, role, f"{name} is the town's {role}.", "Follow your routine."))
                for name, role in (("ana", "carpenter"), ("ben", "shopkeeper"), ("cy", "neighbour"))]

    def client(self, planted_errors: bool = False) -> ScriptedClient:
        """Return a scripted model client running this scenario's policy (optionally with planted infeasible acts)."""
        return ScriptedClient(ScriptedPolicy(planted_errors).decide)

    @staticmethod
    def world_description() -> str:
        """The text a narrated game master gets in place of the engine world (the same facts the world encodes)."""
        return ("A flat stone square from (-32, -61, -32) to (31, -61, 31); people stand at y = -60. "
                "Two oak logs stand at (-26, -60, -20) and (-26, -60, -18). A blank oak sign is at (-28, -60, -28). "
                f"The market sign at {SIGN} reads \"{' / '.join(line for line in SIGN_TEXT if line)}\". "
                f"A chest at {CHEST} holds 4 bread; anyone may open it. "
                "ana (carpenter) starts at (-24.5, -60, -24.5) with nothing. "
                "ben (shopkeeper) starts at (16.5, -60, 16.5) "
                "with a writable book. cy (neighbour) starts at (24.5, -60, 6.5) with nothing. Reach is 4.5 blocks; "
                "talking carries 8 blocks and shouting 24; people see 24 blocks. "
                "Standard Minecraft recipes apply (a log makes 4 planks).")


class ScriptedPolicy:
    """A deterministic, prompt-driven policy: reads the prompt text and returns the next tool call.

    With `planted_errors`, Ana and Ben each end their plan with one act the engine must reject (a narration-error
    plumbing check): Ana mines a block far out of reach, and Ben crafts an iron pickaxe without the materials.
    """

    PLANTED = {"ana": ToolCall("mine", {"x": 6, "y": -61, "z": -20}),
               "ben": ToolCall("craft", {"item": "minecraft:iron_pickaxe", "count": 1})}

    def __init__(self, planted_errors: bool = False):
        """Choose whether the plans end with the planted infeasible acts."""
        self._planted = planted_errors

    def decide(self, system: str, user: str) -> ToolCall:
        """Choose the agent's next action from its prompt alone."""
        name = re.search(r"You are (\w+) at", user).group(1)
        if user.rstrip().endswith("(you may continue what you were doing)."):
            return self._reply(user) if name == "cy" else ToolCall("keep_going", {})
        steps = re.findall(r"\] step (\d+)", self._section(user, "Your notes"))
        step = int(steps[-1]) if steps else 0
        plan = getattr(self, f"_{name}")(user)
        if self._planted and name in self.PLANTED:
            plan.append(self.PLANTED[name])
        if step >= len(plan):
            return ToolCall("wait", {"ticks": 400})
        call = plan[step]
        return ToolCall(call.name, {**call.args, "note": f"step {step + 1}"})

    @staticmethod
    def _section(user: str, title: str) -> str:
        """Return the body of one prompt section."""
        match = re.search(rf"{title}:\n(.*?)(?:\n\n|$)", user, re.S)
        return match.group(1) if match else ""

    def _slot(self, user: str, item: str) -> int:
        """Return the first inventory slot holding `item`, or 0 if none."""
        match = re.search(rf"slot (\d+): \d+ {item}\b", user)
        return int(match.group(1)) if match else 0

    def _sign(self, user: str) -> str:
        """Return the text of the market sign if it's readable now, else '?'."""
        match = re.search(rf"sign at \({SIGN[0]}, {SIGN[1]}, {SIGN[2]}\): \"(.*?)\"", user)
        return match.group(1) if match else "?"

    def _ana(self, user: str) -> list[ToolCall]:
        """Ana: fell a log, make planks, place one, label her workshop, and announce it."""
        return [ToolCall("move_to", {"x": -24, "y": -60, "z": -20}),
                ToolCall("mine", {"x": -26, "y": -60, "z": -20}),
                ToolCall("collect", {}),
                ToolCall("craft", {"item": "minecraft:oak_planks", "count": 4}),
                ToolCall("place", {"x": -24, "y": -60, "z": -23, "item": "minecraft:oak_planks"}),
                ToolCall("move_to", {"x": -27, "y": -60, "z": -26}),
                ToolCall("write_sign", {"x": -28, "y": -60, "z": -28, "lines": ["Ana's", "workshop"]}),
                ToolCall("speak", {"text": "My workshop sign is up."})]

    def _ben(self, user: str) -> list[ToolCall]:
        """Ben: take bread, read the sign aloud, write it down, read it back, and give Cy bread."""
        sign = self._sign(user)
        return [ToolCall("move_to", {"x": 18, "y": -60, "z": 12}),
                ToolCall("open_container", {"x": CHEST[0], "y": CHEST[1], "z": CHEST[2]}),
                ToolCall("chest_take", {"x": CHEST[0], "y": CHEST[1], "z": CHEST[2], "item": "minecraft:bread",
                                        "count": 2}),
                ToolCall("move_to", {"x": 20, "y": -60, "z": 12}),
                ToolCall("speak", {"text": f"The market sign says: {sign}"}),
                ToolCall("write_book", {"slot": self._slot(user, "writable_book"), "title": "Market",
                                        "pages": [sign]}),
                ToolCall("read_book", {"slot": self._slot(user, "written_book")}),
                ToolCall("move_to", {"x": 22, "y": -60, "z": 10}),
                ToolCall("give", {"target": "cy", "item": "minecraft:bread", "count": 1})]

    def _cy(self, user: str) -> list[ToolCall]:
        """Cy: step aside and wait (waits are interrupted when someone speaks)."""
        return [ToolCall("move_to", {"x": 23, "y": -60, "z": 9})]

    def _reply(self, user: str) -> ToolCall:
        """Cy's answer to the latest thing someone else said."""
        heard = [match.group(1) for line in self._section(user, "What you heard").splitlines()
                 if "] cy" not in line and (match := re.search(r": \"(.*)\"$", line))]
        quoted = heard[-1] if heard else ""
        return ToolCall("speak", {"text": f"Cy heard: {' '.join(quoted.split()[:6])}"})
