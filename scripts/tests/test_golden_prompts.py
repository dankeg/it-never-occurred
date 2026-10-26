"""Byte-stable village prompts; run this file with --update to rebaseline intentionally."""

import copy
from itertools import zip_longest
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.llm import ToolCall, arguments_json
from orchestrator.runtime import failure_hint
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS


class GoldenPromptsTest(unittest.TestCase):
    """Pin all six roles' system, first and fifth prompts, conversation history and tools."""

    def snapshots(self):
        """Build the stored bytes from fixed observations and four ordinary runtime-style turns."""
        root = Path(__file__).resolve().parents[2]
        golden = Path(__file__).with_name("golden")
        manifest = ScenarioManifest.load(root / "scenario" / "village_sabotage.json")
        fixture = json.loads((golden / "observation.json").read_text(encoding="utf-8"))
        turns = json.loads((golden / "turns.json").read_text(encoding="utf-8"))
        spawns = {role["name"]: role["spawn"] for role in manifest.roles}
        snapshots = {"tools.json": (json.dumps(TOOL_DEFINITIONS, indent=1, ensure_ascii=False) + "\n").encode("utf-8")}
        for agent in manifest.agents(epoch_turns=8):
            observation = copy.deepcopy(fixture)
            observation["self"].update(name=agent.name, pos=list(spawns[agent.name]))
            agent.memory.absorb(observation, agent.name)
            user = agent.next_prompt(observation, interrupted=False)
            snapshots[f"{agent.name}/system.txt"] = agent.system_prompt().encode("utf-8")
            snapshots[f"{agent.name}/turn1_user.txt"] = user.encode("utf-8")
            for index, turn in enumerate(turns):
                call = ToolCall(turn["action"]["tool"], copy.deepcopy(turn["action"]["args"]))
                outcome = turn["outcome"] + failure_hint(call, turn["outcome"], observation)
                now = observation["gameTime"]
                agent.memory.record_action(now, call, outcome, skill_id=turn.get("skillId"))
                agent.finish_turn(user, call, outcome)
                if "read" in turn:
                    agent.memory.record_read(now, turn["read"])
                following = turn["before_next"]
                observation["gameTime"] = following["gameTime"]
                observation["heard"] = copy.deepcopy(following.get("heard", []))
                observation["changes"] = copy.deepcopy(following.get("changes", []))
                for key in ("skill", "inventory"):
                    if key in following:
                        observation["self"][key] = copy.deepcopy(following[key])
                status = observation["self"]["skill"]
                if status["state"] != "running":
                    agent.memory.resolve_skill(status)
                agent.memory.absorb(observation, agent.name)
                user = agent.next_prompt(observation, interrupted=index == 3)
            snapshots[f"{agent.name}/turn5_user.txt"] = user.encode("utf-8")
            history = [[turn.user, turn.call.name if turn.call else None,
                        arguments_json(turn.call) if turn.call else None, turn.result] for turn in agent.history()]
            snapshots[f"{agent.name}/turn5_history.json"] = (json.dumps(history, indent=1, ensure_ascii=False)
                                                           + "\n").encode("utf-8")
        return snapshots

    def assert_golden_bytes(self, relative, actual):
        """Report the file and first differing line without replacing byte comparison with a digest."""
        path = Path(__file__).with_name("golden") / relative
        self.assertTrue(path.is_file(), f"{path}: first differing line 1 (stored file is missing)")
        expected = path.read_bytes()
        if actual != expected:
            for line, (stored, current) in enumerate(zip_longest(expected.splitlines(keepends=True),
                                                                actual.splitlines(keepends=True)), start=1):
                if stored != current:
                    self.fail(f"{path}: first differing line {line}\nstored: {stored!r}\ncurrent: {current!r}")

    def test_six_roles_turn_one_and_five_prompts_history_and_tools_are_byte_stable(self):
        """All six roles' system, first and interrupted fifth prompts, history, and tools match stored bytes."""
        for relative, actual in self.snapshots().items():
            with self.subTest(file=relative):
                self.assert_golden_bytes(relative, actual)

    def update(self):
        """Rewrite changed stored outputs from the unchanged production renderer and print their paths."""
        golden = Path(__file__).with_name("golden")
        for relative, actual in self.snapshots().items():
            path = golden / relative
            if not path.exists() or path.read_bytes() != actual:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(actual)
                print(f"Updated {path.relative_to(Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        GoldenPromptsTest().update()
    else:
        unittest.main()
