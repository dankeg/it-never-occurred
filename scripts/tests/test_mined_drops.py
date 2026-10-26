"""Pin mined-drop feedback without simulating Minecraft's sight, item motion or pickup physics."""

import copy
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.runtime import SimulationRuntime, failure_hint
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld


class MinedDropFeedbackTests(unittest.TestCase):
    """Only an owning, resolved mine with an unpicked stack adds a location to the next prompt."""

    def status(self, drops=None):
        """Return a terminal mine receipt with one known, unpicked item entity by default."""
        return {"skillId": "builder:1", "skill": "mine", "state": "succeeded", "endedAt": 40,
                "progress": {"drops": [{"uuid": "drop-1", "item": "minecraft:oak_log", "count": 1,
                                        "pos": [-108.75, -60.0, 8.25]}] if drops is None else drops}}

    def agent(self, epoch_turns=1):
        """Start one ordinary agent and record a mine that is awaiting its terminal receipt."""
        agent = Agent(RoleBrief("builder", "builder", "Builds.", "Get logs."), epoch_turns=epoch_turns)
        call = ToolCall("mine", {"x": -109, "y": -60, "z": 8})
        agent.memory.record_action(0, call, "started", skill_id="builder:1")
        agent.finish_turn("previous prompt", call, "started")
        return agent

    def test_full_and_incremental_next_prompts_locate_the_drop_once(self):
        """Both epoch shapes carry one drop line, with no invented pickup or duplicate status announcements."""
        for epoch_turns in (1, 8):
            with self.subTest(epoch_turns=epoch_turns):
                agent = self.agent(epoch_turns)
                agent.memory.resolve_skill(self.status())
                agent.memory.resolve_skill(self.status())
                prompt = agent.next_prompt_for_view("current view", False)
                self.assertEqual(prompt.count("unpicked drop: 1 oak_log at (-109, -60, 8)"), 1)
                self.assertNotIn("your inventory:", prompt)
                restored = Agent.from_json(agent.to_json())
                self.assertIn("unpicked drop:", restored.memory.render())

    def test_empty_or_legacy_progress_leaves_prompt_bytes_unchanged(self):
        """Successful automatic pickups and older servers do not add sections or otherwise change prompts."""
        for epoch_turns in (1, 8):
            with self.subTest(epoch_turns=epoch_turns):
                baseline = self.agent(epoch_turns)
                status = self.status([])
                del status["progress"]
                baseline.memory.resolve_skill(status)
                expected = baseline.next_prompt_for_view("current view", False)
                agent = self.agent(epoch_turns)
                agent.memory.resolve_skill(self.status([]))
                self.assertEqual(agent.next_prompt_for_view("current view", False), expected)

    def test_unowned_failed_and_non_mine_receipts_never_reveal_drops(self):
        """An unrelated bot, failed mine or unrelated skill cannot grant a miner's perception to this agent."""
        cases = [{"skillId": "mayor:1"}, {"state": "failed", "reason": "interrupted"}, {"skill": "collect"}]
        for changed in cases:
            with self.subTest(changed=changed):
                agent = self.agent()
                agent.memory.resolve_skill({**self.status(), **changed})
                self.assertNotIn("unpicked drop:", agent.next_prompt_for_view("current view", False))
        memory = AgentMemory()
        memory.resolve_skill(self.status())
        self.assertEqual(memory.take_fresh(), [])

    def test_runtime_delivers_own_drop_to_the_actual_next_decision(self):
        """The normal status/observe loop, not a hand-built prompt, delivers the feedback only to its miner."""
        def idle(system, user):
            """Take no further action after the terminal mine receipt."""
            return None

        world = FakeWorld({"builder": [-106.5, -60, 8.5], "mayor": [-112.5, -60, 8.5]})
        builder = self.agent(8)
        world.skills["builder"].append(copy.deepcopy(self.status()))
        mayor = Agent(RoleBrief("mayor", "mayor", "Governs.", "Help."))
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [builder, mayor], ScriptedClient(idle), Path(directory),
                                        command_error=CommandError)
            runtime.run(40)
            prompts = {decision.agent: decision.user_prompt for decision in runtime.log.decisions()}
        self.assertIn("unpicked drop: 1 oak_log at (-109, -60, 8)", prompts["builder"])
        self.assertNotIn("unpicked drop:", prompts["mayor"])

    def test_collect_failure_hint_does_not_promise_inventory_pickup(self):
        """Collection distinguishes visible drops and recently remembered reachable mining drops."""
        hint = failure_hint(ToolCall("collect", {}), "not possible: target_not_found", {})
        self.assertIn("in sight within 6 blocks", hint)
        self.assertIn("reachable drops remembered from your recent mining", hint)
        self.assertIn("may still be on the ground", hint)
        self.assertNotIn("straight into your inventory", hint)
        self.assertEqual(failure_hint(ToolCall("collect", {}), "succeeded", {}), "")
