"""Regression tests for fixed incidents (plan §2), one class per fix, named after the commit that made it.

Each test was run once with its fix reverted (the commit's relevant hunk undone on a scratch branch, then discarded)
and failed; its docstring records that as "fails at <commit>^".
"""

from pathlib import Path
import re
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import Decision, DecisionLog, prompt_digest
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import GameMaster, NarratedRuntime, ShadowEngine, narration_error
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS, SimulationRuntime
from orchestrator.tools import TOOL_DEFINITIONS, ToolTranslator
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld

BRIEF = RoleBrief("ann", "shopkeeper", "Runs the shop.", "Keep the shop open.")
OBSERVATION = {"gameTime": 2000, "self": {"name": "ann", "pos": [0.5, -60.0, 0.5], "health": 20.0, "food": 20,
                                          "inventory": [], "skill": {"state": "idle"}},
               "entities": [], "blocks": [], "documents": [], "heard": [], "changes": []}


class Fix9f4f3f5SortedKeyPromptTest(unittest.TestCase):
    """A replay from the file log rebuilds the original prompts (incident D8)."""

    def test_remembered_action_renders_the_same_after_a_log_round_trip(self):
        """An action remembered with the model's argument order and the same action reloaded from `decisions.jsonl`
        (written with sorted keys) give the same prompt and digest. Fails at 9f4f3f5^."""
        args = {"z": 3, "x": 1, "y": -60}  # the order a model happened to write
        with tempfile.TemporaryDirectory() as folder:
            log = DecisionLog(Path(folder) / "decisions.jsonl")
            log.append(Decision(1960, "ann", "d", False, "model", "move_to", args, "m", 0, 0, "prompt"))
            reloaded = DecisionLog(log.path).get("ann", 1960).tool_call
        prompts = []
        for call in (ToolCall("move_to", args), reloaded):
            agent = Agent(BRIEF)
            agent.memory.record_action(1960, call, "started")
            prompts.append(agent.user_prompt(OBSERVATION, interrupted=False))
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(*(prompt_digest("m", "sys", prompt, TOOL_DEFINITIONS) for prompt in prompts))


class Fix6efe019EpochResetTest(unittest.TestCase):
    """An agent stops copying its own failing turn: three identical failures start a new epoch (incident R4)."""

    CALL = ToolCall("speak", {"text": "word " * 60})  # the 283-character speech of the rehearsal, give or take
    FAILED = "failed: Speech needs nonempty text of at most 256 characters"

    def fail(self, agent: Agent, times: int) -> None:
        """Have the agent choose the same failing speech `times` times, recorded as the runtime records it."""
        for tick in range(0, 40 * times, 40):
            user = agent.next_prompt(OBSERVATION, interrupted=False)
            agent.memory.record_action(tick, self.CALL, self.FAILED)
            agent.finish_turn(user, self.CALL, self.FAILED)

    def test_three_identical_failures_drop_the_conversation(self):
        """After the third identical failure the next prompt is a full prompt with no earlier turns, so the model no
        longer sees its repeated turns. Fails at 6efe019^."""
        agent = Agent(BRIEF, epoch_turns=8)
        self.fail(agent, 3)
        prompt = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertEqual(agent.history(), ())
        self.assertTrue(prompt.startswith("Your notes:"))

    def test_two_identical_failures_keep_the_conversation(self):
        """Two identical failures are not yet a loop: the epoch goes on."""
        agent = Agent(BRIEF, epoch_turns=8)
        self.fail(agent, 2)
        prompt = agent.next_prompt(OBSERVATION, interrupted=False)
        self.assertEqual(len(agent.history()), 2)
        self.assertTrue(prompt.startswith("Since your last action:"))


class Fixbba8b2dItemNamesTest(unittest.TestCase):
    """Item names reach the engine as registry ids, however an agent writes them."""

    def test_item_names_are_normalized_for_every_tool_with_an_item(self):
        """'Oak Planks', 'oak_planks' and 'minecraft:oak_planks' all become `minecraft:oak_planks` in the harness
        command of every tool that names an item. Fails at bba8b2d^."""
        translator = ToolTranslator()
        position = {"x": 1, "y": -60, "z": 2}
        calls = {"chest_take": {**position, "count": 1}, "chest_put": {**position, "count": 1},
                 "buy": {**position, "count": 1}, "place": position, "craft": {"count": 4},
                 "give": {"target": "bo", "count": 1}}
        for tool, args in calls.items():
            for written in ("Oak Planks", "oak_planks", "minecraft:oak_planks"):
                with self.subTest(tool=tool, written=written):
                    action = translator.translate("ann", ToolCall(tool, {**args, "item": written}))
                    self.assertEqual(action.args["args"]["item"], "minecraft:oak_planks")

    def test_a_book_copied_from_the_inventory_is_a_written_book(self):
        """A book named as the inventory shows it ('written_book "Report" by mayor') is `minecraft:written_book`
        (extended in 67a630e)."""
        action = ToolTranslator().translate("ann", ToolCall("chest_put", {
            "x": 1, "y": -60, "z": 2, "item": 'written_book "Report" by mayor', "count": 1}))
        self.assertEqual(action.args["args"]["item"], "minecraft:written_book")


class AlternatingPolicy:
    """Opens a container (an instant act), then walks (a 50-tick skill), and so on: decided from memory alone, so the
    same policy runs unchanged under the engine and under a narrator."""

    CHEST = {"x": 1, "y": -60, "z": 1}

    def decide(self, system: str, user: str) -> ToolCall:
        """Walk if the last remembered action opened the container, else open it."""
        actions = re.search(r"Your recent actions:\n(.*?)(?:\n\n|$)", user, re.S).group(1).splitlines()
        if "open_container" in actions[-1]:
            return ToolCall("move_to", {"x": 2, "y": -60, "z": 2})
        return ToolCall("open_container", dict(self.CHEST))

    @staticmethod
    def rule(system: str, user: str) -> ToolCall:
        """An engine-faithful narrator: every act happens, opening is instant and walking takes 50 ticks."""
        tool = user.split(" attempts: ", 1)[1].split(" ", 1)[0]
        return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 50 if tool == "move_to" else 0,
                                    "effects": [], "actor_view": "a room", "observations": []})


class Fixd2f1d30NarratedCadenceTest(unittest.TestCase):
    """Narrated agents decide on the engine runtime's cadence (incidents N1 and R8)."""

    START, END = 15_600, 24_200  # an evening, the night rest, and the next morning

    def engine_ticks(self, folder: Path) -> list[int]:
        """Decision ticks of one agent under the engine runtime, on a scripted world."""
        world = FakeWorld({"ann": [0.5, -60, 0.5]}, game_time=self.START, skill_ticks={"move_to": 50},
                          containers={(1, -60, 1): [{"slot": 0, "item": "minecraft:bread", "count": 1}]})
        runtime = SimulationRuntime(world, [Agent(BRIEF)], ScriptedClient(AlternatingPolicy().decide),
                                    folder / "engine", command_error=CommandError, min_gap=THINKING_TICKS,
                                    rest=NIGHT_REST)
        runtime.run(self.END)
        return [decision.game_time for decision in runtime.log.decisions()]

    def narrated_ticks(self, folder: Path) -> list[int]:
        """Decision ticks of the same agent and policy under the narrated runtime with an engine-faithful narrator."""
        gm = GameMaster(ScriptedClient(AlternatingPolicy.rule), "a room with a chest", {"ann": "a room"})
        runtime = NarratedRuntime([Agent(BRIEF)], ScriptedClient(AlternatingPolicy().decide), gm, folder / "narrated",
                                  min_gap=THINKING_TICKS, rest=NIGHT_REST)
        runtime.run(self.START, self.END)
        return [decision.game_time for decision in runtime.log.decisions()]

    def test_narrated_and_engine_runtimes_decide_at_the_same_ticks(self):
        """With one policy, the narrated runtime decides exactly when the engine runtime does: thinking time after
        instant acts, none after skills, and no decisions in the night rest. Fails at d2f1d30^."""
        with tempfile.TemporaryDirectory() as folder:
            engine, narrated = self.engine_ticks(Path(folder)), self.narrated_ticks(Path(folder))
        self.assertEqual(narrated, engine)
        self.assertIn(15_720, engine)  # 120 ticks after the instant act at 15,600
        self.assertFalse([tick for tick in engine if 16_000 <= tick < 24_000])  # the night rest
        self.assertEqual(min(tick for tick in engine if tick >= 16_000), 24_000)


class Fix648a83dNarrationRulesTest(unittest.TestCase):
    """The game master knows the engine's rules, and its rejections are scored against the engine (incident N2)."""

    @staticmethod
    def rejecting(system: str, user: str) -> ToolCall:
        """A narrator that rejects every act."""
        return ToolCall("resolve", {"accepted": False, "reason": "too far", "duration_ticks": 0, "effects": [],
                                    "actor_view": "a room", "observations": []})

    def test_gm_is_told_that_people_cannot_be_asked(self):
        """The system prompt the GM's model receives lists the agents' acts and the engine's rule that `ask` is for
        villagers only (7 of the first pilot's errors). Fails at 648a83d^."""
        systems = []

        def record(system: str, user: str) -> ToolCall:
            """Keep the system prompt and reject."""
            systems.append(system)
            return self.rejecting(system, user)

        GameMaster(ScriptedClient(record), "a village", {"ann": "a room"}).resolve(0, "ann", ToolCall("ask", {
            "npc": "bo"}))
        self.assertIn("- ask: Ask a villager", systems[0])
        self.assertIn("cannot be asked", systems[0])

    def test_rejection_of_a_feasible_act_is_a_false_rejection(self):
        """Every ruling gets the engine's dry-run verdict, so a GM rejection of an act the engine finds feasible is
        counted as a false rejection. Fails at 648a83d^."""
        world = FakeWorld({"ann": [0.5, -60, 0.5]})
        gm = GameMaster(ScriptedClient(self.rejecting), "a room", {"ann": "a room"})
        walk = ScriptedClient(lambda system, user: ToolCall("move_to", {"x": 2, "y": -60, "z": 2}))
        with tempfile.TemporaryDirectory() as folder:
            runtime = NarratedRuntime([Agent(BRIEF)], walk, gm, Path(folder), shadow=ShadowEngine(world, CommandError))
            runtime.run(0, 40)
        summary = narration_error(runtime.acts)
        self.assertEqual((summary["rejected"], summary["false_rejections"], summary["false_rejections_by_tool"]),
                         (1, 1, {"move_to": 1}))


if __name__ == "__main__":
    unittest.main()
