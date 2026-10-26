"""Pin the runtime's ten behaviours with a scripted world and no external services."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import DecisionLog
from orchestrator.llm import ScriptedClient, ToolCall, UsageLimitReached
from orchestrator.runtime import BudgetExceeded, ReplayDiverged, SimulationRuntime, resting
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld


class DecisionGridTest(unittest.TestCase):
    """Decisions stay on the poll grid and invalid timing is rejected."""

    def policy(self, system, user):
        """Choose a skill that ends at the next forty-tick poll."""
        return ToolCall("wait", {"ticks": 40})

    def test_decisions_start_at_next_poll_and_stay_on_grid(self):
        """A run starting at twenty first decides at forty and only on forty-tick polls."""
        world = FakeWorld({"alice": [0, 64, 0]}, game_time=20)
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        ScriptedClient(self.policy), Path(directory), command_error=CommandError)
            runtime.run(400)
            self.assertEqual([decision.game_time for decision in runtime.log.decisions()], list(range(40, 400, 40)))
            self.assertEqual(runtime.stats.decisions, 9)
            self.assertEqual([args for command, args in world.requests if command == "step"],
                             [{"n": 20}] + [{"n": 40}] * 9)

    def test_start_off_twenty_tick_grid_is_rejected(self):
        """A run whose starting tick is not divisible by twenty raises ValueError."""
        world = FakeWorld({"alice": [0, 64, 0]}, game_time=21)
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [], ScriptedClient(self.policy), Path(directory),
                                        command_error=CommandError)
            with self.assertRaisesRegex(ValueError, "20-tick grid, not at 21"):
                runtime.run(80)
            self.assertEqual(world.requests, [("ping", {})])

    def test_poll_interval_must_be_positive_multiple_of_twenty(self):
        """Zero, negative, and non-twenty-multiple poll intervals raise ValueError."""
        with tempfile.TemporaryDirectory() as directory:
            for interval in (0, -20, 1, 30, 41):
                with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "positive multiple of 20"):
                    SimulationRuntime(FakeWorld({}), [], None, Path(directory), command_error=CommandError,
                                      poll_interval=interval)


class ThinkingTimeTest(unittest.TestCase):
    """Only instant or unsuccessful acts impose the configured thinking delay."""

    def run_action(self, call, *, min_gap=100, rejects=None, containers=None):
        """Repeat a fixed prompt-independent action through tick 160 and return its runtime and world."""
        def policy(system, user):
            """Return the same action for every prompt."""
            return call

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        world = FakeWorld({"alice": [0, 64, 0]}, rejects=rejects,
                          containers={(1, 64, 0): []} if containers is None else containers)
        runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                    ScriptedClient(policy), Path(directory.name), command_error=CommandError,
                                    min_gap=min_gap)
        runtime.run(160)
        return runtime, world

    def test_instant_act_waits_thinking_time(self):
        """An instant container read at zero waits until tick 120 with a hundred-tick gap."""
        runtime, world = self.run_action(ToolCall("open_container", {"x": 1, "y": 64, "z": 0}))
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(0, "done"), (120, "done")])
        self.assertEqual([args for command, args in world.requests if command == "open_container"],
                         [{"name": "alice", "pos": [1, 64, 0]}] * 2)

    def test_rejected_skill_waits_thinking_time(self):
        """A rejected skill at zero waits until tick 120 and records its reason and hint."""
        runtime, world = self.run_action(ToolCall("move_to", {"x": 5, "y": 64, "z": 0}),
                                         rejects={"move_to": "no_path"})
        outcome = "not possible: no_path (no walkable route: it may be across water, enclosed, or not yet built)"
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(0, outcome), (120, outcome)])
        self.assertEqual([args for command, args in world.requests if command == "skill_start"],
                         [{"name": "alice", "skill": "move_to", "args": {"x": 5.5, "y": 64, "z": 0.5}}] * 2)

    def test_drop_waits_for_skill_completion_without_an_instant_action_gap(self):
        """The runtime polls drop completion instead of treating drops as instant information actions."""
        runtime, world = self.run_action(ToolCall("drop", {"item": "bread", "count": 1}))
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(0, "started"), (40, "started"), (80, "started"), (120, "started")])
        self.assertEqual([args for command, args in world.requests if command == "skill_start"],
                         [{"name": "alice", "skill": "drop", "args": {"item": "minecraft:bread", "count": 1}}] * 4)

    def test_failed_instant_act_waits_thinking_time(self):
        """A missing container becomes a failed act and imposes the same hundred-tick thinking delay."""
        runtime, world = self.run_action(ToolCall("open_container", {"x": 1, "y": 64, "z": 0}), containers={})
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(0, "failed: no_container"), (120, "failed: no_container")])
        self.assertEqual(len([command for command, args in world.requests if command == "open_container"]), 2)

    def test_finished_skill_does_not_wait_thinking_time(self):
        """A skill ending at forty permits a new decision at forty despite a hundred-tick minimum gap."""
        runtime, world = self.run_action(ToolCall("wait", {"ticks": 40}))
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(tick, "started") for tick in (0, 40, 80, 120)])
        self.assertEqual([skill["startedAt"] for skill in world.skills["alice"]], [0, 40, 80, 120])

    def test_zero_gap_allows_next_poll_after_instant_act(self):
        """With no thinking delay an instant act at zero is followed by a decision at forty."""
        runtime, world = self.run_action(ToolCall("open_container", {"x": 1, "y": 64, "z": 0}), min_gap=0)
        self.assertEqual([(entry.game_time, entry.outcome) for entry in runtime.log.decisions()],
                         [(tick, "done") for tick in (0, 40, 80, 120)])
        self.assertEqual(len([command for command, args in world.requests if command == "open_container"]), 4)


class NightRestTest(unittest.TestCase):
    """Idle agents rest at night, but heard speech still wakes them at a poll."""

    def policy(self, system, user):
        """Read a container immediately so the agent remains idle."""
        return ToolCall("open_container", {"x": 1, "y": 64, "z": 0})

    def test_idle_agent_rests_until_daybreak(self):
        """An idle agent skips all polls in the night window and decides again at tick 24000."""
        world = FakeWorld({"alice": [0, 64, 0]}, game_time=15_960, containers={(1, 64, 0): []})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        ScriptedClient(self.policy), Path(directory), command_error=CommandError,
                                        rest=(16_000, 24_000))
            runtime.run(24_040)
            self.assertEqual([entry.game_time for entry in runtime.log.decisions()], [15_960, 24_000])
            self.assertEqual([entry.outcome for entry in runtime.log.decisions()], ["done", "done"])

    def test_night_speech_wakes_agent_at_next_poll(self):
        """Speech queued between night polls causes one decision at the next poll without ending nightly rest."""
        world = FakeWorld({"alice": [0, 64, 0]}, game_time=16_000, containers={(1, 64, 0): []})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        ScriptedClient(self.policy), Path(directory), command_error=CommandError,
                                        rest=(16_000, 24_000), min_gap=100)
            runtime.run(16_020)
            self.assertEqual(runtime.log.decisions(), [])
            world.hear("alice", "visitor", "Alice, wake up.")
            runtime.run(16_120)
            self.assertEqual([(entry.game_time, entry.interrupted, entry.outcome) for entry in runtime.log.decisions()],
                             [(16_040, False, "done")])
            self.assertIn('visitor (3 blocks away): "Alice, wake up."', runtime.log.decisions()[0].user_prompt)

    def test_resting_boundaries_and_midnight_wrap(self):
        """Rest windows include their start, exclude their end, wrap midnight, and repeat daily."""
        for rest, ticks in ((None, ((0, False), (20_000, False))),
                            ((16_000, 24_000), ((15_999, False), (16_000, True), (23_999, True),
                                               (24_000, False), (40_000, True))),
                            ((20_000, 4_000), ((0, True), (3_999, True), (4_000, False), (19_999, False),
                                              (20_000, True), (23_999, True), (24_000, True), (28_000, False)))):
            for tick, expected in ticks:
                with self.subTest(rest=rest, tick=tick):
                    self.assertEqual(resting(tick, rest), expected)


class SpeechInterruptTest(unittest.TestCase):
    """Others' completed words interrupt activity, but self-hearing and mid-speech do not."""

    def test_completed_speech_interrupts_listener_at_first_poll(self):
        """A listener's running move is interrupted at the first poll after another agent finishes speaking."""
        def policy(system, user):
            """Start a long move for Alice and one short utterance for Bob."""
            name = re.search(r"You are (\w+) at", user).group(1)
            first = "(tick 0)" in user
            if name == "alice":
                return ToolCall("move_to", {"x": 10, "y": 64, "z": 0}) if first else ToolCall("keep_going", {})
            return ToolCall("speak", {"text": "Alice please listen"}) if first else ToolCall("wait", {"ticks": 400})

        world = FakeWorld({"alice": [0, 64, 0], "bob": [3, 64, 0]}, skill_ticks={"move_to": 400})
        agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in world.bots]
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, agents, ScriptedClient(policy), Path(directory),
                                        command_error=CommandError)
            runtime.run(80)
            self.assertEqual([(entry.game_time, entry.agent, entry.interrupted) for entry in runtime.log.decisions()],
                             [(0, "alice", False), (0, "bob", False), (40, "alice", True), (40, "bob", False)])
            decision = runtime.log.get("alice", 40)
            self.assertTrue(decision.user_prompt.endswith(
                "You were interrupted because someone spoke. Choose your next action (you may continue what you "
                "were doing)."))
            self.assertEqual(agents[0].memory.to_json()["heard"],
                             [{"gameTime": 20, "speaker": "bob", "text": "Alice please listen", "distance": 3}])
            self.assertEqual(world.skills["bob"][0]["progress"], {"heardBy": ["alice"]})
            self.assertNotIn("You were interrupted", runtime.log.get("bob", 40).user_prompt)

    def test_self_hearing_does_not_interrupt_running_skill(self):
        """An agent hearing its own words does not decide while its long-running move continues."""
        def policy(system, user):
            """Choose a move lasting beyond both polls."""
            return ToolCall("move_to", {"x": 10, "y": 64, "z": 0})

        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"move_to": 400})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        ScriptedClient(policy), Path(directory), command_error=CommandError)
            runtime.run(40)
            world.hear("alice", "alice", "My own words.")
            runtime.run(80)
            self.assertEqual([entry.game_time for entry in runtime.log.decisions()], [0])
            self.assertEqual(world.request("skill_status", {"name": "alice"})["state"], "running")

    def test_mid_speech_listener_finishes_own_words_before_deciding(self):
        """A listener still speaking at forty waits until its own utterance finishes at eighty to decide."""
        def policy(system, user):
            """Give Alice eighty ticks of speech and Bob twenty, then wait."""
            if "(tick 0)" not in user:
                return ToolCall("wait", {"ticks": 400})
            name = re.search(r"You are (\w+) at", user).group(1)
            return ToolCall("speak", {"text": "one two three four five six seven eight nine ten eleven twelve"
                                     if name == "alice" else "Alice please listen"})

        world = FakeWorld({"alice": [0, 64, 0], "bob": [3, 64, 0]})
        agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in world.bots]
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, agents, ScriptedClient(policy), Path(directory),
                                        command_error=CommandError)
            runtime.run(120)
            self.assertEqual([(entry.game_time, entry.interrupted) for entry in runtime.log.decisions()
                              if entry.agent == "alice"], [(0, False), (80, False)])
            self.assertEqual(world.skills["alice"][0]["endTick"], 80)
            self.assertEqual(world.skills["alice"][0]["state"], "succeeded")
            self.assertEqual(agents[0].memory.to_json()["heard"][0]["gameTime"], 20)


class KeepGoingOrReplaceTest(unittest.TestCase):
    """Interrupted decisions either preserve the active skill or explicitly replace it."""

    def interrupt_with(self, response):
        """Interrupt a long move at forty and apply the chosen response."""
        def policy(system, user):
            """Start a move at zero and answer its interruption with the supplied action."""
            return ToolCall("move_to", {"x": 10, "y": 64, "z": 0}) if "(tick 0)" in user else response

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"move_to": 400})
        runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                    ScriptedClient(policy), Path(directory.name), command_error=CommandError)
        runtime.run(40)
        old_status = world.request("skill_status", {"name": "alice"})
        request_offset = len(world.requests)
        world.hear("alice", "visitor", "Alice, a message.")
        runtime.run(80)
        return runtime, world, old_status, world.requests[request_offset:]

    def test_keep_going_preserves_skill_without_start_request(self):
        """An interrupted keep_going records carried on without starting or replacing the running skill."""
        runtime, world, old_status, requests = self.interrupt_with(ToolCall("keep_going", {}))
        self.assertEqual(runtime.log.get("alice", 40).outcome, "carried on")
        self.assertTrue(runtime.log.get("alice", 40).interrupted)
        self.assertEqual([args for command, args in requests if command == "skill_start"], [])
        self.assertEqual(world.request("skill_status", {"name": "alice"}), old_status)
        self.assertEqual(old_status["state"], "running")

    def test_keep_going_adds_no_thinking_time_after_the_skill_ends(self):
        """An agent that carries on decides at the first poll after its skill ends, not after thinking time.

        Fails at 47a770e^ (before the fix): thinking time after `keep_going` made the agent wait until tick 160."""
        def policy(system, user):
            """Move at zero, carry on when interrupted at forty, then wait."""
            if "(tick 0)" in user:
                return ToolCall("move_to", {"x": 10, "y": 64, "z": 0})
            return ToolCall("keep_going", {}) if "(tick 40)" in user else ToolCall("wait", {"ticks": 400})

        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"move_to": 60})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        ScriptedClient(policy), Path(directory), command_error=CommandError,
                                        min_gap=100)
            runtime.run(40)
            world.hear("alice", "visitor", "Alice, a message.")
            runtime.run(200)
            self.assertEqual([entry.game_time for entry in runtime.log.decisions()][:3], [0, 40, 80])
            self.assertEqual(runtime.log.get("alice", 40).outcome, "carried on")

    def test_new_skill_cancels_previous_skill_and_updates_memory(self):
        """An interrupted replacement starts a new skill, cancels the old one, and records why it stopped."""
        runtime, world, old_status, requests = self.interrupt_with(ToolCall("wait", {"ticks": 200}))
        self.assertEqual([args for command, args in requests if command == "skill_start"],
                         [{"name": "alice", "skill": "wait", "args": {"ticks": 200}}])
        self.assertEqual(world.skills["alice"][0], {**old_status, "state": "cancelled", "reason": "interrupted"})
        self.assertEqual(world.request("skill_status", {"name": "alice"})["skillId"], "alice:2")
        self.assertEqual(runtime.log.get("alice", 40).outcome, "started")
        self.assertEqual(runtime.agents[0].memory.to_json()["actions"][0]["outcome"],
                         "stopped (you started something else)")


class ReplayTest(unittest.TestCase):
    """Matching decisions replay exactly and a single mismatched digest is handled locally."""

    def policy(self, system, user):
        """Choose the same eighty-tick wait for any prompt."""
        return ToolCall("wait", {"ticks": 80})

    def runtime_at(self, directory, model, source=None):
        """Create a fresh identical two-agent world and a hash-recording runtime."""
        world = FakeWorld({"bob": [3, 64, 0], "alice": [0, 64, 0]})
        agents = [Agent(RoleBrief(name, "worker", "Works.", "Help."), epoch_turns=8) for name in world.bots]
        return SimulationRuntime(world, agents, model, directory, command_error=CommandError, source=source,
                                 hash_region={"min": [0, 60, 0], "max": [10, 70, 10]})

    def trace(self, runtime):
        """Select the decision tick, agent, tool and arguments in replay order."""
        return [(entry.game_time, entry.agent, entry.tool, entry.args) for entry in runtime.log.decisions()]

    def mismatched_source(self, runtime, path):
        """Copy a log with only Bob's final decision digest deliberately changed."""
        records = [asdict(entry) for entry in runtime.log.decisions()]
        records[-1]["digest"] = "deliberately different"
        path.write_text("".join(json.dumps(entry) + "\n" for entry in records), encoding="utf-8")
        return DecisionLog(path)

    def test_identical_run_reuses_all_decisions_without_model(self):
        """An identical fresh world reuses every logged decision with zero model calls and identical hashes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factual = self.runtime_at(root / "factual", ScriptedClient(self.policy))
            factual.run(200)
            replay = self.runtime_at(root / "replay", None, DecisionLog(factual.log.path))
            replay.run(200)
            self.assertEqual(replay.stats.model_calls, 0)
            self.assertEqual(replay.stats.replayed, replay.stats.decisions)
            self.assertEqual(replay.stats.decisions, 6)
            self.assertEqual(self.trace(replay), self.trace(factual))
            self.assertEqual(replay.hashes, factual.hashes)
            self.assertEqual([entry.outcome for entry in replay.log.decisions()], ["started"] * 6)

    def test_one_digest_mismatch_calls_only_that_agent_at_that_tick(self):
        """One changed source digest makes only Bob's decision at tick 160 call the available model."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factual = self.runtime_at(root / "factual", ScriptedClient(self.policy))
            factual.run(200)
            source = self.mismatched_source(factual, root / "changed.jsonl")
            branch = self.runtime_at(root / "branch", ScriptedClient(self.policy), source)
            branch.run(200)
            self.assertEqual(branch.stats.model_calls, 1)
            self.assertEqual(branch.stats.replayed, 5)
            self.assertEqual([(entry.agent, entry.game_time) for entry in branch.log.decisions()
                              if entry.source == "model"], [("bob", 160)])
            self.assertEqual(self.trace(branch), self.trace(factual))
            self.assertEqual(branch.hashes, factual.hashes)

    def test_digest_mismatch_without_model_records_divergence(self):
        """Without a model a changed digest raises ReplayDiverged and writes the exact divergent agent and prompt."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factual = self.runtime_at(root / "factual", ScriptedClient(self.policy))
            factual.run(200)
            source = self.mismatched_source(factual, root / "changed.jsonl")
            replay = self.runtime_at(root / "replay", None, source)
            with self.assertRaisesRegex(ReplayDiverged, "tick 160.*bob"):
                replay.run(200)
            expected = factual.log.get("bob", 160)
            self.assertEqual(json.loads((root / "replay" / "diverged.json").read_text(encoding="utf-8")),
                             [{"gameTime": 160, "agent": "bob", "digest": expected.digest,
                               "user_prompt": expected.user_prompt}])
            self.assertEqual(replay.stats.model_calls, 0)
            self.assertEqual([entry.game_time for entry in replay.log.decisions()], [0, 0, 80, 80])


class BudgetTest(unittest.TestCase):
    """Budgets stop an entire pending batch before calls, including provider usage exhaustion."""

    def test_insufficient_budget_stops_before_any_model_call_and_writes_outputs(self):
        """A budget smaller than the first deciding batch stops before any call and still persists hashes and stats."""
        def policy(system, user):
            """Expose any completion wrongly attempted before the runtime's own budget check."""
            raise UsageLimitReached("budget preflight must not reach the provider")

        world = FakeWorld({"alice": [0, 64, 0], "bob": [3, 64, 0]})
        agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in world.bots]
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, agents, ScriptedClient(policy), Path(directory),
                                        command_error=CommandError, max_model_calls=1)
            with self.assertRaisesRegex(BudgetExceeded, "budget 1.*tick 0"):
                runtime.run(80)
            self.assertEqual(runtime.stats.model_calls, 0)
            self.assertEqual(runtime.stats.decisions, 0)
            self.assertEqual(runtime.log.decisions(), [])
            self.assertEqual(world.requests, [("ping", {}), ("skill_status", {"name": "alice"}),
                                             ("observe", {"name": "alice", "include_terrain": False}),
                                             ("skill_status", {"name": "bob"}),
                                             ("observe", {"name": "bob", "include_terrain": False})])
            self.assertEqual(json.loads((Path(directory) / "hashes.json").read_text(encoding="utf-8")), [])
            self.assertEqual(json.loads((Path(directory) / "stats.json").read_text(encoding="utf-8")),
                             asdict(runtime.stats))

    def test_provider_usage_limit_becomes_budget_exceeded(self):
        """A provider UsageLimitReached aborts the run as BudgetExceeded rather than a no-action decision."""
        class LimitedPolicy:
            """Raise the provider's usage exception for every completion request."""

            model = "limited"

            def complete(self, system, user, tools, seed=None, history=()):
                """Report exhausted provider usage without performing any external work."""
                raise UsageLimitReached("daily allowance exhausted")

        world = FakeWorld({"alice": [0, 64, 0]})
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimulationRuntime(world, [Agent(RoleBrief("alice", "worker", "Works.", "Help."))],
                                        LimitedPolicy(), Path(directory), command_error=CommandError)
            with self.assertRaisesRegex(BudgetExceeded, "Provider usage limit reached") as raised:
                runtime.run(80)
            self.assertIsInstance(raised.exception.__cause__, UsageLimitReached)
            self.assertEqual(runtime.log.decisions(), [])
            self.assertEqual(runtime.stats.model_errors, 0)
            self.assertTrue((Path(directory) / "hashes.json").is_file())
            self.assertTrue((Path(directory) / "stats.json").is_file())


class ModelFailureTest(unittest.TestCase):
    """A client failure becomes a replayable no-action turn without stopping other decisions."""

    def test_runtime_error_records_no_action_and_replays_without_model(self):
        """One failing completion records no action, lets the run continue, and replays its outcome with no calls."""
        def policy(system, user):
            """Fail only Alice's first prompt and otherwise choose eighty ticks of waiting."""
            name = re.search(r"You are (\w+) at", user).group(1)
            if name == "alice" and "(tick 0)" in user:
                raise RuntimeError("scripted completion failed")
            return ToolCall("wait", {"ticks": 80})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = FakeWorld({"alice": [0, 64, 0], "bob": [3, 64, 0]})
            agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in world.bots]
            runtime = SimulationRuntime(world, agents, ScriptedClient(policy), root / "factual",
                                        command_error=CommandError)
            runtime.run(120)
            self.assertEqual([(entry.game_time, entry.agent, entry.tool, entry.outcome)
                              for entry in runtime.log.decisions()],
                             [(0, "alice", None, "no action chosen"), (0, "bob", "wait", "started"),
                              (40, "alice", "wait", "started"), (80, "bob", "wait", "started")])
            self.assertIsNone(runtime.log.get("alice", 0).args)
            self.assertEqual(runtime.stats.model_errors, 1)
            self.assertEqual(runtime.stats.model_calls, 4)
            self.assertEqual(world.game_time, 120)
            replay_world = FakeWorld({"alice": [0, 64, 0], "bob": [3, 64, 0]})
            replay_agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in replay_world.bots]
            replay = SimulationRuntime(replay_world, replay_agents, None, root / "replay",
                                       command_error=CommandError, source=DecisionLog(runtime.log.path))
            replay.run(120)
            self.assertEqual(replay.stats.model_calls, 0)
            self.assertEqual(replay.stats.replayed, 4)
            self.assertEqual([(entry.game_time, entry.agent, entry.tool, entry.args, entry.outcome)
                              for entry in replay.log.decisions()],
                             [(entry.game_time, entry.agent, entry.tool, entry.args, entry.outcome)
                              for entry in runtime.log.decisions()])


class OutputsAndHashesTest(unittest.TestCase):
    """Output logs, trajectories, statistics and optional hash samples have exact cadence."""

    def test_outputs_and_twenty_or_forty_tick_hash_cadence(self):
        """A two-agent run writes each decision and poll and samples hashes only at the requested cadence and region."""
        def policy(system, user):
            """Choose a fixed eighty-tick skill to leave some polls without decisions."""
            return ToolCall("wait", {"ticks": 80})

        region = {"min": [0, 60, 0], "max": [10, 70, 10]}
        for hash_every, hash_region in ((20, region), (40, region), (20, None)):
            with self.subTest(hash_every=hash_every, region=hash_region), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                world = FakeWorld({"bob": [3, 64, 0], "alice": [0, 64, 0]})
                agents = [Agent(RoleBrief(name, "worker", "Works.", "Help.")) for name in world.bots]
                runtime = SimulationRuntime(world, agents, ScriptedClient(policy), root, command_error=CommandError,
                                            hash_region=hash_region, hash_every=hash_every)
                runtime.run(200)
                hashes = json.loads((root / "hashes.json").read_text(encoding="utf-8"))
                self.assertEqual(hashes, runtime.hashes)
                self.assertEqual([entry["gameTime"] for entry in hashes],
                                 list(range(hash_every, 201, hash_every)) if hash_region is not None else [])
                step = {"n": 40}
                if hash_region is not None:
                    step.update(hash_every=hash_every, region=hash_region)
                self.assertEqual([args for command, args in world.requests if command == "step"], [step] * 5)
                decisions = [json.loads(line) for line in (root / "decisions.jsonl").read_text().splitlines()]
                self.assertEqual(decisions, [asdict(entry) for entry in runtime.log.decisions()])
                self.assertEqual(len(decisions), runtime.stats.decisions)
                self.assertEqual(runtime.stats.decisions, 6)
                trajectory = [json.loads(line) for line in (root / "trajectory.jsonl").read_text().splitlines()]
                self.assertEqual([(entry["t"], entry["agent"], entry["pos"]) for entry in trajectory],
                                 [(tick, name, world.bots[name]) for tick in range(0, 200, 40)
                                  for name in ("alice", "bob")])
                self.assertEqual(len(trajectory), 10)
                self.assertEqual(json.loads((root / "stats.json").read_text(encoding="utf-8")), asdict(runtime.stats))


class ForkRoundTripTest(unittest.TestCase):
    """Saved agent memories and thinking deadlines reproduce an uninterrupted continuation."""

    def policy(self, system, user):
        """Wait until 360, read instantly there, and then start a long wait."""
        tick = int(re.search(r"Time: .* \(tick (\d+)\)", user).group(1))
        if tick == 0:
            return ToolCall("wait", {"ticks": 360})
        if tick == 360:
            return ToolCall("open_container", {"x": 1, "y": 64, "z": 0})
        return ToolCall("wait", {"ticks": 320})

    def runtime_at(self, world, agents, directory):
        """Construct a forkable runtime with thinking time and forty-tick hash samples."""
        return SimulationRuntime(world, agents, ScriptedClient(self.policy), directory, command_error=CommandError,
                                 min_gap=100, hash_every=40, hash_region={"min": [0, 60, 0], "max": [10, 70, 10]})

    def trace(self, runtime, after):
        """Select every reproducible decision field after the fork, excluding wall-clock accounting."""
        return [(entry.game_time, entry.agent, entry.tool, entry.args, entry.outcome, entry.interrupted,
                 entry.user_prompt, entry.digest) for entry in runtime.log.decisions() if entry.game_time >= after]

    def test_restored_timing_reproduces_fork_and_missing_timing_diverges(self):
        """Restored thinking time matches an uninterrupted fork, while missing timing changes its continuation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world_a = FakeWorld({"alice": [0, 64, 0]}, containers={(1, 64, 0): []})
            agents = [Agent(RoleBrief("alice", "worker", "Works.", "Help."), epoch_turns=8)]
            uninterrupted = self.runtime_at(world_a, agents, root / "uninterrupted")
            uninterrupted.run(800)
            self.assertEqual([entry.game_time for entry in uninterrupted.log.decisions()], [0, 360, 480])
            world_b = FakeWorld({"alice": [0, 64, 0]}, containers={(1, 64, 0): []})
            prefix = self.runtime_at(world_b, [Agent(RoleBrief("alice", "worker", "Works.", "Help."), epoch_turns=8)],
                                     root / "prefix")
            prefix.run(400)
            snapshot = root / "agents.json"
            prefix.save_state(snapshot)
            state = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(state["readyAt"], {"alice": 460})
            tick, loaded = SimulationRuntime.load_agents(snapshot)
            self.assertEqual(tick, 400)
            self.assertEqual([agent.to_json() for agent in loaded], [agent.to_json() for agent in prefix.agents])
            restored = self.runtime_at(copy.deepcopy(world_b), loaded, root / "restored")
            restored.restore_timing(snapshot)
            restored.run(800)
            self.assertEqual(self.trace(restored, 400), self.trace(uninterrupted, 400))
            self.assertEqual(prefix.hashes + restored.hashes, uninterrupted.hashes)
            tick, loaded_without_timing = SimulationRuntime.load_agents(snapshot)
            self.assertEqual(tick, 400)
            without_timing = self.runtime_at(copy.deepcopy(world_b), loaded_without_timing, root / "without_timing")
            without_timing.run(800)
            self.assertEqual([entry.game_time for entry in without_timing.log.decisions()], [400, 720])
            self.assertNotEqual(self.trace(without_timing, 400), self.trace(uninterrupted, 400))
            self.assertNotEqual(prefix.hashes + without_timing.hashes, uninterrupted.hashes)


if __name__ == "__main__":
    unittest.main()
