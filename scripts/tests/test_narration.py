"""Pin narrated rulings, decision timing, and shadow execution without external services."""

import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.narration import (GameMaster, NarratedRuntime, Resolution, ShadowEngine, narrated_outcomes,
                                    narration_error, outcome_divergence)
from orchestrator.render import ObservationRenderer, clock
from orchestrator.runtime import BudgetExceeded
from scripts.harness import CommandError
from scripts.tests.fake_world import FakeWorld


class ResolutionDefaultsTest(unittest.TestCase):
    """Missing rulings and malformed observations have conservative, exact defaults."""

    def test_reply_without_tool_call_rejects_and_preserves_previous_view(self):
        """A reply without a ruling rejects with no duration, effects, observations, or changed view."""
        def policy(system, user):
            """Return no usable GM tool call."""
            return None

        gm = GameMaster(ScriptedClient(policy), "A village.", {"alice": "Beside the bell."})
        resolution = gm.resolve(0, "alice", ToolCall("wait", {"ticks": 40}))
        self.assertEqual(resolution, Resolution(False, "no ruling", 0, [], "Beside the bell.", {}))
        self.assertEqual(gm.view("alice"), "Beside the bell.")
        self.assertEqual(gm.calls, 1)

    def test_negative_duration_is_clamped_to_zero(self):
        """A negative duration becomes zero while the rest of the ruling and new actor view survive."""
        def policy(system, user):
            """Accept a move with an invalid negative duration."""
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": -40,
                                        "effects": ["Alice moved."], "actor_view": "Inside the workshop.",
                                        "observations": []})

        gm = GameMaster(ScriptedClient(policy), "A village.", {"alice": "Beside the bell."})
        resolution = gm.resolve(40, "alice", ToolCall("move_to", {"x": 4, "y": 64, "z": 2}))
        self.assertEqual(resolution, Resolution(True, "", 0, ["Alice moved."], "Inside the workshop.", {}))
        self.assertEqual(gm.view("alice"), "Inside the workshop.")

    def test_unknown_people_and_non_object_observations_are_dropped(self):
        """Only object observations naming known people reach the resolution's observation map."""
        def policy(system, user):
            """Mix valid observations with unknown people and non-object entries."""
            return ToolCall("resolve", {"accepted": True, "reason": "", "actor_view": "",
                                        "observations": [None, "ben", 7, False, ["cy", "a noise"], {},
                                                         {"person": "stranger", "text": "Do not route."},
                                                         {"person": "ben", "text": "The bell rings."},
                                                         {"person": "cy", "text": "A bird flies past."}]})

        views = {"alice": "Beside the bell.", "ben": "In the shop.", "cy": "At the market."}
        gm = GameMaster(ScriptedClient(policy), "A village.", views)
        resolution = gm.resolve(0, "alice", ToolCall("wait", {"ticks": 40}))
        self.assertEqual(resolution, Resolution(True, "", 0, [], views["alice"],
                                               {"ben": "The bell rings.", "cy": "A bird flies past."}))
        self.assertEqual({person: gm.view(person) for person in views}, views)


class MalformedObservationTest(unittest.TestCase):
    """A GM observation without text must not end a narrated run."""

    def test_observation_without_text_routes_an_empty_perception(self):
        """An observation naming a known person but lacking `text` becomes an empty perception, not a KeyError."""
        def policy(system, user):
            """Accept the act with one observation that has no text."""
            return ToolCall("resolve", {"accepted": True, "reason": "", "actor_view": "",
                                        "observations": [{"person": "ben"}]})

        gm = GameMaster(ScriptedClient(policy), "A village.", {"alice": "Beside the bell.", "ben": "In the shop."})
        resolution = gm.resolve(0, "alice", ToolCall("wait", {"ticks": 40}))
        self.assertEqual(resolution.observations, {"ben": ""})


class ObservationRoutingTest(unittest.TestCase):
    """Speech interrupts named listeners, while other observations only enter noticed memory."""

    def run_observations(self, call, until):
        """Make listeners busy before Zoe's act, routing observations only from that first act."""
        def agent_policy(system, user):
            """Let Zoe perform the specified act while the other people work for four hundred ticks."""
            return call if "Your name is zoe." in system else ToolCall("wait", {"ticks": 400})

        def gm_policy(system, user):
            """Describe Zoe's first act to two listeners and to Zoe herself, but not to Cy."""
            observations = []
            if f"tick 0), zoe attempts: {call.name} " in user:
                observations = [{"person": "alice", "text": "The GM's account of Zoe's act."},
                                {"person": "ben", "text": "The GM's account of Zoe's act."},
                                {"person": "zoe", "text": "An actor must ignore their own entry."}]
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 400,
                                        "observations": observations})

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.agents = {name: Agent(RoleBrief(name, "worker", "Works in town.", "Help the village."))
                       for name in ("zoe", "cy", "ben", "alice")}
        gm = GameMaster(ScriptedClient(gm_policy), "A village.",
                        {name: "Beside the bell." for name in self.agents})
        runtime = NarratedRuntime(list(self.agents.values()), ScriptedClient(agent_policy), gm, Path(directory.name))
        runtime.run(0, until)
        return runtime, gm

    def test_speech_interrupts_busy_named_listeners_at_next_poll_but_not_the_actor(self):
        """Accepted speech delivers the GM's words to each named listener and interrupts only those listeners."""
        runtime, gm = self.run_observations(ToolCall("speak", {"text": "The original words.", "mode": "talk"}), 80)
        self.assertEqual([(decision.game_time, decision.agent, decision.interrupted)
                          for decision in runtime.log.decisions()],
                         [(0, "alice", False), (0, "ben", False), (0, "cy", False), (0, "zoe", False),
                          (40, "alice", True), (40, "ben", True)])
        for name in ("alice", "ben"):
            with self.subTest(listener=name):
                memory = self.agents[name].memory.to_json()
                self.assertEqual(memory["heard"], [{"gameTime": 0, "speaker": "zoe",
                                                   "text": "The GM's account of Zoe's act.", "distance": None}])
                self.assertEqual(memory["noticed"], [])
                prompt = runtime.log.get(name, 40).user_prompt
                self.assertIn("What you heard:", prompt)
                self.assertIn('zoe: "The GM\'s account of Zoe\'s act."', prompt)
                self.assertNotIn("The original words.", prompt)
        for name in ("cy", "zoe"):
            with self.subTest(non_listener=name):
                self.assertEqual(self.agents[name].memory.to_json()["heard"], [])
                self.assertEqual(self.agents[name].memory.to_json()["noticed"], [])
        self.assertEqual(gm.calls, 6)

    def test_non_speech_is_noticed_without_interrupting_and_ignores_the_actor_entry(self):
        """A non-speech observation appears under What you noticed without making busy listeners decide early."""
        runtime, gm = self.run_observations(ToolCall("place", {"x": 1, "y": 64, "z": 2,
                                                              "item": "minecraft:lantern"}), 440)
        self.assertEqual([(decision.game_time, decision.agent, decision.interrupted)
                          for decision in runtime.log.decisions()],
                         [(tick, name, False) for tick in (0, 400) for name in ("alice", "ben", "cy", "zoe")])
        for name in ("alice", "ben"):
            with self.subTest(listener=name):
                self.assertEqual(self.agents[name].memory.to_json()["noticed"],
                                 [{"gameTime": 0, "text": "The GM's account of Zoe's act."}])
                self.assertEqual(self.agents[name].memory.to_json()["heard"], [])
                self.assertIn(f"What you noticed:\n  [{clock(0)}] The GM's account of Zoe's act.",
                              runtime.log.get(name, 400).user_prompt)
        for name in ("cy", "zoe"):
            with self.subTest(non_listener=name):
                self.assertEqual(self.agents[name].memory.to_json()["noticed"], [])
                self.assertEqual(self.agents[name].memory.to_json()["heard"], [])
        self.assertEqual(gm.calls, 8)


class NarratedThinkingTimeTest(unittest.TestCase):
    """GM rejections impose thinking time on the forty-tick decision grid."""

    def test_rejection_at_zero_with_hundred_tick_gap_next_decides_at_120(self):
        """A rejected act reports its reason and delays the next decision until tick 120."""
        def agent_policy(system, user):
            """Try walking to the workshop at every decision."""
            return ToolCall("move_to", {"x": 4, "y": 64, "z": 2})

        def gm_policy(system, user):
            """Reject walking because the gate is locked."""
            return ToolCall("resolve", {"accepted": False, "reason": "locked gate", "duration_ticks": 0})

        agent = Agent(RoleBrief("alice", "worker", "Works in town.", "Help the village."))
        gm = GameMaster(ScriptedClient(gm_policy), "A village.", {"alice": "Beside a locked gate."})
        with tempfile.TemporaryDirectory() as directory:
            runtime = NarratedRuntime([agent], ScriptedClient(agent_policy), gm, Path(directory), min_gap=100)
            runtime.run(0, 160)
        self.assertEqual([(decision.game_time, decision.outcome) for decision in runtime.log.decisions()],
                         [(0, "not possible: locked gate"), (120, "not possible: locked gate")])
        self.assertEqual([entry["outcome"] for entry in agent.memory.to_json()["actions"]],
                         ["not possible: locked gate", "not possible: locked gate"])
        self.assertEqual((runtime.stats.decisions, runtime.stats.model_calls, gm.calls), (2, 2, 2))


class NarratedDecisionControlTest(unittest.TestCase):
    """No-op decisions bypass the GM and model-call budgets include both sides of resolution."""

    def setUp(self):
        """Give each decision-control test its own agent, GM, and temporary logs."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.agent = Agent(RoleBrief("alice", "worker", "Works in town.", "Help the village."))
        self.gm = GameMaster(ScriptedClient(self.accept_instantly), "A village.", {"alice": "Beside the bell."})

    def accept_instantly(self, system, user):
        """Accept ordinary acts without narrated elapsed time."""
        return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 0})

    def wait_policy(self, system, user):
        """Choose one ordinary skill act on every poll."""
        return ToolCall("wait", {"ticks": 40})

    def test_interrupted_keep_going_carries_on_without_a_gm_call(self):
        """An interrupted worker choosing keep_going keeps the original deadline without another GM ruling."""
        def agent_policy(system, user):
            """Have Zoe speak and Alice continue her work when interrupted."""
            if "Your name is zoe." in system:
                return ToolCall("speak", {"text": "Keep working.", "mode": "talk"})
            if "You were interrupted because someone spoke." in user:
                return ToolCall("keep_going", {})
            return ToolCall("wait", {"ticks": 400})

        def gm_policy(system, user):
            """Give both initial acts long durations and deliver only Zoe's speech to Alice."""
            observations = [{"person": "alice", "text": "Keep working."}] if " attempts: speak " in user else []
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 400,
                                        "observations": observations})

        speaker = Agent(RoleBrief("zoe", "worker", "Works in town.", "Help the village."))
        gm = GameMaster(ScriptedClient(gm_policy), "A village.", {"alice": "The square.", "zoe": "The square."})
        runtime = NarratedRuntime([speaker, self.agent], ScriptedClient(agent_policy), gm, self.directory)
        runtime.run(0, 400)
        self.assertEqual([(decision.game_time, decision.agent, decision.tool, decision.interrupted, decision.outcome)
                          for decision in runtime.log.decisions()],
                         [(0, "alice", "wait", False, "done (took 400 ticks)"),
                          (0, "zoe", "speak", False, "done (took 400 ticks)"),
                          (40, "alice", "keep_going", True, "carried on")])
        self.assertEqual([(act["agent"], act["tool"]) for act in runtime.acts], [("alice", "wait"), ("zoe", "speak")])
        self.assertEqual(self.agent.memory.to_json()["actions"][-1],
                         {"gameTime": 40, "tool": "keep_going", "args": {}, "outcome": "carried on", "skillId": None})
        self.assertEqual((runtime.stats.model_calls, gm.calls), (3, 2))

    def test_keep_going_adds_no_thinking_time_after_the_act_ends(self):
        """A narrated agent that carries on decides at the first poll after its act ends, as in the engine runtime.

        Fails at 47a770e^ (before the fix): thinking time after `keep_going` made Alice wait until tick 160."""
        def agent_policy(system, user):
            """Zoe speaks; Alice carries on when interrupted and otherwise waits."""
            if "Your name is zoe." in system:
                return ToolCall("speak", {"text": "Keep working.", "mode": "talk"})
            if "You were interrupted because someone spoke." in user:
                return ToolCall("keep_going", {})
            return ToolCall("wait", {"ticks": 60})

        def gm_policy(system, user):
            """Alice's wait takes 60 ticks; Zoe's speech takes 400 and reaches Alice."""
            speech = " attempts: speak " in user
            observations = [{"person": "alice", "text": "Keep working."}] if speech else []
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 400 if speech else 60,
                                        "observations": observations})

        speaker = Agent(RoleBrief("zoe", "worker", "Works in town.", "Help the village."))
        gm = GameMaster(ScriptedClient(gm_policy), "A village.", {"alice": "The square.", "zoe": "The square."})
        runtime = NarratedRuntime([speaker, self.agent], ScriptedClient(agent_policy), gm, self.directory,
                                  min_gap=100)
        runtime.run(0, 200)
        self.assertEqual([decision.game_time for decision in runtime.log.decisions() if decision.agent == "alice"][:3],
                         [0, 40, 80])

    def test_agent_reply_without_tool_records_no_action_without_calling_gm(self):
        """A missing agent tool call records no action chosen in both the decision log and memory."""
        def policy(system, user):
            """Choose no usable action."""
            return None

        runtime = NarratedRuntime([self.agent], ScriptedClient(policy), self.gm, self.directory)
        runtime.run(0, 40)
        decision = runtime.log.get("alice", 0)
        self.assertEqual((decision.tool, decision.args, decision.outcome), (None, None, "no action chosen"))
        self.assertEqual(self.agent.memory.to_json()["actions"],
                         [{"gameTime": 0, "tool": "(none)", "args": {}, "outcome": "no action chosen",
                           "skillId": None}])
        self.assertEqual(runtime.acts, [])
        self.assertEqual((runtime.stats.decisions, runtime.stats.model_calls, self.gm.calls), (1, 1, 0))

    def test_budget_counts_agent_and_gm_calls_before_the_next_poll(self):
        """A three-call budget allows one agent-plus-GM turn but refuses the next two-call turn."""
        runtime = NarratedRuntime([self.agent], ScriptedClient(self.wait_policy), self.gm, self.directory,
                                  max_model_calls=3)
        with self.assertRaises(BudgetExceeded) as caught:
            runtime.run(0, 80)
        self.assertEqual(str(caught.exception), "Model-call budget 3 would be exceeded at tick 40")
        self.assertEqual([decision.game_time for decision in runtime.log.decisions()], [0])
        self.assertEqual((runtime.stats.model_calls, self.gm.calls, len(runtime.acts)), (1, 1, 1))
        self.assertEqual(json.loads((self.directory / "stats.json").read_text(encoding="utf-8"))["gm_calls"], 1)

    def test_insufficient_budget_stops_before_either_model_is_called(self):
        """A one-call budget rejects the initial agent-plus-GM turn before either model runs."""
        runtime = NarratedRuntime([self.agent], ScriptedClient(self.wait_policy), self.gm, self.directory,
                                  max_model_calls=1)
        with self.assertRaises(BudgetExceeded) as caught:
            runtime.run(0, 40)
        self.assertEqual(str(caught.exception), "Model-call budget 1 would be exceeded at tick 0")
        self.assertEqual((runtime.stats.model_calls, self.gm.calls), (0, 0))
        self.assertEqual(runtime.log.decisions(), [])
        self.assertEqual(runtime.acts, [])
        self.assertEqual(self.agent.memory.to_json()["actions"], [])

    def test_exact_budget_allows_one_agent_and_gm_turn(self):
        """A budget equal to the two required calls permits one complete turn."""
        runtime = NarratedRuntime([self.agent], ScriptedClient(self.wait_policy), self.gm, self.directory,
                                  max_model_calls=2)
        runtime.run(0, 40)
        self.assertEqual((runtime.stats.decisions, runtime.stats.model_calls, self.gm.calls), (1, 1, 1))
        self.assertEqual(runtime.log.get("alice", 0).outcome, "done")


class NarratedShadowRuntimeTest(unittest.TestCase):
    """A FakeWorld shadow supplies physical verdicts, final skill states, and resynchronized views."""

    def run_move(self, skill_ticks=40):
        """Run one narrated move for eighty ticks with temporary logs and a real ShadowEngine."""
        def agent_policy(system, user):
            """Move to the workshop with a private planning note."""
            return ToolCall("move_to", {"x": 4, "y": 64, "z": 2, "note": "Reach the workshop."})

        def gm_policy(system, user):
            """Accept the move and replace the narrator's view for eighty ticks."""
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 80,
                                        "effects": ["Alice moved."], "actor_view": "GM-only room.",
                                        "observations": []})

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"move_to": skill_ticks})
        agent = Agent(RoleBrief("alice", "worker", "Works in town.", "Help the village."))
        gm = GameMaster(ScriptedClient(gm_policy), "A village.", {"alice": "Beside the bell."})
        runtime = NarratedRuntime([agent], ScriptedClient(agent_policy), gm, Path(directory.name),
                                  shadow=ShadowEngine(world, CommandError))
        runtime.run(0, 80)
        return runtime, gm, world, Path(directory.name)

    def test_accepted_skill_has_dry_run_final_state_and_exact_output_files(self):
        """An accepted move starts in the shadow, finishes there, and persists its verdict and GM counters."""
        runtime, gm, world, directory = self.run_move()
        expected_act = {"gameTime": 0, "agent": "alice", "tool": "move_to", "args": {"x": 4, "y": 64, "z": 2},
                        "gm": {"accepted": True, "reason": "", "duration_ticks": 80, "effects": ["Alice moved."],
                               "actor_view": "GM-only room.", "observations": {}},
                        "engine_check": {"feasible": True, "reason": None, "ticks": 40},
                        "engine": "started", "engine_final": "succeeded"}
        self.assertEqual(runtime.acts, [expected_act])
        self.assertEqual((world.game_time, world.bots["alice"], gm.view("alice")),
                         (80, [4.5, 64, 2.5], "GM-only room."))
        action = {"name": "alice", "skill": "move_to", "args": {"x": 4.5, "y": 64, "z": 2.5}}
        self.assertEqual([args for command, args in world.requests if command == "skill_start"],
                         [{**action, "dry_run": True}, action])
        self.assertEqual((directory / "acts.jsonl").read_text(encoding="utf-8"),
                         json.dumps(expected_act, ensure_ascii=False, sort_keys=True) + "\n")
        expected_stats = {"decisions": 1, "model_calls": 1, "replayed": 0, "input_tokens": 0, "output_tokens": 0,
                          "cache_read_tokens": 0, "cache_write_tokens": 0, "model_errors": 0, "model_seconds": 0.0,
                          "model_retries": 0, "gm_calls": 1, "gm_input_tokens": 0, "gm_cached_tokens": 0,
                          "gm_output_tokens": 0}
        self.assertEqual((directory / "stats.json").read_text(encoding="utf-8"),
                         json.dumps(expected_stats, indent=1) + "\n")

    def test_unfinished_skill_is_recorded_as_running_at_run_end(self):
        """A skill still executing when the run ends keeps engine_final equal to running."""
        runtime, gm, world, directory = self.run_move(skill_ticks=120)
        self.assertEqual(runtime.acts[0]["engine_final"], "running")
        self.assertEqual(runtime.acts[0]["engine_check"], {"feasible": True, "reason": None, "ticks": 120})
        self.assertEqual((world.game_time, world.skills["alice"][-1]["state"], world.bots["alice"]),
                         (80, "running", [0, 64, 0]))
        self.assertEqual(json.loads((directory / "acts.jsonl").read_text(encoding="utf-8"))["engine_final"], "running")

    def test_resync_replaces_deciding_agent_view_with_engine_rendering_at_each_poll(self):
        """Resynchronization replaces stale narrator text in both the agent prompt and the GM ruling prompt."""
        gm_views = []

        def agent_policy(system, user):
            """Move on each decision so the engine's next view changes."""
            return ToolCall("move_to", {"x": 4, "y": 64, "z": 2})

        def gm_policy(system, user):
            """Save the current-view prompt and deliberately narrate an engine-inconsistent room."""
            gm_views.append(user.split("alice's current view: ", 1)[1].removesuffix("\n\nResolve this act."))
            return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 40,
                                        "actor_view": "GM-only room."})

        world = FakeWorld({"alice": [0, 64, 0]}, skill_ticks={"move_to": 40})
        agent = Agent(RoleBrief("alice", "worker", "Works in town.", "Help the village."))
        gm = GameMaster(ScriptedClient(gm_policy), "A village.", {"alice": "Invented initial room."})
        expected_views = []
        with tempfile.TemporaryDirectory() as directory:
            runtime = NarratedRuntime([agent], ScriptedClient(agent_policy), gm, Path(directory),
                                      shadow=ShadowEngine(world, CommandError), resync=True)
            for tick in (0, 40):
                observation = world.request("observe", {"name": "alice", "include_terrain": False})
                expected_views.append(ObservationRenderer().render(observation))
                runtime.run(tick, tick + 40)
                self.assertEqual(gm.view("alice"), "GM-only room.")
        self.assertEqual(gm_views, expected_views)
        self.assertNotEqual(expected_views[0], expected_views[1])
        self.assertEqual([decision.game_time for decision in runtime.log.decisions()], [0, 40])
        for decision, expected_view in zip(runtime.log.decisions(), expected_views):
            with self.subTest(tick=decision.game_time):
                self.assertIn(f"\n\n{expected_view}\n\n", decision.user_prompt)
                self.assertNotIn("GM-only room.", decision.user_prompt)
                self.assertNotIn("Invented initial room.", decision.user_prompt)


class ShadowEngineExecutionTest(unittest.TestCase):
    """Shadow execution distinguishes continuation, translation failures, command failures, and instant acts."""

    def setUp(self):
        """Create a shadow with one agent and one accessible container."""
        self.world = FakeWorld({"alice": [0, 64, 0]}, containers={(1, 64, 0): []})
        self.shadow = ShadowEngine(self.world, CommandError)

    def test_keep_going_is_carried_on_without_any_world_request(self):
        """Continuing an existing action reports carried on without sending a command."""
        self.assertEqual(self.shadow.execute("alice", ToolCall("keep_going", {})), "carried on")
        self.assertEqual(self.world.requests, [])

    def test_unknown_tool_is_invalid_without_any_world_request(self):
        """An unknown action reports the exact translator failure before touching the shadow world."""
        self.assertEqual(self.shadow.execute("alice", ToolCall("fly", {})), "invalid action: Unknown tool: fly")
        self.assertEqual(self.world.requests, [])

    def test_command_error_is_failed_with_the_engine_reason(self):
        """Opening a missing container reports failed with the command prefix removed from the reason."""
        self.assertEqual(self.shadow.execute("alice", ToolCall("open_container", {"x": 2, "y": 64, "z": 0})),
                         "failed: no_container")
        self.assertEqual(self.world.requests, [("open_container", {"name": "alice", "pos": [2, 64, 0]})])

    def test_instant_act_is_done_without_starting_a_skill(self):
        """Opening an accessible container reports done and starts no timed skill."""
        self.assertEqual(self.shadow.execute("alice", ToolCall("open_container", {"x": 1, "y": 64, "z": 0})), "done")
        self.assertEqual(self.world.requests, [("open_container", {"name": "alice", "pos": [1, 64, 0]})])
        self.assertEqual(self.world.skills["alice"], [])

    def test_rejected_skill_is_not_possible_with_the_engine_reason(self):
        """A skill rejected by the engine returns not possible rather than started."""
        world = FakeWorld({"alice": [0, 64, 0]}, rejects={"mine": "missing_tool"})
        shadow = ShadowEngine(world, CommandError)
        self.assertEqual(shadow.execute("alice", ToolCall("mine", {"x": 1, "y": 64, "z": 0})),
                         "not possible: missing_tool")
        self.assertEqual(world.requests, [("skill_start", {"name": "alice", "skill": "mine",
                                                          "args": {"x": 1, "y": 64, "z": 0}})])
        self.assertEqual(world.skills["alice"], [])



class NarratedOutcomesTest(unittest.TestCase):
    """The narrated story's outcomes come only from accepted acts, judged by the engine's own rules."""

    MANIFEST = {"success_predicates": {"fair_open": {"all": [
                    {"blocks_present": [{"pos": [56, -60, 20], "block": "minecraft:bell"}]},
                    {"block_beside": {"pos": [56, -60, 20], "block": "minecraft:lantern"}}]}},
                "schematics": {"bridge_span_1": {}, "bridge_span_2": {}, "port_repair": {}}}

    @staticmethod
    def act(time, tool, args, accepted=True, duration=0, agent="builder"):
        """One narrated act record as the runtime writes it."""
        return {"gameTime": time, "agent": agent, "tool": tool, "args": args,
                "gm": {"accepted": accepted, "duration_ticks": duration}}

    def test_outcomes_need_every_span_and_a_lantern_beside_the_bell(self):
        """Rejected acts do not count, a diagonal lantern does not open the fair, and only emeralds handed to the
        builder count as its payment."""
        acts = [self.act(100, "build_section", {"schematic": "bridge_span_1"}, duration=200),
                self.act(400, "build_section", {"schematic": "bridge_span_2"}, accepted=False),
                self.act(900, "build_section", {"schematic": "bridge_span_2"}, duration=300),
                self.act(950, "place", {"x": 57, "y": -60, "z": 21, "item": "lantern"}, agent="mayor"),
                self.act(990, "give", {"target": "freight", "item": "emerald", "count": 6}, agent="mayor"),
                self.act(1000, "give", {"target": "builder", "item": "minecraft:emerald", "count": 24}, agent="mayor"),
                self.act(1100, "place", {"x": 55, "y": -60, "z": 20, "item": "minecraft:lantern"}, duration=20,
                         agent="mayor")]
        narrated = narrated_outcomes(acts, self.MANIFEST)
        self.assertEqual(narrated, {"bridge_complete": 1200, "port_repaired": None, "fair_ready": 1120,
                                    "fair_open": 1120, "builder_paid": 24})
        self.assertEqual(narrated_outcomes(acts[:2], self.MANIFEST)["bridge_complete"], None)
        divergence = outcome_divergence(narrated, {"bridge_complete": True, "fair_open": False})
        self.assertEqual(divergence["bridge_complete"]["diverged"], False)
        self.assertEqual(divergence["fair_open"], {"narrated": True, "narrated_tick": 1120, "engine": False,
                                                   "diverged": True})

    def test_a_ready_fair_opens_at_its_day_and_not_after_the_run(self):
        """With `fair_ready` and a `from_day` fair, an early lantern makes the fair ready at once and open at day 5's
        start, unless the run ends first."""
        ready = self.MANIFEST["success_predicates"]["fair_open"]
        manifest = {**self.MANIFEST, "success_predicates": {"fair_ready": ready,
                                                            "fair_open": {"from_day": 5, "world_query": ready}}}
        acts = [self.act(900, "place", {"x": 55, "y": -60, "z": 20, "item": "lantern"}, duration=20, agent="mayor")]
        self.assertEqual(narrated_outcomes(acts, manifest)["fair_ready"], 920)
        self.assertEqual(narrated_outcomes(acts, manifest)["fair_open"], 120_000)
        self.assertIsNone(narrated_outcomes(acts, manifest, end=24_000)["fair_open"])



class ServiceErrorTest(unittest.TestCase):
    """A GM call that keeps failing is no ruling: the act is refused for a service error, never judged."""

    def test_failed_gm_call_is_an_error_not_a_rejection(self):
        """The resolution is marked as an error and leaves the event log alone; scoring counts it apart."""
        def failing(system, user):
            raise RuntimeError("HTTP 520")
        gm = GameMaster(ScriptedClient(failing), "A village.", {"alice": "Beside the bell."})
        resolution = gm.resolve(0, "alice", ToolCall("speak", {"text": "hi"}))
        self.assertEqual((resolution.accepted, resolution.error, gm.errors, gm.calls), (False, True, 1, 0))
        self.assertEqual(gm._events, [])
        acts = [{"gameTime": 0, "agent": "alice", "tool": "speak", "gm": {"accepted": False, "error": True},
                 "engine_check": {"feasible": True}}]
        scored = narration_error(acts)
        self.assertEqual((scored["gm_errors"], scored["rejected"], scored["false_rejections"]), (1, 0, 0))
