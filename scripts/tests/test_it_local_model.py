"""Server-free IT-12 checks: process refusal, real prompts, reply validity, log evidence and recovery replay."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from orchestrator.decisions import DecisionLog
from orchestrator.llm import LLMResponse, ToolCall
from orchestrator.scenario import ScenarioManifest
from scripts import harness
from scripts.test_e3 import decision_trace
from scripts.test_it12_local_model import (
    MODEL_PORT, MODEL_URL, SERVER_OBSERVER, LocalModelExperiment, golden_conversation, model_processes,
    reply_value, server_observations, valid_tool,
)
from scripts.tests.fake_world import FakeWorld


def response(tool="wait", arguments=None, *, identifier="call_1", text="", count=1):
    """Build a parsed response with a real OpenAI-style raw message, without a provider or socket."""
    arguments = {"ticks": 40} if arguments is None else arguments
    function = {"name": tool, "arguments": json.dumps(arguments)}
    raw = {"choices": [{"message": {"content": text, "tool_calls": [
        {"id": identifier, "type": "function", "function": function} for unused in range(count)]}}]}
    return LLMResponse(ToolCall(tool, arguments) if count else None, text=text, raw=raw)


class RecoveringClient:
    """A hand-written failing-then-successful client for the real runtime, not a mock framework or HTTP server."""

    model = "mlx-local-unit"

    def __init__(self):
        """Begin with a dropped provider connection."""
        self.available = False

    def complete(self, *_args, **_kwargs):
        """Expose the same RuntimeError boundary as a dropped real provider, then return a short wait."""
        if not self.available:
            raise RuntimeError("connection dropped")
        return LLMResponse(ToolCall("wait", {"ticks": 40}), model=self.model)


class PairClient:
    """Return a fixed sequence of parsed replies while retaining each request's seed and actual prompt."""

    def __init__(self, replies):
        """Keep the exact reply sequence for one sampling experiment."""
        self.replies = iter(replies)
        self.requests = []

    def complete(self, system, user, tools, seed=None, history=()):
        """Record arguments and return one preconstructed reply, never making a request."""
        self.requests.append((system, user, seed, history))
        return next(self.replies)


class LocalModelTests(unittest.TestCase):
    """Test the gate's non-model logic using committed golden prompts and real file-backed runtime logs."""

    def test_process_refusal_uses_command_tokens_and_excludes_itself_and_shell_searches(self):
        """Both server forms and the launcher are refused; unrelated tools and substrings are not models."""
        processes = """11 /opt/bin/python -m mlx_lm.server --port 8080
12 /opt/bin/python /repo/scripts/mlx_agent_server.py --port 8090
13 python3 /repo/scripts/local_model.py
14 /bin/ps -axo pid=,command=
15 rg mlx_lm.server
16 zsh -c python -m mlx_lm.server
17 python test_it12_local_model.py
18 python another_local_model.py
19 python /repo/scripts/local_model.py
"""
        self.assertEqual([line.split()[0] for line in model_processes(processes, 19)], ["11", "12", "13"])
        self.assertEqual(model_processes("\nPID COMMAND\n1", 1), [])

    def test_model_is_always_on_reserved_8090(self):
        """Never probe the other agents' model port 8080."""
        self.assertEqual(MODEL_PORT, 8090)
        self.assertEqual(MODEL_URL, "http://127.0.0.1:8090/v1")

    def test_all_six_roles_use_real_golden_conversations(self):
        """The first four users, calls and outcomes come from stored history, followed by the real fifth prompt."""
        manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        for role in manifest.roles:
            with self.subTest(agent=role["name"]):
                system, history, following = golden_conversation(role["name"])
                self.assertEqual(len(history), 4)
                self.assertEqual(system, next(agent for agent in manifest.agents() if agent.name == role["name"])
                                 .system_prompt())
                stored = harness.REPO_ROOT / "scripts/tests/golden" / role["name"] / "turn1_user.txt"
                self.assertEqual(history[0].user, stored.read_text(encoding="utf-8"))
                self.assertIn(f"You are {role['name']} at", following)
                self.assertTrue(all(turn.call is not None for turn in history))

    def test_tool_validation_requires_exactly_one_usable_call(self):
        """No-call, multi-call, unknown-tool and ill-typed arguments cannot pass the concurrency smoke check."""
        self.assertTrue(valid_tool("builder", response()))
        for reply in (response(count=0), response(count=2), response(tool="unknown"),
                      response(tool="move_to", arguments={"x": "bad", "y": 0, "z": 0})):
            with self.subTest(reply=reply):
                self.assertFalse(valid_tool("builder", reply))

    def test_reply_comparison_ignores_only_request_identifiers(self):
        """Repeated request IDs cannot manufacture randomness; real text and argument changes still count."""
        self.assertEqual(reply_value(response(identifier="first")), reply_value(response(identifier="second")))
        self.assertNotEqual(reply_value(response()), reply_value(response(arguments={"ticks": 80})))
        self.assertNotEqual(reply_value(response()), reply_value(response(text="different reply")))

    def test_pairs_use_exactly_ten_unseeded_and_five_matched_seed_pairs(self):
        """The stock control can fail (a), nine varying pairs meet it, and a single seeded mismatch is visible."""
        with tempfile.TemporaryDirectory() as temporary:
            args = harness.experiment_parser("").parse_args(["--runs-dir", temporary])
            args.model = Path(temporary) / "unused-model"
            experiment = LocalModelExperiment(args)
            stock = PairClient([response()] * 20)
            self.assertEqual(experiment._pairs(stock), 0)
            self.assertEqual(len(stock.requests), 20)
            self.assertTrue(all(item[2] is None for item in stock.requests))
            self.assertEqual(len({item[:2] for item in stock.requests}), 1)
            varying = PairClient([reply for index in range(10) for reply in
                                  (response(), response(arguments={"ticks": 80 if index < 9 else 40}))])
            self.assertEqual(experiment._pairs(varying), 9)
            seeded = PairClient([response()] * 9 + [response(arguments={"ticks": 80})])
            self.assertEqual(experiment._pairs(seeded, seeded=True), 1)
            self.assertEqual([item[2] for item in seeded.requests],
                             [seed for seed in range(1, 6) for unused in range(2)])

    def test_log_evidence_is_not_confused_with_prompt_cache_sequence_counts(self):
        """Use actual observed generator caps and token overlap, not unrelated LRU cache-size log messages."""
        text = ("Prompt Cache: 9 sequences, 0.5 GB\nIT12 concurrency decode=1 prompt=1\n"
                "IT12 tokens=901 shared_prefix=0\nIT12 tokens=1100 shared_prefix=896\n")
        self.assertEqual(server_observations(text), ([(1, 1)], [(901, 0), (1100, 896)]))
        self.assertEqual(server_observations("Prompt Cache: 1 sequences"), ([], []))
        self.assertEqual(server_observations("IT12 concurrency decode=4 prompt=4")[0], [(4, 4)])
        compile(SERVER_OBSERVER, "sitecustomize.py", "exec")

    def test_killed_decisions_recover_and_replay_exactly_from_the_file_log(self):
        """The actual runtime records no-actions and resumes after recovery, then replays without a client."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = harness.experiment_parser("").parse_args(["--runs-dir", str(root), "--port", "25700"])
            args.model = root / "unused-model"
            experiment = LocalModelExperiment(args)
            experiment.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
            bots = {role["name"]: role["spawn"] for role in experiment.manifest.roles}
            session = SimpleNamespace(client=FakeWorld(bots), directory=root / "live")
            client = RecoveringClient()
            live = experiment._runtime(session, client)
            live.run(40)
            dropped = live.log.decisions()
            self.assertEqual(live.stats.model_errors, len(bots))
            self.assertEqual(len(dropped), len(bots))
            self.assertTrue(all(item.tool is None and item.outcome == "no action chosen" for item in dropped))
            client.available = True
            live.run(200)
            self.assertEqual(live.game_time(), 200)
            self.assertGreater(live.stats.model_calls, live.stats.model_errors)
            source = DecisionLog(session.directory / "orchestrator/decisions.jsonl")
            replay_session = SimpleNamespace(client=FakeWorld(bots), directory=root / "replay")
            replay = experiment._runtime(replay_session, None, source)
            replay.run(200)
            self.assertEqual(replay.stats.model_calls, 0)
            self.assertEqual(decision_trace(source), decision_trace(replay.log))
            self.assertEqual(len(live.hashes), 10)
            self.assertEqual(live.hashes, replay.hashes)


if __name__ == "__main__":
    unittest.main()
