"""IT-12: exercise the real MLX agent server on port 8090, including the former local-model smoke test.

Procedure: refuse other model processes, send ten unseeded pairs and five seeded pairs of golden agent prompts,
send four concurrent requests, and send each role three successive conversation turns, recording latency and tool
validity. Append another golden turn and require cached tokens >= 90% of the actual shared token prefix. Gate-local
logging observers report the server's real batch limits and token prefixes without changing its behaviour.
Finally, kill an in-flight mlx-local request in a scripted 200-tick village schedule, let failed decisions be
recorded, restart the model, finish, and replay the file log without a model. All decisions and hashes must match.
Negative control: the stock compiled-sampler mlx_lm.server, in .venv-mlx on the same port, must fail the unseeded test.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
from urllib import error, request

if __package__:
    from . import harness
    from .local_model import DEFAULT_MODEL, MLX_PYTHON, MlxServer
    from .run_village import model_client
    from .test_e3 import decision_trace
    from .village_session import VillageSession
else:
    import harness
    from local_model import DEFAULT_MODEL, MLX_PYTHON, MlxServer
    from run_village import model_client
    from test_e3 import decision_trace
    from village_session import VillageSession

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import LLMResponse, OpenAICompatibleClient, ToolCall, Turn  # noqa: E402
from orchestrator.runtime import THINKING_TICKS, ReplayDiverged, SimulationRuntime  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS, TOOL_NAMES, ToolTranslator  # noqa: E402

MODEL_PORT = 8090
MODEL_URL = f"http://127.0.0.1:{MODEL_PORT}/v1"
GOLDEN = Path(__file__).with_name("tests") / "golden"

# This read-only observer lives in the gate's run directory, not in production or the MLX environment.
SERVER_OBSERVER = '''"""Report the real server's batch limits and consecutive prompt-token overlap for IT-12."""
import logging
import mlx_lm.server as server

original_batch_init = server.BatchGenerator.__init__
original_tokenize = server.ResponseGenerator._tokenize
previous_prompt = []

def report_batch(self, *args, **kwargs):
    """Log the limits actually passed to the generator, without changing them."""
    logging.info("IT12 concurrency decode=%s prompt=%s",
                 kwargs.get("completion_batch_size"), kwargs.get("prefill_batch_size"))
    original_batch_init(self, *args, **kwargs)

def report_tokens(self, *args, **kwargs):
    """Log the exact common prefix of consecutive tokenized requests, without retaining model state."""
    result = original_tokenize(self, *args, **kwargs)
    prompt = result[0]
    shared = 0
    for earlier, current in zip(previous_prompt, prompt):
        if earlier != current:
            break
        shared += 1
    logging.info("IT12 tokens=%s shared_prefix=%s", len(prompt), shared)
    previous_prompt[:] = prompt
    return result

server.BatchGenerator.__init__ = report_batch
server.ResponseGenerator._tokenize = report_tokens
'''


def model_processes(output: str, own_pid: int) -> list[str]:
    """Select actual model-server command tokens, not substrings in ps, grep, or this gate's own command."""
    conflicts = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        tokens = shlex.split(parts[1])
        if Path(tokens[0]).name in {"ps", "grep", "rg", "sh", "bash", "zsh"}:
            continue
        if any(token == "mlx_lm.server" or Path(token).name in {"mlx_agent_server.py", "local_model.py"}
               for token in tokens):
            conflicts.append(line.strip())
    return conflicts


def require_idle_model() -> None:
    """Refuse any existing model process or occupied model port before allocating another 24-GB model."""
    output = subprocess.run(["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True).stdout
    conflicts = model_processes(output, os.getpid())
    if conflicts:
        raise RuntimeError("Other model processes are running: " + "; ".join(conflicts))
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", MODEL_PORT))


def golden_conversation(agent: str) -> tuple[str, list[Turn], str]:
    """Read real system, earlier turns and next-turn prompts from the committed golden fixtures."""
    directory = GOLDEN / agent
    history = [Turn(user, ToolCall(tool, json.loads(arguments)) if tool else None, result)
               for user, tool, arguments, result in json.loads(
                   (directory / "turn5_history.json").read_text(encoding="utf-8"))]
    return ((directory / "system.txt").read_text(encoding="utf-8"), history,
            (directory / "turn5_user.txt").read_text(encoding="utf-8"))


def valid_tool(agent: str, response: LLMResponse) -> bool:
    """Require exactly one known tool call with arguments accepted by the production translator."""
    calls = response.raw.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    call = response.tool_call
    if len(calls) != 1 or call is None or call.name not in TOOL_NAMES:
        return False
    try:
        ToolTranslator().translate(agent, call)
    except (ValueError, KeyError, TypeError):
        return False
    return True


def reply_value(response: LLMResponse) -> tuple:
    """Compare actual reply content and tool arguments, ignoring per-request IDs and usage timings."""
    message = response.raw["choices"][0]["message"]
    functions = [call["function"] for call in message.get("tool_calls") or []]
    return message.get("content"), json.dumps(functions, sort_keys=True, ensure_ascii=False)


def server_observations(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Read generator limits and exact token-prefix lengths from the real server's gate-local observer log."""
    limits = [(int(decode), int(prompt)) for decode, prompt in re.findall(
        r"IT12 concurrency decode=(\d+) prompt=(\d+)", text)]
    prefixes = [(int(tokens), int(shared)) for tokens, shared in re.findall(
        r"IT12 tokens=(\d+) shared_prefix=(\d+)", text)]
    return limits, prefixes


class LocalModelExperiment:
    """Own the agent server, stock-server negative control, real prompt probes and interrupted village replay."""

    def __init__(self, args):
        """Keep options without allocating a model or creating output directories."""
        self.args = args
        self.directory = args.runs_dir / "it12_model"
        self.checks = {}
        self.calls = []
        self.server = MlxServer(args.model, MODEL_PORT, log=self.directory / "agent_server.log")

    def _check(self, label: str, passed: bool) -> None:
        """Record and immediately report one positive or negative-control invariant."""
        self.checks[label] = bool(passed)
        print(f"  {'ok ' if passed else 'BAD'} {label}", flush=True)

    def _start_agent(self) -> None:
        """Start MlxServer with read-only, gate-local server logging enabled only in its child environment."""
        previous = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(self.directory / "observer") + (os.pathsep + previous if previous else "")
        try:
            self.server.start()
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous

    def _call(self, client, agent: str, user: str, history=(), seed=None) -> LLMResponse:
        """Time one actual agent request and retain the smoke test's validity, latency and token measurements."""
        system, _, _ = golden_conversation(agent)
        started = time.monotonic()
        response = client.complete(system, user, TOOL_DEFINITIONS, seed=seed, history=history)
        self.calls.append({"agent": agent, "seconds": round(time.monotonic() - started, 3),
                           "valid": valid_tool(agent, response), "tool": getattr(response.tool_call, "name", None),
                           "args": getattr(response.tool_call, "args", None),
                           "input": response.input_tokens, "cached": response.cache_read_tokens,
                           "output": response.output_tokens})
        return response

    def _pairs(self, client, seeded: bool = False) -> int:
        """Count unequal replies in ten unseeded or five identically seeded golden-prompt pairs."""
        _, history, _ = golden_conversation("builder")
        different = 0
        for index in range(5 if seeded else 10):
            seed = index + 1 if seeded else None
            first = self._call(client, "builder", history[0].user, seed=seed)
            second = self._call(client, "builder", history[0].user, seed=seed)
            different += reply_value(first) != reply_value(second)
        return different

    def _prompts(self) -> None:
        """Check sampling, deterministic seeds, concurrent tool calls and all six roles' cached conversations."""
        client = OpenAICompatibleClient(str(self.args.model), MODEL_URL, api_key="", temperature=0.7,
                                        max_tokens=384, extra={"chat_template_kwargs": {"enable_thinking": False}},
                                        timeout=600)
        different = self._pairs(client)
        self._check(f"unseeded pairs differ ({different}/10; need at least 9)", different >= 9)
        self._check("all five seeded pairs match", self._pairs(client, seeded=True) == 0)
        self._check("sampling probes return valid single tool calls", all(call["valid"] for call in self.calls))
        manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        agents = [role["name"] for role in manifest.roles]
        retries = client.retries
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self._call, client, name, golden_conversation(name)[1][0].user)
                       for name in agents[:4]]
            responses = [future.result() for future in futures]
        self._check("four concurrent requests return four valid single tool calls without retries",
                    client.retries == retries and all(valid_tool(name, response)
                                                      for name, response in zip(agents[:4], responses)))
        limits, _ = server_observations((self.directory / "agent_server.log").read_text(encoding="utf-8"))
        self._check("server reports decode and prompt concurrency one", bool(limits) and set(limits) == {(1, 1)})
        for name in agents:
            _, history, _ = golden_conversation(name)
            for index in range(3):
                response = self._call(client, name, history[index].user, history[:index])
                self._check(f"smoke: {name} turn {index + 1} is valid", valid_tool(name, response))
        _, history, following = golden_conversation("builder")
        retries = client.retries
        self._call(client, "builder", history[-1].user, history[:-1])
        response = self._call(client, "builder", following, history)
        _, prefixes = server_observations((self.directory / "agent_server.log").read_text(encoding="utf-8"))
        shared = prefixes[-1][1] if prefixes else 0
        self._check(f"appended turn reuses >=90% of its {shared}-token prefix ({response.cache_read_tokens} cached)",
                    client.retries == retries and shared > 0 and response.cache_read_tokens >= 0.9 * shared)

    def _runtime(self, session, model, source=None):
        """Build the same village runtime for the live and model-free replay phases."""
        return SimulationRuntime(session.client, self.manifest.agents(epoch_turns=8), model,
                                 session.directory / "orchestrator", command_error=harness.CommandError,
                                 source=source, hash_region=self.manifest.data["layout"]["region"], hash_every=20,
                                 model_concurrency=1, min_gap=THINKING_TICKS)

    def _interrupted_run(self) -> None:
        """Kill during a real mlx-local request, record the whole failed batch, restart, finish and replay 200 ticks."""
        args = self.args
        self.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        options = harness.server_options(args)
        options.update(game_port=args.port + 1, heap=args.heap or "3G")
        live = VillageSession(args.runs_dir / "it12_live", args.runs_dir / "village_sabotage",
                              self.manifest, options)
        live.clone()
        provider_args = argparse.Namespace(provider="mlx-local", model=str(args.model), base_url=MODEL_URL,
                                           temperature=0.7)
        with live:
            runtime = self._runtime(live, model_client(provider_args))
            log_path = self.directory / "agent_server.log"
            offset = log_path.stat().st_size
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(runtime.run, live.start + 40)
                deadline = time.monotonic() + 600
                while b"IT12 tokens=" not in log_path.read_bytes()[offset:]:
                    if future.done():
                        future.result()
                        raise RuntimeError("Village did not issue an in-flight model request")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Village model request did not reach tokenization")
                    time.sleep(0.01)
                self.server._process.kill()
                self.server._process.wait(timeout=30)
                future.result(timeout=600)
            dropped = runtime.log.decisions()
            self._check("killed requests become recorded no-action decisions",
                        runtime.stats.model_errors > 0 and len(dropped) == runtime.stats.model_errors
                        and all(item.tool is None and item.outcome == "no action chosen" for item in dropped))
            self._start_agent()
            runtime.run(live.start + 200)
            self._check("200-tick run completes after restart and records recovered tool calls",
                        runtime.game_time() == live.start + 200
                        and runtime.stats.model_calls > runtime.stats.model_errors
                        and any(item.tool is not None for item in runtime.log.decisions()))
        source = DecisionLog(live.directory / "orchestrator/decisions.jsonl")
        replay = VillageSession(args.runs_dir / "it12_replay", args.runs_dir / "village_sabotage",
                                self.manifest, options)
        replay.clone()
        with replay:
            repeated = self._runtime(replay, None, source)
            repeated.run(replay.start + 200)
        self._check("model-free replay makes zero calls and matches decisions",
                    repeated.stats.model_calls == 0 and bool(source.decisions())
                    and decision_trace(source) == decision_trace(repeated.log))
        self._check("model-free replay matches all ten world hashes",
                    len(runtime.hashes) == 10 and runtime.hashes == repeated.hashes)

    def _stock_control(self) -> None:
        """Run the real stock compiled sampler with the same prompt and require its unseeded sampling test to fail."""
        require_idle_model()
        command = [str(MLX_PYTHON), "-m", "mlx_lm.server", "--model", str(self.args.model),
                   "--port", str(MODEL_PORT), "--decode-concurrency", "1", "--prompt-concurrency", "1",
                   "--chat-template-args", json.dumps({"enable_thinking": False}), "--log-level", "INFO"]
        with (self.directory / "stock_server.log").open("ab") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Stock model server exited ({process.returncode})")
                    try:
                        with request.urlopen(MODEL_URL + "/models", timeout=2) as response:
                            json.loads(response.read())
                        break
                    except (error.URLError, TimeoutError, ConnectionError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Stock model server did not become ready")
                        time.sleep(1)
                client = OpenAICompatibleClient(str(self.args.model), MODEL_URL, api_key="", temperature=0.7,
                                                max_tokens=384,
                                                extra={"chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
                offset = len(self.calls)
                different = self._pairs(client)
                self._check("negative control: stock server returns valid replies", all(
                    call["valid"] for call in self.calls[offset:]))
                self._check(f"negative control: stock compiled sampler fails (a) ({different}/10 pairs differ)",
                            different < 9)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)

    def run(self) -> bool:
        """Refuse competing models, reserve artifacts, perform all checks and always stop owned processes."""
        require_idle_model()
        self._check("no other local model process or model-port owner", True)
        if not 1 <= self.args.port <= 65528:
            raise ValueError("IT-12 requires --port in 1..65528")
        if self.args.game_port not in (None, self.args.port + 1):
            raise ValueError("IT-12 requires game port --port + 1")
        fixture = self.args.runs_dir / "village_sabotage"
        if not fixture.is_dir() or not fixture.resolve().is_relative_to(self.args.runs_dir.resolve()):
            raise ValueError(f"Village fixture must be a directory under --runs-dir: {fixture}")
        for path in (self.directory, self.args.runs_dir / "it12_live", self.args.runs_dir / "it12_replay"):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {path}")
        (self.directory / "observer").mkdir(parents=True)
        (self.directory / "observer/sitecustomize.py").write_text(SERVER_OBSERVER, encoding="utf-8")
        try:
            self._start_agent()
            self._prompts()
            self._interrupted_run()
        finally:
            process = self.server._process
            self.server.stop()
            if process is not None:
                process.wait(timeout=30)
            (self.directory / "requests.json").write_text(json.dumps(self.calls, indent=1) + "\n", encoding="utf-8")
        self._stock_control()
        (self.directory / "requests.json").write_text(json.dumps(self.calls, indent=1) + "\n", encoding="utf-8")
        passed = all(self.checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-12: local model sampling, caching, recovery and replay")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install server cleanup before parsing the shared arguments and running the real-model experiment."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    try:
        return 0 if LocalModelExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, harness.HarnessError,
            ReplayDiverged) as failure:
        print(f"  BAD {failure}\nFAIL IT-12: local model experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
