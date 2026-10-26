"""IT-02: reuse OracleReplay while E1 continuously loads a second JVM on control/game ports +2/+3.

Procedure: repeat the existing E1 experiment on fresh base clones until the existing oracle replay finishes;
--model-load additionally serves real golden agent prompts on port 8090, targeting one request every two seconds.
Pass: OracleReplay's 100% identical hashes and decisions with zero model calls, and every completed E1 comparison
passes. The load workers must actually run during the replay. Their errors fail the gate, and all workers and model
processes are joined/stopped on exit. Negative control: disabling the chunk-spawn-order mixin needs a production
switch; no such switch is added. The owner runs that control by reverting the mixin on a scratch branch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

if __package__:
    from . import harness
    from .local_model import DEFAULT_MODEL, MlxServer
    from .test_e1 import E1Experiment
    from .test_it12_local_model import MODEL_PORT, MODEL_URL, golden_conversation, require_idle_model, valid_tool
    from .test_oracle_replay import OracleReplay
else:
    import harness
    from local_model import DEFAULT_MODEL, MlxServer
    from test_e1 import E1Experiment
    from test_it12_local_model import MODEL_PORT, MODEL_URL, golden_conversation, require_idle_model, valid_tool
    from test_oracle_replay import OracleReplay

from orchestrator.llm import OpenAICompatibleClient  # noqa: E402
from orchestrator.runtime import ReplayDiverged  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS  # noqa: E402


def load_options(args, directory) -> argparse.Namespace:
    """Copy E1's options with an isolated run root and only the required +2/+3 load ports."""
    options = argparse.Namespace(**vars(args))
    options.runs_dir = directory
    options.port, options.game_port = args.port + 2, args.port + 3
    return options


class LoadExperiment:
    """Run the existing replay oracle against continuous E1 load and optional real-model load."""

    def __init__(self, args):
        """Keep options, cancellation events and worker evidence without starting anything."""
        self.args = args
        self.root = args.runs_dir / "it02_load"
        self.stop = threading.Event()
        self.e1_ready = threading.Event()
        self.model_ready = threading.Event()
        self.e1_results = []
        self.model_requests = []

    def _e1(self) -> None:
        """Keep completing unmodified E1 comparisons on independent clones until the replay requests stop."""
        while not self.stop.is_set():
            directory = self.root / f"e1_{len(self.e1_results) + 1}"
            directory.mkdir()
            harness.clone_apfs(self.args.runs_dir / "base", directory / "base")
            experiment = E1Experiment(load_options(self.args, directory))
            experiment.setup()
            self.e1_ready.set()
            started = time.monotonic()
            experiment.run()
            experiment.check()
            passed = experiment.report() == 0
            self.e1_results.append({"start": started, "end": time.monotonic(), "passed": passed,
                                    "directory": str(directory)})
            if not passed:
                raise RuntimeError("E1 failed under replay load")

    def _model(self) -> None:
        """Answer one real agent request at a time, targeting two-second start intervals without catch-up bursts."""
        system, history, _ = golden_conversation("builder")
        client = OpenAICompatibleClient(str(DEFAULT_MODEL), MODEL_URL, api_key="", temperature=0.7, max_tokens=384,
                                        extra={"chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
        while not self.stop.is_set():
            started = time.monotonic()
            response = client.complete(system, history[0].user, TOOL_DEFINITIONS)
            valid = valid_tool("builder", response)
            self.model_requests.append({"start": started, "end": time.monotonic(), "valid": valid})
            if not valid:
                raise RuntimeError("Model load returned an invalid tool call")
            self.model_ready.set()
            self.stop.wait(max(0, started + 2 - time.monotonic()))

    def _wait_ready(self, event, future, timeout: float) -> None:
        """Wait for real load-worker progress, propagating failures instead of hanging at a startup barrier."""
        deadline = time.monotonic() + timeout
        while not event.wait(0.05):
            if future.done():
                future.result()
                raise RuntimeError("Load worker exited before becoming ready")
            if time.monotonic() >= deadline:
                raise TimeoutError("Load worker did not become ready")

    def run(self) -> bool:
        """Run OracleReplay unchanged, join load experiments, persist evidence and report each invariant."""
        args = self.args
        if not 1 <= args.port <= 65528:
            raise ValueError("IT-02 requires --port in 1..65528")
        if args.game_port not in (None, args.port + 1):
            raise ValueError("IT-02 requires replay game port --port + 1")
        if not args.oracle or Path(args.oracle).name != args.oracle or args.oracle in {".", ".."}:
            raise ValueError("IT-02 requires --oracle to be one fixture directory name under --runs-dir")
        root = args.runs_dir.resolve()
        for name in ("base", f"village_{args.variant}", args.oracle):
            fixture = root / name
            if not fixture.is_dir() or not fixture.resolve().is_relative_to(root):
                raise ValueError(f"Fixture must be a directory under --runs-dir: {fixture}")
        args.game_port = args.port + 1
        if args.model_load:
            require_idle_model()
            args.heap = args.heap or "3G"
        self.root.mkdir()
        server = MlxServer(DEFAULT_MODEL, MODEL_PORT, log=self.root / "model_server.log") if args.model_load else None
        replay_passed = False
        futures = []
        try:
            if server is not None:
                server.start()
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    futures.append(pool.submit(self._e1))
                    self._wait_ready(self.e1_ready, futures[0], args.startup_timeout)
                    if args.model_load:
                        futures.append(pool.submit(self._model))
                        self._wait_ready(self.model_ready, futures[-1], 600)
                    for future in futures:
                        if future.done():
                            future.result()
                    started = time.monotonic()
                    replay_passed = OracleReplay(args).run()
                    ended = time.monotonic()
                finally:
                    self.stop.set()
                for future in futures:
                    future.result()
        finally:
            self.stop.set()
            active_error = sys.exc_info()[1]
            cleanup_errors = []
            try:
                if server is not None:
                    process = server._process
                    try:
                        server.stop()
                    except (OSError, RuntimeError, subprocess.SubprocessError) as failure:
                        cleanup_errors.append(str(failure))
                    if process is not None:
                        try:
                            process.wait(timeout=30)
                        except (OSError, RuntimeError, subprocess.SubprocessError) as failure:
                            cleanup_errors.append(str(failure))
            finally:
                (self.root / "load.json").write_text(json.dumps({"e1": self.e1_results,
                                                               "model": self.model_requests,
                                                               "cleanup_errors": cleanup_errors}, indent=1) + "\n",
                                                   encoding="utf-8")
            if cleanup_errors and active_error is None:
                raise RuntimeError("Model cleanup failed: " + "; ".join(cleanup_errors))
        checks = {
            "oracle: 100% identical hashes and decisions, zero replay model calls": replay_passed,
            "E1 passes under replay load": bool(self.e1_results) and all(item["passed"] for item in self.e1_results),
            "E1 load overlaps the oracle replay": any(item["start"] < ended and item["end"] > started
                                                     for item in self.e1_results),
        }
        if args.model_load:
            checks["valid model requests overlap the replay"] = any(
                item["valid"] and item["start"] < ended and item["end"] > started for item in self.model_requests)
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        print("  negative control deferred: owner reverts ChunkSpawnOrderMixin on a scratch branch (no switch added)")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-02: oracle determinism under E1 and optional model load")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse shared options and the oracle's existing defaults, then run the experiment."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--model-load", action="store_true", help="also load the local model on port 8090")
    parser.add_argument("--oracle", default="oracle_sabotage_1024")
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--end", type=int, default=48_000)
    parser.add_argument("--epoch-turns", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        return 0 if LoadExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, harness.HarnessError,
            ReplayDiverged) as failure:
        print(f"  BAD {failure}\nFAIL IT-02: load experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
