"""IT-08: run E3's recorded run A twice concurrently, using threads, separate base clones, and separate ports.

Both runs must finish with identical, complete world-hash traces and identical nonempty decision traces. The second
server uses control port + 2 and game port + 3. Negative control: a second real Server using an occupied control
port must fail within five seconds with a port-conflict error and leave no Java process holding its private run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import subprocess
import threading
import time

if __package__:
    from . import harness
    from .agent_scenario import WorkshopScenario
    from .test_e3 import RUN_TICKS, E3Experiment, decision_trace
    from .test_it06_signals import cleanup_java, directory_holders, java_holders
else:
    import harness
    from agent_scenario import WorkshopScenario
    from test_e3 import RUN_TICKS, E3Experiment, decision_trace
    from test_it06_signals import cleanup_java, directory_holders, java_holders


def parallel_options(args: argparse.Namespace) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Copy each worker's arguments independently and validate the required, disjoint port pairs."""
    first, second = argparse.Namespace(**vars(args)), argparse.Namespace(**vars(args))
    first.runs_dir = second.runs_dir = args.runs_dir.resolve()
    first.game_port = args.game_port if args.game_port is not None else args.port + 1
    second.port, second.game_port = args.port + 2, args.port + 3
    ports = (first.port, first.game_port, second.port, second.game_port)
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != 4:
        raise ValueError(f"Parallel servers require four distinct valid ports: {ports}")
    return first, second


def trace_checks(hashes_a: list[dict], hashes_b: list[dict], decisions_a: list[tuple],
                 decisions_b: list[tuple]) -> dict[str, bool]:
    """Compare E3's existing traces, rejecting empty or truncated runs even when both traces match."""
    return {
        "identical complete world-hash traces": hashes_a == hashes_b and len(hashes_a) == RUN_TICKS // 20,
        "identical nonempty decision traces": bool(decisions_a) and decisions_a == decisions_b,
    }


def is_port_conflict(error: BaseException) -> bool:
    """Recognize an occupied/bound port error without accepting unrelated readiness or filesystem failures."""
    message = str(error).lower()
    return ((isinstance(error, OSError) and error.errno == errno.EADDRINUSE)
            or "address already in use" in message or "bindexception" in message
            or ("port" in message and ("in use" in message or "occupied" in message)))


class ParallelExperiment:
    """Run isolated E3 workers concurrently, then exercise the real harness's same-control-port refusal."""

    def __init__(self, args: argparse.Namespace):
        """Create isolated argument copies and paths, keeping setup, execution, and reporting separate."""
        self._options = parallel_options(args)
        self._root = self._options[0].runs_dir
        self._names = ("it08_a", "it08_b")
        self._owner = self._root / "it08_port_owner"
        self._contender = self._root / "it08_port_conflict"
        self._results = {}
        self._checks = {"both concurrent runs finished": False}
        self._completed = False

    def setup(self) -> None:
        """Reserve fresh paths only under runs-dir and clone base for the port-collision negative control."""
        base = self._root / "base"
        if not base.is_dir():
            raise FileNotFoundError(base)
        for directory in (*(self._root / name for name in self._names), self._owner, self._contender):
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        directory_holders(base)
        for directory in (self._owner, self._contender):
            harness.clone_apfs(base, directory)

    def _worker(self, name: str, args: argparse.Namespace, barrier: threading.Barrier):
        """Give each thread its own E3 experiment and scripted policy, starting both run-A calls together."""
        experiment = E3Experiment(args)
        model = WorkshopScenario().client()
        barrier.wait(timeout=5)
        return experiment.run_once(name, model, None)

    def _collision(self) -> None:
        """Attempt a real Server on an occupied control port and inspect for leaks before fallback cleanup."""
        first, second = self._options
        options = harness.server_options(second)
        options.update(port=first.port, startup_timeout=min(first.startup_timeout, 4),
                       request_timeout=min(first.request_timeout, 1), shutdown_timeout=1)
        contender = harness.Server(self._contender, **options)
        try:
            with harness.Server(self._owner, **harness.server_options(first)) as owner:
                error = None
                started = time.monotonic()
                try:
                    contender.start()
                except (OSError, harness.HarnessError) as failure:
                    error = failure
                elapsed = time.monotonic() - started
                self._checks["negative control rejects occupied control port within 5 s"] = (
                    error is not None and is_port_conflict(error) and elapsed <= 5)
                self._checks["negative control leaves no Java process behind"] = not java_holders(self._contender)
                self._checks["negative control preserves the original server"] = (
                    owner.process.poll() is None and harness.game_time(owner.client.request("ping")) >= 0)
                print(f"Negative control: control port {first.port}, {elapsed:.3f}s, error={error}")
        finally:
            try:
                contender.stop()
            finally:
                try:
                    cleanup_java(self._contender)
                finally:
                    cleanup_java(self._owner)

    def run(self) -> None:
        """Collect both thread outcomes independently and always attempt the port-collision negative control."""
        barrier = threading.Barrier(2)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(self._worker, name, args, barrier): name
                           for name, args in zip(self._names, self._options)}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        self._results[name] = future.result()
                    except Exception as error:
                        self._checks[f"{name} completed: {error}"] = False
            self._checks["both concurrent runs finished"] = len(self._results) == 2
            if len(self._results) == 2:
                first, second = (self._results[name] for name in self._names)
                self._checks["both runs used the scripted policy"] = (
                    first.stats.model_calls > 0 and second.stats.model_calls > 0)
                self._checks.update(trace_checks(first.hashes, second.hashes,
                                                decision_trace(first.log), decision_trace(second.log)))
            self._checks["concurrent runs leave no Java holders"] = all(
                not java_holders(self._root / name) for name in self._names if (self._root / name).exists())
            try:
                self._collision()
            except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
                self._checks[f"negative control completed: {error}"] = False
            self._completed = True
        finally:
            for name in self._names:
                directory = self._root / name
                if directory.is_dir():
                    cleanup_java(directory)

    def check(self) -> bool:
        """Print the independent comparisons and negative-control verdict; unfinished runs never pass."""
        for label, passed in self._checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = self._completed and all(self._checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-08: concurrent E3 traces and same-port refusal")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first and run the shared-parser experiment's setup, run, and check phases."""
    harness.stop_on_sigterm()
    try:
        experiment = ParallelExperiment(harness.experiment_parser(__doc__).parse_args(argv))
        experiment.setup()
        experiment.run()
    except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
        print(f"FAIL IT-08: {error}")
        return 1
    return 0 if experiment.check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
