"""IT-06: interrupt E3 with SIGINT and SIGTERM after readiness and seeded delays, each on a fresh base clone.

Each subprocess must exit within 30 seconds, leave no Java holder of its run directory or server.log, and remain
incomplete (nonzero exit and no PASS line). A separate server must stop within 30 seconds after a 24,000-tick burst
(D2). Negative control: SIGTERM a stand-in that owns a real Server but omits stop_on_sigterm; detect its orphaned
Java process before terminating only Java processes still holding that negative control's private directory.
Revert check (2026-10-18): fails at a0b6f5c^ (E3 without stop_on_sigterm leaves its JVM).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

if __package__:
    from . import harness
else:
    import harness


EXIT_TIMEOUT = 30.0
JVM_START = "--- harness JVM start ---"
ORPHAN_STANDIN = """
import json
from pathlib import Path
import sys
import time
from scripts import harness

directory = Path(sys.argv[1])
with harness.Server(directory, **json.loads(sys.argv[2])) as server:
    receipt = directory / 'java.pid.tmp'
    receipt.write_text(str(server.process.pid), encoding='utf-8')
    receipt.replace(directory / 'java.pid')
    while True:
        time.sleep(1)
"""


def lsof_processes(output: str) -> dict[int, str]:
    """Parse lsof's untruncated process/command fields without confusing filenames with PIDs."""
    processes = {}
    current = None
    for line in output.splitlines():
        if line.startswith("p"):
            current = int(line[1:]) if line[1:].isdigit() and int(line[1:]) > 0 else None
        elif line.startswith("c") and current is not None:
            processes[current] = line[1:]
    return processes


def is_java(command: str) -> bool:
    """Recognize the Java executable, not a Python command line mentioning Java or a Java-named file."""
    return Path(command).name == "java"


def directory_holders(directory: Path) -> dict[int, str]:
    """Find open files and cwd holders recursively, explicitly including server.log; fail closed on lsof errors."""
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    logs = sorted(directory.rglob("server.log"))
    result = subprocess.run(["lsof", "-nP", "-Fpc", "+D", str(directory), *(str(path) for path in logs)],
                            capture_output=True, text=True, timeout=10)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise harness.HarnessError(f"Cannot inspect holders of {directory}: {result.stderr.strip() or result.stdout}")
    if result.stdout.strip() and not lsof_processes(result.stdout):
        raise harness.HarnessError(f"Unrecognized lsof process output for {directory}: {result.stdout!r}")
    return lsof_processes(result.stdout)


def java_holders(directory: Path) -> set[int]:
    """Return only actual Java processes holding this run, including a JVM with just server.log open."""
    return {pid for pid, command in directory_holders(directory).items() if is_java(command)}


def cleanup_java(directory: Path) -> None:
    """Revalidate directory ownership before each signal, escalating only this run's observed Java PIDs."""
    owned = java_holders(directory)
    if not owned:
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(owned & java_holders(directory)):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while owned & java_holders(directory):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        else:
            return
    remaining = owned & java_holders(directory)
    if remaining:
        raise harness.HarnessError(f"Java cleanup failed for {directory}: {sorted(remaining)}")


def stop_child(process: subprocess.Popen) -> None:
    """Interrupt only the child we started, then reap it, killing it if its cleanup exceeds the grace period."""
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=5)


def server_log_ready(path: Path, offset: int = 0) -> bool:
    """Require readiness in the latest JVM start, never in the base clone's inherited log history."""
    if not path.is_file():
        return False
    with path.open("rb") as source:
        source.seek(offset)
        text = source.read().decode("utf-8", errors="replace")
    _, marker, latest = text.rpartition(JVM_START)
    return bool(marker and "Done (" in latest and "SimHarness control listening on 127.0.0.1:" in latest)


def wait_until_ready(process: subprocess.Popen, log: Path, timeout: float, *, offset: int = 0,
                     pid_file: Path | None = None) -> None:
    """Wait on this child's log and optional startup receipt, rejecting an early exit or missed deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise harness.HarnessError(f"Child exited ({process.returncode}) before readiness; see {log}")
        if server_log_ready(log, offset) and (pid_file is None or pid_file.is_file()):
            return
        time.sleep(0.02)
    raise harness.HarnessError(f"Child readiness timed out after {timeout:g}s; see {log}")


def run_incomplete(returncode: int | None, output: str) -> bool:
    """Require a reaped nonzero exit and no PASS verdict, not merely an absent completion artifact."""
    return returncode not in (None, 0) and re.search(r"^\s*PASS\b", output, flags=re.MULTILINE) is None


class SignalExperiment:
    """Exercise both E3 interrupt paths, burst shutdown, and an intentionally unhandled SIGTERM orphan."""

    def __init__(self, args: argparse.Namespace):
        """Keep private run paths, seeded interrupt delays, and check results without starting anything."""
        self._args = argparse.Namespace(**vars(args))
        self._args.runs_dir = args.runs_dir.resolve()
        self._args.game_port = args.game_port if args.game_port is not None else args.port + 1
        self._directories = {name: self._args.runs_dir / f"it06_{name}"
                             for name in ("sigint", "sigterm", "burst", "orphan")}
        generator = random.Random(args.base_seed)
        self._delays = [generator.uniform(0.05, 0.5) for _ in range(2)]
        self._checks = {}
        self._completed = False

    def setup(self) -> None:
        """Refuse existing destinations before cloning base into each isolated case under runs-dir."""
        base = self._args.runs_dir / "base"
        if not base.is_dir():
            raise FileNotFoundError(base)
        for directory in self._directories.values():
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        directory_holders(base)
        for name, directory in self._directories.items():
            if name in ("sigint", "sigterm"):
                directory.mkdir()
                harness.clone_apfs(base, directory / "base")
            else:
                harness.clone_apfs(base, directory)

    def _interrupt(self, name: str, signum: signal.Signals, delay: float) -> None:
        """Interrupt a ready E3 subprocess, record evidence before cleanup, and preserve its output."""
        directory = self._directories[name]
        output = directory / "e3.log"
        command = [sys.executable, str(harness.REPO_ROOT / "scripts/test_e3.py"),
                   "--runs-dir", str(directory), "--port", str(self._args.port),
                   "--game-port", str(self._args.game_port or self._args.port + 1),
                   "--base-seed", str(self._args.base_seed), "--startup-timeout", str(self._args.startup_timeout),
                   "--request-timeout", str(self._args.request_timeout)]
        if self._args.heap is not None:
            command += ["--heap", self._args.heap]
        process = None
        try:
            inherited_log = directory / "base/server.log"
            offset = inherited_log.stat().st_size if inherited_log.is_file() else 0
            with output.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=harness.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                wait_until_ready(process, directory / "e3_a/server.log", self._args.startup_timeout, offset=offset)
                time.sleep(delay)
                if process.poll() is not None:
                    raise harness.HarnessError(f"E3 exited before {signum.name} at delay {delay:.3f}s")
                started = time.monotonic()
                process.send_signal(signum)
                try:
                    process.wait(timeout=EXIT_TIMEOUT)
                    self._checks[f"{signum.name} exits within 30 s"] = time.monotonic() - started <= EXIT_TIMEOUT
                except subprocess.TimeoutExpired:
                    self._checks[f"{signum.name} exits within 30 s"] = False
                    process.kill()
                    process.wait(timeout=5)
            self._checks[f"{signum.name} leaves no Java holders"] = not java_holders(directory)
            self._checks[f"{signum.name} run is incomplete"] = run_incomplete(
                process.returncode, output.read_text(encoding="utf-8", errors="replace"))
            print(f"{signum.name}: delay={delay:.3f}s, exit={process.returncode}; {output}")
        finally:
            try:
                if process is not None:
                    stop_child(process)
            finally:
                cleanup_java(directory)

    def _burst(self) -> None:
        """Advance 24,000 lockstep ticks in one request and measure a successful, orphan-free stop."""
        directory = self._directories["burst"]
        server = harness.Server(directory, shutdown_timeout=EXIT_TIMEOUT, **harness.server_options(self._args))
        try:
            server.start()
            before = harness.enable_lockstep(server.client, recording=False)
            after = harness.game_time(server.client.request("step", {"n": 24_000}))
            self._checks["burst advances 24,000 ticks"] = after == before + 24_000
            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started
            self._checks["burst stops cleanly within 30 s"] = elapsed <= EXIT_TIMEOUT
            self._checks["burst leaves no Java holders"] = not java_holders(directory)
            print(f"Burst stop: {elapsed:.3f}s")
        finally:
            try:
                server.stop()
            finally:
                cleanup_java(directory)

    def _orphan(self) -> None:
        """Prove a real Java orphan is detected after unhandled SIGTERM, then clean that private run."""
        directory = self._directories["orphan"]
        process = None
        try:
            server_log = directory / "server.log"
            offset = server_log.stat().st_size if server_log.is_file() else 0
            with (directory / "standin.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-c", ORPHAN_STANDIN, str(directory),
                     json.dumps(harness.server_options(self._args))], cwd=harness.REPO_ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt = directory / "java.pid"
                wait_until_ready(process, server_log, self._args.startup_timeout, offset=offset, pid_file=receipt)
                java_pid = int(receipt.read_text(encoding="utf-8"))
                if java_pid not in java_holders(directory):
                    raise harness.HarnessError("Negative control's ready Java PID was not observed by lsof")
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)
                holders = java_holders(directory)
                self._checks["negative control detects the SIGTERM orphan"] = (
                    process.returncode == -signal.SIGTERM and java_pid in holders)
                print(f"Negative control: exit={process.returncode}, Java holders={sorted(holders)}")
        finally:
            try:
                if process is not None:
                    stop_child(process)
            finally:
                cleanup_java(directory)
        self._checks["negative control orphan is cleaned up"] = not java_holders(directory)

    def run(self) -> None:
        """Run the independent cases sequentially, retaining failure evidence while always attempting cleanup."""
        for name, signum, delay in zip(("sigint", "sigterm"), (signal.SIGINT, signal.SIGTERM), self._delays):
            try:
                self._interrupt(name, signum, delay)
            except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
                self._checks[f"{signum.name} case completed: {error}"] = False
        for name, action in (("burst", self._burst), ("negative control", self._orphan)):
            try:
                action()
            except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
                self._checks[f"{name} case completed: {error}"] = False
        self._completed = True

    def check(self) -> bool:
        """Print each invariant and the negative control separately, passing only a completed successful experiment."""
        for label, passed in self._checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = self._completed and bool(self._checks) and all(self._checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-06: signals, burst shutdown, and orphan detection")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse the shared gate options, and run the three experiment phases."""
    harness.stop_on_sigterm()
    experiment = SignalExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    try:
        experiment.setup()
        experiment.run()
    except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
        print(f"FAIL IT-06: {error}")
        return 1
    return 0 if experiment.check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
