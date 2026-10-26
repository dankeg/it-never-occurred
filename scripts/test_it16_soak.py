"""IT-16: run the scripted village for five days, then replay its file log without a model.

Procedure: clone runs-dir/village_<variant> twice, run M5's VillageScriptedPolicy for --days Minecraft days,
and replay the recorded decisions on the second clone. Sample each JVM's RSS immediately and every 60 wall seconds,
including startup; retain samples and a summary under --runs-dir. Both servers use the same reserved ports in turn.
Pass: both runs finish and shut down without harness failure; the replay makes zero model calls and exactly matches
nonempty decisions and every expected 1,200-tick world hash. Each run's sampled peak RSS and final logical file size
must stay within the estimated constants below. The bounds apply separately to both clones, not to all runs-dir.
Negative control: none specified for IT-16 in the plan. Server-free unit tests change traces and exceed the bounds;
no production fault switch is introduced. Estimated default time: about 15 minutes, extrapolated from M5, unmeasured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

if __package__:
    from . import harness
    from .test_e3 import decision_trace
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import CHECK_EVERY
else:
    import harness
    from test_e3 import decision_trace
    from test_m5_village import VillageScriptedPolicy
    from village_session import CHECK_EVERY

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient  # noqa: E402
from orchestrator.runtime import SimulationRuntime  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402

DAY_TICKS = 24_000
RSS_SAMPLE_SECONDS = 60.0
MAX_RSS_BYTES = 6 * 1024**3  # Estimate: the default 3 GiB heap plus 3 GiB for JVM/native/chunk overhead.
MAX_RUN_BYTES = 8 * 1024**3  # Estimate per clone: ~1 GiB fixture plus several GiB of five-day events and trajectories.


def positive_days(value: str) -> int:
    """Parse a positive whole number of Minecraft days before any run can start."""
    days = int(value)
    if days <= 0:
        raise argparse.ArgumentTypeError("--days must be positive")
    return days


def process_rss_bytes(pid: int) -> int:
    """Read only the owned process's RSS with macOS/Linux ps, whose rss column is in KiB."""
    if pid <= 0:
        raise ValueError("RSS sampling requires a positive PID")
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], check=True, capture_output=True,
                            text=True, timeout=5)
    return parse_rss_bytes(result.stdout)


def parse_rss_bytes(output: str) -> int:
    """Convert a single positive ps RSS value from KiB, rejecting missing or malformed measurements."""
    rss = int(output.strip()) * 1024
    if rss <= 0:
        raise ValueError(f"No positive RSS measurement: {output!r}")
    return rss


def directory_bytes(directory: Path) -> int:
    """Count logical regular-file bytes in this clone, without following symlinks outside it."""
    return sum(path.stat().st_size for path in directory.rglob("*") if not path.is_symlink() and path.is_file())


class SoakExperiment:
    """Own two sequential village runs, their PID-scoped memory sampling, and the IT-16 verdict."""

    def __init__(self, args: argparse.Namespace):
        """Retain CLI options and empty evidence without creating runs or starting processes."""
        self.args = args
        self.root = args.runs_dir.resolve()
        self.names = ("it16_recorded", "it16_replay")
        self.reports: dict[str, dict] = {}
        self.runtimes: dict[str, SimulationRuntime] = {}
        self.error: str | None = None

    def setup(self) -> None:
        """Validate days, reserved ports, local fixture and nonexisting destinations before cloning."""
        if self.args.days <= 0:
            raise ValueError("--days must be positive")
        port = self.args.port
        game_port = self.args.game_port if self.args.game_port is not None else port + 1
        if not (1 <= port <= 65535 and port < game_port <= min(port + 7, 65535)):
            raise ValueError("IT-16 requires a control port and distinct game port within --port + 0..7")
        self.options = {**harness.server_options(self.args), "game_port": game_port}
        self.fixture = self.root / f"village_{self.args.variant}"
        if not self.fixture.resolve().is_relative_to(self.root) or not self.fixture.is_dir():
            raise ValueError(f"Village fixture must be a directory under --runs-dir: {self.fixture}")
        for name in (*self.names, "it16_soak.json"):
            path = self.root / name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {path}")
        self.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario" / f"village_{self.args.variant}.json")

    def _sample_memory(self, server: harness.Server, stopped: threading.Event, report: dict) -> None:
        """Sample the owned JVM from its creation every minute, recording failures rather than losing a thread error."""
        started = time.monotonic()
        try:
            while server.process is None:
                if stopped.wait(0.05):
                    return
            with (Path(report["directory"]) / "it16_rss.jsonl").open("w", encoding="utf-8") as handle:
                while not stopped.is_set():
                    sample = {"seconds": round(time.monotonic() - started, 3), "pid": server.process.pid,
                              "rss_bytes": process_rss_bytes(server.process.pid)}
                    report["rss_samples"].append(sample)
                    handle.write(json.dumps(sample) + "\n")
                    handle.flush()
                    if stopped.wait(RSS_SAMPLE_SECONDS):
                        break
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["rss_errors"].append(f"{type(error).__name__}: {error}")

    def run_once(self, name: str, source: DecisionLog | None) -> None:
        """Clone the fixture, perform M5's fresh-village setup, and run the full requested duration before shutdown."""
        directory = self.root / name
        report = {"directory": str(directory), "completed": False, "start": None, "end": None,
                  "rss_samples": [], "rss_errors": [], "bytes": None}
        self.reports[name] = report
        harness.clone_apfs(self.fixture, directory)
        server = harness.Server(directory, **self.options)
        stopped = threading.Event()
        monitor = threading.Thread(target=self._sample_memory, args=(server, stopped, report), daemon=True)
        monitor.start()
        try:
            with server:
                client = server.client
                report["start"] = harness.enable_lockstep(client, recording=True)
                client.request("restore_bots")
                client.request("config_set", {"exposure_log": 0})
                client.request("structures_set", {"structures": self.manifest.structures()})
                model = ScriptedClient(VillageScriptedPolicy(self.manifest).decide) if source is None else None
                runtime = SimulationRuntime(client, self.manifest.agents(1), model, directory / "orchestrator",
                                            command_error=harness.CommandError, source=source,
                                            hash_region=self.manifest.data["layout"]["region"], hash_every=CHECK_EVERY)
                self.runtimes[name] = runtime
                runtime.run(report["start"] + self.args.days * DAY_TICKS)
                report["end"] = harness.game_time(client.request("ping"))
                client.request("save_snapshot")
                stopped.set()
                monitor.join()
            report["completed"] = True
        finally:
            stopped.set()
            monitor.join()
            report["bytes"] = directory_bytes(directory)
            report["peak_rss_bytes"] = max((sample["rss_bytes"] for sample in report["rss_samples"]), default=None)

    def check(self) -> dict[str, bool]:
        """Compare recorded evidence without hashing content or accepting empty or incomplete matching traces."""
        checks = {}
        for name in self.names:
            report = self.reports.get(name, {})
            samples = report.get("rss_samples", [])
            checks[f"{name}: full duration and clean harness shutdown"] = (
                report.get("completed", False) and report.get("start") is not None
                and report.get("end") == report["start"] + self.args.days * DAY_TICKS)
            checks[f"{name}: JVM RSS sampled successfully every minute"] = (
                bool(samples) and not report.get("rss_errors"))
            checks[f"{name}: sampled peak RSS <= {MAX_RSS_BYTES} bytes"] = (
                bool(samples) and 0 < max(sample["rss_bytes"] for sample in samples) <= MAX_RSS_BYTES)
            checks[f"{name}: run directory <= {MAX_RUN_BYTES} bytes"] = (
                report.get("bytes") is not None and 0 < report["bytes"] <= MAX_RUN_BYTES)
        recorded, replayed = (self.runtimes.get(name) for name in self.names)
        checks["scripted policy recorded nonempty decisions"] = recorded is not None and recorded.stats.model_calls > 0
        checks["replay made zero model calls and reused every decision"] = (
            replayed is not None and replayed.stats.model_calls == 0
            and 0 < replayed.stats.replayed == replayed.stats.decisions == len(replayed.log))
        checks["identical nonempty decision traces"] = (
            recorded is not None and replayed is not None and len(recorded.log) > 0
            and decision_trace(recorded.log) == decision_trace(replayed.log))
        first = self.reports.get(self.names[0], {})
        start, end = first.get("start"), first.get("end")
        checks["identical complete world-hash traces"] = (
            recorded is not None and replayed is not None and start is not None and end is not None
            and bool(recorded.hashes) and recorded.hashes == replayed.hashes
            and [sample.get("gameTime") for sample in recorded.hashes]
            == list(range((start // CHECK_EVERY + 1) * CHECK_EVERY, end + 1, CHECK_EVERY)))
        checks["both clones started at the same game time"] = (
            start is not None and start == self.reports.get(self.names[1], {}).get("start"))
        checks["no harness, replay, sampling or filesystem failure"] = (
            self.error is None and all(not report.get("rss_errors") for report in self.reports.values()))
        return checks

    def run(self) -> bool:
        """Run the scripted soak and model-free file replay, persist evidence, and print every check and verdict."""
        ready = False
        try:
            self.setup()
            ready = True
            self.run_once(self.names[0], None)
            source = DecisionLog(self.root / self.names[0] / "orchestrator" / "decisions.jsonl")
            self.run_once(self.names[1], source)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            self.error = f"{type(error).__name__}: {error}"
        checks = self.check()
        if ready:
            summary = {"days": self.args.days, "rss_interval_seconds": RSS_SAMPLE_SECONDS,
                       "max_rss_bytes": MAX_RSS_BYTES, "max_run_bytes": MAX_RUN_BYTES,
                       "runs": self.reports, "checks": checks, "error": self.error}
            try:
                (self.root / "it16_soak.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            except OSError as error:
                self.error = f"{type(error).__name__}: {error}"
                checks["no harness, replay, sampling or filesystem failure"] = False
        if self.error:
            print(self.error)
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-16: {self.args.days}-day scripted village soak and model-free replay")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse the shared gate options and run only this experiment."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.set_defaults(heap="3G")
    parser.add_argument("--days", type=positive_days, default=5, help="positive Minecraft days to run (default: 5)")
    parser.add_argument("--variant", choices=("sabotage", "accident"), default="sabotage")
    return 0 if SoakExperiment(parser.parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
