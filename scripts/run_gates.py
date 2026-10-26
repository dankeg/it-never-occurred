"""Run a tier of integration gates with one command (plan §7): `python3.11 scripts/run_gates.py --tier fast`.

Tiers (`TIERS`): `fast` (E3, L1, N1 plumbing, IT-01 protocol, IT-04 override, IT-06 signals, IT-07 jar, IT-08 parallel
servers, IT-11 narration, IT-17 file replay and IT-18 cadence), `full` (fast plus E1, E2, E2-skills, O1, S1, M5 + V-G2
for both village variants, IT-02 load, IT-03 forks, IT-05 study integrity, IT-09 predicates, IT-10 audits, IT-15 skills,
IT-19 shutdown, the narrated village, the replay oracle, and probes, which are recorded but never fail the tier),
`model` (IT-12) and `weekly` (V1, IT-14 fixture rebuilds and IT-16 soak). `--list` prints the commands and runs nothing;
`--only E3,L1` runs some gates of a tier. IT timeouts are provisional until measured.

Before any gate, a preflight checks free memory and disk against the tier's needs, checks every fixture the gates use
(it must exist and have a `fixture.json` built from the current manifest, incident W1), and records the code state:
git commit, dirty flag (the diff saved as `code.diff`) and the harness jar's SHA-256 (P10, P11). Before every later
gate the jar is hashed again and the chain stops if it changed (P1). Each gate runs on fresh APFS clones of its
fixtures in `runs/gates_<YYYYMMDD>_<n>/<gate>/`, on its own control and game ports (from 25700-25749; a gate that runs
more servers uses the next six), with its output in `gate.log` there. `gates.json` in the same directory is
rewritten after every gate: exit code, PASS/FAIL line, wall time, artifacts and, for a failure, triage (the first
`diverged.json`, the newest `server.log` tail and the command to reproduce it from the repo root).

Exit status: 0 only if the preflight passed, no gate was skipped for a jar change, and every gating gate passed.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

if __package__:
    from . import harness
    from .provenance import CodeState, FixtureMetadata
else:
    import harness
    from provenance import CodeState, FixtureMetadata

PORTS = range(25700, 25750)  # this agent's port range on the shared machine
PORTS_PER_GATE = 8  # control and game port, and three more pairs for gates that run several servers


@dataclass(frozen=True)
class Gate:
    """Describe a gate's command, source fixtures, timeout and effect on tier success."""

    name: str
    script: str | Path
    args: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    gating: bool = True
    timeout_s: float = 1800  # generous until step 0 measures each gate's time


@dataclass(frozen=True)
class Tier:
    """Group ordered gates with required free resources and a reporting-only budget."""

    name: str
    gates: tuple[Gate, ...]
    memory_gib: float
    disk_gib: float
    budget_minutes: float


# Timeouts are about three times the time measured on 2026-10-18 (at least two minutes): E3 23 s, L1 32 s, N1
# plumbing 10 s, E1 36 s, E2 32 s, E2-skills 40 s, O1 8 s, S1 52 s, M5 + V-G2 342 s per variant, scripted narrated
# village 18 s, V1 1,740 s. Probes are unmeasured and keep the default.
# S4c estimates (unmeasured): IT-09 up to 8 min, IT-10 3 min, IT-11 1 min and IT-15 2 min; timeouts are 3x.
FAST_GATES = (
    Gate("E3", "scripts/test_e3.py", fixtures=("base",), timeout_s=120),
    Gate("L1", "scripts/test_l1.py", fixtures=("base", "_cache"), timeout_s=120),
    Gate("N1-plumbing", "scripts/test_n1_plumbing.py", fixtures=("base",), timeout_s=120),
    Gate("IT-01", "scripts/test_it01_protocol.py", fixtures=("base",), timeout_s=300),
    Gate("IT-04", "scripts/test_it04_override.py", fixtures=("base", "village_sabotage", "_cache"), timeout_s=360),
    Gate("IT-06", "scripts/test_it06_signals.py", fixtures=("base",), timeout_s=300),
    Gate("IT-07", "scripts/test_it07_jar.py", fixtures=("base",), timeout_s=300),
    Gate("IT-08", "scripts/test_it08_parallel.py", fixtures=("base",), timeout_s=300),
    Gate("IT-11", "scripts/test_it11_narration.py", fixtures=("base", "village_sabotage"), timeout_s=180),
    Gate("IT-17", "scripts/test_it17_file_replay.py", fixtures=("base",), timeout_s=300),
    Gate("IT-18", "scripts/test_it18_cadence.py", fixtures=("base",), timeout_s=300),
)

TIERS = {
    "fast": Tier("fast", FAST_GATES, 4, 5, 5),
    "full": Tier("full", FAST_GATES + (
        Gate("IT-02", "scripts/test_it02_load.py", fixtures=("base", "village_sabotage", "oracle_sabotage_1024"),
             timeout_s=1800),
        Gate("IT-03", "scripts/test_it03_forks.py", fixtures=("base", "village_sabotage", "_cache"), timeout_s=1620),
        Gate("IT-05", "scripts/test_it05_study.py", fixtures=("base", "village_sabotage", "_cache"), timeout_s=3240),
        Gate("IT-09", "scripts/test_it09_predicates.py", fixtures=("village_sabotage",), timeout_s=1440),
        Gate("IT-10", "scripts/test_it10_audits.py", fixtures=("village_sabotage",), timeout_s=540),
        Gate("IT-15", "scripts/test_it15_skills.py", fixtures=("village_sabotage",), timeout_s=360),
        Gate("IT-19", "scripts/test_it19_shutdown.py", ("--trials", "8"), ("village_sabotage",), timeout_s=600),
        Gate("E1", "scripts/test_e1.py", fixtures=("base",), timeout_s=120),
        Gate("E2", "scripts/test_e2.py", fixtures=("base", "_cache"), timeout_s=120),
        Gate("E2-skills", "scripts/test_e2_skills.py", fixtures=("base", "_cache"), timeout_s=120),
        Gate("O1", "scripts/test_o1.py", fixtures=("base",), timeout_s=120),
        Gate("S1", "scripts/test_s1.py", fixtures=("base",), timeout_s=180),
        Gate("M5-VG2-sabotage", "scripts/test_m5_village.py", ("--variant", "sabotage", "--replay"),
             ("village_sabotage",), timeout_s=1200),
        Gate("M5-VG2-accident", "scripts/test_m5_village.py", ("--variant", "accident", "--replay"),
             ("village_accident",), timeout_s=1200),
        Gate("VG5-scripted", "scripts/run_narrated_village.py", ("--scripted",), ("village_sabotage",),
             timeout_s=120),
        Gate("oracle-replay", "scripts/test_oracle_replay.py", fixtures=("village_sabotage", "oracle_sabotage_1024"),
             timeout_s=900),
        Gate("probe-ferry", "scripts/probe_ferry.py", fixtures=("base", "village_accident", "_cache"), gating=False),
        Gate("probe-village", "scripts/probe_village.py", fixtures=("village_accident",), gating=False),
    ), 8, 20, 60),
    "model": Tier("model", (
        Gate("IT-12", "scripts/test_it12_local_model.py", ("--heap", "3G"), ("village_sabotage",), timeout_s=1800),
    ), 28, 5, 10),
    # V1 alone (29 min) would take half the full tier's hour: it runs weekly, and before merging a change to
    # Villagers, VillagerNews or RoadGraph (change-to-test map), until a shorter V1 variant exists.
    "weekly": Tier("weekly", (
        Gate("V1", "scripts/test_v1.py", fixtures=("base",), timeout_s=5400),
        # Did not finish within 30 minutes on 2026-10-18 (full tier); it runs weekly and when communication changes.
        Gate("probe-communication", "scripts/probe_communication.py", fixtures=("village_sabotage",), gating=False,
             timeout_s=7200),
        Gate("IT-14", "scripts/test_it14_fixtures.py", timeout_s=14400),
        Gate("IT-16", "scripts/test_it16_soak.py", fixtures=("village_sabotage",), timeout_s=7200),
    ), 8, 40, 240),
}


class GateRunner:
    """Preflight, isolate and supervise a sequential tier while checkpointing its evidence."""

    def __init__(self, tier: Tier, *, runs_dir: Path = harness.RUNS_DIR, jar: Path = harness.MOD_JAR,
                 ports: range = PORTS, python: str = sys.executable):
        """Retain the tier and resolve source paths without creating or running anything."""
        self.tier = tier
        self.runs_dir = Path(runs_dir).resolve()
        self.jar = Path(jar).resolve()
        self.ports = ports
        self.python = python

    def _output_directory(self) -> Path:
        """Atomically reserve the first unused numbered output directory for today's date."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        date = datetime.now().astimezone().strftime("%Y%m%d")
        number = 1
        while True:
            directory = self.runs_dir / f"gates_{date}_{number}"
            try:
                directory.mkdir()
                return directory
            except FileExistsError:
                number += 1

    def _jar_sha256(self) -> str:
        """Stream the harness jar into SHA-256 without loading it into memory."""
        digest = hashlib.sha256()
        with self.jar.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _free_memory(self) -> int:
        """Measure free, inactive and speculative macOS pages using vm_stat's page size."""
        output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=15).stdout
        page_size = re.search(r"page size of (\d+) bytes", output)
        pages = dict(re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.", output, re.MULTILINE))
        if page_size is None or set(pages) != {"free", "inactive", "speculative"}:
            raise ValueError("vm_stat did not report page size and all available-page counts")
        return int(page_size.group(1)) * sum(int(count) for count in pages.values())

    def _fixture_problems(self) -> list[str]:
        """Why any fixture the tier's gates read cannot be used: missing, without `fixture.json`, or stale against its
        scenario manifest (W1). `_cache` (downloaded launchers) is not a world and only has to exist."""
        problems = []
        for fixture in dict.fromkeys(fixture for gate in self.tier.gates for fixture in gate.fixtures):
            if fixture == "_cache":
                if not (self.runs_dir / fixture).is_dir():
                    problems.append(f"{self.runs_dir / fixture} does not exist")
                continue
            manifest = harness.REPO_ROOT / "scenario" / f"{fixture}.json"
            problems += FixtureMetadata.problems(self.runs_dir / fixture, manifest if manifest.exists() else None)
        return problems

    def _preflight(self) -> tuple[dict, str | None]:
        """Collect resource measurements, fixture checks and jar identity, failing closed on missing evidence."""
        result = {"memory_free_bytes": None, "disk_free_bytes": None,
                  "memory_needed_gib": self.tier.memory_gib, "disk_needed_gib": self.tier.disk_gib,
                  "errors": self._fixture_problems()}
        try:
            result["memory_free_bytes"] = self._free_memory()
            if result["memory_free_bytes"] < self.tier.memory_gib * 1024**3:
                result["errors"].append(f"insufficient memory: need {self.tier.memory_gib:g} GiB")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result["errors"].append(f"cannot measure free memory: {error}")
        try:
            result["disk_free_bytes"] = shutil.disk_usage(self.runs_dir).free
            if result["disk_free_bytes"] < self.tier.disk_gib * 1024**3:
                result["errors"].append(f"insufficient disk: need {self.tier.disk_gib:g} GiB")
        except OSError as error:
            result["errors"].append(f"cannot measure free disk: {error}")
        jar_sha256 = None
        try:
            jar_sha256 = self._jar_sha256()
        except OSError as error:
            result["errors"].append(f"cannot hash jar {self.jar}: {error}")
        result["passed"] = not result["errors"]
        return result, jar_sha256

    def _port(self, index: int) -> int:
        """Gate `index`'s control port: blocks of `PORTS_PER_GATE` taken in turn from the port range (gates run one
        at a time, so a block is reused only after every other block)."""
        blocks = len(self.ports) // PORTS_PER_GATE
        return self.ports[(index % blocks) * PORTS_PER_GATE]

    def _command(self, gate: Gate, directory: Path, index: int) -> list[str]:
        """Build the exact repo-root command with an isolated run directory and port pair."""
        port = self._port(index)
        return [self.python, str(gate.script), *gate.args, "--runs-dir", str(directory),
                "--port", str(port), "--game-port", str(port + 1)]

    def list_gates(self) -> None:
        """Print selected gates, commands and fixtures without preflight or filesystem writes."""
        for index, gate in enumerate(self.tier.gates):
            command = self._command(gate, Path("<output>") / gate.name.lower(), index)
            print(f"{gate.name}: {shlex.join(command)}; fixtures: {', '.join(gate.fixtures) or '(none)'}")

    def _stop_group(self, process: subprocess.Popen) -> None:
        """Interrupt the started session's process group, escalating to SIGKILL after a 30-second grace period.

        SIGINT, not SIGTERM: a gate script turns it into `KeyboardInterrupt`, so `harness.Server` stops its JVM, which
        runs in a session of its own that a group signal would not reach (incident P5; C8 makes SIGTERM do the same).
        """
        deadline = time.monotonic() + 30
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            process.wait()
            return
        while True:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(min(0.05, remaining))
        process.wait()

    def _triage(self, directory: Path, command: list[str]) -> dict:
        """Retain the first divergence, newest server-log tail and shell-safe reproduce command."""
        result = {"first_divergence": None, "server_log": None, "reproduce": shlex.join(command)}
        divergences = sorted(directory.rglob("diverged.json"))
        if divergences:
            first = divergences[0]
            result["first_divergence"] = {"path": str(first.relative_to(directory)),
                                          "content": first.read_text(encoding="utf-8", errors="replace")}
        logs = list(directory.rglob("server.log"))
        if logs:
            newest = max(logs, key=lambda path: (path.stat().st_mtime_ns, str(path)))
            with newest.open(encoding="utf-8", errors="replace") as source:
                tail = "".join(deque(source, maxlen=40))
            result["server_log"] = {"path": str(newest.relative_to(directory)), "tail": tail}
        return result

    def _run_gate(self, gate: Gate, directory: Path, record: dict) -> None:
        """Clone fixtures, run one bounded process and collect its result without deleting evidence."""
        started = time.monotonic()
        directory.mkdir()
        log_path = directory / "gate.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                for fixture in gate.fixtures:
                    harness.clone_apfs(self.runs_dir / fixture, directory / fixture)
                process = subprocess.Popen(record["command"], cwd=harness.REPO_ROOT, start_new_session=True,
                                           stdout=log, stderr=subprocess.STDOUT)
                try:
                    record["exit_code"] = process.wait(timeout=gate.timeout_s)
                except subprocess.TimeoutExpired:
                    self._stop_group(process)
                    record["exit_code"] = process.returncode
                    record["reason"] = f"timeout after {gate.timeout_s:g} s"
                except KeyboardInterrupt:  # the runner was interrupted or terminated: stop the gate, then stop
                    self._stop_group(process)
                    raise
            except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
                record["reason"] = str(error)
                log.write(f"FAIL {gate.name}: {error}\n")
        record["seconds"] = time.monotonic() - started
        record["passed"] = record["exit_code"] == 0 and "reason" not in record
        with log_path.open(encoding="utf-8", errors="replace") as log:
            for line in log:
                if line.startswith(("PASS", "FAIL")):
                    record["verdict"] = line.rstrip("\r\n")
        record["artifacts"] = sorted(path.name for path in directory.iterdir()
                                     if path.name not in (*gate.fixtures, "gate.log"))
        if not record["passed"]:
            record["triage"] = self._triage(directory, record["command"])

    def _save(self, directory: Path, report: dict) -> None:
        """Atomically replace gates.json so interrupted saves preserve the last complete checkpoint."""
        temporary = directory / "gates.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(directory / "gates.json")

    def run(self) -> int:
        """Run the tier, checkpoint every result and return its gating exit status."""
        started = time.monotonic()
        started_at = datetime.now().astimezone().isoformat()
        output = self._output_directory()
        preflight, jar_sha256 = self._preflight()
        code = CodeState.capture(self.jar)
        if code.diff:
            (output / "code.diff").write_text(code.diff, encoding="utf-8")
        report = {"tier": self.tier.name, "started_at": started_at, "jar_sha256": jar_sha256, "code": code.summary(),
                  "preflight": preflight, "budget_minutes": self.tier.budget_minutes, "gates": []}
        self._save(output, report)
        skip_reason = None if preflight["passed"] else "preflight failed: " + "; ".join(preflight["errors"])
        for index, gate in enumerate(self.tier.gates):
            directory = output / gate.name.lower()
            record = {"name": gate.name, "gating": gate.gating, "command": self._command(gate, directory, index),
                      "port": self._port(index), "game_port": self._port(index) + 1,
                      "exit_code": None, "passed": False, "verdict": None, "seconds": 0, "artifacts": [],
                      "not_run": False}
            if index and skip_reason is None:
                try:
                    current_sha256 = self._jar_sha256()
                except OSError as error:
                    current_sha256 = f"unavailable ({error})"
                if current_sha256 != jar_sha256:
                    skip_reason = f"jar changed: {jar_sha256} -> {current_sha256}"
            if skip_reason is not None:
                record.update(not_run=True, reason=skip_reason)
            else:
                self._run_gate(gate, directory, record)
            report["gates"].append(record)
            self._save(output, report)
            status = "SKIP" if record["not_run"] else ("PASS" if record["passed"] else "FAIL")
            print(f"{status} {gate.name} {record['seconds']:.3f} s")
        report["total_seconds"] = time.monotonic() - started
        report["budget_met"] = report["total_seconds"] <= self.tier.budget_minutes * 60
        report["disk_free_after_bytes"] = shutil.disk_usage(self.runs_dir).free
        report["passed"] = preflight["passed"] and skip_reason is None and all(
            not record["gating"] or record["passed"] for record in report["gates"])
        self._save(output, report)
        print(output / "gates.json")
        return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Parse the tier selection and either list its commands or run its selected gates."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops the running gate, which stops its servers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--runs-dir", type=Path, default=harness.RUNS_DIR)
    parser.add_argument("--only", help="Comma-separated gate names, executed in tier order")
    parser.add_argument("--list", action="store_true", help="List commands and fixtures without running anything")
    args = parser.parse_args(argv)
    tier = TIERS[args.tier]
    if args.only is not None:
        names = {name.strip() for name in args.only.split(",")}
        unknown = names - {gate.name for gate in tier.gates}
        if unknown:
            parser.error("unknown gate names: " + ", ".join(sorted(unknown)))
        tier = replace(tier, gates=tuple(gate for gate in tier.gates if gate.name in names))
    runner = GateRunner(tier, runs_dir=args.runs_dir)
    if args.list:
        runner.list_gates()
        return 0
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
