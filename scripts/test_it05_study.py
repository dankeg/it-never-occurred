"""IT-05: study integrity with two scripted pairs, one worker, fork day 0 and end day 1 (plan §2, §4, §7).

Run an uninterrupted reference, then clone its finished factual run and fork into a separate study. SIGTERM that
study only after control 1's outcome is saved and intervention 0 is actively recording decisions. Resume it: finished
outcomes and decision logs must retain their bytes and mtimes, the unfinished arm must be renamed intact and rerun,
and the complete report, decision traces and harness hashes must equal the reference. A budget-limited study also
starts from the finished factual/fork so the budget hits a branch, not the factual run. It must record stopped,
exclude partial pairs and retain every completed decision, checked against the reference at its trajectory polls.

Every intervention must replay its own control's digest, tool and args; controls >= 1 must use only model decisions
(scripted controls have identical values; independence is checked by model-only sources). Copy control 1's log and
alter one replayed decision's args as a negative control; the same reuse check must fail. Artifacts stay under runs-dir;
all servers use the supplied eight-port block. Estimated 15-18 minutes, unmeasured; no live model is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from . import harness
    from .test_e3 import decision_trace
    from .test_it06_signals import cleanup_java, java_holders, stop_child
else:
    import harness
    from test_e3 import decision_trace
    from test_it06_signals import cleanup_java, java_holders, stop_child

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.scenario import TICKS_PER_DAY  # noqa: E402

PREFIX = "study_no_saboteur_sabotage"
FINISHED = ("factual", "control_1", "intervention_0", "intervention_1")
WAIT_SECONDS = 1200
EXIT_SECONDS = 30


def snapshot(paths: list[Path]) -> dict[Path, tuple[bytes, int]]:
    """Keep exact bytes and nanosecond mtimes of finished artifacts, without computing checksums."""
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def untouched(saved: dict[Path, tuple[bytes, int]]) -> bool:
    """Require nonempty evidence and the same files, contents and mtimes after reuse or a rename."""
    return bool(saved) and all(path.is_file() and (path.read_bytes(), path.stat().st_mtime_ns) == value
                               for path, value in saved.items())


def decision_log(path: Path) -> DecisionLog:
    """Parse every physical JSONL line and reject duplicate keys that DecisionLog would otherwise collapse."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    log = DecisionLog(path)
    if len(records) != len(log):
        raise ValueError(f"Duplicate decisions in {path}")
    return log


def reuse_matches(control: DecisionLog, intervention: DecisionLog) -> bool:
    """Require real replay coverage and exact own-control digests, tools and arguments at each agent/tick."""
    reused = [decision for decision in intervention.decisions() if decision.source == "replay"]
    if not reused or any(decision.source not in ("model", "replay") for decision in intervention.decisions()):
        return False
    for decision in reused:
        original = control.get(decision.agent, decision.game_time)
        if original is None or (decision.digest, decision.tool, decision.args) != (
                original.digest, original.tool, original.args):
            return False
    return True


def corrupt_reused_args(source: Path, intervention: DecisionLog, destination: Path) -> None:
    """Copy control 1's log, changing only the args of one decision actually replayed by intervention 1."""
    control = decision_log(source)
    if not reuse_matches(control, intervention):
        raise ValueError("Negative control requires a valid own-control replay first")
    target = next(decision for decision in intervention.decisions() if decision.source == "replay")
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        record = json.loads(line)
        if (record["agent"], record["game_time"]) == (target.agent, target.game_time):
            record["args"] = {**(record["args"] or {}), "it05_wrong_control": True}
            lines[index] = json.dumps(record, sort_keys=True) + "\n"
            destination.write_text("".join(lines), encoding="utf-8")
            return
    raise ValueError("Replayed decision missing from control log")


def interrupt_target(directory: Path) -> Path | None:
    """Wait for control 1, not the factual outcome alone, and an unfinished intervention's live runtime logs."""
    active = directory / f"{PREFIX}_intervention_0"
    logs = [active / "orchestrator" / name for name in ("decisions.jsonl", "trajectory.jsonl")]
    if ((directory / f"{PREFIX}_control_1/outcome.json").is_file()
            and not (active / "outcome.json").exists()
            and all(path.is_file() and path.stat().st_size > 0 for path in logs)):
        return active
    return None


def budget_decisions_complete(directory: Path, reference: DecisionLog) -> bool:
    """Use the reference's real readyAt/rest/memory schedule at matching actual polls, not raw observation counts."""
    output = directory / "orchestrator"
    actual = decision_log(output / "decisions.jsonl")
    polls = [json.loads(line) for line in (output / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    if not polls or not actual.decisions():
        return False
    keys = {(record["agent"], record["t"]) for record in polls}
    stop = max(record["t"] for record in polls)
    reference_polls = [json.loads(line) for line in (reference.path.parent / "trajectory.jsonl")
                       .read_text(encoding="utf-8").splitlines()]
    expected = [record for record in decision_trace(reference) if record[0] < stop]
    return (0 < stop < TICKS_PER_DAY and len(keys) == len(polls)
            and polls == [record for record in reference_polls if record["t"] <= stop]
            and any(decision.game_time == stop for decision in reference.decisions())
            and all((decision.agent, decision.game_time) in keys for decision in actual.decisions())
            and decision_trace(actual) == expected
            and len(actual) == stats["decisions"] == stats["model_calls"])


class StudyExperiment:
    """Own isolated reference, interrupted/resumed and budget studies, and report each integrity invariant."""

    def __init__(self, args: argparse.Namespace):
        """Keep the shared options and require the factual and branch servers to share one eight-port block."""
        if not 1 <= args.port <= 65528 or args.game_port not in (None, args.port + 1):
            raise ValueError("IT-05 needs eight ports starting at --port, with --game-port equal to port + 1")
        self._args = args
        self._root = args.runs_dir.resolve() / "it05_study"
        self._reference = self._root / "reference"
        self._resumed = self._root / "resumed"
        self._budget = self._root / "budget"
        self._checks: dict[str, bool] = {}
        self._completed = False

    def setup(self) -> None:
        """Refuse existing artifacts and clone all required fixtures locally, including setup_server's cache."""
        fixtures = ("base", "village_sabotage", "_cache")  # setup_server clones the game files from base (P2)
        for name in fixtures:
            source = self._args.runs_dir / name
            if not source.is_dir():
                raise FileNotFoundError(source)
        self._root.mkdir()
        for directory in (self._reference, self._resumed, self._budget):
            directory.mkdir()
            for name in fixtures:
                harness.clone_apfs(self._args.runs_dir / name, directory / name)

    def _command(self, directory: Path, max_calls: int | None = None) -> list[str]:
        """Build the actual CLI command with day-zero forks, sequential pairs and only block-local ports."""
        args = self._args
        command = [sys.executable, str(harness.REPO_ROOT / "scripts/run_study.py"),
                   "--runs-dir", str(directory), "--study", "no_saboteur", "--variant", "sabotage",
                   "--model", "scripted", "--branches", "2", "--parallel", "1", "--fork-day", "0", "--end-day", "1",
                   "--port", str(args.port), "--game-port", str(args.port + 1), "--branch-port", str(args.port),
                   "--base-seed", str(args.base_seed), "--startup-timeout", str(args.startup_timeout),
                   "--request-timeout", str(args.request_timeout)]
        if args.heap is not None:
            command += ["--heap", args.heap]
        if max_calls is not None:
            command += ["--max-calls", str(max_calls)]
        return command

    def _run_study(self, directory: Path, label: str, max_calls: int | None = None) -> None:
        """Wait for a real study subprocess, saving output and cleaning only this private run on failure."""
        output = directory / f"{label}.log"
        with output.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(self._command(directory, max_calls), cwd=harness.REPO_ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
            try:
                if process.wait(timeout=WAIT_SECONDS) != 0:
                    raise harness.HarnessError(f"Study failed; see {output}")
            finally:
                stop_child(process)
                cleanup_java(directory)

    def _seed_factual(self, directory: Path) -> None:
        """Reuse only the reference's completed factual/fork, keeping budget checks out of the factual runtime."""
        for suffix in ("factual", "fork"):
            name = f"{PREFIX}_{suffix}"
            harness.clone_apfs(self._reference / name, directory / name)

    def _interrupt_and_resume(self) -> bool:
        """SIGTERM after a finished control and live next arm, then verify reuse and intact unfinished renaming."""
        directory = self._resumed
        self._seed_factual(directory)
        with (directory / "sigterm.log").open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(self._command(directory), cwd=harness.REPO_ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + WAIT_SECONDS
                active = interrupt_target(directory)
                while active is None:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise harness.HarnessError("No finished control followed by an active unfinished intervention")
                    time.sleep(0.02)
                    active = interrupt_target(directory)
                saved = snapshot([directory / f"{PREFIX}_{suffix}" / relative
                                  for suffix in ("factual", "control_1")
                                  for relative in ("outcome.json", "orchestrator/decisions.jsonl")])
                if process.poll() is not None or (active / "outcome.json").exists():
                    raise harness.HarnessError("Study ended before SIGTERM could interrupt an unfinished branch")
                process.send_signal(signal.SIGTERM)
                try:
                    code = process.wait(timeout=EXIT_SECONDS)
                    exited = code != 0
                except subprocess.TimeoutExpired:
                    exited = False
                self._checks["SIGTERM exits nonzero within 30 seconds"] = exited
                holders = sorted(java_holders(directory))
                self._checks["SIGTERM leaves no Java holder"] = not holders
                partial = not (active / "outcome.json").exists()
                no_report = not (directory / f"{PREFIX}_report.json").exists()
                self._checks["SIGTERM truly interrupts an unfinished arm without a final report"] = (
                    partial and no_report)
                (directory / "sigterm.json").write_text(json.dumps({
                    "pid": process.pid, "returncode_before_cleanup": process.poll(), "exited_in_30s": exited,
                    "active_arm": active.name, "unfinished_before_cleanup": partial,
                    "no_report_before_cleanup": no_report, "java_holders_before_cleanup": holders,
                }, indent=1) + "\n", encoding="utf-8")
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                stop_child(process)
                cleanup_java(directory)
        self._checks["finished outcomes and decisions untouched by SIGTERM"] = untouched(saved)
        if not (exited and not holders and partial and no_report):
            self._checks["unfinished arm renamed intact and rerun to a saved outcome"] = False
            return False
        partial = snapshot([active / "orchestrator" / name for name in ("decisions.jsonl", "trajectory.jsonl")])
        self._run_study(directory, "resume")
        self._checks["finished outcomes and decisions reused byte-for-byte with original mtimes"] = untouched(saved)
        archives = list(directory.glob(f"{active.name}_unfinished_*"))
        self._checks["unfinished arm renamed intact and rerun to a saved outcome"] = (
            len(archives) == 1 and not (archives[0] / "outcome.json").exists()
            and untouched({archives[0] / path.relative_to(active): value for path, value in partial.items()})
            and (active / "outcome.json").is_file())
        return True

    def _check_pairs(self, directory: Path) -> None:
        """Check each pair's replay coupling and each independent control's model-only decision provenance."""
        for index in range(2):
            control_name = "factual" if index == 0 else f"control_{index}"
            control = decision_log(directory / f"{PREFIX}_{control_name}/orchestrator/decisions.jsonl")
            intervention = decision_log(directory / f"{PREFIX}_intervention_{index}/orchestrator/decisions.jsonl")
            self._checks[f"{directory.name} intervention {index} replays only its own control"] = (
                reuse_matches(control, intervention))
            if index >= 1:
                stats = json.loads(control.path.with_name("stats.json").read_text(encoding="utf-8"))
                self._checks[f"{directory.name} control {index} samples every decision independently"] = (
                    bool(control.decisions()) and all(decision.source == "model" for decision in control.decisions())
                    and stats["model_calls"] == stats["decisions"] == len(control) and stats["replayed"] == 0)

    def _check_budget(self) -> None:
        """Spend exactly the first control poll's calls; require a later branch budget stop and no partial effects."""
        self._seed_factual(self._budget)
        reference = decision_log(self._reference / f"{PREFIX}_control_1/orchestrator/decisions.jsonl")
        if not reference.decisions():
            raise ValueError("Budget check requires nonempty reference control decisions")
        first = reference.decisions()[0].game_time
        limit = sum(decision.game_time == first for decision in reference.decisions())
        self._run_study(self._budget, "budget", limit)
        report = json.loads((self._budget / f"{PREFIX}_report.json").read_text(encoding="utf-8"))
        branch = self._budget / f"{PREFIX}_control_1"
        self._checks["branch budget stop recorded and partial pairs excluded from effects"] = (
            bool(report["stopped"]) and f"budget {limit}" in report["stopped"] and report["pairsFinished"] == 0
            and report["effects"] == {} and report["branches_detail"] == {"control": [], "intervention": []}
            and report["model_calls"] == report["decisions"] == {"control": 0, "intervention": 0}
            and not (branch / "outcome.json").exists()
            and all(not (self._budget / f"{PREFIX}_intervention_{index}").exists() for index in range(2)))
        self._checks["budget JSONL retains every completed decision at its trajectory polls"] = (
            budget_decisions_complete(branch, reference) and len(decision_log(branch / "orchestrator/decisions.jsonl"))
            == limit)

    def run(self) -> None:
        """Run the reference, signal/resume and budget cases, then the wrong-control-args negative control."""
        self._run_study(self._reference, "reference")
        resumed = self._interrupt_and_resume()
        if resumed:
            report = f"{PREFIX}_report.json"
            self._checks["resumed final report exactly equals separate uninterrupted reference"] = (
                (self._reference / report).read_bytes() == (self._resumed / report).read_bytes()
                and json.loads((self._resumed / report).read_text(encoding="utf-8"))["pairsFinished"] == 2)
            for suffix in FINISHED:
                reference = self._reference / f"{PREFIX}_{suffix}/orchestrator"
                output = self._resumed / f"{PREFIX}_{suffix}/orchestrator"
                hashes = json.loads((reference / "hashes.json").read_text(encoding="utf-8"))
                self._checks[f"resumed {suffix} matches reference decisions and nonempty harness hashes"] = (
                    bool(hashes) and hashes == json.loads((output / "hashes.json").read_text(encoding="utf-8"))
                    and decision_trace(decision_log(reference / "decisions.jsonl"))
                    == decision_trace(decision_log(output / "decisions.jsonl")))
            self._check_pairs(self._resumed)
        self._check_pairs(self._reference)
        source = self._resumed if resumed else self._reference
        control = source / f"{PREFIX}_control_1/orchestrator/decisions.jsonl"
        intervention = decision_log(source / f"{PREFIX}_intervention_1/orchestrator/decisions.jsonl")
        damaged = self._root / "wrong_control_1.jsonl"
        before = snapshot([control])
        corrupt_reused_args(control, intervention, damaged)
        self._checks["negative control: changed control 1 args fail the same reuse check"] = (
            untouched(before) and not reuse_matches(decision_log(damaged), intervention))
        self._check_budget()
        self._completed = True

    def check(self) -> bool:
        """Print ok/BAD per invariant and PASS only after all cases, including the negative control, finish."""
        for label, passed in self._checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = self._completed and bool(self._checks) and all(self._checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-05: paired study integrity, resume and budget persistence")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install SIGTERM handling first and run the single experiment with the shared gate argument parser."""
    harness.stop_on_sigterm()
    experiment = StudyExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    try:
        experiment.setup()
        experiment.run()
    except (OSError, ValueError, KeyError, TypeError, harness.HarnessError, subprocess.SubprocessError) as error:
        experiment._checks[f"experiment completed ({error})"] = False  # print the checks made before it stopped
    return 0 if experiment.check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
