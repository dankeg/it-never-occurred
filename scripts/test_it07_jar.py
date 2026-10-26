"""IT-07: jar consistency across E3 starts, a gate chain, and a study branch (plan §2 P1, §4 and §7).

Procedure: run the full E3 recorded run and replay on the supplied base fixture. Inspect each run's latest provenance
start against the current harness jar and HEAD. Run a two-gate GateRunner chain in a run-local temporary directory,
with jar= a copy of the harness jar that the first stand-in rewrites. Finally, try a study branch whose synthetic
factual run records a different jar; its fork deliberately does not exist, so it cannot clone or start a server.

Pass: E3 passes, both runs append current jar/commit provenance, the stand-in chain's first gate passes and its second
is not_run with a 'jar changed' reason, and Study.branch raises a clear HarnessError before touching the branch.
Negative controls: the copied-jar rewrite must stop the next gate, and the mismatched factual jar must stop cloning.
All artifacts stay under --runs-dir; only the supplied base fixture is cloned, and no setup_server call is needed.
Revert check (2026-10-18): fails at fc291dd^ (no jar re-hash between gates).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

if __package__:
    from . import harness, provenance
    from .run_gates import Gate, GateRunner, Tier
    from .run_study import Study
    from .test_e3 import E3Experiment
else:
    import harness
    import provenance
    from run_gates import Gate, GateRunner, Tier
    from run_study import Study
    from test_e3 import E3Experiment


class JarExperiment:
    """Run E3 and two jar-change negative controls, then report every jar/commit invariant."""

    def __init__(self, args: argparse.Namespace):
        """Retain shared gate options and reserve names without creating directories or starting anything."""
        self._args = args
        self._directory = args.runs_dir / "it07_jar"
        self._checks: dict[str, bool] = {}

    def setup(self) -> None:
        """Reserve a fresh artifact directory and count inherited base starts so copied evidence cannot pass."""
        self._directory.mkdir()
        self._inherited_starts = len(provenance.RunProvenance.starts(self._args.runs_dir / "base"))

    def run(self) -> None:
        """Reuse full E3, then run the copied-jar chain and server-free study negative control."""
        self._checks["full E3 recorded run and replay passed"] = E3Experiment(self._args).run()
        self._run_jar_chain()
        self._check_study_refusal()

    def _run_jar_chain(self) -> None:
        """Keep a real two-process GateRunner report; only its copied jar may be rewritten by the first gate."""
        directory = Path(tempfile.mkdtemp(prefix="chain_", dir=self._directory))
        jar = directory / "harness.jar"
        shutil.copy2(harness.MOD_JAR, jar)
        before = provenance.file_sha256(jar)
        first_script = directory / "rewrite_jar.py"
        first_script.write_text(
            '"""Rewrite the copied jar for the IT-07 negative control."""\n'
            'from pathlib import Path\nimport sys\n'
            'Path(sys.argv[1]).write_bytes(b"IT-07 changed jar\\n")\n'
            'print("PASS copied jar rewritten")\n', encoding="utf-8")
        second_script = directory / "second.py"
        second_script.write_text(
            '"""This stand-in must never run after the jar changes."""\n'
            'print("PASS unexpected second gate")\n', encoding="utf-8")
        tier = Tier("it07-jar-change", (
            Gate("First", first_script, (str(jar),), timeout_s=10),
            Gate("Second", second_script, timeout_s=10),
        ), 0, 0, 1)
        status = GateRunner(tier, runs_dir=directory, jar=jar,
                            ports=range(self._args.port, self._args.port + 8)).run()
        reports = list(directory.glob("gates_*/gates.json"))
        if len(reports) != 1:
            raise harness.HarnessError(f"Expected one copied-jar chain report in {directory}")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        first, second = report["gates"]
        self._checks["stand-in chain preflight and first gate passed"] = (
            report["preflight"]["passed"] and report["jar_sha256"] == before
            and first["passed"] and not first["not_run"] and first["exit_code"] == 0)
        self._checks["negative control: jar changed, second gate not_run"] = (
            before is not None and provenance.file_sha256(jar) != before
            and status == 1 and not report["passed"] and second["not_run"] and not second["passed"]
            and second["exit_code"] is None and "jar changed" in second.get("reason", "")
            and not (reports[0].parent / "second").exists())

    def _check_study_refusal(self) -> None:
        """Require a synthetic factual jar mismatch to fail before cloning or moving an unfinished branch."""
        directory = self._directory / "study"
        directory.mkdir()
        study = Study(argparse.Namespace(runs_dir=directory, variant="sabotage", study="honest_freight",
                                         fork_day=1, end_day=2))
        factual = study._log_dir("control", 0)
        factual.mkdir()
        jar = directory / "factual.jar"
        jar.write_bytes(b"IT-07 synthetic factual jar\n")
        recorded = provenance.file_sha256(jar)
        current = provenance.file_sha256(harness.MOD_JAR)
        provenance.RunProvenance.record(factual, provenance.CodeState(None, False, "", (), current))
        provenance.RunProvenance.record(factual, provenance.CodeState(None, False, "", (), recorded))
        branch = study._log_dir("intervention", 0)
        branch.mkdir()
        evidence = branch / "unfinished.txt"
        evidence.write_text("keep unfinished branch", encoding="utf-8")
        refused = False
        try:
            study.branch(directory / "missing_fork", "intervention", 0)
        except harness.HarnessError as error:
            refused = ("jar changed" in str(error) and str(factual) in str(error)
                       and str(recorded) in str(error) and str(current) in str(error))
        self._checks["negative control: factual jar mismatch refused before clone"] = (
            recorded != current and refused and evidence.is_file()
            and evidence.read_text(encoding="utf-8") == "keep unfinished branch"
            and not list(directory.glob(f"{branch.name}_unfinished_*")))

    def _provenance_checks(self) -> dict[str, bool]:
        """Check fresh latest starts, not inherited fixture history, against the jar and HEAD current now."""
        current = provenance.CodeState.capture(harness.MOD_JAR)
        checks = {"current harness jar and HEAD are available": (
            current.jar_sha256 is not None and current.commit is not None and current.error is None)}
        for name in ("e3_a", "e3_b"):
            starts = provenance.RunProvenance.starts(self._args.runs_dir / name)
            latest = starts[-1] if starts else {}
            checks[f"{name} appended server-start provenance"] = len(starts) > self._inherited_starts
            checks[f"{name} latest jar matches current harness jar"] = (
                current.jar_sha256 is not None and latest.get("jar_sha256") == current.jar_sha256)
            checks[f"{name} latest commit matches current HEAD"] = (
                current.commit is not None and latest.get("commit") == current.commit)
        return checks

    def check(self) -> bool:
        """Print each result and the final IT-07 verdict; pass only when E3 and both negative controls pass."""
        self._checks.update(self._provenance_checks())
        for label, passed in self._checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(self._checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-07: E3 provenance, copied-jar chain and factual branch guard")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse shared options, and execute setup/run/check with a failing error verdict."""
    harness.stop_on_sigterm()
    experiment = JarExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    try:
        experiment.setup()
        experiment.run()
        return 0 if experiment.check() else 1
    except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
        print(f"  BAD {error}\nFAIL IT-07: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
