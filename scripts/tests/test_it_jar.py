"""Offline IT-07 checks: latest provenance, a tiny real subprocess chain, and the pre-clone factual jar guard.

Pass when changed jars are rejected, legacy factual runs still reach cloning, and provenance uses only the latest
fresh start. Negative controls use mismatched jar/commit records and the chain's rewritten copy. Missing fork paths
bound compatibility tests before any APFS operation or server; no Minecraft, Gradle, model, or mock framework runs.
"""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import harness, provenance
from scripts.run_study import Study, save_outcome
from scripts.test_it07_jar import JarExperiment


class JarWorkspaceTest(unittest.TestCase):
    """Provide real temporary files and redirect only the current jar path, restoring it after each test."""

    def setUp(self):
        """Create a base fixture and tiny current jar without invoking setup_server or a JVM."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "base").mkdir()
        self.jar = self.root / "current.jar"
        self.jar.write_bytes(b"current unit-test jar")
        self.addCleanup(setattr, harness, "MOD_JAR", harness.MOD_JAR)
        harness.MOD_JAR = self.jar
        self.args = harness.experiment_parser("").parse_args([
            "--runs-dir", str(self.root), "--port", "28000", "--game-port", "28001"])


class StudyJarGuardTest(JarWorkspaceTest):
    """Reject the latest factual jar mismatch before cloning, retaining legacy and resume behavior."""

    def setUp(self):
        """Build a real Study with a synthetic factual directory and a deliberately nonexistent fork."""
        super().setUp()
        self.study = Study(argparse.Namespace(**vars(self.args), variant="sabotage", study="honest_freight",
                                              fork_day=1, end_day=2))
        self.factual = self.study._log_dir("control", 0)
        self.factual.mkdir()
        self.fork = self.root / "missing_fork"
        self.current = provenance.file_sha256(self.jar)

    def record(self, digest: str | None) -> None:
        """Append factual evidence through the same provenance writer used by the harness."""
        provenance.RunProvenance.record(self.factual, provenance.CodeState("factual-commit", False, "", (), digest))

    def assert_reaches_clone(self) -> None:
        """An allowed branch must reach the missing fork's strict resolution, before APFS or server startup."""
        with self.assertRaises(FileNotFoundError) as caught:
            self.study.branch(self.fork, "intervention", 0)
        self.assertEqual(Path(caught.exception.filename), self.fork)
        self.assertFalse(self.study._log_dir("intervention", 0).exists())

    def test_latest_mismatch_refuses_both_arms_without_moving_unfinished_runs(self):
        """A later wrong jar defeats an earlier match, regardless of arm, and preserves branch evidence."""
        self.record(self.current)
        self.record("different-factual-jar")
        for arm, index in (("intervention", 0), ("control", 1)):
            with self.subTest(arm=arm):
                branch = self.study._log_dir(arm, index)
                branch.mkdir()
                (branch / "keep.txt").write_text("unfinished", encoding="utf-8")
                with self.assertRaises(harness.HarnessError) as caught:
                    self.study.branch(self.fork, arm, index)
                self.assertIn("jar changed", str(caught.exception))
                self.assertIn(str(self.factual), str(caught.exception))
                self.assertIn("different-factual-jar", str(caught.exception))
                self.assertIn(self.current, str(caught.exception))
                self.assertEqual((branch / "keep.txt").read_text(encoding="utf-8"), "unfinished")
                self.assertFalse(list(self.root.glob(f"{branch.name}_unfinished_*")))

    def test_matching_latest_start_allows_cloning_despite_older_mismatch(self):
        """Only the latest start's jar matters, not older jars or a different factual commit."""
        self.record("old-jar")
        self.record(self.current)
        self.assert_reaches_clone()

    def test_factual_without_provenance_remains_usable(self):
        """A run recorded before provenance existed is allowed through to cloning."""
        self.assert_reaches_clone()

    def test_empty_provenance_history_remains_usable(self):
        """The provenance reader's empty-history compatibility path still permits cloning."""
        (self.factual / provenance.RunProvenance.FILE).write_text('{"starts": []}', encoding="utf-8")
        self.assert_reaches_clone()

    def test_current_jar_is_rehashed_before_each_branch(self):
        """Replacing the current jar after a matching factual start prevents a new branch."""
        self.record(self.current)
        self.jar.write_bytes(b"changed current unit-test jar")
        with self.assertRaisesRegex(harness.HarnessError, "jar changed"):
            self.study.branch(self.fork, "intervention", 0)

    def test_recorded_missing_jar_does_not_match_an_available_current_jar(self):
        """A recorded null jar is a mismatch, not the legacy absence of provenance."""
        self.record(None)
        with self.assertRaisesRegex(harness.HarnessError, "jar changed"):
            self.study.branch(self.fork, "intervention", 0)

    def test_completed_branch_is_reused_without_checking_a_jar_for_a_new_start(self):
        """A finished outcome remains resumable without cloning or starting a differently built server."""
        self.record("old-jar")
        branch = self.study._log_dir("intervention", 0)
        branch.mkdir()
        outcome = {"welfare": {1: 0.5}, "first_true": {}, "roles": {}}
        save_outcome(branch, outcome)
        self.assertEqual(self.study.branch(self.fork, "intervention", 0), outcome)

    def test_stopped_study_does_not_start_or_jar_check_another_branch(self):
        """A budget stop still short-circuits before jar checking or any branch filesystem work."""
        self.record("old-jar")
        self.study._stopped = "model-call budget exhausted"
        self.assertEqual(self.study.branch(self.fork, "intervention", 0), {"stopped": self.study._stopped})
        self.assertFalse(self.study._log_dir("intervention", 0).exists())


class LatestProvenanceTest(JarWorkspaceTest):
    """Both E3 runs need fresh latest-start provenance; stale inherited history must never suffice."""

    def setUp(self):
        """Simulate copied fixture history followed by a fresh start for each E3 run, without servers."""
        super().setUp()
        provenance.RunProvenance.record(self.root / "base", provenance.CodeState("fixture", False, "", (), "old"))
        self.experiment = JarExperiment(self.args)
        self.experiment.setup()
        self.current = provenance.CodeState.capture(self.jar)
        for name in ("e3_a", "e3_b"):
            directory = self.root / name
            directory.mkdir()
            shutil.copy2(self.root / "base/provenance.json", directory / "provenance.json")
            provenance.RunProvenance.record(directory, self.current)

    def test_latest_current_records_pass_even_with_stale_inherited_starts(self):
        """Historical fixture commits and jars do not cause a false failure after a fresh matching start."""
        checks = self.experiment._provenance_checks()
        self.assertTrue(all(checks.values()), checks)

    def test_latest_wrong_jar_is_detected_for_each_run(self):
        """A later mismatched jar is detected independently for either E3 run."""
        for name in ("e3_a", "e3_b"):
            with self.subTest(run=name):
                provenance.RunProvenance.record(self.root / name, provenance.CodeState(
                    self.current.commit, False, "", (), "changed"))
                self.assertFalse(self.experiment._provenance_checks()[f"{name} latest jar matches current harness jar"])

    def test_latest_wrong_commit_is_detected(self):
        """Matching jar identity does not excuse a latest start recorded under a different commit."""
        provenance.RunProvenance.record(self.root / "e3_b", provenance.CodeState(
            "wrong-head", False, "", (), self.current.jar_sha256))
        self.assertFalse(self.experiment._provenance_checks()["e3_b latest commit matches current HEAD"])

    def test_missing_provenance_cannot_pass(self):
        """Deleting a run's evidence fails its fresh-start, jar and commit checks."""
        (self.root / "e3_a/provenance.json").unlink()
        checks = self.experiment._provenance_checks()
        self.assertTrue(all(not passed for label, passed in checks.items() if label.startswith("e3_a ")))

    def test_matching_inherited_evidence_alone_cannot_pass(self):
        """A matching copied fixture start is not evidence that this run appended a server start."""
        path = self.root / "e3_a/provenance.json"
        path.unlink()
        provenance.RunProvenance.record(path.parent, self.current)
        checks = self.experiment._provenance_checks()
        self.assertFalse(checks["e3_a appended server-start provenance"])
        self.assertTrue(checks["e3_a latest jar matches current harness jar"])

    def test_missing_current_jar_fails_even_with_recorded_evidence(self):
        """An unavailable current build must not be mistaken for matching null provenance."""
        self.jar.unlink()
        checks = self.experiment._provenance_checks()
        self.assertFalse(checks["current harness jar and HEAD are available"])
        self.assertFalse(checks["e3_a latest jar matches current harness jar"])


class JarNegativeControlTest(JarWorkspaceTest):
    """Exercise both gate negative controls offline with real files and tiny Python subprocesses."""

    def test_chain_rewrites_only_its_copy_and_skips_second_gate(self):
        """The real runner persists not_run plus jar changed while leaving the current jar intact."""
        experiment = JarExperiment(self.args)
        experiment.setup()
        original = provenance.file_sha256(self.jar)
        experiment._run_jar_chain()
        self.assertTrue(all(experiment._checks.values()), experiment._checks)
        self.assertEqual(provenance.file_sha256(self.jar), original)
        report_path, = experiment._directory.glob("chain_*/gates_*/gates.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(len(report["gates"]), 2)
        self.assertEqual(report["jar_sha256"], original)
        self.assertTrue(report["gates"][0]["passed"])
        self.assertTrue(report["gates"][1]["not_run"])
        self.assertIn("jar changed", report["gates"][1]["reason"])
        self.assertIsNone(report["gates"][1]["exit_code"])
        self.assertFalse((report_path.parent / "second").exists())

    def test_synthetic_study_negative_control_never_needs_a_fork_or_server(self):
        """The gate's real Study refuses latest factual mismatch and preserves unfinished evidence."""
        experiment = JarExperiment(self.args)
        experiment.setup()
        experiment._check_study_refusal()
        self.assertTrue(all(experiment._checks.values()), experiment._checks)
        self.assertFalse((experiment._directory / "study/missing_fork").exists())

    def test_setup_refuses_to_overwrite_an_existing_experiment(self):
        """Repeated setup cannot silently reuse another experiment's negative-control artifacts."""
        experiment = JarExperiment(self.args)
        experiment.setup()
        with self.assertRaises(FileExistsError):
            experiment.setup()


if __name__ == "__main__":
    unittest.main()
