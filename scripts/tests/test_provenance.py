"""Unit tests for run and fixture provenance (incidents P1, P10, P11, W1) and SIGTERM handling (C8, incident P5)."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.provenance import CodeState, FixtureMetadata, RunProvenance, file_sha256


class GitRepository:
    """A throwaway git repository with one committed file."""

    def __init__(self, root: Path):
        """Create the repository and its first commit."""
        self.root = root
        self.git("init", "-q")
        (root / "code.py").write_text("print('one')\n", encoding="utf-8")
        self.git("add", "code.py")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.test", "commit", "-q", "-m", "first")

    def git(self, *args: str) -> str:
        """Run git in the repository and return its output."""
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True,
                              check=True).stdout


class CodeStateTest(unittest.TestCase):
    """The captured state names the commit, and a dirty tree's diff and untracked files."""

    def setUp(self):
        """A repository and a jar file in a temporary directory."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "repo").mkdir()
        self.repo = GitRepository(self.root / "repo")
        self.jar = self.root / "harness.jar"
        self.jar.write_bytes(b"jar")

    def test_clean_tree_records_commit_and_jar(self):
        """A clean tree is the HEAD commit, not dirty, with no diff, and the jar's SHA-256."""
        state = CodeState.capture(self.jar, self.repo.root)
        self.assertEqual(state.commit, self.repo.git("rev-parse", "HEAD").strip())
        self.assertEqual((state.dirty, state.diff, state.untracked, state.error), (False, "", (), None))
        self.assertEqual(state.jar_sha256, file_sha256(self.jar))

    def test_dirty_tree_records_diff_and_untracked_files(self):
        """A modified tracked file is in the diff, and a new file is listed as untracked (P11)."""
        (self.repo.root / "code.py").write_text("print('two')\n", encoding="utf-8")
        (self.repo.root / "new.py").write_text("x = 1\n", encoding="utf-8")
        state = CodeState.capture(self.jar, self.repo.root)
        self.assertTrue(state.dirty)
        self.assertIn("+print('two')", state.diff)
        self.assertEqual(state.untracked, ("new.py",))

    def test_unreadable_repository_is_recorded_not_raised(self):
        """Outside a git repository the state carries the error and counts as dirty."""
        state = CodeState.capture(self.jar, self.root)
        self.assertIsNone(state.commit)
        self.assertTrue(state.dirty)
        self.assertTrue(state.error.startswith("git: "))


class RunProvenanceTest(unittest.TestCase):
    """Each server start appends one entry; a dirty start saves its diff beside the run."""

    def test_starts_accumulate_and_dirty_diffs_are_saved(self):
        """Two starts give two entries in order; only the dirty one names a saved diff file."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            self.assertEqual(RunProvenance.starts(run), [])
            RunProvenance.record(run, CodeState("abc", False, "", (), "jar1"))
            RunProvenance.record(run, CodeState("abc", True, "diff text\n", ("new.py",), "jar2"))
            starts = RunProvenance.starts(run)
            self.assertEqual([(entry["commit"], entry["jar_sha256"], entry["diff_file"]) for entry in starts],
                             [("abc", "jar1", None), ("abc", "jar2", "provenance_2.diff")])
            self.assertEqual((run / "provenance_2.diff").read_text(encoding="utf-8"), "diff text\n")
            self.assertEqual(starts[1]["untracked"], ["new.py"])
            self.assertNotIn("diff", starts[1])


class FixtureMetadataTest(unittest.TestCase):
    """A fixture is usable only with metadata, and only while its manifest's world-relevant fields are unchanged."""

    def setUp(self):
        """A fixture directory and its manifest in a temporary directory."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.fixture = Path(folder.name) / "village_test"
        self.fixture.mkdir()
        self.manifest = Path(folder.name) / "village_test.json"
        self.manifest.write_text(json.dumps({"seed": 1, "layout": {"bell": [1, 2, 3]}, "success_predicates": {}}),
                                 encoding="utf-8")

    def edit_manifest(self, **fields) -> None:
        """Change top-level manifest fields."""
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.manifest.write_text(json.dumps({**data, **fields}), encoding="utf-8")

    def test_missing_fixture_or_metadata_is_refused(self):
        """A missing directory, or one without `fixture.json`, cannot be used."""
        self.assertIn("does not exist", FixtureMetadata.problems(self.fixture.with_name("absent"))[0])
        self.assertIn("no fixture.json", FixtureMetadata.problems(self.fixture, self.manifest)[0])

    def test_current_fixture_passes_and_non_world_fields_do_not_matter(self):
        """A fixture built from the current manifest has no problems, even after a predicate-only change."""
        FixtureMetadata.write(self.fixture, CodeState("abc", False, "", (), "jar"), manifest=self.manifest)
        self.edit_manifest(success_predicates={"fair_open": True})
        self.assertEqual(FixtureMetadata.problems(self.fixture, self.manifest), [])

    def test_world_field_change_makes_the_fixture_stale(self):
        """A change to a world-relevant field (here the layout) is named in the refusal (W1)."""
        FixtureMetadata.write(self.fixture, CodeState("abc", False, "", (), "jar"), manifest=self.manifest)
        self.edit_manifest(layout={"bell": [9, 9, 9]})
        problems = FixtureMetadata.problems(self.fixture, self.manifest)
        self.assertEqual(len(problems), 1)
        self.assertIn("changed in layout since it was built", problems[0])


class StopOnSigtermTest(unittest.TestCase):
    """After `stop_on_sigterm`, SIGTERM unwinds the script so its cleanup (a server's stop) runs (C8, P5)."""

    def test_sigterm_runs_cleanup(self):
        """A process that called `stop_on_sigterm` runs its `finally` block when terminated."""
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "cleaned"
            ready = Path(folder) / "ready"
            code = ("import pathlib, time\nfrom scripts import harness\nharness.stop_on_sigterm()\n"
                    f"pathlib.Path({str(ready)!r}).touch()\n"
                    f"try:\n    time.sleep(30)\nfinally:\n    pathlib.Path({str(marker)!r}).touch()\n")
            process = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                                       stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            os.kill(process.pid, signal.SIGTERM)
            process.wait(10)
            self.assertTrue(marker.exists())
            self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
