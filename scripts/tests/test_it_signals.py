"""Offline signal-gate logic tests; the only subprocess is a tiny Python file holder, never a server."""

import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from scripts import harness
from scripts.test_it06_signals import (
    JVM_START,
    SignalExperiment,
    cleanup_java,
    directory_holders,
    is_java,
    java_holders,
    lsof_processes,
    run_incomplete,
    server_log_ready,
    stop_child,
    wait_until_ready,
)


class SignalLogicTests(unittest.TestCase):
    """Check lsof field parsing, readiness boundaries, incomplete verdicts, and fresh-directory protection."""

    def test_lsof_fields_identify_processes_not_filenames(self):
        """Retain distinct process identities even when filenames and other commands mention Java."""
        output = "p120\ncjava\nfcwd\nn/a/run\np121\ncPython\nn/java/server.log\np122\nc/jdk/bin/java\n"
        processes = lsof_processes(output)
        self.assertEqual(processes, {120: "java", 121: "Python", 122: "/jdk/bin/java"})
        self.assertEqual({pid for pid, command in processes.items() if is_java(command)}, {120, 122})
        self.assertFalse(is_java("javascript"))
        self.assertFalse(is_java("python test_java.py"))

    def test_lsof_empty_and_malformed_pid_fields(self):
        """Treat a no-holder result as empty and never attach a command to an invalid or previous PID."""
        self.assertEqual(lsof_processes(""), {})
        self.assertEqual(lsof_processes("cjava\np0\ncjava\npwrong\ncjava\np31\ncPython\npbad\ncjava"),
                         {31: "Python"})

    def test_incomplete_requires_nonzero_exit_and_no_pass_line(self):
        """Reject successful, still-running, or falsely passing children while accepting real interruptions."""
        self.assertTrue(run_incomplete(-2, "KeyboardInterrupt\n"))
        self.assertTrue(run_incomplete(1, "FAIL E3\n"))
        self.assertFalse(run_incomplete(None, ""))
        self.assertFalse(run_incomplete(0, ""))
        for output in ("PASS E3: done\n", "prefix\n  PASS E3: done\n", "PASS\n"):
            with self.subTest(output=output):
                self.assertFalse(run_incomplete(1, output))

    def test_readiness_ignores_inherited_and_previous_start_logs(self):
        """Only a fresh start after the inherited byte offset may satisfy both server readiness messages."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            self.assertFalse(server_log_ready(path))
            ready = (f"{JVM_START}\nDone (0.1s)! For help, type help\n"
                     "SimHarness control listening on 127.0.0.1:25700\n")
            path.write_text(ready, encoding="utf-8")
            offset = path.stat().st_size
            self.assertTrue(server_log_ready(path))
            self.assertFalse(server_log_ready(path, offset))
            with path.open("a", encoding="utf-8") as output:
                output.write(f"{JVM_START}\nstarting\n")
            self.assertFalse(server_log_ready(path))
            self.assertFalse(server_log_ready(path, offset))
            with path.open("a", encoding="utf-8") as output:
                output.write("Done (1s)!\n")
            self.assertFalse(server_log_ready(path, offset))
            with path.open("a", encoding="utf-8") as output:
                output.write("SimHarness control listening on 127.0.0.1:25700\n")
            self.assertTrue(server_log_ready(path, offset))

    def test_seeded_delays_and_unrun_experiment_cannot_pass(self):
        """Constructing an experiment is offline and reproducibly chooses the same bounded delays."""
        args = harness.experiment_parser("").parse_args([])
        first, second = SignalExperiment(args), SignalExperiment(args)
        self.assertEqual(first._delays, second._delays)
        self.assertTrue(all(0.05 <= delay <= 0.5 for delay in first._delays))
        self.assertEqual(first._args.game_port, args.port + 1)
        self.assertIsNone(args.game_port)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(first.check())

    def test_setup_refuses_existing_destination_before_any_clone(self):
        """A conflicting output path is rejected before cloning or touching another worker's files."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir()
            destination = root / "it06_orphan"
            destination.mkdir()
            evidence = destination / "keep.txt"
            evidence.write_text("other worker", encoding="utf-8")
            args = harness.experiment_parser("").parse_args(["--runs-dir", str(root)])
            with self.assertRaises(FileExistsError):
                SignalExperiment(args).setup()
            self.assertEqual(evidence.read_text(encoding="utf-8"), "other worker")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["base", "it06_orphan"])


@unittest.skipUnless(shutil.which("lsof"), "lsof is required to inspect real process file holders")
class HolderProcessTests(unittest.TestCase):
    """Use a real, non-server Python process to verify file-holder detection and safe cleanup boundaries."""

    def test_log_only_holder_outside_run_cwd_is_detected_and_non_java_is_not_killed(self):
        """Detect a process holding only a nested server.log, without misclassifying or terminating Python."""
        with tempfile.TemporaryDirectory(prefix="it06 holders ") as temporary:
            root = Path(temporary)
            run = root / "run with spaces"
            (run / "nested").mkdir(parents=True)
            receipt = root / "ready"
            code = ("from pathlib import Path; import sys, time; "
                    "handle = open(sys.argv[1], 'a'); Path(sys.argv[2]).write_text('ready'); time.sleep(60)")
            process = subprocess.Popen([sys.executable, "-c", code, str(run / "nested/server.log"), str(receipt)],
                                       cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while not receipt.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(receipt.exists(), "Tiny file holder did not start")
                self.assertIn(process.pid, directory_holders(run))
                self.assertNotIn(process.pid, java_holders(run))
                cleanup_java(run)
                self.assertIsNone(process.poll())
                with self.assertRaises(harness.HarnessError):
                    wait_until_ready(process, run / "missing.log", 0.01)
                stop_child(process)
                self.assertIsNotNone(process.returncode)
                self.assertNotIn(process.pid, directory_holders(run))
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_empty_and_missing_directories_are_distinct(self):
        """An existing unheld directory is empty; a nonexistent directory is an inspection failure."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(directory_holders(root), {})
            with self.assertRaises(FileNotFoundError):
                directory_holders(root / "missing")
            log = root / "server.log"
            log.write_text("closed log", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                directory_holders(log)
