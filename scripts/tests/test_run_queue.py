"""The run queue starts jobs within its server budget, assigns port blocks and records exits."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_queue import FIRST_PORT, BLOCK, Job, RunQueue  # noqa: E402


class RunQueueTest(unittest.TestCase):
    """Jobs run in order within the budget, each with its own ports."""

    def _submit(self, queue: Path, name: str, servers: int, code: str, order: int) -> Job:
        job = Job(name, queue)
        job.spec_path.write_text(json.dumps({"command": [sys.executable, "-c", code], "servers": servers,
                                             "submitted": f"2026-10-21T00:00:0{order}"}), encoding="utf-8")
        return job

    def test_ports_budget_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            (queue / "jobs").mkdir()
            first = self._submit(queue, "first", 2, "import time; print('{port} {port+1}'); time.sleep(0.3)", 1)
            second = self._submit(queue, "second", 2, "print('{port}')", 2)
            failing = self._submit(queue, "failing", 1, "raise SystemExit(3)", 3)
            RunQueue(max_servers=3, queue=queue).serve(poll=0.05)
            self.assertEqual(first.status["state"], "done")
            self.assertEqual(first.log_path.read_text().split(), [str(FIRST_PORT), str(FIRST_PORT + 1)])
            self.assertEqual(second.status["state"], "done")
            self.assertEqual(failing.status["state"], "failed")
            self.assertEqual(failing.status["exit"], 3)
            # the second job (2 servers) could not start beside the first (2 of 3); the failing one (1) could
            self.assertLessEqual(first.status["ended"], second.status["started"])
            self.assertIn(failing.status["ports"], (FIRST_PORT + BLOCK, FIRST_PORT))

    def test_budget_file_and_disk_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            (queue / "jobs").mkdir()
            runner = RunQueue(max_servers=2, queue=queue)
            self.assertEqual(runner.budget(), 2)
            (queue / "max_servers").write_text("4\n")
            self.assertEqual(runner.budget(), 4)
            (queue / "max_servers").write_text("four")
            self.assertEqual(runner.budget(), 2)
            self.assertTrue(RunQueue(1, queue, min_free=0).disk_ok())
            self.assertFalse(RunQueue(1, queue, min_free=10**18).disk_ok())


if __name__ == "__main__":
    unittest.main()
