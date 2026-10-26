"""Exercise IT-17's log corruption and divergence artifact checks without a server."""

import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.decisions import Decision, DecisionLog
from scripts.test_it17_file_replay import corrupt_digest, divergence_names


class FileReplayTest(unittest.TestCase):
    """Corruption changes one digest byte only and diagnostics must name the exact decision."""

    def test_changes_exactly_one_digest_byte_and_keeps_log_loadable(self):
        """Real DecisionLog records survive the byte edit with all other fields and source untouched."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, damaged = directory / "source.jsonl", directory / "damaged.jsonl"
            log = DecisionLog(source)
            for name in ("ana", "ben"):
                log.append(Decision(40, name, "a" * 64, False, "model", "wait", {"ticks": 40},
                                    "scripted", 0, 0, "prompt"))
            before = source.read_bytes()
            self.assertEqual(corrupt_digest(source, damaged), ("ana", 40))
            after = damaged.read_bytes()
            self.assertEqual(len(before), len(after))
            self.assertEqual(sum(left != right for left, right in zip(before, after)), 1)
            self.assertEqual(source.read_bytes(), before)
            loaded = DecisionLog(damaged)
            self.assertEqual(loaded.get("ben", 40), log.get("ben", 40))
            self.assertNotEqual(loaded.get("ana", 40).digest, log.get("ana", 40).digest)

    def test_empty_log_cannot_be_a_negative_control(self):
        """Missing decisions must not silently turn corruption into a passing negative control."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "empty.jsonl"
            source.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty decision log"):
                corrupt_digest(source, Path(temporary) / "bad.jsonl")

    def test_diagnostic_requires_matching_agent_and_tick(self):
        """A missing artifact, another agent or another tick is not evidence the fault was caught."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diverged.json"
            self.assertFalse(divergence_names(path, "ana", 40))
            path.write_text(json.dumps([{"agent": "ana", "gameTime": 40}]), encoding="utf-8")
            self.assertTrue(divergence_names(path, "ana", 40))
            self.assertFalse(divergence_names(path, "ben", 40))
            self.assertFalse(divergence_names(path, "ana", 80))
