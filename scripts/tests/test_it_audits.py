"""Exercise IT-10's log-copy negative control with the real testimony auditor, without a server."""

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from scripts import harness
from scripts.audit_testimony import TestimonyAudit as ExistingTestimonyAudit
from scripts.test_it10_audits import AuditExperiment, corrupt_origin, main


class AuditControlTest(unittest.TestCase):
    """The gate corrupts one actual origin, preserves the source, and never accepts an empty control."""

    def test_one_corrupt_origin_is_detected_without_changing_source(self):
        """A valid firsthand item passes before copying and fails for exactly the changed origin afterward."""
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "source", Path(temporary) / "negative"
            logs = source / "world/simharness"
            logs.mkdir(parents=True)
            events = [{"id": 1, "type": "block_change", "pos": [1, 2, 3]},
                      {"type": "utterance", "uttId": 1, "speaker": "Alice", "speakerKey": "npc:adult_01",
                       "gameTime": 20, "startedAt": 10, "listeners": [],
                       "news": [{"newsId": "change:1", "originEventId": "change:1", "subject": [1, 2, 3],
                                 "chain": ["Alice"]}]}]
            original = "".join(json.dumps(event) + "\n" for event in events)
            (logs / "events.jsonl").write_text(original, encoding="utf-8")
            observation = {"observer": "npc:adult_01", "gameTime": 5, "changes": [{"id": 1}]}
            (logs / "npc_observations.jsonl").write_text(json.dumps(observation) + "\n", encoding="utf-8")
            self.assertEqual(ExistingTestimonyAudit.load(source).run().violations, [])
            label = corrupt_origin(source, destination)
            audit = ExistingTestimonyAudit.load(destination).run()
            self.assertEqual(audit.checked, 1)
            self.assertEqual(len(audit.violations), 1)
            self.assertIn(label, audit.violations[0])
            self.assertIn("not a logged block change", audit.violations[0])
            self.assertEqual((logs / "events.jsonl").read_text(encoding="utf-8"), original)
            with self.assertRaises(FileExistsError):
                corrupt_origin(source, destination)

    def test_missing_testimony_cannot_pass_the_negative_control(self):
        """No NPC news is a failed prerequisite, rather than a silently successful negative control."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            logs = source / "world/simharness"
            logs.mkdir(parents=True)
            (logs / "events.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No NPC testimony"):
                corrupt_origin(source, Path(temporary) / "negative")

    def test_ports_stay_inside_the_gate_allocation(self):
        """The default game port is adjacent and explicit sockets may not collide or escape the allocation."""
        args = harness.experiment_parser("").parse_args(["--port", "25700"])
        self.assertEqual(AuditExperiment(args).options["game_port"], 25701)
        for port in (25699, 25700, 25708):
            with self.subTest(port=port):
                args.game_port = port
                with self.assertRaises(ValueError):
                    AuditExperiment(args)
        args.game_port = None
        for port in (0, -1, 65529, 65536):
            with self.subTest(base=port):
                args.port = port
                with self.assertRaises(ValueError):
                    AuditExperiment(args)

    def test_invalid_options_print_a_failure_verdict_without_cloning(self):
        """An operational error still produces the required BAD and FAIL lines and a nonzero exit status."""
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            status = main(["--runs-dir", temporary, "--port", "25700", "--game-port", "25708"])
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(status, 1)
        self.assertIn("BAD", output.getvalue())
        self.assertIn("FAIL IT-10:", output.getvalue())
