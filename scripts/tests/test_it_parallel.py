"""Offline argument isolation, trace comparison, and port-error classification for the parallel gate."""

import contextlib
import errno
import io
from pathlib import Path
import tempfile
import unittest

from scripts import harness
from scripts.test_e3 import RUN_TICKS
from scripts.test_it08_parallel import ParallelExperiment, is_port_conflict, parallel_options, trace_checks


class ParallelLogicTests(unittest.TestCase):
    """Exercise parallel-run logic without sockets, server launches, APFS cloning, or fake runtime objects."""

    def test_worker_options_are_isolated_and_use_reserved_port_pairs(self):
        """Preserve shared CLI values while giving each worker a separate Namespace and correct ports."""
        args = harness.experiment_parser("").parse_args(
            ["--runs-dir", "runs/example", "--port", "25700", "--game-port", "25701", "--heap", "2G"])
        original = vars(args).copy()
        first, second = parallel_options(args)
        self.assertIsNot(first, second)
        self.assertIsNot(first, args)
        self.assertEqual((first.port, first.game_port, second.port, second.game_port),
                         (25700, 25701, 25702, 25703))
        self.assertEqual(first.runs_dir, args.runs_dir.resolve())
        self.assertEqual(second.runs_dir, first.runs_dir)
        self.assertEqual(second.heap, "2G")
        second.port = 25800
        first.heap = "3G"
        self.assertEqual(vars(args), original)
        self.assertEqual(first.port, 25700)
        self.assertEqual(second.heap, "2G")

    def test_default_game_port_is_explicit_and_disjoint(self):
        """A missing game-port must not let both cloned servers fall back to the default Minecraft port."""
        args = harness.experiment_parser("").parse_args(["--port", "25700"])
        first, second = parallel_options(args)
        self.assertEqual((first.game_port, second.game_port), (25701, 25703))
        self.assertIsNone(args.game_port)

    def test_colliding_and_out_of_range_ports_are_rejected_offline(self):
        """Fail invalid input before directories or servers can be created."""
        for control, game in ((25700, 25700), (25700, 25702), (25700, 25703), (65533, 65534), (0, 1)):
            with self.subTest(control=control, game=game):
                args = harness.experiment_parser("").parse_args(
                    ["--port", str(control), "--game-port", str(game)])
                with self.assertRaises(ValueError):
                    parallel_options(args)

    def test_equal_complete_traces_pass_without_computing_any_digest(self):
        """Compare existing sample records and decision entries as data, without new content hashing."""
        hashes = [{"gameTime": tick} for tick in range(20, RUN_TICKS + 1, 20)]
        decisions = [(40, "ana", "recorded prompt", "walk_to", "{}", "accepted")]
        checks = trace_checks(hashes, list(hashes), decisions, list(decisions))
        self.assertTrue(all(checks.values()))

    def test_equal_empty_or_truncated_traces_fail(self):
        """Identically missing evidence is not proof of deterministic concurrent execution."""
        self.assertFalse(any(trace_checks([], [], [], []).values()))
        checks = trace_checks([{"gameTime": 20}], [{"gameTime": 20}], [(40, "ana")], [(40, "ana")])
        self.assertFalse(checks["identical complete world-hash traces"])
        self.assertTrue(checks["identical nonempty decision traces"])

    def test_world_and_decision_mismatches_fail_independently(self):
        """A changed sample or decision must fail its own comparison while leaving the other check intact."""
        hashes = [{"gameTime": tick} for tick in range(20, RUN_TICKS + 1, 20)]
        changed = [*hashes[:-1], {"gameTime": RUN_TICKS + 20}]
        checks = trace_checks(hashes, changed, [(40, "ana")], [(40, "ana")])
        self.assertFalse(checks["identical complete world-hash traces"])
        self.assertTrue(checks["identical nonempty decision traces"])
        checks = trace_checks(hashes, hashes, [(40, "ana")], [(80, "ana")])
        self.assertTrue(checks["identical complete world-hash traces"])
        self.assertFalse(checks["identical nonempty decision traces"])

    def test_port_errors_are_specific_to_a_binding_conflict(self):
        """Accept actual occupied-port diagnostics, not timeouts, permission errors, or missing launchers."""
        for error in (OSError(errno.EADDRINUSE, "Address already in use"),
                      harness.HarnessError("java.net.BindException: Address already in use"),
                      harness.HarnessError("control port 25700 is occupied")):
            with self.subTest(error=error):
                self.assertTrue(is_port_conflict(error))
        for error in (TimeoutError("Server readiness timed out"), FileNotFoundError("launcher"),
                      PermissionError(errno.EACCES, "Permission denied"), ConnectionRefusedError("port refused")):
            with self.subTest(error=error):
                self.assertFalse(is_port_conflict(error))

    def test_unrun_experiment_cannot_pass(self):
        """Checking a newly constructed experiment must fail without attempting any external operations."""
        experiment = ParallelExperiment(harness.experiment_parser("").parse_args([]))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(experiment.check())

    def test_setup_preserves_existing_run_before_any_cloning(self):
        """A destination owned by another worker is never overwritten or used as a fixture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base").mkdir()
            destination = root / "it08_b"
            destination.mkdir()
            evidence = destination / "keep.txt"
            evidence.write_text("other worker", encoding="utf-8")
            args = harness.experiment_parser("").parse_args(["--runs-dir", str(root)])
            with self.assertRaises(FileExistsError):
                ParallelExperiment(args).setup()
            self.assertEqual(evidence.read_text(encoding="utf-8"), "other worker")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["base", "it08_b"])
