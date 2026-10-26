"""Exercise IT-16's setup, resource accounting and real runtime replay logic without any server or model service."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import DecisionLog
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.runtime import SimulationRuntime
from scripts import harness
from scripts.test_it16_soak import (CHECK_EVERY, DAY_TICKS, MAX_RSS_BYTES, MAX_RUN_BYTES, SoakExperiment,
                                   directory_bytes, parse_rss_bytes, positive_days, process_rss_bytes)
from scripts.tests.fake_world import FakeWorld


def wait_policy(system: str, user: str) -> ToolCall:
    """Provide deterministic actions to the real runtime without loading a model."""
    return ToolCall("wait", {"ticks": 1_200})


class SoakTests(unittest.TestCase):
    """Reject missing, changed and over-budget evidence without sockets, APFS cloning or a JVM."""

    def setUp(self):
        """Create only an empty temporary fixture and parsed shared arguments for each test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "village_sabotage").mkdir()
        self.args = harness.experiment_parser("").parse_args(["--runs-dir", str(self.root), "--port", "25700"])
        self.args.days, self.args.variant = 1, "sabotage"
        self.experiment = SoakExperiment(self.args)

    def populate_evidence(self):
        """Run real recorded and file-replayed runtimes against FakeWorld, then attach resource measurements."""
        start = 620
        source = None
        for name in self.experiment.names:
            directory = self.root / name
            world = FakeWorld({"ana": [0, 64, 0]}, game_time=start)
            model = ScriptedClient(wait_policy) if source is None else None
            agent = Agent(RoleBrief("ana", "worker", "Wait here.", "Wait."))
            runtime = SimulationRuntime(world, [agent], model, directory / "orchestrator",
                                        command_error=harness.CommandError, source=source,
                                        hash_region=harness.REGION, hash_every=CHECK_EVERY)
            runtime.run(start + DAY_TICKS)
            self.experiment.runtimes[name] = runtime
            self.experiment.reports[name] = {
                "directory": str(directory), "completed": True, "start": start, "end": start + DAY_TICKS,
                "rss_samples": [{"seconds": 0, "pid": os.getpid(), "rss_bytes": MAX_RSS_BYTES}],
                "rss_errors": [], "bytes": MAX_RUN_BYTES,
            }
            source = DecisionLog(directory / "orchestrator" / "decisions.jsonl")

    def test_positive_days(self):
        """Accept positive whole days and reject zero, negatives and fractional values before startup."""
        self.assertEqual(positive_days("5"), 5)
        for value in ("0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_days(value)
        with self.assertRaises(ValueError):
            positive_days("1.5")

    def test_setup_confines_ports_and_paths_without_mutating_shared_arguments(self):
        """Both sequential servers use an explicit game port in the reserved eight-port window."""
        original = vars(self.args).copy()
        self.experiment.setup()
        self.assertEqual(self.experiment.options["port"], 25700)
        self.assertEqual(self.experiment.options["game_port"], 25701)
        self.assertEqual(vars(self.args), original)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["village_sabotage"])
        self.args.game_port = 25707
        self.experiment.setup()
        self.assertEqual(self.experiment.options["game_port"], 25707)

    def test_setup_refuses_invalid_ports_days_and_preexisting_artifacts(self):
        """Do not start a server for invalid input or overwrite another worker's artifacts."""
        for control, game in ((0, 1), (25700, 25700), (25700, 25708), (65535, None)):
            with self.subTest(control=control, game=game), self.assertRaises(ValueError):
                self.args.port, self.args.game_port = control, game
                self.experiment.setup()
        self.args.port, self.args.game_port = 25700, None
        self.args.days = 0
        with self.assertRaises(ValueError):
            self.experiment.setup()
        self.args.days = 1
        for name in (*self.experiment.names, "it16_soak.json"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text("preserve me", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    self.experiment.setup()
                self.assertEqual(path.read_text(encoding="utf-8"), "preserve me")
                path.unlink()

    def test_setup_refuses_missing_and_external_symlink_fixture(self):
        """The fixture must originate inside --runs-dir, not from a fallback runs directory."""
        fixture = self.root / "village_sabotage"
        fixture.rmdir()
        with self.assertRaises(ValueError):
            self.experiment.setup()
        fixture.symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.experiment.setup()

    def test_directory_size_counts_files_not_links_or_unrelated_runs(self):
        """Logical size includes nested artifacts, not directory entries or data beyond a symlink."""
        directory = self.root / "size"
        directory.mkdir()
        (directory / "nested").mkdir()
        (directory / "small.bin").write_bytes(b"abc")
        (directory / "nested" / "other.bin").write_bytes(b"12345")
        (self.root / "outside.bin").write_bytes(b"x" * 1_000)
        (directory / "link.bin").symlink_to(self.root / "outside.bin")
        (directory / "link_dir").symlink_to(self.root / "village_sabotage", target_is_directory=True)
        self.assertEqual(directory_bytes(directory), 8)

    def test_ps_rss_units_and_invalid_measurements(self):
        """Parse actual ps output format offline; malformed, empty or nonpositive data cannot hide a missing sample."""
        self.assertEqual(parse_rss_bytes("  123456\n"), 123456 * 1024)
        for output in ("", "0", "-1", "rss", "12\n34", "1.5"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_rss_bytes(output)
        with self.assertRaises(ValueError):
            process_rss_bytes(0)

    def test_memory_monitor_stops_before_start_without_sampling(self):
        """A setup failure before Popen cannot hang the monitor or sample an unrelated process."""
        stopped = threading.Event()
        stopped.set()
        report = {"directory": str(self.root), "rss_samples": [], "rss_errors": []}
        server = harness.Server(self.root)
        self.experiment._sample_memory(server, stopped, report)
        self.assertEqual(report["rss_samples"], [])
        self.assertEqual(report["rss_errors"], [])
        self.assertFalse((self.root / "it16_rss.jsonl").exists())

    def test_unrun_experiment_and_missing_fixture_fail_without_servers(self):
        """Missing evidence never passes; setup failures still print a final IT-16 FAIL line."""
        self.assertFalse(all(self.experiment.check().values()))
        (self.root / "village_sabotage").rmdir()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertFalse(self.experiment.run())
        self.assertIn("BAD", output.getvalue())
        self.assertIn("FAIL IT-16:", output.getvalue())
        self.assertFalse((self.root / "it16_soak.json").exists())

    def test_real_file_replay_and_inclusive_resource_bounds_pass(self):
        """Unaligned fixture time still yields the exact expected cadence and all decisions reused."""
        self.populate_evidence()
        self.assertTrue(all(self.experiment.check().values()), self.experiment.check())
        first = self.experiment.runtimes[self.experiment.names[0]]
        self.assertEqual(len(first.hashes), DAY_TICKS // CHECK_EVERY)
        self.assertEqual(first.hashes[0]["gameTime"], 1_200)

    def test_decision_and_hash_mutations_fail_independently(self):
        """Changing recorded evidence is detected by comparison, without computing another digest."""
        self.populate_evidence()
        replay = self.experiment.runtimes[self.experiment.names[1]]
        changed = DecisionLog(self.root / "changed.jsonl")
        for index, decision in enumerate(replay.log.decisions()):
            changed.append(replace(decision, outcome="different outcome") if index == 0 else decision)
        replay.log = changed
        checks = self.experiment.check()
        self.assertFalse(checks["identical nonempty decision traces"])
        self.assertTrue(checks["identical complete world-hash traces"])
        replay.hashes[-1] = {"gameTime": 24_000, "hash": {"world": "different server result"}}
        self.assertFalse(self.experiment.check()["identical complete world-hash traces"])

    def test_equal_missing_or_incomplete_hashes_do_not_pass(self):
        """Both sides losing the same checkpoint, or all checkpoints, cannot masquerade as an exact replay."""
        self.populate_evidence()
        for runtime in self.experiment.runtimes.values():
            runtime.hashes.pop()
        self.assertFalse(self.experiment.check()["identical complete world-hash traces"])
        for runtime in self.experiment.runtimes.values():
            runtime.hashes.clear()
        self.assertFalse(self.experiment.check()["identical complete world-hash traces"])

    def test_resource_failure_partial_completion_and_replay_calls_fail(self):
        """Excess RSS/bytes, missing samples, sampler errors, a short run and a replay call each reject the gate."""
        self.populate_evidence()
        name = self.experiment.names[1]
        report = self.experiment.reports[name]
        report["rss_samples"][0]["rss_bytes"] = MAX_RSS_BYTES + 1
        report["bytes"] = MAX_RUN_BYTES + 1
        report["end"] -= 40
        replay = self.experiment.runtimes[name]
        replay.stats.model_calls = 1
        checks = self.experiment.check()
        self.assertFalse(checks[f"{name}: sampled peak RSS <= {MAX_RSS_BYTES} bytes"])
        self.assertFalse(checks[f"{name}: run directory <= {MAX_RUN_BYTES} bytes"])
        self.assertFalse(checks[f"{name}: full duration and clean harness shutdown"])
        self.assertFalse(checks["replay made zero model calls and reused every decision"])
        report["rss_errors"].append("ps unavailable")
        self.assertFalse(self.experiment.check()[f"{name}: JVM RSS sampled successfully every minute"])
        self.assertFalse(self.experiment.check()["no harness, replay, sampling or filesystem failure"])
        report["rss_errors"].clear()
        report["rss_samples"].clear()
        self.assertFalse(self.experiment.check()[f"{name}: JVM RSS sampled successfully every minute"])


if __name__ == "__main__":
    unittest.main()
