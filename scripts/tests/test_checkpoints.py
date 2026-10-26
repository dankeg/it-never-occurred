"""Checkpoint scheduling, saved state and resume bookkeeping, using the existing in-process village world."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.agent import Agent, RoleBrief
from orchestrator.decisions import DecisionLog
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.runtime import SimulationRuntime
from orchestrator.scenario import OutcomeTracker, ScenarioManifest
from scripts import harness, village_session
from scripts.run_study import Study
from scripts.test_m5_village import M5VillageExperiment
from scripts.run_village import prepare_resume
from scripts.tests.test_village_session import ROOT, SessionWorld
from scripts.village_session import CHECK_EVERY, Checkpoints, VillageSession


class CheckpointsTest(unittest.TestCase):
    """Checkpoint writes follow the outcome grid and retain the first save at a tick."""

    def setUp(self):
        """Provide a temporary world directory and a session that never enters a server context."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.session = VillageSession(self.root / "run", self.root / "source", self.manifest, {})
        (self.session.directory / "world").mkdir(parents=True)
        (self.session.directory / "world" / "level.dat").write_text("saved world", encoding="utf-8")

    def test_interval_must_be_positive_and_aligned(self):
        """Reject zero, negative and off-grid intervals; allow any positive outcome-grid multiple."""
        for interval in (0, -CHECK_EVERY, 20, CHECK_EVERY + 20):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "positive multiple"):
                Checkpoints(interval)
        self.assertEqual(Checkpoints(2 * CHECK_EVERY).every, 2 * CHECK_EVERY)

    def test_checkpoints_save_after_checks_and_skip_partial_end(self):
        """At tick 2,400 the saved tracker includes its first true predicate; tick 2,500 adds no checkpoint."""
        world = SessionWorld({}, game_time=620)
        self.session.client = world
        tracker = OutcomeTracker({"present": {"blocks_present": [{"pos": [0, 64, 0], "block": "minecraft:air"}]}})
        runtime = SimulationRuntime(world, [], None, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError)
        with patch.object(harness, "clone_apfs", side_effect=shutil.copytree) as clone:
            self.assertEqual(self.session.run_checked(runtime, tracker, 620, 2_500, Checkpoints(2_400)), 2_500)
        checkpoint = self.session.directory / "checkpoints" / "2400"
        clone.assert_called_once_with(self.session.directory / "world", checkpoint / "world")
        self.assertEqual(sorted(path.name for path in checkpoint.iterdir()), ["orchestrator", "world"])
        self.assertEqual(sorted(path.name for path in (checkpoint / "orchestrator").iterdir()),
                         ["first_true.json", "fork_state.json"])
        self.assertEqual(json.loads((checkpoint / "orchestrator" / "first_true.json").read_text()),
                         {"present": 1_200})
        self.assertEqual(SimulationRuntime.load_agents(checkpoint / "orchestrator" / "fork_state.json"), (2_400, []))
        self.assertEqual([path.name for path in checkpoint.parent.iterdir()], ["2400"])
        self.assertEqual([command for command, _ in world.requests].count("save_snapshot"), 1)
        with patch.object(harness, "clone_apfs") as clone:
            Checkpoints(2_400).save(self.session, runtime, OutcomeTracker({}), 2_400)
        clone.assert_not_called()
        self.assertEqual(json.loads((checkpoint / "orchestrator" / "first_true.json").read_text()),
                         {"present": 1_200})

    def test_latest_selects_numeric_ticks_at_or_before_limit(self):
        """Select by integer time rather than name, ignoring files and unrelated directories."""
        root = self.session.directory / "checkpoints"
        for name in ("1200", "9600", "12000", "notes"):
            (root / name).mkdir(parents=True)
        (root / "24000").touch()
        self.assertEqual(Checkpoints.latest(self.session.directory), root / "12000")
        self.assertEqual(Checkpoints.latest(self.session.directory, 12_000), root / "12000")
        self.assertEqual(Checkpoints.latest(self.session.directory, 11_000), root / "9600")
        with self.assertRaisesRegex(ValueError, "at or before tick 1000"):
            Checkpoints.latest(self.session.directory, 1_000)

    def test_bare_checkpoint_clone_assembles_server_files_without_starting(self):
        """Only checkpoint world and orchestrator directories are cloned into a newly assembled server."""
        checkpoint = self.root / "checkpoint"
        for name in ("world", "orchestrator"):
            (checkpoint / name).mkdir(parents=True)
        session = VillageSession(self.root / "resumed", checkpoint, self.manifest, {}, fresh=False)

        def assemble(name, runs_dir):
            """Stand in for the file-only server assembler, not a server process."""
            directory = runs_dir / name
            directory.mkdir()
            (directory / "fabric-server-launch.jar").touch()
            return directory

        with patch.object(village_session, "setup_server", side_effect=assemble) as setup, \
                patch.object(harness, "clone_apfs", side_effect=shutil.copytree) as clone, \
                patch.object(harness, "Server") as server:
            session.clone()
        setup.assert_called_once_with("resumed", self.root)
        self.assertEqual(clone.call_count, 2)
        self.assertTrue((session.directory / "world").is_dir())
        self.assertTrue((session.directory / "orchestrator").is_dir())
        server.assert_not_called()

    def test_restored_agents_and_timing_reuse_decisions_only_to_crash(self):
        """Restore an append-only prompt epoch and thinking delay, replay the suffix, then call the model anew."""
        world = SessionWorld({"alice": [0, 64, 0]}, game_time=620)
        self.session.client = world
        agent = Agent(RoleBrief("alice", "worker", "Works.", "Help."), epoch_turns=3)
        model = ScriptedClient(lambda system, user: ToolCall("open_container", {"x": 1, "y": 64, "z": 0}))
        runtime = SimulationRuntime(world, [agent], model, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError, min_gap=100)
        tracker = OutcomeTracker({})
        tracker.first_true["fair_ready"] = 600
        with patch.object(harness, "clone_apfs", side_effect=shutil.copytree):
            self.session.run_checked(runtime, tracker, 620, 1_200, Checkpoints(1_200))
        saved_world = copy.deepcopy(world)
        runtime.run(1_320)
        checkpoint = Checkpoints.latest(self.session.directory)
        tick, agents = SimulationRuntime.load_agents(checkpoint / "orchestrator" / "fork_state.json")
        self.assertEqual(tick, 1_200)
        restored = SimulationRuntime(saved_world, agents, model, self.root / "resumed" / "orchestrator",
                                     command_error=harness.CommandError, min_gap=100,
                                     source=DecisionLog(runtime.log.path))
        restored.restore_timing(checkpoint / "orchestrator" / "fork_state.json")
        restored.run(1_440)
        self.assertEqual([(decision.game_time, decision.source) for decision in restored.log.decisions()],
                         [(1_240, "replay"), (1_360, "model")])
        self.assertEqual(restored.stats.model_calls, 1)
        self.assertEqual(restored.stats.replayed, 1)

    def test_checkpointing_keeps_decisions_and_hashes_unchanged(self):
        """Saving snapshots adds no agent observations, prompt changes or world-hash samples."""
        traces, hashes = [], []
        for name, checkpoints in (("plain", None), ("checked", Checkpoints(1_200))):
            world = SessionWorld({"alice": [0, 64, 0]})
            session = VillageSession(self.root / name, self.root / "source", self.manifest, {})
            (session.directory / "world").mkdir(parents=True)
            session.client = world
            agent = Agent(RoleBrief("alice", "worker", "Works.", "Help."), epoch_turns=3)
            runtime = SimulationRuntime(world, [agent], ScriptedClient(lambda system, user: ToolCall("wait",
                                        {"ticks": 40})), session.directory / "orchestrator",
                                        command_error=harness.CommandError, hash_region={"min": [0, 64, 0],
                                        "max": [1, 65, 1]}, hash_every=CHECK_EVERY)
            with patch.object(harness, "clone_apfs", side_effect=shutil.copytree):
                session.run_checked(runtime, OutcomeTracker({}), 0, 2_500, checkpoints)
            traces.append([(decision.game_time, decision.digest, decision.tool_call, decision.outcome)
                           for decision in runtime.log.decisions()])
            hashes.append(runtime.hashes)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(hashes[0], hashes[1])

    def test_village_resume_restores_tracker_and_writes_normal_reports(self):
        """The village runner restores its checkpoint and consumes matching suffix decisions without a server."""
        directory = self.session.directory
        world = SessionWorld({role["name"]: role["spawn"] for role in self.manifest.roles}, game_time=620,
                             containers={tuple(container["pos"]): []
                                         for container in self.manifest.data["containers"].values()})
        self.session.client = world
        model = ScriptedClient(lambda system, user: ToolCall("open_container", {"x": 1, "y": 64, "z": 0}))
        runtime = SimulationRuntime(world, self.manifest.agents(3), model, directory / "orchestrator",
                                    command_error=harness.CommandError, min_gap=100)
        tracker = OutcomeTracker({})
        tracker.first_true["fair_ready"] = 600
        with patch.object(harness, "clone_apfs", side_effect=shutil.copytree):
            self.session.run_checked(runtime, tracker, 620, 1_200, Checkpoints(1_200))
        saved_world = copy.deepcopy(world)
        saved_world.requests.clear()
        runtime.run(1_320)
        checkpoint = prepare_resume(directory, 1_440)
        source = DecisionLog(checkpoint.parent.parent / "orchestrator" / "decisions.jsonl")
        args = harness.experiment_parser("").parse_args(["--runs-dir", str(self.root)])
        args.variant = "sabotage"

        with patch.object(village_session, "setup_server", side_effect=lambda name, runs_dir:
                          (runs_dir / name).mkdir()), \
                patch.object(harness, "clone_apfs", side_effect=shutil.copytree), \
                patch.object(harness, "Server") as server:
            server.return_value.__enter__.return_value.client = saved_world
            resumed, restored_tracker, report = M5VillageExperiment(args).run_once(
                "run", model, source, end_tick=1_440, min_gap=100, resume_from=checkpoint)
        self.assertEqual(restored_tracker.first_true["fair_ready"], 600)
        self.assertEqual(report["end"], 1_440)
        self.assertEqual(resumed.stats.replayed, 6)
        self.assertEqual(resumed.stats.model_calls, 6)
        prefix = {decision.game_time for decision in source.decisions() if decision.game_time < 1_200}
        self.assertTrue(prefix)  # the resumed log carries the run's decisions before the checkpoint
        self.assertEqual({decision.game_time for decision in resumed.log.decisions()}, prefix | {1_240, 1_360})
        self.assertTrue((directory / "m5_report.json").is_file())
        self.assertTrue((directory / "scenario.json").is_file())
        self.assertTrue((directory / "metrics.json").is_file())
        self.assertFalse(any(command in {"config_set", "structures_set"} for command, _ in saved_world.requests))


class ResumeTest(unittest.TestCase):
    """Refusals leave the original run intact, while a valid resume retains its files under the aside name."""

    def test_finished_run_is_refused_before_renaming(self):
        """A final report forbids resuming even if checkpoints exist."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            (run / "checkpoints" / "1200").mkdir(parents=True)
            (run / "m5_report.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finished run"):
                prepare_resume(run, 2_400)
            self.assertTrue(run.exists())
            self.assertEqual(len(list(run.parent.iterdir())), 1)

    def test_run_without_checkpoint_is_refused_before_renaming(self):
        """Neither a missing checkpoint directory nor an empty one changes the unfinished run."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            for empty_root in (False, True):
                with self.subTest(empty_root=empty_root), self.assertRaisesRegex(ValueError, "No checkpoint"):
                    if empty_root:
                        (run / "checkpoints").mkdir()
                    prepare_resume(run, 2_400)
                self.assertTrue(run.exists())

    def test_resume_keeps_old_decisions_and_returns_relocated_checkpoint(self):
        """A valid resume renames rather than deletes its source, selecting the latest usable tick."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            for tick in (1_200, 2_400, 3_600):
                (run / "checkpoints" / str(tick)).mkdir(parents=True)
            (run / "orchestrator").mkdir()
            (run / "orchestrator" / "decisions.jsonl").write_text("old decisions\n", encoding="utf-8")
            checkpoint = prepare_resume(run, 3_000)
            self.assertEqual(checkpoint.name, "2400")
            self.assertTrue(checkpoint.is_dir())
            self.assertTrue(checkpoint.parent.parent.name.startswith("run_unfinished_"))
            self.assertFalse(run.exists())
            self.assertEqual((checkpoint.parent.parent / "orchestrator" / "decisions.jsonl").read_text(),
                             "old decisions\n")

    def test_study_checkpoint_option_keeps_existing_default(self):
        """Studies leave checkpointing disabled unless the interval is explicitly supplied."""
        args = harness.experiment_parser("").parse_args([])
        args.variant, args.study, args.fork_day, args.end_day = "sabotage", "road_closure", 1, 2
        self.assertIsNone(Study(args)._checkpoints)
        args.checkpoint_every = 2_400
        self.assertEqual(Study(args)._checkpoints.every, 2_400)


if __name__ == "__main__":
    unittest.main()
