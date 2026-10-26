"""Pin village setup, first views, and checkpoint requests without starting a server."""

import copy
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.render import ObservationRenderer
from orchestrator.runtime import BudgetExceeded, SimulationRuntime
from orchestrator.scenario import OutcomeTracker, ScenarioManifest
from scripts import harness
from scripts.run_narrated_village import NarratedVillage
from scripts.test_m5_village import M5VillageExperiment
from scripts.tests.fake_world import FakeWorld
from scripts.village_session import VillageSession

ROOT = Path(__file__).resolve().parents[2]


class SessionWorld(FakeWorld):
    """Record village setup commands in addition to FakeWorld's runtime and observation requests."""

    def request(self, command: str, args: dict | None = None) -> dict:
        """Accept only the session's setup commands beyond the usual scripted world protocol."""
        if command in {"lockstep", "await_entities", "recording", "restore_bots", "config_set", "structures_set",
                       "save_snapshot"}:
            self.requests.append((command, copy.deepcopy(args or {})))
            return {}
        return super().request(command, args)


class VillageSessionTest(unittest.TestCase):
    """Shared village operations preserve the exact protocol order and checkpoint boundaries."""

    def setUp(self):
        """Create a configuration-only session and a scripted world, without cloning or entering its context."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.session = VillageSession(self.root / "run", self.root / "source", self.manifest, {})
        self.world = SessionWorld({role["name"]: role["spawn"] for role in self.manifest.roles}, game_time=620)

    def test_fresh_and_fork_setup_requests(self):
        """Only fresh sessions configure exposure and structures, after the shared lockstep and bot restore."""
        shared = [("lockstep", {"enabled": True}), ("await_entities", harness.FORCELOADED_CHUNKS),
                  ("recording", {"enabled": True}), ("ping", {}), ("restore_bots", {})]
        for fresh in (True, False):
            with self.subTest(fresh=fresh):
                world = SessionWorld({}, game_time=620)
                session = VillageSession(self.root / "run", self.root / "source", self.manifest, {}, fresh=fresh)
                session._setup(world)
                expected = shared + ([("config_set", {"exposure_log": 0}),
                                      ("structures_set", {"structures": self.manifest.structures()})] if fresh else [])
                self.assertEqual(world.requests, expected)
                self.assertIs(session.client, world)
                self.assertEqual(session.start, 620)

    def test_first_views_observe_roles_in_manifest_order(self):
        """Initial views render one terrain-free observation per role, without additional requests."""
        self.session._setup(self.world)
        self.world.requests.clear()
        views = self.session.first_views()
        self.assertEqual(list(views), [role["name"] for role in self.manifest.roles])
        self.assertEqual(self.world.requests,
                         [("observe", {"name": role["name"], "include_terrain": False})
                          for role in self.manifest.roles])
        expected_world = FakeWorld(self.world.bots, game_time=620)
        renderer = ObservationRenderer()
        for role in self.manifest.roles:
            self.assertEqual(views[role["name"]], renderer.render(expected_world.request("observe",
                                                                                      {"name": role["name"]})))

    def test_run_checked_preserves_requests_at_checkpoints_and_partial_end(self):
        """From tick 620, check at 1,200, 2,400, and the partial end 2,500, always after the runtime."""
        self.session.client = self.world
        runtime = SimulationRuntime(self.world, [], None, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError)
        tracker = OutcomeTracker({"missing": {"blocks_present": [{"pos": [1, 64, 0], "block": "minecraft:stone"}]}})
        end = self.session.run_checked(runtime, tracker, 620, 2_500)
        check = ("blocks", {"positions": [[1, 64, 0]]})
        expected = ([("ping", {}), ("step", {"n": 20})] + [("step", {"n": 40})] * 14 + [check]
                    + [("ping", {})] + [("step", {"n": 40})] * 30 + [check]
                    + [("ping", {})] + [("step", {"n": 40})] * 2 + [("step", {"n": 20}), check])
        self.assertEqual(self.world.requests, expected)
        self.assertEqual((end, self.world.game_time), (2_500, 2_500))
        self.assertEqual(tracker.first_true, {})

    def test_run_checked_keeps_first_true_and_does_not_repeat_completed_queries(self):
        """A true query records its first checkpoint only, even when the runtime continues."""
        self.session.client = self.world
        runtime = SimulationRuntime(self.world, [], None, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError)
        tracker = OutcomeTracker({"present": {"blocks_present": [{"pos": [1, 64, 0], "block": "minecraft:air"}]}})
        self.session.run_checked(runtime, tracker, 620, 2_500)
        self.assertEqual(tracker.first_true, {"present": 1_200})
        self.assertEqual([request for request in self.world.requests if request[0] == "blocks"],
                         [("blocks", {"positions": [[1, 64, 0]]})])

    def test_run_checked_at_or_after_end_makes_no_requests(self):
        """A completed interval neither runs the runtime nor checks outcomes."""
        self.session.client = self.world
        runtime = SimulationRuntime(self.world, [], None, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError)
        tracker = OutcomeTracker({"missing": {"blocks_present": [{"pos": [1, 64, 0], "block": "minecraft:stone"}]}})
        for now in (620, 1_200):
            with self.subTest(now=now):
                self.assertEqual(self.session.run_checked(runtime, tracker, now, 620), now)
        self.assertEqual(self.world.requests, [])

    def test_run_checked_leaves_budget_handling_to_runner(self):
        """A model-call budget stop propagates before checking an incomplete checkpoint."""
        agent = Agent(RoleBrief("alice", "worker", "Works.", "Help."))
        world = FakeWorld({"alice": [0, 64, 0]})
        self.session.client = world
        model = ScriptedClient(lambda system, user: ToolCall("wait", {"ticks": 40}))
        runtime = SimulationRuntime(world, [agent], model, self.session.directory / "orchestrator",
                                    command_error=harness.CommandError, max_model_calls=0)
        tracker = OutcomeTracker({"missing": {"blocks_present": [{"pos": [1, 64, 0], "block": "minecraft:stone"}]}})
        with self.assertRaises(BudgetExceeded):
            self.session.run_checked(runtime, tracker, 0, 1_200)
        self.assertEqual(world.game_time, 0)
        self.assertFalse(any(command == "blocks" for command, args in world.requests))
        self.assertEqual(tracker.first_true, {})

    def test_existing_directory_is_refused_before_cloning_or_starting(self):
        """Cloning an existing run fails locally even when the source and server files do not exist."""
        self.session.directory.mkdir()
        with self.assertRaisesRegex(FileExistsError, f"Refusing to overwrite {self.session.directory}"):
            self.session.clone()
        self.assertIsNone(self.session.client)
        self.assertIsNone(self.session.start)
        self.assertFalse((self.root / "source").exists())

    def test_missing_source_fails_before_starting(self):
        """A missing source raises before any APFS operation, server startup, or setup request."""
        with self.assertRaises(FileNotFoundError) as caught:
            self.session.clone()
        self.assertEqual(Path(caught.exception.filename), self.root / "source")
        self.assertIsNone(self.session.client)
        self.assertIsNone(self.session.start)
        self.assertFalse(self.session.directory.exists())

    def test_runners_refuse_existing_runs(self):
        """M5 and narrated runs refuse an existing run directory before starting anything."""
        self.session.directory.mkdir()
        args = harness.experiment_parser("").parse_args(["--runs-dir", str(self.root)])
        args.name, args.variant = "run", "sabotage"
        experiment, village = M5VillageExperiment(args), NarratedVillage(args)
        for runner, run in (("m5", lambda: experiment.run_once("run", None, None)), ("narrated", village.run)):
            with self.subTest(runner=runner):
                with self.assertRaisesRegex(FileExistsError, f"Refusing to overwrite {self.session.directory}"):
                    run()
