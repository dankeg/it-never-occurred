"""Unit tests for the study runner: paired statistics, outcome bookkeeping, resume, effects and run configuration."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.agent import Agent, RoleBrief
from orchestrator.llm import ToolCall
from orchestrator.scenario import ScenarioManifest
from scripts.run_study import Study, load_outcome, main, paired_ci, replace_brief, save_outcome, set_aside
from scripts.tests.fake_world import FakeWorld
from scripts.village_session import VillageSession

ROOT = Path(__file__).resolve().parents[2]


class PairedEffectTest(unittest.TestCase):
    """Effects are mean paired differences with a bootstrap CI over pairs."""

    def test_paired_ci(self):
        self.assertEqual(paired_ci([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), (1.0, 1.0, 1.0))
        effect, low, high = paired_ci([0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(effect, 0.5)
        self.assertTrue(0.0 <= low <= 0.5 <= high <= 1.0)
        with self.assertRaises(ValueError):
            paired_ci([1.0], [1.0, 2.0])


class StudyBookkeepingTest(unittest.TestCase):
    """Finished runs are recorded and reloaded; unfinished ones are set aside; effects cover outcome times and roles."""

    def test_outcome_round_trip_and_set_aside(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "study_x_control_1"
            run.mkdir()
            self.assertIsNone(load_outcome(run))
            save_outcome(run, {"welfare": {1: 0.9}, "first_true": {}, "roles": {}})
            self.assertEqual(load_outcome(run)["welfare"], {1: 0.9})
            moved = set_aside(run)
            self.assertFalse(run.exists())
            self.assertTrue(moved.name.startswith("study_x_control_1_unfinished_"))

    def test_effects_on_outcome_times_and_roles(self):
        study = Study.__new__(Study)
        study._fork, study._end = 24_000, 48_000
        study._manifest = ScenarioManifest({"roles": [{"name": "builder"}]})
        outcome = lambda welfare, fair, paid: {"welfare": {1: welfare}, "first_true": {"fair_open": fair},  # noqa: E731
                                               "roles": {"builder": paid}}
        control = [outcome(1.0, 30_000, False), outcome(0.9, None, None)]
        treated = [outcome(0.8, 26_000, True), outcome(0.9, 40_000, True)]
        effects = study._effects(control, treated)
        self.assertAlmostEqual(effects["welfare_day_1"][0], -0.1)
        self.assertEqual(effects["fair_open_tick"][0], -6_000.0)  # never opened counts as the end time
        self.assertEqual(effects["builder_success"][0], 1.0)  # only the pair where both runs have a verdict


class PairedIntervalBoundaryTest(unittest.TestCase):
    """Known paired differences give exact bootstrap means and percentile endpoints."""

    def test_equal_differences_collapse_the_interval(self):
        """Equal differences collapse the mean and both interval endpoints to that difference."""
        self.assertEqual(paired_ci([2.0, 5.0, 9.0], [0.0, 3.0, 7.0]), (-2.0, -2.0, -2.0))

    def test_zero_and_one_differences_have_zero_to_one_interval(self):
        """Differences zero and one give mean one-half and the exact interval zero to one."""
        self.assertEqual(paired_ci([2.0, 4.0], [2.0, 5.0]), (0.5, 0.0, 1.0))


class StudyPersistenceTest(unittest.TestCase):
    """Outcome reloads restore integer days and unfinished directories retain all contents."""

    def test_loading_json_restores_integer_welfare_day_keys(self):
        """Loading a recorded outcome converts every welfare day key back to an integer."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            outcome = {"welfare": {"0": 0.75, "2": 1.0}, "first_true": {"fair_open": None},
                       "roles": {"builder": False}}
            (run / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
            self.assertEqual(load_outcome(run), {**outcome, "welfare": {0: 0.75, 2: 1.0}})

    def test_set_aside_renames_the_directory_without_losing_contents(self):
        """Setting aside a run moves its nested files intact into a sibling directory."""
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "unfinished"
            (run / "orchestrator").mkdir(parents=True)
            (run / "orchestrator/decisions.jsonl").write_text('{"saved": true}\n', encoding="utf-8")
            (run / "partial.bin").write_bytes(b"\x00\xffpartial")
            moved = set_aside(run)
            self.assertFalse(run.exists())
            self.assertEqual(moved.parent, Path(folder))
            self.assertTrue(moved.name.startswith("unfinished_unfinished_"))
            self.assertEqual((moved / "orchestrator/decisions.jsonl").read_text(), '{"saved": true}\n')
            self.assertEqual((moved / "partial.bin").read_bytes(), b"\x00\xffpartial")
            self.assertEqual(sorted(path.relative_to(moved).as_posix() for path in moved.rglob("*")),
                             ["orchestrator", "orchestrator/decisions.jsonl", "partial.bin"])


class CompletedStudyResumeTest(unittest.TestCase):
    """A fully completed study resumes from outcome files and writes its paired report."""

    PREFIX = "study_no_saboteur_sabotage"

    @classmethod
    def finished_study(cls, root: Path, roles: list[dict]) -> tuple[dict, dict]:
        """Save a finished factual run, its fork, and two finished pairs under `root`; return the factual and
        intervention outcomes."""
        (root / f"{cls.PREFIX}_fork").mkdir()
        factual = {"welfare": {0: 1.0, 1: 0.5, 2: 0.75},
                   "first_true": {"bridge_complete": 30000, "fair_open": None},
                   "roles": {role["name"]: False for role in roles},
                   "calls": 4, "decisions": 10, "replayed": 6}
        treated = {**factual, "welfare": {0: 1.0, 1: 0.75, 2: 1.0},
                   "first_true": {"bridge_complete": 26000, "fair_open": 60000},
                   "roles": {role["name"]: True for role in roles},
                   "calls": 1, "decisions": 10, "replayed": 9}
        for suffix, outcome in (("factual", factual), ("control_1", factual),
                                ("intervention_0", treated), ("intervention_1", treated)):
            directory = root / f"{cls.PREFIX}_{suffix}"
            directory.mkdir()
            save_outcome(directory, outcome)
        return factual, treated

    def test_finished_branches_resume_without_server_setup(self):
        """Saved factual, control and intervention outcomes produce the complete report without a server."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(runs_dir=root, variant="sabotage", study="no_saboteur", fork_day=1,
                                      end_day=3, branches=2, parallel=2, model="scripted")
            study = Study(args)
            prefix = self.PREFIX
            factual, treated = self.finished_study(root, study._manifest.roles)
            report = study.run()
            self.assertEqual(report["pairsFinished"], args.branches)
            self.assertEqual(report["branches"], 2)
            self.assertIsNone(report["stopped"])
            self.assertEqual(report["factual"], factual)
            self.assertEqual(report["model_calls"], {"control": 4, "intervention": 2})
            self.assertEqual(report["decisions"], {"control": 10, "intervention": 20})
            self.assertEqual(report["calls_saved_by_gating"], {"control": 0.6, "intervention": 0.9})
            self.assertEqual(report["effects"], {
                "welfare_day_1": (0.25, 0.25, 0.25), "welfare_day_2": (0.25, 0.25, 0.25),
                "bridge_complete_tick": (-4000.0, -4000.0, -4000.0),
                "fair_ready_tick": (0.0, 0.0, 0.0), "fair_open_tick": (-12000.0, -12000.0, -12000.0),
                **{f"{role['name']}_success": (1.0, 1.0, 1.0) for role in study._manifest.roles}})
            self.assertEqual(report["branches_detail"], {
                "control": [{**factual, "calls": 0, "decisions": 0, "replayed": 0}, factual],
                "intervention": [treated, treated]})
            self.assertEqual(json.loads((root / f"{prefix}_report.json").read_text()),
                             json.loads(json.dumps(report)))
            self.assertEqual(sorted(path.name for path in root.iterdir()), sorted([
                f"{prefix}_{suffix}" for suffix in ("factual", "fork", "control_1", "intervention_0",
                                                    "intervention_1", "report.json")]))

    def test_studies_sharing_a_base_reuse_its_factual_run_and_controls(self):
        """With `base`, a second study pairs its own interventions with the base's factual run and controls."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(runs_dir=root, variant="sabotage", study="no_saboteur", fork_day=1,
                                      end_day=3, branches=2, parallel=2, model="scripted")
            factual, treated = self.finished_study(root, Study(args)._manifest.roles)  # the base: PREFIX
            other = "study_honest_freight_sabotage"
            for suffix in ("intervention_0", "intervention_1"):
                (root / f"{other}_{suffix}").mkdir()
                save_outcome(root / f"{other}_{suffix}", treated)
            report = Study(argparse.Namespace(**{**vars(args), "study": "honest_freight", "base": self.PREFIX})).run()
            self.assertEqual((report["base"], report["pairsFinished"], report["factual"]), (self.PREFIX, 2, factual))
            self.assertEqual(report["branches_detail"]["control"][1], factual)
            self.assertTrue((root / f"{other}_report.json").exists())
            self.assertFalse((root / f"{other}_factual").exists() or (root / f"{other}_control_1").exists())

    def test_main_prints_the_resumed_report(self):
        """The command line resumes a finished study and prints its effects and call savings."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.finished_study(root, ScenarioManifest.load(ROOT / "scenario/village_sabotage.json").roles)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--study", "no_saboteur", "--runs-dir", str(root), "--end-day", "3",
                               "--branches", "2", "--parallel", "2"])
        printed = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(sorted(printed), ["branches", "calls_saved_by_gating", "decisions", "effects", "model_calls",
                                           "study", "variant"])
        self.assertEqual(printed["effects"]["welfare_day_2"], [0.25, 0.25, 0.25])
        self.assertEqual(printed["calls_saved_by_gating"], {"control": 0.6, "intervention": 0.9})


class StudyRunConfigurationTest(unittest.TestCase):
    """Study runs check outcomes at every checkpoint, and the model and prompt epochs follow the options."""

    @staticmethod
    def study(**options) -> Study:
        """A road-closure study on the sabotage village, with `options` over scripted defaults."""
        defaults = {"variant": "sabotage", "study": "road_closure", "fork_day": 1, "end_day": 2, "model": "scripted",
                    "provider": "anthropic", "epoch_turns": None, "max_calls": None}
        return Study(argparse.Namespace(**{**defaults, **options}))

    def test_runs_hash_and_check_outcomes_at_every_checkpoint(self):
        """A study run records world hashes and checks the bridge and the fair every 1,200 ticks, keeping the first
        checkpoint at which each was true."""
        study = self.study()
        manifest = study._manifest
        fair = manifest.data["success_predicates"]["fair_ready"]["all"]
        bell, lantern = fair[0]["blocks_present"][0], fair[1]["block_beside"]
        x, y, z = lantern["pos"]
        world = FakeWorld({role["name"]: role["spawn"] for role in manifest.roles},
                          blocks={tuple(bell["pos"]): bell["block"], (x + 1, y, z): lantern["block"]})
        tracker = study._tracker()
        with tempfile.TemporaryDirectory() as folder:
            session = VillageSession(Path(folder), ROOT / "runs/village_sabotage", manifest, {})
            session.client = world
            runtime = study._runtime(world, manifest.agents(study._epoch_turns()), Path(folder), None)
            end = session.run_checked(runtime, tracker, 0, 2_400)
        self.assertEqual(end, 2_400)
        self.assertEqual(tracker.first_true, {"fair_ready": 1_200})  # ready on day 0; it opens only on day 5
        self.assertEqual([record["gameTime"] for record in runtime.hashes], [1_200, 2_400])
        self.assertGreater(runtime.stats.decisions, 0)

    def test_model_and_epochs_follow_the_options(self):
        """The scripted model reads full prompts (one-turn epochs); a live model gets eight unless told otherwise."""
        scripted = self.study()
        self.assertEqual((scripted._model().model, scripted._epoch_turns()), ("scripted", 1))
        live = self.study(model="qwen", provider="openai-compatible", base_url="http://127.0.0.1:9/v1",
                          temperature=1.0, reasoning_effort=None)
        self.assertEqual((live._model().model, live._epoch_turns()), ("qwen", 8))
        self.assertEqual(self.study(epoch_turns=3)._epoch_turns(), 3)


class StudyEffectCoverageTest(unittest.TestCase):
    """Effects cover post-fork welfare, never-completed outcomes and paired role verdicts."""

    def test_effects_exclude_prefork_days_and_unpaired_role_verdicts(self):
        """Only post-fork days and paired verdicts count, with missing outcome times replaced by the end tick."""
        study = Study(argparse.Namespace(variant="sabotage", study="no_saboteur", fork_day=1, end_day=3))
        control = [
            {"welfare": {0: 0.0, 1: 0.5, 2: 0.75}, "first_true": {"fair_open": 30000, "bridge_complete": None},
             "roles": {"builder": False}},
            {"welfare": {0: 0.0, 1: 0.5, 2: 0.75}, "first_true": {"fair_open": None, "bridge_complete": 36000},
             "roles": {"builder": None, "mayor": True}}]
        treated = [
            {"welfare": {0: 1.0, 1: 0.75, 2: 0.5}, "first_true": {"fair_open": 26000, "bridge_complete": 60000},
             "roles": {"builder": True, "mayor": False}},
            {"welfare": {0: 1.0, 1: 0.75, 2: 0.5}, "first_true": {"fair_open": 40000, "bridge_complete": None},
             "roles": {"builder": True, "mayor": None}}]
        self.assertEqual(study._effects(control, treated), {
            "welfare_day_1": (0.25, 0.25, 0.25), "welfare_day_2": (-0.25, -0.25, -0.25),
            "bridge_complete_tick": (12000.0, -12000.0, 36000.0),
            "fair_ready_tick": (0.0, 0.0, 0.0), "fair_open_tick": (-18000.0, -32000.0, -4000.0),
            "builder_success": (1.0, 1.0, 1.0)})


class StudyBriefReplacementTest(unittest.TestCase):
    """Interventions change only an existing agent's private brief."""

    def test_replacement_preserves_the_agent_public_brief_and_memory(self):
        """Replacing a private goal preserves agent identity, public knowledge and existing memory."""
        agent = Agent(RoleBrief("ben", "builder", "public goal", "old secret", "town map"))
        agent.memory.record_action(0, ToolCall("wait", {"ticks": 20}), "started")
        memory = agent.memory.to_json()
        original = agent.brief
        self.assertIs(replace_brief(agent, None), agent)
        self.assertIs(agent.brief, original)
        self.assertIs(replace_brief(agent, "new secret"), agent)
        self.assertEqual(agent.brief, RoleBrief("ben", "builder", "public goal", "new secret", "town map"))
        self.assertEqual(original.private, "old secret")
        self.assertEqual(agent.memory.to_json(), memory)


if __name__ == "__main__":
    unittest.main()
