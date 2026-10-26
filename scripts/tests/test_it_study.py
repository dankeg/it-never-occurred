"""IT-05 artifact checks, interruption readiness and deliberate wrong-control detection without servers."""

from dataclasses import asdict, replace
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from orchestrator.decisions import Decision, DecisionLog
from scripts import harness
from scripts.test_it05_study import (PREFIX, StudyExperiment, budget_decisions_complete, corrupt_reused_args,
                                     decision_log, interrupt_target, reuse_matches, snapshot, untouched)


class StudyIntegrityTest(unittest.TestCase):
    """Synthetic real DecisionLogs exercise the exact checks used by the integration experiment."""

    def setUp(self):
        """Keep all synthetic logs and reports in one temporary directory, with automatic cleanup."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def decision(agent: str = "builder", tick: int = 0, source: str = "model") -> Decision:
        """Create an ordinary scripted decision with explicit prompt identity and a meaningful tool argument."""
        return Decision(tick, agent, "same-prompt", False, source, "wait", {"ticks": 1200}, "scripted", 0, 0, "prompt")

    def log(self, name: str, records: list[Decision]) -> DecisionLog:
        """Persist records through the production append-only log before exercising gate checks."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        log = DecisionLog(path)
        for record in records:
            log.append(record)
        return decision_log(path)

    def experiment(self, *options: str) -> StudyExperiment:
        """Parse the real shared gate options without calling setup or starting any subprocess."""
        args = harness.experiment_parser("unit").parse_args(["--runs-dir", str(self.root), *options])
        return StudyExperiment(args)

    def test_replay_requires_matching_agent_tick_digest_tool_and_args(self):
        """Changing any replay identity field fails even though every other decision still matches."""
        original = self.decision()
        control = self.log("control.jsonl", [original])
        for index, changes in enumerate(({"agent": "mayor"}, {"game_time": 40}, {"digest": "other-prompt"},
                                         {"tool": "move_to"}, {"args": {"ticks": 1}})):
            with self.subTest(changes=changes):
                intervention = self.log(f"bad_{index}.jsonl", [replace(original, source="replay", **changes)])
                self.assertFalse(reuse_matches(control, intervention))
        self.assertTrue(reuse_matches(control, self.log("good.jsonl", [replace(original, source="replay")])))

    def test_model_decisions_need_not_match_but_replay_cannot_be_vacuous(self):
        """Fresh intervention decisions are legal, but empty, model-only and forced-only arms cannot prove reuse."""
        original = self.decision()
        control = self.log("control.jsonl", [original])
        mixed = self.log("mixed.jsonl", [replace(original, source="replay"),
                                        replace(original, game_time=40, args={"ticks": 1})])
        self.assertTrue(reuse_matches(control, mixed))
        for name, records in (("empty", []), ("model", [original]), ("forced", [replace(original, source="forced")])):
            with self.subTest(name=name):
                self.assertFalse(reuse_matches(control, self.log(f"{name}.jsonl", records)))

    def test_negative_control_changes_only_args_of_a_reused_record(self):
        """Corrupt a non-first record actually used by intervention 1 and leave the real control untouched."""
        control = self.log("control_1.jsonl", [self.decision(), self.decision(tick=40)])
        intervention = self.log("intervention_1.jsonl", [self.decision(tick=40, source="replay")])
        saved = snapshot([control.path])
        destination = self.root / "wrong_control_1.jsonl"
        corrupt_reused_args(control.path, intervention, destination)
        damaged = decision_log(destination)
        self.assertTrue(untouched(saved))
        self.assertTrue(reuse_matches(control, intervention))
        self.assertFalse(reuse_matches(damaged, intervention))
        before = [asdict(record) for record in control.decisions()]
        after = [asdict(record) for record in damaged.decisions()]
        self.assertEqual(after[0], before[0])
        self.assertNotEqual(after[1].pop("args"), before[1].pop("args"))
        self.assertEqual(after[1], before[1])

    def test_negative_control_refuses_invalid_or_absent_replay(self):
        """A preexisting mismatch or an empty replay cannot count as evidence the deliberate corruption was caught."""
        control = self.log("control.jsonl", [self.decision()])
        intervention = self.log("intervention.jsonl", [replace(self.decision(source="replay"), digest="wrong")])
        with self.assertRaisesRegex(ValueError, "valid own-control replay"):
            corrupt_reused_args(control.path, intervention, self.root / "bad.jsonl")

    def test_finished_files_require_both_bytes_and_nanosecond_mtimes(self):
        """Equal bytes with a changed mtime, changed bytes with the original mtime, and missing files all fail."""
        path = self.root / "outcome.json"
        path.write_bytes(b"first")
        saved = snapshot([path])
        self.assertTrue(untouched(saved))
        stamp = saved[path][1]
        os.utime(path, ns=(stamp, stamp + 1000000))
        self.assertFalse(untouched(saved))
        path.write_bytes(b"other")
        os.utime(path, ns=(stamp, stamp))
        self.assertFalse(untouched(saved))
        path.unlink()
        self.assertFalse(untouched(saved))
        self.assertFalse(untouched({}))

    def test_snapshot_survives_an_intact_unfinished_directory_rename(self):
        """Relocating an unfinished run must preserve the captured raw logs and their mtimes."""
        log = self.log("unfinished/orchestrator/decisions.jsonl", [self.decision()])
        saved = snapshot([log.path])
        source, destination = self.root / "unfinished", self.root / "unfinished_archived"
        source.rename(destination)
        self.assertTrue(untouched({destination / path.relative_to(source): value for path, value in saved.items()}))

    def test_jsonl_validation_rejects_truncation_blank_lines_and_duplicate_keys(self):
        """Every physical decision line must parse, and duplicate agent/tick records cannot silently collapse."""
        log = self.log("valid.jsonl", [self.decision()])
        data = log.path.read_text(encoding="utf-8")
        for index, invalid in enumerate((data + "{", data + "\n", data + data)):
            with self.subTest(index=index):
                path = self.root / f"invalid_{index}.jsonl"
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    decision_log(path)

    def budget_artifacts(self) -> tuple[Path, DecisionLog]:
        """Write two completed decisions, an idle poll and the next decision-bearing poll that hits the budget."""
        records = [self.decision(), self.decision("mayor"), self.decision(tick=80), self.decision("mayor", 160)]
        reference = self.log("reference.jsonl", records)
        directory = self.root / "budget_control_1"
        self.log("budget_control_1/orchestrator/decisions.jsonl", records[:2])
        output = directory / "orchestrator"
        polls = [{"agent": agent, "t": tick} for tick in (0, 40, 80) for agent in ("builder", "mayor")]
        (output / "trajectory.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in polls), encoding="utf-8")
        reference_polls = polls + [{"agent": agent, "t": 160} for agent in ("builder", "mayor")]
        (reference.path.parent / "trajectory.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in reference_polls), encoding="utf-8")
        (output / "stats.json").write_text(json.dumps({"decisions": 2, "model_calls": 2}), encoding="utf-8")
        return directory, reference

    def test_budget_counts_completed_decisions_not_raw_trajectory_observations(self):
        """Six agent observations include an idle poll and a rejected batch, but only two decisions completed."""
        directory, reference = self.budget_artifacts()
        self.assertTrue(budget_decisions_complete(directory, reference))

    def test_budget_detects_lost_decision_even_when_stats_are_also_truncated(self):
        """A self-consistent shortened log and stats still fail comparison with completed reference polls."""
        directory, reference = self.budget_artifacts()
        output = directory / "orchestrator"
        path = output / "decisions.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        (output / "stats.json").write_text(json.dumps({"decisions": 1, "model_calls": 1}), encoding="utf-8")
        self.assertFalse(budget_decisions_complete(directory, reference))

    def test_budget_requires_stats_and_trajectory_to_account_for_every_decision(self):
        """Wrong call counts and missing decision-bearing observations independently fail the retention check."""
        directory, reference = self.budget_artifacts()
        output = directory / "orchestrator"
        (output / "stats.json").write_text(json.dumps({"decisions": 2, "model_calls": 1}), encoding="utf-8")
        self.assertFalse(budget_decisions_complete(directory, reference))
        (output / "stats.json").write_text(json.dumps({"decisions": 2, "model_calls": 2}), encoding="utf-8")
        path = output / "trajectory.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[1:]), encoding="utf-8")
        self.assertFalse(budget_decisions_complete(directory, reference))

    def test_budget_detects_missing_idle_polls_or_role_observations(self):
        """Even polls with no decisions must contain every role and match the uninterrupted runtime's trajectory."""
        directory, reference = self.budget_artifacts()
        path = directory / "orchestrator/trajectory.jsonl"
        polls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("".join(json.dumps(record) + "\n" for record in polls if record["t"] != 40), encoding="utf-8")
        self.assertFalse(budget_decisions_complete(directory, reference))

    def test_signal_waits_for_control_one_and_active_unfinished_runtime(self):
        """A factual outcome or server directory alone cannot trigger SIGTERM before the first branch is saved."""
        factual = self.root / f"{PREFIX}_factual"
        factual.mkdir()
        (factual / "outcome.json").write_text("{}", encoding="utf-8")
        active = self.root / f"{PREFIX}_intervention_0"
        (active / "orchestrator").mkdir(parents=True)
        self.assertIsNone(interrupt_target(self.root))
        control = self.root / f"{PREFIX}_control_1"
        control.mkdir()
        (control / "outcome.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(interrupt_target(self.root))
        self.log(f"{PREFIX}_intervention_0/orchestrator/decisions.jsonl", [self.decision()])
        trajectory = active / "orchestrator/trajectory.jsonl"
        trajectory.touch()
        self.assertIsNone(interrupt_target(self.root))
        trajectory.write_text('{"agent":"builder","t":0}\n', encoding="utf-8")
        self.assertEqual(interrupt_target(self.root), active)
        (active / "outcome.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(interrupt_target(self.root))

    def test_pair_checks_accept_identical_independently_sampled_controls(self):
        """Scripted controls need not differ, but control 1 must not claim replay or any other non-model source."""
        directory = self.root / "pairs"
        original = self.decision()
        for suffix in ("factual", "control_1", "intervention_0", "intervention_1"):
            decision = replace(original, source="replay") if suffix.startswith("intervention") else original
            self.log(f"pairs/{PREFIX}_{suffix}/orchestrator/decisions.jsonl", [decision])
        stats = directory / f"{PREFIX}_control_1/orchestrator/stats.json"
        stats.write_text(json.dumps({"decisions": 1, "model_calls": 1, "replayed": 0}), encoding="utf-8")
        experiment = self.experiment()
        experiment._check_pairs(directory)
        self.assertTrue(all(experiment._checks.values()))
        stats.write_text(json.dumps({"decisions": 1, "model_calls": 0, "replayed": 1}), encoding="utf-8")
        experiment._check_pairs(directory)
        self.assertFalse(experiment._checks["pairs control 1 samples every decision independently"])
        stats.write_text(json.dumps({"decisions": 1, "model_calls": 1, "replayed": 0}), encoding="utf-8")
        path = directory / f"{PREFIX}_control_1/orchestrator/decisions.jsonl"
        path.write_text(json.dumps(asdict(replace(original, source="replay"))) + "\n", encoding="utf-8")
        experiment._check_pairs(directory)
        self.assertFalse(experiment._checks["pairs control 1 samples every decision independently"])

    def test_check_cannot_pass_an_incomplete_experiment_or_failed_negative_control(self):
        """Only completed experiments with every invariant true print PASS and return success."""
        experiment = self.experiment()
        for completed, checks, expected in ((False, {"reuse": True}, False),
                                           (True, {"negative control": False}, False),
                                           (True, {"reuse": True, "negative control": True}, True)):
            with self.subTest(completed=completed, checks=checks):
                experiment._completed, experiment._checks = completed, checks
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(experiment.check(), expected)
                self.assertIn("PASS IT-05:" if expected else "FAIL IT-05:", output.getvalue())
                self.assertIn("BAD" if not all(checks.values()) else "ok ", output.getvalue())

    def test_command_keeps_factual_and_branch_servers_in_the_assigned_block(self):
        """The real subprocess CLI fixes two pairs, one worker and the minimal day range with no timing switches."""
        experiment = self.experiment("--port", "30000", "--game-port", "30001", "--heap", "2G")
        command = experiment._command(self.root / "case", 2)
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual({key: options[key] for key in ("--port", "--game-port", "--branch-port")},
                         {"--port": "30000", "--game-port": "30001", "--branch-port": "30000"})
        self.assertEqual((options["--branches"], options["--parallel"], options["--fork-day"], options["--end-day"]),
                         ("2", "1", "0", "1"))
        self.assertEqual((options["--model"], options["--max-calls"], options["--heap"]), ("scripted", "2", "2G"))

    def test_invalid_game_port_or_eight_port_block_is_refused(self):
        """Reject ports outside the gate's block instead of silently using a default or another gate's port."""
        for options in (("--port", "0"), ("--port", "65529"), ("--port", "30000", "--game-port", "31000")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.experiment(*options)


if __name__ == "__main__":
    unittest.main()
