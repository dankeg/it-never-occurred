"""Unit tests for seeded skill cases, response schemas, effects, and fork validation."""

import contextlib
import copy
from collections import Counter
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from scripts import harness, test_e2_skills, test_s1


def status(skill="wait", state="succeeded", skill_id="skill-1", started=100, ended=105, args=None):
    """Build a skill-status fixture with the requested identity and state."""
    value = {"skillId": skill_id, "skill": skill, "args": args or {"ticks": 5}, "state": state,
             "progress": {}, "startedAt": started}
    if state not in {"running", "idle"}:
        value["endedAt"] = ended
    if state in {"failed", "cancelled"}:
        value["reason"] = "interrupted"
    return value


def component(time, digest="a"):
    """Return fake component hashes at the supplied time with the chosen digest."""
    return {"gameTime": time, **{key: digest * 64 for key in harness.HASH_KEYS}}


def record(time):
    """Wrap fake component hashes in the sampled-record format."""
    return {"gameTime": time, "hash": component(time)}


def result(case, *, accepted=True, succeeded=True):
    """Build a case-result fixture with chosen acceptance and success values."""
    return {"case": asdict(case), "dry": {"feasible": accepted}, "real": {"accepted": accepted},
            "status": status(case.skill) if accepted else None, "agreement": True,
            "dry_run_pure": True, "success": accepted and succeeded, "errors": []}


class CaseGenerationTests(unittest.TestCase):
    """Exercise fixed-seed coverage and explicit negative skill cases."""
    def test_seed_twenty_per_ten_skills_and_extra_negative_cases(self):
        """Verify seed twenty per ten skills and extra negative cases."""
        cases = test_s1.generate_cases()
        self.assertEqual(cases, test_s1.generate_cases())
        self.assertEqual(test_s1.SEED, 20261016)
        self.assertEqual(Counter(case.skill for case in cases if case.generated),
                         {skill: 20 for skill in test_s1.SKILLS})
        self.assertEqual(len(test_s1.SKILLS), 10)
        self.assertEqual(len(cases), 214)
        self.assertEqual(len({case.label for case in cases}), len(cases))
        self.assertTrue(all(case.expected_reason is None for case in cases if case.generated))
        self.assertEqual(Counter(case.skill for case in cases if not case.generated),
                         {"move_to": 2, "mine": 3, "craft": 2, "place": 3, "place_section": 1,
                          "chest_put": 1, "chest_take": 1, "give": 1})

    def test_required_scenarios_and_loaded_targets(self):
        """Verify required scenarios and loaded targets."""
        cases = test_s1.generate_cases()
        for skill, fixtures in {"move_to": {"wall_gap", "step", "sealed", "water"},
                                "craft": {"table", "no_table", "missing"},
                                "chest_put": {"allowed", "denied"}, "chest_take": {"allowed", "denied"}}.items():
            self.assertTrue(fixtures <= {case.fixture for case in cases if case.skill == skill})
        for case in cases:
            if "x" in case.args:
                self.assertTrue(-32 <= case.args["x"] < 32)
                self.assertTrue(-32 <= case.args["z"] < 32)
            if case.skill == "move_to" and case.generated:
                self.assertLessEqual(((case.args["x"] - 0.5) ** 2 + (case.args["z"] - 0.5) ** 2) ** 0.5, 40)
        self.assertTrue({"dirt", "stone", "oak_log", "oak_planks"} <=
                        {case.fixture for case in cases if case.skill == "mine"})

    def test_schematic_is_exact_supported_order_three_by_one_by_five(self):
        """Verify schematic is exact supported order three by one by five."""
        blocks = test_s1.deck_blocks()
        self.assertEqual(len(blocks), 15)
        self.assertEqual({block["dx"] for block in blocks}, {0, 1, 2})
        self.assertEqual({block["dy"] for block in blocks}, {0})
        self.assertEqual({block["dz"] for block in blocks}, set(range(5)))
        self.assertEqual([block["dx"] for block in blocks[:3]], [0, 1, 2])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            test_s1.write_schematic(directory, "deck")
            self.assertEqual(json.loads((directory / "world/simharness/schematics/deck.json").read_text()),
                             {"blocks": blocks})


class SchemaTests(unittest.TestCase):
    """Exercise dry-run, real-start, status, and inventory response validation."""
    def test_dry_run_requires_real_boolean_and_complete_estimate_even_when_rejected(self):
        """Verify dry run requires real boolean and complete estimate even when rejected."""
        valid = {"feasible": True, "estimate": {"ticks": 10, "materials": {"minecraft:dirt": 1}}}
        self.assertTrue(test_s1.validate_dry(valid))
        self.assertFalse(test_s1.validate_dry({**valid, "feasible": False, "reason": "missing_items"}))
        invalid = [{}, {**valid, "feasible": 1}, {**valid, "estimate": {}},
                   {**valid, "estimate": {"ticks": True, "materials": {}}},
                   {**valid, "estimate": {"ticks": 0, "materials": {"dirt": 1}}},
                   {**valid, "estimate": {"ticks": 0, "materials": {"minecraft:dirt": -1}}},
                   {**valid, "feasible": False}, {**valid, "feasible": False, "reason": "unknown"}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(harness.ProtocolError):
                test_s1.validate_dry(value)

    def test_real_start_and_status_cannot_default_to_success(self):
        """Verify real start and status cannot default to success."""
        for value in ({}, {"accepted": 1}, {"accepted": True}, {"accepted": True, "skillId": True},
                      {"accepted": False}, {"accepted": False, "reason": "engine_error"}):
            with self.subTest(value=value), self.assertRaises(harness.ProtocolError):
                test_s1.validate_start(value)
        self.assertTrue(test_s1.validate_start({"accepted": True, "skillId": "1"}))
        self.assertFalse(test_s1.validate_start({"accepted": False, "reason": "acl_denied"}))
        test_s1.validate_status(status(), skill="wait", skill_id="skill-1")
        for field, value in (("state", "done"), ("state", []), ("skillId", True), ("progress", []), ("args", []),
                             ("startedAt", True), ("endedAt", 99)):
            with self.subTest(field=field), self.assertRaises(harness.ProtocolError):
                test_s1.validate_status({**status(), field: value})
        with self.assertRaises(harness.ProtocolError):
            test_s1.validate_status(status(), skill="mine")
        with self.assertRaises(harness.ProtocolError):
            test_s1.validate_status(status(), skill_id="another")
        failed = status(state="failed")
        failed.pop("reason")
        with self.assertRaises(harness.ProtocolError):
            test_s1.validate_status(failed)

    def test_inventory_schema_rejects_duplicate_slots_and_boolean_counts(self):
        """Verify inventory schema rejects duplicate slots and boolean counts."""
        entries = [{"slot": 0, "item": "minecraft:dirt", "count": 3}]
        self.assertEqual(test_s1.item_counts(entries), {"minecraft:dirt": 3})
        for value in ({}, entries * 2, [{**entries[0], "count": True}], [{**entries[0], "slot": -1}]):
            with self.subTest(value=value), self.assertRaises(harness.ProtocolError):
                test_s1.item_counts(value)


class S1DriverTests(unittest.TestCase):
    """Exercise feasibility agreement, dry-run purity, and physical effects."""
    def test_dry_then_real_has_only_read_barriers_and_no_ticks_between(self):
        """Verify dry then real has only read barriers and no ticks between."""
        case = test_s1.Case("wait", "wait/1", {"ticks": 5})
        calls = []

        def request(command, args=None):
            """Handle a scripted control request using the enclosing test fixture."""
            calls.append((command, copy.deepcopy(args)))
            if command == "ping":
                return {"gameTime": 100}
            if command == "hash":
                return component(100)
            if command == "skill_start":
                if args.get("dry_run"):
                    return {"feasible": True, "estimate": {"ticks": 5, "materials": {}}}
                return {"accepted": True, "skillId": "skill-1"}
            raise AssertionError(command)

        client = Mock()
        client.request.side_effect = request
        with patch.object(test_s1, "prepare_case", return_value={}), \
                patch.object(test_s1, "finish_skill", return_value=status()), \
                patch.object(test_s1, "verify_effect"):
            outcome = test_s1.run_case(client, case)
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["errors"], [])
        self.assertEqual([command for command, _args in calls],
                         ["ping", "hash", "skill_start", "ping", "hash", "skill_start"])
        self.assertEqual(calls[2][1], {"name": "alice", "skill": "wait", "args": {"ticks": 5}, "dry_run": True})
        self.assertEqual(calls[-1][1], {"name": "alice", "skill": "wait", "args": {"ticks": 5}})

    def invoke_case(self, dry, real, *, terminal=None, mutation=False):
        """Run a case with the supplied dry-run, start, and terminal fixtures."""
        client = Mock()
        client.request.side_effect = [dry, real]
        with patch.object(test_s1, "prepare_case", return_value={}), \
                patch.object(test_s1, "fingerprint",
                             side_effect=[component(100), component(100, "b" if mutation else "a")]), \
                patch.object(test_s1, "finish_skill", return_value=terminal or status()), \
                patch.object(test_s1, "verify_effect"):
            return test_s1.run_case(client, test_s1.Case("wait", "wait/1", {"ticks": 5}))

    def test_mutation_disagreement_bad_schema_and_failure_reasons_are_retained(self):
        """Verify mutation disagreement bad schema and failure reasons are retained."""
        dry = {"feasible": True, "estimate": {"ticks": 5, "materials": {}}}
        accepted = {"accepted": True, "skillId": "skill-1"}
        mutated = self.invoke_case(dry, accepted, mutation=True)
        self.assertFalse(mutated["dry_run_pure"])
        self.assertIn("dry-run mutated", str(mutated["errors"]))
        mismatch = self.invoke_case(dry, {"accepted": False, "reason": "timeout"})
        self.assertFalse(mismatch["agreement"])
        self.assertTrue(mismatch["errors"])
        self.assertTrue(self.invoke_case({"feasible": 1}, accepted)["errors"])
        failed = self.invoke_case(dry, accepted, terminal=status(state="failed"))
        self.assertFalse(failed["success"])
        self.assertEqual(failed["status"]["reason"], "interrupted")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            test_s1.print_report([failed], {})
        self.assertIn("reason=interrupted", output.getvalue())

    def test_polling_is_bounded_and_exactly_twenty_ticks(self):
        """Verify polling is bounded and exactly twenty ticks."""
        client = Mock()
        client.request.side_effect = [{"gameTime": 100}, status(state="running"), {"gameTime": 120}, status()]
        self.assertEqual(test_s1.finish_skill(client, "alice", "wait", "skill-1"), status())
        self.assertIn(call("step", {"n": 20}), client.request.call_args_list)
        client.request.side_effect = [{"gameTime": 100}, status(state="running"), {"gameTime": 120},
                                      status(state="running")]
        with self.assertRaisesRegex(harness.HarnessError, "gate_timeout"):
            test_s1.finish_skill(client, "alice", "wait", "skill-1", cap=20)

    def test_collect_prerequisite_mines_not_summons_and_short_deck_gets_fourteen(self):
        """Verify collect prerequisite mines not summons and short deck gets fourteen."""
        client = Mock()
        client.request.return_value = {"inventory": []}
        with patch.object(test_s1, "reset_arena"), patch.object(test_s1, "start_and_finish") as mine, \
                patch.object(test_s1, "item_entities", return_value=[{"type": "minecraft:item"}]):
            test_s1.prepare_case(client, test_s1.Case("collect", "collect", {"radius": 6}, "dirt"))
        mine.assert_called_once_with(client, "alice", "mine", {"x": 0, "y": -60, "z": 3})
        self.assertFalse(any(entry.args[0] in {"give", "bot_break"} for entry in client.request.call_args_list))
        self.assertNotIn("summon", str(client.request.call_args_list))
        with patch.object(test_s1, "reset_arena"), patch.object(test_s1, "supply") as supply:
            test_s1.prepare_case(client, test_s1.Case("place_section", "short", {}, "short"))
        supply.assert_called_once_with(client, "minecraft:oak_planks", 14)

    def test_reset_guarantees_nonempty_fills_without_mass_transient_block_changes(self):
        """Verify reset guarantees nonempty fills without mass transient block changes."""
        client = Mock()

        def request(command, args=None):
            """Handle a scripted control request using the enclosing test fixture."""
            if command == "bot_state":
                return {"inventory": []}
            if command == "dump_entities":
                return {"entities": []}
            return {"gameTime": 120 if command == "step" else 100}

        client.request.side_effect = request
        test_s1.reset_arena(client)
        commands = [entry.args[1]["command"] for entry in client.request.call_args_list if entry.args[0] == "command"]
        self.assertEqual(commands[:4], ["setblock -32 -60 -32 minecraft:barrier",
            "fill -32 -60 -32 31 -55 31 minecraft:air", "setblock -32 -61 -32 minecraft:barrier",
            "fill -32 -61 -32 31 -61 31 minecraft:stone"])
        self.assertFalse(any(text.startswith("fill") and "barrier" in text for text in commands))

    def test_reports_enforce_exact_case_coverage_and_ninety_five_percent(self):
        """Verify reports enforce exact case coverage and ninety five percent."""
        cases = test_s1.generate_cases()
        rows = [result(case, accepted=case.expected_reason is None) for case in cases]
        self.assertTrue(all(value["passed"] for value in test_s1.summarize(rows, cases).values()))
        moves = [row for row in rows if row["case"]["skill"] == "move_to" and row["real"]["accepted"]]
        moves[0]["success"] = False
        self.assertTrue(test_s1.summarize(rows, cases)["move_to"]["passed"])
        moves[1]["success"] = False
        self.assertFalse(test_s1.summarize(rows, cases)["move_to"]["passed"])
        self.assertFalse(test_s1.summarize(rows[:-1], cases)["give"]["passed"])
        self.assertFalse(test_s1.summarize(rows + [rows[0]], cases)["move_to"]["passed"])
        rows[30]["agreement"] = False
        self.assertFalse(test_s1.summarize(rows, cases)["mine"]["passed"])
        rows[50]["dry_run_pure"] = False
        self.assertFalse(test_s1.summarize(rows, cases)["collect"]["passed"])
        rejected = [result(case, accepted=False) for case in cases]
        self.assertFalse(any(value["passed"] for value in test_s1.summarize(rejected, cases).values()))

    def test_s1_main_uses_fresh_clone_lockstep_barrier_recording_and_writes_report(self):
        """Verify s1 main uses fresh clone lockstep barrier recording and writes report."""
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            directory = runs / "s1"
            client = Mock()
            client.request.return_value = {"gameTime": 100}
            server = contextlib.nullcontext(Mock(client=client))
            with patch.object(harness, "clone_apfs", side_effect=lambda source, target: target.mkdir()) as clone, \
                    patch.object(harness, "Server", return_value=server), \
                    patch.object(test_s1, "run_case",
                                 side_effect=lambda control, case: result(case,
                                                                         accepted=case.expected_reason is None)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(test_s1.main(["--runs-dir", str(runs)]), 0)
            clone.assert_called_once_with(runs / "base", directory)
            self.assertEqual(client.request.call_args_list[:4], [call("lockstep", {"enabled": True}),
                call("await_entities", harness.FORCELOADED_CHUNKS), call("recording", {"enabled": True}), call("ping")])
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(len(report["cases"]), 214)
            self.assertTrue(report["passed"])

    def test_s1_rejects_a_different_seed_before_touching_runs(self):
        """Verify s1 rejects a different seed before touching runs."""
        with patch.object(harness, "clone_apfs") as clone, patch.object(harness, "Server") as server:
            with self.assertRaisesRegex(ValueError, "20261016"):
                test_s1.main(["--base-seed", "42"])
        clone.assert_not_called()
        server.assert_not_called()


class ForkValidationTests(unittest.TestCase):
    """Exercise saved skill state, restore checks, and fork comparison."""
    def snapshot(self):
        """Return the fake bot snapshot used by fork validation."""
        snapshot = {name: status(skill, "running", skill_id=name, started=1100)
                    for name, skill in test_e2_skills.SNAPSHOT_SKILLS.items()}
        snapshot["bob"]["progress"] = {"placed": 3, "total": 15}
        return snapshot

    def test_snapshot_requires_both_skills_running_with_real_partial_progress(self):
        """Verify snapshot requires both skills running with real partial progress."""
        snapshot = self.snapshot()
        test_e2_skills.validate_snapshot(snapshot, 1120)
        test_e2_skills.validate_restore(snapshot, copy.deepcopy(snapshot), 1120)
        invalid = [dict(alice=snapshot["alice"]),
                   {**snapshot, "alice": status("move_to", "succeeded", ended=1120)},
                   {**snapshot, "alice": {**snapshot["alice"], "startedAt": 1120}}]
        for placed in (0, 15, True):
            modified = copy.deepcopy(snapshot)
            modified["bob"]["progress"]["placed"] = placed
            invalid.append(modified)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(harness.HarnessError):
                test_e2_skills.validate_snapshot(value, 1120)
        changed = copy.deepcopy(snapshot)
        changed["alice"]["progress"]["waypoint"] = 1
        with self.assertRaisesRegex(harness.HarnessError, "differs"):
            test_e2_skills.validate_restore(snapshot, changed, 1120)

    def test_full_suffix_every_component_every_sample_and_no_empty_trace_pass(self):
        """Verify full suffix every component every sample and no empty trace pass."""
        start, fork = 100, 1120
        full = [record(time) for time in range(120, 4101, 20)]
        suffix = [entry for entry in full if entry["gameTime"] > fork]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(test_e2_skills.compare_suffix(full, suffix, start, fork))
        self.assertIn("149 exact full-component", output.getvalue())
        for actual in ([], suffix[:-1], suffix[1:], list(reversed(suffix)), suffix + [record(4120)]):
            with self.subTest(length=len(actual)), self.assertRaises(harness.ProtocolError):
                test_e2_skills.compare_suffix(full, actual, start, fork)
        for key in harness.HASH_KEYS:
            modified = copy.deepcopy(suffix)
            modified[-1]["hash"][key] = "b" * 64
            with self.subTest(key=key), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(test_e2_skills.compare_suffix(full, modified, start, fork))
            self.assertIn("4100", output.getvalue())
        malformed = copy.deepcopy(suffix)
        malformed[0]["hash"].pop("harness")
        with self.assertRaises(harness.ProtocolError):
            test_e2_skills.compare_suffix(full, malformed, start, fork)

    def test_schedule_is_repeatable_mixed_and_has_identical_unshifted_suffix(self):
        """Verify schedule is repeatable mixed and has identical unshifted suffix."""
        schedule = test_e2_skills.schedule()
        self.assertEqual(schedule, test_e2_skills.schedule())
        self.assertEqual({skill for entries in schedule.values() for _name, skill, _args in entries},
                         {"move_to", "mine", "craft", "place_section"})  # no collect after mine (db7c958)
        self.assertEqual({name: skill for name, skill, _args in schedule[1000]},
                         {"alice": "move_to", "bob": "place_section"})
        self.assertTrue(all(time % 20 == 0 and 0 <= time < 4000 for time in schedule))
        suffix_a = [entry for time in range(1020, 4000, 20) for entry in schedule.get(time, [])]
        suffix_c = [entry for time in range(1020, 4000, 20) for entry in test_e2_skills.schedule().get(time, [])]
        self.assertEqual(suffix_a, suffix_c)

    def test_segment_rejects_wrong_step_hashes_and_running_skill_replacement(self):
        """Verify segment rejects wrong step hashes and running skill replacement."""
        client = Mock()
        client.request.return_value = {"gameTime": 100}
        running = {"alice": status("mine", "running"), "bob": {"state": "idle"}}
        with patch.object(test_e2_skills, "statuses", return_value=running):
            client.request.side_effect = [{"gameTime": 100}, running["alice"]]
            with self.assertRaisesRegex(harness.HarnessError, "replace unfinished"):
                test_e2_skills.run_segment(client, 100, 0, 20, [], [])
        client.request.side_effect = [{"gameTime": 120}, {"gameTime": 139, "hashes": []}]
        with patch.object(test_e2_skills, "statuses", return_value=running), self.assertRaises(harness.ProtocolError):
            test_e2_skills.run_segment(client, 100, 20, 40, [], [])

    def invoke_fork(self, runs, *, changed_restore=False):
        """Run the fork gate with scripted traces and optional restore mutation."""
        events = []
        snapshot = self.snapshot()

        class Control:
            """Provide scripted control responses for the enclosing skill fork test."""
            def __init__(control, directory):
                """Initialize this test fixture without launching a real server."""
                control.directory = directory
                control.current = 100 if directory.name.endswith("_a") else 1120

            def request(control, command, args=None):
                """Handle a scripted control request using the enclosing test fixture."""
                events.append((directory_name := control.directory.name, command, control.current, copy.deepcopy(args)))
                if command in {"ping", "lockstep", "recording", "await_entities"}:
                    return {"gameTime": control.current}
                if command == "skill_status":
                    value = copy.deepcopy(snapshot[args["name"]])
                    if changed_restore and directory_name.endswith("_c"):
                        value["progress"]["restored_wrong"] = True
                    if control.current == 4100:
                        value.update(state="succeeded", endedAt=4100)
                    return value
                if command == "bot_state":
                    return {"pos": [-20.5, -60, -16.5]}
                return {}

        def make_server(directory, **options):
            """Return the scripted server fixture for a run directory."""
            directory = Path(directory)
            control = Control(directory)

            class Server:
                """Expose the enclosing scripted client through a server context fixture."""
                client = control

                def __enter__(server):
                    """Enter the existing fixture context and return its managed object."""
                    events.append((directory.name, "start", control.current, options))
                    return server

                def __exit__(server, *_exc):
                    """Apply the existing fixture cleanup without changing exception handling."""
                    events.append((directory.name, "stop", control.current, None))

            return Server()

        def clone(source, destination):
            """Simulate the clone operation expected by the enclosing test."""
            events.append((Path(destination).name, "clone", str(source), None))
            Path(destination).mkdir()

        def create_shell(name, runs_dir):
            """Create the empty destination shell used by the fork fixture."""
            destination = Path(runs_dir) / name
            destination.mkdir()
            return destination

        def segment(client, start, begin, end, hashes, launches):
            """Append scripted hashes and skill launches for a run segment."""
            events.append((client.directory.name, "segment", begin, end))
            self.assertEqual(client.current, start + begin)
            hashes.extend(record(time) for time in range(start + begin + 20, start + end + 1, 20))
            for elapsed, entries in test_e2_skills.schedule().items():
                if begin <= elapsed < end:
                    launches.extend({"gameTime": start + elapsed, "name": name, "skill": skill, "args": args,
                                     "result": {"accepted": True, "skillId": f"{name}:{elapsed}"}}
                                    for name, skill, args in entries)
            client.current = start + end

        with patch.object(harness, "Server", side_effect=make_server), \
                patch.object(harness, "clone_apfs", side_effect=clone), \
                patch.object(test_e2_skills, "setup_server", side_effect=create_shell), \
                patch.object(test_e2_skills, "run_segment", side_effect=segment), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = test_e2_skills.main(["--runs-dir", str(runs)])
        return exit_code, events, output.getvalue()

    def test_main_forks_live_frozen_world_then_restores_in_a_separate_server(self):
        """Verify main forks live frozen world then restores in a separate server."""
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            code, events, output = self.invoke_fork(runs)
            self.assertEqual(code, 0, output)
            starts = [entry[0] for entry in events if entry[1] == "start"]
            self.assertEqual(starts, ["e2_skills_a", "e2_skills_c"])
            cloned = [entry for entry in events if entry[1] == "clone"]
            self.assertEqual([entry[2] for entry in cloned], [str(runs / "base"), str(runs / "e2_skills_a/world")])
            save_index = next(index for index, entry in enumerate(events) if entry[1] == "save_snapshot")
            clone_index = events.index(cloned[-1])
            stop_index = next(index for index, entry in enumerate(events) if entry[:2] == ("e2_skills_a", "stop"))
            self.assertLess(save_index, clone_index)
            self.assertLess(clone_index, stop_index)
            self.assertEqual([entry[1] for entry in events[save_index + 1:clone_index]], ["ping"])
            restored = [entry for entry in events if entry[0] == "e2_skills_c"]
            self.assertFalse(any(entry[1] in {"spawn_bot", "give", "command", "skill_start"} for entry in restored))
            self.assertEqual(sum(entry[1] == "restore_bots" for entry in restored), 1)
            self.assertLess(next(index for index, entry in enumerate(restored) if entry[1] == "restore_bots"),
                            next(index for index, entry in enumerate(restored) if entry[1] == "segment"))
            report = json.loads((runs / "e2_skills_a/report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["snapshot"], report["restored"])
            self.assertEqual(len(json.loads((runs / "e2_skills_a/hashes.json").read_text())), 200)
            self.assertEqual(len(json.loads((runs / "e2_skills_c/hashes.json").read_text())), 149)

    def test_main_preserves_failure_report_and_partial_hashes_when_restore_is_wrong(self):
        """Verify main preserves failure report and partial hashes when restore is wrong."""
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            code, events, output = self.invoke_fork(runs, changed_restore=True)
            self.assertEqual(code, 1)
            self.assertIn("Restored skill state/status differs", output)
            self.assertFalse(any(entry[:2] == ("e2_skills_c", "segment") for entry in events))
            report = json.loads((runs / "e2_skills_a/report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["errors"])
            self.assertEqual(len(json.loads((runs / "e2_skills_a/hashes.json").read_text())), 200)
            self.assertEqual(json.loads((runs / "e2_skills_c/hashes.json").read_text()), [])

    def test_existing_gate_directories_are_never_overwritten_or_started(self):
        """Verify existing gate directories are never overwritten or started."""
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            for module, directory in ((test_s1, "s1"), (test_e2_skills, "e2_skills_c")):
                (runs / directory).mkdir()
                with patch.object(harness, "Server") as server, patch.object(harness, "clone_apfs") as clone, \
                        patch.object(test_e2_skills, "setup_server") as setup:
                    with self.assertRaises(FileExistsError):
                        module.main(["--runs-dir", str(runs)])
                server.assert_not_called()
                clone.assert_not_called()
                setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
