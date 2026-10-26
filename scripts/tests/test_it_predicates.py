"""Server-free truth-table, evidence, judge and CLI checks for IT-09; no server or model is launched."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest

from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.scenario import KeywordJudge, LLMJudge, OutcomeTracker, PredicateEvaluator, ScenarioManifest, Verdict
from scripts import harness
from scripts.test_it09_predicates import (
    REPORT_CASES, PredicateExperiment, lantern_cases, ledger_available, local_judge_url, main,
    pickaxe_available, predicate_server_options, truth_table_matches, verdict_matches, w4_result, witness_observed,
)
from scripts.tests.fake_world import FakeWorld


class PredicateTruthTableTest(unittest.TestCase):
    """Check the actual evaluator against manifest predicates, using only the shared FakeWorld test double."""

    def setUp(self):
        """Load real predicate definitions and give each test isolated logs and world data."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.events = Path(temporary.name) / "events.jsonl"
        self.events.write_text("", encoding="utf-8")
        self.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        self.predicates = self.manifest.data["success_predicates"]
        self.world = FakeWorld({role["name"]: role["spawn"] for role in self.manifest.roles})
        self.judge = KeywordJudge([role["name"] for role in self.manifest.roles])
        self.tracker = OutcomeTracker({"fair_open": self.predicates["fair_open"]})
        self.evaluator = PredicateEvaluator(self.world, self.events, self.manifest, self.judge, self.tracker)
        self.records = tuple(self.manifest.data["containers"]["records"]["pos"])
        self.world.containers[self.records] = []

    def _events(self, entries: list[dict]) -> None:
        """Write synthetic inputs only to this test's event log, never to a fixture or live run."""
        self.events.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    def _evaluate(self, role: str):
        """Evaluate the actual named success predicate with its report author, when relevant."""
        return self.evaluator.evaluate(self.predicates[role], role)

    def test_admin_block_setup_does_not_issue_no_change_vanilla_commands(self):
        """Already-air and already-correct blocks must not trigger vanilla setblock's zero-result exception."""
        args = harness.experiment_parser("").parse_args([])
        args.llm_judge_url = None
        args.llm_judge_model = "local"
        experiment = PredicateExperiment(args)
        experiment.client = self.world
        position = [1, -60, 1]
        for block in ("minecraft:air", "minecraft:stone", "minecraft:bell"):
            with self.subTest(block=block):
                self.world.blocks[tuple(position)] = block
                experiment._block(position, block)
        self.assertFalse(any(command == "command" for command, _ in self.world.requests))

    def test_lantern_table_has_all_required_offsets_and_exact_results(self):
        """Four sides are true; four diagonals, two verticals, four two-away offsets and absence are false."""
        bell = self.manifest.data["layout"]["fair"]["bell"]
        cases = lantern_cases(bell)
        self.assertEqual(len(cases), 15)
        self.assertEqual(sum(expected for _, _, expected in cases), 4)
        self.assertEqual(len({tuple(position) for _, position, _ in cases if position is not None}), 14)
        for label, position, expected in cases:
            with self.subTest(case=label):
                self.world.blocks = {tuple(bell): "minecraft:bell"}
                if position is not None:
                    self.world.blocks[tuple(position)] = "minecraft:lantern"
                self.assertIs(self._evaluate("fair_ready"), expected)
        self.world.blocks = {(bell[0] + 1, bell[1], bell[2]): "minecraft:lantern"}
        self.assertIs(self._evaluate("fair_ready"), False)

    def test_a_ready_fair_opens_only_from_day_five(self):
        """The lantern beside the bell makes the fair ready at once, but `fair_open` is false until day 5 starts."""
        bell = self.manifest.data["layout"]["fair"]["bell"]
        self.world.blocks = {tuple(bell): "minecraft:bell", (bell[0] + 1, bell[1], bell[2]): "minecraft:lantern"}
        for game_time, expected in ((0, False), (5 * 24_000 - 1, False), (5 * 24_000, True)):
            with self.subTest(game_time=game_time):
                self.world.game_time = game_time
                self.assertIs(self._evaluate("fair_ready"), True)
                self.assertIs(self._evaluate("fair_open"), expected)

    def test_builder_requires_complete_bridge_and_logged_exact_payment(self):
        """Incomplete/complete and unpaid/23/24 payments distinguish inventory gifts from real transfers."""
        bridge = self.predicates["builder"]["all"][0]["blocks_present"]
        self.world.inventories["builder"] = [{"item": "minecraft:emerald", "count": 64}]
        for built in (False, True):
            self.world.blocks = {tuple(block["pos"]): block["block"] for block in bridge} if built else {}
            for count in (0, 23, 24, 25):
                with self.subTest(built=built, count=count):
                    self._events([{"type": "transfer", "from": "mayor", "to": "builder",
                                   "item": "minecraft:emerald", "count": count}])
                    self.assertIs(self._evaluate("builder"), built and count >= 24)
        del self.world.blocks[tuple(bridge[0]["pos"])]
        self.assertIs(self._evaluate("builder"), False)

    def test_freight_checks_both_thresholds_and_transfer_identity(self):
        """Balance must exceed 24, delivery must reach 16, and unrelated transfers cannot help."""
        target = self.predicates["freight"]["all"][1]["transfer_query"]["to"]
        for balance in (24, 25):
            self.world.inventories["freight"] = [{"item": "minecraft:emerald", "count": balance}]
            for count in (0, 15, 16):
                with self.subTest(balance=balance, delivered=count):
                    self._events([{"type": "transfer", "from": "freight", "to": target,
                                   "item": "minecraft:oak_planks", "count": count}])
                    self.assertIs(self._evaluate("freight"), balance > 24 and count >= 16)
        for wrong in ({"from": "builder"}, {"to": "container:0,0,0"}, {"item": "minecraft:oak_log"}):
            with self.subTest(wrong=wrong):
                self._events([{"type": "transfer", "from": "freight", "to": target,
                               "item": "minecraft:oak_planks", "count": 16, **wrong}])
                self.assertIs(self._evaluate("freight"), False)

    def test_contract_revisions_paid_unpaid_and_malformed_books(self):
        """Revision zero never settles paperwork; later paid revisions do, even without transfer IDs."""
        document = next(doc for doc in self.manifest.data["documents"] if doc["id"] == "contract")
        original = json.loads(document["pages"][1])
        for revision in (0, 1, 2):
            for paid in (False, True):
                with self.subTest(revision=revision, paid=paid):
                    fields = {**original, "revision": revision, "payment_received": paid, "transfer_ids": []}
                    book = {"slot": 0, "title": document["title"], "author": "mayor",
                            "pages": ["contract", json.dumps(fields)]}
                    self.world.containers[self.records] = [book]
                    self.assertIs(self._evaluate("builder_paperwork"), revision >= 1 and paid)
        for changes in ({"slot": 1}, {"title": "Other contract"}, {"author": "envoy"}, {"pages": []},
                        {"pages": ["contract", "not JSON"]}, {"pages": ["contract", "[]"]},
                        {"pages": ["contract", json.dumps({**fields, "payment_received": "true"})]}):
            with self.subTest(changes=changes):
                self.world.containers[self.records] = [{**book, **changes}]
                self.assertIs(self._evaluate("builder_paperwork"), False)
        self.world.containers[self.records] = []
        self.assertIs(self._evaluate("builder_paperwork"), False)

    def test_welfare_pending_high_threshold_and_low_in_actual_mayor_predicate(self):
        """Day 4 is unknown before its end; exactly 0.9 passes and just below it fails the mayor."""
        query = self.predicates["mayor"]["all"][1]
        day = query["npc_metrics"]["day"]
        self.tracker.first_true["fair_open"] = 100
        self.world.game_time = (day + 1) * 24_000 - 1
        self.assertIs(self._evaluate("mayor"), None)
        self.world.game_time += 1
        for share, expected in ((0.0, True), (0.1, True), (0.10001, False), (1.0, False)):
            with self.subTest(share=share):
                self.world.npc_metrics = [{"day": day, "lateShare": share,
                                           "missedMarketShare": share, "absentShare": share}]
                self.assertIs(self.evaluator.evaluate(query), expected)
                self.assertIs(self._evaluate("mayor"), expected)

    def test_deadline_boundary_and_envoy_tristate(self):
        """Unknown deadlines propagate None; exactly-deadline completion is late and timely history persists."""
        deadline = self.predicates["mayor"]["all"][0]
        cutoff = (deadline["by_day"] + 1) * 24_000
        self.world.game_time = cutoff - 1
        self.assertIs(self.evaluator.evaluate(deadline), None)
        self.assertIs(self._evaluate("envoy"), None)
        self.world.game_time = cutoff
        self.assertIs(self.evaluator.evaluate(deadline), False)
        self.assertIs(self._evaluate("envoy"), True)
        self.tracker.first_true["fair_open"] = cutoff
        self.assertIs(self.evaluator.evaluate(deadline), False)
        self.tracker.first_true["fair_open"] = cutoff - 1
        self.assertIs(self.evaluator.evaluate(deadline), True)
        self.assertIs(self._evaluate("envoy"), False)
        self.tracker.first_true.clear()
        self.world.containers[self.records] = [{"author": "investigator", "title": "Report",
                                                "pages": [REPORT_CASES[0][1]]}]
        self.assertIs(self._evaluate("envoy"), False)

    def test_fixed_reports_and_other_authors_do_not_fill_missing_reports(self):
        """Both report roles return None without their own report, then true or false rather than None."""
        for role in ("investigator", "security"):
            with self.subTest(role=role):
                self.world.containers[self.records] = []
                self.assertIs(self._evaluate(role), None)
                other = "security" if role == "investigator" else "investigator"
                self.world.containers[self.records] = [{"author": other, "title": "Report",
                                                        "pages": [REPORT_CASES[0][1]]}]
                self.assertIs(self._evaluate(role), None)
                for label, text, verdict in REPORT_CASES:
                    with self.subTest(report=label):
                        self.world.containers[self.records] = [{"author": role, "title": "Report", "pages": [text]}]
                        self.assertIs(self._evaluate(role), verdict.cause == "sabotage" and verdict.culprit == "envoy")

    def test_latest_own_report_decides_but_any_report_can_catch_envoy(self):
        """A later wrong report changes its author's verdict, but an earlier accusation still catches envoy."""
        self.world.containers[self.records] = [
            {"author": "investigator", "title": "Report one", "pages": [REPORT_CASES[0][1]]},
            {"author": "investigator", "title": "Report two", "pages": [REPORT_CASES[-1][1]]},
        ]
        self.assertIs(self._evaluate("investigator"), False)
        self.assertIs(self._evaluate("envoy"), False)

    def test_incident_window_region_actor_and_change_are_all_required(self):
        """Only a named bot removing a landmark block from day 1 onward fails security's incident query."""
        query = self.predicates["security"]["all"][0]
        position = self.predicates["builder"]["all"][0]["blocks_present"][0]["pos"]
        event = {"type": "block_change", "actor": "envoy", "gameTime": 24_000,
                 "pos": position, "to": "minecraft:air"}
        for changes, expected in (({}, False), ({"gameTime": 23_999}, True), ({"actor": "command"}, True),
                                  ({"pos": [100, -60, 100]}, True), ({"to": "minecraft:oak_planks"}, True)):
            with self.subTest(changes=changes):
                self._events([{**event, **changes}])
                self.assertIs(self.evaluator.evaluate(query), expected)


class JudgeAndEvidenceTest(unittest.TestCase):
    """Pin fixed judge texts and require real structured evidence, including strict W4 XFAIL/XPASS handling."""

    def setUp(self):
        """Load evidence coordinates and signed-document identities from the actual sabotage manifest."""
        self.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        self.evidence = self.manifest.data["evidence"]
        self.suspects = [role["name"] for role in self.manifest.roles]

    def test_keyword_fixed_texts_match_every_expected_verdict(self):
        """Canonical, paraphrased, wrong-culprit, accident and unclear texts have distinct expected readings."""
        judge = KeywordJudge(self.suspects)
        for label, text, expected in REPORT_CASES:
            with self.subTest(case=label):
                self.assertTrue(verdict_matches(judge.judge(text), expected))

    def test_llm_judge_plumbing_uses_same_fixed_corpus_without_network(self):
        """The real LLMJudge handles structured scripted replies and normalizes unclear/none exactly."""
        for label, text, expected in REPORT_CASES:
            with self.subTest(case=label):
                def decide(system: str, user: str) -> ToolCall:
                    """Check the actual judge prompt and answer from this fixed case's expected structured verdict."""
                    self.assertIn("only what the report itself", system)
                    self.assertEqual(user, f"Report:\n{text}")
                    return ToolCall("verdict", {"cause": expected.cause or "unclear",
                                               "culprit": expected.culprit or "none",
                                               "evidence": list(reversed(expected.evidence))})
                judge = LLMJudge(ScriptedClient(decide), self.suspects, self.evidence["expected_chain"])
                self.assertTrue(verdict_matches(judge.judge(text), expected))

    def test_verdict_comparison_rejects_wrong_cause_culprit_missing_and_extra_evidence(self):
        """Evidence order is irrelevant, but wrong conclusions and changed evidence sets never pass."""
        expected = REPORT_CASES[0][2]
        for actual in (Verdict("accident", "envoy", expected.evidence), Verdict("sabotage", "mayor", expected.evidence),
                       Verdict("sabotage", "envoy", expected.evidence[:-1]),
                       Verdict("sabotage", "envoy", (*expected.evidence, "invented"))):
            with self.subTest(actual=actual):
                self.assertFalse(verdict_matches(actual, expected))

    def test_pickaxe_requires_item_contents_uuid_tag_and_exact_position(self):
        """Legacy fixtures: a tool name or same-position villager is not the dropped evidence item."""
        tool = {"uuid": "00000000-0000-0000-0000-0000000002be", "item": "minecraft:iron_pickaxe",
                "pos": [-8.5, -60, -3.5], "tag": "village_evidence_pickaxe"}
        entity = {"uuid": tool["uuid"], "type": "minecraft:item", "pos": ",".join(map(str, tool["pos"])),
                  "nbt": f'{{Item:{{id:"{tool["item"]}"}},Tags:["{tool["tag"]}"]}}'}
        self.assertTrue(pickaxe_available(tool, [entity]))
        self.assertFalse(pickaxe_available(tool, []))
        for changes in ({"uuid": "another"}, {"type": "minecraft:villager"}, {"pos": "0,0,0"},
                        {"nbt": tool["tag"]}, {"nbt": tool["item"]}):
            with self.subTest(changes=changes):
                self.assertFalse(pickaxe_available(tool, [{**entity, **changes}]))

    def test_live_pickaxe_requires_its_logged_envoy_drop(self):
        """Live-staged fixtures: the surviving item counts only when joined to the envoy's logged drop of it."""
        tool = self.evidence["tool"]
        self.assertTrue(tool.get("thrower"), "the committed manifest is live-staged")
        entity = {"uuid": tool["uuid"], "type": "minecraft:item", "pos": ",".join(map(str, tool["pos"])),
                  "nbt": f'{{Item:{{id:"{tool["item"]}"}}}}'}
        drop = {"type": "drop", "bot": "envoy", "item": tool["item"], "count": 1, "itemEntity": tool["uuid"],
                "thrower": tool["thrower"]}
        self.assertTrue(pickaxe_available(tool, [entity], drop))
        self.assertFalse(pickaxe_available(tool, [entity]))
        for changes in ({"bot": "builder"}, {"thrower": "someone-else"}, {"count": 2}, {"itemEntity": "other"}):
            with self.subTest(changes=changes):
                self.assertFalse(pickaxe_available(tool, [entity], {**drop, **changes}))

    def test_ledger_requires_actual_book_container_slot_identity_and_purchase_text(self):
        """A similarly titled book elsewhere or an empty signed ledger is not observable purchase evidence."""
        document = next(doc for doc in self.manifest.data["documents"] if doc["id"] == "market_ledger")
        item = {"slot": document["slot"], "item": "minecraft:written_book", "title": document["title"],
                "author": document["author"], "pages": document["pages"]}
        self.assertTrue(ledger_available(document, self.evidence["ledger_entry"], [item]))
        for changes in ({"slot": 1}, {"item": "minecraft:book"}, {"title": "Ledger"}, {"author": "envoy"},
                        {"pages": ["nothing sold"]}):
            with self.subTest(changes=changes):
                self.assertFalse(ledger_available(document, self.evidence["ledger_entry"], [{**item, **changes}]))
        self.assertFalse(ledger_available(document, self.evidence["ledger_entry"], []))

    def test_witness_requires_sight_of_culprit_all_supports_and_firsthand_bridge_news(self):
        """Unrelated, hearsay-only or invented-origin testimony cannot manufacture a legacy witness XPASS."""
        supports = self.evidence["supports"]
        state = {"id": "adult_01", "news": [{"firsthand": True, "subject": "the bridge",
                                            "originEventId": "change:1"}]}
        record = {"npc": state["id"], "entities": [{"name": "envoy", "prov": "entity:envoy-uuid"}],
                  "changes": [{"pos": support["pos"], "actor": "envoy", "to": "minecraft:air",
                               "id": index, "prov": f"change:{index}"}
                              for index, support in enumerate(supports, 1)]}
        self.assertTrue(witness_observed(state, [record], supports, "envoy"))
        self.assertFalse(witness_observed(state, [], supports, "envoy"))
        for changes in ({"npc": "adult_02"}, {"entities": []}, {"changes": record["changes"][:-1]},
                        {"changes": [{**change, "actor": "mayor"} for change in record["changes"]]},
                        {"changes": [{**change, "prov": "change:invented"} for change in record["changes"]]}):
            with self.subTest(record_changes=changes):
                self.assertFalse(witness_observed(state, [{**record, **changes}], supports, "envoy"))
        for changes in ({"firsthand": False}, {"originEventId": "change:unseen"}, {"subject": "the port"}):
            with self.subTest(news_changes=changes):
                altered = {**state, "news": [{**state["news"][0], **changes}]}
                self.assertFalse(witness_observed(altered, [record], supports, "envoy"))

    def test_w4_is_expected_failure_not_skip_and_xpass_fails(self):
        """The exact W4 line is pinned and unexpected availability must require removing the expectation."""
        self.assertEqual(w4_result(False), (True, "XFAIL witness_observations (W4, older fixture)"))
        self.assertEqual(w4_result(True), (False, "XPASS witness_observations (W4, older fixture)"))
        self.assertEqual(w4_result(False, "staged_live"), (False, "FAIL witness_observations"))
        self.assertEqual(w4_result(True, "staged_live"), (True, "PASS witness_observations"))
        for unknown in (None, 0, 1, "skip"):
            with self.subTest(unknown=unknown), self.assertRaises(ValueError):
                w4_result(unknown)

    def test_swapping_two_real_lantern_results_fails_both_comparisons(self):
        """The same strict comparison used by the gate detects both intentionally reversed expectations."""
        cases = [(True, True), (False, False), (None, None)]
        self.assertTrue(truth_table_matches(cases))
        swapped = [(cases[0][0], cases[1][1]), (cases[1][0], cases[0][1])]
        self.assertFalse(truth_table_matches(swapped))
        self.assertTrue(all(not truth_table_matches([case]) for case in swapped))
        for invalid in ([], [(True, 1)], [(False, 0)], [(None, False)], [(None, True)]):
            with self.subTest(invalid=invalid):
                self.assertFalse(truth_table_matches(invalid))


class PredicateOptionsTest(unittest.TestCase):
    """Validate server isolation and optional local judges before any server/model lifecycle can begin."""

    def setUp(self):
        """Restore the caller's signal handler after CLI tests install the shared cleanup handler."""
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)

    def _args(self, **changes) -> argparse.Namespace:
        """Build real shared-parser defaults without invoking a server or creating a directory."""
        args = harness.experiment_parser("IT-09 test").parse_args([])
        args.port = 31000
        args.llm_judge_url = None
        args.llm_judge_model = "local"
        return argparse.Namespace(**{**vars(args), **changes})

    def test_default_game_port_is_assigned_base_plus_one_not_25565(self):
        """None is always resolved explicitly, while all other shared server options are preserved."""
        args = self._args(heap="2G", request_timeout=17)
        options = predicate_server_options(args)
        self.assertEqual(options["port"], 31000)
        self.assertEqual(options["game_port"], 31001)
        self.assertEqual(options["heap"], "2G")
        self.assertEqual(options["request_timeout"], 17)

    def test_explicit_game_ports_must_be_distinct_and_inside_eight_port_window(self):
        """The inclusive +7 boundary works; legacy defaults, collisions and out-of-range TCP ports fail early."""
        for game_port in (31001, 31007):
            with self.subTest(port=game_port):
                self.assertEqual(predicate_server_options(self._args(game_port=game_port))["game_port"], game_port)
        for game_port in (25565, 30999, 31000, 31008, 65536):
            with self.subTest(port=game_port), self.assertRaises(ValueError):
                predicate_server_options(self._args(game_port=game_port))
        for port in (0, -1, 65535, 65536):
            with self.subTest(base=port), self.assertRaises(ValueError):
                predicate_server_options(self._args(port=port))

    def test_only_loopback_judge_urls_are_allowed(self):
        """Local IPv4/IPv6 and localhost work; remote hosts, URL credentials and non-HTTP transports do not."""
        for url in ("http://localhost:8080/v1/", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"):
            with self.subTest(url=url):
                self.assertEqual(local_judge_url(url), url.rstrip("/"))
        for url in ("https://example.com/v1", "http://localhost.example.com/v1", "http://192.168.1.1/v1",
                    "file:///v1", "http://user:secret@localhost/v1", "http://localhost/v1?remote=1",
                    "http://localhost:invalid/v1", "http://localhost/v1#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_judge_url(url)

    def test_constructor_and_optional_local_judge_create_no_files_or_connections(self):
        """Both offline and opt-in judge clients can be configured without starting or contacting a model."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for url in (None, "http://localhost:1/v1"):
                with self.subTest(url=url):
                    experiment = PredicateExperiment(self._args(runs_dir=root, llm_judge_url=url))
                    self.assertEqual(set(experiment.judges), {"keyword", "LLM"} if url else {"keyword"})
                    self.assertEqual(list(root.iterdir()), [])
                    self.assertEqual(experiment.covered, set())

    def test_check_recording_retains_bad_tristate_rows(self):
        """A mismatch remains a failing artifact row instead of becoming a truthy integer or a skipped check."""
        experiment = PredicateExperiment(self._args())
        output = io.StringIO()
        with redirect_stdout(output):
            experiment._expect("no report", None, None)
            experiment._expect("swapped", True, False)
        self.assertTrue(experiment.checks[0]["passed"])
        self.assertFalse(experiment.checks[1]["passed"])
        self.assertIn("ok  no report", output.getvalue())
        self.assertIn("BAD swapped", output.getvalue())

    def test_tick_bursts_are_bounded_and_stop_exactly_at_deadline(self):
        """Advancing the real gate helper reaches a deadline without rounding past its strict boundary."""
        experiment = PredicateExperiment(self._args())
        experiment.client = FakeWorld({}, game_time=13)
        experiment._advance(2500)
        self.assertEqual(experiment.client.game_time, 2500)
        bursts = [args["n"] for command, args in experiment.client.requests if command == "step"]
        self.assertEqual(bursts, [1200, 1200, 87])
        experiment._advance(2500)
        self.assertEqual(experiment.client.game_time, 2500)
        self.assertEqual(sum(command == "step" for command, _ in experiment.client.requests), 3)

    def test_cli_rejects_unsafe_ports_and_remote_judges_before_any_clone(self):
        """Invalid options print BAD and FAIL IT-09 and return one without touching even the run directory."""
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in (["--port", "31000", "--game-port", "25565"],
                              ["--llm-judge-url", "https://example.com/v1"]):
                with self.subTest(arguments=arguments):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        code = main(["--runs-dir", temporary, *arguments])
                    self.assertEqual(code, 1)
                    self.assertIn("BAD ValueError", output.getvalue())
                    self.assertIn("FAIL IT-09:", output.getvalue())
                    self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_missing_fixture_reports_oserror_without_starting_a_server(self):
        """A nonexistent fixture fails during clone source validation, before any JVM lifecycle can begin."""
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--runs-dir", temporary, "--port", "31000"])
            self.assertEqual(code, 1)
            self.assertIn("BAD FileNotFoundError", output.getvalue())
            self.assertIn("FAIL IT-09:", output.getvalue())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_help_exits_before_any_experiment(self):
        """The shared parser and optional judge arguments remain discoverable without a fixture or server."""
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--llm-judge-url", output.getvalue())
        self.assertIn("--runs-dir", output.getvalue())


if __name__ == "__main__":
    unittest.main()
