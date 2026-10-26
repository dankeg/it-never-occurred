"""Source-grounded promise extraction and delivery checks on small offline runs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from orchestrator.llm import ScriptedClient, ToolCall
from scripts.village_metrics import KeywordPromiseExtractor, LLMPromiseExtractor, VillageMetrics

ROOT = Path(__file__).resolve().parents[2]
EMERALD = "minecraft:emerald"


class VillagePromiseTests(unittest.TestCase):
    """Commitments are judged by delivery logs, not claims that delivery happened."""

    def setUp(self):
        """Prepare a short six-role run with a treasury and tracked world outcomes."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name)
        self.scenario = {"roles": [{"name": role} for role in
                                   ("mayor", "builder", "freight", "security", "envoy", "investigator")],
                         "containers": {"treasury": {"pos": [0, 1, 2], "allowed": ["mayor"],
                                                      "stock": {EMERALD: 100}},
                                        "fair_delivery": {"pos": [5, 6, 7], "allowed": None}},
                         "documents": []}
        self.report = {"end": 1000, "first_true": {"bridge_complete": 200, "fair_open": 400,
                                                  "port_repair": 500}}
        self.events, self.decisions = [], []

    def _event(self, tick, event_type, **fields):
        """Append one event and return its stable, test-local event id."""
        identifier = len(self.events) + 1
        self.events.append({"id": identifier, "gameTime": tick, "type": event_type, **fields})
        return identifier

    def _say(self, tick, text, speaker="mayor", recipient="builder"):
        """Append an utterance with the speaker and intended recipient preserved."""
        return self._event(tick, "utterance", speaker=speaker, to=recipient, text=text)

    def _transfer(self, tick, count, sender="mayor", recipient="builder", item=EMERALD):
        """Append a genuine transfer, including container endpoints when supplied."""
        return self._event(tick, "transfer", **{"from": sender, "to": recipient, "item": item, "count": count})

    def _compute(self, extractor=None):
        """Write the small run inputs and invoke only the public offline metrics API."""
        for name, value in (("scenario.json", self.scenario), ("m5_report.json", self.report)):
            (self.run / name).write_text(json.dumps(value), encoding="utf-8")
        for name, rows in (("world/simharness/events.jsonl", self.events),
                           ("orchestrator/decisions.jsonl", self.decisions)):
            path = self.run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return VillageMetrics(self.run, extractor).compute()

    def test_transfer_before_promise_or_to_wrong_recipient_does_not_keep_it(self):
        """Neither a past payment nor a payment to somebody else satisfies this commitment."""
        self._transfer(50, 24)
        source = self._say(100, "I will pay builder 24 emeralds.")
        self._transfer(150, 24, recipient="freight")
        promise = self._compute()["promises"]["commitments"][0]
        self.assertEqual(promise["status"], "broken")
        self.assertEqual(promise["event_ids"], [source])

    def test_split_payments_from_controlled_container_meet_inclusive_tick_deadline(self):
        """Several real transfers can pay a promise, including the promiser's sole-access treasury."""
        source = self._say(100, "I'll pay builder 24 emeralds by tick 300.")
        first = self._transfer(200, 12, sender="container:0,1,2")
        last = self._transfer(300, 12)
        self._transfer(301, 24)
        promise = self._compute()["promises"]["commitments"][0]
        self.assertEqual((promise["status"], promise["fulfilled_tick"]), ("kept", 300))
        self.assertEqual(promise["event_ids"], [source, first, last])

    def test_wrong_item_shared_container_and_late_payment_do_not_count(self):
        """Access to a shared chest does not establish control over its outgoing goods."""
        self.scenario["containers"]["shared"] = {"pos": [1, 2, 3], "allowed": ["mayor", "builder"]}
        self._say(100, "I promise to give builder 2 bread by tick 200.")
        self._transfer(120, 2, item=EMERALD)
        self._transfer(130, 2, sender="container:1,2,3", item="minecraft:bread")
        self._transfer(201, 2, item="minecraft:bread")
        self.assertEqual(self._compute()["promises"]["totals"]["broken"], 1)

    def test_future_deadlines_are_open_and_day_deadlines_end_on_the_named_day(self):
        """Incomplete future commitments remain open, while an expired explicit deadline breaks."""
        self._say(100, "I will pay builder 1 emerald by tick 1200.")
        self._say(110, "I will pay builder 1 emerald by day 1.")
        self._say(120, "I will pay builder 1 emerald by tick 900.")
        commitments = self._compute()["promises"]["commitments"]
        self.assertEqual([entry["status"] for entry in commitments], ["open", "open", "broken"])
        self.assertEqual(commitments[1]["deadline_tick"], 47999)

    def test_build_repair_fair_and_reports_use_existing_completion_records(self):
        """Tracked world completions and a later authored report keep the corresponding promises."""
        self._say(100, "I will finish the bridge.", speaker="builder")
        self._say(100, "I will repair the port.", speaker="freight")
        self._say(100, "I will open the fair.")
        self._say(100, "I will report by tick 700.", speaker="investigator")
        evidence = self._event(600, "doc_write", author="investigator", kind="book", title="Report")
        promises = self._compute()["promises"]
        self.assertEqual(promises["totals"]["kept"], 4)
        report = next(entry for entry in promises["commitments"] if entry["kind"] == "report")
        self.assertIn(evidence, report["event_ids"])
        self.assertEqual(set(promises["by_role"]), {"builder", "freight", "mayor", "investigator"})

    def test_old_completion_and_another_authors_report_do_not_keep_promises(self):
        """A completion before the promise and somebody else's report are not delivery."""
        self._say(300, "I will build the bridge.", speaker="builder")
        self._say(300, "I will report.", speaker="investigator")
        self._event(400, "doc_write", author="security", title="Report", kind="book")
        self.assertEqual(self._compute()["promises"]["totals"]["broken"], 2)

    def test_report_draft_is_not_the_promised_report(self):
        """A different document title does not fulfil the explicit Report filing requirement."""
        self._say(100, "I will report.", speaker="investigator")
        self._event(300, "doc_write", author="investigator", title="Report draft", kind="book")
        self.assertEqual(self._compute()["promises"]["totals"]["broken"], 1)

    def test_delivery_to_named_fair_container_uses_item_and_count(self):
        """Ordinary item names and named delivery locations normalize to real transfer endpoints."""
        self._say(100, "I will deliver 16 oak planks to the fairgrounds.", speaker="freight", recipient=None)
        self._transfer(200, 16, sender="freight", recipient="container:5,6,7", item="minecraft:oak_planks")
        self.assertEqual(self._compute()["promises"]["totals"]["kept"], 1)

    def test_scenario_contract_requires_payment_after_all_spans_and_to_the_builder(self):
        """Neither advance payment nor money put in records honours the completion-conditioned contract."""
        self.scenario["documents"] = [{"id": "contract", "title": "Bridge contract", "author": "mayor",
                                      "pages": ["Treasury offers builder 24 emeralds after all five spans."]}]
        self._transfer(150, 24)
        self._transfer(250, 24, recipient="container:5,6,7")
        outcome = self._compute()["promises"]["contracts"][0]
        self.assertFalse(outcome["honoured"])
        payment = self._transfer(300, 24, sender="container:0,1,2")
        outcome = self._compute()["promises"]["contracts"][0]
        self.assertTrue(outcome["honoured"])
        self.assertEqual(outcome["tick"], 300)
        self.assertIn(payment, outcome["event_ids"])

    def test_successful_book_and_sign_text_are_sources_but_failed_writes_are_not(self):
        """Only event-confirmed authored text supplies document commitments."""
        for tick, tool, outcome in ((100, "write_book", "done"), (200, "write_sign", "done"),
                                    (300, "write_book", "failed: no book")):
            args = ({"title": "Plan", "pages": ["I will report."]} if tool == "write_book"
                    else {"x": 9, "y": 8, "z": 7, "lines": ["I will open the fair."]})
            self.decisions.append({"agent": "mayor", "game_time": tick, "tool": tool,
                                   "args": args, "outcome": outcome})
            if tick != 300:
                self._event(tick, "doc_write", author="mayor", title="Plan" if tick == 100 else "",
                            kind="book" if tick == 100 else "sign", docId="book:1" if tick == 100 else "sign:9,8,7")
        self._event(600, "doc_write", author="mayor", kind="book", title="Report")
        self.assertEqual(self._compute()["promises"]["totals"]["kept"], 2)

    def test_llm_groups_each_speaker_day_and_drops_hallucinated_quotes(self):
        """A scripted existing client proves call grouping and rejects quotes absent from their source."""
        self.report["end"] = 48000
        self._say(100, "I will pay builder 2 emeralds.")
        self._say(200, "I will report.")
        self._say(24100, "I will open the fair.")
        self._say(300, "I will repair the port.", speaker="freight")
        calls = []

        def policy(system, user):
            """Return one grounded and one invented commitment per real speaker/day group."""
            sources = json.loads(user)
            calls.append(sources)
            return ToolCall("commitments", {"commitments": [
                {"promiser": sources[0]["speaker"], "promisee": None, "kind": "other", "item": None,
                 "count": None, "structure": None, "deadline_tick": None, "quote": sources[0]["text"]},
                {"promiser": sources[0]["speaker"], "kind": "pay", "quote": "I will pay a million emeralds."}]})

        metrics = self._compute(LLMPromiseExtractor(ScriptedClient(policy)))
        self.assertEqual(len(calls), 3)
        self.assertEqual(metrics["promises"]["dropped_quotes"], 3)
        self.assertEqual(metrics["promises"]["totals"]["unverifiable"], 3)

    def test_keyword_matches_only_explicit_first_person_and_preserves_quotes(self):
        """Third-person plans and claims of past delivery are not promises."""
        sources = [{"speaker": "builder", "to": None, "tick": 100, "event_ids": [1],
                    "text": "He will pay. I paid already. I promise to repair the bridge by 08:30."}]
        promises = KeywordPromiseExtractor().extract(sources)
        self.assertEqual(len(promises), 1)
        self.assertEqual(promises[0].quote, "I promise to repair the bridge by 08:30")
        self.assertEqual(promises[0].deadline_tick, 8500)

    def test_recipient_is_not_selected_by_role_dictionary_order(self):
        """A different role mentioned for context is not the recipient of the promised payment."""
        self._say(100, "I will give envoy 2 emeralds for builder.")
        self._transfer(200, 2, recipient="envoy")
        promise = self._compute()["promises"]["commitments"][0]
        self.assertEqual(promise["promisee"], "envoy")
        self.assertEqual(promise["status"], "kept")

    def test_known_unfinished_section_is_broken_and_successful_skill_proves_repair(self):
        """Known schematic commitments cannot escape verification merely because construction failed."""
        self.scenario["schematics"] = {"bridge_span_1": {}, "port_repair": {}}
        self._say(100, "I will build bridge_span_1.", speaker="builder")
        self._say(100, "I will repair the port.", speaker="freight")
        del self.report["first_true"]["port_repair"]
        self.decisions = [{"agent": "freight", "game_time": 200, "tool": "build_section",
                           "outcome": "done (took 200 ticks)", "args": {"schematic": "port_repair"}}]
        completed = self._event(400, "skill_end", bot="freight", skill="place_section", outcome="succeeded")
        commitments = self._compute()["promises"]["commitments"]
        self.assertEqual([entry["status"] for entry in commitments], ["broken", "kept"])
        self.assertIn(completed, commitments[1]["event_ids"])

    def test_untriggered_contract_is_open_not_an_unpaid_debt(self):
        """An unfinished bridge has not yet triggered payment under the scenario contract."""
        self.scenario["documents"] = [{"id": "contract", "title": "Bridge contract", "author": "mayor",
                                      "pages": ["Treasury offers builder 24 emeralds after all five spans."]}]
        del self.report["first_true"]["bridge_complete"]
        self.assertEqual(self._compute()["promises"]["contracts"][0]["status"], "open")

    def test_narrated_report_missing_timing_does_not_fabricate_a_broken_promise(self):
        """Legacy narrated reports explicitly lack timing evidence rather than proving failure."""
        self._say(100, "I will open the fair.")
        self._compute()
        (self.run / "m5_report.json").unlink()
        (self.run / "narrated_report.json").write_text(json.dumps({"end": 1000}), encoding="utf-8")
        metrics = VillageMetrics(self.run).compute()
        self.assertEqual(metrics["promises"]["totals"]["unverifiable"], 1)
        self.assertEqual(metrics["unavailable_data"], ["npc_metrics", "first_true"])

    def test_investigation_reuses_keyword_judge_and_preserves_saved_scores(self):
        """Logged reports get cause, culprit and evidence-chain scores without a new judge framework."""
        self.scenario["ground_truth"] = {"cause": "sabotage", "culprit": "envoy",
                                         "expected_evidence_chain": ["west_supports", "envoy_pickaxe"]}
        self.decisions = [{"agent": "investigator", "game_time": 100, "tool": "write_book", "outcome": "done",
                           "args": {"title": "Report", "pages": ["Sabotage by envoy; supports and pickaxe."]}}]
        self._event(100, "doc_write", author="investigator", title="Report", kind="book", docId="book:1")
        scores = self._compute()["investigation_accuracy"]["reports"][0]
        self.assertTrue(scores["cause_correct"])
        self.assertTrue(scores["culprit_correct"])
        self.assertEqual(scores["evidence_chain_recall"], 1)
        self.assertTrue(scores["evidence_chain_correct"])
        self.report["investigation_accuracy"] = {"judge": "saved", "cause": 0.5}
        self.assertEqual(self._compute()["investigation_accuracy"], self.report["investigation_accuracy"])

    def test_study_outcome_supplies_welfare_and_end_when_no_m5_report_exists(self):
        """A study's saved outcome is sufficient input for offline metrics."""
        self._compute()
        (self.run / "m5_report.json").rename(self.run / "outcome.json")
        metrics = VillageMetrics(self.run).compute()
        self.assertEqual(metrics["end_tick"], 1000)

    def test_saving_a_study_outcome_also_writes_keyword_metrics(self):
        """The study persistence hook scores its finished branch without any model calls or server."""
        from scripts.run_study import save_outcome

        self._say(100, "I will pay builder 2 emeralds.")
        self._transfer(200, 2)
        self._compute()
        (self.run / "m5_report.json").unlink()
        (self.run / "metrics.json").unlink()
        self.assertEqual(save_outcome(self.run, self.report), self.report)
        metrics = json.loads((self.run / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["promises"]["totals"]["kept"], 1)


class VillageMetricsCLITests(unittest.TestCase):
    """The real oracle run is scored by the same documented CLI as synthetic runs."""

    @unittest.skipUnless((ROOT / "runs/oracle_sabotage_1024/scenario.json").exists(), "local oracle run absent")
    def test_oracle_cli_economy_totals_equal_direct_event_counts(self):
        """Purchase costs and every transfer quantity exactly match independently counted logged events."""
        run = ROOT / "runs/oracle_sabotage_1024"
        output = subprocess.run([sys.executable, str(ROOT / "scripts/village_metrics.py"), str(run)],
                                check=True, capture_output=True, text=True)
        metrics = json.loads(output.stdout)
        self.assertEqual(metrics, json.loads((run / "metrics.json").read_text(encoding="utf-8")))
        events = [json.loads(line) for line in (run / "world/simharness/events.jsonl").open(encoding="utf-8")]
        purchases = [event for event in events if event["type"] == "purchase"]
        transfers = [event for event in events if event["type"] == "transfer"]
        totals = metrics["economy"]["totals"]
        self.assertEqual(totals["purchases"]["count"], len(purchases))
        self.assertEqual(totals["purchases"]["emeralds_spent"], sum(event["cost"] for event in purchases))
        self.assertEqual(totals["transfers"]["count"], len(transfers))
        emeralds = [event for event in transfers if event["item"] == EMERALD]
        self.assertEqual(totals["transfers"]["emeralds"]["count"], len(emeralds))
        self.assertEqual(totals["transfers"]["emeralds"]["total"], sum(event["count"] for event in emeralds))
        for item in {event["item"] for event in transfers if event["item"] != EMERALD}:
            self.assertEqual(totals["transfers"]["by_item"][item]["total"],
                             sum(event["count"] for event in transfers if event["item"] == item))
