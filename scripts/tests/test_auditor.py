"""Offline auditor assessments use synthetic questions, public run logs and ScriptedClient."""

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.llm import LLMResponse, ScriptedClient, ToolCall
from scripts.auditor import Auditor, VerdictSchema, canonical_json, load_questions, main
from scripts.auditor_scores import AuditorScores
from scripts.cost_ledger import LedgerWriter
from scripts.run_views import RunViews


class RecordingClient(ScriptedClient):
    """Record exact shared request bytes while the scripted policy supplies every answer."""

    def __init__(self, calls, model="scripted", reasoning_effort=None):
        """Keep a finite sequence of tool responses and deterministic token usage."""
        self._calls = iter(calls)
        super().__init__(lambda system, user: next(self._calls), model=model)
        self.reasoning_effort = reasoning_effort
        self.requests = []

    def complete(self, system, user, tools, seed=None, history=()):
        """Capture provider-independent request bytes without sorting away tool key differences."""
        self.requests.append(json.dumps({"system": system, "user": user, "tools": tools,
                                         "seed": seed, "history": history}, ensure_ascii=False).encode("utf-8"))
        response = super().complete(system, user, tools, seed, history)
        return LLMResponse(response.tool_call, model=self.model, input_tokens=20, output_tokens=5,
                           cache_read_tokens=3)


class AuditorTests(unittest.TestCase):
    """Pin request identity, evidence privacy, bounded retries and real-call-only costs."""

    def setUp(self):
        """Write an isolated questions file and run carrying explicit private sentinels."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run-a"
        self.run.mkdir()
        (self.run / "scenario.json").write_text(json.dumps({"roles": [
            {"name": "builder", "brief": "PRIVATE GOAL"}]}), encoding="utf-8")
        self._jsonl("world/simharness/events.jsonl", [
            {"gameTime": 10, "id": 1, "type": "utterance", "speaker": "builder", "text": "Public café",
             "listeners": []},
            {"gameTime": 30, "id": 2, "type": "utterance", "speaker": "builder", "text": "Later speech"}])
        self._jsonl("orchestrator/decisions.jsonl", [
            {"game_time": 20, "agent": "builder", "tool": "wait", "outcome": "started",
             "args": {"ticks": 1, "note": "PRIVATE NOTE"}, "user_prompt": "PRIVATE PROMPT"}])
        self.question_path = self.root / "questions.json"
        self.question_path.write_text(json.dumps([{
            "id": "synthetic", "system": "Use only this synthetic public evidence.",
            "question": "Did the synthetic event occur?",
            "verdict": {"type": "object", "required": ["label", "confidence", "evidence_ticks"],
                        "additionalProperties": False, "properties": {
                            "label": {"type": "string", "enum": ["yes", "no"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_ticks": {"type": "array", "items": {"type": "integer", "minimum": 0}}}}}]),
                                      encoding="utf-8")
        self.question = load_questions(self.question_path)["synthetic"]
        self.verdict = {"label": "yes", "confidence": 0.8, "evidence_ticks": [10]}
        self.ledger_path = self.root / "ledger.jsonl"

    def _jsonl(self, name, rows):
        """Write a small log beneath the temporary run directory."""
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _auditor(self, client, **options):
        """Build an assessor whose cache and ledger cannot touch real runs."""
        return Auditor(client, options.pop("question", self.question), options.pop("view_kind", "both"),
                       ledger=LedgerWriter("auditor", self.ledger_path),
                       cache_dir=options.pop("cache_dir", self.root / "cache"), **options)

    def _client(self, **options):
        """Return one scripted successful verdict with optional model/reasoning identity."""
        return RecordingClient([ToolCall("verdict", self.verdict)], **options)

    def test_requests_are_byte_identical_with_reordered_question_keys(self):
        """Recursive JSON key order cannot change request bytes or cache identity."""
        first_client, second_client = self._client(), self._client()
        reversed_question = json.loads(json.dumps(self.question), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
        first = self._auditor(first_client).assess(self.run)
        second_auditor = self._auditor(second_client, question=reversed_question, cache_dir=self.root / "other")
        second = second_auditor.assess(self.run)
        self.assertEqual(first, second)
        self.assertEqual(first_client.requests, second_client.requests)
        view = RunViews(self.run).render("both")
        expected = sha256(canonical_json({"model": "scripted", "reasoning_effort": None,
                                          "question": self.question, "view": view}).encode("utf-8")).hexdigest()
        self.assertEqual(first["request_digest"], expected)
        self.assertEqual(first["view_chars"], len(view))
        self.assertEqual((first["run"], first["view"], first["question"], first["model"]),
                         ("run-a", "both", "synthetic", "scripted"))
        self.assertEqual(first["verdict"], self.verdict)

    def test_only_public_view_and_file_question_are_sent(self):
        """Private notes, prompts and goals never reach the single required tool request."""
        client = self._client()
        self._auditor(client, until=20).assess(self.run)
        request = json.loads(client.requests[0])
        self.assertEqual(request["system"], self.question["system"])
        self.assertIn(self.question["question"], request["user"])
        self.assertIn("Public café", request["user"])
        self.assertNotIn("Later speech", request["user"])
        self.assertNotIn("PRIVATE", request["user"])
        self.assertNotIn(str(self.run), request["user"])
        self.assertEqual(len(request["tools"]), 1)
        self.assertEqual(request["tools"][0]["name"], "verdict")
        self.assertEqual(request["tools"][0]["parameters"], self.question["verdict"])

    def test_cache_prevents_a_second_call_and_ledger_row(self):
        """A new assessor instance reads the successful verdict without spending tokens."""
        client = self._client()
        first = self._auditor(client).assess(self.run)
        cached_client = RecordingClient([])
        self.assertEqual(self._auditor(cached_client).assess(self.run), first)
        self.assertEqual(cached_client.requests, [])
        rows = [json.loads(line) for line in self.ledger_path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual({key: rows[0][key] for key in ("purpose", "run", "input_tokens", "cached_tokens",
                                                      "output_tokens")},
                         {"purpose": "auditor", "run": "run-a", "input_tokens": 20,
                          "cached_tokens": 3, "output_tokens": 5})
        self.assertEqual(len(list((self.root / "cache").glob("*.json"))), 1)

    def test_identical_views_rebind_cached_run_and_view_metadata(self):
        """Content-addressed cache reuse cannot misattribute an identical run's answer."""
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "scenario.json").write_text("{}", encoding="utf-8")
        other = self.root / "other-empty"
        other.mkdir()
        (other / "scenario.json").write_text("{}", encoding="utf-8")
        first = self._auditor(self._client(), view_kind="speech").assess(empty)
        cached_client = RecordingClient([])
        second = self._auditor(cached_client, view_kind="actions").assess(other)
        self.assertEqual(first["request_digest"], second["request_digest"])
        self.assertEqual((second["run"], second["view"]), ("other-empty", "actions"))
        self.assertEqual(cached_client.requests, [])

    def test_changed_model_reasoning_question_view_or_cutoff_misses_cache(self):
        """Every request-changing identity component produces a separate assessment."""
        results = [self._auditor(self._client()).assess(self.run)]
        results.append(self._auditor(self._client(model="another")).assess(self.run))
        results.append(self._auditor(self._client(reasoning_effort="low")).assess(self.run))
        changed = {**self.question, "question": "Changed?"}
        results.append(self._auditor(self._client(), question=changed).assess(self.run))
        results.append(self._auditor(self._client(), view_kind="speech").assess(self.run))
        results.append(self._auditor(self._client(), until=10).assess(self.run))
        self.assertEqual(len({row["request_digest"] for row in results}), len(results))

    def test_plan_reasoning_attribute_and_question_snapshot_are_used(self):
        """Real plan clients keep effort in _reasoning; caller edits cannot change a prepared question."""
        client = self._client()
        del client.reasoning_effort
        client._reasoning = "low"
        assessor = self._auditor(client)
        self.question["system"] = "Changed after construction"
        self.question["verdict"]["properties"]["label"]["enum"] = ["not the original labels"]
        result = assessor.assess(self.run)
        expected_question = load_questions(self.question_path)["synthetic"]
        expected = sha256(canonical_json({"model": "scripted", "reasoning_effort": "low",
                                          "question": expected_question,
                                          "view": RunViews(self.run).render("both")}).encode("utf-8")).hexdigest()
        self.assertEqual(result["request_digest"], expected)
        self.assertEqual(json.loads(client.requests[0])["system"], expected_question["system"])

    def test_missing_wrong_named_or_invalid_call_is_retried_once(self):
        """A corrected retry uses identical request bytes and charges both responses."""
        for index, bad_call in enumerate((None, ToolCall("wrong", self.verdict), ToolCall("verdict", {}),
                                          ToolCall("verdict", "not an object"))):
            with self.subTest(bad_call=bad_call):
                client = RecordingClient([bad_call, ToolCall("verdict", self.verdict)])
                result = self._auditor(client, cache_dir=self.root / f"retry-{index}").assess(self.run)
                self.assertEqual(result["verdict"], self.verdict)
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(client.requests[0], client.requests[1])
        self.assertEqual(len(self.ledger_path.read_text().splitlines()), 8)

    def test_two_malformed_calls_fail_without_caching(self):
        """Exhaustion reports the validation error; another assessment must call again."""
        client = RecordingClient([None, ToolCall("verdict", {**self.verdict, "confidence": 2})])
        with self.assertRaisesRegex(ValueError, "after 2 attempts.*confidence"):
            self._auditor(client).assess(self.run)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])
        self._auditor(self._client()).assess(self.run)
        self.assertEqual(len(self.ledger_path.read_text().splitlines()), 3)

    def test_schema_enforces_fields_types_bounds_and_additional_properties(self):
        """Malformed values, booleans masquerading as numbers, and nonfinite JSON fail."""
        invalid = [{}, {**self.verdict, "label": "maybe"}, {**self.verdict, "confidence": True},
                   {**self.verdict, "confidence": -0.1}, {**self.verdict, "confidence": float("nan")},
                   {**self.verdict, "confidence": float("inf")}, {**self.verdict, "evidence_ticks": [1.5]},
                   {**self.verdict, "evidence_ticks": [False]}, {**self.verdict, "evidence_ticks": [-1]},
                   {**self.verdict, "extra": "forbidden"}]
        for index, verdict in enumerate(invalid):
            with self.subTest(verdict=verdict):
                client = RecordingClient([ToolCall("verdict", verdict)] * 2)
                with self.assertRaisesRegex(ValueError, "malformed auditor verdict"):
                    self._auditor(client, cache_dir=self.root / str(index)).assess(self.run)
                self.assertEqual(len(client.requests), 2)

    def test_overlong_views_refuse_before_cache_calls_or_ledger(self):
        """The size limit is checked even on cache hits, and equality is allowed."""
        length = len(RunViews(self.run).render("both"))
        self._auditor(self._client(), max_chars=length).assess(self.run)
        ledger_before = self.ledger_path.read_bytes()
        client = RecordingClient([])
        with self.assertRaisesRegex(ValueError, "exceeds --max-chars.*no truncation"):
            self._auditor(client, max_chars=length - 1).assess(self.run)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.ledger_path.read_bytes(), ledger_before)

    def test_private_changes_do_not_invalidate_cache(self):
        """Changing a private goal alone leaves the rendered request identity unchanged."""
        first = self._auditor(self._client()).assess(self.run)
        (self.run / "scenario.json").write_text(json.dumps({"roles": [
            {"name": "builder", "brief": "DIFFERENT PRIVATE GOAL"}]}), encoding="utf-8")
        self.assertEqual(self._auditor(RecordingClient([])).assess(self.run), first)

    def test_corrupted_cached_verdict_is_not_trusted(self):
        """A cached answer must still satisfy the question's schema."""
        first = self._auditor(self._client()).assess(self.run)
        path = self.root / "cache" / f"{first['request_digest']}.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required property"):
            self._auditor(RecordingClient([])).assess(self.run)

    def test_question_loader_rejects_ambiguous_or_invalid_input(self):
        """Questions must come from a well-formed file with unique IDs and supported schemas."""
        for entries in ({}, [self.question, self.question], [{**self.question, "system": ""}],
                        [{**self.question, "verdict": {"type": "object", "$ref": "elsewhere"}}]):
            with self.subTest(entries=entries):
                self.question_path.write_text(json.dumps(entries), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_questions(self.question_path)

    def test_cli_rejects_missing_question_before_building_a_model(self):
        """An invalid question cannot trigger auth, model or server activity."""
        errors = io.StringIO()
        with redirect_stderr(errors), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--questions", str(self.question_path), "--question", "absent", "--model", "never-called",
                  "--out", str(self.root / "results.jsonl"), str(self.run)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unknown question id", errors.getvalue())

    def test_cli_appends_results_and_reuses_the_provider_factory(self):
        """A scripted client exercises CLI provider options, JSONL append and cached reassessment."""
        client = self._client(reasoning_effort="low")
        output = self.root / "results.jsonl"
        output.write_text('{"existing":true}\n', encoding="utf-8")

        def assessor(*args, **kwargs):
            """Keep actual CLI assessments inside the isolated cache and ledger."""
            return Auditor(*args, **kwargs, cache_dir=self.root / "cli-cache",
                           ledger=LedgerWriter("auditor", self.ledger_path))

        text = io.StringIO()
        with patch("scripts.auditor.model_client", return_value=client) as factory, \
                patch("scripts.auditor.Auditor", side_effect=assessor), redirect_stdout(text):
            code = main(["--questions", str(self.question_path), "--question", "synthetic", "--view", "speech",
                         "--provider", "chatgpt-plan", "--model", "scripted", "--reasoning-effort", "low",
                         "--until", "10", "--out", str(output), str(self.run), str(self.run)])
        self.assertEqual(code, 0)
        args = factory.call_args.args[0]
        self.assertEqual((args.provider, args.model, args.reasoning_effort), ("chatgpt-plan", "scripted", "low"))
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(rows[0], {"existing": True})
        self.assertEqual(rows[1], rows[2])
        self.assertEqual(rows[1]["view"], "speech")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len(self.ledger_path.read_text().splitlines()), 1)
        self.assertIn("Assessed 2 run(s)", text.getvalue())

    def test_assessments_feed_the_scorer_without_translation(self):
        """Structured verdicts and metadata form the exact scoring interface end to end."""
        other = self.root / "run-b"
        other.mkdir()
        (other / "scenario.json").write_text("{}", encoding="utf-8")
        negative = {"label": "no", "confidence": 0.2, "evidence_ticks": []}
        client = RecordingClient([ToolCall("verdict", self.verdict), ToolCall("verdict", negative)])
        auditor = self._auditor(client)
        rows = [auditor.assess(directory) for directory in (self.run, other)]
        group = AuditorScores(rows, {"run-a": "yes", "run-b": "no"}, "yes", bootstrap=20).report()["groups"][0]
        self.assertEqual(group["n"], 2)
        self.assertEqual(group["auroc"]["value"], 1)
        self.assertEqual(group["accuracy"]["value"], 1)
        self.assertEqual(group["fpr_at_tpr"]["value"], 0)
        self.assertEqual(len(client.requests), 2)


class VerdictSchemaTests(unittest.TestCase):
    """The verdict subset is enforced, and anything outside it is refused before any model call."""

    def test_types_enums_bounds_arrays_and_extra_fields(self):
        """Nested objects, enums, bounds, typed arrays and additionalProperties false keep JSON semantics."""
        schema = VerdictSchema({"type": "object", "required": ["label"], "additionalProperties": False, "properties": {
            "label": {"type": "string", "enum": ["yes", "no"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ticks": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "flag": {"type": "boolean"},
            "nested": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}}})
        schema.validate({"label": "yes", "confidence": 1, "ticks": [0, 3], "flag": False, "nested": {"n": 2}})
        for value in ({}, {"label": "maybe"}, {"label": "yes", "confidence": 1.5}, {"label": "yes", "confidence": True},
                      {"label": "yes", "ticks": [1.5]}, {"label": "yes", "ticks": [-1]}, {"label": "yes", "flag": 0},
                      {"label": "yes", "nested": {}}, {"label": "yes", "extra": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                schema.validate(value)

    def test_unsupported_or_malformed_schemas_fail_before_assessment(self):
        """Keywords outside the subset, and malformed ones, are refused even in optional properties."""
        for child in ({"type": "unknown"}, {"type": ["null", "string"]}, {"type": "string", "pattern": "^a"},
                      {"type": "number", "minimum": True}, {"type": "string", "enum": []},
                      {"type": "object", "properties": []}, {"type": "object", "required": [1]},
                      {"type": "number", "anyOf": [{"minimum": 0}]}):
            with self.subTest(child=child), self.assertRaises(ValueError):
                VerdictSchema({"type": "object", "properties": {"optional": child}})
