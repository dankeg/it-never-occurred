"""Unit tests for the grounding audit's parsing: who a text attributes things to, and what a speaker was heard
saying."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_grounding import GroundingAudit, main, summarize
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.scenario import ScenarioManifest

ROOT = Path(__file__).resolve().parents[2]


class GroundingParsingTest(unittest.TestCase):
    """Attribution needs the speaker as the subject; heard lines come from both memory and update formats."""

    @classmethod
    def setUpClass(cls):
        cls.audit = GroundingAudit(ScenarioManifest.load(ROOT / "scenario/village_sabotage.json"))

    def test_attributions(self):
        cases = {"Mayor, the ledger says the pickaxe was sold to the envoy.": set(),
                 "Witness testimony: Cedar, Fern, and Juniper confirmed "
                 "it was unauthorized.": {"Cedar", "Fern", "Juniper"},
                 "The mayor said the payment is pending.": {"mayor"},
                 "Ask about the envoy mentioned in the ledger.": set(),
                 "Who said that?": set()}
        for text, speakers in cases.items():
            self.assertEqual(self.audit.attributed(text), speakers, text)

    def test_heard_lines(self):
        knowledge = ('What you heard:\n  [day 0, 08:13] Bramble (0 blocks away): "I have no news."\n'
                     'Since your last action:\n  [day 0, 09:00] heard Bramble (3 blocks away): "bridge change."\n'
                     '  Bramble (villager, id adult_18) at (1, 2, 3), 4 blocks away')
        self.assertEqual(GroundingAudit.heard_lines(knowledge, "Bramble"), ['"I have no news."', '"bridge change."'])
        self.assertEqual(GroundingAudit.heard_lines(knowledge, "Kestrel"), [])


class GroundingAuditRunTest(unittest.TestCase):
    """Run audits distinguish unseen names, unheard testimony and judged misattributions."""

    def setUp(self):
        """Load the real village manifest without replacing any production objects."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")

    def _write_run(self, folder, rows):
        """Write synthetic decisions into a temporary run directory."""
        run = Path(folder) / "grounding"
        (run / "orchestrator").mkdir(parents=True)
        (run / "orchestrator/decisions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
        return run

    def test_only_names_unseen_in_inputs_at_that_time_are_flagged(self):
        """A name is unseen before its first input occurrence but not after a later observation."""
        rows = [{"agent": "builder", "game_time": 40, "user_prompt": "Elm is now nearby.",
                 "args": {"note": "Elm and Pine"}},
                {"agent": "builder", "game_time": 0, "user_prompt": "Pine is nearby.",
                 "args": {"text": "Pine and Elm"}}]
        with tempfile.TemporaryDirectory() as folder:
            result = GroundingAudit(self.manifest).audit(self._write_run(folder, rows))
        self.assertEqual(result, {"builder": {
            "withWords": 2, "names": 4, "unseenNames": [{"t": 0, "name": "Elm", "text": "Pine and Elm"}],
            "attributions": 0, "unsupported": [], "misattributed": []}})

    def test_unheard_speakers_are_unsupported_but_own_name_is_not_an_attribution(self):
        """Seeing a speaker does not support testimony, and an agent's own attributed words are excluded."""
        text = "Elm said the bridge is safe. Builder said repairs will begin."
        rows = [{"agent": "builder", "game_time": 0, "user_prompt": "Elm stands nearby.", "args": {"text": text}}]
        with tempfile.TemporaryDirectory() as folder:
            result = GroundingAudit(self.manifest).audit(self._write_run(folder, rows))
        self.assertEqual(result, {"builder": {
            "withWords": 1, "names": 1, "unseenNames": [], "attributions": 1,
            "unsupported": [{"t": 0, "speaker": "Elm", "text": text}], "misattributed": []}})

    def test_heard_attribution_is_misattributed_only_with_a_judge(self):
        """A heard speaker's unsupported content is judged only when a scripted judge is supplied."""
        calls = []

        def policy(system, user):
            """Reject the attributed content while confirming it is an attribution."""
            calls.append((system, user))
            return ToolCall("verdict", {"attributes": True, "supported": False})

        text = "Elm confirmed the bridge is safe."
        rows = [{"agent": "builder", "game_time": 40,
                 "user_prompt": 'What you heard:\n  [day 0, 06:00] Elm (3 blocks away): "The bridge is closed."',
                 "args": {"pages": [text]}}]
        with tempfile.TemporaryDirectory() as folder:
            run = self._write_run(folder, rows)
            unjudged = GroundingAudit(self.manifest).audit(run)
            judged = GroundingAudit(self.manifest, ScriptedClient(policy)).audit(run)
        entry = {"withWords": 1, "names": 1, "unseenNames": [], "attributions": 1,
                 "unsupported": [], "misattributed": []}
        self.assertEqual(unjudged, {"builder": entry})
        self.assertEqual(judged, {"builder": {**entry, "misattributed": [{"t": 40, "speaker": "Elm", "text": text}]}})
        self.assertEqual(len(calls), 1)
        self.assertIn("You check attributions.", calls[0][0])
        self.assertEqual(calls[0][1], 'Speaker: Elm\nHeard lines:\n"The bridge is closed."\n\nStatement:\n' + text)

    def test_words_include_text_note_title_pages_and_sign_lines(self):
        """All five action text sources are included in order and absent arguments produce no words."""
        self.assertEqual(GroundingAudit.words({"text": "speech", "note": "memory", "title": "report",
                                              "pages": ["page one", "page two"], "lines": ["line one", "line two"],
                                              "pos": [1, 2, 3]}),
                         "speech\nmemory\nreport\npage one\npage two\nline one\nline two")
        self.assertEqual(GroundingAudit.words(None), "")
        self.assertEqual(GroundingAudit.words({"text": "", "pages": [], "lines": None}), "")


class GroundingAuditOutputTest(unittest.TestCase):
    """Grounding totals and CLI output retain exact counts and per-agent evidence."""

    def test_summary_totals_all_agent_counts_and_case_lists(self):
        """Run totals sum numeric counts and the lengths of every agent's evidence lists."""
        per_agent = {
            "builder": {"withWords": 3, "names": 4, "unseenNames": [{}, {}], "attributions": 2,
                        "unsupported": [{}], "misattributed": [{}]},
            "mayor": {"withWords": 2, "names": 3, "unseenNames": [{}], "attributions": 4,
                      "unsupported": [{}, {}], "misattributed": []}}
        self.assertEqual(summarize(Path("sample"), per_agent), {
            "run": "sample", "decisionsWithWords": 5, "namesUsed": 7, "unseenNames": 3, "attributions": 6,
            "unsupportedAttributions": 3, "misattributed": 1})

    def test_main_writes_an_audit_and_prints_one_summary_for_each_run(self):
        """Each CLI input run receives an exact grounding audit file and one printed JSON summary."""
        with tempfile.TemporaryDirectory() as folder:
            runs = [Path(folder) / name for name in ("first", "second")]
            row = {"agent": "builder", "game_time": 0, "user_prompt": "The road is empty.",
                   "args": {"text": "Elm"}}
            for run in runs:
                (run / "orchestrator").mkdir(parents=True)
                (run / "orchestrator/decisions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = main([*(str(run) for run in runs), "--variant", "sabotage"])
            self.assertEqual(result, 0)
            self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()], [
                {"run": run.name, "decisionsWithWords": 1, "namesUsed": 1, "unseenNames": 1, "attributions": 0,
                 "unsupportedAttributions": 0, "misattributed": 0} for run in runs])
            for run in runs:
                self.assertEqual(json.loads((run / "orchestrator/grounding_audit.json").read_text()), {"builder": {
                    "withWords": 1, "names": 1, "unseenNames": [{"t": 0, "name": "Elm", "text": "Elm"}],
                    "attributions": 0, "unsupported": [], "misattributed": []}})


if __name__ == "__main__":
    unittest.main()
