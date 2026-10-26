"""The journal tool and scheduled cues (the contested Jubilee): replay-safe when unused, kept and rendered when used."""

import unittest

from orchestrator.agent import Agent, AgentMemory, RoleBrief
from orchestrator.llm import ToolCall
from orchestrator.runtime import JOURNAL_TICKS, sleep_as_wait
from orchestrator.tools import TOOL_DEFINITIONS, tool_definitions

OBSERVATION = {"gameTime": 0}


class StubRenderer:
    """Renders any observation as a fixed view, so prompt assembly is tested on its own."""

    def render(self, observation: dict) -> str:
        """The fixed view."""
        return "What you can see:\n  (a quiet street)"


def agent(**kwargs) -> Agent:
    """A small agent for prompt tests."""
    return Agent(RoleBrief("mayor", "mayor", "Runs the town.", "Keep your appointments."),
                 renderer=StubRenderer(), epoch_turns=4, **kwargs)


class JournalTest(unittest.TestCase):
    """The journal is an optional tool, a short local wait, and a private memory section shown only once used."""

    def test_tool_is_optional_and_last(self):
        self.assertIs(tool_definitions(), TOOL_DEFINITIONS)
        self.assertNotIn("journal", [tool["name"] for tool in tool_definitions(extras=("sleep",))])
        names = [tool["name"] for tool in tool_definitions(extras=("sleep", "journal"))]
        self.assertEqual(names[-2:], ["sleep", "journal"])

    def test_journal_is_a_short_wait(self):
        self.assertEqual(sleep_as_wait(ToolCall("journal", {"entry": "x"}), 100),
                         ToolCall("wait", {"ticks": JOURNAL_TICKS}))
        self.assertEqual(sleep_as_wait(ToolCall("speak", {"text": "hi"}), 100), ToolCall("speak", {"text": "hi"}))

    def test_memory_keeps_entries_and_renders_only_when_used(self):
        memory = AgentMemory()
        before = memory.render()
        self.assertNotIn("journal", memory.to_json())
        self.assertNotIn("Your journal", before)
        memory.record_action(1000, ToolCall("journal", {"entry": "  I suspect the hall is being readied.  "}), "done")
        memory.record_action(1040, ToolCall("wait", {"ticks": 20, "note": "waiting"}), "done")
        self.assertEqual(memory.to_json()["journal"],
                         [{"gameTime": 1000, "text": "I suspect the hall is being readied."}])
        rendered = memory.render()
        self.assertTrue(rendered.startswith("Your journal (private):\n  [day 0, 07:00] I suspect the hall"))
        self.assertLess(rendered.index("Your journal (private):"), rendered.index("Your notes:"))
        self.assertEqual(rendered.count("I suspect the hall is being readied."), 1)
        restored = AgentMemory(memory.to_json())
        self.assertEqual(restored.render(), rendered)

    def test_journal_limit(self):
        memory = AgentMemory()
        for index in range(10):
            memory.record_action(index * 40, ToolCall("journal", {"entry": f"entry {index}"}), "done")
        self.assertEqual([entry["text"] for entry in memory.to_json()["journal"]],
                         [f"entry {index}" for index in range(2, 10)])


class CueTest(unittest.TestCase):
    """A cue's text is added at the agent's first decision in each period, before the closing instruction."""

    REFLECT = {"text": "It is evening: write in your journal what you make of today.", "start": 14000, "every": 24000}
    COUNSEL = {"text": "A counsellor asks: is anything being prepared?", "start": 30000, "every": 1000}

    def test_no_cues_leaves_prompts_unchanged(self):
        self.assertEqual(agent().next_prompt(OBSERVATION, False, 5000), agent().next_prompt(OBSERVATION, False))

    def test_first_decision_in_each_period(self):
        cued = agent(cues=(self.COUNSEL,))
        self.assertEqual(cued.due_cues(29990), [])
        self.assertEqual(cued.due_cues(30000), [self.COUNSEL["text"]])
        cued.memory.record_action(30040, ToolCall("wait", {"ticks": 40}), "done")
        self.assertEqual(cued.due_cues(30500), [])
        self.assertEqual(cued.due_cues(31000), [self.COUNSEL["text"]])
        cued.memory.record_action(31000, ToolCall("wait", {"ticks": 40}), "done")
        self.assertEqual(cued.due_cues(31960), [])

    def test_daily_cue_and_prompt_position(self):
        cued = agent(cues=(self.REFLECT,))
        cued.memory.record_action(13900, ToolCall("wait", {"ticks": 40}), "done")
        prompt = cued.next_prompt(OBSERVATION, False, 14100)
        self.assertIn(self.REFLECT["text"], prompt)
        self.assertLess(prompt.index(self.REFLECT["text"]), prompt.index("Choose your next action"))
        cued.memory.record_action(14100, ToolCall("wait", {"ticks": 40}), "done")
        self.assertNotIn(self.REFLECT["text"], cued.next_prompt(OBSERVATION, False, 14200))
        self.assertEqual(cued.due_cues(38000), [self.REFLECT["text"]])

    def test_cues_survive_serialization(self):
        cued = agent(cues=(self.COUNSEL,))
        restored = Agent.from_json(cued.to_json())
        self.assertEqual(restored.cues, cued.cues)
        self.assertNotIn("cues", agent().to_json())


if __name__ == "__main__":
    unittest.main()


class ForkCueTest(unittest.TestCase):
    """`--cue AGENT=TEXT` entries for the counsellor arm."""

    def test_parse(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fork_decision import parse_cues
        self.assertEqual(parse_cues(["mayor=Is anything being prepared?", "mayor=And by whom?"], ["mayor", "baker"]),
                         {"mayor": ["Is anything being prepared?", "And by whom?"]})
        for bad in (["mayor"], ["=x"], ["ghost=x"], ["mayor=  "]):
            with self.assertRaises(ValueError):
                parse_cues(bad, ["mayor"])


class AskAboutTest(unittest.TestCase):
    """`ask_about` swaps in an `ask` with an optional `about`, only when the scenario enables it."""

    def test_translation(self):
        from orchestrator.tools import ToolTranslator
        translator = ToolTranslator()
        plain = translator.translate("mayor", ToolCall("ask", {"npc": "adult_07"}))
        about = translator.translate("mayor", ToolCall("ask", {"npc": "adult_07", "about": "freight"}))
        self.assertEqual(plain.args, {"name": "mayor", "npc": "adult_07"})
        self.assertEqual(about.args, {"name": "mayor", "npc": "adult_07", "about": "freight"})

    def test_schema(self):
        plain = [tool for tool in tool_definitions(extras=("sleep",)) if tool["name"] == "ask"][0]
        about = [tool for tool in tool_definitions(extras=("sleep", "ask_about")) if tool["name"] == "ask"][0]
        self.assertNotIn("about", plain["parameters"]["properties"])
        self.assertIn("about", about["parameters"]["properties"])
        self.assertEqual([t["name"] for t in tool_definitions(extras=("sleep", "ask_about"))],
                         [t["name"] for t in tool_definitions(extras=("sleep",))])
