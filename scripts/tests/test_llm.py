"""Offline tests for function-call parsing and provider conversation layouts."""

import unittest

from orchestrator.llm import AnthropicClient, OpenAICompatibleClient, ToolCall, Turn, arguments_json


class FunctionCallParsingTest(unittest.TestCase):
    """Chat-completions parsing returns the first call and preserves the original message."""

    def test_first_function_call_is_selected(self):
        """Only the first function call is selected from a reply with several calls."""
        message = {"content": "walking", "tool_calls": [
            {"function": {"name": "move_to", "arguments": '{"x": 2}'}},
            {"function": {"name": "wait", "arguments": '{"ticks": 40}'}}]}
        self.assertEqual(OpenAICompatibleClient._parse({"choices": [{"message": message}]}),
                         (ToolCall("move_to", {"x": 2}), message))

    def test_no_function_call_returns_none(self):
        """Absent, null and empty tool-call lists all produce no action."""
        for message in ({"content": "thinking"}, {"tool_calls": None}, {"tool_calls": []}):
            with self.subTest(message=message):
                self.assertEqual(OpenAICompatibleClient._parse({"choices": [{"message": message}]}), (None, message))

    def test_malformed_first_arguments_return_none(self):
        """Invalid JSON in the first call is not replaced by a later valid call."""
        message = {"tool_calls": [{"function": {"name": "wait", "arguments": "{"}},
                                  {"function": {"name": "wait", "arguments": '{"ticks": 40}'}}]}
        self.assertEqual(OpenAICompatibleClient._parse({"choices": [{"message": message}]}), (None, message))

    def test_dictionary_arguments_are_accepted(self):
        """Arguments already decoded as a dictionary are accepted unchanged."""
        message = {"tool_calls": [{"function": {"name": "wait", "arguments": {"ticks": 20}}}]}
        self.assertEqual(OpenAICompatibleClient._parse({"choices": [{"message": message}]}),
                         (ToolCall("wait", {"ticks": 20}), message))

    def test_missing_arguments_default_to_an_empty_dictionary(self):
        """A function call with no argument field receives an empty dictionary."""
        message = {"tool_calls": [{"function": {"name": "keep_going"}}]}
        self.assertEqual(OpenAICompatibleClient._parse({"choices": [{"message": message}]}),
                         (ToolCall("keep_going", {}), message))


class ProviderHistoryTest(unittest.TestCase):
    """Provider histories retain no-action results and deterministic argument ordering."""

    def test_anthropic_no_action_result_precedes_the_next_prompt(self):
        """No-action results and tool results share the next user turn in Anthropic history."""
        history = (Turn("first", None, "no action chosen"),
                   Turn("second", ToolCall("wait", {"ticks": 20}), "started"))
        self.assertEqual(AnthropicClient._messages(history, "third"), [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "(no action)"}]},
            {"role": "user", "content": [{"type": "text", "text": "no action chosen"},
                                          {"type": "text", "text": "second"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "wait",
                                               "input": {"ticks": 20}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "started"},
                                          {"type": "text", "text": "third"}]}])

    def test_argument_json_sorts_nested_keys_and_preserves_unicode(self):
        """Argument JSON sorts keys recursively without escaping non-ASCII text."""
        self.assertEqual(arguments_json(ToolCall("write_book", {"z": {"y": 2, "a": 1}, "a": "café"})),
                         '{"a": "café", "z": {"a": 1, "y": 2}}')
