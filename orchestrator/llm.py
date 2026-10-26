"""Model-agnostic LLM clients for agent decisions.

An agent decision is one model call that returns exactly one tool call: the action the agent takes. Three clients
share one small interface:

- `AnthropicClient`: the Anthropic Messages API (tool use), over plain HTTPS.
- `OpenAICompatibleClient`: any OpenAI-compatible chat-completions endpoint with function tools.
- `ScriptedClient`: a deterministic policy function, for $0 development and plumbing tests (E3, L1).

The clients stay thin: they translate the shared request into a provider request and back. Retries are bounded and
only for transient HTTP failures.

A request may carry `history`: the agent's earlier turns in its current epoch (see `orchestrator.agent`). Clients send
them as a conversation (user message, the tool call, its result), so consecutive prompts of one agent share an exact
prefix that a provider's or local server's prompt cache can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import os
import time
from typing import Callable, Protocol, Sequence
from urllib import error, request


@dataclass(frozen=True)
class ToolCall:
    """One action chosen by the model: a tool name and its JSON arguments."""

    name: str
    args: dict


@dataclass(frozen=True)
class Turn:
    """One earlier turn in an agent's conversation: the user message, the action chosen (if any), and its result."""

    user: str
    call: ToolCall | None
    result: str


@dataclass(frozen=True)
class LLMResponse:
    """The parsed outcome of one model call."""

    tool_call: ToolCall | None
    text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw: dict = field(default_factory=dict, repr=False)


class LLMClient(Protocol):
    """Interface shared by all clients: one decision per call."""

    model: str

    def complete(self, system: str, user: str, tools: list[dict], seed: int | None = None,
                 history: Sequence[Turn] = ()) -> LLMResponse:
        """Return the model's single tool call for this system prompt, user message and tool list.

        `seed` makes sampling reproducible where the backend supports it (local servers do; hosted APIs may not).
        `history` holds the earlier turns of the agent's conversation, oldest first; `user` follows them.
        """
        ...


class UsageLimitReached(RuntimeError):
    """The provider's usage allowance is exhausted (e.g. a ChatGPT plan's limit): the run must stop, not continue
    with failed decisions."""


RETRY_NUDGE = "\n\nRespond now with exactly one tool call (one action). Do not write a plan or explanation."
NO_ACTION = "(no action)"


def arguments_json(call: ToolCall) -> str:
    """A tool call's arguments as JSON with sorted keys, so a replayed turn renders exactly like the original."""
    return json.dumps(call.args, sort_keys=True, ensure_ascii=False)


class _HttpJson:
    """Minimal JSON-over-HTTPS helper with bounded retries on transient failures."""

    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}

    def __init__(self, timeout: float = 120.0, retries: int = 4):
        """Configure the per-request timeout and the number of retries."""
        self._timeout = timeout
        self._retries = retries

    def post(self, url: str, headers: dict, body: dict) -> dict:
        """POST a JSON body and return the decoded JSON response, retrying transient failures."""
        data = json.dumps(body).encode("utf-8")
        for attempt in range(self._retries + 1):
            req = request.Request(url, data=data, headers={**headers, "content-type": "application/json"},
                                  method="POST")
            try:
                with request.urlopen(req, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as failure:
                detail = failure.read().decode("utf-8", "replace")[:500]
                if failure.code not in self.RETRYABLE or attempt == self._retries:
                    raise RuntimeError(f"HTTP {failure.code} from {url}: {detail}") from failure
            except (error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as failure:
                # Includes a server dropping the connection mid-request (seen with a local model server).
                if attempt == self._retries:
                    raise RuntimeError(f"Request to {url} failed: {failure}") from failure
            time.sleep(min(30.0, 2.0 ** attempt))
        raise AssertionError("unreachable")


class AnthropicClient:
    """Anthropic Messages API client that forces exactly one tool call per decision."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024, temperature: float = 1.0):
        """Use `api_key`, or the `ANTHROPIC_API_KEY` environment variable."""
        self.model = model
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._http = _HttpJson()

    def complete(self, system: str, user: str, tools: list[dict], seed: int | None = None,
                 history: Sequence[Turn] = ()) -> LLMResponse:
        """Ask for one tool call (`tool_choice: any`) and parse the first tool-use block. The API has no seed.

        The tools, the system prompt and the conversation so far are marked for prompt caching: the first two are the
        same at every decision of an agent, and the conversation is the prefix of the agent's next prompt.
        """
        api_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                     for t in tools]
        if api_tools:
            api_tools[-1] = {**api_tools[-1], "cache_control": {"type": "ephemeral"}}
        messages = self._messages(history, user)
        last = messages[-1]["content"][-1]
        messages[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}
        body = {"model": self.model, "max_tokens": self._max_tokens, "temperature": self._temperature,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": messages, "tools": api_tools, "tool_choice": {"type": "any"}}
        raw = self._http.post(self.URL, {"x-api-key": self._key, "anthropic-version": "2023-06-01"}, body)
        call, texts = None, []
        for block in raw.get("content", []):
            if block.get("type") == "tool_use" and call is None:
                call = ToolCall(block["name"], dict(block.get("input") or {}))
            elif block.get("type") == "text":
                texts.append(block.get("text", ""))
        usage = raw.get("usage", {})
        return LLMResponse(call, "\n".join(texts), raw.get("model", self.model),
                           usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                           usage.get("cache_read_input_tokens") or 0,
                           usage.get("cache_creation_input_tokens") or 0, raw)

    @staticmethod
    def _messages(history: Sequence[Turn], user: str) -> list[dict]:
        """Messages API turns: each tool result shares a user message with the next prompt, as the API alternates."""
        messages, pending = [], []
        for index, turn in enumerate(history):
            messages.append({"role": "user", "content": [*pending, {"type": "text", "text": turn.user}]})
            if turn.call is None:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": NO_ACTION}]})
                pending = [{"type": "text", "text": turn.result}]
            else:
                messages.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": f"call_{index}", "name": turn.call.name, "input": turn.call.args}]})
                pending = [{"type": "tool_result", "tool_use_id": f"call_{index}", "content": turn.result}]
        messages.append({"role": "user", "content": [*pending, {"type": "text", "text": user}]})
        return messages


class OpenAICompatibleClient:
    """OpenAI-compatible chat-completions client (function tools), e.g. a local proxy or another provider."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None, temperature: float = 1.0,
                 max_tokens: int | None = None, extra: dict | None = None, timeout: float = 120.0):
        """Point at `base_url` (ending in /v1). The key defaults to `OPENAI_API_KEY`, and may be empty for local
        servers.

        `extra` is merged into every request body, e.g. `{"chat_template_kwargs": {"enable_thinking": False}}` for a
        local Qwen server.
        """
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = dict(extra or {})
        self._http = _HttpJson(timeout=timeout)
        self.retries = 0  # replies that had no tool call and were asked again

    def complete(self, system: str, user: str, tools: list[dict], seed: int | None = None,
                 history: Sequence[Turn] = ()) -> LLMResponse:
        """Require a tool call and parse the first function call."""
        body = {"model": self.model, "temperature": self._temperature,
                "messages": [{"role": "system", "content": system}, *self._history_messages(history),
                             {"role": "user", "content": user}],
                "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "required",
                **self._extra}
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        if seed is not None:
            body["seed"] = seed
        headers = {"authorization": f"Bearer {self._key}"} if self._key else {}
        raw = self._http.post(self._url, headers, body)
        call, message = self._parse(raw)
        if call is None:
            # Some servers ignore tool_choice; ask once more, firmly. The decision log keeps the original prompt.
            self.retries += 1
            retry = {**body, "messages": [*body["messages"][:-1], {"role": "user", "content": user + RETRY_NUDGE}]}
            raw = self._http.post(self._url, headers, retry)
            call, message = self._parse(raw)
        usage = raw.get("usage", {})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return LLMResponse(call, message.get("content") or "", raw.get("model", self.model),
                           usage.get("prompt_tokens", 0) - cached, usage.get("completion_tokens", 0), cached, 0, raw)

    @staticmethod
    def _history_messages(history: Sequence[Turn]) -> list[dict]:
        """Chat-completions messages for earlier turns: user, assistant tool call, tool result."""
        messages = []
        for index, turn in enumerate(history):
            messages.append({"role": "user", "content": turn.user})
            if turn.call is None:
                messages += [{"role": "assistant", "content": NO_ACTION}, {"role": "user", "content": turn.result}]
                continue
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call_{index}", "type": "function",
                 "function": {"name": turn.call.name, "arguments": arguments_json(turn.call)}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{index}", "content": turn.result})
        return messages

    @staticmethod
    def _parse(raw: dict) -> tuple[ToolCall | None, dict]:
        """The first function call in a chat-completions response (None if absent or malformed), and the message."""
        message = raw["choices"][0]["message"]
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            arguments = function.get("arguments") or "{}"
            try:
                return ToolCall(function["name"],
                                json.loads(arguments) if isinstance(arguments, str) else arguments), message
            except json.JSONDecodeError:
                return None, message
        return None, message


class ScriptedClient:
    """Deterministic stand-in for a model: `policy(system, user) -> ToolCall`. Used to test the runtime at $0."""

    def __init__(self, policy: Callable[[str, str], ToolCall], model: str = "scripted"):
        """Wrap a pure policy function. The same prompt must always yield the same action."""
        self.model = model
        self._policy = policy

    def complete(self, system: str, user: str, tools: list[dict], seed: int | None = None,
                 history: Sequence[Turn] = ()) -> LLMResponse:
        """Return the policy's action for this prompt (deterministic, so `seed` is unused; it sees no history)."""
        return LLMResponse(self._policy(system, user), model=self.model)
