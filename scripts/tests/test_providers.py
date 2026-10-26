"""Unit tests for the model clients over HTTP: request bodies, the single retry, usage, plan usage limits, token
refresh under two threads, and the sign-in callback.

A loopback server answers each request with a recorded provider response and keeps the requests it got, so the
clients' real request code runs with no network. These are the only unit tests that open sockets (127.0.0.1 only).
One test points `orchestrator.chatgpt.TOKEN_URL` at the loopback server, since the token endpoint is a module constant.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from urllib import error, parse, request

from orchestrator import chatgpt
from orchestrator.chatgpt import USAGE_ERRORS, ChatGPTAuth, ChatGPTPlanClient, Credentials, input_items
from orchestrator.llm import RETRY_NUDGE, AnthropicClient, OpenAICompatibleClient, ToolCall, Turn, UsageLimitReached
from orchestrator.tools import TOOL_DEFINITIONS

HISTORY = (Turn("u0", ToolCall("move_to", {"z": 1, "x": 2}), "started"),)
WAIT = {"name": "wait", "arguments": json.dumps({"ticks": 20})}


class RecordedProvider:
    """A loopback HTTP server that answers each request with the next recorded response and keeps every request."""

    def __init__(self, responses: list[tuple[int, bytes]], delay: float = 0.0):
        """Queue `(status, body)` responses, each sent `delay` seconds after its request arrives."""
        self.requests: list[dict] = []
        self._responses = list(responses)
        self._delay = delay
        self._lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            """Hands every POST to the provider."""

            def do_POST(self):  # noqa: N802 (http.server API)
                """Record the request and send the next recorded response."""
                provider.answer(self)

            def log_message(self, *_args):
                """Keep test output quiet."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True)

    @property
    def url(self) -> str:
        """The server's base URL."""
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "RecordedProvider":
        """Start serving."""
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        """Stop serving and close the socket."""
        self._server.shutdown()
        self._server.server_close()

    def answer(self, handler: BaseHTTPRequestHandler) -> None:
        """Record one request (path, lower-cased headers, body) and write the next recorded response."""
        body = handler.rfile.read(int(handler.headers.get("content-length", 0)))
        with self._lock:
            self.requests.append({"path": handler.path, "body": body,
                                  "headers": {key.lower(): value for key, value in handler.headers.items()}})
            status, payload = self._responses.pop(0)
        time.sleep(self._delay)
        handler.send_response(status)
        handler.send_header("content-length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def bodies(self) -> list[dict]:
        """The JSON request bodies received, in order."""
        return [json.loads(item["body"]) for item in self.requests]


def chat_reply(tool_call: dict | None, usage: dict | None = None) -> tuple[int, bytes]:
    """A recorded chat-completions response with an optional function call."""
    message = {"role": "assistant", "content": "" if tool_call else "I will walk to the chest."}
    if tool_call:
        message["tool_calls"] = [{"id": "c1", "type": "function", "function": tool_call}]
    return 200, json.dumps({"model": "local", "choices": [{"message": message}], "usage": usage or {}}).encode()


def event_stream(*events: dict) -> tuple[int, bytes]:
    """A recorded Responses API event stream."""
    return 200, b"".join(b"event: x\ndata: " + json.dumps(event).encode() + b"\n\n" for event in events)


CALL_EVENT = {"type": "response.output_item.done", "item": {"type": "function_call", **WAIT}}
TEXT_EVENT = {"type": "response.output_text.delta", "delta": "Let me think."}
COMPLETED = {"type": "response.completed", "response": {"usage": {
    "input_tokens": 100, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 60}}}}


class OpenAICompatibleRequestTest(unittest.TestCase):
    """Chat-completions requests carry the conversation and options; a reply without a tool call is asked once more."""

    def test_request_carries_history_tools_seed_and_extra_options(self):
        """The body holds system, history, user, function tools, `tool_choice: required`, the seed and extra options."""
        with RecordedProvider([chat_reply(WAIT, {"prompt_tokens": 100, "completion_tokens": 7,
                                                 "prompt_tokens_details": {"cached_tokens": 60}})]) as provider:
            client = OpenAICompatibleClient("local", provider.url + "/v1", api_key="k", temperature=0.5,
                                            max_tokens=64, extra={"top_k": 20})
            response = client.complete("sys", "u1", TOOL_DEFINITIONS[:2], seed=7, history=HISTORY)
        body, sent = provider.bodies()[0], provider.requests[0]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["headers"]["authorization"], "Bearer k")
        self.assertEqual([message["role"] for message in body["messages"]], ["system", "user", "assistant", "tool",
                                                                             "user"])
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"x": 2, "z": 1}')
        self.assertEqual(body["tools"], [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS[:2]])
        self.assertEqual((body["tool_choice"], body["seed"], body["max_tokens"], body["top_k"], body["temperature"]),
                         ("required", 7, 64, 20, 0.5))
        self.assertEqual(response.tool_call, ToolCall("wait", {"ticks": 20}))
        self.assertEqual((response.input_tokens, response.cache_read_tokens, response.output_tokens), (40, 60, 7))

    def test_unseeded_request_without_key_has_no_seed_or_authorization(self):
        """An unseeded call sends no `seed`, and a local server without a key gets no authorization header."""
        with RecordedProvider([chat_reply(WAIT)]) as provider:
            OpenAICompatibleClient("local", provider.url + "/v1", api_key="").complete("sys", "u1", TOOL_DEFINITIONS)
        self.assertNotIn("seed", provider.bodies()[0])
        self.assertNotIn("authorization", provider.requests[0]["headers"])

    def test_reply_without_tool_call_is_asked_once_more_with_a_nudge(self):
        """A reply with no tool call is retried once, with the same messages and the nudge appended to the user's."""
        with RecordedProvider([chat_reply(None), chat_reply(WAIT)]) as provider:
            client = OpenAICompatibleClient("local", provider.url + "/v1", api_key="")
            response = client.complete("sys", "u1", TOOL_DEFINITIONS, history=HISTORY)
        first, second = provider.bodies()
        self.assertEqual(second["messages"][:-1], first["messages"][:-1])
        self.assertEqual(second["messages"][-1], {"role": "user", "content": "u1" + RETRY_NUDGE})
        self.assertEqual((client.retries, response.tool_call), (1, ToolCall("wait", {"ticks": 20})))

    def test_second_reply_without_tool_call_is_no_action(self):
        """If the retry also has no tool call, the decision has none; there is no third request."""
        with RecordedProvider([chat_reply(None), chat_reply(None)]) as provider:
            response = OpenAICompatibleClient("local", provider.url + "/v1", api_key="").complete("sys", "u1", [])
        self.assertIsNone(response.tool_call)
        self.assertEqual(len(provider.requests), 2)

    def test_rejected_request_fails_without_retry(self):
        """A non-transient HTTP error raises at once, with the server's text."""
        with RecordedProvider([(400, b"bad request")]) as provider:
            with self.assertRaisesRegex(RuntimeError, "HTTP 400 .*bad request"):
                OpenAICompatibleClient("local", provider.url + "/v1", api_key="").complete("sys", "u1", [])
        self.assertEqual(len(provider.requests), 1)


class AnthropicRequestTest(unittest.TestCase):
    """Messages API requests mark the stable prefix for prompt caching and force one tool call."""

    def test_layout_cache_marks_and_usage(self):
        """System, the last tool and the last user block carry cache marks; usage includes cache reads and writes."""
        reply = {"model": "claude-x", "content": [{"type": "text", "text": "ok"},
                                                  {"type": "tool_use", "name": "wait", "input": {"ticks": 20}}],
                 "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 90,
                           "cache_creation_input_tokens": 30}}
        with RecordedProvider([(200, json.dumps(reply).encode())]) as provider:
            client = AnthropicClient("claude-x", api_key="k")
            client.URL = provider.url + "/v1/messages"
            response = client.complete("sys", "u1", TOOL_DEFINITIONS[:3], history=HISTORY)
        body, headers = provider.bodies()[0], provider.requests[0]["headers"]
        ephemeral = {"type": "ephemeral"}
        self.assertEqual((headers["x-api-key"], headers["anthropic-version"]), ("k", "2023-06-01"))
        self.assertEqual(body["system"], [{"type": "text", "text": "sys", "cache_control": ephemeral}])
        self.assertEqual([tool.get("cache_control") for tool in body["tools"]], [None, None, ephemeral])
        self.assertEqual(body["tools"][0]["input_schema"], TOOL_DEFINITIONS[0]["parameters"])
        self.assertEqual(body["messages"][-1]["content"][-1],
                         {"type": "text", "text": "u1", "cache_control": ephemeral})
        self.assertNotIn("cache_control", body["messages"][0]["content"][-1])
        self.assertEqual(body["tool_choice"], {"type": "any"})
        self.assertEqual((response.tool_call, response.text), (ToolCall("wait", {"ticks": 20}), "ok"))
        self.assertEqual((response.input_tokens, response.output_tokens, response.cache_read_tokens,
                          response.cache_write_tokens), (10, 5, 90, 30))


class ChatGPTPlanRequestTest(unittest.TestCase):
    """Plan requests are streamed Responses API calls with no sampling options; usage limits stop the run."""

    def setUp(self):
        """A signed-in auth with a valid token, saved in a temporary directory."""
        self._folder = tempfile.TemporaryDirectory()
        path = Path(self._folder.name) / "chatgpt.json"
        self.auth = ChatGPTAuth(Credentials("app", "host", "sub", "e@x", "id", "tok", "r1", time.time() + 3600), path)

    def tearDown(self):
        """Remove the temporary directory."""
        self._folder.cleanup()

    def client(self, provider: RecordedProvider) -> ChatGPTPlanClient:
        """A plan client that sends to the loopback provider."""
        client = ChatGPTPlanClient("gpt-x", self.auth, reasoning_effort="low")
        client.URL = provider.url + "/v1/responses"
        return client

    def test_streamed_request_and_usage(self):
        """The body has instructions, input items, function tools and the plan's fixed options; usage is parsed."""
        with RecordedProvider([event_stream(CALL_EVENT, COMPLETED)]) as provider:
            response = self.client(provider).complete("sys", "u1", TOOL_DEFINITIONS[:1], seed=5, history=HISTORY)
        body, headers = provider.bodies()[0], provider.requests[0]["headers"]
        self.assertEqual((headers["authorization"], headers["accept"]), ("Bearer tok", "text/event-stream"))
        self.assertEqual((body["instructions"], body["input"]), ("sys", input_items(HISTORY, "u1")))
        tool = TOOL_DEFINITIONS[0]
        self.assertEqual(body["tools"], [{"type": "function", "name": tool["name"], "description": tool["description"],
                                          "parameters": tool["parameters"]}])
        self.assertEqual({key: body[key] for key in ("tool_choice", "parallel_tool_calls", "store", "stream",
                                                     "reasoning")},
                         {"tool_choice": "required", "parallel_tool_calls": False, "store": False, "stream": True,
                          "reasoning": {"effort": "low"}})
        self.assertFalse({"seed", "temperature", "max_output_tokens"} & set(body))
        self.assertEqual(response.tool_call, ToolCall("wait", {"ticks": 20}))
        self.assertEqual((response.input_tokens, response.cache_read_tokens, response.output_tokens), (40, 60, 9))

    def test_stream_without_call_is_asked_once_more_with_a_nudge(self):
        """A streamed reply with only text is retried once, with the nudge appended to the last user input."""
        with RecordedProvider([event_stream(TEXT_EVENT, COMPLETED), event_stream(CALL_EVENT, COMPLETED)]) as provider:
            client = self.client(provider)
            response = client.complete("sys", "u1", TOOL_DEFINITIONS, history=HISTORY)
        first, second = provider.bodies()
        self.assertEqual(second["input"][:-1], first["input"][:-1])
        self.assertEqual(second["input"][-1], {"role": "user", "content": "u1" + RETRY_NUDGE})
        self.assertEqual((client.retries, response.tool_call), (1, ToolCall("wait", {"ticks": 20})))

    def test_plan_usage_limit_stops_without_retry(self):
        """An HTTP error naming a plan usage code raises `UsageLimitReached` after a single request."""
        for code in USAGE_ERRORS:
            refusal = (429, json.dumps({"error": {"code": code}}).encode())
            with self.subTest(code=code), RecordedProvider([refusal]) as provider:
                with self.assertRaises(UsageLimitReached):
                    self.client(provider).complete("sys", "u1", TOOL_DEFINITIONS)
                self.assertEqual(len(provider.requests), 1)


class TokenRefreshTest(unittest.TestCase):
    """The access token is refreshed once when it is about to expire, even when two threads ask at once (M6)."""

    def setUp(self):
        """Point the token endpoint at nothing yet; keep the original to restore."""
        self._token_url = chatgpt.TOKEN_URL
        self._folder = tempfile.TemporaryDirectory()
        self.path = Path(self._folder.name) / "config" / "chatgpt.json"

    def tearDown(self):
        """Restore the token endpoint and remove the temporary directory."""
        chatgpt.TOKEN_URL = self._token_url
        self._folder.cleanup()

    def test_expired_token_is_refreshed_once_for_two_threads(self):
        """Two threads asking for an expired token cause one refresh; both get the new token, which is saved."""
        auth = ChatGPTAuth(Credentials("app", "host", "sub", "e@x", "id", "old", "r1", time.time() - 10), self.path)
        tokens = {"access_token": "new", "refresh_token": "r2", "expires_in": 3600}
        results: list[str] = []
        with RecordedProvider([(200, json.dumps(tokens).encode())], delay=0.3) as provider:
            chatgpt.TOKEN_URL = provider.url + "/oauth/token"
            threads = [threading.Thread(target=lambda: results.append(auth.access_token())) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
        self.assertEqual(results, ["new", "new"])
        self.assertEqual(len(provider.requests), 1)
        form = dict(parse.parse_qsl(provider.requests[0]["body"].decode()))
        self.assertEqual({key: form[key] for key in ("grant_type", "refresh_token", "client_id")},
                         {"grant_type": "refresh_token", "refresh_token": "r1", "client_id": "app"})
        self.assertEqual(Credentials.load(self.path).refresh_token, "r2")
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_valid_token_is_used_without_refresh(self):
        """A token valid for more than five minutes is returned as is."""
        auth = ChatGPTAuth(Credentials("app", "host", "sub", "e@x", "id", "tok", "r1", time.time() + 600), self.path)
        with RecordedProvider([]) as provider:
            chatgpt.TOKEN_URL = provider.url + "/oauth/token"
            self.assertEqual(auth.access_token(), "tok")
        self.assertEqual(provider.requests, [])

    def test_not_signed_in_raises(self):
        """Without credentials there is no token, and the error says how to sign in."""
        with self.assertRaisesRegex(RuntimeError, "chatgpt_login"):
            ChatGPTAuth(None, self.path).access_token()


class SignInCallbackTest(unittest.TestCase):
    """The sign-in listener returns the redirect's query parameters and gives up after its timeout."""

    def test_redirect_parameters_are_returned(self):
        """Another path gets a 404 and is ignored; the callback's query parameters are returned."""
        callback = chatgpt._Callback.start()
        statuses: list[int] = []

        def browser():
            """Visit a wrong path, then the redirect URI."""
            try:
                request.urlopen(callback.redirect_uri.replace("/callback", "/other"), timeout=5)
            except error.HTTPError as failure:
                statuses.append(failure.code)
            with request.urlopen(callback.redirect_uri + "?code=c1&state=s1", timeout=5) as response:
                statuses.append(response.status)

        thread = threading.Thread(target=browser)
        thread.start()
        params = callback.wait(10)
        thread.join(5)
        self.assertEqual(params, {"code": "c1", "state": "s1"})
        self.assertEqual(statuses, [404, 200])

    def test_wait_times_out(self):
        """With no redirect before the deadline, waiting raises `TimeoutError`."""
        with self.assertRaises(TimeoutError):
            chatgpt._Callback.start().wait(0)


if __name__ == "__main__":
    unittest.main()
