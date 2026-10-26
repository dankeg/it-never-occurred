"""Unit tests for the ChatGPT plan client's offline pieces: PKCE, request input, and event-stream parsing."""

import base64
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import time
import unittest

from orchestrator.chatgpt import (ISSUER, USAGE_ERRORS, ChatGPTAuth, ChatGPTPlanClient, Credentials, StreamInterrupted,
                                  input_items, jwt_claims, pkce_pair, read_events)
from orchestrator import chatgpt
from orchestrator.llm import ToolCall, Turn, UsageLimitReached


def event(payload: dict) -> bytes:
    """One server-sent event data line."""
    return b"data: " + json.dumps(payload).encode() + b"\n"


class ChatGPTPlanTest(unittest.TestCase):
    """The plan client builds standard Responses API input and reads streamed function calls and usage."""

    def test_pkce_challenge_is_s256_of_verifier(self):
        verifier, challenge = pkce_pair()
        digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(challenge, digest)

    def test_history_becomes_input_items(self):
        items = input_items((Turn("u0", ToolCall("move_to", {"z": 1, "x": 2}), "started"),
                             Turn("u1", None, "no action")), "u2")
        self.assertEqual([item.get("type") or item["role"] for item in items],
                         ["user", "function_call", "function_call_output", "user", "assistant", "user", "user"])
        self.assertEqual(items[1]["arguments"], '{"x": 2, "z": 1}')
        self.assertEqual(items[2], {"type": "function_call_output", "call_id": "call_0", "output": "started"})

    def test_reads_function_call_and_usage(self):
        call, text, usage = read_events([
            b"event: response.output_item.done\n",
            event({"type": "response.output_item.done",
                   "item": {"type": "function_call", "name": "move_to", "arguments": json.dumps({"x": 5})}}),
            event({"type": "response.completed", "response": {"usage": {
                "input_tokens": 100, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 60}}}})])
        self.assertEqual(call, ToolCall("move_to", {"x": 5}))
        self.assertEqual(usage["input_tokens_details"]["cached_tokens"], 60)

    def test_usage_limit_and_incomplete_streams_raise(self):
        with self.assertRaises(UsageLimitReached):
            read_events([event({"type": "response.failed",
                                "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}})])
        with self.assertRaises(StreamInterrupted):
            read_events([event({"type": "response.output_text.delta", "delta": "hi"})])
        with self.assertRaises(StreamInterrupted):
            read_events([event({"type": "error", "error": {"code": "server_error"}})])

    def test_an_interrupted_stream_is_retried(self):
        """A stream that ends before completing is asked again, like a network error, and the retry's call is used."""
        complete = [event({"type": "response.output_item.done",
                           "item": {"type": "function_call", "name": "wait", "arguments": "{}"}}),
                    event({"type": "response.completed", "response": {"usage": {"input_tokens": 3}}})]
        replies = [[event({"type": "response.output_text.delta", "delta": "cut"})], complete]

        class Reply(list):
            """A urlopen result usable as a context manager."""
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False

        attempts = []
        def urlopen(req, timeout):
            attempts.append(req)
            return Reply(replies[len(attempts) - 1])
        auth = type("Auth", (), {"access_token": lambda self: "token"})()
        original_urlopen, original_sleep = chatgpt.request.urlopen, chatgpt.time.sleep
        chatgpt.request.urlopen, chatgpt.time.sleep = urlopen, lambda seconds: None
        try:
            call, _, usage = ChatGPTPlanClient("gpt-6-luna", auth)._stream({"model": "gpt-6-luna"})
        finally:
            chatgpt.request.urlopen, chatgpt.time.sleep = original_urlopen, original_sleep
        self.assertEqual((call, usage, len(attempts)), (ToolCall("wait", {}), {"input_tokens": 3}, 2))

    def test_any_5xx_is_retried_and_backoff_is_bounded(self):
        """A CDN's 520 is retried like a 502; the backoff grows to a one-minute cap."""
        from email.message import Message
        from urllib import error as urlerror
        complete = [event({"type": "response.output_item.done",
                           "item": {"type": "function_call", "name": "wait", "arguments": "{}"}}),
                    event({"type": "response.completed", "response": {"usage": {}}})]

        class Reply(list):
            """A urlopen result usable as a context manager."""
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False

        attempts = []
        def urlopen(req, timeout):
            attempts.append(req)
            if len(attempts) == 1:
                raise urlerror.HTTPError("https://example", 520, "unknown", Message(), None)
            return Reply(complete)
        auth = type("Auth", (), {"access_token": lambda self: "token"})()
        original_urlopen, original_sleep = chatgpt.request.urlopen, chatgpt.time.sleep
        chatgpt.request.urlopen, chatgpt.time.sleep = urlopen, lambda seconds: None
        try:
            call, _, _ = ChatGPTPlanClient("gpt-6-luna", auth)._stream({"model": "gpt-6-luna"})
        finally:
            chatgpt.request.urlopen, chatgpt.time.sleep = original_urlopen, original_sleep
        self.assertEqual((call, len(attempts)), (ToolCall("wait", {}), 2))
        self.assertEqual([ChatGPTPlanClient.backoff(n) for n in range(6)], [5.0, 10.0, 20.0, 40.0, 60.0, 60.0])


class RecordedEventStreamTest(unittest.TestCase):
    """Recorded byte streams pin calls, text, completion requirements and error classifications."""

    def test_first_function_call_text_and_complete_usage_are_returned(self):
        """The first call is retained alongside joined text and the exact completed usage object."""
        usage = {"input_tokens": 100, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 60}}
        lines = [event({"type": "response.output_text.delta", "delta": "I will "}),
                 event({"type": "response.output_item.done", "item": {
                     "type": "function_call", "name": "move_to", "arguments": '{"x": 5}'}}),
                 event({"type": "response.output_item.done", "item": {
                     "type": "function_call", "name": "wait", "arguments": '{"ticks": 40}'}}),
                 event({"type": "response.output_text.delta", "delta": "walk."}),
                 event({"type": "response.completed", "response": {"usage": usage}})]
        self.assertEqual(read_events(lines), (ToolCall("move_to", {"x": 5}), "I will walk.", usage))

    def test_text_only_stream_ignores_non_data_lines_and_done(self):
        """Non-data lines, empty data and DONE markers do not alter a completed text-only reply."""
        lines = [b": keepalive\n", b"event: response.output_text.delta\n", b"id: 1\n", b"\n", b"data:\n",
                 event({"type": "response.output_text.delta", "delta": "hello "}), b"data: [DONE]\n",
                 event({"type": "response.output_text.delta", "delta": "world"}),
                 event({"type": "response.completed", "response": {}}), b"data: [DONE]\n"]
        self.assertEqual(read_events(lines), (None, "hello world", {}))

    def test_stream_without_completed_event_raises_even_after_done(self):
        """A DONE marker cannot substitute for the required response.completed event."""
        with self.assertRaisesRegex(RuntimeError, "^Responses stream ended before response.completed$"):
            read_events([event({"type": "response.output_text.delta", "delta": "unfinished"}), b"data: [DONE]\n"])

    def test_each_usage_error_code_raises_usage_limit_reached(self):
        """Every configured usage error is classified as UsageLimitReached in either failure envelope."""
        for code in USAGE_ERRORS:
            for kind in ("response.failed", "error"):
                with self.subTest(code=code, kind=kind):
                    detail = {"code": code, "message": "allowance exhausted"}
                    payload = {"type": kind, **({"response": {"error": detail}} if kind == "response.failed"
                                               else {"error": detail})}
                    with self.assertRaisesRegex(UsageLimitReached, code):
                        read_events([event(payload)])

    def test_other_error_codes_raise_runtime_error(self):
        """Non-usage API failures raise RuntimeError rather than a usage-limit exception."""
        for payload in ({"type": "error", "error": {"code": "invalid_request", "message": "bad tool"}},
                        {"type": "response.failed", "response": {"error": {"code": "invalid_request"}}}):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(RuntimeError, "invalid_request") as raised:
                    read_events([event(payload)])
                self.assertNotIsInstance(raised.exception, UsageLimitReached)

    def test_malformed_function_arguments_produce_no_call(self):
        """Malformed function argument JSON yields no call while preserving text and completion usage."""
        self.assertEqual(read_events([
            event({"type": "response.output_item.done", "item": {
                "type": "function_call", "name": "wait", "arguments": "{"}}),
            event({"type": "response.output_text.delta", "delta": "bad arguments"}),
            event({"type": "response.completed", "response": {"usage": {"output_tokens": 2}}})]),
            (None, "bad arguments", {"output_tokens": 2}))


class CredentialPersistenceTest(unittest.TestCase):
    """Credential files are owner-only atomic replacements that round-trip all fields."""

    def test_save_creates_parents_and_atomically_replaces_existing_credentials(self):
        """Saving creates parents, writes mode 0600 and replaces an existing file without leaving a temp file."""
        credentials = Credentials("client", "host", "subject", "user@example.test", "id-token", "access-token",
                                  "refresh-token", 1234567890.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested/config/chatgpt.json"
            self.assertFalse(path.parent.exists())
            credentials.save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()), asdict(credentials))
            self.assertEqual(Credentials.load(path), credentials)
            self.assertEqual(list(path.parent.iterdir()), [path])
            original_inode = path.stat().st_ino
            path.chmod(0o644)
            updated = replace(credentials, access_token="new-access", refresh_token="new-refresh", expires_at=234.0)
            updated.save(path)
            self.assertNotEqual(path.stat().st_ino, original_inode)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(Credentials.load(path), updated)
            self.assertEqual(json.loads(path.read_text()), asdict(updated))
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_stale_world_readable_temporary_file_does_not_leak_its_mode(self):
        """A leftover `chatgpt.tmp` with mode 0644 does not make the saved credentials readable by others."""
        credentials = Credentials("client", "host", "subject", "user@example.test", "id-token", "access-token",
                                  "refresh-token", 1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "chatgpt.json"
            stale = path.with_suffix(".tmp")
            stale.write_text("old")
            stale.chmod(0o644)
            credentials.save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(Credentials.load(path), credentials)


class IdTokenValidationTest(unittest.TestCase):
    """Offline ID-token checks decode unpadded claims and enforce issuer, audience, expiry and nonce."""

    def _token(self, claims):
        """Encode synthetic claims as an unpadded JWT payload without generating a signature."""
        payload = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).rstrip(b"=").decode()
        return f"header.{payload}.signature"

    def test_jwt_claims_decode_an_unpadded_payload(self):
        """JWT payloads missing required base64 padding decode to the original claims."""
        token = self._token({"sub": "a"})
        self.assertNotEqual(len(token.split(".")[1]) % 4, 0)
        self.assertNotIn("=", token)
        self.assertEqual(jwt_claims(token), {"sub": "a"})

    def test_wrong_issuer_audience_expiry_or_nonce_is_rejected(self):
        """A wrong issuer, audience, expired timestamp or nonce independently fails ID-token validation."""
        valid = {"iss": ISSUER, "aud": "client", "exp": time.time() + 3600, "nonce": "expected", "sub": "person"}
        for changed in ({"iss": "https://wrong.example.test"}, {"aud": "someone-else"},
                        {"exp": time.time() - 3600}, {"nonce": "different"}):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, "^ID token failed validation$"):
                    ChatGPTAuth._accept(self._token({**valid, **changed}), "client", "expected")

    def test_matching_string_and_list_audiences_are_accepted(self):
        """Valid tokens accept either a matching audience string or a list containing the client id."""
        for audience in ("client", ["other-client", "client"]):
            with self.subTest(audience=audience):
                claims = {"iss": ISSUER, "aud": audience, "exp": time.time() + 3600,
                          "nonce": "expected", "sub": "person"}
                self.assertEqual(ChatGPTAuth._accept(self._token(claims), "client", "expected"), claims)


if __name__ == "__main__":
    unittest.main()
