"""Agent decisions on a ChatGPT plan, through "Sign in with ChatGPT" for locally run apps.

A Plus or Pro subscriber can let a locally run app use their plan for Responses API requests: the app registers itself
at the first browser sign-in (OAuth with PKCE, no client secret) and receives tokens whose `chatgpt.tokens.use.direct`
scope bills requests to the plan's allowance (see developers.openai.com/siwc/token-sharing-open-source).

- `ChatGPTAuth` signs in once (`sign_in`, interactive), keeps the credentials in `~/.config/minecraftsim/chatgpt.json`
  (owner-only), and refreshes the hourly access token as needed.
- `ChatGPTPlanClient` is an `LLMClient`: one streamed Responses API request per decision, with the agent's earlier
  turns as input items and the tools as function tools. The plan flow accepts no sampling parameters, no
  `max_output_tokens` and no stored responses, so requests set `store: false` and `stream: true` only.

When the plan's (or the app's) usage limit is reached, requests raise `UsageLimitReached`, which ends a run like a
model-call budget instead of turning into no-action decisions.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Iterable, Sequence
from urllib import error, parse, request
import uuid
import webbrowser

from .llm import NO_ACTION, RETRY_NUDGE, LLMResponse, ToolCall, Turn, UsageLimitReached, arguments_json

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
APP_NAME = "MinecraftSim"
CREDENTIALS = Path.home() / ".config/minecraftsim/chatgpt.json"
USAGE_ERRORS = ("subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable")


@dataclass
class Credentials:
    """One signed-in account for this app on this machine (the issued client and its tokens)."""

    client_id: str
    ext_agent_host_id: str
    subject: str
    email: str
    id_token: str
    access_token: str
    refresh_token: str
    expires_at: float

    def save(self, path: Path = CREDENTIALS) -> None:
        """Write atomically with owner-only permissions."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(descriptor, 0o600)  # a stale temporary file keeps its old mode otherwise
        with os.fdopen(descriptor, "w") as handle:
            json.dump(asdict(self), handle)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path = CREDENTIALS) -> Credentials:
        """Read saved credentials (run `scripts/chatgpt_login.py` first)."""
        return cls(**json.loads(path.read_text()))


def pkce_pair() -> tuple[str, str]:
    """A PKCE verifier and its S256 challenge (base64url without padding)."""
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def jwt_claims(token: str) -> dict:
    """The claims of a JWT, without verifying its signature (see `ChatGPTAuth._accept`)."""
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class ChatGPTAuth:
    """Signs this app in with ChatGPT and keeps a valid access token."""

    _shared: ChatGPTAuth | None = None
    _shared_lock = threading.Lock()

    def __init__(self, credentials: Credentials | None = None, path: Path = CREDENTIALS):
        """Use saved credentials, if any; `sign_in` creates them."""
        self.credentials = credentials
        self._path = path
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path = CREDENTIALS) -> ChatGPTAuth:
        """Auth from the saved credentials."""
        return cls(Credentials.load(path), path)

    @classmethod
    def shared(cls) -> ChatGPTAuth:
        """One auth per process. Refresh tokens rotate, so clients that refreshed independently (e.g. parallel study
        branches) would invalidate each other's tokens."""
        with cls._shared_lock:
            if cls._shared is None:
                # Without saved credentials, cached judgements still work; a real call raises "Not signed in".
                cls._shared = cls.load() if CREDENTIALS.exists() else cls()
            return cls._shared

    def sign_in(self, timeout: float = 600.0) -> Credentials:
        """Run the browser sign-in once (the user approves plan use), then save and return the credentials.

        A first sign-in registers the app (`client_id=dynamic_agent_client`) and receives its own client id; later
        sign-ins reuse that id and the host id.
        """
        known = self.credentials
        host_id = known.ext_agent_host_id if known else f"urn:uuid:{uuid.uuid4()}"
        verifier, challenge = pkce_pair()
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        callback = _Callback.start()
        query = {"client_id": known.client_id if known else "dynamic_agent_client", "agent_name_hint": APP_NAME,
                 "ext_agent_host_id": host_id, "response_type": "code", "redirect_uri": callback.redirect_uri,
                 "scope": SCOPES, "resource": RESOURCE, "state": state, "nonce": nonce,
                 "code_challenge_method": "S256", "code_challenge": challenge}
        if known:
            query["login_hint"] = known.email
        url = f"{AUTHORIZE_URL}?{parse.urlencode(query)}"
        print(f"Approve MinecraftSim's use of your ChatGPT plan in the browser:\n{url}", flush=True)
        webbrowser.open(url)
        params = callback.wait(timeout)
        if params.get("state") != state:
            raise RuntimeError("Sign-in state mismatch")
        if "error" in params:
            raise RuntimeError(f"Sign-in refused: {params['error']}")
        client_id = params.get("client_id") or (known.client_id if known else None)
        if not client_id or client_id == "dynamic_agent_client":
            raise RuntimeError("Sign-in did not return the app's client id")
        tokens = _post_form(TOKEN_URL, {"grant_type": "authorization_code", "code": params["code"],
                                        "client_id": client_id, "code_verifier": verifier,
                                        "redirect_uri": callback.redirect_uri, "resource": RESOURCE})
        claims = self._accept(tokens["id_token"], client_id, nonce)
        self.credentials = Credentials(client_id, host_id, claims["sub"], claims.get("email", ""), tokens["id_token"],
                                       tokens["access_token"], tokens["refresh_token"],
                                       time.time() + tokens.get("expires_in", 3600))
        self.credentials.save(self._path)
        return self.credentials

    def access_token(self) -> str:
        """A valid access token, refreshed a few minutes before it expires (thread-safe)."""
        with self._lock:
            credentials = self.credentials
            if credentials is None:
                raise RuntimeError("Not signed in: run scripts/chatgpt_login.py")
            if time.time() > credentials.expires_at - 300:
                tokens = _post_form(TOKEN_URL, {"grant_type": "refresh_token",
                                                "refresh_token": credentials.refresh_token,
                                                "client_id": credentials.client_id, "resource": RESOURCE})
                credentials.access_token = tokens["access_token"]
                credentials.refresh_token = tokens.get("refresh_token", credentials.refresh_token)
                credentials.expires_at = time.time() + tokens.get("expires_in", 3600)
                credentials.save(self._path)
            return credentials.access_token

    @staticmethod
    def _accept(id_token: str, client_id: str, nonce: str) -> dict:
        """Check the ID token's issuer, audience, expiry and nonce. It comes straight from the token endpoint over
        TLS, which authenticates the issuer (OpenID Connect Core 3.1.3.7), so its signature is not re-verified."""
        claims = jwt_claims(id_token)
        audience = claims.get("aud")
        if (claims.get("iss") != ISSUER or client_id not in (audience if isinstance(audience, list) else [audience])
                or claims.get("exp", 0) < time.time() or claims.get("nonce") != nonce):
            raise RuntimeError("ID token failed validation")
        return claims


class _Callback:
    """A one-shot loopback HTTP listener for the OAuth redirect."""

    def __init__(self, server: HTTPServer):
        self._server = server
        self.params: dict | None = None
        self.redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"

    @classmethod
    def start(cls) -> _Callback:
        """Listen on an ephemeral loopback port."""
        holder: list[_Callback] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server API)
                url = parse.urlparse(self.path)
                if url.path == "/callback":
                    holder[0].params = dict(parse.parse_qsl(url.query))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"MinecraftSim is signed in. You can close this tab.")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_args):
                pass

        callback = cls(HTTPServer(("127.0.0.1", 0), Handler))
        holder.append(callback)
        return callback

    def wait(self, timeout: float) -> dict:
        """Serve until the redirect arrives; return its query parameters."""
        deadline = time.monotonic() + timeout
        self._server.timeout = 1.0
        try:
            while self.params is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Sign-in was not completed in time")
                self._server.handle_request()
        finally:
            self._server.server_close()
        return self.params


def _post_form(url: str, fields: dict) -> dict:
    """POST a form and return the JSON response, with the server's error text on failure."""
    req = request.Request(url, data=parse.urlencode(fields).encode(),
                          headers={"content-type": "application/x-www-form-urlencoded"}, method="POST")
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read())
    except error.HTTPError as failure:
        raise RuntimeError(f"HTTP {failure.code} from {url}: "
                           f"{failure.read().decode('utf-8', 'replace')[:300]}") from failure


def input_items(history: Sequence[Turn], user: str) -> list[dict]:
    """Responses API input: each earlier turn as a user message, the function call and its output; then `user`."""
    items: list[dict] = []
    for index, turn in enumerate(history):
        items.append({"role": "user", "content": turn.user})
        if turn.call is None:
            items += [{"role": "assistant", "content": NO_ACTION}, {"role": "user", "content": turn.result}]
            continue
        items.append({"type": "function_call", "call_id": f"call_{index}", "name": turn.call.name,
                      "arguments": arguments_json(turn.call)})
        items.append({"type": "function_call_output", "call_id": f"call_{index}", "output": turn.result})
    items.append({"role": "user", "content": user})
    return items


class StreamInterrupted(RuntimeError):
    """A response stream that ended before completing, or a server-side error event: transient, worth a retry."""


def read_events(lines: Iterable[bytes]) -> tuple[ToolCall | None, str, dict]:
    """Parse a Responses API event stream: the first function call, any text, and the usage on completion.

    Raises `UsageLimitReached` for plan usage errors, `StreamInterrupted` for an incomplete stream or a `server_error`
    event, and `RuntimeError` for other failures.
    """
    call, text, usage, completed = None, [], {}, False
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        kind = event.get("type", "")
        if kind == "response.output_item.done" and event["item"].get("type") == "function_call" and call is None:
            item = event["item"]
            try:
                call = ToolCall(item["name"], json.loads(item.get("arguments") or "{}"))
            except json.JSONDecodeError:
                call = None
        elif kind == "response.output_text.delta":
            text.append(event.get("delta", ""))
        elif kind == "response.completed":
            usage, completed = event["response"].get("usage") or {}, True
        elif kind in ("response.failed", "error"):
            detail = event.get("response", {}).get("error") or event.get("error") or event
            code = detail.get("code", "") if isinstance(detail, dict) else ""
            kind = (UsageLimitReached if code in USAGE_ERRORS else StreamInterrupted if code == "server_error"
                    else RuntimeError)
            raise kind(f"Responses API: {detail}")
    if not completed:
        raise StreamInterrupted("Responses stream ended before response.completed")
    return call, "".join(text), usage


class ChatGPTPlanClient:
    """`LLMClient` for a model in the signed-in user's ChatGPT plan (Responses API, streamed)."""

    URL = f"{RESOURCE}/responses"
    RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, model: str, auth: ChatGPTAuth, reasoning_effort: str | None = None, timeout: float = 120.0,
                 retries: int = 6):
        """`model` is a slug from `list_models`; `reasoning_effort` (e.g. "low") is sent when given. `timeout` is the
        socket inactivity limit per attempt (a stalled stream once held an episode for 910 s at 300 s)."""
        self.model = model
        self._auth = auth
        self._reasoning = reasoning_effort
        self._timeout = timeout
        self._retries = retries
        self.retries = 0  # replies that had no tool call and were asked again

    @staticmethod
    def backoff(attempt: int) -> float:
        """Seconds before retry `attempt` (0-based): 5, 10, 20, 40, then 60, about three minutes over six retries, so a
        short provider outage does not end a long run."""
        return min(60.0, 5.0 * 2.0 ** attempt)

    def list_models(self) -> list[dict]:
        """The models in the user's plan catalog (slug, display name, description, reasoning levels, ...)."""
        req = request.Request(f"{RESOURCE}/models", headers={"authorization": f"Bearer {self._auth.access_token()}"})
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read()).get("models", [])

    def complete(self, system: str, user: str, tools: list[dict], seed: int | None = None,
                 history: Sequence[Turn] = ()) -> LLMResponse:
        """Require a function call; ask once more if the reply had none. The plan flow has no seed or sampling knobs."""
        body = {"model": self.model, "instructions": system, "input": input_items(history, user),
                "tools": [{"type": "function", "name": t["name"], "description": t["description"],
                           "parameters": t["parameters"]} for t in tools],
                "tool_choice": "required", "parallel_tool_calls": False, "store": False, "stream": True}
        if self._reasoning:
            body["reasoning"] = {"effort": self._reasoning}
        call, text, usage = self._stream(body)
        if call is None:
            self.retries += 1
            body["input"] = input_items(history, user + RETRY_NUDGE)
            call, text, usage = self._stream(body)
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        return LLMResponse(call, text, self.model, usage.get("input_tokens", 0) - cached, usage.get("output_tokens", 0),
                           cached, 0, {"usage": usage})

    def _stream(self, body: dict) -> tuple[ToolCall | None, str, dict]:
        """POST one streamed request, retrying transient failures (honouring Retry-After)."""
        data = json.dumps(body).encode()
        for attempt in range(self._retries + 1):
            req = request.Request(self.URL, data=data, method="POST", headers={
                "authorization": f"Bearer {self._auth.access_token()}", "content-type": "application/json",
                "accept": "text/event-stream"})
            try:
                with request.urlopen(req, timeout=self._timeout) as response:
                    return read_events(response)
            except error.HTTPError as failure:
                detail = failure.read().decode("utf-8", "replace")[:500]
                if any(code in detail for code in USAGE_ERRORS):
                    raise UsageLimitReached(f"ChatGPT plan usage: {detail}") from failure
                retryable = failure.code in self.RETRYABLE or 500 <= failure.code < 600  # e.g. a CDN's 520
                if not retryable or attempt == self._retries:
                    raise RuntimeError(f"HTTP {failure.code} from {self.URL}: {detail}") from failure
                wait = float(failure.headers.get("retry-after") or self.backoff(attempt))
            except (error.URLError, OSError, HTTPException, StreamInterrupted) as failure:  # transient: retry
                if attempt == self._retries:
                    raise RuntimeError(f"Request to {self.URL} failed: {failure}") from failure
                wait = self.backoff(attempt)
            time.sleep(wait)
        raise AssertionError("unreachable")
