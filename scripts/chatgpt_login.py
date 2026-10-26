"""Sign MinecraftSim in with ChatGPT so agent runs can use your ChatGPT plan (provider `chatgpt-plan`).

Opens your browser to approve this app's use of your plan, then saves the credentials owner-only in
`~/.config/minecraftsim/chatgpt.json`. Revoke them, or set this app's weekly usage cap, in ChatGPT's settings.

Usage:
  python3.11 scripts/chatgpt_login.py                # sign in (or reuse a saved sign-in) and list your plan's models
  python3.11 scripts/chatgpt_login.py --test <slug>  # also make one tool-call request with that model
  python3.11 scripts/chatgpt_login.py --again        # sign in again (e.g. after revoking)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.chatgpt import CREDENTIALS, ChatGPTAuth, ChatGPTPlanClient  # noqa: E402

TEST_TOOL = {"name": "move_to", "description": "Walk to a block position.",
             "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                                                             "z": {"type": "integer"}}, "required": ["x", "y", "z"]}}


def main(argv: list[str] | None = None) -> int:
    """Sign in if needed, list models, and optionally test one model with a tool call."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", metavar="MODEL", default=None, help="make one tool-call request with this model")
    parser.add_argument("--again", action="store_true", help="sign in again even if credentials are saved")
    args = parser.parse_args(argv)
    if CREDENTIALS.exists() and not args.again:
        auth = ChatGPTAuth.load()
    else:
        auth = ChatGPTAuth(ChatGPTAuth.load().credentials if CREDENTIALS.exists() else None)
        auth.sign_in()
    print(f"Signed in as {auth.credentials.email} (app client {auth.credentials.client_id}).")
    client = ChatGPTPlanClient(args.test or "", auth)
    print("Models in your plan:")
    for model in client.list_models():
        if model.get("visibility") != "hide":
            print(f"  {model['slug']:14s} {model.get('description', '')}")
    if args.test:
        started = time.monotonic()
        response = client.complete("You are a person in a small town. Act with one tool call.",
                                   "You stand at (0, -60, 0). Walk to the well at (5, -60, 3).", [TEST_TOOL])
        print(f"{args.test}: tool call {response.tool_call} in {time.monotonic() - started:.1f} s; "
              f"input {response.input_tokens} (+{response.cache_read_tokens} cached), output {response.output_tokens}")
        if response.tool_call is None:
            print("No tool call: this model or flow may not support function tools.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
