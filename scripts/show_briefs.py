"""The instructions each agent received in one episode, exactly as sent to the model ($0, offline).

Each agent's system prompt is fixed for the whole episode: the world's rules, what everyone in town knows, and the
role's public description and private brief (`orchestrator/agent.py`), built from the episode's `scenario.json`.
Each printed prompt is checked against the episode's log: together with the agent's first recorded observation
prompt and its tools, it must reproduce the SHA-256 digest recorded with that decision.

Usage: `python3 scripts/show_briefs.py runs/jub_L2_1 [--agent mayor] [--private-only]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.decisions import prompt_digest  # noqa: E402
from orchestrator.jubilee import JubileeManifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print each agent's system prompt (or only its private brief) and whether it matches the log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--agent", help="only this agent (e.g. mayor, secretary, innkeeper)")
    parser.add_argument("--private-only", action="store_true", help="print only each role's private brief")
    args = parser.parse_args(argv)
    manifest = JubileeManifest(json.loads((args.run / "scenario.json").read_text(encoding="utf-8")))
    first = {}
    for line in (args.run / "orchestrator" / "decisions.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") == "model" and row["agent"] not in first:
            first[row["agent"]] = row
    for agent in manifest.agents(epoch_turns=8):
        if args.agent and agent.name != args.agent:
            continue
        system = agent.system_prompt()
        row = first.get(agent.name)
        check = ("no model decision logged" if row is None else
                 "matches the logged digest" if prompt_digest(row["model"], system, row["user_prompt"], agent.tools)
                 == row["digest"] else "DOES NOT match the logged digest")
        print(f"===== {agent.name} ({agent.brief.role}): {check}")
        print(agent.brief.private if args.private_only else system)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
