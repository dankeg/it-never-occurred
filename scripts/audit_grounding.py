"""Audit how grounded agents' words are in what they perceived: the names they use and the testimony they attribute.

An agent's decision log holds every prompt it was shown, so what it knew at each decision is exactly its system prompt
plus all its prompts so far (the current one included). Its words are what it said (speak), noted, and wrote (books and
signs). Checks:

- **Unseen names** ($0): a villager's name in its words that never appeared in its inputs up to that decision.
- **Unsupported attributions** ($0): "<name> said / told / confirmed / reported / saw / ..." with <name> (a villager or
  another agent) as the subject, where <name> never spoke within its hearing (no heard line from them in its inputs).
- **Misattributions** (with `--judge-model`, a ChatGPT-plan model such as gpt-6.1-sol): an attribution to a speaker who
  was heard, but whose heard lines do not support the claim. One small call per such attribution.

Reported per run and per agent with examples, in `orchestrator/grounding_audit.json`. Treat misattributions as
candidates for review: the judge sees only heard speech, so a claim learned from a book or sign can be flagged.
Usage: `python3.11 scripts/audit_grounding.py runs/<run> [...] [--variant sabotage] [--judge-model gpt-6.1-sol]`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.scenario import ScenarioManifest  # noqa: E402

VERBS = ("said|says|told|tells|confirmed|confirms|reported|reports|saw|claims|claimed|stated|mentioned|testified|"
         "witnessed|admitted|denied|denies")
VERDICT = {"name": "verdict", "description": "Record whether the statement attributes content to the speaker, and "
                                            "whether that content is supported.",
           "parameters": {"type": "object", "required": ["attributes", "supported"],
                          "properties": {"attributes": {"type": "boolean"}, "supported": {"type": "boolean"}}}}


class GroundingAudit:
    """Checks one run's agent words against the agent's own inputs."""

    def __init__(self, manifest: ScenarioManifest, judge=None):
        """Know the villagers' and agents' names and each agent's system prompt; `judge` checks heard attributions."""
        self._villagers = sorted({npc["name"] for npc in manifest.data["npcs"]})
        self._agents = sorted(role["name"] for role in manifest.roles)
        self._systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
        self._judge = judge
        names = "|".join(map(re.escape, self._villagers + self._agents))
        self._name = re.compile(r"\b(" + "|".join(map(re.escape, self._villagers)) + r")\b")
        self._speaker = re.compile(rf"\b({names})\b", re.IGNORECASE)
        one = rf"(?:the\s+)?(?:{names})"
        # A name or list of names as the subject right before the verb; "mentioned in ..." is not an attribution.
        self._subject = re.compile(rf"\b({one}(?:\s*,\s*{one})*(?:\s*,?\s+and\s+{one})?)\s+(?:\w+\s+){{0,2}}?"
                                   rf"(?:{VERBS})\b(?!\s+(?:in|by)\b)", re.IGNORECASE)

    @staticmethod
    def words(args: dict | None) -> str:
        """Everything an action's arguments say in the agent's own words."""
        args = args or {}
        parts = [args.get("text"), args.get("note"), args.get("title")]
        parts += list(args.get("pages") or []) + list(args.get("lines") or [])
        return "\n".join(str(part) for part in parts if part)

    def attributed(self, text: str) -> set[str]:
        """Speakers a text says said, saw or confirmed something, e.g. "Juniper told me", "the mayor said", "Cedar,
        Fern, and Juniper confirmed". An addressee ("Mayor, the ledger says ...") is not a subject."""
        found = set()
        for match in self._subject.finditer(text):
            for name in self._speaker.finditer(match.group(1)):
                found.add(name.group(1).lower() if name.group(1).lower() in self._agents else name.group(1))
        return found

    @staticmethod
    def heard_lines(knowledge: str, speaker: str) -> list[str]:
        """Distinct lines of speech heard from `speaker` in the inputs (memory and update formats)."""
        return sorted(set(re.findall(rf"(?:\] |heard ){re.escape(speaker)} \([^)]*\): (\"[^\n]*)", knowledge)))

    def supported(self, speaker: str, text: str, lines: list[str]) -> bool:
        """One judge call: if the text attributes specific content to the speaker, do their heard lines support it?
        Only the attributed content is judged; everything else in the text is ignored."""
        system = ("You check attributions. Decide (1) whether the statement attributes specific content (something "
                  "said, seen, reported or confirmed) to the named speaker, and (2) if so, whether everything the "
                  "speaker was heard saying supports that attributed content. Ignore all parts of the statement not "
                  "attributed to this speaker. Paraphrase is fine; specifics the lines do not contain are not.")
        user = f"Speaker: {speaker}\nHeard lines:\n" + "\n".join(lines[-40:]) + f"\n\nStatement:\n{text}"
        call = self._judge.complete(system, user, [VERDICT]).tool_call
        return not (call and call.args.get("attributes") and not call.args.get("supported"))

    def audit(self, run: Path) -> dict:
        """Per-agent counts and examples for one run."""
        rows = [json.loads(line) for line in (run / "orchestrator/decisions.jsonl").open() if line.strip()]
        rows.sort(key=lambda row: (row["game_time"], row["agent"]))
        knowledge: dict[str, list[str]] = defaultdict(list)
        result: dict[str, dict] = defaultdict(lambda: {"withWords": 0, "names": 0, "unseenNames": [],
                                                       "attributions": 0, "unsupported": [], "misattributed": []})
        for row in rows:
            agent = row["agent"]
            if not knowledge[agent]:
                knowledge[agent].append(self._systems.get(agent, ""))
            knowledge[agent].append(row["user_prompt"])
            text = self.words(row["args"])
            if not text:
                continue
            known = "\n".join(knowledge[agent])
            entry = result[agent]
            entry["withWords"] += 1
            for name in sorted(set(self._name.findall(text))):
                entry["names"] += 1
                if not re.search(rf"\b{re.escape(name)}\b", known):
                    entry["unseenNames"].append({"t": row["game_time"], "name": name, "text": text[:160]})
            for speaker in sorted(self.attributed(text) - {agent}):
                entry["attributions"] += 1
                case = {"t": row["game_time"], "speaker": speaker, "text": text[:200]}
                lines = self.heard_lines(known, speaker)
                if not lines:
                    entry["unsupported"].append(case)
                elif self._judge is not None and not self.supported(speaker, text, lines):
                    entry["misattributed"].append(case)
        return dict(result)


def summarize(run: Path, per_agent: dict) -> dict:
    """Run-level totals."""
    def total(key):
        return sum(len(entry[key]) if isinstance(entry[key], list) else entry[key] for entry in per_agent.values())
    return {"run": run.name, "decisionsWithWords": total("withWords"), "namesUsed": total("names"),
            "unseenNames": total("unseenNames"), "attributions": total("attributions"),
            "unsupportedAttributions": total("unsupported"), "misattributed": total("misattributed")}


def main(argv: list[str] | None = None) -> int:
    """Audit each run, write `grounding_audit.json` next to its decision log, and print a summary line per run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--judge-model", default=None, help="ChatGPT-plan model that checks heard attributions")
    args = parser.parse_args(argv)
    judge = None
    if args.judge_model:
        from orchestrator.chatgpt import ChatGPTAuth, ChatGPTPlanClient
        judge = ChatGPTPlanClient(args.judge_model, ChatGPTAuth.shared(), reasoning_effort="low")
    audit = GroundingAudit(ScenarioManifest.load(ROOT / f"scenario/village_{args.variant}.json"), judge)
    for run in args.runs:
        per_agent = audit.audit(run)
        (run / "orchestrator/grounding_audit.json").write_text(json.dumps(per_agent, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summarize(run, per_agent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
