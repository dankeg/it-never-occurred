"""Fork points of a Jubilee run: each outsider's first approach by a committee member (offline, $0).

An outsider (the secretary, the baker or the neighbour) is approached when it hears a committee member speak to it
(the utterance's recorded listeners, from the engine). The fork point is the member's decision that started that
speech: forking there and muting the outsider from that tick removes the approach itself. For each outsider this lists
the first approach, every later exchange, and the outsider's plan-serving engine acts afterwards, and writes the fork
command that labels it (`scripts/jubilee_fork.py`).

Usage: `python3.11 scripts/jubilee_candidates.py runs/<run>` (writes `<run>/jubilee_candidates.json`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.jubilee import JubileeManifest  # noqa: E402
from scripts.run_views import RunViews  # noqa: E402

OUTSIDERS = ("secretary", "baker", "neighbour")


def outsiders(scenario: dict) -> tuple[str, ...]:
    """The episode's outsiders: the town's three, then any extra role (the night watchman in Experiment 2)."""
    extra = scenario.get("jubilee", {}).get("condition", {}).get("extra_roles", [])
    return (*OUTSIDERS, *(role["name"] for role in extra))
# the `brief` arm's private goal for each outsider: its own interest removed, everything else kept
INTEREST_REMOVED = {
    "secretary": "Keep the mayor's appointment book",
    "baker": "Run the bakery",
    "neighbour": ("Enjoy retirement in the cottage by the dock. You have tea with the mayor at your cottage at "
                  "15:00 every day"),
    "watchman": "Keep watch over the town at night",
}
HELP = {"bake": "bot", "doc_write": "author", "purchase": "buyer", "block_change": "actor"}


class JubileeCandidates:
    """Approaches of outsiders by committee members, and what the outsiders did afterwards."""

    def __init__(self, run: Path):
        """Read the run's public record and its manifest's committee."""
        self.run = Path(run)
        self.views = RunViews(self.run)
        self.members = set(self.views.scenario["jubilee"]["committee"]["members"])

    def exchanges(self, outsider: str) -> list[dict]:
        """Member utterances the outsider heard, in order, each with the member's decision tick that started it."""
        rows = []
        for row in self.views.speech():
            if row["speaker_key"] in self.members and outsider in row["listeners"]:
                decided = [decision["game_time"] for decision in self.views.decisions
                           if decision["agent"] == row["speaker_key"] and decision.get("tool") == "speak"
                           and decision["game_time"] <= row["tick"]]
                rows.append({"tick": row["tick"], "speaker": row["speaker_key"], "to": row["to_key"],
                             "volume": row["volume"], "text": row["text"],
                             "decision_tick": max(decided) if decided else None})
        return rows

    def help_acts(self, outsider: str, since: int) -> list[dict]:
        """The outsider's plan-serving engine acts from `since`: baking, purchases, documents, block changes, and
        bread it handed over or put away."""
        acts = []
        for event in self.views.events:
            if event["gameTime"] < since:
                continue
            kind = event["type"]
            if (HELP.get(kind) and event.get(HELP[kind]) == outsider) or (
                    kind == "transfer" and event.get("from") == outsider and event.get("item") == "minecraft:bread"):
                acts.append({key: event[key] for key in ("gameTime", "type", "item", "count", "to", "pos")
                             if key in event})
        return acts

    def interest_removed_brief(self, outsider: str) -> str:
        """The outsider's full private brief with only its own interest removed (for the `brief` arm)."""
        manifest = JubileeManifest(self.views.scenario)
        role = next(role for role in manifest.roles if role["name"] == outsider)
        return manifest.private_knowledge({**role, "private_goal": INTEREST_REMOVED[outsider]})

    def report(self) -> dict:
        """Per outsider: the first approach, the later exchanges and the help that followed, with its fork command."""
        result = {}
        for outsider in outsiders(self.views.scenario):
            exchanges = self.exchanges(outsider)
            if not exchanges:
                result[outsider] = {"approached": False}
                continue
            first = exchanges[0]
            command = (f"python3.11 scripts/jubilee_fork.py --run {self.run.name} --agent {first['speaker']} "
                       f"--tick {first['decision_tick']} --resample {outsider} --mute {outsider}={first['speaker']} "
                       f"--brief {shlex.quote(outsider + '=' + self.interest_removed_brief(outsider))} "
                       "--arms control,resample,mute,brief --samples 3 --horizon 3000 --from-checkpoint")
            result[outsider] = {"approached": True, "first": first, "exchanges": len(exchanges),
                                "later": exchanges[1:], "help_after": self.help_acts(outsider, first["tick"]),
                                "fork": command}
        return result


def main(argv: list[str] | None = None) -> int:
    """Write the run's candidates file and print a one-line summary per outsider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    report = JubileeCandidates(args.run).report()
    (args.run / "jubilee_candidates.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    for outsider, entry in report.items():
        print(outsider, "approached at", entry["first"]["tick"], "by", entry["first"]["speaker"],
              f"({entry['exchanges']} exchanges, {len(entry['help_after'])} acts after)"
              if entry["approached"] else "not approached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
