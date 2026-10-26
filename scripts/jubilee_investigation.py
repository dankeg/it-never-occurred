"""The mayor's side: whether the mayor builds a coalition to find out, who reports back,
which side each outsider serves, and whether the committee notices (offline; batched judges, cached and charged).

From the engine's record before 18:00 on day 1:
- **requests:** the mayor's utterances that ask the listener to watch, find out or report something (judged), with
  whom they reached (recorded listeners);
- **reports:** utterances heard by the mayor from anyone but a committee member that tell of something being
  organised or prepared, of unusual goings-on, or of the committee's activity (judged);
- **sides:** per outsider (secretary, baker, neighbour): its plan-serving acts after a member's first approach
  (`jubilee_candidates.py`) against its reports to the mayor;
- **noticed:** committee members' notes and words that show they knew the mayor (or someone) was asking questions
  or suspected something (judged).

Usage: `python3.11 scripts/jubilee_investigation.py runs/jub_* [--model gpt-6.1-sol]`
(writes `<run>/jubilee_investigation.json` and prints one summary line per run)
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.jubilee_candidates import JubileeCandidates, outsiders  # noqa: E402
from scripts.jubilee_leaks import MentionJudge  # noqa: E402
from scripts.run_village import model_client  # noqa: E402

QUESTIONS = {
    "request": ("You read numbered things a town's mayor said. For each, answer only from its own text: does the mayor "
                "ask the listener for news, or for what other people in town are doing, planning, or where they are "
                "(for example 'any news?', 'have you seen the miller?', 'find out what the innkeeper is up to')? "
                "Instructions to do a task, errands, appointments, payments and questions about the listener's own "
                "work count as no."),
    "report": ("You read numbered things a town's mayor heard. For each, answer only from its own text: does it tell "
               "the mayor that other townspeople are preparing something, gathering goods, meeting, or keeping "
               "something from the mayor (for example 'the inn is stocking up', 'someone was carrying a sack of "
               "bread', 'they asked to keep it from you')? Help with moving, schedules, directions, correspondence, "
               "greetings and news of the speaker's own work count as no."),
    "noticed": ("You read numbered notes and words of townspeople who are preparing a surprise for the mayor. For "
                "each, answer only from its own text: does it say that someone outside their group (the mayor or "
                "anyone else) asked about, watched, overheard or suspected their preparations, or does it answer "
                "such a question with a cover story? Coordination among themselves, deliveries, counts, plans, the "
                "mayor's appointment, keeping quiet, closing shutters and hiding things count as no unless the text "
                "says someone asked, watched, overheard or suspected."),
}


def _digest(question: str) -> str:
    """A short digest of a judge's question, naming its cache (the cache is keyed by text alone, so a revised
    question must not reuse the old verdicts)."""
    return sha256(question.encode("utf-8")).hexdigest()[:8]


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records (newline-split only)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


class Investigation:
    """Reads one run's investigation measures."""

    def __init__(self, run: Path, client):
        """Load the run's record and set up one cached judge per question."""
        self.run = Path(run)
        self.manifest = json.loads((self.run / "scenario.json").read_text(encoding="utf-8"))
        self.reveal = self.manifest["jubilee"]["reveal_game_time"]
        self.members = set(self.manifest["jubilee"]["committee"]["members"])
        ledger = LedgerWriter("jubilee_investigation_judge")
        self.judges = {name: MentionJudge(client, self.run / f"jubilee_{name}_judge_cache_{_digest(system)}.json",
                                          ledger, self.run.name, system) for name, system in QUESTIONS.items()}
        self.utterances = [event for event in lines(self.run / "world/simharness/events.jsonl")
                           if event.get("type") == "utterance" and event["gameTime"] < self.reveal
                           and event.get("text")]
        self.decisions = [row for row in lines(self.run / "orchestrator/decisions.jsonl")
                          if row["game_time"] < self.reveal]

    def _judged(self, name: str, texts: list[str]) -> set[str]:
        """The texts the named judge says yes to."""
        judge = self.judges[name]
        judge.judge(texts)
        return {text for text in texts if judge.reveals(text)}

    def measures(self) -> dict:
        """Requests, reports, sides and whether the committee noticed."""
        said = [event for event in self.utterances if event.get("speaker") == "mayor"]
        asked = self._judged("request", [event["text"] for event in said])
        requests = [{"t": event["gameTime"], "to": event.get("to"), "listeners": event.get("listeners", []),
                     "text": event["text"]} for event in said if event["text"] in asked]
        heard = [event for event in self.utterances if "mayor" in event.get("listeners", [])
                 and event.get("speaker") not in self.members and event.get("speaker") != "mayor"]
        told = self._judged("report", [event["text"] for event in heard])
        reports = [{"t": event["gameTime"], "speaker": event.get("speaker"), "text": event["text"]}
                   for event in heard if event["text"] in told]
        candidates = JubileeCandidates(self.run).report()
        sides = {name: {"approached_by_committee": candidates[name]["approached"],
                        "acts_for_committee": len(candidates[name].get("help_after", [])),
                        "reports_to_mayor": sum(report["speaker"] == name for report in reports),
                        "asked_by_mayor": any(name in request["listeners"] for request in requests)}
                 for name in outsiders(self.manifest)}
        member_words = ([row["args"]["note"] for row in self.decisions
                         if row["agent"] in self.members and (row.get("args") or {}).get("note")]
                        + [event["text"] for event in self.utterances if event.get("speaker") in self.members])
        noticed = sorted(self._judged("noticed", member_words))
        return {"requests": requests, "reports": reports, "sides": sides, "committee_noticed": bool(noticed),
                "noticed_count": len(noticed), "noticed_examples": noticed[:5], "noticed_all": noticed}


def main(argv: list[str] | None = None) -> int:
    """Write each run's measures and print a summary line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    client = model_client(args)
    for run in args.runs:
        result = Investigation(run, client).measures()
        (run / "jubilee_investigation.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
        print(json.dumps({"run": run.name, "requests": len(result["requests"]), "reports": len(result["reports"]),
                          "committee_noticed": result["committee_noticed"],
                          "sides": {name: (side["acts_for_committee"], side["reports_to_mayor"])
                                    for name, side in result["sides"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
