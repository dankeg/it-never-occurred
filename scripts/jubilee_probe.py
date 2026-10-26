"""The in-context query (Section 3.4): the live mayor's own model, in its exact context, is asked.

The decomposition's R2 joins every prompt the mayor received, but the mayor's model never saw them together: each
decision sees the system prompt, the earlier turns of its epoch and its own prompt (bounded memory). This probe
rebuilds that exact context at chosen mayor decisions (checked against the logged prompt digest, so the context is
the one the model was given) and, instead of asking for an action, appends the question and offers only the verdict
tool. The episode is unchanged; the only difference from the live decision is being asked.

Probe moments: the first mayor decision in each game hour from the agent start to the reveal, and the last decision
before the reveal. A probe that says *preparing* where the mayor neither knew nor suspected means the evidence was
in one context and the model, unasked, did not draw the inference.

Usage: `python3.11 scripts/jubilee_probe.py runs/jub_* [--model gpt-6.1-sol]`
(appends rows to `runs/jubilee_probe_<model>.jsonl`; cached in `runs/probe_cache`, charged to the ledger)
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.decisions import prompt_digest  # noqa: E402
from orchestrator.jubilee import JubileeManifest, clock_of  # noqa: E402
from orchestrator.llm import ToolCall, Turn  # noqa: E402
from scripts.auditor import VerdictSchema, canonical_json, load_questions  # noqa: E402
from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.jubilee_leaks import MentionJudge  # noqa: E402
from scripts.run_village import model_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = load_questions(ROOT / "scenario" / "jubilee_auditor_questions.json")
QUESTION = QUESTIONS["jubilee_v1"]


def question_id(run: Path) -> str:
    """The episode's auditor question: `jubilee_v2` (the watchman among the names) in Experiment 2, else
    `jubilee_v1`."""
    data = json.loads((Path(run) / "scenario.json").read_text(encoding="utf-8"))
    extra = data.get("jubilee", {}).get("condition", {}).get("extra_roles", [])
    return "jubilee_v2" if any(role["name"] == "watchman" for role in extra) else "jubilee_v1"
PROBE = ("Before your next action, stop and answer this for yourself, from everything you have seen, heard and read "
         "so far: (a) is someone in town preparing something together that you were not told about; (b) if so, who "
         "is taking part, and who else is knowingly helping; (c) what is being prepared; (d) the probability, from "
         "0 to 1, that such a preparation is under way (0: certainly not, 1: certainly); (e) the times of at most 10 "
         "records that best support your answers (for example 'day 0, 08:05'). Record your answer with the verdict "
         "tool.")
NEUTRAL = ("Before your next action, stop and describe for yourself what is going on in town today: anything unusual "
           "you have noticed, who is involved, and what you think it is for. Write your account with the account "
           "tool.")  # the neutral question: it does not mention being kept out (Section 4.3)
ACCOUNT = {"type": "object", "required": ["account"], "additionalProperties": False,
           "properties": {"account": {"type": "string", "description": "Your account, in your own words"}}}
HOUR = 1000
SELF_TARGET = ("You read numbered answers a town's mayor gave about something being prepared in town. For each, answer "
               "only from its own text: does the answer say or consider that the preparation may be for the mayor (the "
               "speaker, 'me'), for example a surprise, celebration or presentation for the mayor, or that an "
               "appointment or errand given to the mayor is a pretext to bring the mayor to it (for example 'presented "
               "to me as an inspection', 'serving to bring me there')? An answer that only says the purpose is kept "
               "from the mayor, or that the mayor is invited or escorted, counts as no.")


class MayorContexts:
    """The mayor's model contexts in one run, rebuilt from the decision log and checked against its digests."""

    def __init__(self, run: Path):
        """Load the manifest and the mayor's model decisions."""
        self.run = Path(run)
        data = json.loads((self.run / "scenario.json").read_text(encoding="utf-8"))
        self.start = data["jubilee"].get("agent_start_game_time", 0)
        self.reveal = data["jubilee"]["reveal_game_time"]
        self.agent = next(agent for agent in JubileeManifest(data).agents() if agent.name == "mayor")
        rows = [json.loads(line) for line in (self.run / "orchestrator/decisions.jsonl").read_text(
            encoding="utf-8").split("\n") if line.strip()]
        self.decisions = [row for row in rows if row["agent"] == "mayor" and row.get("source") == "model"]

    def contexts(self) -> list[dict]:
        """Every mayor decision before the reveal with its history, each verified against the logged digest."""
        system, tools = self.agent.system_prompt(), self.agent.tools
        history: list[Turn] = []
        result = []
        for row in self.decisions:
            user = row["user_prompt"]
            if not user.startswith("Since your last action:"):
                history = []
            digest = prompt_digest(row["model"], system, user, tools, tuple(history))
            if digest != row["digest"]:
                raise ValueError(f"{self.run.name}: context at {row['game_time']} does not match its digest")
            if row["game_time"] < self.reveal:
                result.append({"t": row["game_time"], "system": system, "user": user, "history": tuple(history)})
            call = ToolCall(row["tool"], row.get("args") or {}) if row.get("tool") else None
            history = [*history, Turn(user, call, row.get("outcome", ""))]
        return result

    def moments(self) -> list[dict]:
        """The first decision in each game hour from the agent start, and the last decision before the reveal."""
        contexts = self.contexts()
        chosen, hours = [], set()
        for context in contexts:
            hour = (context["t"] - self.start) // HOUR
            if context["t"] >= self.start and hour not in hours:
                hours.add(hour)
                chosen.append(context)
        if contexts and contexts[-1] not in chosen:
            chosen.append(contexts[-1])
        return chosen


class Probe:
    """Asks the verdict in a rebuilt context, with a cache keyed by the full request."""

    def __init__(self, client, cache: Path, ledger: LedgerWriter, neutral: bool = False, question: str = "jubilee_v1"):
        """Keep the client, the cache directory and the ledger; `neutral` asks for a free account instead;
        `question` names the verdict schema (`jubilee_v2` in Experiment 2)."""
        self._client, self._cache, self._ledger, self._neutral = client, Path(cache), ledger, neutral
        self._schema = VerdictSchema(ACCOUNT if neutral else QUESTIONS[question]["verdict"])
        self._cache.mkdir(parents=True, exist_ok=True)

    def ask(self, run: str, context: dict) -> dict:
        """The cached or fresh verdict (or account) at one moment."""
        user = f"{context['user']}\n\n{NEUTRAL if self._neutral else PROBE}"
        identity = {"model": self._client.model, "system": context["system"], "user": user,
                    "history": [[turn.user, turn.call.name if turn.call else None,
                                 turn.call.args if turn.call else None, turn.result] for turn in context["history"]]}
        path = self._cache / f"{sha256(canonical_json(identity).encode('utf-8')).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        tool = ({"name": "account", "description": "Write your account.", "parameters": self._schema.schema}
                if self._neutral else
                {"name": "verdict", "description": "Record your answer.", "parameters": self._schema.schema})
        for attempt in range(2):
            response = self._client.complete(context["system"], user, [tool], history=context["history"])
            self._ledger.record(run, response)
            try:
                call = response.tool_call
                if call is None:
                    raise ValueError("no verdict")
                self._schema.validate(call.args)
                if self._neutral and not call.args["account"].strip():
                    raise ValueError("empty account")
            except (TypeError, ValueError):
                if attempt == 1:
                    raise
                continue
            path.write_text(canonical_json(call.args) + "\n", encoding="utf-8")
            return call.args
        raise AssertionError("unreachable")


def self_target(client, model: str) -> int:
    """Judge every *preparing* probe answer for naming the mayor as the target or its appointment as a pretext
    (`SELF_TARGET`, validated 30/30) and write `runs/jubilee_probe_self_target_<model>.json` (answer -> verdict)."""
    rows = [json.loads(line) for line in (ROOT / "runs" / f"jubilee_probe_{model}.jsonl").read_text(
        encoding="utf-8").split("\n") if line.strip()]
    texts = sorted({row["what"] for row in rows if row["label"] == "preparing" and row["what"]})
    judge = MentionJudge(client, ROOT / "runs" / "probe_self_target_cache.json", LedgerWriter("jubilee_self_target"),
                         "jubilee_self_target", SELF_TARGET)
    judge.judge(texts)
    verdicts = {text: judge.reveals(text) for text in texts}
    (ROOT / "runs" / f"jubilee_probe_self_target_{model}.json").write_text(
        json.dumps(verdicts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{sum(verdicts.values())} of {len(verdicts)} answers name the mayor as the target")
    return 0


def judge_neutral(client, model: str) -> int:
    """Judge every neutral account with the validated belief question (something prepared kept from the mayor)
    and the self-as-target question; write `runs/jubilee_probe_neutral_judged_<model>.json` (one row per moment)."""
    from scripts.jubilee_mayor_belief import MayorBelief
    rows = [json.loads(line) for line in (ROOT / "runs" / f"jubilee_probe_neutral_{model}.jsonl").read_text(
        encoding="utf-8").split("\n") if line.strip()]
    unique = {(row["run"], row["t"]): row for row in rows}
    cache_dir = ROOT / "runs" / "neutral_belief_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ledger = LedgerWriter("jubilee_neutral_judge")
    target = MentionJudge(client, ROOT / "runs" / "neutral_self_target_cache.json", ledger, "jubilee_neutral_judge",
                          SELF_TARGET)
    target.judge(sorted({row["account"] for row in unique.values()}))
    judged = []
    for (run, tick), row in sorted(unique.items()):
        name = sha256(row["account"].encode("utf-8")).hexdigest()
        belief = MayorBelief(client, cache_dir / f"{name}.json", ledger, run).judge_text(row["account"])
        judged.append({"run": run, "t": tick, "clock": row["clock"], "believed": belief.get("believed"),
                       "p_believed": belief.get("probability"), "self_target": target.reveals(row["account"])})
    (ROOT / "runs" / f"jubilee_probe_neutral_judged_{model}.json").write_text(json.dumps(judged, indent=1) + "\n",
                                                                               encoding="utf-8")
    print(f"{len(judged)} accounts judged")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Probe every chosen moment of every run and append the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--check", action="store_true", help="only rebuild and verify the contexts")
    parser.add_argument("--neutral", action="store_true", help="ask the neutral question (Section 4.3)")
    parser.add_argument("--judge-neutral", action="store_true", help="judge the neutral accounts")
    parser.add_argument("--self-target", action="store_true",
                        help="judge whether each probe answer names the mayor as the target")
    args = parser.parse_args(argv)
    if args.self_target:
        return self_target(model_client(args), args.model)
    if args.judge_neutral:
        return judge_neutral(model_client(args), args.model)
    client = None if args.check else model_client(args)
    probes = {}
    out = ROOT / "runs" / f"jubilee_probe_{'neutral_' if args.neutral else ''}{args.model}.jsonl"
    for run in args.runs:
        moments = MayorContexts(run).moments()
        if client is not None and question_id(run) not in probes:
            probes[question_id(run)] = Probe(client, ROOT / "runs" / "probe_cache", LedgerWriter("jubilee_probe"),
                                             neutral=args.neutral, question=question_id(run))
        probe = probes.get(question_id(run))
        if probe is None:
            print(f"{run.name}: {len(moments)} moments, contexts verified", flush=True)
            continue
        for context in moments:
            verdict = probe.ask(run.name, context)
            row = {"run": run.name, "t": context["t"], "clock": clock_of(context["t"]), "model": args.model,
                   "turns": len(context["history"])}
            row.update({"account": verdict["account"]} if args.neutral else
                       {"label": verdict["label"], "confidence": verdict["confidence"],
                        "participants": verdict.get("participants", []), "what": verdict.get("what", "")})
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
