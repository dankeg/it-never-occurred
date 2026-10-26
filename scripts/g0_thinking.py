"""Gate G0 (DECISION_MODEL_PLAN §4): does reasoning raise a local model's agreement with Sol's logged choices?

Every logged decision of a Sol run is sent to the local model (a running `scripts/local_model.py` server) exactly as
Sol received it: its state document rebuilt from the decision log and verified against the logged digest (with the
prompt context of the commit that recorded the run), and the tool definitions of that commit. Each decision is asked
once with thinking off and once with thinking on, each with Qwen's recommended sampling for that mode. A reply agrees
when its call matches Sol's under the formal Match (`same_choice`). Results are written one line per call to
`<out>.jsonl` (resumable: finished calls are skipped), and the summary to `<out>.json`: agreement per mode, overall and
on non-speech decisions, the paired difference with a bootstrap interval, and seconds and output tokens per decision.

Usage: `python3.11 scripts/g0_thinking.py --run sol_pilot_sabotage --prompt-context runs/ctx556.json --out runs/g0`
(the local model must be serving; this script never starts or stops it). Go if thinking raises agreement by at least
5 points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.candidates import same_choice  # noqa: E402
from orchestrator.decisions import DecisionLog, prompt_digest  # noqa: E402
from orchestrator.llm import OpenAICompatibleClient, Turn  # noqa: E402

MODES = {
    # Qwen's recommended sampling: non-thinking 0.7 / 0.8 / 20 with presence 1.5 (as the agents run);
    # thinking 0.6 / 0.95 / 20
    "off": {"temperature": 0.7, "max_tokens": 384,
            "extra": {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.8, "top_k": 20,
                      "presence_penalty": 1.5}},
    "on": {"temperature": 0.6, "max_tokens": 8192,
           "extra": {"chat_template_kwargs": {"enable_thinking": True}, "top_p": 0.95, "top_k": 20}},
}
FREE_TEXT = {"speak", "write_book", "write_sign"}


def states(decisions, system: dict, tools: list) -> list[tuple]:
    """(decision, history) for every decision, with the history rebuilt as the dataset builder does; raises if a
    rebuilt prompt does not reproduce the logged digest (the model would not see what Sol saw)."""
    result, turns = [], {}
    for decision in decisions:
        if decision.user_prompt.startswith("Your notes:"):
            turns[decision.agent] = []
        history = tuple(turns.setdefault(decision.agent, []))
        digest = prompt_digest(decision.model, system[decision.agent], decision.user_prompt, tools, history)
        if digest != decision.digest:
            raise ValueError(f"State of {decision.agent} at {decision.game_time} does not match its digest")
        result.append((decision, history))
        turns[decision.agent].append(Turn(decision.user_prompt, decision.tool_call, decision.outcome))
    return result


class ThinkingGate:
    """Asks the local model every logged decision in both modes and summarizes agreement with Sol."""

    def __init__(self, args):
        """Load the decision log and the prompt context."""
        self._args = args
        context = json.loads(Path(args.prompt_context).read_text(encoding="utf-8"))
        self._system, self._tools = context["system"], context["tools"]
        log = DecisionLog(args.runs_dir / args.run / "orchestrator" / "decisions.jsonl")
        self._cases = states(log.decisions(), self._system, self._tools)
        self._path = Path(str(args.out) + ".jsonl")

    def _done(self) -> dict:
        """Calls already recorded, keyed by (mode, agent, game time)."""
        if not self._path.exists():
            return {}
        rows = [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {(row["mode"], row["agent"], row["game_time"]): row for row in rows}

    def run(self) -> dict:
        """Ask every missing (mode, decision) pair, mode by mode in time order, then write and return the summary."""
        done = self._done()
        for mode in self._args.modes.split(","):
            settings = MODES[mode]
            client = OpenAICompatibleClient(self._args.model, self._args.base_url, api_key="",
                                            temperature=settings["temperature"], max_tokens=settings["max_tokens"],
                                            extra=settings["extra"], timeout=1800.0)
            for decision, history in self._cases:
                key = (mode, decision.agent, decision.game_time)
                if key in done:
                    continue
                started = time.monotonic()
                try:
                    response = client.complete(self._system[decision.agent], decision.user_prompt, self._tools,
                                               history=history)
                    call, error = response.tool_call, None
                    tokens = response.output_tokens
                except RuntimeError as failure:
                    call, error, tokens = None, str(failure)[:300], 0
                row = {"mode": mode, "agent": decision.agent, "game_time": decision.game_time,
                       "logged": {"name": decision.tool, "args": decision.args},
                       "call": None if call is None else {"name": call.name, "args": call.args},
                       "agree": same_choice(call, decision.tool_call), "free_text": decision.tool in FREE_TEXT,
                       "seconds": round(time.monotonic() - started, 2), "output_tokens": tokens, "error": error}
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[key] = row
        summary = self.summary(list(done.values()))
        Path(str(self._args.out) + ".json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        return summary

    @staticmethod
    def summary(rows: list[dict], samples: int = 2000, seed: int = 0) -> dict:
        """Agreement per mode (all and non-speech), the paired on-minus-off difference with a 95% bootstrap interval
        over decisions asked in both modes, and time and output tokens per decision."""
        by_mode: dict[str, dict] = {}
        for row in rows:
            by_mode.setdefault(row["mode"], {})[(row["agent"], row["game_time"])] = row
        result = {}
        for mode, table in by_mode.items():
            values = list(table.values())
            non_speech = [row for row in values if not row["free_text"]]
            result[mode] = {"decisions": len(values),
                            "agreement": sum(row["agree"] for row in values) / len(values),
                            "non_speech_decisions": len(non_speech),
                            "non_speech_agreement": (sum(row["agree"] for row in non_speech) / len(non_speech)
                                                     if non_speech else None),
                            "no_call": sum(row["call"] is None for row in values),
                            "seconds_median": statistics.median(row["seconds"] for row in values),
                            "seconds_mean": statistics.fmean(row["seconds"] for row in values),
                            "output_tokens_mean": statistics.fmean(row["output_tokens"] for row in values)}
        if {"on", "off"} <= set(by_mode):
            both = sorted(set(by_mode["on"]) & set(by_mode["off"]))
            for label, keys in (("all", both), ("non_speech", [k for k in both if not by_mode["on"][k]["free_text"]])):
                differences = [by_mode["on"][k]["agree"] - by_mode["off"][k]["agree"] for k in keys]
                generator = random.Random(seed)
                means = sorted(statistics.fmean(generator.choice(differences) for _ in differences)
                               for _ in range(samples))
                result[f"difference_{label}"] = {"pairs": len(keys), "on_minus_off": statistics.fmean(differences),
                                                 "ci95": [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]}
        return result


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the gate, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--prompt-context", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="off,on")
    parser.add_argument("--model", default=str(Path.home() / "models/Qwen3.6-35B-A3B-4bit"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    print(json.dumps(ThinkingGate(parser.parse_args(argv)).run(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
