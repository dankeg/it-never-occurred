"""IT-04: replace one logged village decision using DecisionFork's real swap arm.

Record a short scripted village run, replay its prefix/control, and swap a builder act for a short wait at the
same tick. Pass: exactly that decision is forced and accepted, outcome.json marks the override, and other agents
reuse matching logged prompts/actions and call the scripted model only where their digests differ. The control
must replay the factual decisions and hashes without calls. Negative control: DecisionFork must refuse the
identical logged act through its same_choice guard, before creating another world or making any call.
"""

from __future__ import annotations

import argparse
import json

if __package__:
    from . import harness
    from .fork_decision import DecisionFork
    from .test_e3 import decision_trace
    from .test_m5_village import M5VillageExperiment, VillageScriptedPolicy
else:
    import harness
    from fork_decision import DecisionFork
    from test_e3 import decision_trace
    from test_m5_village import M5VillageExperiment, VillageScriptedPolicy

from orchestrator.candidates import same_choice  # noqa: E402
from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS  # noqa: E402

RUN_TICKS = 3_000  # factual run length after the fixture's agent start (6,000 on live-staged fixtures)


def reuse_checks(factual: DecisionLog, swapped: DecisionLog, target: str, tick: int, end: int) -> dict[str, bool]:
    """Check actual other-agent decisions against matching factual prompts, rejecting vacuous or false reuse."""
    others = [decision for decision in swapped.decisions()
              if decision.agent != target and tick <= decision.game_time < end]
    matching, correct = 0, bool(others)
    for decision in others:
        logged = factual.get(decision.agent, decision.game_time)
        if logged is not None and logged.digest == decision.digest:
            matching += 1
            correct &= (decision.source == "replay" and decision.tool == logged.tool and decision.args == logged.args)
        else:
            correct &= decision.source == "model"
    return {"other agents reused at least one unchanged prompt": matching > 0,
            "other-agent reuse stops exactly at changed prompts": correct}


class OverrideExperiment:
    """Exercise the in-process decision-fork runner, including its refusal of an identical swapped act."""

    def __init__(self, args: argparse.Namespace):
        """Fix the scripted sabotage village and assigned port pair without adding production options."""
        self.args = argparse.Namespace(**vars(args), variant="sabotage")
        self.args.game_port = self.args.port + 1

    def run(self) -> bool:
        """Record a factual run, run a real control/swap fork, and check decision-level gating and its guard."""
        village = M5VillageExperiment(self.args)
        end = village._manifest.data["layout"]["agent_start_game_time"] + RUN_TICKS
        factual, _, report = village.run_once(
            "it04_factual", ScriptedClient(VillageScriptedPolicy(village._manifest).decide), None,
            end_tick=end, min_gap=THINKING_TICKS, rest=NIGHT_REST)
        first = factual.log.decisions()[0].game_time
        point = next((decision for decision in factual.log.decisions()
                      if decision.agent == "builder" and first < decision.game_time < end - 400
                      and decision.tool not in (None, "wait", "keep_going")), None)
        if point is None:
            raise harness.HarnessError("No post-start builder decision available to override")
        swap = ToolCall("wait", {"ticks": 200})
        args = argparse.Namespace(**vars(self.args), run="it04_factual", agent=point.agent, tick=point.game_time,
                                  swap=json.dumps({"name": swap.name, "args": swap.args}),
                                  horizon=end - point.game_time, arms="swap", model="scripted",
                                  provider="anthropic", max_calls=None, epoch_turns=1, forced=False)
        fork = DecisionFork(args)
        result = fork.run()
        swapped = DecisionLog(fork.directory("swap") / "orchestrator/decisions.jsonl")
        forced = [decision for decision in swapped.decisions() if decision.source == "forced"]
        negative = False
        identical = argparse.Namespace(**vars(args))
        identical.swap = json.dumps({"name": point.tool, "args": point.args})
        try:
            DecisionFork(identical)
        except ValueError as error:
            negative = same_choice(point.tool_call, point.tool_call) and "matches the logged choice" in str(error)
        control = fork.directory("control") / "orchestrator"
        checks = {
            "factual run finished with model decisions": report["stopped"] is None and factual.stats.model_calls > 0,
            "swap executed at its tick with the forced call": (
                len(forced) == 1 and forced[0].game_time == point.game_time and forced[0].agent == point.agent
                and forced[0].tool_call == swap and forced[0].outcome == "started"),
            "outcome and decision log mark the override": (
                result["stopped"] is None and result["arms"]["swap"]["override_applied"]
                and result["arms"]["swap"]["point"]["swap"] == {"name": swap.name, "args": swap.args}),
            "control made zero model calls": result["arms"]["control"]["calls"] == 0,
            "control replayed factual decisions and hashes": (
                decision_trace(DecisionLog(control / "decisions.jsonl")) == decision_trace(factual.log)
                and json.loads((control / "hashes.json").read_text(encoding="utf-8")) == factual.hashes),
            **reuse_checks(factual.log, swapped, point.agent, point.game_time, end),
            "negative control: identical act refused by same_choice": negative,
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-04: forced decision, gated reuse, and identical-act refusal")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse common gate arguments, and run only the requested experiment."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if OverrideExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-04: override experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
