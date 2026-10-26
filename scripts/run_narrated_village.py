"""N1 at village scale (gate V-G5): the village crisis under a narrating game master, with an engine shadow.

The agents (same briefs, prompts and conversation layout as engine runs) act in a world that only a game master (GM)
narrates. The GM starts from a complete text description of the village (`ScenarioManifest.gm_world`) and from each
agent's first view, which is the engine's own rendering of what that agent sees at the start. Every act the GM accepts
is also executed, at the GM's time, in a shadow copy of the real village that nobody sees.

Reported in `narrated_report.json`: the narration-error rate (GM-accepted acts the engine rejects or that fail there,
by reason), timeline compression (acts the GM finished while the engine was still running them), the outcomes the
shadow world actually reached, and model calls for agents and GM.

`--scripted` runs at $0: the scripted village policy as agents and a naive GM that accepts every act instantly and lets
everyone hear all speech, the extreme the shadow must catch. Live runs use `--provider chatgpt-plan` (or mlx-local),
with the GM on `--gm-model` (default: the agents' model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

if __package__:
    from . import harness
    from .provenance import FixtureMetadata
    from .run_village import model_client
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import VillageSession
else:
    import harness
    from provenance import FixtureMetadata
    from run_village import model_client
    from test_m5_village import VillageScriptedPolicy
    from village_session import VillageSession

from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.narration import (GameMaster, NarratedRuntime, ShadowEngine, narrated_outcomes,  # noqa: E402
                                    narration_error, outcome_divergence)
from orchestrator.runtime import NIGHT_REST, POLL, THINKING_TICKS, BudgetExceeded  # noqa: E402
from orchestrator.scenario import KeywordJudge, LLMJudge, PredicateEvaluator, ScenarioManifest  # noqa: E402

SPAN = 6_000  # default run length in ticks after the fixture's start (live-staged fixtures start agents at 6,000)


class NaiveVillageGM:
    """A scripted GM at the naive extreme: every act is possible and instant, and everyone hears all speech."""

    def __init__(self, people: list[str]):
        """Know the agents' names (the people who can hear)."""
        self._people = people

    def resolve(self, system: str, user: str) -> ToolCall:
        """Accept the act in the GM prompt with zero duration, keeping the actor's view as it was."""
        match = re.search(r"\), (\w+) attempts: (\w+) (\{.*\})\n(\w+)'s current view: (.*?)\n\nResolve this act\.$",
                          user, re.S)
        agent, tool, args, view = match.group(1), match.group(2), json.loads(match.group(3)), match.group(5)
        heard = [{"person": other, "text": args.get("text", "")} for other in self._people
                 if tool == "speak" and other != agent]
        return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 0,
                                    "effects": [f"{agent} did {tool}"], "actor_view": view, "observations": heard})


class NarratedVillage:
    """One narrated run of the village with its engine shadow."""

    def __init__(self, args):
        """Keep options and load the scenario."""
        self._args = args
        self._manifest = ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario"
                                               / f"village_{args.variant}.json")

    def _models(self):
        """The agents' model and the GM's: scripted at $0, else live clients (the GM may use a different model)."""
        if self._args.scripted:
            people = [role["name"] for role in self._manifest.roles]
            return (ScriptedClient(VillageScriptedPolicy(self._manifest).decide),
                    ScriptedClient(NaiveVillageGM(people).resolve, model="naive-gm"))
        agents = model_client(self._args)
        gm = model_client(argparse.Namespace(**{**vars(self._args), "model": self._args.gm_model or self._args.model}))
        return agents, gm

    def run(self) -> dict:
        """Run the narrated village to the end time and write the report."""
        args, manifest = self._args, self._manifest
        directory = args.runs_dir / args.name
        session = VillageSession(directory, args.runs_dir / f"village_{args.variant}", manifest,
                                 harness.server_options(args))
        session.clone()
        agent_model, gm_model = self._models()
        epoch_turns = 1 if args.scripted else args.epoch_turns  # the scripted policy reads full prompts
        with session:
            client, now = session.client, session.start
            views = session.first_views()
            config = manifest.data
            goods = {key: config[key] for key in ("carry_limits", "fresh_days", "bake_ticks", "villager_purchases")
                     if key in config}
            gm = GameMaster(gm_model, manifest.gm_world(), views, **goods)
            agents = manifest.agents(epoch_turns)
            runtime = NarratedRuntime(agents, agent_model, gm, directory / "orchestrator",
                                      shadow=ShadowEngine(client, harness.CommandError),
                                      max_model_calls=args.max_calls, min_gap=THINKING_TICKS, rest=NIGHT_REST,
                                      resync=args.resync)
            start = (now + POLL - 1) // POLL * POLL
            end = args.ticks if args.ticks is not None else start + SPAN
            stopped = None
            try:
                runtime.run(start, end)
            except BudgetExceeded as limit:
                stopped = str(limit)
            client.request("save_snapshot")
            suspects = [role["name"] for role in manifest.roles]
            judge = (LLMJudge(agent_model, suspects, manifest.data["evidence"]["expected_chain"])
                     if not args.scripted else KeywordJudge(suspects))
            shadow = PredicateEvaluator(client, directory / "world/simharness/events.jsonl", manifest, judge)
            predicates = manifest.data["success_predicates"]
            outcomes = {"fair_ready": shadow.evaluate(predicates["fair_ready"]),
                        "fair_open": shadow.evaluate(predicates["fair_open"]),
                        "bridge_complete": shadow.evaluate(predicates["builder"]["all"][0]),
                        "roles": {role["name"]: shadow.evaluate(predicates[role["name"]], role["name"])
                                  for role in manifest.roles}}
        narrated = narrated_outcomes(runtime.acts, manifest.data, end)
        report = {"start": start, "end": end, "stopped": stopped, "narration_error": narration_error(runtime.acts),
                  "fixture": FixtureMetadata.read(directory),
                  "shadow_outcomes": outcomes, "narrated_outcomes": narrated,
                  "outcome_divergence": outcome_divergence(narrated, outcomes),
                  "agent_calls": runtime.stats.model_calls, "gm_calls": gm.calls,
                  "model_errors": runtime.stats.model_errors, "gm_errors": gm.errors,
                  "gm_tokens": {"input": gm.input_tokens, "cached": gm.cached_tokens, "output": gm.output_tokens},
                  "acts": len(runtime.acts)}
        (directory / "narrated_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return report


def main(argv: list[str] | None = None) -> int:
    """Print the narrated report and fail scripted runs with no activity or an early stop."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage", "accident_daily_life", "sabotage_daily_life"),
                        default="sabotage")
    parser.add_argument("--scripted", action="store_true", help="$0 plumbing: scripted agents and a naive GM")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--gm-model", default=None, help="the GM's model (default: the agents' model)")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ticks", type=int, default=None,
                        help=f"end game time (default: {SPAN:,} ticks after the fixture's start)")
    parser.add_argument("--max-calls", type=int, default=None, help="budget of agent plus GM calls")
    parser.add_argument("--epoch-turns", type=int, default=8)
    parser.add_argument("--resync", action="store_true",
                        help="judge each GM ruling from the engine's true view of the acting agent (no compounding)")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    args.name = args.name or f"narrated_{'scripted' if args.scripted else args.model}_{args.variant}"
    report = NarratedVillage(args).run()
    print(json.dumps(report, indent=1))
    if args.scripted:
        reasons = []
        if report["acts"] == 0:
            reasons.append("acts == 0 (no narrated acts)")
        if report["gm_calls"] == 0:
            reasons.append("gm_calls == 0 (no GM calls)")
        if report["stopped"] is not None:
            reasons.append(f"stopped early: {report['stopped']}")
        for reason in reasons:
            print(f"FAIL VG5-scripted: {reason}")
        if reasons:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
