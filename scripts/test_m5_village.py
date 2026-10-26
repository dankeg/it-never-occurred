"""M5 end to end ($0): the village crisis with all six role agents, a scripted policy, and automatic outcome metrics.

The policy stands in for an LLM and is driven by the prompt (`step N` notes, like the workshop scenario):
- the **builder** walks to the forest, mines four logs with its own hands, collects them, crafts planks, and rebuilds
  the five bridge sections in order from the west bank;
- the **mayor** walks from the town hall to the east-bank fairgrounds (a long route via the ford while the bridge is
  out) and places the lantern beside the bell;
- everyone else waits.

The run goes from the built village to the end of day 1. Outcomes are tracked every 1,200 ticks with read-only
predicates: when the bridge was complete, when the fair opened, and welfare on day 0 against day 1. The scripted
plans prove the engine can carry a full crisis end to end; the live run replaces the policy with LLM agents.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__:
    from . import harness
    from .provenance import FixtureMetadata
    from .test_e3 import decision_trace
    from .village_session import CHECK_EVERY, Checkpoints, VillageSession
    from .village_metrics import VillageMetrics
else:
    import harness
    from provenance import FixtureMetadata
    from test_e3 import decision_trace
    from village_session import CHECK_EVERY, Checkpoints, VillageSession
    from village_metrics import VillageMetrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.runtime import BudgetExceeded, SimulationRuntime  # noqa: E402
from orchestrator.scenario import KeywordJudge, LLMJudge, OutcomeTracker, PredicateEvaluator, ScenarioManifest  # noqa: E402

END_TICK = 48_000
TREES = ((-109, 8), (-109, 16))


class VillageScriptedPolicy:
    """Prompt-driven plans for the builder and the mayor; the other roles wait. The mayor also pays the builder, which
    shows the builder's success predicate (bridge built and paid) can be reached."""

    def __init__(self, manifest: ScenarioManifest):
        """Read the bridge sections and fair location from the manifest."""
        self._sections = [(name, spec) for name, spec in sorted(manifest.data["schematics"].items())
                          if name.startswith("bridge_span_")]
        self._lantern = manifest.data["layout"]["fair"]["lantern"]

    def decide(self, system: str, user: str) -> ToolCall:
        """Return the next planned action for the agent named in the prompt."""
        name = re.search(r"You are (\w+) at", user).group(1)
        notes = re.search(r"Your notes:\n(.*?)(?:\n\n|$)", user, re.S).group(1)
        steps = re.findall(r"\] step (\d+)", notes)
        step = int(steps[-1]) if steps else 0
        plan = self.builder() if name == "builder" else self.mayor() if name == "mayor" else []
        if step >= len(plan):
            return ToolCall("wait", {"ticks": 1200})
        call = plan[step]
        return ToolCall(call.name, {**call.args, "note": f"step {step + 1}"})

    def builder(self) -> list[ToolCall]:
        """Harvest logs, craft planks, then build each bridge section from its standing point."""
        plan = []
        for x, z in TREES:
            plan += [ToolCall("move_to", {"x": x + 2, "y": -60, "z": z}),
                     ToolCall("mine", {"x": x, "y": -60, "z": z}), ToolCall("mine", {"x": x, "y": -59, "z": z}),
                     ToolCall("collect", {})]
        plan.append(ToolCall("craft", {"item": "minecraft:oak_planks", "count": 16}))
        for name, spec in self._sections:
            stand = [int(value // 1) for value in spec["stand"]]
            plan += [ToolCall("move_to", {"x": stand[0], "y": stand[1], "z": stand[2]}),
                     ToolCall("build_section", {"schematic": name, "origin": spec["origin"]})]
        return plan

    def mayor(self) -> list[ToolCall]:
        """Place the lantern beside the bell; later, once the bridge is long done, pay the builder its 24 emeralds
        where it finished (beside the last section's standing point)."""
        x, y, z = self._lantern
        stand = [int(value // 1) for value in self._sections[-1][1]["stand"]]
        return [ToolCall("move_to", {"x": x, "y": y, "z": z + 2}),
                ToolCall("place", {"x": x, "y": y, "z": z, "item": "minecraft:lantern"}),
                ToolCall("wait", {"ticks": 1200}), ToolCall("wait", {"ticks": 1200}),
                ToolCall("move_to", {"x": stand[0] + 1, "y": stand[1], "z": stand[2] + 1}),
                ToolCall("give", {"target": "builder", "item": "minecraft:emerald", "count": 24})]


VARIANTS = ("accident", "sabotage", "accident_daily_life", "sabotage_daily_life", "sabotage_coverup",
            "sabotage_coverup2")

class M5VillageExperiment:
    """Runs the scripted village crisis from the built world to the end of day 1 and reports outcomes.

    With `--replay`, it also replays the run from its decision log with no model (gate V-G2, exactness at scale):
    the replay must make zero model calls and reproduce every decision and every checkpoint world hash.
    """

    def __init__(self, args):
        """Keep parsed CLI options and load the scenario manifest."""
        self._args = args
        self._manifest = ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario"
                                               / f"village_{args.variant}.json")

    def run_once(self, name: str, model, source: DecisionLog | None, end_tick: int = END_TICK,
                 max_model_calls: int | None = None, model_concurrency: int | None = None,
                 min_gap: int = 0, rest: tuple[int, int] | None = None,
                 epoch_turns: int = 1, judge_client=None, checkpoints: Checkpoints | None = None,
                 resume_from: Path | None = None) -> tuple[SimulationRuntime, OutcomeTracker, dict]:
        """Run the village from a fresh clone of the built world to `end_tick`, tracking outcomes every CHECK_EVERY.

        `epoch_turns` > 1 gives agents append-only conversation prompts (the scripted policy reads full prompts, so 1).
        Reports are judged by keywords ($0), or by `judge_client` (an LLM client) when given.
        `resume_from` restores a checkpoint's agents, timing and outcome tracker instead of starting fresh.

        Returns the runtime, the outcome tracker and the run report. A run stopped by the model-call budget still
        reports what happened up to that point (`stopped`).
        """
        args, manifest = self._args, self._manifest
        directory = args.runs_dir / name
        session = VillageSession(directory, resume_from if resume_from is not None
                                 else args.runs_dir / f"village_{args.variant}", manifest,
                                 harness.server_options(args), fresh=resume_from is None)
        session.clone()
        agents = (manifest.agents(epoch_turns) if resume_from is None
                  else SimulationRuntime.load_agents(directory / "orchestrator" / "fork_state.json")[1])
        predicates = manifest.data["success_predicates"]
        tracker = OutcomeTracker({"bridge_complete": predicates["builder"]["all"][0],
                                  "fair_ready": predicates["fair_ready"], "fair_open": predicates["fair_open"]})
        prefix_hashes = []
        if resume_from is not None:
            tracker.first_true.update(json.loads((directory / "orchestrator" / "first_true.json")
                                                 .read_text(encoding="utf-8")))
            prefix_hashes = Checkpoints.carry_prefix(resume_from, directory)
        with session:
            client, now = session.client, session.start
            evaluator = PredicateEvaluator(client, directory / "world/simharness/events.jsonl")
            runtime = SimulationRuntime(client, agents, model, directory / "orchestrator",
                                        command_error=harness.CommandError, source=source,
                                        hash_region=manifest.data["layout"]["region"], hash_every=CHECK_EVERY,
                                        max_model_calls=max_model_calls, model_concurrency=model_concurrency,
                                        min_gap=min_gap, rest=rest)
            if resume_from is not None:
                runtime.restore_timing(directory / "orchestrator" / "fork_state.json")
                runtime.hashes.extend(prefix_hashes)
            stopped = None
            try:
                now = session.run_checked(runtime, tracker, now, end_tick, checkpoints)
            except BudgetExceeded as error:
                stopped = str(error)
                now = harness.game_time(client.request("ping"))
            welfare = {day: round(evaluator.welfare(day), 3) for day in range(now // 24_000)}
            metrics = client.request("npc_metrics", {})["days"]
            client.request("save_snapshot")  # flushes the event log for transfer and incident predicates
            suspects = [role["name"] for role in manifest.roles]
            judge = (LLMJudge(judge_client, suspects, manifest.data["evidence"]["expected_chain"])
                     if judge_client is not None else KeywordJudge(suspects))
            judged = PredicateEvaluator(client, directory / "world/simharness/events.jsonl", manifest, judge, tracker)
            outcomes = {role["name"]: judged.evaluate(predicates[role["name"]], role["name"])
                        for role in manifest.roles}
            paperwork = judged.evaluate(predicates["builder_paperwork"]) if "builder_paperwork" in predicates else None
        report = {"end": now, "stopped": stopped, "stats": runtime.stats.__dict__, "first_true": tracker.first_true,
                  "fixture": FixtureMetadata.read(directory),
                  "welfare": welfare, "outcomes": outcomes, "builder_paperwork": paperwork,
                  "npc_metrics": metrics, "hashes": len(runtime.hashes)}
        if not (directory / "scenario.json").exists():
            (directory / "scenario.json").write_text(json.dumps(manifest.data, indent=1) + "\n", encoding="utf-8")
        (directory / "m5_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        VillageMetrics(directory).compute()
        return runtime, tracker, report

    def run(self) -> bool:
        """Run (and optionally replay), print the report, and return true if all checks hold."""
        args = self._args
        policy = VillageScriptedPolicy(self._manifest)
        runtime, tracker, report = self.run_once(f"m5_village_{args.variant}", ScriptedClient(policy.decide), None)
        welfare = report["welfare"]
        for decision in runtime.log.decisions():
            if decision.agent in {"builder", "mayor"} and decision.tool != "wait":
                print("  ", decision.game_time, decision.agent, decision.tool, decision.outcome)
        for agent in runtime.agents:
            if agent.name in {"builder", "mayor"}:
                print(f"--- {agent.name} ---\n{agent.memory.render()}\n")
        print(json.dumps({"stats": runtime.stats.__dict__, "first_true": tracker.first_true, "welfare": welfare,
                          "fixture": report["fixture"],
                          "outcomes": report["outcomes"], "hashes": len(runtime.hashes)}, indent=1))
        start_hashes = END_TICK // CHECK_EVERY - min(d.game_time for d in runtime.log.decisions()) // CHECK_EVERY
        checks = {
            "the bridge was rebuilt on day 0": tracker.by_day("bridge_complete", 0),
            "the fair was ready (lantern beside the bell) on day 0": tracker.by_day("fair_ready", 0),
            "the builder succeeded (bridge built and paid)": report["outcomes"]["builder"] is True,
            "villager welfare on day 1 meets the mayor's threshold (0.9)": welfare[1] >= 0.9,
            "world hashes recorded every 1,200 ticks": len(runtime.hashes) == start_hashes,
        }
        if args.replay:
            replay, _, _ = self.run_once(f"vg2_replay_{args.variant}", None, runtime.log)
            print(f"replay: {replay.stats}")
            checks["V-G2: the replay made zero model calls"] = replay.stats.model_calls == 0
            checks["V-G2: identical decisions"] = decision_trace(replay.log) == decision_trace(runtime.log)
            checks["V-G2: identical world hashes at every checkpoint"] = replay.hashes == runtime.hashes
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} M5 village end to end ({args.variant})"
              + (" + V-G2 replay" if args.replay else ""))
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the experiment, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=VARIANTS,
                        default="sabotage")
    parser.add_argument("--replay", action="store_true", help="also replay from the decision log (gate V-G2)")
    return 0 if M5VillageExperiment(parser.parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
