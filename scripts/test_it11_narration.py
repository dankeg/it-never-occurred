"""IT-11: exercise N1 narration against the real workshop engine, then complete the scripted narrated village.

The scripted GM obeys the ask rule in its system prompt. Deliberately planted faults compress Ana's speech, accept
her next act while she is speaking, accept Ben's distant mine and material-free placement, and reject Cy's feasible
wait. The real shadow supplies all verdicts. Check each fault's first/downstream classification with narration_error,
the false rejection, duration ratio and GM tokens per call. Negative control: a GameMaster built here without
ENGINE_RULES accepts ask aimed at Ben; its real-engine rejection must be a first error. The rules-on ask must reject.
Finally invoke the unchanged run_narrated_village --scripted entry point and check its saved completion report.
All clones and artifacts stay under --runs-dir; no model is contacted. Estimated time: 1 minute; timeout: 3 minutes.
Runtime cadence parity is IT-18, not part of this gate.
Revert check (2026-10-18): fails at 648a83d^ (the GM without the engine rules).
"""

from __future__ import annotations

import argparse
import json
import math
import re

if __package__:
    from . import harness, run_narrated_village
    from .agent_scenario import WorkshopScenario
else:
    import harness
    import run_narrated_village
    from agent_scenario import WorkshopScenario

from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.narration import (  # noqa: E402
    ENGINE_RULES, GameMaster, NarratedRuntime, ShadowEngine, narration_error,
)

PLANS = {
    "ana": [ToolCall("speak", {"text": " ".join(["word"] * 60)}),
            ToolCall("move_to", {"x": -23, "y": -60, "z": -24})],
    "ben": [ToolCall("mine", {"x": -26, "y": -60, "z": -20}),
            ToolCall("place", {"x": 17, "y": -60, "z": 16, "item": "minecraft:diamond_block"})],
    "cy": [ToolCall("wait", {"ticks": 80}), ToolCall("ask", {"npc": "ben"}),
           ToolCall("wait", {"ticks": 200})],
}


def planned_action(system: str, user: str) -> ToolCall | None:
    """Choose the next planted act from the actor's own numbered notes, then leave the actor idle."""
    name = re.search(r"You are (\w+) at", user).group(1)
    notes = re.search(r"Your notes:\n(.*?)(?:\n\n|$)", user, re.S).group(1)
    steps = re.findall(r"\] step (\d+)", notes)
    index = int(steps[-1]) if steps else 0
    if index >= len(PLANS[name]):
        return None
    call = PLANS[name][index]
    return ToolCall(call.name, {**call.args, "note": f"step {index + 1}"})


def scripted_ruling(system: str, user: str) -> ToolCall:
    """Apply the system's explicit participant rule for ask and inject the other documented GM faults."""
    match = re.search(r"\), (\w+) attempts: (\w+) (\{.*\})\n\w+'s current view: (.*?)\n\nResolve this act\.$",
                      user, re.S)
    agent, tool, args, view = match.group(1), match.group(2), json.loads(match.group(3)), match.group(4)
    rejected_ask = tool == "ask" and "not villagers and cannot be asked" in system
    false_rejection = agent == "cy" and tool == "wait" and args["ticks"] == 200
    accepted = not (rejected_ask or false_rejection)
    duration = args["ticks"] if accepted and tool == "wait" else 0
    return ToolCall("resolve", {"accepted": accepted, "reason": "target_not_found" if rejected_ask else
                                "planted false rejection" if false_rejection else "", "duration_ticks": duration,
                                "effects": [], "actor_view": view, "observations": []})


def classification(acts: list[dict], index: int, key: str) -> int:
    """Measure one real act's contribution using the production scorer, preserving its preceding divergences."""
    return narration_error(acts[:index + 1])[key] - narration_error(acts[:index])[key]


def planted_checks(acts: list[dict], control: list[dict]) -> dict[str, bool]:
    """Require exact engine reasons and independent first/downstream scoring for every planted error class."""
    checks = {}
    cases = [("out of reach", "ben", "mine", "out_of_reach", "infeasible_first"),
             ("missing items", "ben", "place", "missing_items", "infeasible_downstream"),
             ("act during own speech", "ana", "move_to", "busy", "infeasible_downstream")]
    for label, agent, tool, reason, key in cases:
        matches = [(index, act) for index, act in enumerate(acts) if act["agent"] == agent and act["tool"] == tool]
        checks[f"{label}: exact reason and {key}"] = len(matches) == 1 and (
            matches[0][1].get("engine") == f"not possible: {reason}"
            and classification(acts, matches[0][0], key) == 1)
    asks = [act for act in acts if act["tool"] == "ask"]
    checks["rules-on GM rejects ask aimed at an agent"] = len(asks) == 1 and (
        not asks[0]["gm"]["accepted"] and asks[0].get("engine_check", {}).get("reason") == "target_not_found")
    false = [(index, act) for index, act in enumerate(acts)
             if act["agent"] == "cy" and act["tool"] == "wait" and act["args"]["ticks"] == 200]
    checks["feasible rejection counted before divergence"] = len(false) == 1 and (
        not false[0][1]["gm"]["accepted"] and (false[0][1].get("engine_check") or {}).get("feasible") is True
        and classification(acts, false[0][0], "false_rejections_first") == 1)
    summary = narration_error(acts)
    ratio = summary["duration_ratio_median"]
    checks["speech compression precedes the downstream busy error"] = summary["compressed_by_tool"].get("speak") == 1
    checks["duration ratio is measured and reported"] = ratio is not None and math.isfinite(ratio) and ratio > 0
    checks["negative control: rules removed accepts ask; engine scores a first error"] = len(control) == 1 and (
        control[0]["tool"] == "ask" and control[0]["gm"]["accepted"]
        and control[0].get("engine") == "not possible: target_not_found"
        and narration_error(control)["infeasible_first"] == 1)
    return checks


class NarrationExperiment:
    """Measure planted narration errors on the real engine and run the existing scripted village workflow."""

    def __init__(self, args):
        """Keep harness options and bind both sockets to the gate's eight-port allocation."""
        self.args = args
        if not 1 <= args.port <= 65528:
            raise ValueError("--port must leave room for its eight-port allocation (1..65528)")
        self.options = harness.server_options(args)
        self.options["game_port"] = args.game_port if args.game_port is not None else args.port + 1
        if self.options["game_port"] == args.port or self.options["game_port"] not in range(args.port, args.port + 8):
            raise ValueError("--game-port must differ from --port and lie within --port + 0..7")

    def run(self) -> bool:
        """Run workshop faults and the rules-removed control before invoking the scripted narrated village."""
        directory = self.args.runs_dir / "it11_workshop"
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        harness.clone_apfs(self.args.runs_dir / "base", directory)
        scenario = WorkshopScenario()
        with harness.Server(directory, **self.options) as server:
            harness.enable_lockstep(server.client, recording=False)
            scenario.setup(server.client)
            shadow = ShadowEngine(server.client, harness.CommandError)
            views = {name: shadow.view(name) for name in scenario.BOTS}
            gm = GameMaster(ScriptedClient(scripted_ruling, model="rule-following-with-planted-faults"),
                            scenario.world_description(), views)
            runtime = NarratedRuntime(scenario.agents(), ScriptedClient(planned_action), gm,
                                      directory / "orchestrator", shadow=shadow)
            now = server.client.request("ping")["gameTime"]
            start = now + (-now % 40)
            runtime.run(start, start + 160)
            control_gm = GameMaster(ScriptedClient(scripted_ruling, model="rules-removed"),
                                    scenario.world_description(), views)
            control_gm._system = control_gm._system.replace("\n\n" + ENGINE_RULES, "")

            def ask_agent(system: str, user: str) -> ToolCall:
                """Attempt the same invalid participant under the rules-removed GameMaster."""
                return ToolCall("ask", {"npc": "ben"})

            control = NarratedRuntime([agent for agent in scenario.agents() if agent.name == "cy"],
                                      ScriptedClient(ask_agent), control_gm, directory / "negative_orchestrator",
                                      shadow=shadow)
            control.run(start + 160, start + 200)
            server.client.request("save_snapshot")
        checks = planted_checks(runtime.acts, control.acts)
        tokens = {"calls": gm.calls, "input_tokens_per_call": gm.input_tokens / gm.calls if gm.calls else 0,
                  "output_tokens_per_call": gm.output_tokens / gm.calls if gm.calls else 0,
                  "note": "Scripted clients report zero tokens; no live model measurement."}
        village_args = ["--scripted", "--runs-dir", str(self.args.runs_dir), "--port", str(self.args.port),
                        "--game-port", str(self.options["game_port"]), "--name", "it11_narrated_village",
                        "--base-seed", str(self.args.base_seed), "--startup-timeout", str(self.args.startup_timeout),
                        "--request-timeout", str(self.args.request_timeout)]
        if self.args.heap is not None:
            village_args += ["--heap", self.args.heap]
        status = run_narrated_village.main(village_args)
        village = json.loads((self.args.runs_dir / "it11_narrated_village/narrated_report.json").read_text(
            encoding="utf-8"))
        checks["scripted narrated village completes"] = status == 0 and village["stopped"] is None and (
            village["end"] == village["start"] + run_narrated_village.SPAN and village["acts"] > 0
            and village["gm_calls"] > 0)
        report = {"narration_error": narration_error(runtime.acts), "negative": narration_error(control.acts),
                  "gm_usage": tokens, "village": village, "checks": checks}
        (directory / "it11_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=1))
        for label, passed in checks.items():
            print(f"  {'ok' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-11: engine-scored narration and scripted village completion")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse shared harness options and run the narration gate."""
    harness.stop_on_sigterm()
    args: argparse.Namespace = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if NarrationExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-11: narration experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
