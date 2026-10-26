"""N1 plumbing ($0): the narrated runtime and its engine shadow catch infeasible acts that a game master accepts.

The workshop agents (scripted policy, with two planted infeasible acts) run under a scripted "naive" game master: it
accepts every act and says every act is instant. Every GM-accepted act is mirrored into a shadow engine world. The
check passes if the shadow judged every accepted act, and flagged both planted acts with the right engine reason. The
live N1 swaps in a real LLM game master once model spend is approved.
"""

from __future__ import annotations

import json
import re

if __package__:
    from . import harness
    from .agent_scenario import WorkshopScenario
else:
    import harness
    from agent_scenario import WorkshopScenario

from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.narration import GameMaster, NarratedRuntime, ShadowEngine, narration_error  # noqa: E402
from orchestrator.render import ObservationRenderer  # noqa: E402

RUN_TICKS = 2000
EXPECTED = {("ana", "mine"): {"not possible: out_of_reach", "not possible: not_visible"},
            ("ben", "craft"): {"not possible: missing_items", "not possible: needs_crafting_table"}}


class NaiveGameMaster:
    """A scripted GM at the naive extreme: everything is possible and instant; speech reaches everyone."""

    AGENTS = ("ana", "ben", "cy")

    def resolve(self, system: str, user: str) -> ToolCall:
        """Accept the act described in the GM prompt, with zero duration and no change to the actor's view."""
        match = re.search(r"\), (\w+) attempts: (\w+) (\{.*\})\n(\w+)'s current view: (.*?)\n\nResolve this act\.$",
                          user, re.S)
        agent, tool, args, view = match.group(1), match.group(2), json.loads(match.group(3)), match.group(5)
        heard = [{"person": other, "text": args.get("text", "")} for other in self.AGENTS
                 if tool == "speak" and other != agent]
        return ToolCall("resolve", {"accepted": True, "reason": "", "duration_ticks": 0,
                                    "effects": [f"{agent} did {tool}."], "actor_view": view, "observations": heard})


class N1PlumbingExperiment:
    """Runs the narrated workshop with a shadow engine and checks the oracle on the planted acts."""

    def __init__(self, args):
        """Keep parsed CLI options."""
        self._args = args
        self._scenario = WorkshopScenario()

    def run(self) -> bool:
        """Run the narrated scenario and print the narration-error summary and checks. Returns true on PASS."""
        directory = self._args.runs_dir / "n1_plumbing"
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        harness.clone_apfs(self._args.runs_dir / "base", directory)
        with harness.Server(directory, **harness.server_options(self._args)) as server:
            start = harness.enable_lockstep(server.client, recording=False)
            self._scenario.setup(server.client)
            renderer = ObservationRenderer()
            views = {name: renderer.render(server.client.request("observe", {"name": name, "include_terrain": False}))
                     for name in self._scenario.BOTS}
            gm = GameMaster(ScriptedClient(NaiveGameMaster().resolve, model="naive-gm"),
                            self._scenario.world_description(), views)
            runtime = NarratedRuntime(self._scenario.agents(), self._scenario.client(planted_errors=True), gm,
                                      directory / "orchestrator",
                                      shadow=ShadowEngine(server.client, harness.CommandError))
            runtime.run(start, start + RUN_TICKS)
        summary = narration_error(runtime.acts)
        print(json.dumps(summary, indent=1))
        for act in (a for a in runtime.acts if a["tool"] != "wait"):
            print(f"  {act['gameTime']} {act['agent']}: {act['tool']} {json.dumps(act['args'])} -> GM accepted; "
                  f"engine {act.get('engine')}, finally {act.get('engine_final', '-')}")
        planted = {(act["agent"], act["tool"]): act["engine"] for act in runtime.acts
                   if (act["agent"], act["tool"]) in EXPECTED}
        checks = {
            "every act was accepted by the naive GM": all(act["gm"]["accepted"] for act in runtime.acts),
            "the shadow judged every accepted act": all("engine" in act for act in runtime.acts),
            "both planted acts were flagged with the right reason":
                all(planted.get(key) in reasons for key, reasons in EXPECTED.items()),
            "the planted acts count as infeasible": summary["infeasible"] >= len(EXPECTED),
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} N1 plumbing: "
              f"{summary['infeasible']}/{summary['accepted']} accepted acts "
              f"were infeasible in the engine; {summary['compressed']} were compressed in time")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the experiment, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    return 0 if N1PlumbingExperiment(harness.experiment_parser(__doc__).parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
