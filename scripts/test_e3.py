"""E3: an agent run replays exactly from its decision log (identical decisions and world hashes, zero model calls).

Run A drives the workshop scenario's three agents with the scripted policy and records every decision. Run B rebuilds
the same world and runs the agents with no model at all, reusing A's decisions only when the prompt digest matches.
The gate passes if B makes zero model calls, reproduces A's decision sequence, and A's world-hash trace every 20 ticks.
"""

from __future__ import annotations

import json

if __package__:
    from . import harness
    from .agent_scenario import REGION, WorkshopScenario
else:
    import harness
    from agent_scenario import REGION, WorkshopScenario

from orchestrator.decisions import DecisionLog  # noqa: E402  (agent_scenario puts the repo root on sys.path)
from orchestrator.runtime import SimulationRuntime  # noqa: E402

RUN_TICKS = 4000


def decision_trace(log: DecisionLog) -> list[tuple]:
    """Return the comparable part of each decision: time, agent, prompt digest, tool, arguments and outcome."""
    return [(d.game_time, d.agent, d.digest, d.tool, json.dumps(d.args, sort_keys=True), d.outcome)
            for d in log.decisions()]


class E3Experiment:
    """Runs the recorded run (A) and the model-free replay (B), then compares them."""

    def __init__(self, args):
        """Keep parsed CLI options."""
        self._args = args
        self._scenario = WorkshopScenario()

    def run_once(self, name: str, model, source: DecisionLog | None) -> SimulationRuntime:
        """Build a fresh world from the base run, set up the scenario, and run the agents for RUN_TICKS."""
        directory = self._args.runs_dir / name
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        harness.clone_apfs(self._args.runs_dir / "base", directory)
        with harness.Server(directory, **harness.server_options(self._args)) as server:
            start = harness.enable_lockstep(server.client, recording=False)
            self._scenario.setup(server.client)
            runtime = SimulationRuntime(server.client, self._scenario.agents(), model, directory / "orchestrator",
                                        command_error=harness.CommandError, source=source, hash_region=REGION)
            runtime.run(start + RUN_TICKS)
            for agent in runtime.agents:
                print(f"--- {agent.name} (run {name}) ---\n{agent.memory.render()}\n")
        return runtime

    def run(self) -> bool:
        """Run A then B and print the verdict. Returns true on PASS."""
        recorded = self.run_once("e3_a", self._scenario.client(), None)
        replayed = self.run_once("e3_b", None, recorded.log)
        trace_a, trace_b = decision_trace(recorded.log), decision_trace(replayed.log)
        tools = sorted({entry[3] for entry in trace_a if entry[3]})
        print(f"A: {recorded.stats}\nB: {replayed.stats}\nTools used in A: {tools}\nA's decisions:")
        for entry in trace_a:
            print(f"  {entry[0]} {entry[1]}: {entry[3]} {entry[4]} -> {entry[5]}")
        checks = {
            "A called the model": recorded.stats.model_calls > 0,
            "B made zero model calls": replayed.stats.model_calls == 0,
            "identical decision sequences": trace_a == trace_b,
            "identical world-hash traces": (
                recorded.hashes == replayed.hashes and len(recorded.hashes) == RUN_TICKS // 20),
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} E3: {len(trace_a)} decisions, {len(recorded.hashes)} hashes")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the experiment, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    return 0 if E3Experiment(harness.experiment_parser(__doc__).parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
