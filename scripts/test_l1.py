"""L1 (plumbing, $0): gated counterfactual branches reuse factual decisions wherever an agent's prompt is unchanged.

The factual run F drives the workshop scenario with the scripted policy and forks at FORK_TICKS: it saves the world
snapshot and the agents' state, and both are cloned into two branches. Each branch restores in a fresh JVM and runs to
the end with F's decision log as its source; it calls the model only when an agent's prompt digest differs from F's.

- **Null branch** (no intervention): must make zero model calls and reproduce F's decisions and world hashes exactly.
- **Sign branch** (the market sign is rewritten at the fork): Ben reads a different sign, says something different, and
  Cy replies differently. Only agents in that cone should call the model. Ana, who never perceives any of it, must make
  zero calls and repeat her factual decisions exactly.

The live L1 (10 stochastic LLM branches, effect with CIs) reuses this machinery once model spend is approved.
"""

from __future__ import annotations

from collections import Counter

if __package__:
    from . import harness
    from .agent_scenario import REGION, WorkshopScenario
    from .setup_server import setup_server
    from .test_e3 import decision_trace
else:
    import harness
    from agent_scenario import REGION, WorkshopScenario
    from setup_server import setup_server
    from test_e3 import decision_trace

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.runtime import SimulationRuntime  # noqa: E402

RUN_TICKS = 4000
FORK_TICKS = 60  # fork at tick 160: after everyone's first decision, before Ben reaches the sign
CHANGED_SIGN = ("Market closed", "today", "", "")


class L1Experiment:
    """Runs the factual run with a fork point, then the null and sign branches, and compares them."""

    def __init__(self, args):
        """Keep parsed CLI options."""
        self._args = args
        self._scenario = WorkshopScenario()

    def _fresh(self, name: str):
        """Return a new run directory path, refusing to overwrite an existing one."""
        directory = self._args.runs_dir / name
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        return directory

    def run_factual(self, branches: list[str]) -> tuple[SimulationRuntime, int]:
        """Run F, forking its world and agent state into each branch directory. Returns F's runtime and fork time."""
        directory = self._fresh("l1_factual")
        branch_dirs = [setup_server(self._fresh(name).name, self._args.runs_dir) for name in branches]
        harness.clone_apfs(self._args.runs_dir / "base", directory)
        with harness.Server(directory, **harness.server_options(self._args)) as server:
            start = harness.enable_lockstep(server.client, recording=False)
            self._scenario.setup(server.client)
            runtime = SimulationRuntime(server.client, self._scenario.agents(), self._scenario.client(),
                                        directory / "orchestrator", command_error=harness.CommandError,
                                        hash_region=REGION)
            fork = start + FORK_TICKS
            runtime.run(fork)
            server.client.request("save_snapshot")
            for branch in branch_dirs:
                harness.clone_apfs(directory / "world", branch / "world")
                (branch / "orchestrator").mkdir()
                runtime.save_state(branch / "orchestrator" / "fork_state.json")
            runtime.run(start + RUN_TICKS)
        self._end = start + RUN_TICKS
        return runtime, fork

    def run_branch(self, name: str, fork: int, factual: DecisionLog, sign: tuple[str, ...] | None) -> SimulationRuntime:
        """Restore a branch at the fork, apply its intervention, and run it with F's decisions as the source."""
        directory = self._args.runs_dir / name
        restored_time, agents = SimulationRuntime.load_agents(directory / "orchestrator" / "fork_state.json")
        with harness.Server(directory, **harness.server_options(self._args)) as server:
            start = harness.enable_lockstep(server.client)
            if start != fork or restored_time != fork:
                raise harness.HarnessError(f"Branch {name} restored at {start}/{restored_time}, expected {fork}")
            server.client.request("restore_bots")
            if sign is not None:
                self._scenario.change_sign(server.client, sign)
            runtime = SimulationRuntime(server.client, agents, self._scenario.client(), directory / "orchestrator",
                                        command_error=harness.CommandError, source=factual, hash_region=REGION)
            runtime.run(self._end)
            for agent in runtime.agents:
                print(f"--- {agent.name} (branch {name}) ---\n{agent.memory.render()}\n")
        return runtime

    @staticmethod
    def calls_by_agent(runtime: SimulationRuntime) -> dict[str, int]:
        """Count model calls per agent in a run's decision log."""
        counts = Counter(d.agent for d in runtime.log.decisions() if d.source == "model")
        return {agent.name: counts.get(agent.name, 0) for agent in runtime.agents}

    def run(self) -> bool:
        """Run F and both branches, print per-agent call counts and the checks, and return true on PASS."""
        factual, fork = self.run_factual(["l1_null", "l1_sign"])
        null = self.run_branch("l1_null", fork, factual.log, None)
        signed = self.run_branch("l1_sign", fork, factual.log, CHANGED_SIGN)
        suffix = [entry for entry in decision_trace(factual.log) if entry[0] >= fork]
        factual_hashes = [record for record in factual.hashes if record["gameTime"] > fork]
        by_agent = lambda trace, name: [entry for entry in trace if entry[1] == name]  # noqa: E731
        sign_calls = self.calls_by_agent(signed)
        print(f"F: {factual.stats}\nnull: {null.stats} calls={self.calls_by_agent(null)}\n"
              f"sign: {signed.stats} calls={sign_calls}")
        checks = {
            "null branch made zero model calls": null.stats.model_calls == 0,
            "null branch repeats F's decisions": decision_trace(null.log) == suffix,
            "null branch repeats F's world hashes": null.hashes == factual_hashes,
            "sign branch: ben and cy called the model": sign_calls["ben"] > 0 and sign_calls["cy"] > 0,
            "sign branch: ana made zero model calls": sign_calls["ana"] == 0,
            "sign branch: ana repeats F's decisions": (
                by_agent(decision_trace(signed.log), "ana") == by_agent(suffix, "ana")),
            "sign branch world diverged": signed.hashes != factual_hashes,
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} L1 plumbing: sign-branch calls {sign_calls} of "
              f"{signed.stats.decisions} decisions")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the experiment, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    return 0 if L1Experiment(harness.experiment_parser(__doc__).parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
