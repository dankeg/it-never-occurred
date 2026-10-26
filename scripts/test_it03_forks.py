"""IT-03: snapshot five live fork points, including a running skill, speech, and the night rest.

Record the scripted village through tick 16,400, saving three world/agent snapshots: a running non-wait skill,
tick 15,000, and tick 16,020 inside rest. Record the workshop separately for an idle agent's thinking gap and a
mid-speech snapshot (the village policy makes no instant acts and never speaks). Resume all five in fresh JVMs with
saved agent memory, restore_timing and file-log replay. Pass: every post-fork decision and every checkpoint world hash
matches its uninterrupted factual run, with zero model calls. Hashes are sampled every 200 ticks, including during
rest. Negative control: resume the thinking-gap snapshot without restore_timing, allowing the same scripted policy on
replay misses; its world hashes must differ.
"""

from __future__ import annotations

import argparse
import json

if __package__:
    from . import harness
    from .agent_scenario import REGION, WorkshopScenario
    from .setup_server import setup_server
    from .test_e3 import decision_trace
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import VillageSession
else:
    import harness
    from agent_scenario import REGION, WorkshopScenario
    from setup_server import setup_server
    from test_e3 import decision_trace
    from test_m5_village import VillageScriptedPolicy
    from village_session import VillageSession

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient  # noqa: E402
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS, SimulationRuntime  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402

HASH_EVERY = 200
VILLAGE_END = 16_400
WORKSHOP_SEARCH_TICKS = 4_000


def snapshot_labels(now: int, statuses: dict[str, dict], ready_at: dict[str, int]) -> set[str]:
    """Identify actual running activities and an unexpired idle thinking gap without observing/draining memory."""
    labels = set()
    for name, status in statuses.items():
        if status["state"] == "running":
            if status.get("skill") == "speak":
                labels.add("speech")
            elif status.get("skill") not in (None, "wait"):
                labels.add("skill")
        elif ready_at.get(name, 0) > now:
            labels.add("thinking")
    if NIGHT_REST[0] <= now % 24_000 < NIGHT_REST[1]:
        labels.add("night")
    return labels


def suffix_checks(recorded: SimulationRuntime, resumed: SimulationRuntime, tick: int, end: int,
                  hash_every: int = 20) -> dict[str, bool]:
    """Compare complete factual suffixes, excluding the already-recorded hash at the snapshot tick itself."""
    hashes = [entry for entry in recorded.hashes if entry["gameTime"] > tick]
    decisions = [entry for entry in decision_trace(recorded.log) if entry[0] >= tick]
    return {
        "zero model calls": resumed.stats.model_calls == 0,
        "identical decisions": bool(recorded.log.decisions()) and decision_trace(resumed.log) == decisions,
        "identical complete world hashes": (
            len(hashes) == end // hash_every - tick // hash_every and bool(hashes) and resumed.hashes == hashes),
    }


class ForksExperiment:
    """Record two factual worlds, snapshot five verified states, and replay their suffixes in fresh servers."""

    def __init__(self, args: argparse.Namespace):
        """Keep isolated paths, use only the assigned port pair, and load the village/workshop definitions."""
        self.args = argparse.Namespace(**vars(args))
        self.args.game_port = self.args.port + 1
        self.manifest = ScenarioManifest.load(harness.REPO_ROOT / "scenario/village_sabotage.json")
        self.workshop = WorkshopScenario()
        self.points: list[dict] = []
        self.checks: dict[str, bool] = {}

    def _runtime(self, session: VillageSession, agents: list, model, region: dict,
                 source: DecisionLog | None = None) -> SimulationRuntime:
        """Use the unchanged runtime with village thinking/rest rules and dense harness-provided hashes."""
        return SimulationRuntime(session.client, agents, model, session.directory / "orchestrator",
                                 command_error=harness.CommandError, source=source, hash_region=region,
                                 hash_every=HASH_EVERY, min_gap=THINKING_TICKS, rest=NIGHT_REST)

    def _snapshot(self, session: VillageSession, runtime: SimulationRuntime, label: str, scenario: str) -> None:
        """Save world and agent state exactly as Study does, keeping the selected status as an artifact."""
        tick = runtime.game_time()
        fork = setup_server(f"it03_fork_{label}", self.args.runs_dir)
        session.save_fork(fork, runtime)
        statuses = {agent.name: session.client.request("skill_status", {"name": agent.name})
                    for agent in runtime.agents}
        self.points.append({"label": label, "scenario": scenario, "tick": tick, "fork": fork.name,
                            "statuses": statuses, "readyAt": dict(runtime._ready_at)})
        (self.args.runs_dir / "it03_points.json").write_text(json.dumps(self.points, indent=1) + "\n",
                                                         encoding="utf-8")

    def _village(self) -> SimulationRuntime:
        """Select an early running skill, then record daytime and sleeping snapshots."""
        session = VillageSession(self.args.runs_dir / "it03_village", self.args.runs_dir / "village_sabotage",
                                 self.manifest, harness.server_options(self.args))
        session.clone()
        with session:
            runtime = self._runtime(session, self.manifest.agents(1),
                                    ScriptedClient(VillageScriptedPolicy(self.manifest).decide),
                                    self.manifest.data["layout"]["region"])
            now, selected = session.start, set()
            if now >= 8_000:
                raise harness.HarnessError(f"Village fixture starts too late for early forks: {now}")
            while now + 20 < 8_000 and not selected:
                now += 20
                runtime.run(now)
                statuses = {agent.name: session.client.request("skill_status", {"name": agent.name})
                            for agent in runtime.agents}
                if "skill" in snapshot_labels(now, statuses, runtime._ready_at):
                    self._snapshot(session, runtime, "skill", "village")
                    selected.add("skill")
            if selected != {"skill"}:
                raise harness.HarnessError("The village never reached a running non-wait skill before tick 8,000")
            runtime.run(15_000)
            self._snapshot(session, runtime, "daytime", "village")
            runtime.run(16_020)
            self._snapshot(session, runtime, "night", "village")
            runtime.run(VILLAGE_END)
        return runtime

    def _speech(self) -> SimulationRuntime:
        """Use the real workshop policy to find a thinking gap and a running utterance, then finish its factual run
        uninterrupted."""
        session = VillageSession(self.args.runs_dir / "it03_workshop", self.args.runs_dir / "base", self.manifest,
                                 harness.server_options(self.args), fresh=False)
        session.clone()
        with session:
            session.client.request("recording", {"enabled": False})
            self.workshop.setup(session.client)
            runtime = self._runtime(session, self.workshop.agents(), self.workshop.client(), REGION)
            now, end, selected = session.start, session.start + WORKSHOP_SEARCH_TICKS, set()
            while now < end and selected != {"thinking", "speech"}:
                now += 20
                runtime.run(now)
                statuses = {agent.name: session.client.request("skill_status", {"name": agent.name})
                            for agent in runtime.agents}
                available = snapshot_labels(now, statuses, runtime._ready_at) - selected
                for label in ("speech", "thinking"):
                    if label in available:
                        self._snapshot(session, runtime, label, "workshop")
                        selected.add(label)
                        break
            if selected != {"thinking", "speech"}:
                raise harness.HarnessError(f"Workshop never reached both live fork conditions: {selected}")
            end = now + 800
            self.workshop_end = end
            runtime.run(end)
        return runtime

    def _resume(self, point: dict, *, restore: bool = True) -> SimulationRuntime:
        """Load a saved fork into a new JVM and replay, omitting timing only for the scripted negative control."""
        village = point["scenario"] == "village"
        name = f"it03_{'resume' if restore else 'no_timing'}_{point['label']}"
        session = VillageSession(self.args.runs_dir / name, self.args.runs_dir / point["fork"], self.manifest,
                                 harness.server_options(self.args), fresh=False)
        session.clone()
        state = session.directory / "orchestrator/fork_state.json"
        tick, agents = SimulationRuntime.load_agents(state)
        factual = self.village if village else self.speech
        end = VILLAGE_END if village else self.workshop_end
        model = (None if restore else ScriptedClient(VillageScriptedPolicy(self.manifest).decide) if village
                 else self.workshop.client())
        with session:
            if session.start != tick or tick != point["tick"]:
                raise harness.HarnessError(f"Restored at {session.start}, expected saved tick {point['tick']}")
            runtime = self._runtime(session, agents, model,
                                    self.manifest.data["layout"]["region"] if village else REGION,
                                    DecisionLog(factual.log.path))
            if restore:
                runtime.restore_timing(state)
            runtime.run(end if restore else min(end, tick + 800))
        return runtime

    def run(self) -> bool:
        """Check exact replay at every saved point and a real world-hash divergence when timing is omitted."""
        self.village = self._village()
        self.speech = self._speech()
        labels = {point["label"] for point in self.points}
        self.checks["five distinct fork conditions were saved"] = (
            len(self.points) == 5 and labels == {"skill", "thinking", "daytime", "night", "speech"}
            and len({point["tick"] for point in self.points}) == 5)
        for point in self.points:
            conditions = snapshot_labels(point["tick"], point["statuses"], point["readyAt"])
            if point["label"] != "daytime":
                self.checks[f"{point['label']}: verified live fork condition"] = point["label"] in conditions
            factual = self.village if point["scenario"] == "village" else self.speech
            end = VILLAGE_END if point["scenario"] == "village" else self.workshop_end
            resumed = self._resume(point)
            self.checks.update({f"{point['label']} at {point['tick']}: {label}": passed
                                for label, passed in suffix_checks(
                                    factual, resumed, point["tick"], end, HASH_EVERY).items()})
        point = next(point for point in self.points if point["label"] == "thinking")
        negative = self._resume(point, restore=False)
        factual = self.village if point["scenario"] == "village" else self.speech
        expected = [entry for entry in factual.hashes if point["tick"] < entry["gameTime"] <= point["tick"] + 800]
        self.checks["negative control: skipping restore_timing changes world hashes"] = (
            bool(expected) and len(negative.hashes) == len(expected) and negative.hashes != expected)
        for label, passed in self.checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(self.checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-03: five exact forks and the missing-timing negative control")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse common gate arguments, and return success only for the complete experiment."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if ForksExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-03: fork experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
