"""IT-18: compare the workshop policy's engine and narrated decision ticks per agent.

Procedure: run SimulationRuntime with THINKING_TICKS and NIGHT_REST, recording actual skill endings and heard
speech. Run NarratedRuntime with the same timing and an engine-faithful scripted GM: it rules on each act as the
engine did (accepted with the engine's actual duration, or rejected with the engine's reason), and delivers speech to
the listeners the engine reached, at the polls where they heard it, with engine-rendered views. Speech is delivered
through the runtime's interruption flag before each poll, since a `Resolution` has no delayed observation.
Pass: identical decision ticks, with instant-action thinking time and a speech interruption both exercised.
Negative control: removing narrated thinking time must break the tick comparison. No production code changes.
Revert check (2026-10-18): fails at d2f1d30^ (narrated runtime without thinking time or rest).
"""

from __future__ import annotations

import json

if __package__:
    from . import harness
    from .agent_scenario import REGION, WorkshopScenario
    from .test_e3 import RUN_TICKS
else:
    import harness
    from agent_scenario import REGION, WorkshopScenario
    from test_e3 import RUN_TICKS

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.narration import NarratedRuntime, Resolution  # noqa: E402
from orchestrator.render import ObservationRenderer  # noqa: E402
from orchestrator.runtime import NIGHT_REST, POLL, THINKING_TICKS, SimulationRuntime  # noqa: E402


def decision_ticks(log: DecisionLog) -> dict[str, list[int]]:
    """Group actual runtime decisions by agent without inferring or normalizing their timing."""
    ticks: dict[str, list[int]] = {}
    for decision in log.decisions():
        ticks.setdefault(decision.agent, []).append(decision.game_time)
    return ticks


def changed_tick(expected: dict[str, list[int]], actual: dict[str, list[int]]) -> bool:
    """Detect an actual mistimed decision, not merely a truncated run or missing trailing decisions."""
    return any(left != right for agent, ticks in actual.items()
               for left, right in zip(expected.get(agent, []), ticks))


def thinking_exercised(log: DecisionLog) -> bool:
    """Require an instant act followed by the first thinking-delayed poll, rather than only a configured delay."""
    previous = {}
    for decision in log.decisions():
        prior = previous.get(decision.agent)
        if (prior is not None and prior.outcome == "done" and not decision.interrupted
                and THINKING_TICKS <= decision.game_time - prior.game_time < THINKING_TICKS + POLL):
            return True
        previous[decision.agent] = decision
    return False


def skill_duration(decision, start: dict | None, endings: dict[str, dict]) -> int:
    """Read the actual end; a post-poll cancellation remains busy through that poll, unlike a tick-end finish."""
    if decision.outcome != "started":
        return 0
    if start is None or start["skillId"] not in endings:
        raise harness.HarnessError(f"Missing actual skill end for {decision.agent} at {decision.game_time}")
    ending = endings[start["skillId"]]
    duration = ending["gameTime"] - decision.game_time
    if duration < 0:
        raise harness.HarnessError("Skill ended before its recorded decision")
    return duration + (ending["outcome"] == "cancelled")


class EngineRecord:
    """Forward the real engine client, keeping only observations and successful skill-start identities."""

    def __init__(self, client):
        """Wrap an existing ready client; do not implement a substitute engine."""
        self.client = client
        self.now = 0
        self.observations: dict[tuple[str, int], dict] = {}
        self.starts: dict[tuple[str, int], dict] = {}

    def request(self, command: str, args: dict | None = None) -> dict:
        """Delegate every request unchanged and retain the real responses needed by the scripted GM."""
        result = self.client.request(command, args)
        self.now = result.get("gameTime", self.now)
        if command == "observe":
            self.observations[(args["name"], self.now)] = result
        if command in ("skill_start", "speak", "ask") and result.get("accepted"):
            self.starts[(args["name"], self.now)] = result
        return result


class EngineFaithfulGM:
    """A scripted GM whose rulings, durations and speech deliveries come exclusively from the recorded engine run."""

    def __init__(self, log: DecisionLog, record: EngineRecord, endings: dict[str, dict]):
        """Keep per-agent action sequences, measured durations and actual speech recipients."""
        self.decisions = {name: [decision for decision in log.decisions()
                                 if decision.agent == name and decision.outcome != "carried on"]
                          for name in decision_ticks(log)}
        self.indices = dict.fromkeys(self.decisions, 0)
        self.record = record
        self.endings = endings
        self.pending: list[tuple[int, str, int, dict]] = []
        self.now = 0
        self.calls = self.input_tokens = self.cached_tokens = self.output_tokens = 0
        self.renderer = ObservationRenderer()

    def view(self, agent: str) -> str:
        """Use the real engine view at this poll, even if narration has already drifted in cadence."""
        observation = self.record.observations.get((agent, self.now))
        if observation is None:
            raise harness.HarnessError(f"No recorded engine view for {agent} at {self.now}")
        return self.renderer.render(observation)

    def resolve(self, game_time: int, agent: str, call) -> Resolution:
        """Rule as the engine did: reject what it rejected, else accept with its measured duration and speech."""
        index = self.indices[agent]
        decisions = self.decisions[agent]
        if index >= len(decisions):
            raise harness.HarnessError(f"Narrated policy exhausted engine actions for {agent} at {game_time}")
        decision = decisions[index]
        expected = decision.tool_call
        if expected is None or call != expected:
            raise harness.HarnessError(f"Narrated policy differs for {agent} at {game_time}: {call} != {expected}")
        self.indices[agent] += 1
        self.calls += 1
        if decision.outcome.startswith(("not possible", "failed", "invalid")):
            return Resolution(False, decision.outcome, 0, [], self.view(agent), {})
        start = self.record.starts.get((agent, decision.game_time))
        duration = skill_duration(decision, start, self.endings)
        if call.name == "speak" and start is not None:
            for (recipient, tick), observation in self.record.observations.items():
                for order, utterance in enumerate(observation.get("heard", [])):
                    if recipient != agent and utterance.get("prov") == f"utt:{start.get('uttId')}":
                        if tick <= game_time:
                            raise harness.HarnessError("Narrated speech started after its recorded delivery poll")
                        self.pending.append((tick, recipient, order, utterance))
        return Resolution(True, "", duration, [], self.view(agent), {})

    def deliver(self, runtime: NarratedRuntime, now: int) -> None:
        """Deliver only speeches actually started, at their engine-observed polls, before the normal scheduler runs."""
        agents = {agent.name: agent for agent in runtime.agents}
        remaining = []
        for tick, recipient, order, utterance in sorted(self.pending, key=lambda item: item[:3]):
            if tick <= now:
                agents[recipient].memory.record_heard(utterance["gameTime"], utterance["speaker"], utterance["text"])
                runtime._interrupted[recipient] = True
            else:
                remaining.append((tick, recipient, order, utterance))
        self.pending = remaining


class CadenceExperiment:
    """Measure engine timing once and independently exercise the narrated runtime's decision scheduler."""

    def __init__(self, args):
        """Retain options and the shared workshop scenario without starting a server."""
        self.args = args
        self.scenario = WorkshopScenario()
        self.errors: dict[str, str] = {}

    def setup(self) -> None:
        """Create only an isolated base clone and reserve fresh narrated artifact directories."""
        self.directory = self.args.runs_dir / "it18_engine"
        for path in (self.directory, self.args.runs_dir / "it18_narrated", self.args.runs_dir / "it18_no_thinking"):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        harness.clone_apfs(self.args.runs_dir / "base", self.directory)

    def narrated(self, name: str, gap: int) -> NarratedRuntime:
        """Run the unchanged narrated scheduler with recorded engine views, not a prerecorded decision schedule."""
        gm = EngineFaithfulGM(self.engine.log, self.record, self.endings)
        runtime = NarratedRuntime(self.scenario.agents(), self.scenario.client(), gm, self.args.runs_dir / name,
                                  min_gap=gap, rest=NIGHT_REST)
        try:
            for now in range(self.start + (-self.start % POLL), self.until, POLL):
                gm.now = now
                gm.deliver(runtime, now)
                runtime.run(now, min(now + POLL, self.until))
        except harness.HarnessError as error:
            self.errors[name] = str(error)
            print(f"  {name}: {error}")
        return runtime

    def run(self) -> bool:
        """Record actual world execution, then compare normal and deliberately broken narrated cadence."""
        self.setup()
        with harness.Server(self.directory, **harness.server_options(self.args)) as server:
            self.start = harness.enable_lockstep(server.client, recording=False)
            self.scenario.setup(server.client)
            self.until = self.start + RUN_TICKS
            self.record = EngineRecord(server.client)
            self.engine = SimulationRuntime(self.record, self.scenario.agents(), self.scenario.client(),
                                            self.directory / "orchestrator", command_error=harness.CommandError,
                                            hash_region=REGION, min_gap=THINKING_TICKS, rest=NIGHT_REST)
            self.engine.run(self.until)
            server.client.request("step", {"n": 400})
        events_path = self.directory / "world/simharness/events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.endings = {event["skillId"]: event for event in events if event.get("type") == "skill_end"}
        self.normal = self.narrated("it18_narrated", THINKING_TICKS)
        self.negative = self.narrated("it18_no_thinking", 0)
        return self.check()

    def check(self) -> bool:
        """Require non-vacuous parity, thinking-time coverage, a real interruption and a sensitive negative control."""
        decisions = self.engine.log.decisions()
        instant = thinking_exercised(self.engine.log)
        interrupted = any(decision.interrupted for decision in decisions)
        engine_ticks, narrated_ticks = decision_ticks(self.engine.log), decision_ticks(self.normal.log)
        checks = {
            "engine made decisions for every workshop agent": set(engine_ticks) == set(self.scenario.BOTS),
            "instant actions exercised thinking time": instant,
            "speech interrupted an engine skill": interrupted,
            "identical decision ticks per agent": engine_ticks == narrated_ticks and "it18_narrated" not in self.errors,
            "negative control: no thinking time breaks cadence": (
                changed_tick(engine_ticks, decision_ticks(self.negative.log))),
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        report = {"engine": engine_ticks, "narrated": narrated_ticks,
                  "no_thinking": decision_ticks(self.negative.log), "errors": self.errors,
                  "engine_rejections": [{"agent": decision.agent, "tick": decision.game_time,
                                         "tool": decision.tool, "outcome": decision.outcome}
                                        for decision in decisions
                                        if decision.outcome.startswith(("failed:", "not possible:"))]}
        (self.args.runs_dir / "it18_cadence.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-18: workshop decision cadence")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse common gate arguments, and run the cadence experiment."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if CadenceExperiment(args).run() else 1
    except (OSError, ValueError, harness.HarnessError) as error:
        print(f"  BAD {error}\nFAIL IT-18: cadence experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
