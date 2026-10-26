"""Counterfactual forks of one logged decision (DELEGATE_SPEC §8): control, swap, placebo, swap-hold, resample, brief
and mute arms.

A decision point is an agent `x` and a tick `t` at which the factual run's log holds a decision; `â` is the act swapped
in (it must not match the logged choice). Run directories sit in `--runs-dir` under the prefix
`fork_<run>_<agent>_<t>`:

- `_control`: the factual run replayed from its starting world (`runs/village_<variant>`, set up as the factual run
  was) with no model. At `t`, before the poll at `t`, the world and every agent are saved into `_fork` (as
  `run_study.py` saves its fork), with the evidence exposure and outcome-tracker state so far. The replay then
  continues to `t + H`: the control arm (the logged factual continuation).
- `_swap`, `_placebo`, `_swap_hold`: each an APFS clone of `_fork`, run like `run_study.py`'s branches with the factual
  log as source and the arm's `DecisionOverride` (swap: `x` takes `â` at `t`; placebo: `x` is sampled afresh at `t`;
  swap-hold: also every other agent takes its logged action at its logged polls in `[t, t + H]`). Later decisions
  reuse the factual log wherever an agent's prompt is unchanged, and call the model elsewhere (`--model scripted` is
  the $0 village policy).
- `_resample`: the agents given by `--resample AGENT` (default: `x`) are asked afresh at every decision from `t`, never
  reusing the factual log; every other agent reuses it where its prompt is unchanged. This is the control for the
  next two arms, which change only one thing on top of it.
- `_brief`: as `_resample`, and the private briefs supplied by `--brief AGENT=TEXT` are replaced at `t`.
- `_mute`: as `_resample`, and from `t` each `--mute LISTENER=SPEAKER` listener no longer perceives that speaker's
  speech (an intervention on perception; the world, and every other listener, still hear it).
Explicit `--samples K` pairs sampled arms by seed (`resample_k`, `brief_k` and `mute_k` share seed k); names gain `_k`
only when K > 1. Use one `--resample` set per prefix, as the arm names do not record it.

With `--from-checkpoint`, control starts at the latest factual checkpoint at or before `t`. Missing exposure starts
empty there, and every outcome records `exposure_from` rather than implying that earlier exposure was recovered.

Each arm writes `outcome.json` with the outcome vector's inputs at `t + H` (role predicates, sections complete, fair
open, emeralds, evidence each agent has perceived, call counts, held actions the engine rejected); the prefix writes
`<prefix>_report.json` with all arms. Divergence and labels are computed later from these files. Arms run one at a
time; a finished arm is reused and an unfinished one set aside (renamed, never deleted), as in `run_study.py`.

`--forced` replays a factual run recorded by older prompt code (its digests no longer match): every agent is held to its
logged actions in the prefix and the control arm. The other arms then reuse no logged decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

if __package__:
    from . import harness
    from .run_study import replace_brief, save_outcome, set_aside
    from .run_village import model_client
    from .setup_server import setup_server
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import CHECK_EVERY, Checkpoints, VillageSession
else:
    import harness
    from run_study import replace_brief, save_outcome, set_aside
    from run_village import model_client
    from setup_server import setup_server
    from test_m5_village import VillageScriptedPolicy
    from village_session import CHECK_EVERY, Checkpoints, VillageSession

from orchestrator.candidates import same_choice  # noqa: E402
from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402
from orchestrator.perception import EvidenceExposure, PerceptionTap  # noqa: E402
from orchestrator.runtime import (BudgetExceeded, DecisionOverride, NIGHT_REST, SimulationRuntime,  # noqa: E402
                                  THINKING_TICKS)
from orchestrator.scenario import KeywordJudge, OutcomeTracker, PredicateEvaluator, ScenarioManifest  # noqa: E402

ARMS = ("control", "swap", "placebo", "swap_hold", "resample", "brief", "mute")
REJECTED = ("not possible", "failed", "invalid")


def parse_call(text: str) -> ToolCall:
    """A tool call from JSON `{"name": ..., "args": {...}}` (the `--swap` argument)."""
    data = json.loads(text)
    return ToolCall(data["name"], dict(data.get("args") or {}))


def parse_briefs(entries: list[str], agents: list[str]) -> dict[str, str]:
    """Parse repeatable `AGENT=TEXT` entries, preserving nonempty text and rejecting unknown agents."""
    briefs = {}
    for entry in entries:
        name, separator, text = entry.partition("=")
        if not separator or not name:
            raise ValueError("Expected --brief AGENT=TEXT")
        if name not in agents:
            raise ValueError(f"Unknown brief agent: {name}")
        if not text.strip():
            raise ValueError(f"The brief for {name} must be nonempty")
        briefs[name] = text
    return briefs


def parse_cues(entries: list[str], agents: list[str]) -> dict[str, list[str]]:
    """Parse repeatable `AGENT=TEXT` cue entries (texts put to AGENT each hour from the fork, in the brief arm)."""
    cues: dict[str, list[str]] = {}
    for entry in entries:
        name, separator, text = entry.partition("=")
        if not separator or not name:
            raise ValueError("Expected --cue AGENT=TEXT")
        if name not in agents:
            raise ValueError(f"Unknown cue agent: {name}")
        if not text.strip():
            raise ValueError(f"The cue for {name} must be nonempty")
        cues.setdefault(name, []).append(text)
    return cues


def parse_mutes(entries: list[str], agents: list[str]) -> frozenset[tuple[str, str]]:
    """Parse repeatable `LISTENER=SPEAKER` entries into (listener, speaker) pairs of known, distinct agents."""
    pairs = set()
    for entry in entries:
        listener, separator, speaker = entry.partition("=")
        if not separator or listener not in agents or speaker not in agents or listener == speaker:
            raise ValueError(f"Expected --mute LISTENER=SPEAKER with two different agents, not {entry!r}")
        pairs.add((listener, speaker))
    return frozenset(pairs)


def held_rejections(arm: DecisionLog, factual: DecisionLog) -> list[dict]:
    """Held decisions the engine rejected although the factual run's same action at the same tick was accepted."""
    rejected = []
    for decision in arm.decisions():
        logged = factual.get(decision.agent, decision.game_time)
        if (decision.source == "held" and logged is not None and decision.outcome.startswith(REJECTED)
                and not logged.outcome.startswith(REJECTED)):
            rejected.append({"agent": decision.agent, "game_time": decision.game_time, "outcome": decision.outcome,
                             "logged_outcome": logged.outcome})
    return rejected


@dataclass(frozen=True)
class ForkPoint:
    """One decision point of a factual run, the optional act swapped in there, and who the resample-based arms resample
    and mute."""

    run: str
    agent: str
    game_time: int
    swap: ToolCall | None = None
    resample_agents: frozenset[str] = frozenset()
    mute: frozenset[tuple[str, str]] = frozenset()

    def override(self, arm: str, agents: list[str], horizon: int) -> DecisionOverride | None:
        """The runtime override of an arm (None for the control arm, which only replays)."""
        if arm == "swap":
            return DecisionOverride(self.agent, self.game_time, call=self.swap)
        if arm == "placebo":
            return DecisionOverride(self.agent, self.game_time, resample=True)
        if arm == "swap_hold":
            return DecisionOverride(self.agent, self.game_time, call=self.swap,
                                    hold=frozenset(name for name in agents if name != self.agent),
                                    hold_from=self.game_time, hold_until=self.game_time + horizon)
        if arm in {"resample", "brief"}:
            return DecisionOverride(self.agent, self.game_time, resample_agents=self.resample_agents)
        if arm == "mute":
            return DecisionOverride(self.agent, self.game_time, resample_agents=self.resample_agents, mute=self.mute)
        if arm == "control":
            return None
        raise ValueError(f"Unknown arm: {arm}")

    def to_json(self) -> dict:
        """The point as JSON, for outcome files."""
        return {"run": self.run, "agent": self.agent, "game_time": self.game_time,
                "swap": {"name": self.swap.name, "args": self.swap.args} if self.swap is not None else None,
                "resample": sorted(self.resample_agents), "mute": sorted(list(pair) for pair in self.mute)}


class OutcomeInputs:
    """Reads the outcome vector's inputs (§8) from a world at the end of an arm, without changing it."""

    def __init__(self, manifest: ScenarioManifest):
        """Keep the manifest: roles, predicates and schematics."""
        self._manifest = manifest

    def collect(self, client, events: Path, tracker: OutcomeTracker, exposure: EvidenceExposure, end: int) -> dict:
        """Role predicates (reports judged by keywords), each schematic complete, fair open, each agent's emeralds and
        the evidence items each agent knows at `end`. Flushes the event log first (transfer predicates read it)."""
        client.request("save_snapshot")
        manifest = self._manifest
        names = [role["name"] for role in manifest.roles]
        evaluator = PredicateEvaluator(client, events, manifest, KeywordJudge(names), tracker)
        predicates = manifest.data["success_predicates"]
        sections = {structure["name"]: evaluator.evaluate({"blocks_present": [
            {"pos": block["pos"], "block": block["block"]} for block in structure["blocks"]]})
            for structure in manifest.structures()}
        emeralds = {name: sum(entry["count"] for entry in client.request("bot_state", {"name": name})["inventory"]
                              if entry["item"] == "minecraft:emerald") for name in names}
        return {"roles": {name: evaluator.evaluate(predicates[name], name) for name in names},
                "sections": sections, "fair_open": evaluator.evaluate(predicates["fair_open"]),
                "emeralds": emeralds,
                "information": {name: sorted(exposure.items(name, end)) for name in names},
                "first_true": dict(tracker.first_true)}


class DecisionFork:
    """Runs the arms of one decision point: the control replay with its fork, then each other arm from the fork."""

    def __init__(self, args):
        """Keep options, load the manifest and the factual log, and check the point."""
        self._args = args
        self._manifest = self._load_manifest()
        self._agents = [role["name"] for role in self._manifest.roles]
        self._arms = [arm for arm in getattr(args, "arms", "control,swap,placebo").split(",") if arm]
        unknown = sorted(set(self._arms) - set(ARMS))
        if unknown:
            raise ValueError(f"Unknown arms: {unknown}")
        samples = getattr(args, "samples", None)
        self._samples = samples if samples is not None else 1
        if self._samples <= 0:
            raise ValueError("--samples must be positive")
        self._briefs = parse_briefs(getattr(args, "brief", None) or [], self._agents)
        self._cues = parse_cues(getattr(args, "cue", None) or [], self._agents)
        if "brief" in self._arms and not self._briefs:
            raise ValueError("The brief arm requires at least one --brief AGENT=TEXT")
        swap = getattr(args, "swap", None)
        if not swap and set(self._arms) & {"swap", "swap_hold"}:
            raise ValueError("--swap is required for swap and swap_hold arms")
        self._factual = DecisionLog(args.runs_dir / args.run / "orchestrator" / "decisions.jsonl")
        mute = parse_mutes(getattr(args, "mute", None) or [], self._agents)
        if "mute" in self._arms and not mute:
            raise ValueError("The mute arm requires at least one --mute LISTENER=SPEAKER")
        resample = frozenset(getattr(args, "resample", None) or [args.agent])
        if not resample <= set(self._agents):
            raise ValueError(f"Unknown --resample agents: {sorted(resample - set(self._agents))}")
        self.point = ForkPoint(args.run, args.agent, args.tick, parse_call(swap) if swap else None, resample, mute)
        logged = self._factual.get(args.agent, args.tick)
        if logged is None:
            raise ValueError(f"{args.run} has no decision by {args.agent} at tick {args.tick}")
        if self.point.swap is not None and same_choice(self.point.swap, logged.tool_call):
            raise ValueError("The swapped act matches the logged choice; a fork needs a different act")
        self._checkpoint = (Checkpoints.latest(args.runs_dir / args.run, args.tick)
                            if getattr(args, "from_checkpoint", False) else None)
        self._exposure_from = (int(self._checkpoint.name) if self._checkpoint is not None
                               and not (self._checkpoint / "orchestrator/exposure.json").exists() else None)
        self._end = args.tick + args.horizon
        self._prefix = f"fork_{args.run}_{args.agent}_{args.tick}"
        self._outcomes = self._outcome_inputs()

    def _load_manifest(self) -> ScenarioManifest:
        """The scenario of the factual run (the village variant's committed manifest)."""
        return ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario"
                                     / f"village_{self._args.variant}.json")

    def _fixture(self) -> Path:
        """The built world the factual run started from."""
        return self._args.runs_dir / f"village_{self._args.variant}"

    def _exposure(self, known: dict | None = None) -> EvidenceExposure:
        """The evidence-exposure record arms keep (optionally restored)."""
        return EvidenceExposure(self._manifest, known)

    def _begin(self, client, now: int) -> int:
        """Prepare a world started from the fixture as the factual run did before its agents acted; returns the
        game time agents start at (the village starts them at once)."""
        return now

    def _outcome_inputs(self) -> OutcomeInputs:
        """What each arm's outcome vector is read from."""
        return OutcomeInputs(self._manifest)

    def directory(self, name: str) -> Path:
        """A run directory of this point (`control`, `fork`, or another arm)."""
        return self._args.runs_dir / f"{self._prefix}_{name}"

    def _arm_name(self, arm: str, sample: int) -> str:
        """Keep historical single-arm names, adding the pair index only for multiple samples, and the label (if any)
        to every arm but the control, which a later labelled fork of the same point reuses."""
        name = f"{arm}_{sample}" if arm != "control" and self._samples > 1 else arm
        label = getattr(self._args, "label", None)
        return f"{name}_{label}" if label and arm != "control" else name

    def _tracker(self) -> OutcomeTracker:
        """Track when the bridge was complete and the fair opened (as the study runner does)."""
        predicates = self._manifest.data["success_predicates"]
        return OutcomeTracker({"bridge_complete": predicates["builder"]["all"][0],
                               "fair_open": predicates["fair_open"]})

    def _runtime(self, client, agents, directory: Path, model, override,
                 sample_seed: int | None = None) -> SimulationRuntime:
        """A village runtime as the factual run was configured, with the factual log as source."""
        return SimulationRuntime(client, agents, model, directory / "orchestrator", command_error=harness.CommandError,
                                 source=self._factual, hash_region=self._manifest.data["layout"]["region"],
                                 hash_every=CHECK_EVERY, max_model_calls=self._args.max_calls,
                                 sample_seed=sample_seed,
                                 model_concurrency=1 if self._args.provider == "mlx-local" else None,
                                 min_gap=THINKING_TICKS, rest=NIGHT_REST, override=override)

    def _model(self):
        """The continuation model: the scripted village policy ($0) or a live client."""
        if self._args.model == "scripted":
            return ScriptedClient(VillageScriptedPolicy(self._manifest).decide)
        return model_client(self._args)

    def _finish(self, arm: str, client, directory: Path, runtime: SimulationRuntime, tracker: OutcomeTracker,
                exposure: EvidenceExposure, sample: int = 1) -> dict:
        """Save and return an arm's outcome inputs."""
        if arm not in {"control", "brief"} and not runtime.override_applied:
            raise RuntimeError(f"The {arm} arm never reached {self.point.agent} at tick {self.point.game_time}")
        outcome = {"arm": self._arm_name(arm, sample), "point": self.point.to_json(), "end": self._end,
                   **self._outcomes.collect(client, directory / "world/simharness/events.jsonl", tracker, exposure,
                                            self._end),
                   "calls": runtime.stats.model_calls, "decisions": runtime.stats.decisions,
                   "replayed": runtime.stats.replayed, "override_applied": runtime.override_applied,
                   "held_rejected": held_rejections(runtime.log, self._factual)}
        if self._exposure_from is not None:
            outcome["exposure_from"] = self._exposure_from
        return save_outcome(directory, outcome)

    def control(self) -> dict:
        """Replay the factual run to the point, save the fork there, then continue to `t + H` (the control arm)."""
        args, manifest = self._args, self._manifest
        directory, fork = self.directory("control"), self.directory("fork")
        saved = self._saved(directory)
        if saved is not None and fork.exists():
            return saved
        for path in (directory, fork):
            if path.exists():
                raise FileExistsError(f"{path} exists without a finished control arm; rename it to start over")
        setup_server(fork.name, args.runs_dir)
        source = self._checkpoint if self._checkpoint is not None else self._fixture()
        session = VillageSession(directory, source, manifest, harness.server_options(args),
                                 fresh=self._checkpoint is None)
        session.clone()
        exposure, tracker = self._exposure(), self._tracker()
        state = directory / "orchestrator"
        if self._checkpoint is None:
            agents = manifest.agents(args.epoch_turns)
        else:
            _, agents = SimulationRuntime.load_agents(state / "fork_state.json")
            tracker.first_true.update(json.loads((state / "first_true.json").read_text(encoding="utf-8")))
            if (state / "exposure.json").exists():
                exposure = self._exposure(json.loads((state / "exposure.json").read_text(encoding="utf-8")))
        override = DecisionOverride(hold=frozenset(self._agents)) if args.forced else None
        with session:
            client, now = session.client, session.start
            if self._checkpoint is None:
                now = self._begin(client, now)
            runtime = self._runtime(PerceptionTap(client, exposure), agents, directory, None, override)
            if self._checkpoint is not None:
                runtime.restore_timing(state / "fork_state.json")
            now = session.run_checked(runtime, tracker, now, self.point.game_time)
            session.save_fork(fork, runtime)
            (fork / "orchestrator" / "exposure.json").write_text(json.dumps(exposure.to_json()) + "\n",
                                                                 encoding="utf-8")
            (fork / "orchestrator" / "first_true.json").write_text(json.dumps(tracker.first_true) + "\n",
                                                                   encoding="utf-8")
            session.run_checked(runtime, tracker, now, self._end)
            return self._finish("control", client, directory, runtime, tracker, exposure)

    def branch(self, arm: str, sample: int = 1) -> dict:
        """Run one arm from the fork to `t + H` in its own clone; return its outcome inputs."""
        args, fork = self._args, self.directory("fork")
        directory = self.directory(self._arm_name(arm, sample))
        saved = self._saved(directory)
        if saved is not None:
            return saved
        if directory.exists():
            set_aside(directory)
        session = VillageSession(directory, fork, self._manifest, harness.server_options(args), fresh=False)
        session.clone()
        state = directory / "orchestrator"
        _, agents = SimulationRuntime.load_agents(state / "fork_state.json")
        if arm == "brief":
            agents = [replace_brief(agent, self._briefs.get(agent.name)) for agent in agents]
            for agent in agents:  # a counsellor's question each hour from the fork (a cue), on top of any brief
                agent.cues = (*agent.cues, *({"text": text, "start": self.point.game_time, "every": 1000}
                                             for text in self._cues.get(agent.name, ())))
        exposure = self._exposure(json.loads((state / "exposure.json").read_text(encoding="utf-8")))
        tracker = self._tracker()
        tracker.first_true.update(json.loads((state / "first_true.json").read_text(encoding="utf-8")))
        with session:
            client, now = session.client, session.start
            runtime = self._runtime(PerceptionTap(client, exposure), agents, directory, self._model(),
                                    self.point.override(arm, self._agents, args.horizon),
                                    sample_seed=sample if getattr(args, "samples", None) is not None else None)
            runtime.restore_timing(state / "fork_state.json")
            session.run_checked(runtime, tracker, now, self._end)
            return self._finish(arm, client, directory, runtime, tracker, exposure, sample)

    @staticmethod
    def _saved(directory: Path) -> dict | None:
        """A finished arm's saved outcome, or None."""
        path = directory / "outcome.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def run(self) -> dict:
        """The control arm (with the fork), then the other requested arms in order; writes and returns the report. A
        model-call budget or usage limit stops the remaining arms."""
        report = {"point": self.point.to_json(), "horizon": self._args.horizon, "arms": {}, "stopped": None}
        report["arms"]["control"] = self.control()
        try:
            for arm in self._arms:
                if arm == "control":
                    continue
                for sample in range(1, self._samples + 1):
                    report["arms"][self._arm_name(arm, sample)] = self.branch(arm, sample)
        except BudgetExceeded as limit:
            report["stopped"] = str(limit)
        label = getattr(self._args, "label", None)
        path = self._args.runs_dir / f"{self._prefix}{'_' + label if label else ''}_report.json"
        path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return report


def fork_parser(description: str):
    """The options every decision fork takes (the village and the Jubilee forks share them)."""
    parser = harness.experiment_parser(description)
    parser.add_argument("--run", required=True, help="the factual run's directory name under --runs-dir")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--tick", type=int, required=True)
    parser.add_argument("--swap", help='the act swapped in: {"name": ..., "args": {...}} (required for swap arms)')
    parser.add_argument("--resample", action="append", default=[], metavar="AGENT",
                        help="an agent the resample, brief and mute arms ask afresh from t (default: --agent)")
    parser.add_argument("--mute", action="append", default=[], metavar="LISTENER=SPEAKER",
                        help="for the mute arm: the listener stops perceiving the speaker's speech from t")
    parser.add_argument("--brief", action="append", default=[], metavar="AGENT=TEXT",
                        help="a replacement private brief")
    parser.add_argument("--cue", action="append", default=[], metavar="AGENT=TEXT",
                        help="for the brief arm: a question put to AGENT at its first decision in each game hour "
                             "from the fork")
    parser.add_argument("--samples", type=int, default=None, help="positive samples per arm (default: 1, unseeded)")
    parser.add_argument("--from-checkpoint", action="store_true",
                        help="replay from the latest checkpoint at or before t")
    parser.add_argument("--arms", default="control,swap,placebo", help="comma-separated, from " + ",".join(ARMS))
    parser.add_argument("--horizon", type=int, default=2400)
    parser.add_argument("--label", default=None,
                        help="suffix for the arms (all but the control) and the report, so a second fork of a point "
                             "with different arms reuses the control and the fork but none of the earlier arms")
    parser.add_argument("--epoch-turns", type=int, default=8, help="as the factual run was configured")
    parser.add_argument("--forced", action="store_true", help="the factual run was recorded by older prompt code")
    parser.add_argument("--model", default="scripted", help="'scripted' ($0 plumbing) or a live model id")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="chatgpt-plan")
    parser.add_argument("--reasoning-effort", default=None, help="for --provider chatgpt-plan, e.g. low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=None, help="per-arm budget of model calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the arms of one decision point, and print a short summary."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = fork_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    report = DecisionFork(parser.parse_args(argv)).run()
    print(json.dumps({"point": report["point"], "stopped": report["stopped"],
                      "arms": {arm: {key: outcome[key] for key in ("calls", "decisions", "replayed", "fair_open")}
                               for arm, outcome in report["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
