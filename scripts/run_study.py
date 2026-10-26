"""Counterfactual studies on the village (gate V-G4): K paired branches, cone-gated re-query, effects with CIs.

One factual run goes from the built village to `--end-day`, forking at the start of `--fork-day`: the world snapshot
and every agent's state are saved. The study then runs K pairs of branches from that fork, each branch in its own
server (branches run in parallel on separate ports):
- **Control k** applies no intervention and is an independent sample of what happens after the fork: pair 0's control
  is the factual run itself; controls 1..K-1 restore the fork and run with no source log, so every decision is sampled
  afresh.
- **Intervention k** applies the study's intervention at the fork and runs with control k's decision log as its source.
  Wherever an agent's prompt is unchanged it reuses control k's decision (like common random numbers), so only agents
  inside the intervention's cone call the model; that saving is reported.

Pairs are independent of each other and coupled within, so effects are mean paired differences (intervention k minus
control k) with percentile bootstrap CIs over pairs: villager welfare per day, the ticks at which the bridge was
complete and the fair opened (a run where it never happened counts as the end time), and each role's success predicate
(1/0).

With `--base NAME`, the factual run, its fork and the controls are named after NAME instead of the study, so several
studies forking from the same run on the same day share them: each study runs only its interventions, paired with the
shared controls. Run such studies one after another (the first runs the shared part).

`--checkpoint-every N` also saves world, agent and outcome-tracker snapshots during factual runs and branches; N must
be a positive multiple of 1,200. Without this option, no checkpoints are saved.

Every finished run saves `outcome.json`, so a study resumes: finished runs are reused, and an unfinished branch
directory is set aside (renamed, never deleted) and run again. A model-call budget or the provider's usage limit stops
the branch it hits, no further branches start, and the report covers the pairs that finished. Reusing one control for
every branch would make the controls identical (zero variance) and the interventions correlated, which understates the
uncertainty.

The three studies from the MVP scope:
- `road_closure`: the main west road is walled off on the fork day;
- `no_saboteur`: the envoy is a genuine guest;
- `honest_freight`: the freight operator is honest.

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import random
from pathlib import Path
import threading
import time

if __package__:
    from . import harness, provenance
    from .run_village import model_client
    from .setup_server import setup_server
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import CHECK_EVERY, Checkpoints, VillageSession
    from .village_metrics import VillageMetrics
else:
    import harness
    import provenance
    from run_village import model_client
    from setup_server import setup_server
    from test_m5_village import VillageScriptedPolicy
    from village_session import CHECK_EVERY, Checkpoints, VillageSession
    from village_metrics import VillageMetrics

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import ScriptedClient  # noqa: E402
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS, BudgetExceeded, SimulationRuntime  # noqa: E402
from orchestrator.scenario import (KeywordJudge, LLMJudge, OutcomeTracker, PredicateEvaluator,  # noqa: E402
                                   ScenarioManifest, TICKS_PER_DAY)


@dataclass(frozen=True)
class Intervention:
    """What a study changes at the fork: world commands, and/or replaced private briefs for some agents."""

    commands: tuple[str, ...] = ()
    private_briefs: dict = field(default_factory=dict)


STUDIES = {
    "road_closure": Intervention(
        commands=("fill -13 -60 20 -10 -60 20 minecraft:cobblestone_wall",)),  # full width: path + shoulders
    "no_saboteur": Intervention(private_briefs={
        "envoy": "Your private goal: Enjoy the fair as a guest. You had nothing to do with the bridge collapse."}),
    "honest_freight": Intervention(private_briefs={
        "freight": "Your private goal: Deliver goods to the fair honestly and charge fair prices."}),
}


def paired_ci(control: list[float], treated: list[float], samples: int = 2000, seed: int = 0) -> tuple:
    """Mean paired difference treated[k] - control[k] and its percentile 95% bootstrap CI, resampling pairs."""
    generator = random.Random(seed)
    differences = [after - before for before, after in zip(control, treated, strict=True)]
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    effects = sorted(mean([generator.choice(differences) for _ in differences]) for _ in range(samples))
    return mean(differences), effects[int(0.025 * samples)], effects[int(0.975 * samples) - 1]


OUTCOME = "outcome.json"


def save_outcome(directory: Path, outcome: dict) -> dict:
    """Record a finished outcome and its offline metrics, then return the outcome for study resume."""
    (directory / OUTCOME).write_text(json.dumps(outcome, indent=1) + "\n", encoding="utf-8")
    if (directory / "scenario.json").exists():
        VillageMetrics(directory).compute()
    return outcome


def load_outcome(directory: Path) -> dict | None:
    """A finished run's saved outcome, or None if the run did not finish."""
    path = directory / OUTCOME
    if not path.exists():
        return None
    outcome = json.loads(path.read_text(encoding="utf-8"))
    outcome["welfare"] = {int(day): value for day, value in outcome["welfare"].items()}
    return outcome


def set_aside(directory: Path) -> Path:
    """Rename an unfinished run directory (kept for inspection) so the run can start again."""
    target = directory.with_name(f"{directory.name}_unfinished_{time.strftime('%Y%m%d-%H%M%S')}")
    directory.rename(target)
    return target


class Study:
    """Runs one study: the factual run with its fork, then control and intervention branches in parallel."""

    def __init__(self, args):
        """Keep options, load the manifest, and fix the fork and end times."""
        self._args = args
        interval = getattr(args, "checkpoint_every", None)
        self._checkpoints = Checkpoints(interval) if interval is not None else None
        self._manifest = ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario"
                                               / f"village_{args.variant}.json")
        self._intervention = STUDIES[args.study]
        self._fork = args.fork_day * TICKS_PER_DAY
        self._end = args.end_day * TICKS_PER_DAY
        self._prefix = f"study_{args.study}_{args.variant}"
        self._base = getattr(args, "base", None) or self._prefix  # names the factual run, fork and controls
        self._stopped: str | None = None
        self._live: set[VillageSession] = set()  # branch sessions running now, for an interrupt to stop (C8)
        self._live_lock = threading.Lock()

    def _model(self):
        """The agents' model: the scripted village policy (for $0 plumbing) or a live client."""
        if self._args.model == "scripted":
            return ScriptedClient(VillageScriptedPolicy(self._manifest).decide)
        return model_client(self._args)

    def _epoch_turns(self) -> int:
        """Decisions per prompt epoch: 1 for the scripted policy (it reads full prompts), else the CLI value or 8."""
        if self._args.epoch_turns is not None:
            return self._args.epoch_turns
        return 1 if self._args.model == "scripted" else 8

    def _runtime(self, client, agents, directory: Path, source: DecisionLog | None,
                 sample_seed: int | None = None) -> SimulationRuntime:
        """A village runtime with hashing at the outcome-check interval; each branch samples with its own seed."""
        return SimulationRuntime(client, agents, self._model(), directory / "orchestrator",
                                 command_error=harness.CommandError, source=source,
                                 hash_region=self._manifest.data["layout"]["region"], hash_every=CHECK_EVERY,
                                 max_model_calls=self._args.max_calls, sample_seed=sample_seed,
                                 model_concurrency=1 if self._args.provider == "mlx-local" else None,
                                 min_gap=THINKING_TICKS, rest=NIGHT_REST)

    def _tracker(self) -> OutcomeTracker:
        """Track when the bridge was completed and when the fair opened."""
        predicates = self._manifest.data["success_predicates"]
        return OutcomeTracker({"bridge_complete": predicates["builder"]["all"][0],
                               "fair_ready": predicates["fair_ready"], "fair_open": predicates["fair_open"]})

    def factual(self) -> Path:
        """Run the factual village, saving the fork (world and agents) at the fork time. Returns the fork directory."""
        args = self._args
        directory, fork = args.runs_dir / f"{self._base}_factual", args.runs_dir / f"{self._base}_fork"
        saved = load_outcome(directory)
        if saved is not None and fork.exists():  # resume: the factual run and its fork are done
            self._factual_outcome = saved
            return fork
        for path in (directory, fork):
            if path.exists():
                raise FileExistsError(f"{path} exists without a finished factual run; rename it to start over")
        fork = setup_server(fork.name, args.runs_dir)
        session = VillageSession(directory, args.runs_dir / f"village_{args.variant}", self._manifest,
                                 harness.server_options(args))
        session.clone()
        tracker = self._tracker()
        with session:
            client, now = session.client, session.start
            runtime = self._runtime(client, self._manifest.agents(self._epoch_turns()), directory, None)
            now = session.run_checked(runtime, tracker, now, self._fork, self._checkpoints)
            session.save_fork(fork, runtime)
            session.run_checked(runtime, tracker, now, self._end, self._checkpoints)
            self._factual_outcome = save_outcome(directory, self._outcome(client, directory, runtime, tracker))
        return fork

    def _log_dir(self, arm: str, index: int) -> Path:
        """The run directory of a branch; pair 0's control is the factual run. Controls (and the factual run) are
        named after the base, interventions after the study."""
        name = "factual" if (arm, index) == ("control", 0) else f"{arm}_{index}"
        return self._args.runs_dir / f"{self._base if arm == 'control' else self._prefix}_{name}"

    def _branch_options(self, arm: str, index: int) -> dict:
        """Use paired ports from branch-port when supplied, otherwise preserve the original branch ports."""
        options = harness.server_options(self._args)
        base = getattr(self._args, "branch_port", None)
        if base is None:
            options.update(port=25610 + index * 2 + (arm == "intervention"),
                           game_port=25700 + index * 2 + (arm == "intervention"))
        else:
            port = base + 4 * index + 2 * (arm == "intervention")
            options.update(port=port, game_port=port + 1)
        return options

    def branch(self, fork: Path, arm: str, index: int) -> dict:
        """Run one branch from the fork in its own server; return its outcome. A control samples afresh; intervention
        k reuses control k's decisions where prompts are unchanged. Before cloning, require the factual run's latest
        recorded jar to match the current jar; factual runs predating provenance remain usable."""
        directory = self._log_dir(arm, index)
        saved = load_outcome(directory)
        if saved is not None:
            return saved
        if self._stopped:
            return {"stopped": self._stopped}
        factual = self._log_dir("control", 0)
        starts = provenance.RunProvenance.starts(factual)
        if starts:
            recorded_jar = starts[-1].get("jar_sha256")
            current_jar = provenance.file_sha256(harness.MOD_JAR)
            if recorded_jar != current_jar:
                raise harness.HarnessError(f"Cannot branch {factual}: jar changed "
                                           f"(factual {recorded_jar}, current {current_jar})")
        if directory.exists():
            set_aside(directory)
        options = self._branch_options(arm, index)
        session = VillageSession(directory, fork, self._manifest, options, fresh=False)
        session.clone()
        _, agents = SimulationRuntime.load_agents(directory / "orchestrator" / "fork_state.json")
        if arm == "intervention":
            agents = [replace_brief(agent, self._intervention.private_briefs.get(agent.name)) for agent in agents]
        source = (DecisionLog(self._log_dir("control", index) / "orchestrator" / "decisions.jsonl")
                  if arm == "intervention" else None)
        tracker = self._tracker()
        tracker.first_true.update({name: when for name, when in self._factual_outcome["first_true"].items()
                                   if when is not None and when <= self._fork})
        with self._live_lock:
            if self._stopped:
                return {"stopped": self._stopped}
            self._live.add(session)
        try:
            with session:
                client, now = session.client, session.start
                if arm == "intervention":
                    for command in self._intervention.commands:
                        client.request("command", {"command": command})
                runtime = self._runtime(client, agents, directory, source,
                                        sample_seed=(1 + 2 * index + (arm == "intervention"))
                                        if self._args.seeded else None)
                runtime.restore_timing(directory / "orchestrator" / "fork_state.json")
                try:
                    session.run_checked(runtime, tracker, now, self._end, self._checkpoints)
                except BudgetExceeded as limit:
                    self._stopped = str(limit)  # no further branches start
                    return {"stopped": str(limit)}
                return save_outcome(directory, self._outcome(client, directory, runtime, tracker))
        finally:
            with self._live_lock:
                self._live.discard(session)

    def _interrupt(self) -> None:
        """Start no further branch and terminate the running branches' servers, so their threads fail and the
        executor can shut down; their directories stay unfinished, to be set aside on resume."""
        with self._live_lock:
            self._stopped = "interrupted"
            live = list(self._live)
        for session in live:
            session.kill()

    def _outcome(self, client, directory: Path, runtime: SimulationRuntime, tracker: OutcomeTracker) -> dict:
        """Welfare per completed day, outcome times, role predicates, and model-call counts."""
        client.request("save_snapshot")
        if not (directory / "scenario.json").exists():
            (directory / "scenario.json").write_text(json.dumps(self._manifest.data, indent=1) + "\n", encoding="utf-8")
        suspects = [role["name"] for role in self._manifest.roles]
        judge = (LLMJudge(model_client(self._args), suspects, self._manifest.data["evidence"]["expected_chain"])
                 if self._args.model != "scripted" and self._args.provider in ("mlx-local", "chatgpt-plan")
                 else KeywordJudge(suspects))
        evaluator = PredicateEvaluator(client, directory / "world/simharness/events.jsonl", self._manifest, judge,
                                       tracker)
        predicates = self._manifest.data["success_predicates"]
        return {"end": self._end, "npc_metrics": client.request("npc_metrics", {})["days"],
                "welfare": {day: evaluator.welfare(day) for day in range(self._end // TICKS_PER_DAY)},
                "first_true": dict(tracker.first_true),
                "roles": {role["name"]: evaluator.evaluate(predicates[role["name"]], role["name"])
                          for role in self._manifest.roles},
                "builder_paperwork": (evaluator.evaluate(predicates["builder_paperwork"])
                                      if "builder_paperwork" in predicates else None),
                "calls": runtime.stats.model_calls, "decisions": runtime.stats.decisions,
                "replayed": runtime.stats.replayed}

    def run(self) -> dict:
        """Run the factual run, then the controls, then the interventions (each needs its control's log); aggregate
        paired effects over the pairs that finished, and call savings."""
        fork = self.factual()
        pairs = range(self._args.branches)
        with ThreadPoolExecutor(max_workers=self._args.parallel) as pool:
            try:
                controls = [{**self._factual_outcome, "calls": 0, "decisions": 0, "replayed": 0}]  # pair 0: no run
                controls += list(pool.map(lambda index: self.branch(fork, "control", index), pairs[1:]))
                ready = [index for index in pairs if "stopped" not in controls[index]]
                treated = dict(zip(ready, pool.map(lambda index: self.branch(fork, "intervention", index), ready)))
            except KeyboardInterrupt:  # SIGINT or SIGTERM (C8): the branch threads never see it, so stop their servers
                self._interrupt()
                raise
        finished = [index for index in ready if "stopped" not in treated[index]]
        arms = {"control": [controls[index] for index in finished],
                "intervention": [treated[index] for index in finished]}
        calls = {arm: sum(o["calls"] for o in outcomes) for arm, outcomes in arms.items()}
        decisions = {arm: sum(o["decisions"] for o in outcomes) for arm, outcomes in arms.items()}
        report = {"study": self._args.study, "variant": self._args.variant, "branches": self._args.branches,
                  "base": self._base,
                  "fixture": provenance.FixtureMetadata.read(self._log_dir("control", 0)),
                  "pairsFinished": len(finished), "stopped": self._stopped, "factual": self._factual_outcome,
                  "effects": self._effects(arms["control"], arms["intervention"]) if finished else {},
                  "model_calls": calls, "decisions": decisions,
                  "calls_saved_by_gating": {arm: 1 - calls[arm] / decisions[arm] for arm in arms if decisions[arm]},
                  "branches_detail": arms}
        (self._args.runs_dir / f"{self._prefix}_report.json").write_text(json.dumps(report, indent=1) + "\n",
                                                                         encoding="utf-8")
        return report

    def _effects(self, control: list[dict], treated: list[dict]) -> dict:
        """Paired effects with CIs: welfare per post-fork day, outcome ticks (never = the end time), role successes
        (1/0, over the pairs where both runs have a verdict)."""
        effects = {}
        for day in range(self._fork // TICKS_PER_DAY, self._end // TICKS_PER_DAY):
            effects[f"welfare_day_{day}"] = paired_ci([o["welfare"][day] for o in control],
                                                      [o["welfare"][day] for o in treated])
        for name in ("bridge_complete", "fair_ready", "fair_open"):
            tick = lambda outcome: outcome["first_true"].get(name) or self._end  # noqa: E731
            effects[f"{name}_tick"] = paired_ci([tick(o) for o in control], [tick(o) for o in treated])
        for role in self._manifest.roles:
            both = [(c["roles"][role["name"]], t["roles"][role["name"]]) for c, t in zip(control, treated)
                    if c["roles"].get(role["name"]) is not None and t["roles"].get(role["name"]) is not None]
            if both:
                effects[f"{role['name']}_success"] = paired_ci([float(c) for c, _ in both], [float(t) for _, t in both])
        return effects


def replace_brief(agent, private: str | None):
    """Return the agent with its private brief replaced (unchanged if `private` is None)."""
    if private is not None:
        agent.brief = replace(agent.brief, private=private)
    return agent


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the study, and print effects and call savings."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--fork-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=2)
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--base", default=None,
                        help="name of a factual run, fork and controls that several studies share")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--branch-port", type=int, default=None,
                        help="branch control port P + 4*k + 2*a (a=0 control, 1 intervention); game port one higher")
    parser.add_argument("--model", default="scripted", help="'scripted' ($0 plumbing) or a live model id")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="anthropic")
    parser.add_argument("--reasoning-effort", default=None, help="for --provider chatgpt-plan, e.g. low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=None, help="per-run budget of model calls")
    parser.add_argument("--epoch-turns", type=int, default=None,
                        help="decisions per append-only prompt epoch (default: 1 for the scripted policy, else 8)")
    parser.add_argument("--seeded", action="store_true",
                        help="seed every model call (reproducible fresh reruns; slower on local servers)")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="save a checkpoint every N ticks")
    report = Study(parser.parse_args(argv)).run()
    print(json.dumps({key: report[key] for key in ("study", "variant", "branches", "effects", "model_calls",
                                                   "decisions", "calls_saved_by_gating")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
