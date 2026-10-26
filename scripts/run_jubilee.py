"""Run the Jubilee in the town with LLM (or scripted) agents, read its outcomes, and optionally replay it.

This spends money with a live model: `--max-calls` is a hard budget. The run starts from the built town
(`runs/town_jubilee`), runs to the reveal (18:00 on day 1), reads the great hall there, runs one more hour for the
mayor's arrival, and writes `jubilee_report.json`. `--condition FILE` replaces the manifest's condition knobs for this
episode (recorded in the run's `scenario.json`). There is no fixed night rest: agents choose when to `sleep`.
`--checkpoint-every`, `--resume` and `--replay` work as in `run_village.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

if __package__:
    from . import harness
    from .jubilee_outcomes import HOUR, JubileeOutcomes
    from .provenance import FixtureMetadata
    from .run_village import model_client, prepare_resume
    from .test_m5_village import decision_trace
    from .village_session import Checkpoints, VillageSession
else:
    import harness
    from jubilee_outcomes import HOUR, JubileeOutcomes
    from provenance import FixtureMetadata
    from run_village import model_client, prepare_resume
    from test_m5_village import decision_trace
    from village_session import Checkpoints, VillageSession

from orchestrator.decisions import DecisionLog  # noqa: E402  (test_m5_village sets sys.path)
from orchestrator.jubilee import JubileeManifest  # noqa: E402
from orchestrator.runtime import THINKING_TICKS, BudgetExceeded, SimulationRuntime  # noqa: E402
from orchestrator.scenario import OutcomeTracker  # noqa: E402

FIXTURE = "town_jubilee"


class JubileeExperiment:
    """One Jubilee episode from the built town (or a checkpoint), with its outcomes."""

    def __init__(self, args, condition: dict | None = None):
        """Load the town's manifest and apply this episode's condition knobs, if any."""
        self._args = args
        data = json.loads((args.runs_dir / FIXTURE / "scenario.json").read_text(encoding="utf-8"))
        if condition is not None:
            data["jubilee"] = {**data["jubilee"], "condition": condition}
        self.manifest = JubileeManifest(data)
        self.outcomes = JubileeOutcomes(data)
        self.agent_start = data["jubilee"].get("agent_start_game_time", data["layout"].get("agent_start_game_time", 0))

    def run_once(self, name: str, model, source: DecisionLog | None, *, max_model_calls: int | None = None,
                 epoch_turns: int = 8, checkpoints: Checkpoints | None = None, resume_from: Path | None = None,
                 end_tick: int | None = None) -> tuple[SimulationRuntime, dict]:
        """Run to the reveal, read the hall, run on to `end_tick` (default an hour after the reveal), and report."""
        args, manifest = self._args, self.manifest
        reveal = self.outcomes.reveal
        end = end_tick if end_tick is not None else reveal + HOUR
        directory = args.runs_dir / name
        session = VillageSession(directory, resume_from if resume_from is not None else args.runs_dir / FIXTURE,
                                 manifest, harness.server_options(args), fresh=resume_from is None)
        session.clone()
        agents = (manifest.agents(epoch_turns) if resume_from is None
                  else SimulationRuntime.load_agents(directory / "orchestrator" / "fork_state.json")[1])
        tracker = OutcomeTracker({})
        prefix_hashes = Checkpoints.carry_prefix(resume_from, directory) if resume_from is not None else []
        hall = None
        with session:
            client, now = session.client, session.start
            if resume_from is None:  # a checkpoint keeps the settings, people, stock and time it saved
                if manifest.engine_overrides:
                    client.request("config_set", manifest.engine_overrides)
                for request in manifest.setup_requests():
                    client.request(request["request"], request["args"])
                start = self.agent_start
                if now < start:  # the town (villagers included) lives on until agents start, at 08:00 by default
                    now = harness.game_time(client.request("step", {"n": start - now}))
            runtime = SimulationRuntime(client, agents, model, directory / "orchestrator",
                                        command_error=harness.CommandError, source=source,
                                        hash_region=manifest.data["layout"]["region"], hash_every=1_200,
                                        max_model_calls=max_model_calls, min_gap=THINKING_TICKS, rest=None,
                                        perception_log=True)
            if resume_from is not None:
                runtime.restore_timing(directory / "orchestrator" / "fork_state.json")
                runtime.hashes.extend(prefix_hashes)
            stopped = None
            try:
                if now < reveal:
                    now = session.run_checked(runtime, tracker, now, min(reveal, end), checkpoints)
                if now == reveal:
                    hall = self.outcomes.hall(client)
                if now < end:
                    now = session.run_checked(runtime, tracker, now, end, checkpoints)
            except BudgetExceeded as error:
                stopped = str(error)
                now = harness.game_time(client.request("ping"))
            client.request("save_snapshot")
        result = (self.outcomes.result(hall, directory / "orchestrator" / "trajectory.jsonl").to_json()
                  if hall is not None else None)
        report = {"end": now, "stopped": stopped, "stats": runtime.stats.__dict__, "outcome": result,
                  "condition": manifest.condition, "fixture": FixtureMetadata.read(directory),
                  "hashes": len(runtime.hashes)}
        if not (directory / "scenario.json").exists() or resume_from is None:
            (directory / "scenario.json").write_text(json.dumps(manifest.data, indent=1) + "\n", encoding="utf-8")
        (directory / "jubilee_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return runtime, report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one Jubilee episode, print its report, and optionally replay it."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="chatgpt-plan")
    parser.add_argument("--reasoning-effort", default=None, help="for --provider chatgpt-plan, e.g. low")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, required=True, help="hard budget of model calls")
    parser.add_argument("--ticks", type=int, default=None, help="end here instead of an hour after the reveal")
    parser.add_argument("--name", required=True, help="run directory name")
    parser.add_argument("--condition", type=Path, default=None, help="JSON file of condition knobs")
    parser.add_argument("--resume", action="store_true", help="restart --name from its latest checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="save a checkpoint every N ticks")
    parser.add_argument("--epoch-turns", type=int, default=8)
    parser.add_argument("--replay", action="store_true", help="then replay from the decision log and compare")
    parser.add_argument("--replay-only", action="store_true",
                        help="replay the finished run --name from its decision log and compare with its saved hashes")
    args = parser.parse_args(argv)
    condition = json.loads(args.condition.read_text(encoding="utf-8")) if args.condition else None
    experiment = JubileeExperiment(args, condition)
    if args.replay_only:
        recorded = args.runs_dir / args.name
        log = DecisionLog(recorded / "orchestrator" / "decisions.jsonl")
        end = json.loads((recorded / "jubilee_report.json").read_text(encoding="utf-8"))["end"]
        hashes = json.loads((recorded / "orchestrator" / "hashes.json").read_text(encoding="utf-8"))
        return replay_and_compare(experiment, args, log, hashes, end, None)
    checkpoints = Checkpoints(args.checkpoint_every) if args.checkpoint_every is not None else None
    end = args.ticks if args.ticks is not None else experiment.outcomes.reveal + HOUR
    resume_from = prepare_resume(args.runs_dir / args.name, end) if args.resume else None
    source = (DecisionLog(resume_from.parent.parent / "orchestrator" / "decisions.jsonl")
              if resume_from is not None else None)
    runtime, report = experiment.run_once(args.name, model_client(args), source, max_model_calls=args.max_calls,
                                          epoch_turns=args.epoch_turns, checkpoints=checkpoints,
                                          resume_from=resume_from, end_tick=end)
    print(json.dumps({key: report[key] for key in ("end", "stopped", "stats", "outcome")}, indent=1))
    if not args.replay:
        return 0
    return replay_and_compare(experiment, args, runtime.log, runtime.hashes, report["end"], resume_from)


def replay_and_compare(experiment: JubileeExperiment, args, log: DecisionLog, hashes: list, end: int,
                       resume_from: Path | None) -> int:
    """Replay `args.name` from its decision log with no model and compare decisions and world hashes; a passing
    replay keeps only its logs and report (disk)."""
    replay, _ = experiment.run_once(f"{args.name}_replay", None, log, epoch_turns=args.epoch_turns,
                                    resume_from=resume_from, end_tick=end)
    same = (replay.stats.model_calls == 0 and decision_trace(replay.log) == decision_trace(log)
            and replay.hashes == hashes)
    print(f"{'PASS' if same else 'FAIL'} replay of {args.name}: {replay.stats.replayed} decisions reused, "
          f"{replay.stats.model_calls} model calls, {len(replay.hashes)} hashes")
    if same:  # a passing replay has shown what it can; keep its logs and report, not another world (disk)
        replay_dir = args.runs_dir / f"{args.name}_replay"
        for path in replay_dir.iterdir():
            if path.name not in ("orchestrator", "jubilee_report.json", "scenario.json"):
                shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
