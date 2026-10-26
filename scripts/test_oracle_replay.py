"""The replay oracle (plan §0): a recorded live village run replays exactly from its decision log with today's code.

A run recorded with a live model (the oracle, by default `oracle_sabotage_1024`: two days, Luna, recorded from main
`7047c5b` after the sprint's doors, evidence witnessing, ferry fixes and tier-1 tools changed the prompts) is
replayed from the built village with no model, reusing a logged decision only where today's prompt has the same digest.
The gate passes if the replay makes zero model calls and reproduces the oracle's decision trace and its world-hash trace
exactly. Behaviour-preserving cleanups (C3, C5, C8 and the like) must keep it passing; an intentional prompt change
needs a fresh recorded oracle. The oracle directory is read from `--runs-dir` (the runner clones it there); only its
`orchestrator/decisions.jsonl` and `orchestrator/hashes.json` are used. The replay is configured as `run_village.py`
records: epoch turns 8, thinking time and night rest.
"""

from __future__ import annotations

import json

if __package__:
    from . import harness
    from .test_e3 import decision_trace
    from .test_m5_village import M5VillageExperiment
else:
    import harness
    from test_e3 import decision_trace
    from test_m5_village import M5VillageExperiment

from orchestrator.decisions import DecisionLog  # noqa: E402  (test_m5_village puts the repo root on sys.path)
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS  # noqa: E402


class OracleReplay:
    """Replays the oracle run with no model and compares decisions and world hashes with the recorded ones."""

    def __init__(self, args):
        """Keep parsed options: the oracle's directory name, variant, end tick and epoch turns."""
        self._args = args
        self._oracle = args.runs_dir / args.oracle / "orchestrator"

    def run(self) -> bool:
        """Replay to the oracle's end, print each check, and return true on PASS."""
        source = DecisionLog(self._oracle / "decisions.jsonl")
        recorded_hashes = json.loads((self._oracle / "hashes.json").read_text(encoding="utf-8"))
        replay, _, report = M5VillageExperiment(self._args).run_once(
            f"{self._args.oracle}_oracle_replay", None, source, end_tick=self._args.end, min_gap=THINKING_TICKS,
            rest=NIGHT_REST, epoch_turns=self._args.epoch_turns)
        checks = {
            "zero model calls": replay.stats.model_calls == 0 and report["stopped"] is None,
            f"identical decision trace ({len(source)} decisions)": decision_trace(replay.log) == decision_trace(source),
            f"identical world hashes ({len(recorded_hashes)})": replay.hashes == recorded_hashes,
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} oracle replay of {self._args.oracle}: {replay.stats.replayed} decisions "
              f"reused, {replay.stats.model_calls} model calls, {len(replay.hashes)} hashes")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse options, replay the oracle, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--oracle", default="oracle_sabotage_1024", help="the recorded run's directory name")
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--end", type=int, default=48_000, help="the recorded run's end game time")
    parser.add_argument("--epoch-turns", type=int, default=8, help="as the run was recorded")
    return 0 if OracleReplay(parser.parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
