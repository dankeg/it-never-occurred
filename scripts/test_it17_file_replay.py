"""IT-17: replay E3's workshop from decisions.jsonl, not the in-memory decision index.

Procedure: record E3 run A, load its file into a fresh DecisionLog, and replay with no model in a fresh world.
Pass: zero model calls, identical decision traces and world hashes. Negative control: change one byte of one
decision's digest in a copy; replay must raise ReplayDiverged and write diverged.json naming that agent and tick.
Revert check (2026-10-18): fails at 9f4f3f5^ (remembered arguments in key order).
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import harness
    from .agent_scenario import WorkshopScenario
    from .test_e3 import RUN_TICKS, E3Experiment, decision_trace
else:
    import harness
    from agent_scenario import WorkshopScenario
    from test_e3 import RUN_TICKS, E3Experiment, decision_trace

from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.runtime import ReplayDiverged  # noqa: E402


def corrupt_digest(source: Path, destination: Path) -> tuple[str, int]:
    """Copy the log changing exactly one ASCII digest byte; return the damaged decision's agent and tick."""
    data = source.read_bytes()
    offset = 0
    for line in data.splitlines(keepends=True):
        if line.strip():
            record = json.loads(line)
            digest = record["digest"].encode("ascii")
            if not digest:
                raise ValueError("Cannot corrupt an empty decision digest")
            field = line.index(b'"digest"')
            start = line.index(b'"', line.index(b":", field) + 1) + 1
            replacement = b"0" if digest[:1] != b"0" else b"1"
            index = offset + start
            destination.write_bytes(data[:index] + replacement + data[index + 1:])
            return record["agent"], record["game_time"]
        offset += len(line)
    raise ValueError("Cannot corrupt an empty decision log")


def divergence_names(path: Path, agent: str, tick: int) -> bool:
    """Whether the runtime's actual divergence artifact identifies the deliberately damaged decision."""
    if not path.is_file():
        return False
    return any(item.get("agent") == agent and item.get("gameTime") == tick
               for item in json.loads(path.read_text(encoding="utf-8")))


class FileReplayExperiment:
    """Record and replay real workshop worlds, then verify detection of a damaged on-disk log."""

    def __init__(self, args):
        """Retain CLI options without creating directories or starting servers."""
        self.args = args
        self.e3 = E3Experiment(args)
        self.negative = False

    def setup(self) -> None:
        """Refuse to overwrite any artifact directory this experiment will own."""
        for name in ("it17_a", "it17_replay", "it17_corrupt", "it17_bad_decisions.jsonl"):
            path = self.args.runs_dir / name
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")

    def run(self) -> bool:
        """Use E3's fresh-runtime path for the original, file replay and deliberate divergence."""
        self.setup()
        self.recorded = self.e3.run_once("it17_a", WorkshopScenario().client(), None)
        source = self.args.runs_dir / "it17_a/orchestrator/decisions.jsonl"
        self.replayed = self.e3.run_once("it17_replay", None, DecisionLog(source))
        damaged = self.args.runs_dir / "it17_bad_decisions.jsonl"
        agent, tick = corrupt_digest(source, damaged)
        try:
            self.e3.run_once("it17_corrupt", None, DecisionLog(damaged))
        except ReplayDiverged:
            self.negative = divergence_names(
                self.args.runs_dir / "it17_corrupt/orchestrator/diverged.json", agent, tick)
        return self.check()

    def check(self) -> bool:
        """Report separate positive replay invariants and the corruption negative control."""
        checks = {
            "recorded decisions with the scripted model": self.recorded.stats.model_calls > 0,
            "file replay made zero model calls": self.replayed.stats.model_calls == 0,
            "identical decision traces": decision_trace(self.recorded.log) == decision_trace(self.replayed.log),
            "identical nonempty world hashes": (
                self.recorded.hashes == self.replayed.hashes and len(self.recorded.hashes) == RUN_TICKS // 20),
            "negative control: damaged digest identifies agent and tick": self.negative,
        }
        for label, passed in checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-17: replay from the file log")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse the common gate arguments, and run the experiment."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if FileReplayExperiment(args).run() else 1
    except (OSError, ValueError, harness.HarnessError, ReplayDiverged) as error:
        print(f"  BAD {error}\nFAIL IT-17: file replay could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
