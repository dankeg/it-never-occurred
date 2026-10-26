"""Counterfactual forks of one Jubilee decision: the village fork runner with the town's scenario and outcomes.

Arms, samples, checkpoints and names are exactly `fork_decision.py`'s (`control`, `resample`, `brief`, `mute`, and
the rest). What differs: the factual run's own `scenario.json` (with its condition) is the manifest; the fixture is
the built town; there is no fixed night rest; the perception log is on; and each arm's outcome is the town's state at
`t + H`: the great hall (fresh bread on the tables, lights), each agent's plan-relevant engine acts since `t` (bakes,
bread moved, purchases, documents written, changes in the hall), emeralds, and the physical leak evidence each agent
perceived (no model: speech is not judged here).

Usage: `python3.11 scripts/jubilee_fork.py --run R --agent baker --tick T --arms control,resample,mute
--mute baker=innkeeper --samples 3 --horizon 3000 --from-checkpoint --model gpt-6.1-sol ...`
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

if __package__:
    from . import harness
    from .fork_decision import DecisionFork, fork_parser
    from .jubilee_leaks import JubileeLeaks
    from .jubilee_outcomes import JubileeOutcomes
    from .run_village import model_client
else:
    import harness
    from fork_decision import DecisionFork, fork_parser
    from jubilee_leaks import JubileeLeaks
    from jubilee_outcomes import JubileeOutcomes
    from run_village import model_client

from orchestrator.jubilee import JubileeManifest  # noqa: E402
from orchestrator.runtime import THINKING_TICKS, SimulationRuntime  # noqa: E402
from orchestrator.scenario import OutcomeTracker  # noqa: E402

FIXTURE = "town_jubilee"


class NullExposure:
    """The Jubilee keeps no evidence-item record (its perception log has everything); the fork runner's interface."""

    def observe(self, agent: str, observation: dict) -> None:
        """Nothing to learn."""

    def read(self, agent: str, text: str, game_time: int) -> None:
        """Nothing to learn."""

    def items(self, agent: str, end: int) -> list[str]:
        """No evidence items."""
        return []

    def to_json(self) -> dict:
        """Nothing to save."""
        return {}


class JubileeOutcomeInputs:
    """Reads a Jubilee arm's outcome at its end: the hall, acts since the fork, emeralds, and physical leaks."""

    ACTS = {"bake": ("bot",), "purchase": ("buyer",), "doc_write": ("author",)}

    def __init__(self, manifest: JubileeManifest, fork_tick: int):
        """Keep the manifest and the fork's tick (acts are counted from it)."""
        self._manifest, self._tick = manifest, fork_tick
        self._hall = JubileeOutcomes(manifest.data)
        self._names = [role["name"] for role in manifest.roles]

    def _acts(self, events: Path) -> dict[str, dict[str, int]]:
        """Per agent, the plan-relevant engine acts since the fork."""
        counts = {name: Counter() for name in self._names}
        for line in Path(events).read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("gameTime", 0) < self._tick:
                continue
            kind = event.get("type")
            for field in self.ACTS.get(kind, ()):
                if event.get(field) in counts:
                    counts[event[field]][kind] += 1
            if kind == "transfer" and event.get("item") == "minecraft:bread":
                for field in ("from", "to"):
                    if event.get(field) in counts:
                        counts[event[field]]["bread_moved"] += event.get("count", 1)
            if kind == "block_change" and event.get("actor") in counts:
                counts[event["actor"]]["block_change"] += 1
        return {name: dict(sorted(counter.items())) for name, counter in counts.items()}

    def collect(self, client, events: Path, tracker: OutcomeTracker, exposure: NullExposure, end: int) -> dict:
        """The arm's outcome vector inputs at `end` (flushes the event log first)."""
        client.request("save_snapshot")
        emeralds = {name: sum(entry["count"] for entry in client.request("bot_state", {"name": name})["inventory"]
                              if entry["item"] == "minecraft:emerald") for name in self._names}
        leaks = JubileeLeaks(Path(events).parents[2], manifest=self._manifest.data)
        evidence = leaks.evidence(before=end)
        return {"hall": asdict(self._hall.hall(client)), "acts": self._acts(events), "emeralds": emeralds,
                "leaks": leaks.summary(evidence), "mayor_unaware": leaks.mayor_kept_unaware(evidence)}


class JubileeFork(DecisionFork):
    """A decision fork of a Jubilee run."""

    def _load_manifest(self) -> JubileeManifest:
        """The factual run's own manifest, so the arms keep its condition."""
        data = json.loads((self._args.runs_dir / self._args.run / "scenario.json").read_text(encoding="utf-8"))
        return JubileeManifest(data)

    def _fixture(self) -> Path:
        """The built town."""
        return self._args.runs_dir / FIXTURE

    def _begin(self, client, now: int) -> int:
        """As `run_jubilee.py`: apply the condition's engine settings and setup requests (extra people, scarcer
        stock), then let the town live on to the agent start."""
        if self._manifest.engine_overrides:
            client.request("config_set", self._manifest.engine_overrides)
        for request in self._manifest.setup_requests():
            client.request(request["request"], request["args"])
        start = self._manifest.data["jubilee"].get("agent_start_game_time", 0)
        if now < start:
            now = harness.game_time(client.request("step", {"n": start - now}))
        return now

    def _exposure(self, known: dict | None = None) -> NullExposure:
        """No evidence-item record."""
        return NullExposure()

    def _outcome_inputs(self) -> JubileeOutcomeInputs:
        """The town's outcome at the end of each arm, acts counted from the fork."""
        return JubileeOutcomeInputs(self._manifest, self._args.tick)

    def _tracker(self) -> OutcomeTracker:
        """No running predicates; outcomes are read at the end."""
        return OutcomeTracker({})

    def _runtime(self, client, agents, directory: Path, model, override,
                 sample_seed: int | None = None) -> SimulationRuntime:
        """The Jubilee runtime as the factual run was configured: no fixed rest, perception log on."""
        return SimulationRuntime(client, agents, model, directory / "orchestrator", command_error=harness.CommandError,
                                 source=self._factual, hash_region=self._manifest.data["layout"]["region"],
                                 hash_every=1_200, max_model_calls=self._args.max_calls, sample_seed=sample_seed,
                                 min_gap=THINKING_TICKS, rest=None, override=override, perception_log=True)

    def _model(self):
        """A live client (the Jubilee has no scripted village policy)."""
        if self._args.model == "scripted":
            raise ValueError("Jubilee forks need a live model (--model)")
        return model_client(self._args)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the arms of one Jubilee decision point, and print a short summary."""
    harness.stop_on_sigterm()
    report = JubileeFork(fork_parser(__doc__).parse_args(argv)).run()
    print(json.dumps({"point": report["point"], "stopped": report["stopped"],
                      "arms": {arm: {key: outcome[key] for key in ("calls", "decisions", "replayed", "mayor_unaware")}
                               for arm, outcome in report["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
