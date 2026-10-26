"""Build decision records by exact, model-free replay, or verify state documents offline.

For older prompts, extract their code with
`git archive <commit> orchestrator scenario | tar -x -C <dir>`, then run in that directory:
`python3.11 -c 'import json; from orchestrator.scenario import ScenarioManifest; from
orchestrator.tools import TOOL_DEFINITIONS; print(json.dumps({"system": {agent.name:
agent.system_prompt() for agent in ScenarioManifest.load("scenario/village_sabotage.json").agents()},
"tools": TOOL_DEFINITIONS}))' > <ctx.json>` (join the Python command onto one line).
Pass that file as --prompt-context; --states-only never starts a server or writes output.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from itertools import zip_longest
import json
from pathlib import Path
import sys

if __package__:
    from . import harness
    from .test_m5_village import CHECK_EVERY
else:
    import harness
    from test_m5_village import CHECK_EVERY

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.agent import Agent
from orchestrator.candidates import Candidate, CandidateGenerator, EngineDryRun, MAX_CANDIDATES, same_choice
from orchestrator.decisions import Decision, DecisionLog, prompt_digest
from orchestrator.llm import Turn, arguments_json
from orchestrator.perception import (DecisionContext, EvidenceExposure, PerceptionFeatures,
                                    PerceptionTap, VillagerTracks)
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS, DecisionOverride, SimulationRuntime
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import tool_definitions


class StateDocuments:
    """Reconstruct each agent's logged prompt epochs and verify their original digests."""

    def __init__(self, decisions: Sequence[Decision], system: dict[str, str], tools: list[dict]):
        """Build independent JSON-ready states in game-time and agent order using the supplied context."""
        self.states: dict[tuple[str, int], dict] = {}
        self.verified: dict[tuple[str, int], bool] = {}
        histories: dict[str, list[Turn]] = {}
        for decision in sorted(decisions, key=lambda item: (item.game_time, item.agent)):
            name, user = decision.agent, decision.user_prompt
            if user.startswith("Your notes:"):
                histories[name] = []
            history = histories.setdefault(name, [])
            key = (name, decision.game_time)
            self.states[key] = {
                "system": system[name],
                "history": [{"user": turn.user, "tool": turn.call.name if turn.call else None,
                             "arguments": arguments_json(turn.call) if turn.call else None,
                             "result": turn.result} for turn in history],
                "user": user,
            }
            self.verified[key] = (prompt_digest(decision.model, system[name], user, tools, history)
                                  == decision.digest)
            history.append(Turn(user, decision.tool_call, decision.outcome))

    @property
    def verified_count(self) -> int:
        """Return how many state documents reproduce the recorded prompt digest."""
        return sum(self.verified.values())


class DecisionCollector:
    """Capture candidates and observations during logged polls, before any actions are applied."""

    def __init__(self, decisions: Sequence[Decision], generator: CandidateGenerator, agents: Sequence[Agent]):
        """Index logged points and the same live agents supplied to the runtime."""
        self._points = {(decision.agent, decision.game_time) for decision in decisions}
        self._generator = generator
        self._agents = {agent.name: agent for agent in agents}
        self.observations: dict[tuple[str, int], dict] = {}
        self.candidates: dict[tuple[str, int], list[Candidate]] = {}
        self.views: dict[int, dict[str, dict]] = {}

    def on_observe(self, agent: str, observation: dict) -> None:
        """Generate only at logged points, using the actor's current memory and unmodified view."""
        key = (agent, observation["gameTime"])
        if key in self._points:
            self.candidates[key] = self._generator.generate(agent, observation, self._agents[agent].memory)
            self.observations[key] = deepcopy(observation)

    def finish_poll(self, game_time: int, observations: dict[str, dict]) -> None:
        """Keep every agent's view from the completed poll for later feature computation."""
        self.views[game_time] = deepcopy(observations)


class DecisionDatasetBuilder:
    """Load a recorded run, collect a model-free replay, and report dataset exactness."""

    def __init__(self, args):
        """Load the manifest, source decisions and either current or historical prompt context."""
        self._args = args
        self._manifest = ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario"
                                               / f"village_{args.variant}.json")
        self._recorded = args.runs_dir / args.run
        source_path = self._recorded / "orchestrator/decisions.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self._source = DecisionLog(source_path)
        self._decisions = self._source.decisions()
        if args.prompt_context is not None:
            context = json.loads(args.prompt_context.read_text(encoding="utf-8"))
        else:
            context = {"system": {agent.name: agent.system_prompt()
                                  for agent in self._manifest.agents(args.epoch_turns)},
                       "tools": tool_definitions(daily_hunger=self._manifest.data.get("daily_hunger", False))}
        self._states = StateDocuments(self._decisions, context["system"], context["tools"])

    @staticmethod
    def record(run: str, decision: Decision, state: dict, state_verified: bool, observation: dict,
               candidates: Sequence[Candidate], features: PerceptionFeatures, context: DecisionContext) -> dict:
        """Assemble one JSON-ready record without a server, matching the first equivalent candidate."""
        frontier = decision.tool_call
        index = next((index for index, candidate in enumerate(candidates)
                      if same_choice(candidate.call, frontier)), None)
        return {"run": run, "agent": decision.agent, "game_time": decision.game_time,
                "state": state, "state_verified": state_verified, "observation": observation,
                "candidates": [{"call": asdict(candidate.call), "label": candidate.label,
                                "distance": float(candidate.distance), "target": list(candidate.target),
                                "features": features.features(candidate, context)} for candidate in candidates],
                "frontier": asdict(frontier) if frontier is not None else None,
                "frontier_index": index, "representable": index is not None,
                "free_text": decision.tool in {"speak", "write_book", "write_sign"}, "split": None}

    @staticmethod
    def compare_hashes(recorded: Sequence[dict], replayed: Sequence[dict], through: int) -> dict:
        """Compare complete hash samples through the replay endpoint, including missing or extra samples."""
        expected = [sample for sample in recorded if sample["gameTime"] <= through]
        actual = [sample for sample in replayed if sample["gameTime"] <= through]
        for left, right in zip_longest(expected, actual):
            if left != right:
                return {"matched": False, "first_mismatch": min(sample["gameTime"]
                        for sample in (left, right) if sample is not None)}
        return {"matched": True, "first_mismatch": None}

    @staticmethod
    def compare_outcomes(recorded: Sequence[Decision], replayed: Sequence[Decision]) -> list[dict]:
        """Report differing outcomes as well as missing or unexpected replay decisions."""
        expected = {(decision.agent, decision.game_time): decision.outcome for decision in recorded}
        actual = {(decision.agent, decision.game_time): decision.outcome for decision in replayed}
        mismatches = []
        for agent, game_time in sorted(expected.keys() | actual.keys(), key=lambda key: (key[1], key[0])):
            key = (agent, game_time)
            if key not in expected or key not in actual or expected[key] != actual[key]:
                mismatches.append({"agent": agent, "game_time": game_time,
                                   "recorded": expected.get(key), "replayed": actual.get(key)})
        return mismatches

    @staticmethod
    def summary(records: Sequence[dict], hash_check: dict, outcome_mismatches: Sequence[dict]) -> dict:
        """Summarize coverage, candidate counts and all exactness checks; empty shares are zero."""
        count = len(records)
        represented = sum(record["representable"] for record in records)
        non_free = [record for record in records if not record["free_text"]]
        per_tool = {}
        for record in records:
            tool = record["frontier"]["name"] if record["frontier"] is not None else "(none)"
            counts = per_tool.setdefault(tool, {"decisions": 0, "representable": 0})
            counts["decisions"] += 1
            counts["representable"] += record["representable"]
        for counts in per_tool.values():
            counts["share"] = counts["representable"] / counts["decisions"]
        return {"decisions": count, "states_verified": sum(record["state_verified"] for record in records),
                "representable_share": represented / count if count else 0.0,
                "non_free_text_representable_share": (sum(record["representable"] for record in non_free)
                                                       / len(non_free) if non_free else 0.0),
                "per_tool": per_tool,
                "mean_candidates": sum(len(record["candidates"]) for record in records) / count if count else 0.0,
                "candidate_cap_hits": sum(len(record["candidates"]) == MAX_CANDIDATES for record in records),
                "hash_check": hash_check,
                "outcome_mismatches": {"count": len(outcome_mismatches), "first": list(outcome_mismatches[:5])}}

    def run(self) -> int:
        """Verify states offline or write replay records and their summary, failing on inexactness."""
        verified, count = self._states.verified_count, len(self._decisions)
        print(f"{verified} of {count} state documents verified")
        if self._args.states_only:
            return 0 if verified == count else 1
        if not self._decisions:
            raise ValueError("Cannot replay an empty decision log")
        records, hash_check, outcomes = self._replay()
        summary = self.summary(records, hash_check, outcomes)
        name = self._args.replay_name or f"{self._args.run}_dataset_replay"
        output = self._args.out or self._args.runs_dir / name / "decision_dataset.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        Path(str(output) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0 if verified == count and hash_check["matched"] and not outcomes else 1

    def _replay(self) -> tuple[list[dict], dict, list[dict]]:
        """Clone the starting world and collect each logged poll with the village runtime's exact setup."""
        args, manifest = self._args, self._manifest
        name = args.replay_name or f"{args.run}_dataset_replay"
        directory = args.runs_dir / name
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}")
        harness.clone_apfs(args.runs_dir / f"village_{args.variant}", directory)
        times = sorted({decision.game_time for decision in self._decisions})
        exposure = EvidenceExposure(manifest)
        agents = manifest.agents(args.epoch_turns)
        with harness.Server(directory, **harness.server_options(args)) as server:
            client = server.client
            harness.enable_lockstep(client, recording=True)
            client.request("restore_bots")
            client.request("config_set", {"exposure_log": 0})
            client.request("structures_set", {"structures": manifest.structures()})
            generator = CandidateGenerator(manifest, EngineDryRun(client, harness.CommandError))
            collector = DecisionCollector(self._decisions, generator, agents)
            tap = PerceptionTap(client, exposure, collector.on_observe)
            override = DecisionOverride(hold=frozenset(agent.name for agent in agents)) if args.forced else None
            runtime = SimulationRuntime(tap, agents, None, directory / "orchestrator",
                                        command_error=harness.CommandError, source=self._source,
                                        hash_region=manifest.data["layout"]["region"], hash_every=CHECK_EVERY,
                                        min_gap=THINKING_TICKS, rest=NIGHT_REST, override=override)
            for game_time in times:
                runtime.run(game_time + 20)
                collector.finish_poll(game_time, tap.latest)
        tracks = VillagerTracks.load(self._recorded / "world/simharness/npc_observations.jsonl")
        features = PerceptionFeatures(manifest, args.horizon)
        records = []
        for decision in self._decisions:
            key = (decision.agent, decision.game_time)
            context = DecisionContext(decision.agent, decision.game_time, collector.views[decision.game_time],
                                      exposure, tracks)
            records.append(self.record(args.run, decision, self._states.states[key], self._states.verified[key],
                                       collector.observations[key], collector.candidates[key], features, context))
        recorded_hashes = json.loads((self._recorded / "orchestrator/hashes.json").read_text(encoding="utf-8"))
        hash_check = self.compare_hashes(recorded_hashes, runtime.hashes, times[-1] + 20)
        replayed = DecisionLog(directory / "orchestrator/decisions.jsonl").decisions()
        return records, hash_check, self.compare_outcomes(self._decisions, replayed)


def main(argv: list[str] | None = None) -> int:
    """Parse dataset and harness options and delegate to the builder."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--run", required=True, help="the recorded run's directory name under --runs-dir")
    parser.add_argument("--variant", choices=("accident", "sabotage", "accident_daily_life", "sabotage_daily_life"),
                        default="sabotage")
    parser.add_argument("--forced", action="store_true",
                        help="hold every agent to its logged actions (runs recorded by older prompt code)")
    parser.add_argument("--prompt-context", type=Path,
                        help="system prompts and tools of the commit that recorded the run (see above)")
    parser.add_argument("--epoch-turns", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2400)
    parser.add_argument("--replay-name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--states-only", action="store_true", help="verify state documents offline only")
    return DecisionDatasetBuilder(parser.parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
