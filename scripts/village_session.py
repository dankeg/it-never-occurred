"""One village run's server session, shared by every village runner (C3: the copies had drifted once).

`VillageSession` clones a fixture (`runs/village_<variant>`, a fresh village) or a fork into the run directory, starts
its server, turns lockstep and recording on and restores the bots; a fresh village also gets the exposure log off and
the known structures (a fork inherits both through its snapshot). It also runs a runtime to an end time with the
outcome tracker checked at every `CHECK_EVERY`, saves a fork, and renders each role's first view (narrated runs). The
runners keep their own logic: what to run, sources, overrides, interventions and outcomes. Optional `Checkpoints` save
the same world and agent state alongside outcome times at aligned ticks for resuming or starting decision forks.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
import sys

if __package__:
    from . import harness
    from .setup_server import setup_server
else:
    import harness
    from setup_server import setup_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.render import ObservationRenderer  # noqa: E402
from orchestrator.runtime import SimulationRuntime  # noqa: E402
from orchestrator.scenario import OutcomeTracker, PredicateEvaluator, ScenarioManifest  # noqa: E402

CHECK_EVERY = 1_200


class Checkpoints:
    """Save a run's world, agents and outcome times at aligned intervals, without replacing earlier saves."""

    def __init__(self, every: int):
        """Require a positive interval on the outcome-check grid."""
        if every <= 0 or every % CHECK_EVERY:
            raise ValueError(f"checkpoint interval must be a positive multiple of {CHECK_EVERY}")
        self.every = every

    def save(self, session: VillageSession, runtime: SimulationRuntime, tracker: OutcomeTracker, tick: int) -> None:
        """Save an aligned tick once, using the same APFS world clone and runtime state as a fork."""
        if tick % self.every:
            return
        directory = session.directory / "checkpoints" / str(tick)
        if directory.exists():
            return
        directory.mkdir(parents=True)
        session.save_fork(directory, runtime)
        (directory / "orchestrator" / "first_true.json").write_text(json.dumps(tracker.first_true) + "\n",
                                                                    encoding="utf-8")

    @staticmethod
    def carry_prefix(checkpoint: Path, directory: Path) -> list[dict]:
        """Copy the checkpointed run's decisions and trajectory before the checkpoint into a run resumed from it, so the
        resumed run's logs are complete; return the world hashes up to the checkpoint (the hash at its tick is recorded
        before it is saved) for its runtime."""
        tick, target = int(checkpoint.name), directory / "orchestrator"
        source = checkpoint.parent.parent / "orchestrator"
        for name, key in (("decisions.jsonl", "game_time"), ("trajectory.jsonl", "t")):
            path = source / name
            if path.exists():
                rows = [line for line in path.read_text(encoding="utf-8").split("\n")  # not splitlines():
                        if line.strip() and json.loads(line)[key] < tick]       # agents' text may hold U+2028
                (target / name).write_text("".join(row + "\n" for row in rows), encoding="utf-8")
        hashes = source / "hashes.json"
        return ([entry for entry in json.loads(hashes.read_text(encoding="utf-8")) if entry["gameTime"] <= tick]
                if hashes.exists() else [])

    @staticmethod
    def latest(directory: Path, tick: int | None = None) -> Path:
        """Return the latest checkpoint at or before `tick` (any tick when omitted), or refuse an empty run."""
        root = directory / "checkpoints"
        candidates = [path for path in root.iterdir() if path.is_dir() and path.name.isdecimal()
                      and (tick is None or int(path.name) <= tick)] if root.is_dir() else []
        if not candidates:
            limit = f" at or before tick {tick}" if tick is not None else ""
            raise ValueError(f"No checkpoint in {directory}{limit}")
        return max(candidates, key=lambda path: int(path.name))


class VillageSession:
    """Own one cloned village run, its server, and its lockstep client for a single context.

    `clone()` first (a runner may read the clone, e.g. a fork's saved agents, before the server starts), then enter.
    """

    def __init__(self, directory: Path, source: Path, manifest: ScenarioManifest, server_options: dict, *,
                 fresh: bool = True):
        """Keep the run paths, the scenario and the server options (ports included)."""
        self.directory = directory
        self._source = source
        self._manifest = manifest
        self._server_options = server_options
        self._fresh = fresh
        self._server = None
        self.client = None
        self.start = None

    def clone(self) -> None:
        """Refuse an existing run and clone its source, assembling server files for a bare checkpoint."""
        if self.directory.exists():
            raise FileExistsError(f"Refusing to overwrite {self.directory}")
        if not self._source.exists():  # report a missing fork or fixture as such, not as a server-setup failure
            raise FileNotFoundError(errno.ENOENT, "Nothing to clone", str(self._source))
        if not self._fresh and not (self._source / "fabric-server-launch.jar").exists():
            setup_server(self.directory.name, self.directory.parent)
            for name in ("world", "orchestrator"):
                harness.clone_apfs(self._source / name, self.directory / name)
        else:
            harness.clone_apfs(self._source, self.directory)

    def __enter__(self) -> VillageSession:
        """Start and restore the cloned run's lockstep server using the runner's server options."""
        self._server = harness.Server(self.directory, **self._server_options)
        server = self._server.__enter__()
        try:
            self._setup(server.client)
        except BaseException:
            self._server.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc) -> None:
        """Stop the owned server without suppressing an exception from the run."""
        self._server.__exit__(*exc)

    def kill(self) -> None:
        """Terminate this session's server now (from another thread), so the run using it fails and unwinds."""
        if self._server is not None:
            self._server.kill()

    def _setup(self, client) -> None:
        """Enable recording and restore bots, configuring exposure and structures only for a fresh village."""
        self.client = client
        self.start = harness.enable_lockstep(client, recording=True)
        client.request("restore_bots")
        if self._fresh:
            client.request("config_set", {"exposure_log": 0})
            client.request("structures_set", {"structures": self._manifest.structures()})

    def first_views(self) -> dict[str, str]:
        """Render each role's initial observation in manifest order without terrain."""
        renderer = ObservationRenderer()
        return {role["name"]: renderer.render(self.client.request("observe", {"name": role["name"],
                                                                              "include_terrain": False}))
                for role in self._manifest.roles}

    def run_checked(self, runtime: SimulationRuntime, tracker: OutcomeTracker, now: int, end: int,
                    checkpoints: Checkpoints | None = None) -> int:
        """Run to `end`, checking outcomes and optionally saving aligned checkpoints after each step."""
        evaluator = PredicateEvaluator(self.client, self.directory / "world/simharness/events.jsonl")
        while now < end:
            now = min(end, (now // CHECK_EVERY + 1) * CHECK_EVERY)
            runtime.run(now)
            tracker.check(evaluator, now)
            if checkpoints is not None:
                checkpoints.save(self, runtime, tracker, now)
        return now

    def save_fork(self, fork: Path, runtime: SimulationRuntime) -> None:
        """Flush and clone the world into a fork, then save the orchestrator's state beside it."""
        self.client.request("save_snapshot")
        harness.clone_apfs(self.directory / "world", fork / "world")
        (fork / "orchestrator").mkdir()
        runtime.save_state(fork / "orchestrator" / "fork_state.json")
