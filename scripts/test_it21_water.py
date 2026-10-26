"""IT-21: agents in water stay afloat, swim to a bank, and replay exactly ($0).

Procedure: on a fresh village_sabotage clone, put a role bot at the bottom of the river away from the bridge and the
port, and leave it idle for 600 ticks: it must stay alive, its observation must say it is in water, and no `death`
event may be logged. Snapshot it there and clone the world. Then `move_to` a point on the west bank through the
model-facing translator: it must succeed and end on land. A second server restored from the snapshot must issue the
same `move_to` at the same tick and reproduce every world hash (sampled every 20 ticks) of the first; a `kill`
afterwards must log one `death` event with its cause and position.

Pass: every printed check is ok. Artifacts stay under --runs-dir/it21_water; only --port + 0..7 are used.
Estimate: about two minutes (unmeasured).
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import harness
    from .test_it15_skills import port_options
    from .village_session import VillageSession
else:
    import harness
    from test_it15_skills import port_options
    from village_session import VillageSession

from orchestrator.llm import ToolCall  # noqa: E402
from orchestrator.render import ObservationRenderer  # noqa: E402
from orchestrator.runtime import POLL  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import ToolTranslator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOT = "envoy"
RIVER_BED = (0.5, -63, 100.5)  # the river spans x -6..6, y -63..-61; z 100 is far from the bridge and the port
WEST_BANK = (-12, -60, 100)
IDLE_TICKS = 600
SWIM_TICKS = 400
HASH_EVERY = 20


class IT21WaterExperiment:
    """Own a village session and a fork of it; check survival, the swim out, and exact replay."""

    def __init__(self, args):
        """Validate ports and load the sabotage manifest without starting anything."""
        self.args = args
        self.options = port_options(args)
        self.directory = args.runs_dir / "it21_water"
        self.fork = args.runs_dir / "it21_water_fork"
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.region = json.loads((ROOT / "scenario/village_sabotage.json").read_text(
            encoding="utf-8"))["layout"]["fixture_hash_region"]
        self.translator = ToolTranslator()
        self.checks: list[dict] = []
        self.client = None

    def _check(self, label: str, passed: bool, detail: str = "") -> None:
        """Print and retain one check with the evidence needed to diagnose it."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if detail else ""), flush=True)

    def _step(self, ticks: int) -> None:
        """Advance lockstep time."""
        self.client.request("step", {"n": ticks})

    def _now(self) -> int:
        """The current game time."""
        return harness.game_time(self.client.request("ping"))

    def _deaths(self, directory: Path) -> list[dict]:
        """The run's `death` events so far (the log is flushed on save)."""
        self.client.request("save_snapshot")
        path = directory / "world/simharness/events.jsonl"
        return [event for event in (json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
                                    if line.strip()) if event.get("type") == "death"]

    def _swim(self) -> tuple[dict, list[int]]:
        """Send the move_to and step to its end or SWIM_TICKS, sampling world hashes; return status and hashes."""
        action = self.translator.translate(BOT, ToolCall("move_to", {"x": WEST_BANK[0], "y": WEST_BANK[1],
                                                                     "z": WEST_BANK[2]}))
        self.client.request(action.command, action.args)
        hashes, status = [], {}
        for _ in range(SWIM_TICKS // HASH_EVERY):
            self._step(HASH_EVERY)
            hashes.append(self.client.request("hash", {"region": self.region})["total"])
            status = self.client.request("skill_status", {"name": BOT})
        return status, hashes

    def _idle_in_water(self) -> None:
        """Place the bot on the river bed, leave it idle, and check it survives and knows it is in water."""
        self.client.request("skill_cancel", {"name": BOT})
        self.client.request("command", {"command": f"tp {BOT} {RIVER_BED[0]} {RIVER_BED[1]} {RIVER_BED[2]} 0 0"})
        self._step(POLL)
        if self._now() % POLL:
            self._step(POLL - self._now() % POLL)
        observation = self.client.request("observe", {"name": BOT, "include_terrain": False})
        self._check("observation reports water", observation["self"].get("in_water") is True)
        self._check("prompt says 'You are in water.'", "You are in water." in ObservationRenderer().render(observation))
        self._step(IDLE_TICKS)
        state = self.client.request("bot_state", {"name": BOT})
        self._check(f"alive after {IDLE_TICKS} idle ticks in water", state.get("health", 0) > 0,
                    f"health {state.get('health')}, pos {state.get('pos')}")
        self._check("no death event", not self._deaths(self.directory))

    def run(self) -> bool:
        """Run the factual session with a snapshot in water, resume the snapshot, and compare."""
        session = VillageSession(self.directory, self.args.runs_dir / "village_sabotage", self.manifest, self.options)
        session.clone()
        factual, resumed, start = [], [], None
        try:
            with session:
                self.client = session.client
                self._idle_in_water()
                self.client.request("save_snapshot")
                harness.clone_apfs(self.directory / "world", self.fork / "world")
                (self.fork / "orchestrator").mkdir()
                start = self._now()
                status, factual = self._swim()
                state = self.client.request("bot_state", {"name": BOT})
                self._check("move_to from the river succeeds", status.get("state") == "succeeded",
                            json.dumps(status)[:300])
                self._check("ends on the west bank, out of water",
                            state["pos"][0] < -6 and not self.client.request(
                                "observe", {"name": BOT, "include_terrain": False})["self"].get("in_water"),
                            f"pos {state['pos']}")
            fork = VillageSession(self.args.runs_dir / "it21_water_resumed", self.fork, self.manifest, self.options,
                                 fresh=False)
            fork.clone()
            with fork:
                self.client = fork.client
                self._check("resumed at the snapshot tick", self._now() == start, f"{self._now()} vs {start}")
                _, resumed = self._swim()
                self.client.request("command", {"command": f"kill {BOT}"})  # after the comparison window
                self._step(1)
                deaths = self._deaths(self.args.runs_dir / "it21_water_resumed")
                self._check("a death is logged as an engine event",
                            len(deaths) == 1 and deaths[0].get("bot") == BOT and deaths[0].get("cause")
                            and deaths[0].get("pos"), json.dumps(deaths)[:300])
            differing = [index for index, (a, b) in enumerate(zip(factual, resumed)) if a != b]
            self._check(f"replay from the snapshot: {len(factual)} hashes equal", bool(factual) and resumed == factual,
                        f"{len(differing)} differ, first at sample {differing[0]}" if differing else "")
        except (harness.HarnessError, OSError, ValueError, KeyError) as error:
            self._check("gate completed", False, str(error))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        report = {"passed": passed, "checks": self.checks}
        (self.directory / "it21_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-21: agents in water", flush=True)
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup, parse the shared gate options, and return zero only for a passing IT-21."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if IT21WaterExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-21: water experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
