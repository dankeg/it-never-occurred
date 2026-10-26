"""IT-22: opt-in night sight, villager testimony, visible loads and exact replay ($0).

For the reviewer to run, never as part of the unit suite. Clone runs/village_sabotage, clear a small test corridor
beside a dwelling villager, and check darkness, placed/held lanterns, daylight, near sight and nine-loaf loads.
Explicitly disabled flags must preserve the default observation byte for byte at the same tick. Replay a short
night segment from a saved fork, comparing observations, testimony and world hashes. Only --port + 0..7 are used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

if __package__:
    from . import harness
    from .test_it15_skills import port_options
    from .village_session import VillageSession
else:
    import harness
    from test_it15_skills import port_options
    from village_session import VillageSession

from orchestrator.render import ObservationRenderer  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OBSERVER = "investigator"
TARGET = "envoy"


class IT22NightExperiment:
    """Own a disposable village clone and compare both observers with a saved exact replay."""

    def __init__(self, args):
        """Load the existing fixture manifest and validate ports without starting a server."""
        self.args = args
        self.options = port_options(args)
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.region = json.loads((ROOT / "scenario/village_sabotage.json").read_text(
            encoding="utf-8"))["layout"]["fixture_hash_region"]
        self.directory = args.runs_dir / "it22_night"
        self.snapshot = args.runs_dir / "it22_night_fork"
        self.client = None
        self.checks: list[dict] = []
        self.npc = ""
        self.origin: list[int] = []
        self._support_placed = False

    def _check(self, label: str, passed: bool, detail: str = "") -> None:
        """Retain and print a check without hiding later failures."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if detail else ""), flush=True)

    def _command(self, command: str) -> None:
        """Change only this probe's disposable clone through logged commands."""
        self.client.request("command", {"command": command})

    def _now(self) -> int:
        """Read elapsed game time independently of the day-time phase."""
        return harness.game_time(self.client.request("ping"))

    def _observe(self) -> dict:
        """Collect the investigating bot's own, engine-filtered view."""
        return self.client.request("observe", {"name": OBSERVER, "include_terrain": False})

    @staticmethod
    def _target(entities: list[dict]) -> dict | None:
        """Find only the target person, not an NPC or another nearby bot."""
        return next((entity for entity in entities if entity.get("name") == TARGET), None)

    def _teleport_target(self, distance: int = 12) -> None:
        """Put the target down the cleared corridor, beyond or within the eight-block cutoff."""
        horizontal, height, depth = self.origin
        self._command(f"tp {TARGET} {horizontal + distance + 0.5} {height} {depth + 0.5} 0 0")

    def _lamp(self, block: str) -> None:
        """Hang a supported lantern above the target, or remove only the lantern without blocking its sightline."""
        horizontal, height, depth = self.origin
        if block == "minecraft:lantern":
            # vanilla setblock fails when nothing changes, so the support is placed once and then stays
            if not self._support_placed:
                self._command(f"setblock {horizontal + 12} {height + 3} {depth} minecraft:stone")
                self._support_placed = True
            block = "minecraft:lantern[hanging=true]"
        self._command(f"setblock {horizontal + 12} {height + 2} {depth} {block}")

    def _sample(self, label: str, visible: bool) -> dict:
        """Check the same sight condition for a bot and a scheduled, stationary villager."""
        since = self._now() + 1
        self.client.request("step", {"n": 20})
        observation = self._observe()
        sighting = self._target(observation.get("entities", []))
        self._check(f"{label}: bot sight", (sighting is not None) == visible)
        records = self.client.request("npc_testimony", {"id": self.npc, "since": since})["records"]
        self._check(f"{label}: villager sampled", bool(records))
        if records:
            witnessed = self._target(records[-1].get("entities", []))
            self._check(f"{label}: villager sight", (witnessed is not None) == visible)
        return observation

    def _prepare(self) -> None:
        """Choose a dwelling NPC and clear all occlusion and emitting blocks around the test path."""
        for definition in self.manifest.data["npcs"]:
            state = self.client.request("npc_state", {"id": definition["id"]})
            if state.get("trip") is None:
                self.npc = definition["id"]
                self.origin = [math.floor(coordinate) for coordinate in state["pos"]]
                break
        if not self.npc:
            raise ValueError("The fixture has no dwelling villager for the night probe")
        horizontal, height, depth = self.origin
        self._command(f"fill {horizontal - 6} {height} {depth - 6} "
                      f"{horizontal + 18} {height + 8} {depth + 6} minecraft:air")
        self._command(f"fill {horizontal - 6} {height - 1} {depth - 6} "
                      f"{horizontal + 18} {height - 1} {depth + 6} minecraft:gravel")
        for name in (OBSERVER, TARGET):
            self.client.request("skill_cancel", {"name": name})
        self._command(f"tp {OBSERVER} {horizontal + 0.5} {height} {depth + 1.5} 0 0")
        self._teleport_target()
        self._command(f"clear {TARGET}")
        self._command("time set 18000")
        self.client.request("step", {"n": 20})
        self._observe()
        baseline = self._observe()
        self.client.request("config_set", {"night_sight": False, "visible_loads": {}})
        disabled = self._observe()
        self._check("flags off: byte-identical default observation",
                    json.dumps(baseline, separators=(",", ":")) == json.dumps(disabled, separators=(",", ":")))
        self._check("flags off: distant dark target remains visible", self._target(disabled["entities"]) is not None)
        self.client.request("config_set", {"night_sight": True, "visible_loads": {"minecraft:bread": 9}})

    def _conditions(self) -> None:
        """Check night, light, inventory-band and day conditions against both admitted observers."""
        self._sample("unlit at twelve blocks at night", False)
        self._lamp("minecraft:lantern")
        self._sample("under a placed lantern", True)
        self._lamp("minecraft:air")
        self._command(f"item replace entity {TARGET} weapon.mainhand with minecraft:lantern")
        self._sample("holding a lantern", True)
        self._command(f"clear {TARGET}")
        self._sample("after the held lantern is removed", False)
        self._teleport_target(6)
        self._sample("unlit within eight blocks", True)
        self._teleport_target()
        self._command("time set 1000")
        self._sample("unlit by day", True)
        self._command(f"give {TARGET} minecraft:bread 8")
        observation = self._sample("eight bread by day", True)
        self._check("eight bread is not a sack", "load" not in (self._target(observation["entities"]) or {}))
        self._command(f"give {TARGET} minecraft:bread 1")
        observation = self._sample("nine bread by day", True)
        load = {"item": "minecraft:bread", "count_band": "sack"}
        self._check("nine bread reveals only a sack", (self._target(observation["entities"]) or {}).get("load") == load)
        self._check("prompt renders the sack", "carrying a sack of bread" in ObservationRenderer().render(observation))
        records = self.client.request("npc_testimony", {"id": self.npc, "since": self._now() - 19})["records"]
        self._check("testimony records the same count-free load", bool(records)
                    and (self._target(records[-1]["entities"]) or {}).get("load") == load)
        self._command("time set 18000")
        self._sample("an unlit sack is still invisible at night", False)

    def _segment(self) -> list[dict]:
        """Repeat a short scripted night segment, retaining hashes and both observers' exact evidence."""
        samples = []
        for block in ("minecraft:lantern", "minecraft:air", "minecraft:lantern"):
            self._lamp(block)
            since = self._now() + 1
            self.client.request("step", {"n": 20})
            observation = self._observe()
            testimony = self.client.request("npc_testimony", {"id": self.npc, "since": since})
            samples.append({"hash": self.client.request("hash", {"region": self.region})["total"],
                            "observation": observation, "testimony": testimony})
        return samples

    def run(self) -> bool:
        """Run the factual clone, restore its snapshot into another clone, and compare the complete segment."""
        session = VillageSession(self.directory, self.args.runs_dir / "village_sabotage", self.manifest, self.options)
        session.clone()
        try:
            with session:
                self.client = session.client
                self._prepare()
                self._conditions()
                self.client.request("save_snapshot")
                harness.clone_apfs(self.directory / "world", self.snapshot / "world")
                (self.snapshot / "orchestrator").mkdir()
                start = self._now()
                factual = self._segment()
            resumed = VillageSession(self.args.runs_dir / "it22_night_resumed", self.snapshot, self.manifest,
                                     self.options, fresh=False)
            resumed.clone()
            with resumed:
                self.client = resumed.client
                self._check("fork restores the exact tick", self._now() == start)
                replay = self._segment()
            self._check("short night replay: observations, testimony and hashes match exactly", factual == replay)
        except (harness.HarnessError, OSError, ValueError, KeyError) as error:
            self._check("gate completed", False, str(error))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        (self.directory / "it22_report.json").write_text(
            json.dumps({"passed": passed, "checks": self.checks}, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-22: night sight and visible loads", flush=True)
        return passed


def main(argv: list[str] | None = None) -> int:
    """Run only on explicit invocation, using the standard experiment port/run options."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if IT22NightExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-22: night experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
