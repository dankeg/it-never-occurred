"""Probe all five M7b ferry cases on disposable village clones, never the source.

Only explicit CLI execution starts Minecraft. Setup teleports the existing builder
and freight bots and supplies repair materials; port_repair itself executes through
the grounded skill. A checkpoint-aligned mid-river snapshot resumes in a fresh JVM.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

if __package__:
    from . import harness
    from .setup_server import REPO_ROOT, setup_server
else:
    import harness
    from setup_server import REPO_ROOT, setup_server


REGION = {"x0": -14, "y0": -64, "z0": 34, "x1": 14, "y1": -54, "z1": 49}


class FerryProbe:
    """Own the grounded repair, ferry crossing and exact fresh-JVM fork probe."""

    def __init__(self, config: dict, source: Path, scratch: Path):
        """Retain manifest and distinct clone paths without filesystem or JVM effects."""
        self.config = config
        self.source = source.expanduser().resolve()
        self.scratch = scratch.expanduser().absolute()
        self.fork = self.scratch.with_name(self.scratch.name + "_fork")
        self.ferry = config["layout"]["ferry"]
        self.landings = self.ferry["landings"]
        self.rider = "freight"
        self.second_rider = "ferry_second"
        self.client: harness.Client | None = None
        self.hashes_a: list[dict] = []
        self.hashes_c: list[dict] = []
        self._scratch_created = False
        self._fork_created = False
        self.phase = "setup"
        self.report = {"passed": False, "variant": config["variant"], "ferry": self.ferry,
                       "hash_every": harness.HASH_EVERY, "region": REGION, "cases": {}, "errors": []}

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        """Fail the current dependent probe sequence without staging a substitute result."""
        if not condition:
            raise harness.HarnessError(message)

    def _record(self, detail: dict) -> None:
        """Retain one completed spec case and print its capability result."""
        self.report["cases"][self.phase] = {"passed": True, **detail}
        print(f"PASS {self.phase}")

    def _command(self, text: str) -> dict:
        """Run an existing administrative command only on the active scratch JVM."""
        return self.client.request("command", {"command": text})

    @staticmethod
    def _vector(value, label: str) -> list[float]:
        """Require a finite three-coordinate protocol position."""
        if (not isinstance(value, list) or len(value) != 3
                or any(type(coordinate) not in (int, float) or not math.isfinite(coordinate)
                       for coordinate in value)):
            raise harness.ProtocolError(f"Invalid {label}: {value!r}")
        return list(value)

    def _feet(self, landing: str) -> list[float]:
        """Convert a board support block to the exact standing centre above it."""
        horizontal, height, depth = self.landings[landing]["board"]
        return [horizontal + 0.5, height + 1, depth + 0.5]

    def _boat_position(self) -> list[float]:
        """Inspect the registered UUID through the existing regional entity dump."""
        entities = self.client.request("dump_entities", {"region": REGION})["entities"]
        boats = [entity for entity in entities if entity.get("uuid") == self.ferry["boat"]]
        self._require(len(boats) == 1 and boats[0].get("type") == "minecraft:oak_boat",
                      f"Missing or wrong registered boat entity: {boats}")
        position = boats[0]["pos"]
        if not isinstance(position, str):
            raise harness.ProtocolError(f"Invalid dump_entities position: {position!r}")
        return self._vector([float(coordinate) for coordinate in position.split(",")], "boat position")

    def _status(self, name: str) -> dict:
        """Read ordinary skill fields without requiring a ferry-specific status API."""
        status = self.client.request("skill_status", {"name": name})
        if status.get("state") not in {"idle", "running", "succeeded", "failed", "cancelled"}:
            raise harness.ProtocolError(f"Invalid {name} skill status: {status}")
        return status

    def _teleport(self, name: str, position) -> None:
        """Stage an idle bot without interrupting or cancelling an active ferry."""
        self._require(self._status(name)["state"] != "running", f"Cannot stage active bot {name}")
        self._command(f"tp {name} {' '.join(map(str, position))} 0 0")

    def _dry(self, name: str, skill: str, args: dict) -> dict:
        """Use and validate the existing feasibility/estimate envelope."""
        result = self.client.request("skill_start", {
            "name": name, "skill": skill, "args": args, "dry_run": True})
        estimate = result.get("estimate")
        if (type(result.get("feasible")) is not bool or not isinstance(estimate, dict)
                or type(estimate.get("ticks")) is not int or estimate["ticks"] < 0
                or not isinstance(estimate.get("materials"), dict)):
            raise harness.ProtocolError(f"Invalid {skill} dry-run response: {result}")
        return result

    def _start(self, name: str, skill: str, args: dict) -> dict:
        """Require real skill acceptance and retain its standard skillId."""
        result = self.client.request("skill_start", {"name": name, "skill": skill, "args": args})
        self._require(result.get("accepted") is True and result.get("skillId") is not None,
                      f"{name}/{skill} rejected: {result}")
        return result

    def _advance(self, target: int, hashes: list[dict] | None = None) -> None:
        """Advance to an absolute game time, validating every requested 20-tick sample."""
        before = harness.game_time(self.client.request("ping"))
        self._require(target >= before, f"Cannot advance backwards from {before} to {target}")
        while before < target:
            after = min(target, (before // harness.HASH_EVERY + 1) * harness.HASH_EVERY)
            args = {"n": after - before}
            if hashes is not None:
                args.update(hash_every=harness.HASH_EVERY, region=REGION)
            result = self.client.request("step", args)
            if hashes is None:
                if harness.game_time(result) != after:
                    raise harness.ProtocolError(f"step did not advance from {before} to {after}: {result}")
            else:
                hashes.extend(harness.step_hashes(result, before, after))
            before = after

    def probe_damaged_dock(self) -> None:
        """Case 1: require the missing east board to reject a dry-run with no_dock."""
        self.phase = "1_damaged_dock"
        self._require(self._boat_position() == self.landings["east"]["moor"],
                      "Built village must already have its registered boat exactly at the east moor")
        board = " ".join(map(str, self.landings["east"]["board"]))
        self._command(f"execute if block {board} minecraft:air")
        self._teleport(self.rider, self._feet("east"))
        before = harness.game_time(self.client.request("ping"))
        result = self._dry(self.rider, "ferry", {"to": "west"})
        self._require(result["feasible"] is False and result.get("reason") == "no_dock",
                      f"Damaged board did not gate the ferry: {result}")
        self._require(harness.game_time(self.client.request("ping")) == before, "Dry-run advanced time")
        spawn = next(role["spawn"] for role in self.config["roles"] if role["name"] == self.rider)
        self._teleport(self.rider, spawn)
        self._record({"dry_run": result})

    def probe_repair(self) -> None:
        """Case 2: execute builder's actual port_repair, then check freight's duration."""
        self.phase = "2_repair_and_estimate"
        section = self.config["schematics"]["port_repair"]
        path = self.scratch / "world/simharness/schematics/port_repair.json"
        self._require(json.loads(path.read_text(encoding="utf-8"))["blocks"] == section["blocks"],
                      "Port schematic on disk differs from the scenario manifest")
        bill = dict(Counter(block["block"] for block in section["blocks"]))
        self._require(bill == section["bill"] == {"minecraft:oak_planks": 9}, "Unexpected port repair bill")
        self._teleport("builder", section["stand"])
        for item, count in bill.items():
            self.client.request("give", {"name": "builder", "item": item, "count": count})
        args = {"schematic": "port_repair", "origin": section["origin"]}
        dry = self._dry("builder", "place_section", args)
        self._require(dry["feasible"] is True and dry["estimate"]["materials"] == bill,
                      f"Grounded port repair is infeasible: {dry}")
        repair = self._start("builder", "place_section", args)
        deadline = harness.game_time(self.client.request("ping")) + 500
        while harness.game_time(self.client.request("ping")) < deadline:
            self._advance(min(deadline, harness.game_time(self.client.request("ping")) + 5))
            status = self._status("builder")
            self._require(status.get("skillId") == repair["skillId"], "Builder lost the repair skill")
            if status["state"] != "running":
                break
        self._require(status["state"] == "succeeded", f"Grounded port repair did not succeed: {status}")
        self._require(status.get("progress", {}).get("placed") == len(section["blocks"]),
                      f"Repair did not physically place all nine planks: {status}")
        for block in section["blocks"]:
            position = [section["origin"][index] + block[offset]
                        for index, offset in enumerate(("dx", "dy", "dz"))]
            self._command(f"execute if block {' '.join(map(str, position))} {block['block']}")
        self._teleport("builder", section["stand"])
        self._teleport(self.rider, self._feet("east"))
        result = self._dry(self.rider, "ferry", {"to": "west"})
        self.travel_ticks = math.ceil(math.dist(self.landings["east"]["moor"], self.landings["west"]["moor"])
                                      / self.ferry["speed"])
        self.duration = 20 + self.travel_ticks + 20
        self._require(result["feasible"] is True and result["estimate"]["ticks"] == self.duration
                      and result["estimate"]["materials"] == {}, f"Incorrect ferry estimate: {result}")
        self._record({"repair_status": status, "dry_run": result, "expected_ticks": self.duration,
                      "staged_materials": bill})

    def _snapshot_state(self) -> dict:
        """Capture public running status and physical positions without advancing time."""
        return {"gameTime": harness.game_time(self.client.request("ping")),
                "status": self._status(self.rider),
                "rider_pos": self._vector(self.client.request("bot_state", {"name": self.rider})["pos"],
                                          "rider position"), "boat_pos": self._boat_position()}

    def _fork_mid_river(self) -> dict:
        """Start the crossing and clone only its flushed world at a mid-river checkpoint."""
        self.phase = "4_mid_river_fork"
        midpoint = 20 + self.travel_ticks // 2
        current = harness.game_time(self.client.request("ping"))
        self._advance(current + (-(current + midpoint)) % harness.HASH_EVERY)
        self.start_tick = harness.game_time(self.client.request("ping"))
        self.started = self._start(self.rider, "ferry", {"to": "west"})
        self.fork_tick = self.start_tick + midpoint
        self.end_tick = self.start_tick + self.duration
        self.trace_end = ((self.end_tick + harness.HASH_EVERY - 1) // harness.HASH_EVERY) * harness.HASH_EVERY
        self._advance(self.fork_tick, self.hashes_a)
        snapshot = self._snapshot_state()
        status = snapshot["status"]
        self._require(status["state"] == "running" and status.get("skill") == "ferry"
                      and status.get("skillId") == self.started["skillId"]
                      and status.get("startedAt") == self.start_tick and status.get("args") == {"to": "west"},
                      f"Snapshot is not the original running ferry: {status}")
        progress = status.get("progress", {})
        self._require(progress.get("ticks") == midpoint and progress.get("phase") == "crossing",
                      f"Missing interior ferry progress at snapshot: {status}")
        east, west, boat = self.landings["east"]["moor"], self.landings["west"]["moor"], snapshot["boat_pos"]
        fraction = sum((boat[index] - east[index]) * (west[index] - east[index]) for index in range(3))
        fraction /= math.dist(east, west) ** 2
        self._require(0.25 <= fraction <= 0.75 and boat[1] == east[1] and boat[2] == east[2] == west[2]
                      and snapshot["rider_pos"][0] == boat[0] and snapshot["rider_pos"][2] == boat[2],
                      f"Snapshot is not halfway across with its rider: {snapshot}")
        replacement = self.client.request("skill_start", {
            "name": self.rider, "skill": "wait", "args": {"ticks": 1}})
        cancelled = self.client.request("skill_cancel", {"name": self.rider})
        for label, result in (("replacement", replacement), ("mid-river cancellation", cancelled)):
            self._require(result.get("accepted") is False and result.get("reason") == "busy",
                          f"Ferry allowed {label}: {result}")
        self._require(self._snapshot_state() == snapshot, "Refused interruptions changed the running ferry")
        self.client.request("save_snapshot")
        self._require(harness.game_time(self.client.request("ping")) == self.fork_tick,
                      "Snapshot advanced the continuous run")
        harness.clone_apfs(self.scratch / "world", self.fork / "world")
        self._require(harness.game_time(self.client.request("ping")) == self.fork_tick,
                      "Frozen APFS clone advanced the continuous run")
        self.report.update(start_tick=self.start_tick, snapshot_tick=self.fork_tick,
                           end_tick=self.end_tick, trace_end=self.trace_end, snapshot=snapshot,
                           interruption={"replacement": replacement, "cancel": cancelled})
        return snapshot

    def _finish_trip(self, directory: Path, hashes: list[dict]) -> dict:
        """Refuse landing cancellation and require exact, audited successful completion."""
        self._advance(self.end_tick - 1, hashes)
        landing = self._snapshot_state()
        self._require(landing["status"]["state"] == "running"
                      and landing["status"].get("progress", {}).get("phase") == "landing"
                      and landing["status"].get("progress", {}).get("ticks") == self.duration - 1
                      and landing["boat_pos"] == self.landings["west"]["moor"],
                      f"Ferry is not still landing one tick before completion: {landing}")
        cancelled = self.client.request("skill_cancel", {"name": self.rider})
        self._require(cancelled.get("accepted") is False and cancelled.get("reason") == "busy",
                      f"Ferry allowed destination-moor cancellation before completion: {cancelled}")
        self._require(self._snapshot_state() == landing, "Refused landing cancellation changed the ferry")
        self._advance(self.end_tick, hashes)
        finished = self._snapshot_state()
        status = finished["status"]
        self._require(status["state"] == "succeeded" and status.get("skill") == "ferry"
                      and status.get("skillId") == self.started["skillId"]
                      and status.get("startedAt") == self.start_tick and status.get("endedAt") == self.end_tick
                      and status.get("progress", {}).get("ticks") == self.duration
                      and status.get("progress", {}).get("phase") == "landed",
                      f"Ferry did not complete in exactly {self.duration} ticks: {status}")
        self._require(finished["boat_pos"] == self.landings["west"]["moor"]
                      and finished["rider_pos"] == self._feet("west"),
                      f"Ferry/rider did not reach their exact landing centres: {finished}")
        self.client.request("save_snapshot")
        with (directory / "world/simharness/events.jsonl").open(encoding="utf-8") as stream:
            trips = [event for line in stream if line.strip() for event in (json.loads(line),)
                     if event.get("type") == "ferry_trip" and event.get("rider") == self.rider
                     and event.get("startTick") == self.start_tick]
        self._require(len(trips) == 1 and trips[0].get("from") == "east" and trips[0].get("to") == "west"
                      and trips[0].get("endTick") == self.end_tick and trips[0].get("outcome") == "succeeded",
                      f"Missing/incorrect successful ferry_trip event: {trips}")
        self._advance(self.trace_end, hashes)
        return {"finished": finished, "event": trips[0], "landing_cancel": cancelled}

    def probe_second_rider(self) -> None:
        """Case 5: an east-board second rider cannot summon the boat back from west."""
        self.phase = "5_one_physical_boat"
        before = self._boat_position()
        self.client.request("spawn_bot", {"name": self.second_rider, "yaw": 0, "pitch": 0,
                                          **dict(zip(("x", "y", "z"), self._feet("east")))})
        dry = self._dry(self.second_rider, "ferry", {"to": "west"})
        real = self.client.request("skill_start", {
            "name": self.second_rider, "skill": "ferry", "args": {"to": "west"}})
        self._require(dry["feasible"] is False and dry.get("reason") == "no_boat"
                      and real.get("accepted") is False and real.get("reason") == "no_boat",
                      f"East rider could use the west-moored boat: dry={dry}, real={real}")
        self._require(before == self._boat_position() == self.landings["west"]["moor"],
                      "Second rider moved or summoned the boat")
        self._record({"dry_run": dry, "real_start": real})

    def _prepare_clones(self) -> None:
        """Reject existing/nested paths, prepare C's server shell and clone the built village."""
        paths = (self.source, self.scratch.resolve(), self.fork.resolve())
        for index, directory in enumerate(paths):
            for other in paths[index + 1:]:
                if directory == other or directory in other.parents or other in directory.parents:
                    raise ValueError("Source, scratch and fork must be separate, non-nested directories")
        for directory in (self.scratch, self.fork):
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        setup_server(self.fork.name, self.fork.parent)
        self._fork_created = True
        harness.clone_apfs(self.source, self.scratch)
        self._scratch_created = True

    def run(self, options: dict) -> int:
        """Run both isolated JVMs sequentially and retain reports, hashes and clone evidence."""
        try:
            self._prepare_clones()
            options = {**options, "base_seed": self.config["seed"]}
            with harness.Server(self.scratch, **options) as server:
                self.client = server.client
                harness.enable_lockstep(self.client)
                self.client.request("restore_bots")
                self.probe_damaged_dock()
                self.probe_repair()
                snapshot = self._fork_mid_river()
                self.phase = "3_real_crossing"
                continuous = self._finish_trip(self.scratch, self.hashes_a)
                self._record(continuous)
                self.probe_second_rider()
            self.phase = "4_mid_river_fork"
            with harness.Server(self.fork, **options) as server:
                self.client = server.client
                restored_tick = harness.enable_lockstep(self.client)
                self._require(restored_tick == self.fork_tick,
                              f"Fresh JVM restored at {restored_tick}, expected {self.fork_tick}")
                self.client.request("restore_bots")
                restored = self._snapshot_state()
                self._require(restored == snapshot, f"Restored ferry state differs: A={snapshot}, C={restored}")
                resumed = self._finish_trip(self.fork, self.hashes_c)
                self._require(resumed["finished"] == continuous["finished"],
                              "Fresh-JVM final positions/status differ")
            expected = [record for record in self.hashes_a if record["gameTime"] > self.fork_tick]
            times = list(range(self.fork_tick + harness.HASH_EVERY, self.trace_end + 1, harness.HASH_EVERY))
            self._require(bool(times) and [record["gameTime"] for record in expected] == times
                          and [record["gameTime"] for record in self.hashes_c] == times,
                          "Incomplete 20-tick post-fork hash suffix")
            self._require(harness.compare_hashes(expected, self.hashes_c, "ferry mid-river fork"),
                          "Full-component post-fork hashes differ; see A/C hashes.json")
            self._record({"restored": restored, "matching_hashes": len(times), "sample_ticks": times})
            self.report["passed"] = len(self.report["cases"]) == 5
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            message = f"{self.phase}: {type(error).__name__}: {error}"
            self.report["errors"].append(message)
            self.report["cases"][self.phase] = {"passed": False, "error": message}
            print(f"FAIL {message}")
        finally:
            self.client = None
            if self._scratch_created:
                harness.write_hashes(self.scratch / "hashes.json", self.hashes_a)
                (self.scratch / "report.json").write_text(
                    json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if self._fork_created:
                harness.write_hashes(self.fork / "hashes.json", self.hashes_c)
        print(f"{'PASS' if self.report['passed'] else 'FAIL'} ferry probe; source never launched: {self.source}")
        print(f"Scratch retained: {self.scratch}; fresh-JVM fork: {self.fork}")
        return 0 if self.report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Parse an explicit runtime request without starting anything on module import."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="accident")
    parser.add_argument("--scratch", type=Path, help="New APFS clone; defaults to <runs-dir>/probe_ferry_<variant>")
    args = parser.parse_args(argv)
    try:
        path = REPO_ROOT / "scenario" / f"village_{args.variant}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["variant"] != args.variant:
            raise ValueError("Manifest variant differs from --variant")
        source = args.runs_dir / f"village_{args.variant}"
        scratch = args.scratch or args.runs_dir / f"probe_ferry_{args.variant}"
        return FerryProbe(config, source, scratch).run(harness.server_options(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL setup: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
