"""E2 skills: 4,000 ticks and an exact fresh-JVM fork of two running skills."""

from __future__ import annotations

import json

if __package__:
    from . import harness, test_s1 as skills
    from .setup_server import setup_server
else:
    import harness
    import test_s1 as skills
    from setup_server import setup_server


RUN_TICKS = 4000
FORK_TICKS = 1020
SNAPSHOT_SKILLS = {"alice": "move_to", "bob": "place_section"}


def schedule() -> dict[int, list[tuple[str, str, dict]]]:
    """Return a fresh copy of the fixed two-bot skill launch schedule.

    Mined blocks go straight into the inventory, so the schedule has no `collect` after `mine`: since 14dcb6d,
    collecting with no dropped items in sight is correctly rejected (S1 covers `collect` with real drops)."""
    return {
        0: [("alice", "mine", {"x": -24, "y": -60, "z": -11})],
        300: [("alice", "craft", {"item": "minecraft:oak_planks", "count": 8})],
        500: [("alice", "move_to", {"x": -24.5, "y": -60, "z": -16.5})],
        1000: [("alice", "move_to", {"x": 12.5, "y": -60, "z": -16.5}),
               ("bob", "place_section", {"schematic": "e2_skills_deck", "origin": [2, -61, 0]})],
        1400: [("alice", "mine", {"x": 12, "y": -60, "z": -15})],
        1800: [("alice", "craft", {"item": "minecraft:stick", "count": 8})],
        2000: [("alice", "move_to", {"x": -24.5, "y": -60, "z": -16.5})],
        2400: [("alice", "mine", {"x": -24, "y": -60, "z": -15})],
        2800: [("alice", "craft", {"item": "minecraft:crafting_table", "count": 1})],
        3000: [("bob", "move_to", {"x": 8.5, "y": -60, "z": 0.5})],
        3400: [("bob", "craft", {"item": "minecraft:oak_planks", "count": 4})],
        3600: [("bob", "place_section", {"schematic": "e2_skills_deck", "origin": [2, -61, 10]})],
    }


def setup(client: harness.Client) -> None:
    """Build fixtures and supply both bots only in the fresh A run."""
    client.request("config_set", {"reach": 4.5})
    skills.command(client, "setblock -32 -60 -32 minecraft:barrier")
    skills.command(client, "fill -32 -60 -32 31 -55 31 minecraft:air")
    skills.command(client, "setblock -32 -61 -32 minecraft:barrier")
    skills.command(client, "fill -32 -61 -32 31 -61 31 minecraft:stone")
    for offset_z in (0, 10):
        skills.command(client, f"fill 2 -62 {offset_z} 4 -61 {offset_z + 4} minecraft:air")
    for position in ((-24, -60, -11), (12, -60, -15), (-24, -60, -15)):
        skills.command(client, f"setblock {' '.join(map(str, position))} minecraft:dirt")
    for name, position in (("alice", (-24.5, -60, -12.5)), ("bob", (0.5, -60, 0.5))):
        client.request("spawn_bot", {"name": name, "x": position[0], "y": position[1], "z": position[2],
                                     "yaw": 0.0, "pitch": 0.0})
    skills.supply(client, "minecraft:stone_pickaxe", 1)
    client.request("bot_select", {"name": "alice", "slot": 0})
    skills.supply(client, "minecraft:oak_log", 8)
    skills.supply(client, "minecraft:oak_planks", 30, "bob")
    skills.supply(client, "minecraft:oak_log", 1, "bob")


def statuses(client: harness.Client) -> dict:
    """Read validated skill statuses in the fixed snapshot-bot order."""
    return {name: skills.validate_status(client.request("skill_status", {"name": name}))
            for name in SNAPSHOT_SKILLS}


def validate_snapshot(snapshot: dict, game_time: int) -> None:
    """Require both forked skills to be running with a partially built deck."""
    if not isinstance(snapshot, dict) or set(snapshot) != set(SNAPSHOT_SKILLS):
        raise harness.ProtocolError(f"Incomplete two-bot snapshot: {snapshot!r}")
    for name, skill in SNAPSHOT_SKILLS.items():
        status = skills.validate_status(snapshot[name], skill=skill)
        skills.check(status["state"] == "running" and status["startedAt"] < game_time,
                     f"Snapshot requires {name}/{skill} genuinely mid-execution: {status}")
    progress = snapshot["bob"]["progress"]
    skills.check(type(progress.get("placed")) is int and type(progress.get("total")) is int
                 and 0 < progress["placed"] < progress["total"] == 15,
                 f"Snapshot deck must be partially built: {progress}")


def validate_restore(expected: dict, actual: dict, game_time: int) -> None:
    """Require valid, exactly equal pre-save and restored skill snapshots."""
    validate_snapshot(expected, game_time)
    validate_snapshot(actual, game_time)
    skills.check(expected == actual, f"Restored skill state/status differs: A={expected}, C={actual}")


def run_segment(client: harness.Client, start: int, begin: int, end: int, hashes: list[dict],
                launches: list[dict]) -> None:
    """Run an unshifted schedule segment and append validated hashes and launches."""
    skills.check(0 <= begin < end <= RUN_TICKS and begin % 20 == end % 20 == 0,
                 "Invalid skills schedule segment")
    skills.check(harness.game_time(client.request("ping")) == start + begin, "Schedule started at wrong game time")
    actions = schedule()
    active_skills = statuses(client)
    for elapsed in range(begin, end, 20):
        for name, skill, args in actions.get(elapsed, []):
            previous = skills.validate_status(client.request("skill_status", {"name": name}))
            skills.check(previous["state"] in {"idle", "succeeded"},
                         f"Schedule would replace unfinished/failed skill at {start + elapsed}: {previous}")
            result = client.request("skill_start", {"name": name, "skill": skill, "args": args})
            launches.append({"gameTime": start + elapsed, "name": name, "skill": skill, "args": args, "result": result})
            skills.check(skills.validate_start(result), f"Scheduled skill rejected at {start + elapsed}: {result}")
            active = skills.validate_status(client.request("skill_status", {"name": name}),
                                            skill=skill, skill_id=result["skillId"])
            skills.check(active.get("args") == args, f"Scheduled skill args changed: {active}")
            active_skills[name] = active
        before, after = start + elapsed, start + elapsed + 20
        stepped = client.request("step", {"n": 20, "hash_every": 20, "region": skills.REGION})
        hashes.extend(harness.step_hashes(stepped, before, after))
        for name, status in statuses(client).items():
            previous = active_skills[name]
            if previous["state"] != "idle":
                skills.validate_status(status, skill=previous["skill"], skill_id=previous["skillId"])
                skills.check(status["state"] != "idle", f"Scheduled {name} lost its active skill at {after}")
            skills.check(status["state"] not in {"failed", "cancelled"},
                         f"Scheduled {name} failed at {after}: {status}")
            active_skills[name] = status


def validate_trace(records: list[dict], before: int, after: int) -> None:
    """Require every 20-tick sample and all existing hash components in order."""
    expected_times = list(range((before // 20 + 1) * 20, after + 1, 20))
    if not isinstance(records, list) or len(records) != len(expected_times):
        raise harness.ProtocolError(f"Incomplete hash trace for ({before}, {after}]")
    for record, expected_time in zip(records, expected_times):
        if not isinstance(record, dict) or harness.game_time(record) != expected_time:
            raise harness.ProtocolError(f"Missing/reordered hash at {expected_time}: {record!r}")
        harness.hash_components(record.get("hash"), expected_time)


def compare_suffix(hashes_a: list[dict], hashes_c: list[dict], start: int, snapshot_time: int) -> bool:
    """Validate and compare the full fork suffix, printing its existing verdict."""
    skills.check(start < snapshot_time < start + RUN_TICKS, "Snapshot must be inside the 4,000-tick run")
    validate_trace(hashes_a, start, start + RUN_TICKS)
    validate_trace(hashes_c, snapshot_time, start + RUN_TICKS)
    expected = [record for record in hashes_a if record["gameTime"] > snapshot_time]
    for left, right in zip(expected, hashes_c):
        if left != right:
            print(f"FAIL E2 skills: first mismatch at game time {left['gameTime']}")
            print("A: " + json.dumps(left, sort_keys=True))
            print("C: " + json.dumps(right, sort_keys=True))
            return False
    print(f"PASS E2 skills: {len(expected)} exact full-component suffix hashes (every 20 ticks)")
    return True


class E2SkillsExperiment:
    """Own the skill fork while preserving schedule, exception, and flush order."""

    def __init__(self, args):
        """Keep parsed CLI options for the two skill-run lifecycles."""
        self._args = args

    def setup(self) -> None:
        """Preflight both paths and clone A after creating C's server shell."""
        args = self._args
        self._directory_a, self._directory_c = (
            args.runs_dir / name for name in ("e2_skills_a", "e2_skills_c"))
        for directory in (self._directory_a, self._directory_c):
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        self._directory_c = setup_server("e2_skills_c", args.runs_dir)
        harness.clone_apfs(args.runs_dir / "base", self._directory_a)
        self._hashes_a, self._hashes_c, self._launches_a, self._launches_c = [], [], [], []
        self._report = {"passed": False, "run_ticks": RUN_TICKS, "hash_every": 20, "errors": []}

    def run(self) -> None:
        """Run A and restored C, checking inside the original try/finally scope."""
        args = self._args
        directory_a, directory_c = self._directory_a, self._directory_c
        hashes_a, hashes_c = self._hashes_a, self._hashes_c
        launches_a, launches_c = self._launches_a, self._launches_c
        report = self._report
        try:
            skills.write_schematic(directory_a, "e2_skills_deck")
            with harness.Server(directory_a, **harness.server_options(args)) as server:
                start = harness.enable_lockstep(server.client)
                setup(server.client)
                report["start"] = start
                run_segment(server.client, start, 0, FORK_TICKS, hashes_a, launches_a)
                snapshot_time = start + FORK_TICKS
                snapshot = statuses(server.client)
                validate_snapshot(snapshot, snapshot_time)
                position = server.client.request("bot_state", {"name": "alice"}).get("pos")
                skills.check(isinstance(position, list) and len(position) == 3
                             and -24.5 < position[0] < 12.0, f"move_to made no partial horizontal progress: {position}")
                report.update(snapshot_time=snapshot_time, snapshot=snapshot)
                server.client.request("save_snapshot")
                skills.check(harness.game_time(server.client.request("ping")) == snapshot_time,
                             "A advanced during snapshot")
                harness.clone_apfs(directory_a / "world", directory_c / "world")
                skills.check(harness.game_time(server.client.request("ping")) == snapshot_time,
                             "A advanced during frozen clone")
                run_segment(server.client, start, FORK_TICKS, RUN_TICKS, hashes_a, launches_a)
                skills.check(all(status["state"] == "succeeded" for status in statuses(server.client).values()),
                             "A ended with unfinished skills")
            with harness.Server(directory_c, **harness.server_options(args)) as server:
                restored_time = harness.enable_lockstep(server.client)
                skills.check(restored_time == snapshot_time, f"C started at {restored_time}, expected {snapshot_time}")
                server.client.request("restore_bots")
                restored = statuses(server.client)
                report["restored"] = restored
                validate_restore(snapshot, restored, snapshot_time)
                run_segment(server.client, start, FORK_TICKS, RUN_TICKS, hashes_c, launches_c)
                skills.check(all(status["state"] == "succeeded" for status in statuses(server.client).values()),
                             "C ended with unfinished skills")
            self.check(start, snapshot_time)
        except skills.ERRORS as error:
            message = f"{type(error).__name__}: {error}"
            report["errors"].append(message)
            print(f"FAIL E2 skills: {message}")
        finally:
            harness.write_hashes(directory_a / "hashes.json", hashes_a)
            harness.write_hashes(directory_c / "hashes.json", hashes_c)
            report.update(launches_a=launches_a, launches_c=launches_c)
            (directory_a / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")

    def check(self, start: int, snapshot_time: int) -> None:
        """Compare post-fork launches and hashes before the mandatory log flush."""
        suffix_launches = [entry for entry in self._launches_a if entry["gameTime"] >= snapshot_time]
        skills.check(suffix_launches == self._launches_c,
                     "A/C post-fork command schedules or acceptance results differ")
        self._report["passed"] = compare_suffix(self._hashes_a, self._hashes_c, start, snapshot_time)
        if not self._report["passed"]:
            self._report["errors"].append("Full post-fork hash suffix differs; see the A/C hashes.json files")

    def report(self) -> int:
        """Print the final verdict after logs are flushed and return its status."""
        print(f"{'PASS' if self._report['passed'] else 'FAIL'} E2 skills")
        return 0 if self._report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run phases with checks inside their error scope."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = E2SkillsExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
