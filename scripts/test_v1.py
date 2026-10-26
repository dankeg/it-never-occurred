"""V1: a physical road closure, exact welfare arithmetic, and a restored replay.

The town has one-block-wide roads and 5x5 intersections spaced 12 blocks apart.
Its middle road extends to the school: open school trips are 96--103 blocks,
and closing an interior segment adds 24 blocks. Day 0 starts from the prepared
base, before any scheduled departure; the fork is exactly game time 24,000.

Trip paths contain surface-node xyz arrays. ``path`` and ``pathLength`` describe
the latest A* route, with length in edges, not cumulative physical distance.
Each segment records the actual pre-snap position, cumulative travelled distance
and movingTicks, and the departure/reroute end-tick. School statuses are on_time,
late, or absent. Only the existing runtime world-state hashes are compared.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

if __package__:
    from . import harness
else:
    import harness


DAY_TICKS = 24_000
SPEED = 0.12
SURFACE_Y = -61
GRID = tuple(range(-24, 25, 12))
REGION = {"x0": -60, "z0": -24, "x1": 60, "z1": 24, "y0": SURFACE_Y, "y1": SURFACE_Y}
HASH_REGION = {"x0": -62, "z0": -26, "x1": 62, "z1": 26, "y0": -64, "y1": -56}
CLOSURE_TICK = DAY_TICKS + 500
ERRORS = (harness.HarnessError, OSError, ValueError, KeyError, TypeError)


def check(condition: bool, message: str) -> None:
    """Raise a harness failure when a V1 invariant does not hold."""
    if not condition:
        raise harness.HarnessError(message)


def integer(value, label: str) -> int:
    """Return a nonnegative integer or fail with the supplied field label."""
    check(type(value) is int and value >= 0, f"Invalid {label}: {value!r}")
    return value


def number(value, label: str) -> float:
    """Return a finite nonnegative number, rejecting booleans and invalid data."""
    check(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          f"Invalid {label}: {value!r}")
    return float(value)


def vector(value, label: str) -> tuple:
    """Validate three finite coordinates and return them as a tuple."""
    check(isinstance(value, list) and len(value) == 3
          and all(type(coordinate) in (int, float) and math.isfinite(coordinate) for coordinate in value),
          f"Invalid {label}: {value!r}")
    return tuple(value)


def close(actual: float, expected: float, label: str) -> None:
    """Require equality within the existing welfare-arithmetic tolerances."""
    check(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
          f"{label}: got {actual}, expected {expected}")


def path_nodes(value) -> list[tuple]:
    """Validate integer surface nodes and each cardinal road edge."""
    check(isinstance(value, list), f"Invalid path: {value!r}")
    nodes = [vector(node, "path node") for node in value]
    check(all(all(type(coordinate) is int for coordinate in node) for node in nodes),
          "Path nodes must be integer surface coordinates")
    for origin, target in zip(nodes, nodes[1:]):
        check(abs(origin[0] - target[0]) + abs(origin[2] - target[2]) == 1
              and abs(origin[1] - target[1]) <= 1, f"Non-road edge: {origin} -> {target}")
    return nodes


def centre(node: tuple) -> tuple:
    """Return the standing position above the centre of a surface node."""
    return node[0] + 0.5, node[1] + 1, node[2] + 0.5


def position_along(points: list[tuple], distance: float) -> tuple:
    """Interpolate distance along a polyline, rejecting excess distance."""
    for origin, target in zip(points, points[1:]):
        length = math.dist(origin, target)
        if length and distance <= length:
            return tuple(start + (end - start) * distance / length for start, end in zip(origin, target))
        distance -= length
    close(distance, 0, "Distance beyond segment endpoint")
    return points[-1]


def expected_ticks(trip: dict) -> int:
    """Reconstruct moving ticks from polylines, including interrupted segments.

    A reroute is logged after that tick's movement. Its movingTicks prefix is
    verified against the previous segment, not treated as a free teleport or
    rounded together with the new segment. Initial movement starts at depart+1.
    Reroute remainders honor the mover's 1e-9 floating-point budget tolerance.
    """
    depart = integer(trip["departTick"], "departTick")
    segments = trip["segments"]
    check(isinstance(segments, list) and bool(segments), f"Missing movement segments: {trip}")
    previous = None
    for index, segment in enumerate(segments):
        tick = integer(segment["tick"], "segment tick")
        moving = integer(segment["movingTicks"], "segment movingTicks")
        travelled = number(segment["travelled"], "segment travelled")
        snap = number(segment["snapDistance"], "segment snapDistance")
        position = vector(segment["position"], "segment position")
        nodes = path_nodes(segment["path"])
        check(tick == depart + moving, "Segment end-tick disagrees with its moving-tick prefix")
        if index == 0:
            check(tick == depart and moving == 0 and travelled == 0 and snap == 0,
                  "Initial segment has movement before departure")
            if nodes:
                close(math.dist(position, centre(nodes[0])), 0, "Initial position")
        if previous is not None:
            previous_tick, previous_moving, previous_travelled, previous_points, previous_length = previous
            check(tick >= previous_tick and moving >= previous_moving, "Reroute segments are reordered")
            walked = SPEED * (moving - previous_moving)
            check(walked < previous_length + 1e-7, "Reroute occurred after the trip already arrived")
            close(travelled, previous_travelled + walked, "Reroute accumulated distance")
            projected = position_along(previous_points, min(walked, previous_length))
            close(math.dist(position, projected), 0, "Reroute actual position")
        points = [position, *[centre(node) for node in nodes]]
        if nodes:
            close(snap, math.dist(position, points[1]), "Snap distance")
        else:
            check(snap == 0, "An unreachable segment cannot have a snap destination")
        length = snap + math.fsum(math.dist(origin, target) for origin, target in zip(nodes, nodes[1:]))
        previous = tick, moving, travelled, points, length
    tick, moving, travelled, _points, remaining = previous
    check(trip["path"] == segments[-1]["path"], "Trip path is not its latest route")
    check(integer(trip["pathLength"], "pathLength") == max(0, len(nodes) - 1), "Wrong latest A* edge length")
    if "travelled" in trip:
        close(number(trip["travelled"], "trip travelled"), travelled + remaining, "Total physical travel length")
    remaining_ticks = math.ceil(remaining / SPEED)
    if len(segments) > 1 and remaining > 0:
        remaining_ticks = max(1, math.ceil(max(0, remaining - 1e-9) / SPEED))
    expected = moving + remaining_ticks
    check(integer(trip["ticks"], "trip ticks") == expected,
          f"{trip['npc']}/{trip['activity']}: moving ticks {trip['ticks']} != {expected}")
    if segments[-1]["path"]:
        arrive = integer(trip["arriveTick"], "arriveTick")
        check(arrive == depart + expected, f"Arrival must be departTick + ticks: {trip}")
    else:
        check(trip["outcome"] in {"unreachable", "absent", "missed"} and trip["arriveTick"] is None,
              "Invalid early unreachable termination")
    return expected


def town_setup() -> dict:
    """Return the deterministic town, household, and NPC configuration."""
    places = {f"home_{index}": [-36 - index, SURFACE_Y, 0] for index in range(10)}
    places.update(work_0=[48, SURFACE_Y, 0], work_1=[24, SURFACE_Y, -24],
                  work_2=[-24, SURFACE_Y, 24], market=[36, SURFACE_Y, 0], school=[60, SURFACE_Y, 0])
    market_days = ([0, 1, 3, 5], [1, 2, 4, 6], [0, 2, 4], [3, 5, 6])
    npcs = [{"id": f"adult_{index:02d}", "name": f"V1 Adult {index:02d}", "role": "adult",
             "home": f"home_{index % 10}", "work": f"work_{index % 3}",
             "market_days": list(market_days[index % len(market_days)])} for index in range(22)]
    npcs.extend({"id": f"child_{index:02d}", "name": f"V1 Child {index:02d}", "role": "child",
                 "home": f"home_{index}", "school": "school"} for index in range(8))
    return {"region": REGION, "surfaces": ["minecraft:dirt_path"], "places": places, "npcs": npcs, "speed": SPEED}


def command(client: harness.Client, text: str) -> None:
    """Send the supplied server command without changing its arguments."""
    client.request("command", {"command": text})


def build_town(client: harness.Client) -> None:
    """Build the physical road grid and configure the scheduled town."""
    bounds = harness.FORCELOADED_CHUNKS
    check(bounds["cx0"] * 16 <= HASH_REGION["x0"] <= HASH_REGION["x1"] < (bounds["cx1"] + 1) * 16
          and bounds["cz0"] * 16 <= HASH_REGION["z0"] <= HASH_REGION["z1"] < (bounds["cz1"] + 1) * 16,
          "Town must fit inside the base's force-loaded chunks")
    command(client, "setblock -62 -60 -26 minecraft:barrier")
    command(client, "fill -62 -60 -26 62 -57 26 minecraft:air")
    command(client, "fill -62 -61 -26 62 -61 26 minecraft:stone")
    for coordinate in GRID:
        command(client, f"fill -24 -61 {coordinate} 24 -61 {coordinate} minecraft:dirt_path")
        command(client, f"fill {coordinate} -61 -24 {coordinate} -61 24 minecraft:dirt_path")
    command(client, "fill -60 -61 0 60 -61 0 minecraft:dirt_path")
    client.request("npc_setup", town_setup())


def run_until(client: harness.Client, end: int, hashes: list[dict]) -> None:
    """Advance lockstep to the target time and append existing runtime hashes."""
    current = harness.game_time(client.request("ping"))
    check(current <= end, f"Already past requested game time {end}")
    while current < end:
        following = min(current + 200, end)
        result = client.request("step", {"n": following - current, "hash_every": harness.HASH_EVERY,
                                         "region": HASH_REGION})
        hashes.extend(harness.step_hashes(result, current, following))
        current = following


def day_metrics(client: harness.Client, day: int) -> dict:
    """Request and validate the metrics response for one scheduled day."""
    days = client.request("npc_metrics", {"day": day}).get("days")
    check(isinstance(days, list) and all(isinstance(record, dict) for record in days), "Invalid npc_metrics days")
    selected = [record for record in days if record.get("day") == day]
    check(len(selected) == 1, f"Expected exactly one metrics record for day {day}")
    return selected[0]


def road_node(node: tuple) -> bool:
    """Test whether a surface node lies on the configured town roads."""
    horizontal, height, vertical = node
    return height == SURFACE_Y and (
        vertical == 0 and -60 <= horizontal <= 60
        or -24 <= horizontal <= 24 and -24 <= vertical <= 24 and (horizontal in GRID or vertical in GRID))


def verdict(activity: str, arrival: int, day: int) -> tuple:
    """Compute the scheduled activity outcome from its arrival tick."""
    phase = arrival - day * DAY_TICKS
    if activity == "work":
        return phase <= 3000, "arrived", None
    if activity == "market":
        return phase <= 11000, "arrived" if phase <= 11000 else "missed", None
    if activity == "school":
        status = "on_time" if phase <= 2500 else "late" if phase <= 4000 else "absent"
        return phase <= 2500, "absent" if phase > 4000 else "arrived", status
    check(activity == "home", f"Unknown activity: {activity!r}")
    return True, "arrived", None


def validate_day(record: dict, day: int, closure: tuple | None = None) -> dict:
    """Validate routes, timing, and welfare arithmetic and return indexed trips."""
    setup = town_setup()
    people = {npc["id"]: npc for npc in setup["npcs"]}
    rows, trips = record["npcs"], record["trips"]
    check(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "Invalid daily NPC rows")
    check(Counter(row.get("npc") for row in rows) == Counter(people.keys()),
          "Daily metrics must cover all 30 NPCs once")
    check(isinstance(trips, list) and all(isinstance(trip, dict) for trip in trips), "Invalid daily trips")
    indexed = {}
    for trip in trips:
        npc, activity = trip["npc"], trip["activity"]
        check(npc in people and trip["day"] == day, f"Trip has wrong NPC/day: {trip}")
        key = npc, activity
        check(key not in indexed, f"Duplicate trip: {key}")
        indexed[key] = trip
        expected_ticks(trip)
        check(len(trip["segments"]) == 1 and bool(trip["path"]),
              "This pre-departure closure fixture must not reroute or become unreachable")
        nodes = path_nodes(trip["path"])
        check(all(road_node(node) for node in nodes), f"Trip left the built road graph: {key}")
        if closure is not None:
            check(closure not in nodes, f"Closed trip crossed the wall: {key}")
        check(trip["from"] in setup["places"] and trip["to"] in setup["places"], f"Unknown place: {key}")
        check(nodes[0] == tuple(setup["places"][trip["from"]])
              and nodes[-1] == tuple(setup["places"][trip["to"]]), f"Wrong path endpoints: {key}")
        check(day * DAY_TICKS <= trip["departTick"] <= trip["arriveTick"] < (day + 1) * DAY_TICKS,
              f"Trip outside its day: {key}")
        expected = verdict(activity, trip["departTick"] + trip["ticks"], day)
        check(type(trip["onTime"]) is bool and (trip["onTime"], trip["outcome"]) == expected[:2],
              f"Deadline arithmetic disagrees with trip result: {key}: {trip}")
    commutes = []
    for row in rows:
        npc = row["npc"]
        person = people[npc]
        primary = "work" if person["role"] == "adult" else "school"
        activities = {primary, "home"}
        market = person["role"] == "adult" and day % 7 in person["market_days"]
        if market:
            activities.add("market")
        check({activity for identity, activity in indexed if identity == npc} == activities,
              f"Incomplete or extra itinerary: {npc}")
        outbound = indexed[npc, primary]
        start, stop = (1000, 1400) if primary == "work" else (1500, 1700)
        check(start <= outbound["departTick"] - day * DAY_TICKS < stop, f"Wrong departure offset: {npc}")
        check(outbound["from"] == person["home"] and outbound["to"] == person[primary],
              f"Wrong scheduled destination: {npc}")
        home = indexed[npc, "home"]
        check(home["to"] == person["home"], f"NPC did not return home: {npc}")
        if market:
            visit = indexed[npc, "market"]
            check(visit["from"] == person["work"] and visit["to"] == "market"
                  and visit["departTick"] == day * DAY_TICKS + 9000
                  and home["from"] == "market" and home["departTick"] >= visit["arriveTick"],
                  f"Wrong market itinerary: {npc}")
        else:
            leave = 9000 if primary == "work" else 8000
            check(home["from"] == person[primary] and home["departTick"] == day * DAY_TICKS + leave,
                  f"Wrong return schedule: {npc}")
        commute = sum(indexed[npc, activity]["ticks"] for activity in activities)
        check(integer(row["commuteTicks"], "commuteTicks") == commute, f"Wrong daily commute: {npc}")
        commutes.append(commute)
        if primary == "work":
            check(row.get("workOnTime") is outbound["onTime"], f"Wrong workOnTime: {npc}")
            check(row["workStatus"] == ("on_time" if outbound["onTime"] else "late")
                  and row["schoolStatus"] == "not_scheduled", f"Wrong work/school status: {npc}")
        else:
            check(row["schoolStatus"] == verdict("school", outbound["arriveTick"], day)[2],
                  f"Wrong schoolStatus: {npc}")
            check(row.get("workOnTime") is None and row["workStatus"] == "not_scheduled",
                  f"Child has a work status: {npc}")
        visited = indexed[npc, "market"]["outcome"] == "arrived" if market else None
        check(row.get("marketVisited") is visited, f"Wrong marketVisited: {npc}")
    close(number(record["meanCommute"], "meanCommute"), sum(commutes) / len(people), "Mean commute")
    check(record["maxCommute"] == max(commutes), "Wrong max commute")
    late = sum(row["workStatus"] == "late" or row["schoolStatus"] == "late" for row in rows)
    markets = [row for row in rows if row.get("marketVisited") is not None]
    missed = sum(row.get("marketVisited") is False for row in markets)
    absent = sum(row["schoolStatus"] == "absent" for row in rows)
    for share, expected in (("lateShare", late / 30),
                            ("missedMarketShare", missed / len(markets) if markets else 0),
                            ("absentShare", absent / 8)):
        close(number(record[share], share), expected, share)
    affected = record["affectedNpcs"]
    check(isinstance(affected, list) and len(affected) == len(set(affected))
          and set(affected) <= people.keys(), "Invalid affectedNpcs")
    return indexed


def select_closure(trips: dict) -> tuple:
    """Choose the interior road node used by the baseline school trips."""
    school_departures = {trip["departTick"] for (_npc, activity), trip in trips.items() if activity == "school"}
    check(len(school_departures) > 1, "School slack fixture requires varied departures")
    candidates = [(coordinate, SURFACE_Y, 0) for coordinate in (-18, -6, 6, 18)]
    counts = {node: sum(node in path_nodes(trip["path"]) for trip in trips.values()) for node in candidates}
    selected = min(candidates, key=lambda node: (-counts[node], node))
    school_users = sum(activity == "school" and selected in path_nodes(trip["path"])
                       for (_npc, activity), trip in trips.items())
    check(counts[selected] > 0 and school_users == 8, "Closure must cut all eight school routes")
    print(f"V1 selects road node {selected}: {counts[selected]} open trips, including all 8 school commutes")
    return selected


def compare_branches(opened: dict, closed: dict, closure: tuple) -> dict:
    """Check affected and unaffected trips and print explained outcome changes."""
    check(opened.keys() == closed.keys(), "Open/closed trip identities differ")
    affected, unaffected, longer, changes = [], [], [], []
    for key, original in opened.items():
        diverted = closed[key]
        if key[1] in {"work", "school"}:
            check(original["departTick"] == diverted["departTick"], f"Closure changed schedule offsets: {key}")
        if closure not in path_nodes(original["path"]):
            check(original == diverted, f"Unaffected trip changed: {key}")
            unaffected.append(list(key))
        else:
            expected = expected_ticks(diverted) - expected_ticks(original)
            actual = diverted["ticks"] - original["ticks"]
            check(actual == expected, f"Wrong travel-tick delta for {key}: {actual} != {expected}")
            check(diverted["path"] != original["path"], f"Affected route did not change: {key}")
            affected.append(list(key))
            if actual > 0:
                longer.append(list(key))
        before = verdict(key[1], original["departTick"] + expected_ticks(original), 1)
        after = verdict(key[1], diverted["departTick"] + expected_ticks(diverted), 1)
        if before != after:
            check(key[1] != "home" and original["arriveTick"] != diverted["arriveTick"],
                  f"Outcome changed without an arrival/deadline change: {key}")
            thresholds = {"work": (3000,), "market": (11000,), "school": (2500, 4000)}[key[1]]
            crossed = [deadline for deadline in thresholds
                       if (original["arriveTick"] <= DAY_TICKS + deadline)
                       != (diverted["arriveTick"] <= DAY_TICKS + deadline)]
            check(bool(crossed), f"Unexplained deadline outcome change: {key}")
            changes.append({"npc": key[0], "activity": key[1], "deadlines": crossed,
                            "openArrival": original["arriveTick"], "closedArrival": diverted["arriveTick"],
                            "openLength": original["pathLength"], "closedLength": diverted["pathLength"],
                            "openResult": list(before), "closedResult": list(after)})
    check(bool(affected) and bool(unaffected) and bool(longer), "V1 exactness/unaffected checks would be vacuous")
    check(bool(changes), "No NPC crossed a deadline; welfare comparison would be vacuous")
    for change in changes:
        print("V1 explained outcome: " + json.dumps(change, sort_keys=True))
    return {"affectedTrips": affected, "unaffectedTrips": unaffected, "longerTrips": longer,
            "outcomeChanges": changes}


def states(client: harness.Client) -> dict:
    """Read each configured NPC's state in deterministic town order."""
    return {npc["id"]: client.request("npc_state", {"id": npc["id"]}) for npc in town_setup()["npcs"]}


def run_branch(directory: Path, args, baseline: dict, snapshot: dict, closure: tuple | None) -> dict:
    """Restore day 1, optionally close the road, and persist branch hashes."""
    hashes = []
    with harness.Server(directory, **harness.server_options(args)) as server:
        client = server.client
        restored = harness.enable_lockstep(client)
        check(restored == DAY_TICKS, f"Restored at {restored}, expected start of day 1")
        client.request("restore_bots")
        check(states(client) == snapshot, "NPC state did not survive the day-1 fork")
        check(day_metrics(client, 0) == baseline, "Day-0 welfare did not survive the fork")
        run_until(client, CLOSURE_TICK, hashes)
        if closure is not None:
            horizontal, surface, vertical = closure
            command(client, f"fill {horizontal} {surface + 1} {vertical - 1} "
                            f"{horizontal} {surface + 1} {vertical + 1} minecraft:cobblestone_wall")
            check(harness.game_time(client.request("ping")) == CLOSURE_TICK,
                  "Wall command advanced time instead of acting at phase 500")
        run_until(client, 2 * DAY_TICKS, hashes)
        metrics = day_metrics(client, 1)
        validate_day(metrics, 1, closure)
        final = states(client)
        client.request("save_snapshot")
    harness.write_hashes(directory / "hashes.json", hashes)
    return {"metrics": metrics, "states": final, "hashes": hashes}


def matching_hashes(expected: list[dict], actual: list[dict], label: str) -> None:
    """Require exact full-component traces and print the labeled match count."""
    check(bool(expected) and len(expected) == len(actual), f"Incomplete {label} hash trace")
    for original, repeated in zip(expected, actual):
        check(original == repeated, f"{label}: runtime hash mismatch at {original['gameTime']}: "
              f"first={original}, repeated={repeated}")
    print(f"PASS V1 {label}: {len(expected)} matching full-component runtime hashes")


class V1Experiment:
    """Own the day-1 road forks and exact replay/welfare comparison phases."""

    def __init__(self, args):
        """Keep parsed CLI options for the baseline and restored branches."""
        self._args = args

    def setup(self) -> None:
        """Preflight all four destinations before cloning the baseline."""
        args = self._args
        self._directory, self._open_directory, self._closed_directory, self._repeat_directory = (
            args.runs_dir / name for name in ("v1", "v1_open", "v1_closed", "v1_closed_repeat"))
        for destination in (self._directory, self._open_directory, self._closed_directory, self._repeat_directory):
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {destination}")
        harness.clone_apfs(args.runs_dir / "base", self._directory)
        self._report = {"passed": False, "speed": SPEED, "forkTick": DAY_TICKS, "closureTick": CLOSURE_TICK,
                        "roadWidth": 1, "gridIntersections": [5, 5], "spacing": 12, "adults": 22, "children": 8,
                        "errors": []}

    def run(self) -> None:
        """Run the baseline and forks, retaining checks inside their error scope."""
        args = self._args
        directory, open_directory = self._directory, self._open_directory
        closed_directory, repeat_directory = self._closed_directory, self._repeat_directory
        report = self._report
        try:
            baseline_hashes = []
            with harness.Server(directory, **harness.server_options(args)) as server:
                client = server.client
                start = harness.enable_lockstep(client)
                check(0 <= start < 1000, "Base must be in day 0, before the first possible departure")
                report["baselineStart"] = start
                build_town(client)
                run_until(client, DAY_TICKS, baseline_hashes)
                baseline = day_metrics(client, 0)
                validate_day(baseline, 0)
                snapshot = states(client)
                client.request("save_snapshot")
                check(harness.game_time(client.request("ping")) == DAY_TICKS, "Baseline advanced during save_snapshot")
                for destination in (open_directory, closed_directory, repeat_directory):
                    harness.clone_apfs(directory, destination)
                    check(harness.game_time(client.request("ping")) == DAY_TICKS, "Baseline advanced during APFS clone")
            harness.write_hashes(directory / "hashes.json", baseline_hashes)
            report["baseline"] = baseline
            opened = run_branch(open_directory, args, baseline, snapshot, None)
            open_trips = validate_day(opened["metrics"], 1)
            closure = select_closure(open_trips)
            report["closureNode"] = list(closure)
            closed = run_branch(closed_directory, args, baseline, snapshot, closure)
            repeated = run_branch(repeat_directory, args, baseline, snapshot, closure)
            self.check(opened, closed, repeated, closure, open_trips)
        except ERRORS as error:
            message = f"{type(error).__name__}: {error}"
            report["errors"].append(message)
            print(f"FAIL V1: {message}")

    def check(self, opened: dict, closed: dict, repeated: dict, closure: tuple, open_trips: dict) -> None:
        """Compare runtime hashes, welfare, and explained closure effects in order."""
        matching_hashes(closed["hashes"], repeated["hashes"], "closed/restored replay")
        for label, branch in (("closed", closed), ("repeated", repeated)):
            matching_hashes([entry for entry in opened["hashes"] if entry["gameTime"] <= CLOSURE_TICK],
                            [entry for entry in branch["hashes"] if entry["gameTime"] <= CLOSURE_TICK],
                            f"open/{label} before closure")
        check(closed["metrics"] == repeated["metrics"] and closed["states"] == repeated["states"],
              "Closed replay welfare or final NPC state differs")
        comparison = compare_branches(open_trips, validate_day(closed["metrics"], 1, closure), closure)
        check(not opened["metrics"]["affectedNpcs"], "Unchanged day-1 roads reported affected NPCs")
        check(set(closed["metrics"]["affectedNpcs"]) == {npc for npc, _activity in comparison["affectedTrips"]},
              "Affected-NPC metrics disagree with the changed open/closed routes")
        self._report.update(comparison)
        self._report.update(open=opened["metrics"], closed=closed["metrics"], passed=True)

    def report(self) -> int:
        """Write the original report before printing and returning the gate verdict."""
        report = self._report
        (self._directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
        print(f"{'PASS' if report['passed'] else 'FAIL'} V1")
        return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run phases with checks inside their error scope."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = V1Experiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
