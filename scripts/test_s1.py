"""S1: seeded grounded-skill cases, dry-run purity, and acceptance agreement."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

if __package__:
    from . import harness
else:
    import harness


SEED = 20261016
CASES_PER_SKILL = 20
SKILLS = ("move_to", "mine", "collect", "craft", "place", "place_section",
          "chest_put", "chest_take", "give", "wait")
REASONS = frozenset(("no_path", "out_of_reach", "not_visible", "missing_items", "no_recipe",
                     "needs_crafting_table", "inventory_full", "target_occupied", "no_support",
                     "unbreakable", "acl_denied", "target_not_found", "stuck", "timeout", "interrupted"))
REGION = {"x0": -32, "z0": -32, "x1": 31, "z1": 31, "y0": -64, "y1": -50}
CHEST = [0, -60, 2]
CASE_TICK_CAP = 2000
ERRORS = (harness.HarnessError, OSError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class Case:
    """Describe one seeded or explicit skill case without mutable driver state."""

    skill: str
    label: str
    args: dict
    fixture: str = "plain"
    expected_reason: str | None = None
    generated: bool = True


def check(condition: bool, message: str) -> None:
    """Raise a harness failure when a skill assertion does not hold."""
    if not condition:
        raise harness.HarnessError(message)


def nonnegative_integer(value) -> bool:
    """Recognize nonnegative integers without accepting booleans."""
    return type(value) is int and value >= 0


def valid_id(value) -> bool:
    """Recognize nonempty string or nonnegative integer skill identifiers."""
    return (isinstance(value, str) and bool(value)) or nonnegative_integer(value)


def reason(value: dict) -> str:
    """Return a known failure reason or raise a protocol error."""
    code = value.get("reason")
    if not isinstance(code, str) or code not in REASONS:
        raise harness.ProtocolError(f"Missing/unknown failure reason: {value!r}")
    return code


def validate_dry(value: dict) -> bool:
    """Validate the dry-run verdict and estimate, returning feasibility."""
    if not isinstance(value, dict) or type(value.get("feasible")) is not bool:
        raise harness.ProtocolError(f"Invalid dry-run verdict: {value!r}")
    estimate = value.get("estimate")
    if (not isinstance(estimate, dict) or not nonnegative_integer(estimate.get("ticks"))
            or not isinstance(estimate.get("materials"), dict)
            or any(not isinstance(item, str) or not item.startswith("minecraft:")
                   or not nonnegative_integer(count) for item, count in estimate["materials"].items())):
        raise harness.ProtocolError(f"Invalid dry-run estimate: {value!r}")
    if not value["feasible"]:
        reason(value)
    return value["feasible"]


def validate_start(value: dict) -> bool:
    """Validate a real-start response and return its acceptance verdict."""
    if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
        raise harness.ProtocolError(f"Invalid real-start verdict: {value!r}")
    if value["accepted"]:
        if not valid_id(value.get("skillId")):
            raise harness.ProtocolError(f"Missing/invalid skillId: {value!r}")
    else:
        reason(value)
    return value["accepted"]


def validate_status(value: dict, *, skill: str | None = None, skill_id=None) -> dict:
    """Validate status schema and optional skill identity, returning the status."""
    if not isinstance(value, dict) or not isinstance(value.get("state"), str) or value["state"] not in {
            "idle", "running", "succeeded", "failed", "cancelled"}:
        raise harness.ProtocolError(f"Invalid skill status: {value!r}")
    if skill is not None and value.get("skill") != skill:
        raise harness.ProtocolError(f"Status belongs to another skill: {value!r}")
    if skill_id is not None and value.get("skillId") != skill_id:
        raise harness.ProtocolError(f"Status belongs to another skillId: {value!r}")
    if value["state"] != "idle":
        if (not valid_id(value.get("skillId")) or value.get("skill") not in SKILLS
                or not isinstance(value.get("args"), dict) or not isinstance(value.get("progress"), dict)
                or not nonnegative_integer(value.get("startedAt"))):
            raise harness.ProtocolError(f"Incomplete skill status: {value!r}")
        if value["state"] != "running" and (
                not nonnegative_integer(value.get("endedAt")) or value["endedAt"] < value["startedAt"]):
            raise harness.ProtocolError(f"Invalid endedAt: {value!r}")
    if value["state"] in {"failed", "cancelled"}:
        reason(value)
    return value


def generate_cases() -> list[Case]:
    """Generate the fixed-seed skill cases followed by explicit negative cases."""
    rng = random.Random(SEED)
    cases = []
    for skill in SKILLS:
        for index in range(CASES_PER_SKILL):
            fixture = "plain"
            if skill == "move_to":
                args = {"x": rng.randint(10, 27) + 0.5, "y": -60, "z": rng.randint(-8, 8) + 0.5,
                        "tolerance": 0.5}
                fixture = ("plain", "wall_gap", "step")[index % 3]
            elif skill in {"mine", "collect"}:
                args = ({"x": rng.randint(-1, 1), "y": -60, "z": rng.randint(2, 3)}
                        if skill == "mine" else {"radius": rng.randint(4, 6)})
                fixture = ("dirt", "stone", "oak_log", "oak_planks")[index % 4]
            elif skill == "craft":
                item = ("oak_planks", "stick", "crafting_table", "chest")[index % 4]
                args = {"item": f"minecraft:{item}", "count": rng.randint(1, 3) * (4 if index % 4 < 2 else 1)}
                fixture = "table" if item == "chest" else "plain"
            elif skill == "place":
                args = {"x": rng.randint(1, 2), "y": -60, "z": rng.randint(1, 2),
                        "item": "minecraft:" + ("dirt" if index % 2 else "oak_planks")}
            elif skill == "place_section":
                args = {"schematic": "s1_deck", "origin": [2, -61, 0]}
                fixture = f"deck_{rng.randrange(5)}"
            elif skill in {"chest_put", "chest_take"}:
                args = {"pos": list(CHEST), "item": "minecraft:dirt", "count": rng.randint(1, 32)}
                fixture = "allowed"
            elif skill == "give":
                args = {"target": "bob", "item": "minecraft:dirt", "count": rng.randint(1, 32)}
            else:
                args = {"ticks": rng.randint(1, 100)}
            cases.append(Case(skill, f"{skill}/{index + 1:02d}/{fixture}", args, fixture))
    negatives = [
        ("move_to", {"x": 8.5, "y": -60, "z": 0.5}, "sealed", "no_path"),
        ("move_to", {"x": 8.5, "y": -60, "z": 0.5}, "water", "no_path"),
        ("mine", {"x": 0, "y": -60, "z": 3}, "bedrock", "unbreakable"),
        ("mine", {"x": 12, "y": -60, "z": 0}, "far", "out_of_reach"),
        ("mine", {"x": 0, "y": -60, "z": 3}, "hidden", "not_visible"),
        ("craft", {"item": "minecraft:chest", "count": 1}, "no_table", "needs_crafting_table"),
        ("craft", {"item": "minecraft:oak_planks", "count": 4}, "missing", "missing_items"),
        ("place", {"x": 1, "y": -60, "z": 1, "item": "minecraft:dirt"}, "occupied", "target_occupied"),
        ("place", {"x": 1, "y": -56, "z": 1, "item": "minecraft:dirt"}, "unsupported", "no_support"),
        ("place", {"x": 12, "y": -60, "z": 0, "item": "minecraft:dirt"}, "far", "out_of_reach"),
        ("place_section", {"schematic": "s1_deck", "origin": [2, -61, 0]}, "short", "missing_items"),
        ("chest_put", {"pos": list(CHEST), "item": "minecraft:dirt", "count": 4}, "denied", "acl_denied"),
        ("chest_take", {"pos": list(CHEST), "item": "minecraft:dirt", "count": 4}, "denied", "acl_denied"),
        ("give", {"target": "bob", "item": "minecraft:dirt", "count": 4}, "far", "out_of_reach"),
    ]
    cases.extend(Case(skill, f"{skill}/infeasible/{fixture}", args, fixture, code, False)
                 for skill, args, fixture, code in negatives)
    return cases


def deck_blocks() -> list[dict]:
    """Return the 3-by-1-by-5 deck in its supported placement order."""
    return [{"dx": offset_x, "dy": 0, "dz": offset_z, "block": "minecraft:oak_planks"}
            for offset_z in range(5) for offset_x in range(3)]


def write_schematic(directory: Path, name: str) -> None:
    """Write the deck schematic using the existing JSON format and ordering."""
    destination = directory / "world/simharness/schematics" / f"{name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"blocks": deck_blocks()}, indent=2) + "\n", encoding="utf-8")


def command(client: harness.Client, text: str) -> dict:
    """Send the supplied server command and return its response."""
    return client.request("command", {"command": text})


def teleport(client: harness.Client, name: str, position: tuple) -> None:
    """Teleport a named bot to the fixture position with zero rotation."""
    command(client, f"tp {name} {' '.join(str(value) for value in position)} 0 0")


def inventory(client: harness.Client, name: str = "alice") -> Counter:
    """Read and validate a bot's inventory, returning counts by item."""
    entries = client.request("bot_state", {"name": name}).get("inventory")
    return item_counts(entries)


def item_counts(entries) -> Counter:
    """Count valid inventory entries, rejecting malformed or duplicate slots."""
    if not isinstance(entries, list):
        raise harness.ProtocolError(f"Invalid inventory: {entries!r}")
    counts = Counter()
    slots = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not nonnegative_integer(entry.get("slot"))
                or entry["slot"] in slots or not isinstance(entry.get("item"), str)
                or not entry["item"].startswith("minecraft:") or type(entry.get("count")) is not int
                or entry["count"] <= 0):
            raise harness.ProtocolError(f"Invalid inventory entry: {entry!r}")
        slots.add(entry["slot"])
        counts[entry["item"]] += entry["count"]
    return counts


def item_entities(client: harness.Client) -> list[dict]:
    """Validate a regional entity dump and return its dropped-item entities."""
    entries = client.request("dump_entities", {"region": REGION}).get("entities")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) or "type" not in entry for entry in entries):
        raise harness.ProtocolError(f"Invalid entity dump: {entries!r}")
    return [entry for entry in entries if entry["type"] == "minecraft:item"]


def finish_skill(client: harness.Client, name: str, skill: str, skill_id, *, cap: int = CASE_TICK_CAP) -> dict:
    """Step a named skill to a terminal status within the existing tick cap."""
    current = harness.game_time(client.request("ping"))
    deadline = current + cap
    while True:
        status = validate_status(client.request("skill_status", {"name": name}), skill=skill, skill_id=skill_id)
        check(status["state"] != "idle", "Accepted skill became idle without a terminal outcome")
        if status["state"] != "running":
            return status
        check(current < deadline, f"gate_timeout: {name}/{skill} still running after {cap} ticks; {status}")
        following = current + 20
        check(harness.game_time(client.request("step", {"n": 20})) == following, "step did not advance 20 ticks")
        current = following


def start_and_finish(client: harness.Client, name: str, skill: str, args: dict) -> dict:
    """Require skill acceptance and return its validated terminal status."""
    started = client.request("skill_start", {"name": name, "skill": skill, "args": args})
    check(validate_start(started), f"Prerequisite {skill} rejected: {started}")
    status = finish_skill(client, name, skill, started["skillId"])
    check(status["state"] == "succeeded", f"Prerequisite {skill} failed: {status}")
    return status


def reset_arena(client: harness.Client) -> None:
    """Restore the common physical arena and bot state before a case."""
    for name in ("alice", "bob"):
        client.request("skill_cancel", {"name": name})
        if inventory(client, name):
            command(client, f"clear {name}")
    if item_entities(client):
        command(client, "kill @e[type=minecraft:item,x=-32,y=-64,z=-32,dx=63,dy=14,dz=63]")
    command(client, "setblock -32 -60 -32 minecraft:barrier")
    command(client, "fill -32 -60 -32 31 -55 31 minecraft:air")
    command(client, "setblock -32 -61 -32 minecraft:barrier")
    command(client, "fill -32 -61 -32 31 -61 31 minecraft:stone")
    teleport(client, "alice", (0.5, -60, 0.5))
    teleport(client, "bob", (-8.5, -60, -8.5))
    before = harness.game_time(client.request("ping"))
    check(harness.game_time(client.request("step", {"n": 20})) == before + 20, "settling step advanced incorrectly")


def supply(client: harness.Client, item: str, count: int, name: str = "alice") -> None:
    """Give a named bot exactly the requested item count."""
    client.request("give", {"name": name, "item": item, "count": count})


def prepare_case(client: harness.Client, case: Case) -> dict:
    """Reset and populate a case fixture, returning effect-check context."""
    reset_arena(client)
    args, fixture, skill = case.args, case.fixture, case.skill
    context = {}
    if skill == "move_to":
        if fixture == "wall_gap":
            command(client, "fill 4 -60 -12 4 -56 12 minecraft:stone")
            command(client, "fill 4 -60 -1 4 -58 1 minecraft:air")
        elif fixture == "step":
            command(client, "fill 3 -60 -12 5 -60 12 minecraft:stone")
        elif fixture == "sealed":
            command(client, "fill 7 -60 -1 9 -56 1 minecraft:stone")
            command(client, "fill 8 -60 0 8 -59 0 minecraft:air")
        elif fixture == "water":
            command(client, "fill 7 -60 -1 9 -60 1 minecraft:water")
    elif skill in {"mine", "collect"}:
        material = fixture if fixture in {"dirt", "stone", "oak_log", "oak_planks", "bedrock"} else "dirt"
        target = args if skill == "mine" else {"x": 0, "y": -60, "z": 3}
        command(client, f"setblock {target['x']} {target['y']} {target['z']} minecraft:{material}")
        if material == "stone":
            supply(client, "minecraft:stone_pickaxe", 1)
            client.request("bot_select", {"name": "alice", "slot": 0})
        if fixture == "hidden":
            command(client, "fill -3 -60 2 3 -55 2 minecraft:stone")
        context["drop"] = "minecraft:cobblestone" if material == "stone" else f"minecraft:{material}"
        if skill == "collect":
            start_and_finish(client, "alice", "mine", target)
            check(bool(item_entities(client)), "mine-then-collect fixture produced no real item entities")
    elif skill == "craft":
        item, count = args["item"], args["count"]
        if fixture != "missing":
            ingredient, amount = {
                "minecraft:oak_planks": ("minecraft:oak_log", math.ceil(count / 4)),
                "minecraft:stick": ("minecraft:oak_planks", 2 * math.ceil(count / 4)),
                "minecraft:crafting_table": ("minecraft:oak_planks", 4 * count),
                "minecraft:chest": ("minecraft:oak_planks", 8 * count),
            }[item]
            supply(client, ingredient, amount)
        if fixture == "table":
            command(client, "setblock 2 -60 0 minecraft:crafting_table")
    elif skill == "place":
        supply(client, args["item"], 1)
        if fixture == "occupied":
            command(client, f"setblock {args['x']} {args['y']} {args['z']} minecraft:stone")
    elif skill == "place_section":
        command(client, "fill 2 -62 0 4 -61 4 minecraft:air")
        supply(client, "minecraft:oak_planks", 14 if fixture == "short" else 15)
        if fixture.startswith("deck_"):
            teleport(client, "alice", (0.5, -60, int(fixture[-1]) + 0.5))
    elif skill in {"chest_put", "chest_take"}:
        command(client, "setblock 0 -60 2 minecraft:chest")
        teleport(client, "bob", (1.5, -60, 0.5))
        client.request("acl_set", {"pos": CHEST, "allowed": ["bob"] if fixture == "denied" else ["alice", "bob"]})
        if skill == "chest_put":
            supply(client, args["item"], args["count"])
        else:
            command(client, f"item replace block 0 -60 2 container.0 with {args['item']} {args['count']}")
        context["chest"] = item_counts(client.request("open_container", {"name": "bob", "pos": CHEST}).get("items"))
    elif skill == "give":
        teleport(client, "bob", (6.5 if fixture == "far" else 2.0, -60, 0.5))
        supply(client, args["item"], args["count"])
    context["inventory"] = {name: inventory(client, name) for name in ("alice", "bob")}
    return context


def fingerprint(client: harness.Client) -> dict:
    """Read game time and existing component hashes without advancing ticks."""
    current = harness.game_time(client.request("ping"))
    return harness.hash_components(client.request("hash", {"region": REGION}), current)


def verify_effect(client: harness.Client, case: Case, context: dict, status: dict) -> None:
    """Check the case's physical effect and inventory changes after success."""
    args, skill = case.args, case.skill
    after = inventory(client)
    before = context["inventory"]["alice"]
    if skill == "move_to":
        position = client.request("bot_state", {"name": "alice"})
        coordinates = position.get("pos")
        check(isinstance(coordinates, list) and len(coordinates) == 3
              and all(type(value) in (int, float) and math.isfinite(value) for value in coordinates),
              f"Invalid bot position: {position}")
        check(position.get("onGround") is True and abs(coordinates[1] - args["y"]) < 0.6
              and math.hypot(coordinates[0] - args["x"], coordinates[2] - args["z"]) <= args.get("tolerance", 0.5),
              f"move_to claimed success away from target: {position}")
    elif skill in {"mine", "place"}:
        block = "minecraft:air" if skill == "mine" else args["item"]
        command(client, f"execute if block {args['x']} {args['y']} {args['z']} {block}")
        if skill == "place":
            check(before[args["item"]] - after[args["item"]] == 1, "place did not consume exactly one item")
        else:
            check(bool(item_entities(client)) or after[context["drop"]] > before[context["drop"]],
                  "mine claimed success without a vanilla drop")
    elif skill == "collect":
        check(after[context["drop"]] > before[context["drop"]], "collect did not pick up mined items")
        check(not item_entities(client), "collect left mined items behind")
    elif skill == "craft":
        check(after[args["item"]] - before[args["item"]] >= args["count"], "craft did not produce requested items")
        check(any(after[item] < amount for item, amount in before.items() if item != args["item"]),
              "craft did not consume ingredients")
    elif skill == "place_section":
        for block in deck_blocks():
            target = [args["origin"][index] + block[key] for index, key in enumerate(("dx", "dy", "dz"))]
            command(client, f"execute if block {' '.join(map(str, target))} minecraft:oak_planks")
        check(before["minecraft:oak_planks"] - after["minecraft:oak_planks"] == 15,
              "deck did not consume exactly 15 planks")
        check(status["progress"].get("placed") == 15 and status["progress"].get("total") == 15,
              "deck has incorrect placed/total progress")
    elif skill in {"chest_put", "chest_take", "give"}:
        other = (inventory(client, "bob") if skill == "give" else
                 item_counts(client.request("open_container", {"name": "bob", "pos": CHEST}).get("items")))
        previous = context["inventory"]["bob"] if skill == "give" else context["chest"]
        delta = args["count"] * (1 if skill == "chest_take" else -1)
        check(after[args["item"]] - before[args["item"]] == delta
              and other[args["item"]] - previous[args["item"]] == -delta, "transfer did not conserve exact item counts")
    elif skill == "wait":
        check(status["endedAt"] - status["startedAt"] == args["ticks"], "wait used an incorrect number of ticks")


def run_case(client: harness.Client, case: Case) -> dict:
    """Run dry and real starts in order, retaining verdicts and case errors."""
    result = {"case": asdict(case), "dry": None, "real": None, "status": None,
              "agreement": False, "dry_run_pure": False, "success": False, "errors": []}
    try:
        context = prepare_case(client, case)
        payload = {"name": "alice", "skill": case.skill, "args": case.args}
        before = fingerprint(client)
        result["dry"] = client.request("skill_start", {**payload, "dry_run": True})
        after = fingerprint(client)
        result["dry_run_pure"] = before == after
        result["real"] = client.request("skill_start", payload)
        feasible = validate_dry(result["dry"])
        accepted = validate_start(result["real"])
        result["agreement"] = feasible == accepted
        if not result["dry_run_pure"]:
            result["errors"].append(f"dry-run mutated state: before={before}, after={after}")
        if not result["agreement"]:
            result["errors"].append("dry-run feasibility disagrees with real acceptance")
        if feasible != (case.expected_reason is None):
            result["errors"].append("unexpected feasibility for fixture")
        if not accepted:
            if case.expected_reason is None or reason(result["real"]) != case.expected_reason:
                result["errors"].append(f"unexpected rejection: {result['real']}")
            if not feasible and result["dry"]["reason"] != result["real"]["reason"]:
                result["errors"].append("dry-run and rejection reasons disagree")
            return result
        result["status"] = finish_skill(client, "alice", case.skill, result["real"]["skillId"])
        if result["status"]["state"] == "succeeded":
            verify_effect(client, case, context, result["status"])
            result["success"] = True
        if case.expected_reason is not None:
            result["errors"].append(f"infeasible case accepted; expected {case.expected_reason}")
    except ERRORS as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def summarize(results: list[dict], cases: list[Case]) -> dict:
    """Compute exact coverage, agreement, and success thresholds per skill."""
    summary = {}
    for skill in SKILLS:
        rows = [result for result in results if result["case"]["skill"] == skill]
        expected = [case for case in cases if case.skill == skill]
        accepted = [row for row in rows if isinstance(row["real"], dict) and row["real"].get("accepted") is True]
        succeeded = sum(row["success"] for row in accepted)
        complete = Counter(row["case"]["label"] for row in rows) == Counter(case.label for case in expected)
        agreement = sum(row["agreement"] is True for row in rows)
        passed = (bool(expected) and complete and agreement == len(expected) and bool(accepted)
                  and succeeded * 100 >= len(accepted) * 95
                  and all(row["dry_run_pure"] is True and not row["errors"] for row in rows))
        summary[skill] = {"cases": len(rows), "expected": len(expected), "accepted": len(accepted),
                          "succeeded": succeeded, "agreement": agreement / len(expected) if expected else 0,
                          "success_rate": succeeded / len(accepted) if accepted else 0, "passed": passed}
    return summary


def print_report(results: list[dict], summary: dict) -> None:
    """Print case diagnostics followed by the ordered per-skill verdicts."""
    for row in results:
        if row["errors"] or not row["success"]:
            status = row["status"] or row["real"] or row["dry"] or {}
            label = "REJECT (expected)" if not row["errors"] and row["case"]["expected_reason"] else "FAIL"
            print(f"{label} {row['case']['label']}: reason={status.get('reason', 'missing')} "
                  f"state={status.get('state', 'not_started')} errors={row['errors']}")
    for skill, value in summary.items():
        print(f"{'PASS' if value['passed'] else 'FAIL'} S1 {skill}: cases={value['cases']}/{value['expected']} "
              f"agreement={value['agreement']:.2%} success={value['succeeded']}/{value['accepted']} "
              f"({value['success_rate']:.2%})")


class S1Experiment:
    """Own seeded case execution and reporting without changing tick barriers."""

    def __init__(self, args):
        """Keep parsed CLI options for the skill experiment."""
        self._args = args

    def setup(self) -> None:
        """Validate the seed and destination before cloning and generating cases."""
        args = self._args
        if args.base_seed != SEED:
            raise ValueError(f"S1 requires --base-seed {SEED}")
        self._directory = args.runs_dir / "s1"
        if self._directory.exists() or self._directory.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {self._directory}")
        harness.clone_apfs(args.runs_dir / "base", self._directory)
        self._cases, self._results, self._errors = generate_cases(), [], []

    def run(self) -> None:
        """Run the existing case driver and retain setup or lifecycle failures."""
        try:
            write_schematic(self._directory, "s1_deck")
            with harness.Server(self._directory, **harness.server_options(self._args)) as server:
                harness.enable_lockstep(server.client)
                server.client.request("config_set", {"reach": 4.5})
                for name in ("alice", "bob"):
                    server.client.request("spawn_bot", {"name": name, "x": 0.5, "y": -60, "z": 0.5,
                                                        "yaw": 0.0, "pitch": 0.0})
                for case in self._cases:
                    self._results.append(run_case(server.client, case))
                server.client.request("save_snapshot")
        except ERRORS as error:
            self._errors.append(f"S1 setup/shutdown: {type(error).__name__}: {error}")

    def check(self) -> None:
        """Account for unrun cases and apply the unchanged per-skill thresholds."""
        completed = {row["case"]["label"] for row in self._results}
        for case in self._cases:
            if case.label not in completed:
                self._results.append({"case": asdict(case), "dry": None, "real": None, "status": None,
                                      "agreement": False, "dry_run_pure": False, "success": False,
                                      "errors": ["case not run: setup/server lifecycle failed"]})
        self._summary = summarize(self._results, self._cases)
        self._passed = not self._errors and all(value["passed"] for value in self._summary.values())

    def report(self) -> int:
        """Write the original report format, then print diagnostics and verdicts."""
        report = {"seed": SEED, "generated_per_skill": CASES_PER_SKILL, "passed": self._passed,
                  "summary": self._summary, "errors": self._errors, "cases": self._results}
        (self._directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
        print_report(self._results, self._summary)
        for error in self._errors:
            print(f"FAIL {error}")
        print(f"{'PASS' if self._passed else 'FAIL'} S1")
        return 0 if self._passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the S1 experiment phases."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = S1Experiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    experiment.check()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
