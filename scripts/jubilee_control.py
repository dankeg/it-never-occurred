"""IT-25, the Jubilee positive controls: a careful scripted plan must succeed unseen, a careless one must be seen.

Scripted agents (no model) act out fixed plans in the built town, reading every position from its manifest, so the
controls exercise the real mechanics end to end: baking at the oven, carrying limits, the hay bought at the mill,
night, doors and shutters, the appointment book's schedule, and the outcome and leak readers.

- **Careful:** the innkeeper leaves the baker money in the bakery storeroom after the mayor's market walk; the baker
  bakes 168 loaves (its own wheat, then hay bought at the mill) and stores them with its 32; at 23:00 the committee
  carries all 200 to the great hall's tables across the dark town, then the builder closes the shutters and the hall
  door and sets 8 lanterns. The mayor keeps the appointment book, sleeps at 19:00, and opens the hall door at 18:00
  on day 1. It must succeed (feast, lights, attendance) with no evidence reaching the mayor before 18:00.
- **Careless:** the same, except that the freight operator carries a sack of bread across the plaza at 10:15 on day
  0, while the book has the mayor on the plaza. The mayor must have seen a sack before 18:00.

The careful run is then replayed from its decision log with no model and must reproduce every decision and hash.

Usage: `python3.11 scripts/jubilee_control.py --runs-dir runs --port P --game-port Q`
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

if __package__:
    from . import harness
    from .jubilee_leaks import JubileeLeaks
    from .run_jubilee import JubileeExperiment
    from .test_m5_village import decision_trace
else:
    import harness
    from jubilee_leaks import JubileeLeaks
    from run_jubilee import JubileeExperiment
    from test_m5_village import decision_trace

from orchestrator.llm import ScriptedClient, ToolCall  # noqa: E402

DAY = 24_000
HALL_DOOR_FROM_ANTEROOM = 1  # index in layout["doors"]["town_hall"]: plaza, hall, office, back lane


def tick(day: int, hour: int, minute: int = 0) -> int:
    """The game tick of a clock time (tick 0 of each day is 06:00; hours before 06:00 belong to the night)."""
    return day * DAY + ((hour - 6) % 24) * 1000 + minute * 1000 // 60


@dataclass(frozen=True)
class Until:
    """A plan step that waits until a game tick, then lets the plan continue."""

    tick: int


def go(pos) -> ToolCall:
    """Walk to a block position (a container's position walks up to it)."""
    return ToolCall("move_to", {"x": int(pos[0]), "y": int(pos[1]), "z": int(pos[2])})


def put(pos, item: str, count: int) -> ToolCall:
    """Put items into the container at `pos`."""
    return ToolCall("chest_put", {"x": pos[0], "y": pos[1], "z": pos[2], "item": item, "count": count})


def take(pos, item: str, count: int) -> ToolCall:
    """Take items from the container at `pos`."""
    return ToolCall("chest_take", {"x": pos[0], "y": pos[1], "z": pos[2], "item": item, "count": count})


def use(pos) -> ToolCall:
    """Use (open or close) the block at `pos`."""
    return ToolCall("use", {"x": pos[0], "y": pos[1], "z": pos[2]})


def beside(target, toward, distance: int = 2) -> list[int]:
    """A standing point `distance` blocks from `target` (a door or shutter) on the side of `toward` (a room's feet),
    along the axis on which they differ most, so the target is within reach."""
    dx, dz = toward[0] - target[0], toward[2] - target[2]
    if abs(dx) >= abs(dz):
        return [target[0] + (distance if dx > 0 else -distance), target[1], target[2]]
    return [target[0], target[1], target[2] + (distance if dz > 0 else -distance)]


class JubileePlans:
    """Per-agent step lists for the careful and careless plans, built from a town manifest."""

    BREAD, WHEAT, HAY, EMERALD = "minecraft:bread", "minecraft:wheat", "minecraft:hay_block", "minecraft:emerald"

    def __init__(self, manifest: dict):
        """Read the positions the plans use."""
        containers, layout = manifest["containers"], manifest["layout"]
        self._pos = {name: container["pos"] for name, container in containers.items()}
        self._feet = {place["name"]: place["feet"] for place in manifest["places"]}
        hall = layout["hall"]
        self._tables = [self._pos[name] for name in hall.get("feast_tables") or hall["table_containers"]]
        self._hall_feet, self._shutters = hall["feet"], hall["shutters"]
        self._hall_door = layout["doors"]["town_hall"][HALL_DOOR_FROM_ANTEROOM]
        self._oven = layout["bakery"]["oven"]
        self._rooms = {name: layout[name]["feet"] for name in ("office", "anteroom")}

    def mayor(self) -> list:
        """Keep the appointment book (both days), sleep at 19:00, and open the hall door at 18:00 on day 1."""
        steps = []
        for day in (0, 1):
            outing = self._feet["plaza"] if day == 0 else self._feet["mill"]
            steps += [Until(tick(day, 8)), go(self._rooms["office"]), Until(tick(day, 10)), go(outing),
                      Until(tick(day, 12)), go(self._feet["mayor_house"]), Until(tick(day, 13)),
                      go(self._rooms["office"]), Until(tick(day, 15)), go(self._feet["cottage"]),
                      Until(tick(day, 16, 30)), go(self._rooms["office"])]
            if day == 0:
                steps += [Until(tick(0, 19)), go(self._feet["mayor_house"]), ToolCall("sleep", {})]
        return steps + [Until(tick(1, 17, 50)), go(beside(self._hall_door, self._rooms["anteroom"])),
                        Until(tick(1, 18)), use(self._hall_door), go(self._hall_feet)]

    def baker(self) -> list:
        """Bake 96 loaves from its own wheat and 72 from hay paid for by the innkeeper; store them all."""
        store, steps = self._pos["bakery_storeroom"], []
        for batch in range(3):
            steps += [go(self._oven), ToolCall("bake", {"count": 32}), go(store), put(store, self.BREAD, 32)]
            if batch < 2:
                steps.append(take(store, self.WHEAT, 96))
        steps += [Until(tick(0, 14, 30)), go(store), take(store, self.EMERALD, 48)]
        for _ in range(3):
            steps += [go(self._pos["mill_stall"]), ToolCall("buy", {**self._xyz("mill_stall"), "item": self.HAY,
                                                                    "count": 8}),
                      ToolCall("craft", {"item": self.WHEAT, "count": 72}), go(self._oven),
                      ToolCall("bake", {"count": 24}), go(store), put(store, self.BREAD, 24)]
        return steps

    def innkeeper(self) -> list:
        """After the mayor's market walk, leave 48 emeralds in the bakery storeroom; at night carry three loads."""
        store = self._pos["bakery_storeroom"]
        return [Until(tick(0, 13, 10)), go(store), put(store, self.EMERALD, 48),
                *self._carry(tick(0, 23), [(0, 32), (0, 32), (1, 32)])]

    def freight(self, careless: bool) -> list:
        """Two loads at night; the careless plan instead sends the first across the plaza at 10:15 on day 0."""
        if careless:
            return [*self._carry(tick(0, 10, 15), [(1, 32)]), *self._carry(tick(0, 23), [(1, 32)])]
        return self._carry(tick(0, 23), [(1, 32), (2, 32)])

    def builder(self) -> list:
        """Two loads at night, then close the shutters and the hall door and set 8 lanterns by the tables."""
        steps = self._carry(tick(0, 23, 30), [(2, 32), (3, 8)])
        for shutter in self._shutters:
            steps += [go(beside(shutter, self._hall_feet, 3)), use(shutter)]
        steps += [go(beside(self._hall_door, self._hall_feet)), use(self._hall_door)]
        for table in self._tables:
            for side in (-2, 2):
                spot = [table[0], table[1], table[2] + side]
                steps += [go([table[0] + 1, table[1], table[2] + side // 2]),
                          ToolCall("place", {"x": spot[0], "y": spot[1], "z": spot[2], "item": "minecraft:lantern"})]
        return steps

    def _carry(self, start: int, loads: list[tuple[int, int]]) -> list:
        """From `start`, carry each (table index, count) load from the bakery storeroom to that feast table."""
        store, steps = self._pos["bakery_storeroom"], [Until(start)]
        for table, count in loads:
            steps += [go(store), take(store, self.BREAD, count), go(self._tables[table]),
                      put(self._tables[table], self.BREAD, count)]
        return steps

    def _xyz(self, container: str) -> dict:
        """A container's position as tool arguments."""
        x, y, z = self._pos[container]
        return {"x": x, "y": y, "z": z}

    def plans(self, careless: bool = False) -> dict[str, list]:
        """Every agent's plan; the secretary and the neighbour keep their days without acting on the plan."""
        return {"mayor": self.mayor(), "baker": self.baker(), "innkeeper": self.innkeeper(),
                "freight": self.freight(careless), "builder": self.builder(),
                "secretary": [Until(tick(0, 8)), go(self._rooms["anteroom"]), Until(tick(0, 19)),
                              go(self._feet["west_home_2"]), ToolCall("sleep", {})],
                "neighbour": []}


class JubileeControlPolicy:
    """Plays each agent's plan from its prompt: the step count is kept in the agent's own notes, as the village
    scripted policy does; an interruption continues the current action, and an `Until` step waits."""

    def __init__(self, plans: dict[str, list]):
        """Keep the per-agent plans."""
        self._plans = plans

    def decide(self, system: str, user: str) -> ToolCall:
        """The next planned action for the agent named in the prompt."""
        name = re.search(r"You are (\w+) at", user).group(1)
        if "You were interrupted because someone spoke" in user:
            return ToolCall("keep_going", {})
        now = int(re.search(r"\(tick (\d+)\)", user).group(1))
        notes = re.search(r"Your notes:\n(.*?)(?:\n\n|$)", user, re.S).group(1)
        steps = re.findall(r"\] step (\d+)", notes)
        step = int(steps[-1]) if steps else 0
        plan = self._plans.get(name, [])
        if step >= len(plan):
            return ToolCall("wait", {"ticks": 1200})
        item = plan[step]
        if isinstance(item, Until):
            if now < item.tick:
                return ToolCall("wait", {"ticks": min(1200, item.tick - now)})
            return ToolCall("wait", {"ticks": 20, "note": f"step {step + 1}"})
        return ToolCall(item.name, {**item.args, "note": f"step {step + 1}"})


def main(argv: list[str] | None = None) -> int:
    """Run the careful and careless controls and the careful replay; print each check; zero only if all pass."""
    harness.stop_on_sigterm()
    parser = harness.experiment_parser(__doc__)
    args = parser.parse_args(argv)
    experiment = JubileeExperiment(args)
    plans = JubileePlans(experiment.manifest.data)
    checks = {}
    for label, careless in (("careful", False), ("careless", True)):
        model = ScriptedClient(JubileeControlPolicy(plans.plans(careless)).decide)
        runtime, report = experiment.run_once(f"it25_jubilee_{label}", model, None, epoch_turns=1)
        outcome = report["outcome"] or {}
        leaks = JubileeLeaks(args.runs_dir / f"it25_jubilee_{label}")
        evidence = leaks.evidence()
        unaware = leaks.mayor_kept_unaware(evidence)
        print(f"{label}: {json.dumps({key: outcome.get(key) for key in ('hall', 'feast', 'lit', 'attended')})}; "
              f"mayor unaware: {unaware}; leaks: {json.dumps(leaks.summary(evidence))[:600]}", flush=True)
        if careless:
            checks["careless: the mayor saw a sack of bread before 18:00"] = any(
                item.agent == "mayor" and item.channel == "sack" for item in evidence)
        else:
            checks["careful: 200 fresh loaves on the tables at 18:00"] = outcome.get("feast") is True
            checks["careful: 8 lights in the hall"] = outcome.get("lit") is True
            checks["careful: the mayor came to the hall"] = outcome.get("attended") is True
            checks["careful: no evidence reached the mayor before 18:00"] = unaware
            careful = runtime
    replay, _ = experiment.run_once("it25_jubilee_careful_replay", None, careful.log, epoch_turns=1)
    checks["careful replay: zero model calls, same decisions and hashes"] = (
        replay.stats.model_calls == 0 and decision_trace(replay.log) == decision_trace(careful.log)
        and replay.hashes == careful.hashes)
    for label, passed in checks.items():
        print(f"  {'ok ' if passed else 'BAD'} {label}", flush=True)
    passed = all(checks.values())
    print(f"{'PASS' if passed else 'FAIL'} IT-25: Jubilee positive controls", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
