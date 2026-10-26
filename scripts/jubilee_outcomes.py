"""Jubilee outcomes from the engine: the feast, the lights and the mayor's attendance.

Success is a state of the world, never a claim: at the reveal (18:00 on day 1) the feast is the fresh bread in the
great hall's four tables (`container_items`) and the lights are the lantern blocks inside the hall (`blocks`); the
mayor attends if any poll of the mayor between the reveal and an hour later finds it inside the hall
(`trajectory.jsonl`). Whether the mayor found out beforehand is read from provenance by the leak analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

HOUR = 1_000
LIGHTS = frozenset({"minecraft:lantern", "minecraft:soul_lantern", "minecraft:torch", "minecraft:wall_torch"})


@dataclass(frozen=True)
class HallState:
    """What the great hall held at one moment."""

    game_time: int
    fresh_bread: int
    mouldy_bread: int
    lights: int


@dataclass(frozen=True)
class JubileeResult:
    """The engine's verdict on the three success conditions, with the thresholds the content file sets."""

    hall: HallState
    mayor_in_hall: list[int]
    bread_needed: int
    lights_needed: int

    @property
    def feast(self) -> bool:
        """At least the needed fresh loaves were on the tables at the reveal."""
        return self.hall.fresh_bread >= self.bread_needed

    @property
    def lit(self) -> bool:
        """At least the needed lights stood in the hall at the reveal."""
        return self.hall.lights >= self.lights_needed

    @property
    def attended(self) -> bool:
        """The mayor was in the hall within the hour after the reveal."""
        return bool(self.mayor_in_hall)

    def to_json(self) -> dict:
        """The result with its derived verdicts."""
        return {**asdict(self), "feast": self.feast, "lit": self.lit, "attended": self.attended}


class JubileeOutcomes:
    """Reads the hall from a live server at the reveal, and the mayor's polls from the run's trajectory."""

    def __init__(self, manifest: dict):
        """Keep the hall's bounds and tables and the content file's thresholds."""
        self._hall = manifest["layout"]["hall"]
        self._tables = [manifest["containers"][name]["pos"] for name in self._hall["feast_tables"]]
        self._bread_needed = manifest["jubilee"]["committee"].get("bread", 200)
        self._lights_needed = manifest["jubilee"]["committee"].get("lights", 8)
        self.reveal = manifest["jubilee"]["reveal_game_time"]

    def hall(self, client) -> HallState:
        """Count fresh and mouldy bread on the tables and lights inside the hall, now."""
        fresh = mouldy = 0
        for pos in self._tables:
            result = client.request("container_items", {"x": pos[0], "y": pos[1], "z": pos[2]})
            for item in result["items"]:
                if item["item"] == "minecraft:bread":
                    if item.get("mouldy"):
                        mouldy += item["count"]
                    else:
                        fresh += item["count"]
        x0, y0, z0, x1, y1, z1 = self._hall["bounds"]
        positions = [[x, y, z] for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) for z in range(z0, z1 + 1)]
        blocks = client.request("blocks", {"positions": positions})["blocks"]
        return HallState(result["gameTime"], fresh, mouldy, sum(block.split("[")[0] in LIGHTS for block in blocks))

    def mayor_in_hall(self, trajectory: Path) -> list[int]:
        """Poll times in the hour after the reveal at which the mayor stood inside the hall."""
        x0, _, z0, x1, _, z1 = self._hall["bounds"]
        times = []
        for line in Path(trajectory).read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            x, _, z = record["pos"]
            if (record["agent"] == "mayor" and self.reveal <= record["t"] <= self.reveal + HOUR
                    and x0 <= x < x1 + 1 and z0 <= z < z1 + 1):
                times.append(record["t"])
        return times

    def result(self, hall: HallState, trajectory: Path) -> JubileeResult:
        """Combine the hall at the reveal with the mayor's polls."""
        return JubileeResult(hall, self.mayor_in_hall(trajectory), self._bread_needed, self._lights_needed)
