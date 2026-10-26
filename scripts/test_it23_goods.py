"""IT-23: opt-in loads, fresh bread, baking, owned stalls and actual villager market sales ($0).

For Claude to run, never as part of pytest: clone runs/village_sabotage, stage a smoker and stalls in a quiet
corner, and use model-facing tools. Check exact-count carrying refusals and hints, per-stack freshness and mouldy
eating refusal, count-only baking (four loaves, twelve wheat, 160 ticks, visible activity and lit/unlit smoker),
cancellation and late failure, owner stocking/takings and non-owner buy-only access. Actual adult market arrivals
must produce ordered purchases or exact sold_out records, consume freshest nonmouldy stock first, and pay per unit.
Compare flags-off configuration/state/hashes with an untouched checkpoint branch, then replay an in-flight bake
checkpoint with identical hashes, inventory and events. Existing fixtures are never modified.

Pass: every check is ok. Artifacts: --runs-dir/it23_goods*; uses only --port + 0..7, sequentially.
Run: python scripts/test_it23_goods.py --runs-dir runs --port 25623
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
from orchestrator.render import ObservationRenderer, stack_text  # noqa: E402
from orchestrator.runtime import failure_hint  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import ToolTranslator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOT = "builder"
OBSERVER = "investigator"
BREAD = "minecraft:bread"
WHEAT = "minecraft:wheat"
EMERALD = "minecraft:emerald"
OVEN = [100, -60, 101]
STALL = [101, -60, 100]
EMPTY_STALL = [101, -60, 102]
CHEST = [99, -60, 100]
OFF = {"carry_limits": {}, "fresh_days": None, "villager_purchases": []}
ON = {"carry_limits": {BREAD: 32, WHEAT: 96}, "fresh_days": 2, "bake_ticks": 40, "villager_purchases": []}


def counts(items: list[dict]) -> dict[str, int]:
    """Sum stacks without conflating their component-sensitive freshness descriptions."""
    result: dict[str, int] = {}
    for item in items:
        result[item["item"]] = result.get(item["item"], 0) + item["count"]
    return result


def market_arrivals(npcs: dict, since: int, until: int) -> list[tuple[int, str]]:
    """Read completed adult market visits, not work at the market, failed paths or departure intentions."""
    result = []
    for day in npcs["days"].values():
        for trip in day["trips"]:
            arrival = trip.get("arriveTick")
            if (arrival is not None and since <= arrival <= until and trip["activity"] == "market"
                    and trip["to"] == "market" and npcs["npcs"][trip["npc"]]["role"] == "adult"):
                result.append((arrival, trip["npc"]))
    return sorted(result)


def sale_arrivals(events: list[dict], stall: list[int]) -> list[tuple[int, str]]:
    """Keep event order (do not sort away a real same-tick ordering bug)."""
    shop = "container:" + ",".join(map(str, stall))
    return [(event["gameTime"], event["buyer"] if event["type"] == "purchase" else event["npc"])
            for event in events if (event["type"] == "purchase" and event.get("shop") == shop
                                   or event["type"] == "sold_out" and event.get("stall") == stall)]


class IT23GoodsExperiment:
    """Own isolated, sequential sessions and retain exact replay evidence for the goods gate."""

    def __init__(self, args):
        """Validate ports and load the old village manifest without starting any server."""
        self.args = args
        self.options = port_options(args)
        self.directory = args.runs_dir / "it23_goods"
        self.active_directory = self.directory
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.region = json.loads((ROOT / "scenario/village_sabotage.json").read_text(
            encoding="utf-8"))["layout"]["fixture_hash_region"]
        self.translator = ToolTranslator()
        self.renderer = ObservationRenderer()
        self.checks: list[dict] = []
        self.client = None
        self.bake_start = 0
        self.bake_cursor = 0

    def _check(self, label: str, passed: bool, detail: str = "") -> None:
        """Print and retain one evidence-bearing assertion."""
        self.checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  {'ok ' if passed else 'BAD'} {label}" + (f": {detail}" if detail else ""), flush=True)

    def _now(self) -> int:
        """Use the server's game time, never wall time or dayTime."""
        return harness.game_time(self.client.request("ping"))

    def _step(self, ticks: int) -> None:
        """Advance only requested ticks in bounded lockstep batches."""
        while ticks:
            batch = min(ticks, 200)
            before = self._now()
            self.client.request("step", {"n": batch})
            if self._now() != before + batch:
                raise harness.ProtocolError("step advanced by an unexpected number of ticks")
            ticks -= batch

    def _command(self, command: str) -> dict:
        """Stage this disposable clone using administrative commands, not agent abilities."""
        return self.client.request("command", {"command": command})

    def _fill(self, corner: tuple[int, int, int], other: tuple[int, int, int], block: str) -> None:
        """Fill a box, skipping vanilla's fill when every block already matches (it fails when nothing changes)."""
        (x0, y0, z0), (x1, y1, z1) = corner, other
        positions = [[x, y, z] for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) for z in range(z0, z1 + 1)]
        found = self.client.request("blocks", {"positions": positions})["blocks"]
        if any(block_id.split("[")[0] != block for block_id in found):
            self._command(f"fill {x0} {y0} {z0} {x1} {y1} {z1} {block}")

    def _inventory(self, name: str = BOT) -> list[dict]:
        """Read a bot's component-aware public inventory."""
        return self.client.request("bot_state", {"name": name})["inventory"]

    def _clear(self, name: str = BOT) -> None:
        """Remove existing stacks without relying on vanilla clear succeeding on an empty inventory."""
        self.client.request("skill_cancel", {"name": name})
        for item in self._inventory(name):
            slot = item["slot"]
            if not 0 <= slot < 36:
                raise harness.HarnessError(f"Unexpected inventory slot {slot}")
            address = f"hotbar.{slot}" if slot < 9 else f"inventory.{slot - 9}"
            self._command(f"item replace entity {name} {address} with minecraft:air")

    def _item(self, target: str, slot: int, item: str, count: int, day: int | None = None) -> None:
        """Seed a known entity hotbar or container slot, with the real custom-data baking stamp."""
        if day is not None:
            item += f"[minecraft:custom_data={{simharness_baked_day:{day}}}]"
        address = f"block {target} container.{slot}" if " " in target else f"entity {target} hotbar.{slot}"
        self._command(f"item replace {address} with {item} {count}")

    def _items(self, pos: list[int]) -> list[dict]:
        """Inspect authoritative stall contents without draining agent observations."""
        return self.client.request("container_items", {"pos": pos})["items"]

    def _empty(self, pos: list[int]) -> None:
        """Clear only occupied slots in a staged container, leaving its shop and owner registration intact."""
        for item in self._items(pos):
            self._command(f"item replace block {' '.join(map(str, pos))} container.{item['slot']} with minecraft:air")

    def _observe(self, name: str = BOT) -> dict:
        """Read only what this observer perceives."""
        return self.client.request("observe", {"name": name, "include_terrain": False})

    def _start(self, call: ToolCall, name: str = BOT, *, dry: bool = False) -> dict:
        """Use the model-facing translator, including count-only bake and canonical item IDs."""
        action = self.translator.translate(name, call)
        return self.client.request(action.command, {**action.args, **({"dry_run": True} if dry else {})})

    def _act(self, call: ToolCall, name: str = BOT) -> dict:
        """Bound a skill's execution and retain immediate structured rejections verbatim."""
        started = self._start(call, name)
        if not started.get("accepted"):
            return started
        for _ in range(601):
            status = self.client.request("skill_status", {"name": name})
            if status.get("state") != "running":
                return status
            self._step(1)
        raise harness.HarnessError(f"{name} {call.name} still running after 600 ticks")

    @staticmethod
    def _at(tool: str, pos: list[int], item: str, count: int) -> ToolCall:
        """Make a container tool call in the same coordinates an agent supplies."""
        return ToolCall(tool, {"x": pos[0], "y": pos[1], "z": pos[2], "item": item, "count": count})

    def _saved(self) -> dict:
        """Flush and read canonical harness state for actual trips and persistence assertions."""
        self.client.request("save_snapshot")
        return json.loads((self.active_directory / "world/simharness/harness_state.json").read_text(encoding="utf-8"))

    def _events(self, since: int = 0) -> list[dict]:
        """Read flushed engine events, preserving their original ordering and fields."""
        self.client.request("save_snapshot")
        return [event for line in (self.active_directory / "world/simharness/events.jsonl").read_text(
            encoding="utf-8").split("\n") if line.strip()
                if (event := json.loads(line))["id"] > since]

    def _checkpoint(self, label: str) -> Path:
        """Copy a complete mid-tick-boundary world; refuse any existing destination."""
        self.client.request("save_snapshot")
        path = self.args.runs_dir / f"it23_goods_{label}_checkpoint"
        path.mkdir()
        harness.clone_apfs(self.active_directory / "world", path / "world")
        (path / "orchestrator").mkdir()
        return path

    def _hashes(self, ticks: int = 80) -> list[dict]:
        """Sample every component hash, not just an aggregate or the final tick."""
        result = []
        while ticks:
            batch = min(ticks, 20)
            self._step(batch)
            result.append({"tick": self._now(), "hash": self.client.request("hash", {"region": self.region})})
            ticks -= batch
        return result

    def _lit(self) -> bool:
        """Query the actual smoker block property without advancing time."""
        coords = " ".join(map(str, OVEN))
        try:
            self._command(f"execute if block {coords} minecraft:smoker[lit=true]")
            return True
        except harness.CommandError:
            self._command(f"execute if block {coords} minecraft:smoker[lit=false]")
            return False

    def _stage(self) -> None:
        """Place one visible reachable oven, a supply chest and two stalls away from village traffic."""
        self._fill((97, -61, 97), (106, -61, 106), "minecraft:stone_bricks")
        self._fill((97, -60, 97), (106, -57, 106), "minecraft:air")
        for pos, block in ((OVEN, "smoker"), (STALL, "chest"), (EMPTY_STALL, "chest"), (CHEST, "chest")):
            self._command(f"setblock {' '.join(map(str, pos))} minecraft:{block}")
        for name, pos in ((BOT, "100.5 -60 100.5"), (OBSERVER, "100.5 -60 104.5"),
                          ("security", "102.5 -60 100.5")):
            self._clear(name)
            self._command(f"tp {name} {pos} 0 0")
        self._step(20)
        self.client.request("shop_set", {"pos": STALL, "prices": {BREAD: 3}})
        self.client.request("shop_set", {"pos": EMPTY_STALL, "prices": {BREAD: 5}})
        self.client.request("acl_set", {"pos": STALL, "allowed": None, "owner": BOT})

    def _limits_and_freshness(self) -> None:
        """Check exact-count refusals, useful carrying hints and separate fresh/mouldy stack descriptions."""
        self._item(BOT, 0, BREAD, 30)
        self._item(BOT, 1, EMERALD, 32)
        self._item(" ".join(map(str, CHEST)), 0, BREAD, 10)
        self._item(" ".join(map(str, STALL)), 0, BREAD, 10)
        for tool, pos in (("chest_take", CHEST), ("buy", STALL)):
            call = self._at(tool, pos, "bread", 3)
            before = self._inventory(), self._items(pos)
            rejected = self._act(call)
            self._check(f"{tool} refuses 33 bread with exact structured limit facts",
                        rejected.get("accepted") is False and rejected.get("reason") == "too_heavy"
                        and rejected.get("item") == BREAD and rejected.get("limit") == 32
                        and rejected.get("have") == 30 and rejected.get("remaining") == 2, str(rejected))
            hint = failure_hint(call, "not possible: too_heavy", self._observe(), rejected)
            self._check(f"{tool} exposes a useful hint", all(text in hint for text in ("32", "30", "2 more"))
                        and bool(rejected.get("hint")), hint)
            self._check(f"{tool} refusal changes neither inventory nor chest",
                        before == (self._inventory(), self._items(pos)))
        accepted = self._act(self._at("chest_take", CHEST, "bread", 2))
        self._check("exactly the remaining carrying capacity succeeds", accepted.get("state") == "succeeded"
                    and counts(self._inventory()).get(BREAD) == 32, str(accepted))
        self._clear()
        today = self._now() // 24_000
        self._item(BOT, 0, BREAD, 2, today - 3)
        self._item(BOT, 1, BREAD, 3, today)
        for slot, day in ((0, today - 3), (1, today)):
            self._item(" ".join(map(str, CHEST)), slot, BREAD, 2, day)
        observation = self._observe()
        bread = [item for item in observation["self"]["inventory"] if item["item"] == BREAD]
        self._check("inventory describes each bread stack's own freshness", len(bread) == 2
                    and [(item.get("baked_day"), item.get("mouldy")) for item in bread]
                    == [(today - 3, True), (today, False)], str(bread))
        rendered = self.renderer.render(observation)
        self._check("inventory renders mouldy and fresh bread separately", "mouldy bread" in rendered
                    and f"baked day {today}" in rendered, rendered)
        opened = self._start(ToolCall("open_container", dict(zip(("x", "y", "z"), CHEST))))
        rendered = ", ".join(stack_text(item) for item in opened.get("items", []))
        self._check("open_container renders mouldy and fresh stacks", "mouldy bread" in rendered
                    and f"baked day {today}" in rendered, rendered)
        before = self._inventory()
        rejected = self._act(ToolCall("eat", {"item": "bread"}))
        self._check("eat refuses the first, mouldy loaf without consuming it", rejected.get("reason") == "mouldy"
                    and self._inventory() == before, str(rejected))

    def _bake_setup(self) -> None:
        """Start a count-only four-loaf batch, check visible activity and checkpoint after forty ticks."""
        self._clear()
        self._item(BOT, 0, WHEAT, 12)
        self.bake_cursor = max((event["id"] for event in self._events()), default=0)
        self.bake_start = self._now()
        call = ToolCall("bake", {"count": 4})
        action = self.translator.translate(BOT, call)
        self._check("bake translates count only (no agent-selected oven)", action.args["args"] == {"count": 4})
        estimate = self._start(call, dry=True)
        self._check("four loaves estimate exactly 160 ticks", estimate.get("feasible") is True
                    and estimate.get("estimate", {}).get("ticks") == 160, str(estimate))
        started = self._start(call)
        self._check("count-only bake finds the reachable smoker", started.get("accepted") is True, str(started))
        if not started.get("accepted"):
            raise harness.HarnessError("Cannot checkpoint a bake that did not start")
        self._check("the smoker lights at bake start", self._lit())
        self._step(40)
        observer = self._observe(OBSERVER)
        baker = next((entity for entity in observer.get("entities", []) if entity.get("name") == BOT), {})
        self._check("witness sees baking at the selected oven", baker.get("activity", {}).get("skill") == "bake"
                    and baker["activity"].get("pos") == OVEN, str(baker))
        self._check("witness prompt says baking at the oven", "baking at the oven" in self.renderer.render(observer))
        self._check("wheat and bread are unchanged before completion", counts(self._inventory()) == {WHEAT: 12})

    def _bake_tail(self) -> dict:
        """Continue an existing forty-tick bake without restarting it; retain exact replay evidence."""
        hashes = self._hashes(100)
        hashes.extend(self._hashes(19))
        status = self.client.request("skill_status", {"name": BOT})
        self._check("baking is still running and lit at tick 159", self._now() == self.bake_start + 159
                    and status.get("state") == "running" and self._lit()
                    and counts(self._inventory()) == {WHEAT: 12}, str(status))
        hashes.extend(self._hashes(1))
        status = self.client.request("skill_status", {"name": BOT})
        inventory = self._inventory()
        bread = [item for item in inventory if item["item"] == BREAD]
        self._check("four loaves consume twelve wheat at exactly tick 160", self._now() == self.bake_start + 160
                    and status.get("state") == "succeeded" and counts(inventory) == {BREAD: 4}, str(status))
        self._check("baked bread is stamped today", bool(bread)
                    and all(item.get("baked_day") == self._now() // 24_000 and item.get("mouldy") is False
                            for item in bread),
                    str(bread))
        self._check("the smoker unlights after baking", not self._lit())
        events = self._events(self.bake_cursor)
        baked = [event for event in events if event["type"] == "bake" and event.get("bot") == BOT]
        endings = [event for event in events if event["type"] == "skill_end" and event.get("bot") == BOT
                   and event.get("skill") == "bake"]
        self._check("bake and skill_end each log one successful completion", len(baked) == len(endings) == 1
                    and baked[0].get("count") == 4 and baked[0].get("pos") == OVEN
                    and baked[0].get("outcome") == endings[0].get("outcome") == "succeeded", str(baked + endings))
        return {"hashes": hashes, "inventory": inventory, "status": status, "events": events}

    def _bake_failures(self) -> None:
        """Cancellation and completion-time overload leave ingredients intact and always unlight the oven."""
        self._clear()
        self._item(BOT, 0, WHEAT, 12)
        started = self._start(ToolCall("bake", {"count": 4}))
        self._check("cancellable batch starts and lights the oven", started.get("accepted") is True and self._lit())
        self._step(20)
        self.client.request("skill_cancel", {"name": BOT})
        self._check("cancelling baking unlights the smoker and spends no wheat", not self._lit()
                    and self.client.request("skill_status", {"name": BOT}).get("state") == "cancelled"
                    and counts(self._inventory()) == {WHEAT: 12})
        started = self._start(ToolCall("bake", {"count": 4}))
        self._check("late-failure batch starts before inventory changes",
                    started.get("accepted") is True and self._lit())
        self._step(20)
        self._item(BOT, 1, BREAD, 30, self._now() // 24_000)
        self._step(140)
        status = self.client.request("skill_status", {"name": BOT})
        details = status.get("progress", {})
        self._check("late overload fails atomically with details in status.progress",
                    status.get("state") == "failed" and status.get("reason") == "too_heavy"
                    and details.get("item") == BREAD and details.get("limit") == 32
                    and details.get("have") == 30 and details.get("remaining") == 2 and bool(details.get("hint"))
                    and counts(self._inventory()) == {WHEAT: 12, BREAD: 30} and not self._lit(), str(status))
        self._clear()
        rejected = self._act(ToolCall("bake", {"count": 1}))
        self._check("baking without wheat refuses without lighting the oven",
                    rejected.get("reason") == "not_enough_wheat" and not self._lit(), str(rejected))
        self._item(BOT, 0, WHEAT, 3)
        self._command(f"setblock {' '.join(map(str, OVEN))} minecraft:air")
        rejected = self._act(ToolCall("bake", {"count": 1}))
        self._check("count-only bake refuses when no visible reachable smoker exists",
                    rejected.get("reason") == "no_oven" and counts(self._inventory()) == {WHEAT: 3}, str(rejected))
        self._command(f"setblock {' '.join(map(str, OVEN))} minecraft:smoker")

    def _ownership(self) -> None:
        """An owner stocks the stall and withdraws earnings; a nonowner can only buy."""
        self._clear()
        self._item(BOT, 0, BREAD, 4, self._now() // 24_000)
        self._empty(STALL)
        stocked = self._act(self._at("chest_put", STALL, "bread", 4))
        self._check("owner stocks its own shop", stocked.get("state") == "succeeded", str(stocked))
        self._clear("security")
        self._item("security", 0, EMERALD, 12)
        self._item("security", 1, BREAD, 1, self._now() // 24_000)
        for tool in ("chest_take", "chest_put"):
            rejected = self._act(self._at(tool, STALL, "bread", 1), "security")
            self._check(f"nonowner cannot {tool} at an owned shop", rejected.get("accepted") is False
                        and rejected.get("reason") == "shop_use_buy", str(rejected))
        bought = self._act(self._at("buy", STALL, "bread", 2), "security")
        self._check("nonowner can buy at the posted per-unit price", bought.get("state") == "succeeded"
                    and counts(self._inventory("security")) == {BREAD: 3, EMERALD: 6}, str(bought))
        taken = self._act(self._at("chest_take", STALL, "emerald", 6))
        self._check("owner takes stall takings", taken.get("state") == "succeeded"
                    and counts(self._inventory()).get(EMERALD) == 6, str(taken))

    def _villagers(self) -> None:
        """Correlate every sale with a completed adult market trip, including same-tick contention and failures."""
        saved = self._saved()["npcs"]
        departure = saved["schedule"]["adultEvening"]
        day = self._now() // 24_000 + (self._now() % 24_000 >= departure - 1)
        start = day * 24_000 + departure - 1
        self._step(start - self._now())
        self._empty(STALL)
        for slot, amount, baked in ((0, 64, day - 3), (1, 4, day - 1), (2, 2, day)):
            self._item(" ".join(map(str, STALL)), slot, BREAD, amount, baked)
        cursor = max((event["id"] for event in self._events()), default=0)
        self.client.request("config_set", {"villager_purchases": [
            {"stall": STALL, "item": BREAD, "count": 2}, {"stall": EMPTY_STALL, "item": BREAD, "count": 2}]})
        self._check("configuring purchases alone does not purchase", not self._events(cursor))
        for _ in range(250):
            self._step(20)
            if counts(self._items(STALL)).get(BREAD, 0) < 70:
                break
        initial = self._events(cursor)
        sold = sum(event.get("count", 0) for event in initial if event["type"] == "purchase")
        slots = {item["slot"]: item for item in self._items(STALL) if item["item"] == BREAD}
        self._check("first arrivals consume newest bread before older fresh bread, never mouldy stock", sold > 0
                    and slots.get(0, {}).get("count") == 64
                    and slots.get(2, {}).get("count", 0) == max(0, 2 - sold)
                    and slots.get(1, {}).get("count", 0) == max(0, 4 - max(0, sold - 2)), str(slots))
        self._step(max(0, day * 24_000 + 23_900 - self._now()))
        events = [event for event in self._events(cursor) if event["type"] in {"purchase", "sold_out"}]
        npcs = self._saved()["npcs"]
        arrivals = market_arrivals(npcs, start, self._now())
        self._check("actual adult market arrivals occurred", bool(arrivals), str(arrivals))
        for pos in (STALL, EMPTY_STALL):
            self._check(f"stall {pos}: one result per actual arrival in tick/id order",
                        bool(arrivals) and sale_arrivals(events, pos) == arrivals, str(sale_arrivals(events, pos)))
        purchases = [event for event in events if event["type"] == "purchase"]
        self._check("three exact two-loaf purchases pay six emeralds apiece", len(purchases) == 3
                    and all(event.get("item") == BREAD and event.get("count") == 2 and event.get("cost") == 6
                            for event in purchases), str(purchases))
        self._check("mouldy stock remains while eighteen emeralds reach the stall",
                    counts(self._items(STALL)) == {BREAD: 64, EMERALD: 18}, str(self._items(STALL)))
        sold_out = [event for event in events if event["type"] == "sold_out"]
        self._check("sold_out has only the exact contract fields plus the standard event header", bool(sold_out)
                    and all(set(event) == {"id", "gameTime", "type", "npc", "stall", "item"}
                            and event["item"] == BREAD for event in sold_out), str(sold_out[:3]))
        self._check("depleted fresh stock and a completely empty stall both log sold_out",
                    any(event["stall"] == STALL for event in sold_out)
                    and any(event["stall"] == EMPTY_STALL for event in sold_out)
                    and not self._items(EMPTY_STALL))

    def _resume(self, checkpoint: Path, label: str) -> VillageSession:
        """Prepare a resumed branch with no fresh-session configuration writes."""
        self.active_directory = self.args.runs_dir / f"it23_goods_{label}_resumed"
        session = VillageSession(self.active_directory, checkpoint, self.manifest, self.options, fresh=False)
        session.clone()
        return session

    def run(self) -> bool:
        """Only an explicit CLI execution starts servers; all are owned by context managers and run sequentially."""
        session = VillageSession(self.directory, self.args.runs_dir / "village_sabotage", self.manifest, self.options)
        session.clone()
        evidence = {}
        try:
            with session:
                self.client = session.client
                baseline = self._saved()
                off_checkpoint = self._checkpoint("off")
                before = self.client.request("hash", {"region": self.region})
                self.client.request("config_set", OFF)
                self._check("explicitly disabled goods flags preserve the full legacy hash",
                            self.client.request("hash", {"region": self.region}) == before)
                self._check("disabled flags preserve canonical config and saved NPC bytes", self._saved() == baseline)
                off_hashes = self._hashes()
                self.client.request("config_set", ON)
                self._stage()
                self._limits_and_freshness()
                self._bake_setup()
                bake_checkpoint = self._checkpoint("bake")
                baked = self._bake_tail()
                self._bake_failures()
                self._ownership()
                self._villagers()
            with self._resume(off_checkpoint, "off") as resumed:
                self.client = resumed.client
                untouched = self._hashes()
                self._check("flags-off branch is byte-identical to untouched legacy replay", untouched == off_hashes)
                evidence["flags_off"] = {"configured": off_hashes, "untouched": untouched}
            with self._resume(bake_checkpoint, "bake") as resumed:
                self.client = resumed.client
                self._check("checkpoint restores the running bake at tick forty", self._now() == self.bake_start + 40
                            and self.client.request("skill_status", {"name": BOT}).get("state") == "running"
                            and self._lit())
                replayed = self._bake_tail()
                self._check("in-flight bake replay: every hash, inventory, status and event is identical",
                            replayed == baked)
                evidence["baking"] = {"factual": baked, "replayed": replayed}
        except (harness.HarnessError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self._check("gate completed", False, str(error))
        passed = bool(self.checks) and all(check["passed"] for check in self.checks)
        report = {"passed": passed, "checks": self.checks, "evidence": evidence}
        (self.directory / "it23_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} IT-23: goods", flush=True)
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse shared gate options and return zero only for a passing, explicitly invoked server probe."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if IT23GoodsExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-23: goods probe could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
