"""Server-free welfare and economy metrics from small, real-format run logs."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.village_metrics import VillageMetrics


EMERALD = "minecraft:emerald"
BREAD = "minecraft:bread"
PLANKS = "minecraft:oak_planks"
ROLES = ("mayor", "builder", "freight", "envoy", "investigator", "security")


def event(tick, event_type, **fields):
    """Build an engine event without invoking the server or a mock."""
    return {"id": tick, "gameTime": tick, "type": event_type, **fields}


def observation(tick, name, emeralds):
    """Expose an agent's actual inventory, including an explicitly empty one."""
    inventory = [{"slot": 0, "item": EMERALD, "count": emeralds}] if emeralds else []
    return {"gameTime": tick, "observer": name, "items": [
        {"category": "self", "name": name, "inventory": inventory}]}


class VillageMetricsTests(unittest.TestCase):
    """Exercise only welfare and economy through persisted synthetic inputs."""

    def setUp(self):
        """Create a two-day run with six roles, two stalls and a treasury."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)
        self._json("scenario.json", {
            "roles": [{"name": name, "kit": {EMERALD: 64}} for name in ROLES],
            "containers": {
                "treasury": {"pos": [0, 64, 0], "stock": {EMERALD: 128}, "allowed": ["mayor"]},
                "stall_1": {"pos": [10, 64, 20], "stock": {BREAD: 64}, "prices": {BREAD: 2},
                            "allowed": None},
                "stall_2": {"pos": [20, 64, 20], "sign_pos": [21, 65, 24], "stock": {PLANKS: 64},
                            "prices": {PLANKS: 3}, "allowed": None}}})
        self._json("m5_report.json", {"end": 48000, "npc_metrics": []})
        for filename in ("world/simharness/events.jsonl", "world/simharness/observations.jsonl",
                         "orchestrator/decisions.jsonl"):
            self._jsonl(filename, [])

    def _json(self, filename, value):
        """Write a JSON fixture under this test's isolated run directory."""
        (self.run_dir / filename).write_text(json.dumps(value), encoding="utf-8")

    def _jsonl(self, filename, rows):
        """Write actual newline-delimited log records, creating their parent directory."""
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _compute(self):
        """Compute through the public API and verify the returned metrics are persisted."""
        metrics = VillageMetrics(self.run_dir, extractor=None).compute()
        self.assertIsInstance(metrics, dict)
        self.assertEqual(json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8")), metrics)
        return metrics

    def test_purchases_count_transactions_and_group_days_and_totals(self):
        """Purchase count means transactions; quantity belongs in each item's items total."""
        self._jsonl("world/simharness/events.jsonl", [
            event(23999, "purchase", buyer="builder", shop="stall_1", item=BREAD, count=3, cost=6),
            event(24000, "purchase", buyer="builder", shop="stall_2", item=PLANKS, count=4, cost=12),
            event(47999, "purchase", buyer="mayor", shop="stall_1", item=BREAD, count=2, cost=4)])
        economy = self._compute()["economy"]
        self.assertEqual(set(economy["days"]), {"0", "1"})
        self.assertEqual(economy["days"]["0"]["purchases"], {
            "count": 1, "emeralds_spent": 6,
            "by_buyer": {"builder": {"count": 1, "emeralds_spent": 6}},
            "by_item": {BREAD: {"count": 1, "emeralds_spent": 6, "items": 3}}})
        self.assertEqual(economy["days"]["1"]["purchases"]["count"], 2)
        self.assertEqual(economy["days"]["1"]["purchases"]["emeralds_spent"], 16)
        self.assertEqual(economy["totals"]["purchases"], {
            "count": 3, "emeralds_spent": 22,
            "by_buyer": {"builder": {"count": 2, "emeralds_spent": 18},
                         "mayor": {"count": 1, "emeralds_spent": 4}},
            "by_item": {BREAD: {"count": 2, "emeralds_spent": 10, "items": 5},
                        PLANKS: {"count": 1, "emeralds_spent": 12, "items": 4}}})

    def test_transfers_group_emerald_pairs_and_other_item_quantities(self):
        """Directed emerald pairs and other items retain transaction counts and quantities."""
        self._jsonl("world/simharness/events.jsonl", [
            event(1, "transfer", **{"from": "mayor", "to": "builder", "item": EMERALD, "count": 4}),
            event(2, "transfer", **{"from": "mayor", "to": "builder", "item": EMERALD, "count": 6}),
            event(3, "transfer", **{"from": "freight", "to": "builder", "item": PLANKS, "count": 8}),
            event(24000, "transfer", **{"from": "builder", "to": "mayor", "item": EMERALD, "count": 2}),
            event(24001, "transfer", **{"from": "freight", "to": "mayor", "item": PLANKS, "count": 3})])
        economy = self._compute()["economy"]
        self.assertEqual(economy["days"]["0"]["transfers"], {
            "count": 3, "emeralds": {"count": 2, "total": 10, "by_pair": [
                {"from": "mayor", "to": "builder", "count": 2, "total": 10}]},
            "by_item": {PLANKS: {"count": 1, "total": 8}}})
        transfers = economy["totals"]["transfers"]
        self.assertEqual(transfers["count"], 5)
        self.assertEqual(transfers["emeralds"]["count"], 3)
        self.assertEqual(transfers["emeralds"]["total"], 12)
        self.assertCountEqual(transfers["emeralds"]["by_pair"], [
            {"from": "mayor", "to": "builder", "count": 2, "total": 10},
            {"from": "builder", "to": "mayor", "count": 1, "total": 2}])
        self.assertEqual(transfers["by_item"], {PLANKS: {"count": 2, "total": 11}})
        self.assertEqual(economy["days"]["1"]["transfers"]["emeralds"]["total"], 2)

    def test_holdings_follow_the_latest_prompt_inventory_and_later_flows(self):
        """A role's holding is its latest prompt's emeralds (its kit before any prompt) plus later transfers and
        purchases; a prompt at or after the boundary does not count, and every role is known, so the Gini is too."""
        def decision(tick, agent, inventory):
            """A logged decision whose prompt shows the agent's inventory."""
            return {"agent": agent, "game_time": tick, "tool": "wait", "args": {}, "outcome": "started",
                    "user_prompt": f"You are {agent}.\nInventory:\n{inventory}Last activity: none.\n"}
        self._jsonl("orchestrator/decisions.jsonl", [
            decision(100, "mayor", "  slot 0: 7 emerald\n  slot 1: 5 emerald\n  slot 2: 99 bread\n"),
            decision(150, "freight", "  slot 0: 3 oak_planks\n"),
            decision(24000, "mayor", "  slot 0: 999 emerald\n")])
        self._jsonl("world/simharness/events.jsonl", [
            event(50, "transfer", **{"from": "envoy", "to": "mayor", "item": EMERALD, "count": 9}),
            event(200, "transfer", **{"from": "mayor", "to": "builder", "item": EMERALD, "count": 4}),
            event(300, "purchase", buyer="mayor", shop="stall_1", item=BREAD, count=1, cost=2)])
        day = self._compute()["economy"]["days"]["0"]
        self.assertEqual(day["holdings"], {"mayor": 6, "builder": 68, "freight": 0, "envoy": 55,
                                           "investigator": 64, "security": 64})
        self.assertAlmostEqual(day["gini"], VillageMetrics.gini([6, 68, 0, 55, 64, 64]))

    def test_treasury_tracks_only_its_emerald_transfers(self):
        """Treasury stock carries forward independently of role snapshots, goods and purchases."""
        treasury = "container:0,64,0"
        self._jsonl("world/simharness/events.jsonl", [
            event(1, "transfer", **{"from": treasury, "to": "builder", "item": EMERALD, "count": 24}),
            event(2, "transfer", **{"from": "mayor", "to": treasury, "item": EMERALD, "count": 8}),
            event(3, "transfer", **{"from": treasury, "to": "builder", "item": PLANKS, "count": 99}),
            event(4, "purchase", buyer="mayor", shop="stall_1", item=BREAD, count=1, cost=2),
            event(5, "transfer", **{"from": "mayor", "to": "envoy", "item": EMERALD, "count": 50}),
            event(24000, "transfer", **{"from": treasury, "to": "mayor", "item": EMERALD, "count": 12})])
        days = self._compute()["economy"]["days"]
        self.assertEqual(days["0"]["treasury"], 112)
        self.assertEqual(days["1"]["treasury"], 100)

    def test_partial_last_day_is_included_and_carries_snapshots_forward(self):
        """A run ending inside day one includes its transactions and the last known holdings."""
        self._json("m5_report.json", {"end": 24010, "npc_metrics": []})
        self._jsonl("world/simharness/events.jsonl", [
            event(24009, "purchase", buyer="mayor", shop="stall_1", item=BREAD, count=1, cost=2)])
        self._jsonl("world/simharness/observations.jsonl", [
            observation(23999, "mayor", 6), observation(24011, "mayor", 99)])
        days = self._compute()["economy"]["days"]
        self.assertEqual(set(days), {"0", "1"})
        self.assertEqual(days["0"]["purchases"]["count"], 0)
        self.assertEqual(days["1"]["purchases"]["count"], 1)
        self.assertEqual(days["1"]["holdings"]["mayor"], 62)  # its kit, less the purchase
        self.assertEqual(days["1"]["treasury"], 128)

    def test_gini_matches_known_distributions(self):
        """Equality, zero wealth and concentrated holdings have their standard Gini values."""
        for holdings, expected in (([4] * 6, 0), ([0] * 6, 0), ([0, 10], 0.5),
                                   ([1, 2, 3], 2 / 9), ([12, 0, 0, 0, 0, 0], 5 / 6)):
            with self.subTest(holdings=holdings):
                self.assertAlmostEqual(VillageMetrics.gini(holdings), expected)

    def test_prices_require_successful_matching_stall_sign_writes(self):
        """Only confirmed writes at configured signs change prices; malformed edits are reported."""
        default_sign, explicit_sign = [10, 64, 22], [21, 65, 24]
        cases = [
            (24010, default_sign, "bread", "4 emeralds", "done", "mayor", default_sign),
            (24015, default_sign, "bread", "6 emeralds", "done", "mayor", default_sign),
            (24020, default_sign, "bread", "ask me", "done", "mayor", default_sign),
            (24025, explicit_sign, "oak_planks", "7 emeralds", "done", "mayor", explicit_sign),
            (24030, default_sign, "bread", "99 emeralds", "failed: blocked", "mayor", default_sign),
            (24040, [10, 64, 23], "bread", "99 emeralds", "done", "mayor", [10, 64, 23]),
            (24050, [20, 64, 22], "oak_planks", "88 emeralds", "done", "mayor", [20, 64, 22]),
            (24070, default_sign, "bread", "66 emeralds", "done", "builder", default_sign),
            (24080, default_sign, "bread", "55 emeralds", "done", "mayor", [99, 64, 22]),
            (24090, default_sign, "bread", "77 emeralds", "done", None, default_sign)]
        decisions, writes = [], []
        for tick, position, item, price, outcome, author, document_pos in cases:
            lines = ["Stall 1" if item == "bread" else "Stall 2", item, price, "per item"]
            decisions.append({"game_time": tick, "agent": "mayor", "tool": "write_sign", "outcome": outcome,
                              "args": dict(zip(("x", "y", "z"), position), lines=lines)})
            if author is not None:
                writes.append(event(tick, "doc_write", author=author, kind="sign", title=lines[0],
                                    docId="sign:" + ",".join(map(str, document_pos))))
                writes[-1]["id"] = 900000 + tick
        self._jsonl("orchestrator/decisions.jsonl", list(reversed(decisions)))
        self._jsonl("world/simharness/events.jsonl", list(reversed(writes)))
        days = self._compute()["economy"]["days"]
        self.assertEqual(days["0"]["posted_prices"], {"stall_1": {BREAD: 2}, "stall_2": {PLANKS: 3}})
        self.assertEqual(days["1"]["posted_prices"], {"stall_1": {BREAD: 6}, "stall_2": {PLANKS: 7}})
        self.assertEqual(days["0"]["unparseable_sign_edits"], [])
        self.assertEqual(len(days["1"]["unparseable_sign_edits"]), 1)

    def test_welfare_uses_scheduled_denominators_and_counts_late_school_arrivals(self):
        """Commutes summarize NPCs, but each activity's shares exclude unscheduled villagers."""
        npcs = []
        for index, commute in enumerate((10, 30, 50, 70, 90, 110)):
            npcs.append({"npc": f"npc_{index}", "commuteTicks": commute,
                         "workOnTime": (True, False, None, None, None, None)[index],
                         "workStatus": ("on_time", "late", *(["not_scheduled"] * 4))[index],
                         "marketVisited": (True, False, None, None, None, None)[index],
                         "marketStatus": ("visited", "missed", *(["not_scheduled"] * 4))[index],
                         "schoolStatus": ("not_scheduled", "not_scheduled", "on_time", "late", "absent",
                                          "not_scheduled")[index]})
        idle = {**npcs[-1], "commuteTicks": 0}
        self._json("m5_report.json", {"end": 48000, "npc_metrics": [
            {"day": 0, "npcs": npcs}, {"day": 1, "npcs": [idle]}]})
        welfare = self._compute()["welfare"]
        self.assertEqual(welfare["0"], {
            "mean_commute_ticks": 60, "median_commute_ticks": 60, "work_on_time_share": 0.5,
            "market_visited_share": 0.5, "school_reached_share": 2 / 3})
        self.assertEqual(welfare["1"], {
            "mean_commute_ticks": 0, "median_commute_ticks": 0, "work_on_time_share": None,
            "market_visited_share": None, "school_reached_share": None})


if __name__ == "__main__":
    unittest.main()
