"""Server-free checks of IT-23 evidence selection and the count-only baking protocol."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orchestrator.llm import ToolCall
from scripts.test_it23_goods import (
    BREAD, OVEN, STALL, WHEAT, IT23GoodsExperiment, counts, market_arrivals, sale_arrivals,
)


class GoodsProbeHelpersTest(unittest.TestCase):
    """Prevent false positives from intended visits, sorted-away ordering bugs or a restarted bake."""

    def test_counts_sums_separate_freshness_stacks_without_mutation(self):
        """Different baking days still contribute to the same exact-count carrying limit."""
        items = [{"item": BREAD, "count": 2, "baked_day": -3},
                 {"item": BREAD, "count": 30, "baked_day": 0}, {"item": WHEAT, "count": 12}]
        self.assertEqual(counts(items), {BREAD: 32, WHEAT: 12})
        self.assertEqual(items[0]["baked_day"], -3)

    def test_only_completed_adult_market_trips_are_actual_arrivals(self):
        """Work at market, children, departures and unreachable trips must never justify a purchase."""
        def trip(npc="adult", arrival=100, activity="market", to="market", **extra):
            return {"npc": npc, "arriveTick": arrival, "activity": activity, "to": to, **extra}

        saved = {"npcs": {"adult": {"role": "adult"}, "child": {"role": "child"}},
                 "days": {"0": {"trips": [trip(), trip(arrival=99), trip(arrival=201), trip(arrival=None),
                                           trip(activity="work"), trip(npc="child"), trip(to="home"),
                                           trip(arrival=200, outcome="missed")]}}}
        self.assertEqual(market_arrivals(saved, 100, 200), [(100, "adult"), (200, "adult")])

    def test_expected_arrivals_sort_by_tick_and_then_raw_npc_id(self):
        """Historical trip-list append order is not the required market queue ordering."""
        saved = {"npcs": {name: {"role": "adult"} for name in ("alice", "bob", "zoe")},
                 "days": {"0": {"trips": [
                     {"npc": name, "arriveTick": tick, "activity": "market", "to": "market"}
                     for tick, name in ((101, "alice"), (100, "zoe"), (100, "bob"))]}}}
        self.assertEqual(market_arrivals(saved, 0, 200), [(100, "bob"), (100, "zoe"), (101, "alice")])

    def test_sale_arrivals_preserves_actual_event_order_and_filters_other_stalls(self):
        """A same-tick queue reversal remains observable, rather than sorted into a passing comparison."""
        shop = "container:" + ",".join(map(str, STALL))
        events = [{"type": "purchase", "gameTime": 100, "buyer": "zoe", "shop": shop},
                  {"type": "purchase", "gameTime": 100, "buyer": "elsewhere", "shop": "container:1,2,3"},
                  {"type": "sold_out", "gameTime": 100, "npc": "bob", "stall": STALL},
                  {"type": "sold_out", "gameTime": 100, "npc": "other", "stall": [1, 2, 3]},
                  {"type": "npc_trip", "gameTime": 100, "npc": "trip"}]
        self.assertEqual(sale_arrivals(events, STALL), [(100, "zoe"), (100, "bob")])
        self.assertNotEqual(sale_arrivals(events, STALL), [(100, "bob"), (100, "zoe")])

    def test_initialization_and_count_only_dry_run_never_start_servers(self):
        """Importing/constructing the probe is safe; only explicitly entering run owns a server."""
        args = SimpleNamespace(runs_dir=Path("/unused"), port=25623, game_port=None, heap=None, base_seed=0,
                               startup_timeout=1, request_timeout=1)
        with patch("scripts.test_it23_goods.harness.Server") as server:
            experiment = IT23GoodsExperiment(args)
            experiment.client = Mock()
            experiment._start(ToolCall("bake", {"count": 4}), dry=True)
        server.assert_not_called()
        experiment.client.request.assert_called_once_with(
            "skill_start", {"name": "builder", "skill": "bake", "args": {"count": 4}, "dry_run": True})

    def test_item_staging_uses_real_custom_data_and_does_not_translate_an_oven_argument(self):
        """Negative baked days survive SNBT staging exactly, and bake remains count-only."""
        experiment = IT23GoodsExperiment.__new__(IT23GoodsExperiment)
        experiment._command = Mock()
        experiment._item("builder", 0, BREAD, 2, -3)
        experiment._command.assert_called_once_with(
            "item replace entity builder hotbar.0 with "
            "minecraft:bread[minecraft:custom_data={simharness_baked_day:-3}] 2")
        experiment._command.reset_mock()
        experiment._item(" ".join(map(str, OVEN)), 2, WHEAT, 12)
        experiment._command.assert_called_once_with(
            "item replace block 100 -60 101 container.2 with minecraft:wheat 12")


if __name__ == "__main__":
    unittest.main()
