"""Behavior tests for grounded candidate generation, Match, and engine dry-run translation."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from orchestrator.agent import AgentMemory
from orchestrator.candidates import Candidate, CandidateGenerator, DryRun, EngineDryRun, known_contents
from orchestrator.candidates import representable, same_choice
from orchestrator.llm import ToolCall
from orchestrator.scenario import ScenarioManifest
from orchestrator.tools import TOOL_DEFINITIONS, ToolTranslator

ROOT = Path(__file__).resolve().parents[2]


def observation(position=(-56, -60, 62), inventory=(), entities=(), blocks=(), documents=(), structures=(), skill=None):
    """Return a small raw observation at a village position, using the supplied visible facts and inventory."""
    return {"self": {"pos": list(position), "inventory": list(inventory),
                     "skill": skill or {"state": "idle"}}, "entities": list(entities), "blocks": list(blocks),
            "documents": list(documents), "structures": list(structures)}


def player(name, position, distance):
    """Return an observed player with the supplied name, actual position, and engine-measured distance."""
    return {"type": "minecraft:player", "name": name, "pos": list(position), "distance": distance}


def villager(identity, position, distance):
    """Return a named observed villager with a prefixed NPC id and the supplied position and distance."""
    return {"type": "minecraft:villager", "name": "Ada", "npcId": f"npc:{identity}",
            "pos": list(position), "distance": distance}


def calls_for(candidates, tool):
    """Return the offered calls for one tool, preserving their final candidate order."""
    return [candidate.call for candidate in candidates if candidate.call.name == tool]


class RecordingFeasibility:
    """Record skill checks and return fixed-duration verdicts with explicit call, tool, or count rejections."""

    def __init__(self, ticks=15):
        """Start with the given duration, an empty call log, and no rejected tools, calls, or count limits."""
        self.ticks = ticks
        self.calls = []
        self.rejected_tools = set()
        self.rejected_calls = []
        self.limits = {}

    def __call__(self, agent, call):
        """Log an actor's call and return whether it passes the configured explicit rejections and count limit."""
        self.calls.append((agent, call))
        feasible = (call.name not in self.rejected_tools and call not in self.rejected_calls
                    and call.args.get("count", 1) <= self.limits.get(call.name, 64))
        return DryRun(feasible, self.ticks, None if feasible else "not feasible")


class CandidateGeneratorTest(unittest.TestCase):
    """Real village facts determine visible, accessible, feasible, and bounded choices without engine access."""

    def setUp(self):
        """Load the real manifest and create a fresh recording skill checker and candidate generator for each test."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.checker = RecordingFeasibility()
        self.generator = CandidateGenerator(self.manifest, self.checker)

    def test_move_to_includes_road_points_and_private_containers(self):
        """The investigator can walk to public road points and the private records chest, with exact labels."""
        candidates = self.generator.generate("investigator", observation())
        moves = calls_for(candidates, "move_to")
        self.assertIn(ToolCall("move_to", {"x": -60, "y": -60, "z": 74}), moves)
        records = next(candidate for candidate in candidates if candidate.call ==
                       ToolCall("move_to", {"x": -56, "y": -60, "z": 62}))
        self.assertEqual(records.label, 'move_to {"x":-56,"y":-60,"z":62} | '
                         'walk to the records chest, 0 blocks, about 15 ticks')
        self.assertEqual((records.target, records.distance), ((-56, -60, 62), 0))
        market = self.generator.generate("investigator", observation((-46, -60, -15)))
        self.assertIn(ToolCall("move_to", {"x": -46, "y": -60, "z": -15}), calls_for(market, "move_to"))

    def test_move_to_does_not_reveal_another_roles_unseen_container(self):
        """Being beside an unseen private chest does not reveal it, even to an actor with a master key."""
        for agent, position in (("investigator", (109, -60, -45)), ("security", (-64, -60, 62))):
            with self.subTest(agent=agent):
                candidates = self.generator.generate(agent, observation(position))
                self.assertNotIn(ToolCall("move_to", dict(zip(("x", "y", "z"), position))),
                                 calls_for(candidates, "move_to"))

    def test_move_to_includes_floored_standing_points_and_ferry_landings(self):
        """Schematic standing points floor negative coordinates, and ferry boarding points are map destinations."""
        candidates = self.generator.generate("builder", observation((-6.5, -60, 0.5)))
        self.assertIn(ToolCall("move_to", {"x": -7, "y": -60, "z": 0}), calls_for(candidates, "move_to"))
        standing = next(candidate for candidate in candidates if "bridge_span_1 standing point" in candidate.label)
        self.assertIn("walk to the bridge_span_1 standing point", standing.label)
        candidates = self.generator.generate("freight", observation((-5, -60, 41)))
        self.assertIn(ToolCall("move_to", {"x": -5, "y": -60, "z": 41}), calls_for(candidates, "move_to"))
        market = self.generator.generate("freight", observation((-46, -60, -13)))
        self.assertIn(ToolCall("move_to", {"x": -46, "y": -60, "z": -13}), calls_for(market, "move_to"))

    def test_move_to_includes_visible_people_containers_and_structures(self):
        """Visible non-role players, villagers, inaccessible chests, and structures supply integer destinations."""
        view = observation(entities=[player("visitor", (-54.2, -60, 62.9), 2),
                                     villager("adult_07", (-58.1, -60, 64.5), 3),
                                     player("investigator", (-56, -60, 62), 0)],
                           blocks=[{"pos": [-64, -60, 62], "block": "minecraft:chest", "distance": 8}],
                           structures=[{"name": "port_repair", "pos": [-55, -60, 63], "built": 1, "total": 9}])
        candidates = self.generator.generate("investigator", view)
        moves = calls_for(candidates, "move_to")
        for target in ((-55, -60, 62), (-59, -60, 64), (-64, -60, 62), (-55, -60, 63)):
            self.assertIn(ToolCall("move_to", dict(zip(("x", "y", "z"), target))), moves)
        self.assertTrue(any("Ada (villager adult_07)" in candidate.label for candidate in candidates))
        self.assertFalse(any(call.args.get("to") == "visitor" for call in calls_for(candidates, "speak")))

    def test_move_to_keeps_the_first_name_for_duplicate_positions(self):
        """A public road point wins over people and structures occupying the same block."""
        view = observation((-60, -60, 74), entities=[player("mayor", (-59.8, -60, 74.9), 1)],
                           structures=[{"name": "duplicate", "pos": [-60, -60, 74]}])
        candidates = self.generator.generate("investigator", view)
        matches = [candidate for candidate in candidates if candidate.call ==
                   ToolCall("move_to", {"x": -60, "y": -60, "z": 74})]
        self.assertEqual(len(matches), 1)
        self.assertIn("walk to the town hall", matches[0].label)

    def test_move_to_drops_infeasible_targets_and_stops_after_24_feasible_checks(self):
        """Movement checks run nearest first, skip a rejected nearby destination, and refill to 24 choices."""
        rejected = ToolCall("move_to", {"x": -56, "y": -60, "z": 62})
        self.checker.rejected_calls.append(rejected)
        candidates = self.generator.generate("investigator", observation())
        moves = calls_for(candidates, "move_to")
        checked = [call for _, call in self.checker.calls if call.name == "move_to"]
        self.assertEqual(len(moves), 24)
        self.assertEqual(len(checked), 25)
        self.assertNotIn(rejected, moves)
        distances = [math.dist((-56, -60, 62), [call.args[axis] for axis in ("x", "y", "z")]) for call in checked]
        self.assertEqual(distances, sorted(distances))

    def test_move_to_breaks_equal_distance_check_ties_by_name(self):
        """Equidistant destinations are dry-run by their public names rather than coordinate order."""
        view = observation(structures=[{"name": "Zulu", "pos": [-57, -60, 62]},
                                       {"name": "Alpha", "pos": [-55, -60, 62]}])
        self.generator.generate("investigator", view)
        checked = [call.args["x"] for _, call in self.checker.calls if call.name == "move_to"]
        self.assertLess(checked.index(-55), checked.index(-57))

    def test_shout_is_offered_beyond_talk_range(self):
        """Addressed speech includes distance boundaries, both broadcasts, and text only in dry-run placeholders."""
        view = observation(entities=[player("mayor", (-48, -60, 62), 8),
                                     player("builder", (-47.9, -60, 62), 8.1),
                                     player("freight", (-32, -60, 62), 24),
                                     player("envoy", (-31.9, -60, 62), 24.1),
                                     player("visitor", (-55, -60, 62), 1),
                                     player("investigator", (-56, -60, 62), 0)])
        candidates = self.generator.generate("investigator", view)
        speech = [candidate for candidate in candidates if candidate.call.name == "speak"]
        self.assertEqual([candidate.call.args for candidate in speech],
                         [{"to": "builder", "volume": "shout"}, {"to": "freight", "volume": "shout"},
                          {"to": "mayor", "volume": "talk"}, {"volume": "shout"}, {"volume": "talk"},
                          {"volume": "whisper"}])
        self.assertEqual(speech[0].label, 'speak {"to":"builder","volume":"shout"} | '
                         'speak to builder (shout), 8 blocks away; heard within 24 blocks')
        for candidate in speech:
            self.assertNotIn("text", candidate.call.args)
            self.assertNotIn("Hello.", candidate.label)
            self.assertNotIn("ticks", candidate.label)
            self.assertEqual((candidate.target, candidate.distance), ((-56, -60, 62), 0))
        checked = [call for _, call in self.checker.calls if call.name == "speak"]
        self.assertEqual(len(checked), 6)
        self.assertTrue(all(call.args["text"] == "Hello." for call in checked))

    def test_infeasible_speech_is_dropped(self):
        """A rejected shout dry run removes that broadcast without removing feasible talk."""
        self.checker.rejected_calls.append(ToolCall("speak", {"volume": "shout", "text": "Hello."}))
        calls = calls_for(self.generator.generate("investigator", observation()), "speak")
        self.assertEqual(calls, [ToolCall("speak", {"volume": "talk"}), ToolCall("speak", {"volume": "whisper"})])

    def test_ask_only_offers_villagers_within_talk_range(self):
        """News requests remove the NPC prefix, preserve actual target coordinates, and include the duration."""
        view = observation(entities=[villager("adult_07", (-48, -60, 62.5), 8),
                                     villager("adult_08", (-47.9, -60, 62), 8.1)])
        candidates = self.generator.generate("investigator", view)
        self.assertEqual(calls_for(candidates, "ask"), [ToolCall("ask", {"npc": "adult_07"})])
        choice = next(candidate for candidate in candidates if candidate.call.name == "ask")
        self.assertEqual(choice.label, 'ask {"npc":"adult_07"} | '
                         'ask Ada (villager adult_07) for news, 8 blocks away, about 15 ticks')
        self.assertEqual(choice.target, (-48, -60, 62.5))
        self.assertEqual(choice.distance, math.dist(view["self"]["pos"], choice.target))

    def test_container_access_requires_observed_reach_and_permission(self):
        """Container reach uses observed eye distance, access lists are enforced, and the master key overrides them."""
        block = {"pos": [-56, -60, 62], "block": "minecraft:chest", "distance": 4.5}
        view = observation((-60.5, -60, 62), blocks=[block],
                           inventory=[{"slot": 0, "item": "minecraft:bread", "count": 2}])
        for agent, expected in (("investigator", 1), ("freight", 0), ("security", 1)):
            with self.subTest(agent=agent):
                candidates = self.generator.generate(agent, view)
                self.assertEqual(len(calls_for(candidates, "open_container")), expected)
                self.assertEqual(len(calls_for(candidates, "chest_put")), expected)
        block["distance"] = 4.5001
        self.assertFalse(calls_for(self.generator.generate("security", view), "open_container"))
        self.assertFalse(calls_for(self.generator.generate("investigator", observation()), "open_container"))

    def test_container_takes_use_latest_reads_not_manifest_stock(self):
        """Only remembered stacks can be taken, with latest counts and exact public container labels."""
        memory = AgentMemory()
        memory.record_read(1, "container at (-56, -60, 62): 1 emerald")
        memory.record_read(2, "container at (-56, -60, 62): 1 written_book, 64 emerald")
        view = observation(blocks=[{"pos": [-56, -60, 62], "block": "minecraft:chest", "distance": 2}],
                           inventory=[{"slot": 0, "item": "minecraft:bread", "count": 2}])
        candidates = self.generator.generate("investigator", view, memory)
        takes = calls_for(candidates, "chest_take")
        self.assertEqual({(call.args["item"], call.args["count"]) for call in takes},
                         {("written_book", 1), ("emerald", 64)})
        labels = {candidate.call.name: candidate.label for candidate in candidates
                  if candidate.call.name in {"open_container", "chest_put"}}
        self.assertEqual(labels["open_container"], 'open_container {"x":-56,"y":-60,"z":62} | '
                         'look inside the records chest')
        self.assertEqual(labels["chest_put"], 'chest_put {"count":2,"item":"bread","x":-56,"y":-60,"z":62} | '
                         'put 2 bread into the records chest, about 15 ticks')
        take = next(candidate for candidate in candidates if candidate.call.name == "chest_take"
                    and candidate.call.args["item"] == "written_book")
        self.assertEqual(take.label, 'chest_take {"count":1,"item":"written_book","x":-56,"y":-60,"z":62} | '
                         'take 1 written_book from the records chest, about 15 ticks')
        self.assertEqual(take.target, (-56, -60, 62))
        treasury = observation((-64, -60, 62), blocks=[{"pos": [-64, -60, 62],
                                                      "block": "minecraft:chest", "distance": 2}])
        self.assertFalse(calls_for(self.generator.generate("mayor", treasury), "chest_take"))

    def test_buy_bisects_maximum_count_without_requiring_container_access(self):
        """A priced chest can be bought from without opening permission, using one and the bisected maximum."""
        self.manifest.data["containers"]["stall_1"]["allowed"] = ["mayor"]
        self.checker.limits["buy"] = 17
        view = observation((-62, -60, -12), blocks=[{"pos": [-62, -60, -13],
                                                  "block": "minecraft:chest", "distance": 2}])
        candidates = self.generator.generate("freight", view)
        self.assertFalse(calls_for(candidates, "open_container"))
        self.assertEqual(sorted(call.args["count"] for call in calls_for(candidates, "buy")), [1, 17])
        choice = next(candidate for candidate in candidates if candidate.call.name == "buy"
                      and candidate.call.args["count"] == 17)
        self.assertEqual(choice.label, 'buy {"count":17,"item":"oak_planks","x":-62,"y":-60,"z":-13} | '
                         'buy 17 oak_planks from the stall 1 chest for 34 emeralds, about 15 ticks')
        self.assertEqual(choice.target, (-62, -60, -13))
        checks = [call.args["count"] for _, call in self.checker.calls if call.name == "buy"]
        self.assertLessEqual(len(checks), 7)
        self.assertEqual(checks.count(1), 1)
        self.assertEqual(checks.count(17), 1)

    def test_craft_bisects_only_schematic_bill_items(self):
        """Crafting enumerates bill items, not unrelated held items, with the feasible endpoint counts and label."""
        self.checker.limits["craft"] = 9
        candidates = self.generator.generate("builder", observation(inventory=[
            {"slot": 0, "item": "minecraft:bread", "count": 4}]))
        calls = calls_for(candidates, "craft")
        self.assertEqual(sorted((call.args["item"], call.args["count"]) for call in calls),
                         [("oak_planks", 1), ("oak_planks", 9)])
        craft = ToolCall("craft", {"item": "oak_planks", "count": 9})
        choice = next(candidate for candidate in candidates if candidate.call == craft)
        self.assertEqual(choice.label, 'craft {"count":9,"item":"oak_planks"} | craft 9 oak_planks, about 15 ticks')
        self.assertEqual((choice.target, choice.distance), ((-56, -60, 62), 0))

    def test_bisection_handles_zero_one_and_64_feasible_items(self):
        """Both quantity tools drop infeasible count one, avoid duplicate singletons, and reach MAX_COUNT."""
        view = observation((-62, -60, -12), blocks=[{"pos": [-62, -60, -13],
                                                  "block": "minecraft:chest", "distance": 2}])
        for maximum, expected in ((0, []), (1, [1]), (64, [1, 64])):
            with self.subTest(maximum=maximum):
                self.checker.limits = {"buy": maximum, "craft": maximum}
                candidates = self.generator.generate("freight", view)
                for tool in ("buy", "craft"):
                    self.assertEqual(sorted(call.args["count"] for call in calls_for(candidates, tool)), expected)

    def test_give_only_hands_whole_stacks_to_nearby_other_roles(self):
        """Handovers use entity reach and actual positions, excluding self, non-role players, and distant roles."""
        view = observation(inventory=[{"slot": 0, "item": "minecraft:bread", "count": 3}],
                           entities=[player("mayor", (-51.5, -60, 62.2), 4.5),
                                     player("builder", (-51.4, -60, 62), 4.6),
                                     player("visitor", (-55, -60, 62), 1),
                                     player("investigator", (-56, -60, 62), 0)])
        candidates = self.generator.generate("investigator", view)
        self.assertEqual(calls_for(candidates, "give"),
                         [ToolCall("give", {"target": "mayor", "item": "bread", "count": 3})])
        choice = next(candidate for candidate in candidates if candidate.call.name == "give")
        self.assertEqual(choice.label, 'give {"count":3,"item":"bread","target":"mayor"} | '
                         'hand 3 bread to mayor, about 15 ticks')
        self.assertEqual(choice.target, (-51.5, -60, 62.2))

    def test_drop_offers_carried_stacks_at_the_actors_feet(self):
        """Dropping needs no nearby recipient and deduplicates repeated stacks while ignoring empty slots."""
        view = observation(inventory=[{"slot": 0, "item": "minecraft:bread", "count": 3},
                                      {"slot": 1, "item": "minecraft:bread", "count": 3},
                                      {"slot": 2, "item": "minecraft:oak_planks", "count": 0}])
        candidates = self.generator.generate("investigator", view)
        self.assertEqual(calls_for(candidates, "drop"), [ToolCall("drop", {"item": "bread", "count": 3})])
        choice = next(candidate for candidate in candidates if candidate.call.name == "drop")
        self.assertEqual(choice.target, tuple(view["self"]["pos"]))
        self.assertEqual(choice.distance, 0)
        self.assertEqual(choice.label, 'drop {"count":3,"item":"bread"} | '
                         'drop 3 bread on the ground, about 15 ticks')
        self.assertIn(("investigator", choice.call), self.checker.calls)
        self.assertEqual(candidates, self.generator.generate("investigator", view))
        self.checker.rejected_tools.add("drop")
        self.assertEqual(calls_for(self.generator.generate("investigator", view), "drop"), [])
        self.assertEqual(calls_for(self.generator.generate("investigator", observation()), "drop"), [])

    def test_build_section_requires_its_full_bill_summed_across_slots(self):
        """A section needs all six planks across slots and targets its origin rather than its standing point."""
        inventory = [{"slot": 0, "item": "minecraft:oak_planks", "count": 3},
                     {"slot": 1, "item": "minecraft:oak_planks", "count": 3}]
        view = observation((-6.5, -60, 0.5), inventory=inventory)
        candidates = self.generator.generate("builder", view)
        choice = next(candidate for candidate in candidates if candidate.call.name == "build_section"
                      and candidate.call.args["schematic"] == "bridge_span_1")
        self.assertEqual(choice.call.args, {"schematic": "bridge_span_1", "origin": [-6, -61, 0]})
        self.assertEqual(choice.label, 'build_section {"origin":[-6,-61,0],"schematic":"bridge_span_1"} | '
                         'build bridge_span_1 (6 oak_planks), about 15 ticks')
        self.assertEqual(choice.target, (-6, -61, 0))
        inventory[1]["count"] = 2
        self.assertFalse(calls_for(self.generator.generate("builder", view), "build_section"))

    def test_build_section_uses_standing_point_reach_and_dry_run(self):
        """Standing-point reach includes 4.5 but not more, and an engine rejection still removes the section."""
        inventory = [{"slot": 0, "item": "minecraft:oak_planks", "count": 6}]
        near = observation((-11, -60, 0.5), inventory=inventory)
        far = observation((-11.001, -60, 0.5), inventory=inventory)
        self.assertEqual(len(calls_for(self.generator.generate("builder", near), "build_section")), 1)
        self.assertFalse(calls_for(self.generator.generate("builder", far), "build_section"))
        self.checker.rejected_tools.add("build_section")
        self.assertFalse(calls_for(self.generator.generate("builder", near), "build_section"))

    def test_mine_skips_fluids_and_containers_and_keeps_nearest_eight(self):
        """Mining uses observed reach, chooses eight blocks by distance then position, and labels the short block id."""
        blocks = [{"pos": [-58 + index % 4, -60, 61 + index // 4], "block": "minecraft:stone",
                   "distance": 1 + index // 2 * 0.2} for index in range(12)]
        excluded = [{"pos": [-56, -60, 62], "block": "minecraft:stone", "distance": 0.5},
                    {"pos": [-55, -61, 62], "block": "minecraft:water", "distance": 0.5},
                    {"pos": [-54, -61, 62], "block": "minecraft:lava", "distance": 0.5},
                    {"pos": [-52, -60, 62], "block": "minecraft:stone", "distance": 4.5001}]
        blocks = [block for block in blocks if block["pos"] != [-56, -60, 62]]
        candidates = self.generator.generate("investigator", observation(blocks=list(reversed(blocks + excluded))))
        mines = [candidate for candidate in candidates if candidate.call.name == "mine"]
        expected = sorted(blocks, key=lambda block: (block["distance"], block["pos"]))[:8]
        self.assertEqual({candidate.target for candidate in mines}, {tuple(block["pos"]) for block in expected})
        checked = [call for _, call in self.checker.calls if call.name == "mine"]
        self.assertEqual([call.args for call in checked],
                         [dict(zip(("x", "y", "z"), block["pos"])) for block in expected])
        choice = next(candidate for candidate in mines if candidate.target == (-58, -60, 61))
        self.assertEqual(choice.label, 'mine {"x":-58,"y":-60,"z":61} | '
                         'break the stone at (-58, -60, 61), about 15 ticks')

    def test_place_requires_held_items_and_skips_observed_correct_blocks(self):
        """Placement selects eight nearest missing schematic blocks, including wrong observed blocks but not built
        ones."""
        view = observation((-3.5, -60, 0.5), inventory=[{"slot": 0, "item": "minecraft:oak_planks", "count": 1}],
                           blocks=[{"pos": [-4, -61, 0], "block": "minecraft:oak_planks", "distance": 2.12},
                                   {"pos": [-5, -61, 0], "block": "minecraft:stone", "distance": 2.35}])
        candidates = self.generator.generate("builder", view)
        places = [candidate for candidate in candidates if candidate.call.name == "place"]
        self.assertEqual(len(places), 8)
        self.assertNotIn((-4, -61, 0), [candidate.target for candidate in places])
        self.assertIn((-5, -61, 0), [candidate.target for candidate in places])
        checked = [call for _, call in self.checker.calls if call.name == "place"]
        eye = (-3.5, -58.38, 0.5)
        distances = [math.dist(eye, [call.args[axis] + 0.5 for axis in ("x", "y", "z")]) for call in checked]
        self.assertEqual(distances, sorted(distances))
        self.assertTrue(all(distance <= 4.5 for distance in distances))
        self.assertFalse(calls_for(self.generator.generate("builder", observation((-3.5, -60, 0.5))), "place"))

    def test_fair_lantern_place_reach_is_eye_to_block_centre(self):
        """The fair lantern is offered at exactly eye reach, even when its block position is farther from the feet."""
        inventory = [{"slot": 0, "item": "minecraft:lantern", "count": 1}]
        near = observation((62, -61.12, 20.5), inventory=inventory)
        candidates = self.generator.generate("freight", near)
        choices = [candidate for candidate in candidates if candidate.call.name == "place"]
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].label, 'place {"item":"lantern","x":57,"y":-60,"z":20} | '
                         'place lantern at (57, -60, 20), about 15 ticks')
        self.assertEqual(choices[0].target, (57, -60, 20))
        self.assertGreater(choices[0].distance, 4.5)
        far = observation((62.0001, -61.12, 20.5), inventory=inventory)
        self.assertFalse(calls_for(self.generator.generate("freight", far), "place"))

    def test_collect_is_offered_only_for_visible_dropped_items_and_is_checked(self):
        """A visible dropped item offers one collection skill at the actor's feet, subject to a dry run."""
        self.assertFalse(calls_for(self.generator.generate("investigator", observation()), "collect"))
        view = observation(entities=[{"type": "minecraft:item", "pos": [-40, -60, 62], "distance": 16}])
        candidates = self.generator.generate("investigator", view)
        choice = next(candidate for candidate in candidates if candidate.call.name == "collect")
        self.assertEqual(choice.label, 'collect {} | pick up dropped items nearby, about 15 ticks')
        self.assertEqual((choice.target, choice.distance), ((-56, -60, 62), 0))
        self.checker.rejected_tools.add("collect")
        self.assertFalse(calls_for(self.generator.generate("investigator", view), "collect"))

    def test_book_and_sign_calls_use_slots_and_positions_without_free_text(self):
        """Read and write candidates preserve document metadata, inventory slots, and observed sign reach."""
        view = observation(inventory=[{"slot": 2, "item": "minecraft:written_book", "count": 1,
                                       "doc": {"title": "Report", "author": "mayor"}},
                                      {"slot": 7, "item": "minecraft:writable_book", "count": 1}],
                           documents=[{"kind": "sign", "pos": [-55, -60, 62], "distance": 4.5},
                                      {"kind": "sign", "pos": [-54, -60, 62], "distance": 4.5001},
                                      {"kind": "lectern", "pos": [-57, -60, 62], "distance": 1}])
        candidates = self.generator.generate("investigator", view)
        expected = {
            "read_book": 'read_book {"slot":2} | read the written book "Report" by mayor in slot 2',
            "write_book": 'write_book {"slot":7} | write and sign a book (writable book in slot 7)',
            "write_sign": 'write_sign {"x":-55,"y":-60,"z":62} | write on the sign at (-55, -60, 62)',
        }
        for tool, label in expected.items():
            choices = [candidate for candidate in candidates if candidate.call.name == tool]
            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0].label, label)
            self.assertEqual(choices[0].target, (-55, -60, 62) if tool == "write_sign" else (-56, -60, 62))
            self.assertFalse({"text", "title", "pages", "lines"}.intersection(choices[0].call.args))

    def test_book_without_observed_metadata_uses_renderer_defaults(self):
        """A written book without optional metadata still gets a readable label without invented title or author."""
        view = observation(inventory=[{"slot": 0, "item": "minecraft:written_book", "count": 1}])
        candidates = self.generator.generate("investigator", view)
        choice = next(candidate for candidate in candidates if candidate.call.name == "read_book")
        self.assertEqual(choice.label, 'read_book {"slot":0} | read the written book "" by ? in slot 0')

    def test_ferry_requires_nearby_boarding_and_targets_the_other_landing(self):
        """Ferries are checked from within three blocks and use the destination board as their perception target."""
        candidates = self.generator.generate("freight", observation((-8, -60, 41)))
        self.assertEqual(calls_for(candidates, "ferry"), [ToolCall("ferry", {"to": "east"})])
        choice = next(candidate for candidate in candidates if candidate.call.name == "ferry")
        self.assertEqual(choice.label, 'ferry {"to":"east"} | ride the boat to the east ferry landing, about 15 ticks')
        self.assertEqual((choice.target, choice.distance), ((4, -60, 41), 12))
        self.assertFalse(calls_for(self.generator.generate("freight", observation((-8.001, -60, 41))), "ferry"))
        reverse = self.generator.generate("freight", observation((4, -60, 41)))
        self.assertEqual(calls_for(reverse, "ferry"), [ToolCall("ferry", {"to": "west"})])

    def test_waits_are_always_offered_and_keep_going_requires_a_running_skill(self):
        """The three waits never need feasibility, while continuation appears only for running activity."""
        self.checker.rejected_tools.update(tool["name"] for tool in TOOL_DEFINITIONS)
        for state in ("idle", "running", "failed"):
            with self.subTest(state=state):
                running = observation(skill={"state": state, "skill": "move_to"})
                candidates = self.generator.generate("investigator", running)
                self.assertEqual({call.args["ticks"] for call in calls_for(candidates, "wait")}, {100, 600, 1200})
                continuing = [candidate for candidate in candidates if candidate.call.name == "keep_going"]
                self.assertEqual(len(continuing), int(state == "running"))
                if continuing:
                    self.assertEqual(continuing[0].label, 'keep_going {} | carry on with the current move_to')
        waits = [candidate.label for candidate in candidates if candidate.call.name == "wait"]
        self.assertIn('wait {"ticks":100} | wait 100 ticks', waits)
        self.assertTrue(all(call.name not in {"wait", "keep_going"} for _, call in self.checker.calls))

    def test_only_skill_candidates_are_sent_to_feasibility(self):
        """Information actions and text-free book and sign writing bypass feasibility, while every checked tool is a
        skill."""
        view = observation(blocks=[{"pos": [-56, -60, 62], "block": "minecraft:chest", "distance": 2}],
                           inventory=[{"slot": 0, "item": "minecraft:written_book", "count": 1},
                                      {"slot": 1, "item": "minecraft:writable_book", "count": 1}],
                           documents=[{"kind": "sign", "pos": [-55, -60, 62], "distance": 1}],
                           skill={"state": "running", "skill": "move_to"})
        self.generator.generate("investigator", view)
        skills = {"move_to", "mine", "collect", "craft", "place", "build_section", "chest_put", "chest_take",
                  "buy", "give", "use", "drop", "hold", "eat", "ferry", "speak", "ask"}
        self.assertTrue(all(agent == "investigator" and call.name in skills for agent, call in self.checker.calls))

    def test_missing_dry_run_duration_omits_duration_suffixes(self):
        """Feasible calls without tick estimates remain candidates, with no fabricated duration in their labels."""
        self.checker.ticks = None
        candidates = self.generator.generate("investigator", observation())
        records = next(candidate for candidate in candidates if "walk to the records chest" in candidate.label)
        self.assertEqual(records.label, 'move_to {"x":-56,"y":-60,"z":62} | walk to the records chest, 0 blocks')
        self.assertTrue(all(", about" not in candidate.label for candidate in candidates))

    def test_crowded_cap_preserves_speech_questions_waits_and_continuation(self):
        """More than 64 raw choices retain all protected tools and prefer nearby remaining acts over distant moves."""
        entities = [villager(f"adult_{index:02}", (-55, -60, 62), 1) for index in range(30)]
        entities += [player(role["name"], (-54, -60, 62), 2) for role in self.manifest.roles
                     if role["name"] != "investigator"]
        documents = [{"kind": "sign", "pos": [-58 + index % 5, -60, 60 + index // 5], "distance": 3}
                     for index in range(20)]
        blocks = [{"pos": [-58 + index % 4, -60, 61 + index // 4], "block": "minecraft:stone", "distance": 2}
                  for index in range(8)]
        view = observation(entities=entities, documents=documents, blocks=blocks,
                           skill={"state": "running", "skill": "mine"})
        candidates = self.generator.generate("investigator", view)
        self.assertEqual(len(candidates), 64)
        for tool, count in (("speak", 13), ("ask", 30), ("wait", 3), ("keep_going", 1)):
            self.assertEqual(len(calls_for(candidates, tool)), count)
        self.assertNotIn(ToolCall("move_to", {"x": -60, "y": -60, "z": 74}), calls_for(candidates, "move_to"))
        unprotected = [candidate for candidate in candidates if candidate.call.name not in
                       {"speak", "ask", "wait", "keep_going"}]
        self.assertTrue(all(candidate.distance <= 4.5 for candidate in unprotected))

    def test_results_are_deduplicated_deterministic_tool_sorted_and_do_not_mutate_inputs(self):
        """Repeated observed facts yield one call, stable tool-distance-label order, and no observation or memory
        edits."""
        sign = {"kind": "sign", "pos": [-55, -60, 62], "distance": 1}
        view = observation(documents=[sign, deepcopy(sign)],
                           entities=[villager("adult_07", (-55.5, -60, 62.5), 1)] * 2,
                           blocks=[{"pos": [-56, -60, 62], "block": "minecraft:chest", "distance": 2}],
                           inventory=[{"slot": 0, "item": "minecraft:bread", "count": 2},
                                      {"slot": 1, "item": "minecraft:bread", "count": 2}])
        memory = AgentMemory()
        memory.record_read(1, "container at (-56, -60, 62): 1 emerald")
        before = deepcopy((view, memory.to_json(), self.manifest.data))
        candidates = self.generator.generate("investigator", view, memory)
        self.assertEqual(candidates, self.generator.generate("investigator", view, memory))
        self.assertEqual((view, memory.to_json(), self.manifest.data), before)
        keys = [(candidate.call.name, json.dumps(candidate.call.args, sort_keys=True)) for candidate in candidates]
        self.assertEqual(len(keys), len(set(keys)))
        order = {tool["name"]: index for index, tool in enumerate(TOOL_DEFINITIONS)}
        sort_keys = [(order[candidate.call.name], candidate.distance, candidate.label) for candidate in candidates]
        self.assertEqual(sort_keys, sorted(sort_keys))
        self.assertEqual(len(calls_for(candidates, "write_sign")), 1)
        self.assertEqual(len(calls_for(candidates, "ask")), 1)
        self.assertEqual(len(calls_for(candidates, "chest_put")), 1)


class MatchTest(unittest.TestCase):
    """Match compares only each tool's specified choice-defining fields and safely handles missing arguments."""

    def test_move_tolerance_is_three_rounded_blocks_per_axis(self):
        """Every movement axis tolerates three rounded blocks, but four on any axis fails."""
        origin = ToolCall("move_to", {"x": -56.2, "y": -60.1, "z": 62.2})
        self.assertTrue(same_choice(origin, ToolCall("move_to", {"x": -53, "y": -57, "z": 65})))
        for axis in ("x", "y", "z"):
            with self.subTest(axis=axis):
                args = {"x": -56, "y": -60, "z": 62}
                args[axis] += 4
                self.assertFalse(same_choice(origin, ToolCall("move_to", args)))

    def test_speech_normalizes_addressees_defaults_to_talk_and_ignores_text(self):
        """Speech ignores words and notes but distinguishes volume and normalized addressee."""
        first = ToolCall("speak", {"to": " NPC:Mayor ", "text": "one", "note": "first"})
        self.assertTrue(same_choice(first, ToolCall("speak", {"to": "mayor", "volume": "talk", "text": "two"})))
        self.assertFalse(same_choice(first, ToolCall("speak", {"to": "mayor", "volume": "shout"})))
        self.assertFalse(same_choice(first, ToolCall("speak", {"to": "builder"})))
        self.assertTrue(same_choice(ToolCall("speak", {}), ToolCall("speak", {"to": "  ", "volume": "talk"})))
        self.assertTrue(same_choice(ToolCall("ask", {"npc": " NPC:Adult_07 "}),
                                    ToolCall("ask", {"npc": "adult_07"})))
        self.assertFalse(same_choice(ToolCall("ask", {"npc": "adult_07"}), ToolCall("ask", {"npc": "adult_08"})))

    def test_container_trade_and_placement_match_position_and_normalized_item(self):
        """Position and normalized item determine transfers and placement, independently of counts and notes."""
        for tool in ("chest_take", "chest_put", "buy", "place"):
            with self.subTest(tool=tool):
                first = ToolCall(tool, {"x": -56, "y": -60, "z": 62, "item": "Oak Planks", "count": 64})
                args = {"x": -56, "y": -60, "z": 62, "item": "minecraft:oak_planks", "count": 1, "note": "new"}
                self.assertTrue(same_choice(first, ToolCall(tool, args)))
                self.assertFalse(same_choice(first, ToolCall(tool, {**args, "x": -55})))
                self.assertFalse(same_choice(first, ToolCall(tool, {**args, "item": "stone"})))

    def test_mining_opening_and_sign_writing_match_only_position(self):
        """These positional tools ignore notes and free text, but not a changed coordinate."""
        for tool in ("mine", "open_container", "write_sign"):
            with self.subTest(tool=tool):
                first = ToolCall(tool, {"x": -56, "y": -60, "z": 62, "lines": ["one"]})
                args = {"x": -56, "y": -60, "z": 62, "lines": ["two"], "note": "new"}
                self.assertTrue(same_choice(first, ToolCall(tool, args)))
                self.assertFalse(same_choice(first, ToolCall(tool, {**args, "z": 63})))

    def test_giving_and_crafting_match_normalized_targets_and_items(self):
        """Giving matches a normalized person and item, while crafting ignores count and uses the item alone."""
        quoted = ToolCall("give", {"target": " NPC:Mayor ", "item": "written_book \"Report\" by mayor", "count": 1})
        self.assertTrue(same_choice(quoted, ToolCall("give", {"target": "mayor", "item": "minecraft:written_book",
                                                              "count": 2})))
        self.assertFalse(same_choice(ToolCall("give", {"target": "mayor", "item": "bread"}),
                                     ToolCall("give", {"target": "builder", "item": "bread"})))
        self.assertTrue(same_choice(ToolCall("craft", {"item": "Oak Planks", "count": 1}),
                                    ToolCall("craft", {"item": "minecraft:oak_planks", "count": 64})))
        self.assertFalse(same_choice(ToolCall("craft", {"item": "bread"}), ToolCall("craft", {"item": "stone"})))

    def test_drop_matches_the_normalized_item_without_quantity_or_note(self):
        """The delegate can vary drop quantities but cannot silently switch the item being discarded."""
        choice = ToolCall("drop", {"item": "Oak Planks", "count": 1, "note": "discard"})
        self.assertTrue(same_choice(choice, ToolCall("drop", {"item": "minecraft:oak_planks", "count": 64})))
        self.assertFalse(same_choice(choice, ToolCall("drop", {"item": "bread", "count": 1})))
        self.assertFalse(same_choice(choice, ToolCall("craft", {"item": "oak_planks"})))

    def test_section_ferry_and_books_match_their_named_choice_fields(self):
        """Building matches section, ferry matches destination, and book acts match slot while ignoring prose."""
        self.assertTrue(same_choice(ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [-6, -61, 0]}),
                                    ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [0, 0, 0]})))
        self.assertFalse(same_choice(ToolCall("build_section", {"schematic": "bridge_span_1"}),
                                     ToolCall("build_section", {"schematic": "bridge_span_2"})))
        self.assertTrue(same_choice(ToolCall("ferry", {"to": "east"}), ToolCall("ferry", {"to": "east", "note": "go"})))
        self.assertFalse(same_choice(ToolCall("ferry", {"to": "east"}), ToolCall("ferry", {"to": "west"})))
        for tool in ("read_book", "write_book"):
            with self.subTest(tool=tool):
                self.assertTrue(same_choice(ToolCall(tool, {"slot": 2}),
                                            ToolCall(tool, {"slot": 2, "title": "Report", "pages": ["text"]})))
                self.assertFalse(same_choice(ToolCall(tool, {"slot": 2}), ToolCall(tool, {"slot": 3})))

    def test_collect_wait_and_continuation_match_the_tool_alone(self):
        """Tool-alone choices ignore waiting durations, notes, and other arguments."""
        for tool in ("collect", "wait", "keep_going"):
            with self.subTest(tool=tool):
                self.assertTrue(same_choice(ToolCall(tool, {}), ToolCall(tool, {"ticks": 1200, "note": "later"})))

    def test_none_different_tools_and_missing_arguments_do_not_raise(self):
        """None and differing tools never match; two missing choice fields match, but a missing field is not a value."""
        self.assertFalse(same_choice(None, None))
        self.assertFalse(same_choice(None, ToolCall("wait", {})))
        self.assertFalse(same_choice(ToolCall("wait", {}), None))
        self.assertFalse(same_choice(ToolCall("wait", {}), ToolCall("collect", {})))
        for tool in ("move_to", "speak", "ask", "mine", "open_container", "write_sign", "chest_take", "chest_put",
                     "buy", "place", "give", "craft", "build_section", "ferry", "read_book", "write_book"):
            with self.subTest(tool=tool):
                self.assertTrue(same_choice(ToolCall(tool, {}), ToolCall(tool, {})))
        for tool, args in (("move_to", {"x": 0}), ("craft", {"item": "bread"}), ("read_book", {"slot": 0})):
            self.assertFalse(same_choice(ToolCall(tool, {}), ToolCall(tool, args)))

    def test_representable_requires_some_matching_candidate(self):
        """Representability uses Match rather than exact call equality, and rejects absent or unmatched choices."""
        candidates = [Candidate(ToolCall("craft", {"item": "oak_planks", "count": 1}), "craft", 0, (-56, -60, 62))]
        self.assertTrue(representable(candidates, ToolCall("craft", {"item": "minecraft:oak_planks", "count": 64})))
        self.assertFalse(representable(candidates, ToolCall("craft", {"item": "bread"})))
        self.assertFalse(representable(candidates, None))
        self.assertFalse(representable([], ToolCall("wait", {})))


class KnownContentsTest(unittest.TestCase):
    """Container memory parsing uses only retained reads, never hidden inventory or book text."""

    def test_latest_container_read_wins_and_preserves_stack_order(self):
        """Later reads replace an entire container, while listed item stacks retain their runtime order."""
        memory = AgentMemory()
        memory.record_read(1, "container at (-56, -60, 62): 10 bread")
        memory.record_read(2, "container at (-64, -60, 62): 64 emerald")
        memory.record_read(3, "container at (-56, -60, 62): 1 written_book, 64 emerald, 2 written_book")
        self.assertEqual(known_contents(memory),
                         {(-56, -60, 62): [("written_book", 1), ("emerald", 64), ("written_book", 2)],
                          (-64, -60, 62): [("emerald", 64)]})

    def test_empty_clears_previous_contents_and_book_reads_are_ignored(self):
        """An empty read removes earlier remembered stacks, and book prose resembling a container is not parsed."""
        memory = AgentMemory()
        memory.record_read(1, "container at (-56, -60, 62): 1 written_book")
        memory.record_read(2, 'book "Report" by mayor: container at (-64, -60, 62): 64 emerald')
        memory.record_read(3, "container at (-56, -60, 62): empty")
        self.assertEqual(known_contents(memory), {(-56, -60, 62): []})
        self.assertEqual(known_contents(None), {})
        self.assertEqual(known_contents(AgentMemory()), {})


class CommandError(Exception):
    """A local engine-client command failure used to exercise dry-run error handling without requests."""


class RecordingClient:
    """An in-process request recorder returning one supplied response or raising one supplied command error."""

    def __init__(self, response=None, error=None):
        """Store the supplied response or default feasible estimate, optional error, and an empty request log."""
        self.response = response if response is not None else {"feasible": True, "estimate": {"ticks": 40}}
        self.error = error
        self.requests = []

    def request(self, command, args):
        """Record a command and arguments, then return the configured response or raise the configured error."""
        self.requests.append((command, args))
        if self.error is not None:
            raise self.error
        return self.response


class EngineDryRunTest(unittest.TestCase):
    """Dry runs translate real tools and inspect in-process responses without starting a server or connecting."""

    def test_drop_is_dry_run_as_a_timed_inventory_skill(self):
        """Dropping is checked by the engine with explicit normalized item/count and no scratchpad note."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError)
        self.assertEqual(checker("investigator", ToolCall("drop", {"item": "Oak Planks", "count": 3,
                                                                   "note": "discard"})), DryRun(True, 40))
        self.assertEqual(client.requests, [("skill_start", {"name": "investigator", "skill": "drop",
                                 "args": {"item": "minecraft:oak_planks", "count": 3}, "dry_run": True})])

    def test_skill_request_puts_dry_run_at_the_top_level(self):
        """Movement translation retains the centred path target and adds dry_run beside the skill arguments."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError, ToolTranslator())
        self.assertEqual(checker("investigator", ToolCall("move_to", {"x": -56, "y": -60, "z": 62})), DryRun(True, 40))
        expected = {"name": "investigator", "skill": "move_to", "args": {"x": -55.5, "y": -60, "z": 62.5},
                    "dry_run": True}
        self.assertEqual(client.requests, [("skill_start", expected)])

    def test_container_skill_and_section_requests_preserve_translator_semantics(self):
        """Transfer items are normalized and section tools become place_section without moving dry_run inside args."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError)
        checker("investigator", ToolCall("chest_take", {"x": -56, "y": -60, "z": 62, "item": "Oak Planks", "count": 6}))
        checker("builder", ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [-6, -61, 0]}))
        self.assertEqual(client.requests, [
            ("skill_start", {"name": "investigator", "skill": "chest_take", "args": {
                "pos": [-56, -60, 62], "item": "minecraft:oak_planks", "count": 6}, "dry_run": True}),
            ("skill_start", {"name": "builder", "skill": "place_section", "args": {
                "schematic": "bridge_span_1", "origin": [-6, -61, 0]}, "dry_run": True}),
        ])

    def test_speak_and_ask_requests_use_their_own_top_level_commands(self):
        """Speech and questions receive dry_run at the top level without being wrapped in skill_start."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError)
        speech = ToolCall("speak", {"to": "mayor", "volume": "talk", "text": "Hello.", "note": "ignored"})
        checker("investigator", speech)
        checker("investigator", ToolCall("ask", {"npc": "adult_07"}))
        self.assertEqual(client.requests, [
            ("speak", {"name": "investigator", "to": "mayor", "volume": "talk", "text": "Hello.", "dry_run": True}),
            ("ask", {"name": "investigator", "npc": "adult_07", "dry_run": True}),
        ])

    def test_command_errors_and_infeasible_responses_keep_the_reason(self):
        """Engine command failures and explicit infeasibility both produce negative verdicts with their reasons."""
        client = RecordingClient(error=CommandError("no path"))
        self.assertEqual(EngineDryRun(client, CommandError)("builder", ToolCall("craft", {"item": "oak_planks"})),
                         DryRun(False, reason="no path"))
        client = RecordingClient({"feasible": False, "reason": "busy"})
        self.assertEqual(EngineDryRun(client, CommandError)("builder", ToolCall("craft", {"item": "oak_planks"})),
                         DryRun(False, reason="busy"))
        client = RecordingClient({"feasible": True})
        self.assertEqual(EngineDryRun(client, CommandError)("builder", ToolCall("collect", {})), DryRun(True))

    def test_translation_errors_are_infeasible_without_requests(self):
        """Value, key, and type errors from actual translation return invalid reasons without contacting the client."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError)
        for call in (ToolCall("fly", {}), ToolCall("mine", {}), ToolCall("move_to", {"x": None, "y": -60, "z": 62})):
            with self.subTest(call=call):
                verdict = checker("investigator", call)
                self.assertFalse(verdict.feasible)
                self.assertTrue(verdict.reason.startswith("invalid: "))
        self.assertEqual(client.requests, [])

    def test_keep_going_and_information_actions_make_no_requests(self):
        """Continuation and translated instant actions return feasible without engine requests."""
        client = RecordingClient()
        checker = EngineDryRun(client, CommandError)
        for call in (ToolCall("keep_going", {}), ToolCall("open_container", {"x": -56, "y": -60, "z": 62}),
                     ToolCall("read_book", {"slot": 1}),
                     ToolCall("write_book", {"slot": 1, "title": "Report", "pages": []}),
                     ToolCall("write_sign", {"x": -55, "y": -60, "z": 62, "lines": []})):
            with self.subTest(call=call):
                self.assertEqual(checker("investigator", call), DryRun(True))
        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
