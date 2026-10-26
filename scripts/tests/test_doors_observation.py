"""Door hints, visible per-position states and narrated village knowledge without a server."""

import copy
from pathlib import Path
import unittest

from orchestrator.llm import ToolCall
from orchestrator.narration import gm_system_prompt
from orchestrator.render import ObservationRenderer
from orchestrator.runtime import failure_hint
from orchestrator.scenario import ScenarioManifest


GENERIC_HINT = " (no walkable route: it may be across water, enclosed, or not yet built)"
DOOR_HINT = " (a closed door may be in the way: `use` it to open it)"


def notable_block(name: str, position: list[int], distance: float, **properties) -> dict:
    """Build a visible notable block with explicit Minecraft state properties."""
    return {"block": f"minecraft:{name}", "pos": position, "distance": distance, "properties": properties}


class DoorFailureHintTest(unittest.TestCase):
    """Only a failed walk and an explicitly closed door in sight justify the additional hint."""

    def setUp(self):
        """Attempt to walk toward a nearby building."""
        self.call = ToolCall("move_to", {"x": 3, "y": -60, "z": 0})

    def hint(self, observation):
        """Explain a failed walk using only its supplied observation."""
        return failure_hint(self.call, "not possible: no_path", observation)

    def test_closed_door_appends_exact_hint_and_preserves_generic_explanation(self):
        """An explicit false property on either door material supplies the same actionable suffix."""
        for name in ("oak_door", "iron_door"):
            for opened in ("false", False):
                with self.subTest(name=name, opened=opened):
                    block = notable_block(name, [1, -60, 0], 1, open=opened)
                    self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT + DOOR_HINT)

    def test_open_door_leaves_hint_generic(self):
        """An observed open door provides no evidence for the closed-door explanation."""
        for opened in ("true", True):
            with self.subTest(opened=opened):
                block = notable_block("oak_door", [1, -60, 0], 1, open=opened)
                self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT)

    def test_missing_or_unknown_state_does_not_mean_closed(self):
        """Missing properties and unknown values cannot justify the suffix, including a numeric zero."""
        block = notable_block("oak_door", [1, -60, 0], 1)
        self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT)
        del block["properties"]
        self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT)
        for opened in (None, "unknown", 0):
            with self.subTest(opened=opened):
                block = notable_block("oak_door", [1, -60, 0], 1, open=opened)
                self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT)

    def test_other_blocks_with_false_open_property_are_not_doors(self):
        """A barrel's open property must not be mistaken for a closed door."""
        for name in ("barrel", "chest", "lever"):
            with self.subTest(name=name):
                block = notable_block(name, [1, -60, 0], 1, open="false")
                self.assertEqual(self.hint({"blocks": [block]}), GENERIC_HINT)

    def test_unseen_or_remembered_door_does_not_supply_a_hint(self):
        """Door records outside visible notable blocks are not evidence of a current obstruction."""
        block = notable_block("oak_door", [1, -60, 0], 1, open="false")
        self.assertEqual(self.hint({}), GENERIC_HINT)
        for category in ("changes", "terrain", "seen_actions", "memory"):
            with self.subTest(category=category):
                self.assertEqual(self.hint({"blocks": [], category: [block]}), GENERIC_HINT)

    def test_mixed_visible_doors_add_suffix_once_without_mutating_observation(self):
        """An open neighbour does not hide the closed door; repeated explanations are deterministic."""
        observation = {"blocks": [notable_block("oak_door", [1, -60, 0], 1, open="true"),
                                  notable_block("oak_door", [2, -60, 0], 2, open="false"),
                                  notable_block("oak_door", [3, -60, 0], 3, open="false")]}
        original = copy.deepcopy(observation)
        self.assertEqual(self.hint(observation), GENERIC_HINT + DOOR_HINT)
        self.assertEqual(self.hint(observation), GENERIC_HINT + DOOR_HINT)
        self.assertEqual(observation, original)
        self.assertEqual(failure_hint(self.call, "failed: no_path", observation), GENERIC_HINT + DOOR_HINT)

    def test_other_actions_and_outcomes_do_not_get_the_door_hint(self):
        """A closed door is not a reason to alter successful movement or unrelated failures."""
        observation = {"blocks": [notable_block("oak_door", [1, -60, 0], 1, open="false")]}
        for outcome in ("started", "done", "failed: interrupted"):
            with self.subTest(outcome=outcome):
                self.assertEqual(failure_hint(self.call, outcome, observation), "")
        self.assertEqual(failure_hint(ToolCall("mine", {}), "failed: no_path", observation), "")
        self.assertEqual(failure_hint(None, "failed: no_path", observation), "")


class DoorRendererTest(unittest.TestCase):
    """Notable door states stay attached to their positions without changing other blocks or bounds."""

    @staticmethod
    def render(blocks, **limits):
        """Render visible blocks in a minimal complete harness observation."""
        observation = {"gameTime": 0, "self": {"name": "ben", "pos": [0, -60, 0], "health": 20,
                                               "food": 20, "inventory": []}, "blocks": blocks}
        return ObservationRenderer(**limits).render(observation)

    def test_mixed_states_are_reported_per_position_in_nearest_order(self):
        """A single block-type group can contain open, closed and unknown-state doors."""
        blocks = [notable_block("oak_door", [3, -60, 0], 3, open="false"),
                  notable_block("oak_door", [1, -60, 0], 1, open="true"),
                  notable_block("oak_door", [2, -60, 0], 2)]
        self.assertIn("  oak_door: (1, -60, 0) (open), (2, -60, 0), (3, -60, 0) (closed)",
                      self.render(blocks).splitlines())

    def test_trapdoors_and_gates_display_each_explicit_state_consistently(self):
        """Door-like barriers accept string and boolean state properties, not string truthiness."""
        for name in ("oak_door", "oak_trapdoor", "oak_fence_gate"):
            for opened, closed in (("true", "false"), (True, False)):
                with self.subTest(name=name, opened=opened):
                    blocks = [notable_block(name, [1, -60, 0], 1, open=opened),
                              notable_block(name, [2, -60, 0], 2, open=closed)]
                    self.assertIn(f"  {name}: (1, -60, 0) (open), (2, -60, 0) (closed)",
                                  self.render(blocks).splitlines())

    def test_absent_or_unknown_state_is_not_rendered_as_closed(self):
        """Legacy observations and unrecognized values keep the original coordinate-only line."""
        block = notable_block("oak_door", [1, -60, 0], 1)
        del block["properties"]
        for properties in (None, {}, {"open": None}, {"open": "unknown"}, {"open": 0}):
            with self.subTest(properties=properties):
                observed = block if properties is None else {**block, "properties": properties}
                self.assertIn("  oak_door: (1, -60, 0)", self.render([observed]).splitlines())

    def test_non_door_output_is_unchanged_even_with_open_property(self):
        """Other notable blocks keep their original grouping, positions and lack of state labels."""
        blocks = [notable_block("barrel", [1, -60, 0], 1, open="false"),
                  notable_block("barrel", [2, -60, 0], 2, open="true"),
                  notable_block("chest", [3, -60, 0], 3)]
        plain = [{key: value for key, value in block.items() if key != "properties"} for block in blocks]
        self.assertEqual(self.render(blocks), self.render(plain))
        self.assertIn("  barrel: (1, -60, 0), (2, -60, 0)", self.render(blocks).splitlines())

    def test_position_and_block_limits_still_bound_mixed_door_groups(self):
        """Hidden positions are counted, not printed, and observations are not sorted in place."""
        blocks = [notable_block("oak_door", [position, -60, 0], position,
                                open="true" if position % 2 else "false") for position in (4, 3, 2, 1)]
        original = copy.deepcopy(blocks)
        text = self.render(blocks, max_blocks=3, max_per_block=2)
        self.assertIn("  oak_door: (1, -60, 0) (open), (2, -60, 0) (closed), and 1 more nearby",
                      text.splitlines())
        self.assertNotIn("(3, -60, 0)", text)
        self.assertNotIn("(4, -60, 0)", text)
        self.assertEqual(blocks, original)

    def test_equal_distance_doors_use_position_as_deterministic_tiebreaker(self):
        """Changing input order cannot change which state belongs to which position."""
        blocks = [notable_block("oak_door", [2, -60, 0], 1, open="false"),
                  notable_block("oak_door", [1, -60, 0], 1, open="true")]
        self.assertEqual(self.render(blocks), self.render(list(reversed(blocks))))


class DoorKnowledgeTest(unittest.TestCase):
    """Paired village doors are public knowledge and individually located for the narrated GM."""

    def setUp(self):
        """Add paired lower-half door coordinates to an existing public village manifest."""
        path = Path(__file__).resolve().parents[2] / "scenario/village_sabotage.json"
        self.manifest = ScenarioManifest.load(path)
        self.manifest.data["layout"]["doors"] = {
            "school": [[-25, -60, -16], [-24, -60, -16]],
            "east_home_4": [[30, -60, 20], [31, -60, 20]],
        }

    def test_common_knowledge_states_doors_as_facts_without_a_purpose(self):
        """Everyone knows doors start open, that use toggles them, and what a closed door blocks; no reason to close
        one is suggested (briefs and shared knowledge state facts and goals, never tactics)."""
        common = self.manifest.common_knowledge()
        self.assertIn("Buildings have doors (two per doorway), all open at the start", common)
        self.assertIn("`use` opens or closes a door", common)
        self.assertIn("A closed door blocks sight and passage", common)
        self.assertNotIn("privacy", common.lower())

    def test_gm_world_locates_both_lower_halves_for_every_building(self):
        """The GM receives exact paired coordinates, including the house containing the envoy's lodging."""
        world = self.manifest.gm_world()
        self.assertIn("Building doors (oak, all open at the start; lower-half positions):", world)
        self.assertIn("  school: (-25, -60, -16), (-24, -60, -16)", world.splitlines())
        self.assertIn("  east_home_4: (30, -60, 20), (31, -60, 20)", world.splitlines())

    def test_door_layout_rendering_does_not_depend_on_mapping_order(self):
        """Building names sort consistently while each pair retains its recorded position order."""
        world = self.manifest.gm_world()
        doors = self.manifest.data["layout"]["doors"]
        self.manifest.data["layout"]["doors"] = dict(reversed(list(doors.items())))
        self.assertEqual(self.manifest.gm_world(), world)

    def test_legacy_manifests_do_not_invent_door_positions_or_new_knowledge(self):
        """Missing and empty layout records preserve existing door-free scenario descriptions."""
        del self.manifest.data["layout"]["doors"]
        common, world = self.manifest.common_knowledge(), self.manifest.gm_world()
        self.assertNotIn("Buildings have doors", common)
        self.assertNotIn("Building doors", world)
        self.manifest.data["layout"]["doors"] = {}
        self.assertEqual(self.manifest.common_knowledge(), common)
        self.assertEqual(self.manifest.gm_world(), world)

    def test_gm_rules_keep_uses_witnessable_and_state_door_rules(self):
        """Narration retains use timing and witness notices alongside the sight and passage rules."""
        rules = gm_system_prompt([])
        self.assertIn("`use` takes 5 ticks", rules)
        self.assertIn("Nearby witnesses can see the action", rules)
        self.assertIn("These are factual action notices, not spoken words", rules)
        self.assertIn("Buildings have doors (two per doorway), all open at the start", rules)
        self.assertIn("A closed door blocks", rules)


if __name__ == "__main__":
    unittest.main()
