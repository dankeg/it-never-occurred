"""Unit tests for absolute-game-time inputs and below-front targeting."""

import copy
import unittest

from scripts import scripted_routine as routine


class RoutineTests(unittest.TestCase):
    """Exercise persistent inputs, cycle boundaries, and cardinal targets."""
    def test_every_phase_and_turn(self):
        """Verify every phase and turn."""
        for game_time, yaw, forward, jump, sprint in (
            (0, 0, 1, False, False), (199, 0, 1, False, False),
            (200, 90, 1, True, False), (299, 90, 1, True, False),
            (300, 90, 1, False, True), (349, 90, 1, False, True),
            (350, 90, 0, False, False), (399, 90, 0, False, False),
            (400, 90, 1, False, False), (600, 180, 1, True, False),
            (1400, 0, 1, True, False), (24000, 0, 1, False, False),
        ):
            with self.subTest(game_time=game_time):
                commands = routine.actions_for(game_time)
                self.assertEqual(commands[0]["cmd"], "bot_input")
                inputs = commands[0]["args"]
                self.assertEqual((inputs["yaw"], inputs["forward"], inputs["jump"], inputs["sprint"]),
                                 (yaw, forward, jump, sprint))
                self.assertFalse(inputs["sneak"])
                self.assertEqual(inputs["strafe"], 0)

    def test_break_place_once_per_cycle_and_never_respawn_or_refill(self):
        """Verify break place once per cycle and never respawn or refill."""
        for game_time in range(1600):
            commands = routine.actions_for(game_time)
            names = [command["cmd"] for command in commands]
            expected = ["bot_input", "bot_break", "bot_select", "bot_place"]
            self.assertEqual(names, expected if game_time % 400 == 300 else ["bot_input"])
            if len(commands) > 1:
                self.assertEqual(commands[1]["target"], routine.BELOW_FRONT)
                self.assertEqual(commands[3]["target"], routine.BELOW_FRONT)
        self.assertEqual([command["cmd"] for command in routine.initial_commands()],
                         ["spawn_bot", "give", "bot_select"])
        self.assertEqual(routine.initial_commands()[1]["args"]["count"], 64)

    def test_pure_at_arbitrary_absolute_restore_times(self):
        """Verify pure at arbitrary absolute restore times."""
        expected = {game_time: copy.deepcopy(routine.actions_for(game_time))
                    for game_time in (100, 299, 300, 350, 12100, 12300, 24100)}
        for game_time in reversed(expected):
            self.assertEqual(routine.actions_for(game_time), expected[game_time])
        commands = routine.actions_for(300)
        commands[0]["args"]["yaw"] = -999
        commands[1]["args"]["x"] = 999
        self.assertEqual(routine.actions_for(300), expected[300])

    def test_batch_boundaries_preserve_inputs_and_instantaneous_actions(self):
        """Verify batch boundaries preserve inputs and instantaneous actions."""
        for game_time in (0, 100, 199, 200, 299, 300, 349, 350, 399, 12100):
            after = routine.next_action_time(game_time)
            self.assertGreater(after, game_time)
            inputs = routine.actions_for(game_time)[0]
            for interior in range(game_time + 1, after):
                self.assertEqual(routine.actions_for(interior), [inputs])

    def test_invalid_times(self):
        """Verify invalid times."""
        for value in (-1, 0.5, True, "100"):
            for function in (routine.actions_for, routine.next_action_time):
                with self.subTest(value=value, function=function), self.assertRaises(ValueError):
                    function(value)

    def test_below_front_handles_cardinals_and_negative_flooring(self):
        """Verify below front handles cardinals and negative flooring."""
        for yaw, expected in (
            (0, {"x": -1, "y": -61, "z": 1}),
            (90, {"x": -2, "y": -61, "z": 0}),
            (180, {"x": -1, "y": -61, "z": -1}),
            (270, {"x": 0, "y": -61, "z": 0}),
            (-90, {"x": 0, "y": -61, "z": 0}),
        ):
            with self.subTest(yaw=yaw):
                self.assertEqual(routine.below_front({"pos": [-0.1, -59.2, 0.5], "yRot": yaw}), expected)
        for position in ([0, 1], [float("nan"), 0, 0], [0, True, 0]):
            with self.assertRaises(ValueError):
                routine.below_front({"pos": position, "yRot": 0})

    def test_below_front_targets_ground_while_jump_walking(self):
        """Verify below front targets ground while jump walking."""
        target = routine.below_front({"pos": [0.5, -58.75, 0.5], "yRot": 90})
        self.assertEqual(target, {"x": -1, "y": -61, "z": 0})


if __name__ == "__main__":
    unittest.main()
