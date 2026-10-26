"""Absolute-time, persistent-input schedule; initialization is deliberately separate."""

from __future__ import annotations

import math


BOT_NAME = "alice"
CYCLE_TICKS = 400
BELOW_FRONT = "below_front"
GROUND_BLOCK_Y = -61


def _check_time(game_time: int) -> None:
    """Reject schedule times that are not nonnegative integers."""
    if type(game_time) is not int or game_time < 0:
        raise ValueError("game_time must be a nonnegative integer")


class ScriptedRoutine:
    """Define pure absolute-time commands with no mutable cursor or RNG use."""

    @staticmethod
    def initial_commands() -> list[dict]:
        """Only fresh E1/A runs may spawn and give supplies, never restored runs."""
        return [
            {"cmd": "spawn_bot", "args": {"name": BOT_NAME, "x": 0.5, "y": -60.0,
                                           "z": 0.5, "yaw": 0.0, "pitch": 0.0}},
            {"cmd": "give", "args": {"name": BOT_NAME, "item": "minecraft:dirt", "count": 64}},
            {"cmd": "bot_select", "args": {"name": BOT_NAME, "slot": 0}},
        ]

    @staticmethod
    def actions_for(game_time: int) -> list[dict]:
        """Return input for this time and any instantaneous boundary actions.

        A cycle is [0,200) walk, [200,300) turn/jump-walk, [300,350)
        sprint, [350,400) stop. Break and replacement occur before tick 300.
        Yaw turns another 90 degrees each cycle. The driver resolves both
        below-front targets from one bot_state, after setting this tick's input.
        No start-time offset or mutable cursor participates in the schedule.
        """
        _check_time(game_time)
        cycle, phase = divmod(game_time, CYCLE_TICKS)
        yaw = float(((cycle + (phase >= 200)) % 4) * 90)
        commands = [{"cmd": "bot_input", "args": {
            "name": BOT_NAME,
            "forward": 0.0 if phase >= 350 else 1.0,
            "strafe": 0.0,
            "jump": 200 <= phase < 300,
            "sneak": False,
            "sprint": 300 <= phase < 350,
            "yaw": yaw,
            "pitch": 0.0,
        }}]
        if phase == 300:
            commands.extend([
                {"cmd": "bot_break", "args": {"name": BOT_NAME}, "target": BELOW_FRONT},
                {"cmd": "bot_select", "args": {"name": BOT_NAME, "slot": 0}},
                {"cmd": "bot_place", "args": {"name": BOT_NAME, "face": "up"}, "target": BELOW_FRONT},
            ])
        return commands

    @staticmethod
    def next_action_time(game_time: int) -> int:
        """First later tick at which input or an instantaneous action changes."""
        _check_time(game_time)
        cycle, phase = divmod(game_time, CYCLE_TICKS)
        return cycle * CYCLE_TICKS + next(boundary for boundary in (200, 300, 350, 400) if boundary > phase)

    @staticmethod
    def below_front(state: dict) -> dict[str, int]:
        """One cardinal block ahead on the flat-world surface, even during a jump."""
        position = state["pos"]
        yaw = state["yRot"]
        if (not isinstance(position, list) or len(position) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in position)
                or type(yaw) not in (int, float) or not math.isfinite(yaw)):
            raise ValueError(f"Invalid bot_state position/rotation: {state!r}")
        direction = math.floor(yaw / 90.0 + 0.5) % 4
        offset_x, offset_z = ((0, 1), (-1, 0), (0, -1), (1, 0))[direction]
        return {"x": math.floor(position[0]) + offset_x,
                "y": GROUND_BLOCK_Y,
                "z": math.floor(position[2]) + offset_z}


def initial_commands() -> list[dict]:
    """Return fresh-run initialization through the compatible module API."""
    return ScriptedRoutine.initial_commands()


def actions_for(game_time: int) -> list[dict]:
    """Return absolute-time commands through the compatible module API."""
    return ScriptedRoutine.actions_for(game_time)


def next_action_time(game_time: int) -> int:
    """Return the next schedule boundary through the compatible module API."""
    return ScriptedRoutine.next_action_time(game_time)


def below_front(state: dict) -> dict[str, int]:
    """Resolve the ground target through the compatible module API."""
    return ScriptedRoutine.below_front(state)
