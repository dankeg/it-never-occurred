"""The lockstep agent runtime: observe, decide, act, step.

The world advances only when the runtime steps it. Polls fall on game times divisible by `poll_interval`, so a
factual run and its branches poll at the same ticks wherever they fork. At every poll, in agent-name order:

1. Each agent's skill status is read, and the agent observes. Observing drains what it heard and saw into its memory.
2. An agent decides if it has no running skill, or if someone else spoke to it (an interruption) while it was not
   itself speaking; a speaker finishes its words and hears what was said at its next decision.
3. Each deciding agent's prompt is built and digested. If a source log holds a decision for the same agent and game time
   with the same digest, that decision is reused. Otherwise the model is called; calls run in parallel across agents.
   A counterfactual fork may pass a `DecisionOverride` that swaps, resamples or holds decisions (DELEGATE_SPEC §8).
4. The decisions are applied in agent-name order, then the world steps to the next poll.

Every input to the world is a deterministic function of the decision log. A replay from the log therefore reproduces the
world-hash trace exactly (gate E3), and a branch reuses factual decisions wherever an agent's prompt is unchanged
(gate L1).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

from .agent import Agent
from .decisions import Decision, DecisionLog, prompt_digest
from .llm import LLMClient, LLMResponse, ToolCall, Turn, UsageLimitReached
from .render import block_text, door_state, short_id, stack_text
from .tools import ToolTranslator, item_id

PAUSE_TICKS = 100
THINKING_TICKS = 100  # after an instant or failed action, an agent decides again 5 s later
NIGHT_REST = (16_000, 24_000)  # agents rest from 22:00 to 06:00 unless someone speaks to them
POLL = 40  # agents decide at game times divisible by this
TICKS_PER_DAY = 24_000


def resting(now: int, rest: tuple[int, int] | None) -> bool:
    """Whether game time `now` falls in the nightly rest window `rest` (day phases, may wrap midnight)."""
    if rest is None:
        return False
    start, end = rest
    phase = now % 24_000
    return (start <= phase < end) if start < end else (phase >= start or phase < end)


PERCEIVED_ENTITY = ("name", "type", "pos", "heldItem", "activity", "load")
PERCEIVED_LISTS = ("heard", "chatter", "shout_seen", "changes", "seen_actions")


def perceived(now: int, agent: str, observation: dict) -> dict:
    """One poll's perception record for `perceptions.jsonl`: the people seen (what they hold, do and carry) and the
    speech, changes and actions perceived, without sight-ray details or terrain; empty sections are left out."""
    record = {"t": now, "agent": agent}
    entities = [{key: entity[key] for key in PERCEIVED_ENTITY if entity.get(key) is not None}
                for entity in observation.get("entities", [])]
    if entities:
        record["entities"] = entities
    for key in PERCEIVED_LISTS:
        if observation.get(key):
            record[key] = observation[key]
    return record


CARRIED = ("minecraft:bread", "minecraft:wheat", "minecraft:hay_block", "minecraft:lantern")


def carried(observation: dict) -> dict[str, int]:
    """The Jubilee goods an agent holds (summed over its inventory), for the trajectory: how much bread, wheat, hay
    and lanterns it carried at each poll."""
    counts: dict[str, int] = {}
    for stack in observation.get("self", {}).get("inventory", []):
        if stack["item"] in CARRIED:
            name = stack["item"].removeprefix("minecraft:")
            counts[name] = counts.get(name, 0) + stack["count"]
    return dict(sorted(counts.items()))


JOURNAL_TICKS = 20  # writing a journal entry takes a moment and changes nothing in the world


def sleep_as_wait(call: ToolCall | None, now: int) -> ToolCall | None:
    """The world action for a decision: `sleep` is a wait until the next 06:00 (day phase 0), which speech interrupts
    like any running action; `journal` is a short wait (the entry is kept in the decision log and the agent's memory);
    every other call is unchanged."""
    if call is not None and call.name == "journal":
        return ToolCall("wait", {"ticks": JOURNAL_TICKS})
    if call is None or call.name != "sleep":
        return call
    return ToolCall("wait", {"ticks": TICKS_PER_DAY - now % TICKS_PER_DAY})


def ferry_hint(call: ToolCall, observation: dict) -> str:
    """Say which way a ferry request failed: the rider asked for its own bank, or the boat is moored elsewhere.

    The river runs through x = -6..6, so a position's bank is the sign of its x; the boat's mooring is named only
    when the rider can see the boat."""
    position = observation.get("self", {}).get("pos")
    side = None if position is None else "west" if position[0] < 0 else "east"
    if side is not None and call.args.get("to") == side:
        return f" (you are already on the {side} bank: name the other landing, and stand where the boat is moored)"
    boats = [entity for entity in observation.get("entities", []) if entity.get("type", "").endswith("_boat")]
    if boats:
        moored = "west" if boats[0]["pos"][0] < 0 else "east"
        return f" (the boat is moored at the {moored} landing, not here; it does not come when called)"
    return " (the boat is not moored at this landing)"


def failure_hint(call: ToolCall | None, outcome: str, observation: dict, details: dict | None = None) -> str:
    """A short explanation for common failures, like a game's on-screen message (empty if none applies).

    Agents only see bare reason codes otherwise, and tend to repeat the same failing action. Optional rejection
    details provide the engine's hint or the affected item, limit, have and remaining counts.
    """
    if call is None or not outcome.startswith(("not possible", "failed", "invalid")):
        return ""
    if "in_water_no_bank" in outcome:
        return " (you are in water and no bank is within 16 blocks)"
    inventory = observation.get("self", {}).get("inventory", [])
    if "too_heavy" in outcome:
        details = details or {}
        if isinstance(details.get("hint"), str) and details["hint"]:
            return f" ({details['hint']})"
        identifier = details.get("item", "bread" if call.name == "bake" else call.args.get("item"))
        limit, have, remaining = details.get("limit"), details.get("have"), details.get("remaining")
        subject = str(call.args.get("target", "the recipient")) if call.name == "give" else "you"
        if isinstance(identifier, str):
            identifier = item_id(identifier)
            if call.name != "give":
                if limit is None:
                    limit = observation.get("self", {}).get("carry_limits", {}).get(identifier)
                if have is None:
                    have = sum(entry["count"] for entry in inventory if item_id(entry["item"]) == identifier)
            facts = []
            if limit is not None:
                facts.append(f"{subject} can carry {limit} {short_id(identifier)} at most")
            if have is not None:
                verb = "have" if subject == "you" else "has"
                facts.append(f"{subject} {verb} {have} {short_id(identifier)}")
            if remaining is not None:
                facts.append(f"{subject} can take {remaining} more")
            if facts:
                return " (" + "; ".join(facts) + "; partial amounts are not taken)"
        return " (that would exceed the carrying limit; partial amounts are not taken)"
    if "no_oven" in outcome:
        return " (baking needs an oven, a smoker within reach and in sight)"
    if "not_enough_wheat" in outcome:
        wheat = sum(entry["count"] for entry in inventory if item_id(entry["item"]) == "minecraft:wheat")
        return f" (each loaf needs 3 wheat from your inventory; you have {wheat})"
    if "mouldy" in outcome:
        return " (mouldy bread cannot be eaten)"
    if call.name == "speak" and "characters" in outcome:
        return (f" (yours was {len(str(call.args.get('text', '')).split())} words; "
                "say it in at most about 100, or in two parts)")
    if call.name == "speak" and "busy" in outcome:
        return " (you are still saying your previous words; speech takes about a second per 3 words)"
    if call.name == "ask" and "busy" in outcome:
        return " (that villager is talking with someone else right now)"
    if "rate_limited" in outcome:
        return " (you can shout only once every 30 seconds; talk or whisper instead, or wait)"
    if call.name == "use" and "not_usable" in outcome:
        return " (use works on doors, trapdoors, gates, bells, levers and buttons, not beds or other blocks)"
    if call.name == "ferry" and "no_boat" in outcome:
        return ferry_hint(call, observation)
    if call.name == "collect" and "target_not_found" in outcome:
        return (" (no dropped items in sight within 6 blocks, or reachable drops remembered from your recent mining; "
                "mined items may still be on the ground)")
    if call.name == "ask" and "target_not_found" in outcome:
        return (" (ask only works on a villager within 8 blocks, using the villager id shown in what you can see; "
                "to talk to a person, use speak with 'to')")
    if call.name == "ask" and "reach" in outcome:
        npc = str(call.args.get("npc", "")).removeprefix("npc:")
        if npc not in {str(entity.get("npcId", "")).removeprefix("npc:") for entity in observation.get("entities", [])}:
            return f" ({npc} is not in sight; villagers move between home, work and the market during the day)"
    if "reach" in outcome:
        return " (stand within about 4 blocks of it first)"
    if call.name == "read_book":
        slots = [entry["slot"] for entry in inventory if entry["item"] == "minecraft:written_book"]
        return f" (written books you carry are in slots {slots})" if slots else " (you carry no written book)"
    if "cannot_afford" in outcome:
        emeralds = sum(entry["count"] for entry in inventory if entry["item"] == "minecraft:emerald")
        return f" (you have {emeralds} emeralds)"
    if call.name == "move_to" and "no_path" in outcome:
        hint = " (no walkable route: it may be across water, enclosed, or not yet built)"
        if any(short_id(block.get("block", "")).endswith("_door") and door_state(block) == "closed"
               for block in observation.get("blocks", [])):
            hint += " (a closed door may be in the way: `use` it to open it)"
        return hint
    if call.name == "drop" and "missing_items" in outcome:
        return " (you can only drop items you carry; check the item and count in your inventory)"
    if call.name == "hold" and "missing_items" in outcome:
        return " (you can only hold an item you carry; check the item in your inventory)"
    if call.name == "eat" and "missing_items" in outcome:
        return " (you can only eat food you carry; check your inventory)"
    if call.name == "eat" and "not_food" in outcome:
        return " (that item is not food; choose food from your inventory)"
    if "missing_items" in outcome:
        return " (you don't carry the materials it needs; check your inventory)"
    return ""


class BudgetExceeded(RuntimeError):
    """Raised before a model call that would exceed the run's model-call budget."""


class ReplayDiverged(RuntimeError):
    """Raised when a prompt has no matching source decision and the runtime has no model to ask."""


@dataclass(frozen=True)
class DecisionOverride:
    """How a counterfactual fork changes a run's decisions (DELEGATE_SPEC §8); it needs a source log.

    At the decision point (`agent`, `game_time`): with a `call` (swap), the agent takes that call without asking the
    model; with `resample` (placebo), the model is asked afresh even if the source log holds a matching decision.
    Agents in `hold` (swap-hold) take exactly the source log's decision at each of their logged polls from `hold_from`
    to `hold_until` (inclusive; None is open-ended), without a prompt match or a model call, and decide at no other poll
    in that window. Holding every agent replays a run's logged actions even when its prompts were built by older code.

    From `game_time` on, agents in `resample_agents` are asked afresh at every decision, never reusing the source log
    (the control for an arm that changes those agents' briefs, whose prompts then never match). A (listener, speaker)
    pair in `mute` removes everything the listener would hear of that speaker (speech, chatter, shouting seen) from
    `game_time` on: an information intervention on the listener's perception, not on the world. Factual action
    notices remain perceivable; they are not speech.

    Under an override, a source decision is reused when the prompt digest matches under the source decision's own model,
    so a branch continued by another model still reuses the logged decisions of agents whose prompts did not change.
    """

    agent: str | None = None
    game_time: int | None = None
    call: ToolCall | None = None
    resample: bool = False
    hold: frozenset[str] = frozenset()
    hold_from: int = 0
    hold_until: int | None = None
    resample_agents: frozenset[str] = frozenset()
    mute: frozenset[tuple[str, str]] = frozenset()

    def at(self, agent: str, now: int) -> bool:
        """Whether (`agent`, `now`) is the overridden decision point."""
        return agent == self.agent and now == self.game_time

    def holds(self, agent: str, now: int) -> bool:
        """Whether `agent` is held to its logged decisions at game time `now`."""
        return agent in self.hold and now >= self.hold_from and (self.hold_until is None or now <= self.hold_until)

    def resampling(self, agent: str, now: int) -> bool:
        """Whether `agent` must be asked afresh at `now` (it is in `resample_agents`, from `game_time` on)."""
        return agent in self.resample_agents and self.game_time is not None and now >= self.game_time

    def perceived(self, listener: str, observation: dict, now: int) -> dict:
        """The observation `listener` perceives at `now`: muted speakers' speech removed, otherwise unchanged."""
        muted = {speaker for pair_listener, speaker in self.mute if pair_listener == listener}
        if not muted or self.game_time is None or now < self.game_time:
            return observation
        return {**observation, **{key: [item for item in observation.get(key, [])
                                       if item.get("kind") in {"block_use", "transfer"}
                                       or item.get("speaker") not in muted]
                                  for key in ("heard", "chatter", "shout_seen") if key in observation}}


@dataclass
class RunStats:
    """Counts for one run: decisions made, model calls, replayed decisions, and tokens used.

    `input_tokens` counts uncached input; cached prompt prefixes are counted separately (read and written).
    `model_retries` counts extra requests a client made because a reply had no tool call (not in `model_calls`).
    """

    decisions: int = 0
    model_calls: int = 0
    replayed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_errors: int = 0
    model_seconds: float = 0.0
    model_retries: int = 0


@dataclass
class _Pending:
    """An agent that must decide at this poll, with the observation it decides on."""

    agent: Agent
    observation: dict
    running: dict | None
    interrupted: bool
    system: str = ""
    user: str = ""
    history: tuple[Turn, ...] = ()
    digest: str = ""
    result: dict | None = None


class SimulationRuntime:
    """Drives role agents through a lockstep world and records decisions and world hashes."""

    def __init__(self, client, agents: list[Agent], model: LLMClient | None, out_dir: Path, *,
                 command_error: type[Exception], source: DecisionLog | None = None, poll_interval: int = 40,
                 hash_region: dict | None = None, hash_every: int = 20, max_model_calls: int | None = None,
                 translator: ToolTranslator | None = None, sample_seed: int | None = None,
                 model_concurrency: int | None = None, min_gap: int = 0, rest: tuple[int, int] | None = None,
                 override: DecisionOverride | None = None, perception_log: bool = False):
        """Configure a run.

        `client` is a harness control client and `command_error` the exception it raises when the server rejects a
        command. `source` is the decision log to reuse decisions from (the factual run for a branch, or the run itself
        for a replay). With `model=None`, any decision missing from `source` raises `ReplayDiverged`.
        `poll_interval` must be a multiple of 20 so world hashes stay on the 20-tick grid; polls happen at game times
        divisible by it. World hashes of `hash_region` are recorded at game times divisible by `hash_every`; with no
        region, nothing is hashed.

        With a `sample_seed`, each model call is seeded from it and the prompt digest, so on a backend that honours
        seeds the same prompt in the same run always yields the same decision (branches: one seed each). Without one
        (the default), calls are unseeded; replays still reproduce runs exactly through the decision log, and local
        servers can batch and reuse prompt prefixes, which seeded requests prevent.
        `model_concurrency` caps simultaneous model calls (1 for a local server that serves one request at a time).
        `min_gap` is thinking time: after an action that finished at once or failed, the agent decides again only
        `min_gap` ticks later. `rest` is a nightly window of day phases (e.g. 16,000 to 24,000, 22:00 to 06:00) when
        idle agents sleep and don't decide. Speech addressed to an agent wakes it in both cases.
        `override` changes decisions for a counterfactual fork (see `DecisionOverride`); `override_applied` then says
        whether its decision point was reached.
        With `perception_log`, every poll appends what each agent perceived (people with what they hold, do and
        carry; speech; changes; actions) to `perceptions.jsonl`, the record leak analyses read, and each trajectory
        record gains the Jubilee goods the agent carries; both are outputs only.
        """
        if poll_interval <= 0 or poll_interval % 20:
            raise ValueError("poll_interval must be a positive multiple of 20")
        if override is not None and source is None:
            raise ValueError("a decision override needs a source log")
        self._client = client
        self._agents = sorted(agents, key=lambda agent: agent.name)
        self._model = model
        self._out_dir = Path(out_dir)
        self._command_error = command_error
        self._source = source
        self._poll_interval = poll_interval
        self._hash_region = hash_region
        self._hash_every = hash_every
        self._max_model_calls = max_model_calls
        self._translator = translator or ToolTranslator()
        self._sample_seed = sample_seed
        self._model_concurrency = model_concurrency
        self._min_gap = min_gap
        self._rest = rest
        self._override = override
        self._perception_log = perception_log
        self.override_applied = False
        self._ready_at: dict[str, int] = {}
        self._retries_at_start = getattr(model, "retries", 0)
        self.log = DecisionLog(self._out_dir / "decisions.jsonl")
        self.hashes: list[dict] = []
        self.stats = RunStats()

    @property
    def agents(self) -> list[Agent]:
        """The run's agents, in decision order (by name)."""
        return list(self._agents)

    @property
    def model_name(self) -> str:
        """The model id used in prompt digests (the source log's model when replaying without a model)."""
        return self._model.model if self._model is not None else "replay"

    def game_time(self) -> int:
        """Return the world's current game time."""
        return self._client.request("ping")["gameTime"]

    def run(self, until: int) -> RunStats:
        """Run polls and steps until game time `until`, then write hashes and stats. Returns the run's stats."""
        now = self.game_time()
        if now % 20:
            raise ValueError(f"Runs must start on the 20-tick grid, not at {now}")
        try:
            while now < until:
                if now % self._poll_interval == 0:
                    self._poll(now)
                ticks = min(self._poll_interval - now % self._poll_interval, until - now)
                args = {"n": ticks}
                if self._hash_region is not None:
                    args.update(hash_every=self._hash_every, region=self._hash_region)
                result = self._client.request("step", args)
                self.hashes.extend(result.get("hashes", []))
                now += ticks
                if result["gameTime"] != now:
                    raise RuntimeError(f"step advanced to {result['gameTime']}, expected {now}")
        finally:
            self._write_outputs()
        return self.stats

    def save_state(self, path: Path) -> None:
        """Write the agents (briefs and memories) at the current game time, for forking."""
        state = {"gameTime": self.game_time(), "agents": [agent.to_json() for agent in self._agents],
                 "readyAt": dict(self._ready_at)}
        Path(path).write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def load_agents(path: Path) -> tuple[int, list[Agent]]:
        """Read a `save_state` file and return its game time and restored agents."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return state["gameTime"], [Agent.from_json(data) for data in state["agents"]]

    def restore_timing(self, path: Path) -> None:
        """Restore each agent's earliest next decision from a `save_state` file (for exact branches)."""
        self._ready_at = dict(json.loads(Path(path).read_text(encoding="utf-8")).get("readyAt", {}))

    def _poll(self, now: int) -> None:
        """Observe every agent, then decide and act for those that must decide now."""
        pending, trajectory, perceptions = [], [], []
        for agent in self._agents:
            status = self._client.request("skill_status", {"name": agent.name})
            running = status if status["state"] == "running" else None
            observation = self._client.request("observe", {"name": agent.name, "include_terrain": False})
            if self._override is not None and self._override.mute:
                observation = self._override.perceived(agent.name, observation, now)
            if running is None:
                progress = status.get("progress") or {}
                reason = status.get("reason") or progress.get("reason")
                if status["state"] == "failed" and reason in {"too_heavy", "no_oven", "not_enough_wheat", "mouldy"}:
                    previous = next((action for action in agent.memory.to_json()["actions"]
                                     if action.get("skillId") is not None
                                     and action["skillId"] == status.get("skillId")), None)
                    if previous is not None:
                        call = ToolCall(previous["tool"], previous["args"])
                        hint = failure_hint(call, f"failed: {reason}", observation, progress)
                        status = {**status, "reason": f"{reason}{hint}"}
                agent.memory.resolve_skill(status)
            trajectory.append({"t": now, "agent": agent.name, "pos": [round(v, 2) for v in observation["self"]["pos"]],
                               "skill": status.get("skill"), "state": status["state"]})
            if self._perception_log:
                perceptions.append(perceived(now, agent.name, observation))
                carrying = carried(observation)
                if carrying:
                    trajectory[-1]["carrying"] = carrying
            spoke = agent.memory.absorb(observation, agent.name)
            speaking = running is not None and running.get("skill") == "speak"  # a speaker finishes the sentence first
            if self._override is not None and self._override.holds(agent.name, now):
                logged = self._source.get(agent.name, now)
                if logged is not None:  # a held agent decides exactly at its logged polls
                    pending.append(_Pending(agent, observation, running, interrupted=logged.interrupted))
            elif (spoke and not speaking) or (running is None and self._awake(agent.name, now)):
                pending.append(_Pending(agent, observation, running, interrupted=running is not None))
        self._record_trajectory(trajectory)
        if perceptions:
            self._append_jsonl("perceptions.jsonl", perceptions)
        if not pending:
            return
        decisions = self._decide(now, pending)
        for item, decision in zip(pending, decisions):
            outcome = self._apply(now, item, decision.tool_call)
            self.log.append(replace(decision, outcome=outcome))

    def _decide(self, now: int, pending: list[_Pending]) -> list[Decision]:
        """Return one decision per pending agent: reused from the source log where the prompt matches, else a model
        call."""
        decisions: list[Decision | None] = []
        misses = []
        for item in pending:
            item.system = item.agent.system_prompt()
            item.user = item.agent.next_prompt(item.observation, item.interrupted, now)
            item.history = item.agent.history()
            reused = self._forced(now, item) or self._reuse(now, item)
            decisions.append(reused)
            if reused is None:
                misses.append(len(decisions) - 1)
        if misses:
            if self._model is None:
                names = [pending[index].agent.name for index in misses]
                diverged = [{"gameTime": now, "agent": pending[index].agent.name, "digest": pending[index].digest,
                             "user_prompt": pending[index].user} for index in misses]
                self._out_dir.mkdir(parents=True, exist_ok=True)
                (self._out_dir / "diverged.json").write_text(json.dumps(diverged, indent=1) + "\n", encoding="utf-8")
                raise ReplayDiverged(f"No matching source decision at tick {now} for {names} (see diverged.json)")
            if self._max_model_calls is not None and self.stats.model_calls + len(misses) > self._max_model_calls:
                raise BudgetExceeded(f"Model-call budget {self._max_model_calls} would be exceeded at tick {now}")
            with ThreadPoolExecutor(max_workers=min(len(misses), self._model_concurrency or len(misses))) as pool:
                responses = list(pool.map(lambda index: self._ask(pending[index]), misses))
            for index, (response, seconds) in zip(misses, responses):
                item, call = pending[index], response.tool_call
                decisions[index] = Decision(now, item.agent.name, item.digest, item.interrupted, "model",
                                            call.name if call else None, call.args if call else None,
                                            response.model, response.input_tokens, response.output_tokens, item.user,
                                            cached_tokens=response.cache_read_tokens, seconds=round(seconds, 3))
                self.stats.model_calls += 1
                self.stats.model_seconds += seconds
                self.stats.input_tokens += response.input_tokens
                self.stats.output_tokens += response.output_tokens
                self.stats.cache_read_tokens += response.cache_read_tokens
                self.stats.cache_write_tokens += response.cache_write_tokens
        self.stats.decisions += len(pending)
        return decisions

    def _awake(self, name: str, now: int) -> bool:
        """Whether an idle agent decides now: past its thinking time, and not in the nightly rest window."""
        return now >= self._ready_at.get(name, 0) and not resting(now, self._rest)

    def _ask(self, item: _Pending) -> tuple[LLMResponse, float]:
        """One model call and its wall time in seconds. A call that still fails after the client's retries becomes a
        decision with no action (recorded as a model error), so one bad request cannot end a long run; replays
        reproduce it exactly."""
        started = time.monotonic()
        try:
            response = self._model.complete(item.system, item.user, item.agent.tools, seed=self._seed(item.digest),
                                            history=item.history)
        except UsageLimitReached as limit:
            raise BudgetExceeded(f"Provider usage limit reached: {limit}") from limit
        except RuntimeError as failure:
            self.stats.model_errors += 1
            response = LLMResponse(None, text=f"model error: {failure}"[:300], model=self.model_name)
        return response, time.monotonic() - started

    def _seed(self, digest: str) -> int | None:
        """A 31-bit sampling seed from the run's sample seed and the prompt digest (None when unseeded)."""
        if self._sample_seed is None:
            return None
        return int(hashlib.sha256(f"{self._sample_seed}:{digest}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF

    def _forced(self, now: int, item: _Pending) -> Decision | None:
        """The override's decision for this agent and tick: a held agent's logged decision (`held`) or the swapped call
        (`forced`). None when the override does not decide it."""
        override = self._override
        if override is None:
            return None
        name = item.agent.name
        if override.holds(name, now):
            logged = self._source.get(name, now)
            item.digest = prompt_digest(self.model_name, item.system, item.user, item.agent.tools, item.history)
            return Decision(now, name, item.digest, item.interrupted, "held", logged.tool, logged.args, logged.model, 0,
                            0, item.user)
        if override.at(name, now) and override.call is not None:
            self.override_applied = True
            item.digest = prompt_digest(self.model_name, item.system, item.user, item.agent.tools, item.history)
            return Decision(now, name, item.digest, item.interrupted, "forced", override.call.name,
                            dict(override.call.args), self.model_name, 0, 0, item.user)
        return None

    def _reuse(self, now: int, item: _Pending) -> Decision | None:
        """Return the source decision for this agent and tick if its prompt digest matches, else None. A placebo
        override never reuses at its decision point; under any override, digests are compared under the source
        decision's model."""
        cached = self._source.get(item.agent.name, now) if self._source is not None else None
        model = self.model_name if self._model is not None else (cached.model if cached else "replay")
        item.digest = prompt_digest(model, item.system, item.user, item.agent.tools, item.history)
        override = self._override
        if override is not None and ((override.resample and override.at(item.agent.name, now))
                                     or override.resampling(item.agent.name, now)):
            self.override_applied = True
            return None
        digest = item.digest
        if override is not None and cached is not None and cached.model != model:
            digest = prompt_digest(cached.model, item.system, item.user, item.agent.tools, item.history)
        if cached is None or cached.digest != digest:
            return None
        self.stats.replayed += 1
        return Decision(now, item.agent.name, digest, item.interrupted, "replay", cached.tool, cached.args,
                        cached.model, 0, 0, item.user)

    def _apply(self, now: int, item: _Pending, call: ToolCall | None) -> str:
        """Carry out one decision in the world, record it in the agent's memory, and return its immediate outcome."""
        agent, memory = item.agent, item.agent.memory
        outcome, skill_id, read = self._execute(agent.name, item, sleep_as_wait(call, now))
        outcome += failure_hint(call, outcome, item.observation, item.result)
        carried_on = outcome == "carried on"  # the running skill continues; the agent decides when it ends
        if skill_id is None and self._min_gap and not carried_on:
            self._ready_at[agent.name] = now + self._min_gap
        memory.record_action(now, call or ToolCall("(none)", {}), outcome, skill_id=skill_id)
        agent.finish_turn(item.user, call, outcome)
        if skill_id is not None and item.running is not None:
            memory.resolve_skill({"skillId": item.running["skillId"], "state": "stopped",
                                  "reason": "you started something else"})
        if read is not None:
            memory.record_read(now, read)
            if self._perception_log:
                self._append_jsonl("perceptions.jsonl", [{"t": now, "agent": agent.name, "read": read}])
        return outcome

    def _execute(self, bot: str, item: _Pending, call: ToolCall | None) -> tuple[str, str | None, str | None]:
        """Send one decision to the world. Returns (outcome, started skill id, text the agent read)."""
        if call is None:
            return "no action chosen", None, None
        try:
            action = self._translator.translate(bot, call)
        except (ValueError, KeyError, TypeError) as error:
            return f"invalid action: {error}", None, None
        if action is None:
            if item.running is not None:
                return "carried on", None, None
            action = self._translator.translate(bot, ToolCall("wait", {"ticks": PAUSE_TICKS}))
        try:
            result = self._client.request(action.command, action.args)
        except self._command_error as error:
            return f"failed: {str(error).split(': ', 1)[-1]}", None, None
        item.result = result
        if action.is_skill:
            if not result.get("accepted"):
                return f"not possible: {result.get('reason')}", None, None
            return "started", result["skillId"], None
        if call.name == "open_container":
            contents = ", ".join(stack_text(entry) for entry in result.get("items", []))
            return "done", None, f"container at {block_text(action.args['pos'])}: {contents or 'empty'}"
        if call.name == "read_book":
            return "done", None, (f"book \"{result.get('title', '')}\" by {result.get('author', '?')}: "
                                  + " | ".join(result.get("pages", [])))
        return "done", None, None

    def _record_trajectory(self, records: list[dict]) -> None:
        """Append each agent's position and activity at this poll to `trajectory.jsonl` (for viewers and analysis)."""
        self._append_jsonl("trajectory.jsonl", records)

    def _append_jsonl(self, name: str, records: list[dict]) -> None:
        """Append compact JSON lines to one of the run's output files."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with (self._out_dir / name).open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
                              for record in records)

    def _write_outputs(self) -> None:
        """Write the world-hash trace and run stats next to the decision log."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.stats.model_retries = getattr(self._model, "retries", 0) - self._retries_at_start
        (self._out_dir / "hashes.json").write_text(json.dumps(self.hashes) + "\n", encoding="utf-8")
        (self._out_dir / "stats.json").write_text(json.dumps(asdict(self.stats), indent=1) + "\n", encoding="utf-8")
