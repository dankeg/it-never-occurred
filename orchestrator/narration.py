"""The narrated baseline (M6, gate N1): the same agents, with acts resolved by an LLM game master instead of the engine.

`GameMaster` is a Concordia-style narrator. It keeps the world as text, checks each act for physical plausibility, and
describes the act's effects, its duration, and what each person perceives. `NarratedRuntime` runs the same agents,
prompts and 40-tick decision grid as the engine runtime, with the GM in place of the world.

An optional `ShadowEngine` executes every GM-accepted act in a real engine world at the same game time. Its verdict
(accepted or not, the reason code, and how the skill really ended) is the oracle for the narration-error rate. Agents
never see the shadow. See `docs/M6_NARRATION_DESIGN.md`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .agent import Agent
from .decisions import Decision, DecisionLog, prompt_digest
from .llm import LLMClient, LLMResponse, ToolCall, UsageLimitReached
from .render import ObservationRenderer, clock, short_id
from .runtime import BudgetExceeded, RunStats, resting
from .tools import TOOL_DEFINITIONS, ToolTranslator, tool_definitions

GM_RULES = """You are the game master of a simulation of a small town, in a world that works like Minecraft. Time \
passes in ticks (20 ticks = 1 second). People act one at a time. For each act, check whether it is physically \
possible given the current world state (where the person is, distances, what they carry, tools and materials, \
recipes, who may open which container), and describe its effects. Reject impossible acts and say why. Give a \
realistic duration in ticks (walking is about 4 blocks per second). Say what the acting person now perceives (their \
location, what they carry, what is around them), and what anyone else nearby perceives of the act; people far away \
perceive nothing. Keep the world consistent with everything that has happened."""

ENGINE_RULES = """Rules the world enforces exactly:
- `ask` questions a villager (an id such as adult_07) within 8 blocks. The people listed in the world description are \
not villagers and cannot be asked; to talk with a person, one uses `speak`.
- Speech lasts one tick per 0.15 words (about 3 words per second), 1.5 times as long when shouted. Until a person's \
speech has finished, they cannot start any other act, including speaking again.
- A person may start a new act while still walking or working: the new act replaces the old one, which stops where \
it is.
- Whispering carries 3 blocks, talking 8, shouting 24. People only catch words while in range for the whole speech. \
Whispers use the same speaking rate as talk, with no shout cooldown. Beyond 3 but within 8 blocks, a person who \
can see the whisperer notices talking but cannot make out any words; without sight they perceive nothing.
- A person can shout at most once every 600 ticks (30 seconds); talk and whisper are not rate-limited.
- `use` takes 5 ticks by default (use_ticks): open or close a door, trapdoor or gate, ring a bell, flip a lever or \
press a button using the block's vanilla action. Reject out-of-reach targets with out_of_reach and other blocks \
with not_usable. Beds are not usable. Nearby witnesses can see the action; everyone within 48 blocks of a rung \
bell hears: bell rang at (x, y, z). These are factual action notices, not spoken words.
- Buildings have doors (two per doorway), all open at the start; `use` opens or closes a door. A closed door blocks \
sight and passage.
- A container's owner does not lock it: when allowed is null anyone may open or take from it. An allowed list \
still restricts access. Taking someone else's owned items is a visible transfer, not speech.
- Dropping items takes 10 ticks and requires carrying the requested item and count. The items land at the person's \
feet, anyone can pick them up (the dropper takes them back only with collect, not by standing on them), and they \
vanish after about 6,000 ticks (five minutes) on the ground.
- Holding an item takes 0 ticks and requires carrying that item (otherwise missing_items). Select its hotbar slot, \
or swap it into the selected hotbar slot if needed, without consuming or transferring items. Only people who can \
see the person can see the held item; this reveals no inventory contents or counts.
- Reach (for containers, blocks, stalls, giving and building) is about 4.5 blocks from where the person actually \
stands."""


def gm_system_prompt(tools: list[dict], *, night_sight: bool = False, visible_loads: dict[str, int] | None = None,
                     carry_limits: dict[str, int] | None = None, fresh_days: int | None = None,
                     bake_ticks: int | None = None, villager_purchases: list[dict] | None = None) -> str:
    """The GM's acts and hard rules, appending one plain sentence per enabled sight, load or goods mechanic; with
    none enabled the prompt is the default one."""
    actions = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in tools)
    rules = []
    if night_sight:
        rules.append("- At night, from 19:00 to 05:00, people and villagers can see a target beyond 8 blocks only if "
                     "it is within 5 blocks of a light-emitting block or is a person holding a light-emitting item in "
                     "their main hand. Sight within 8 blocks works as by day, line of sight is still required "
                     "(including to notice someone whispering), and hearing is unchanged.")
    if visible_loads:
        thresholds = " or ".join(f"{threshold} {short_id(item).replace('_', ' ')}"
                                 for item, threshold in sorted(visible_loads.items()))
        rules.append(f"- When a visible person's inventory totals at least {thresholds}, describe them to people and "
                     "villagers as carrying a sack of the matching item. Reveal only the sack and item, never exact "
                     "inventory counts or unconfigured items.")
    if carry_limits:
        limits = ", ".join(f"{limit} {short_id(item)}" for item, limit in sorted(carry_limits.items()))
        rules.append(f"- A person can carry at most {limits} across their whole inventory; any gain exceeding a "
                     "limit fails with too_heavy, and a partial amount is never taken.")
    if fresh_days is not None:
        rules.append("- Bread records its baked day, with unstamped bread counted as day 0 and negative days allowed; "
                     f"it is shown as mouldy and cannot be eaten when today minus its baked day is greater than "
                     f"{fresh_days}.")
    if bake_ticks is not None:
        rules.append("- Baking at an unlit smoker within reach and in sight turns 3 wheat per loaf into 1–32 bread at "
                     f"{bake_ticks} ticks per loaf, stamps the current day and respects carrying limits, while the "
                     "oven glows and others see baking at the oven until the batch finishes or is interrupted.")
    if carry_limits or fresh_days is not None or bake_ticks is not None or villager_purchases:
        rules.append("- A stall's owner may stock its chest and take its takings; everyone else must buy at the "
                     "posted price.")
    if villager_purchases:
        rules.append("- Villagers on market visits buy the configured counts of the freshest non-mouldy stock at "
                     "the posted emerald price or leave without buying when the stall is sold out.")
    prompt = f"{GM_RULES}\n\nActs people can attempt:\n{actions}\n\n{ENGINE_RULES}"
    return prompt + ("\n" + "\n".join(rules) if rules else "")


RESOLVE_TOOL = {
    "name": "resolve",
    "description": "Resolve one person's act.",
    "parameters": {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean", "description": "Whether the act is physically possible and happens."},
            "reason": {"type": "string", "description": "Why it was rejected (empty if accepted)."},
            "duration_ticks": {"type": "integer", "minimum": 0, "description": "How long the act takes."},
            "effects": {"type": "array", "items": {"type": "string"},
                        "description": "What changed in the world (positions, inventories, blocks, documents)."},
            "actor_view": {"type": "string",
                           "description": "What the acting person now perceives: location, "
                           "what they carry, surroundings."},
            "observations": {"type": "array", "description": "What other people perceive of this act.",
                             "items": {"type": "object", "properties": {"person": {"type": "string"},
                                                                       "text": {"type": "string"}},
                                       "required": ["person", "text"]}},
        },
        "required": ["accepted", "reason", "duration_ticks", "effects", "actor_view", "observations"],
    },
}


@dataclass(frozen=True)
class Resolution:
    """The GM's ruling on one act."""

    accepted: bool
    reason: str
    duration_ticks: int
    effects: list[str]
    actor_view: str
    observations: dict[str, str]
    error: bool = False  # the GM's call failed after retries: no ruling, and not a rejection by judgment


class GameMaster:
    """A narrated GM: the world is text (a description plus an event log), and acts are resolved by model judgment."""

    def __init__(self, model: LLMClient, world: str, views: dict[str, str], max_events: int = 80, *,
                 night_sight: bool = False, visible_loads: dict[str, int] | None = None,
                 carry_limits: dict[str, int] | None = None, fresh_days: int | None = None,
                 bake_ticks: int | None = None, villager_purchases: list[dict] | None = None):
        """Start from the world and each initial view, opting into sight, load and goods rules only when configured
        (kept across tool reconfiguration)."""
        self.model = model
        self._rules = {"night_sight": night_sight, "visible_loads": dict(visible_loads or {}),
                       "carry_limits": carry_limits, "fresh_days": fresh_days, "bake_ticks": bake_ticks,
                       "villager_purchases": villager_purchases}
        self.configure_tools()
        self._world = world
        self._views = dict(views)
        self._events: list[str] = []
        self._max_events = max_events
        self.calls = 0
        self.errors = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0

    def view(self, agent: str) -> str:
        """The agent's current view, as last narrated."""
        return self._views[agent]

    def sync_view(self, agent: str, view: str) -> None:
        """Replace the agent's narrated view (used to judge each ruling from the true state)."""
        self._views[agent] = view

    def configure_tools(self, *, daily_hunger: bool = False, extras: tuple[str, ...] = ()) -> None:
        """Match the GM's available acts to the agents' scenario, keeping the configured sight, load and goods
        rules."""
        tools = tool_definitions(daily_hunger=daily_hunger, extras=extras)
        self._system = gm_system_prompt(tools, **self._rules)

    def resolve(self, game_time: int, agent: str, call: ToolCall) -> Resolution:
        """Rule on `agent`'s act at `game_time`, then update the event log and the agent's view."""
        args = {key: value for key, value in call.args.items() if key != "note"}
        recent = "\n".join(self._events[-self._max_events:]) or "(nothing yet)"
        user = (f"World at the start:\n{self._world}\n\nWhat has happened since (oldest first):\n{recent}\n\n"
                f"Now ({clock(game_time)}, tick {game_time}), {agent} attempts: {call.name} {json.dumps(args)}\n"
                f"{agent}'s current view: {self._views[agent]}\n\nResolve this act.")
        try:
            response = self.model.complete(self._system, user, [RESOLVE_TOOL])
        except UsageLimitReached as limit:
            raise BudgetExceeded(f"Provider usage limit reached: {limit}") from limit
        except RuntimeError:  # still failing after the client's retries: no ruling, so the run goes on
            self.errors += 1
            return Resolution(False, "the game master could not rule (a service error); try again", 0, [],
                              self._views[agent], {}, error=True)
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.cached_tokens += response.cache_read_tokens
        self.output_tokens += response.output_tokens
        ruling = (response.tool_call.args if response.tool_call is not None else {})
        resolution = Resolution(bool(ruling.get("accepted", False)), str(ruling.get("reason", "no ruling")),
                                max(0, int(ruling.get("duration_ticks", 0) or 0)), list(ruling.get("effects", [])),
                                str(ruling.get("actor_view") or self._views[agent]),
                                {item["person"]: str(item.get("text", "")) for item in ruling.get("observations", [])
                                 if isinstance(item, dict) and item.get("person") in self._views})
        verdict = "happened" if resolution.accepted else f"was impossible ({resolution.reason})"
        self._events.append(f"[tick {game_time}] {agent}: {call.name} {json.dumps(args)} {verdict}. "
                            + " ".join(resolution.effects))
        self._views[agent] = resolution.actor_view
        return resolution


class ShadowEngine:
    """Executes GM-accepted acts in a real engine world and reports the engine's verdict on each."""

    def __init__(self, client, command_error: type[Exception], translator: ToolTranslator | None = None):
        """Wrap a harness client whose world starts in the same state as the GM's description."""
        self._client = client
        self._command_error = command_error
        self._translator = translator or ToolTranslator()

    def advance(self, game_time: int) -> None:
        """Step the shadow world forward to `game_time`."""
        now = self._client.request("ping")["gameTime"]
        if game_time > now:
            self._client.request("step", {"n": game_time - now})

    def skill_state(self, agent: str) -> str:
        """Return how the agent's current or last skill stands in the engine (e.g. `running`, `failed (no_path)`)."""
        status = self._client.request("skill_status", {"name": agent})
        return status["state"] + (f" ({status['reason']})" if status.get("reason") else "")

    def check(self, agent: str, call: ToolCall) -> dict | None:
        """The engine's dry-run verdict on an act that runs over time: whether it is feasible now, why not, and its
        estimated ticks. None for instant acts (documents, containers), which have no dry run."""
        try:
            action = self._translator.translate(agent, call)
        except (ValueError, KeyError, TypeError) as error:
            return {"feasible": False, "reason": f"invalid action: {error}", "ticks": None}
        if action is None or not action.is_skill:
            return None
        try:
            result = self._client.request(action.command, {**action.args, "dry_run": True})
        except self._command_error as error:
            return {"feasible": False, "reason": str(error).split(": ", 1)[-1], "ticks": None}
        return {"feasible": bool(result.get("feasible")), "reason": result.get("reason"),
                "ticks": (result.get("estimate") or {}).get("ticks")}

    def view(self, agent: str) -> str:
        """The agent's true current view, rendered exactly as in engine runs."""
        return ObservationRenderer().render(self._client.request("observe", {"name": agent, "include_terrain": False}))

    def execute(self, agent: str, call: ToolCall) -> str:
        """Execute one act now and return the engine's immediate verdict."""
        try:
            action = self._translator.translate(agent, call)
        except (ValueError, KeyError, TypeError) as error:
            return f"invalid action: {error}"
        if action is None:
            return "carried on"
        try:
            result = self._client.request(action.command, action.args)
        except self._command_error as error:
            return f"failed: {str(error).split(': ', 1)[-1]}"
        if action.is_skill:
            return "started" if result.get("accepted") else f"not possible: {result.get('reason')}"
        return "done"


class NarratedRuntime:
    """The agent loop under GM resolution, on the same decision grid as the engine runtime.

    An agent decides when its last narrated act has finished (by the GM's duration), or when the GM says it heard
    someone speak. As in the engine runtime, `min_gap` is thinking time after an act that starts no skill (instant or
    rejected), and idle agents sleep through the nightly `rest` window unless spoken to. Agent decisions at one poll
    run in parallel. Their acts are then resolved by the GM one at a time,
    in agent-name order, so each ruling sees the previous ones.
    """

    def __init__(self, agents: list[Agent], model: LLMClient, gm: GameMaster, out_dir: Path, *,
                 shadow: ShadowEngine | None = None, poll_interval: int = 40, max_model_calls: int | None = None,
                 min_gap: int = 0, rest: tuple[int, int] | None = None, resync: bool = False):
        """Configure a narrated run. `max_model_calls` bounds agent and GM calls together. With `resync` (which needs a
        shadow), each deciding agent's view is reset to the engine's true view before it decides, so every GM ruling
        starts from the true state of the acting agent: a per-ruling error rate, free of compounding."""
        self._agents = sorted(agents, key=lambda agent: agent.name)
        self._model = model
        self._gm = gm
        daily_hunger = any(agent.daily_hunger for agent in self._agents)
        extras = next((agent.extras for agent in self._agents if agent.extras), ())
        if daily_hunger or extras:
            self._gm.configure_tools(daily_hunger=daily_hunger, extras=extras)
        self._shadow = shadow
        self._poll_interval = poll_interval
        self._max_model_calls = max_model_calls
        self._min_gap = min_gap
        self._rest = rest
        self._resync = resync
        self._translator = ToolTranslator()
        self._out_dir = Path(out_dir)
        self.log = DecisionLog(self._out_dir / "decisions.jsonl")
        self.acts: list[dict] = []
        self.stats = RunStats()
        self._busy_until = {agent.name: 0 for agent in self._agents}
        self._ready_at = {agent.name: 0 for agent in self._agents}
        self._interrupted = {agent.name: False for agent in self._agents}
        self._open_act: dict[str, dict] = {}

    @property
    def agents(self) -> list[Agent]:
        """The run's agents, in decision order (by name)."""
        return list(self._agents)

    def run(self, start: int, until: int) -> RunStats:
        """Run decision polls from `start` to `until` (game times on the poll grid), then write the act log."""
        try:
            now = start
            while now < until:
                if now % self._poll_interval == 0:
                    self._poll(now)
                now += self._poll_interval - now % self._poll_interval
            if self._shadow is not None:
                self._shadow.advance(until)
                for agent, record in self._open_act.items():
                    record["engine_final"] = self._shadow.skill_state(agent)
        finally:
            self._write_outputs()
        return self.stats

    def _poll(self, now: int) -> None:
        """Let due agents decide, then resolve their acts through the GM (and the shadow, if any)."""
        due = [agent for agent in self._agents if self._interrupted[agent.name] or (
            now >= max(self._busy_until[agent.name], self._ready_at[agent.name]) and not resting(now, self._rest))]
        if not due:
            return
        if self._max_model_calls is not None and self._calls() + 2 * len(due) > self._max_model_calls:
            raise BudgetExceeded(f"Model-call budget {self._max_model_calls} would be exceeded at tick {now}")
        if self._shadow is not None:
            self._shadow.advance(now)
            if self._resync:
                for agent in due:
                    self._gm.sync_view(agent.name, self._shadow.view(agent.name))
        prompts = []
        for agent in due:  # prompts are built exactly as in engine runs (epochs, updates, history), from the GM's view
            interrupted = self._interrupted[agent.name] and now < self._busy_until[agent.name]
            user = agent.next_prompt_for_view(self._gm.view(agent.name), interrupted)
            prompts.append((agent, interrupted, agent.system_prompt(), user, agent.history()))
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            responses = list(pool.map(lambda item: self._ask(item[2], item[3], item[4], item[0].tools), prompts))
        for (agent, interrupted, system, user, history), response in zip(prompts, responses):
            call = response.tool_call
            self.stats.model_calls += 1
            self.stats.decisions += 1
            self.stats.input_tokens += response.input_tokens
            self.stats.output_tokens += response.output_tokens
            self.stats.cache_read_tokens += response.cache_read_tokens
            outcome = self._resolve(now, agent, call, interrupted)
            started_skill = outcome.startswith("done") and self._is_skill(agent.name, call)
            if self._min_gap and outcome != "carried on" and not started_skill:
                self._ready_at[agent.name] = now + self._min_gap
            agent.finish_turn(user, call, outcome)
            self.log.append(Decision(now, agent.name,
                                     prompt_digest(self._model.model, system, user, agent.tools, history),
                                     interrupted, "model", call.name if call else None, call.args if call else None,
                                     response.model, response.input_tokens, response.output_tokens, user, outcome,
                                     cached_tokens=response.cache_read_tokens))

    def _ask(self, system: str, user: str, history, tools: list[dict] | None = None) -> LLMResponse:
        """One agent call. A call that still fails after the client's retries becomes a decision with no action
        (recorded as a model error), as in engine runs, so one bad request cannot end a long narrated run."""
        try:
            return self._model.complete(system, user, TOOL_DEFINITIONS if tools is None else tools, history=history)
        except UsageLimitReached as limit:
            raise BudgetExceeded(f"Provider usage limit reached: {limit}") from limit
        except RuntimeError as failure:
            self.stats.model_errors += 1
            return LLMResponse(None, text=f"model error: {failure}"[:300], model=self._model.model)

    def _resolve(self, now: int, agent: Agent, call: ToolCall | None, interrupted: bool) -> str:
        """Have the GM rule on one act, update memories and timing, check it in the shadow, and return the outcome."""
        name, memory = agent.name, agent.memory
        self._interrupted[name] = False
        if call is None:
            memory.record_action(now, ToolCall("(none)", {}), "no action chosen")
            return "no action chosen"
        if call.name == "keep_going" and interrupted:
            memory.record_action(now, call, "carried on")
            return "carried on"
        record = {"gameTime": now, "agent": name, "tool": call.name,
                  "args": {k: v for k, v in call.args.items() if k != "note"}}
        if self._shadow is not None and name in self._open_act:
            self._open_act[name]["engine_final"] = self._shadow.skill_state(name)
        resolution = self._gm.resolve(now, name, call)
        record["gm"] = {key: value for key, value in asdict(resolution).items() if key != "error" or value}
        if self._shadow is not None:
            record["engine_check"] = self._shadow.check(name, call)
        if resolution.accepted:
            outcome = "done" if resolution.duration_ticks == 0 else f"done (took {resolution.duration_ticks} ticks)"
            self._busy_until[name] = now + resolution.duration_ticks
            if self._shadow is not None:
                record["engine"] = self._shadow.execute(name, call)
                if record["engine"] == "started":
                    self._open_act[name] = record
                else:
                    self._open_act.pop(name, None)
        else:
            outcome = f"not possible: {resolution.reason}"
        memory.record_action(now, call, outcome)
        for person, text in resolution.observations.items():
            other = next(a for a in self._agents if a.name == person)
            if call.name == "speak" and person != name:
                other.memory.record_heard(now, name, text)
                self._interrupted[person] = True
            elif person != name:
                other.memory.record_noticed(now, text)
        self.acts.append(record)
        return outcome

    def _is_skill(self, agent: str, call: ToolCall) -> bool:
        """Whether the engine would run this act as a skill (which ends on its own) rather than at once."""
        try:
            action = self._translator.translate(agent, call)
        except (ValueError, KeyError, TypeError):
            return False
        return action is not None and action.is_skill

    def _calls(self) -> int:
        """Agent plus GM model calls so far."""
        return self.stats.model_calls + self._gm.calls

    def _write_outputs(self) -> None:
        """Write the per-act log (GM rulings and engine verdicts) and run stats."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with (self._out_dir / "acts.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.acts:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stats = {**asdict(self.stats), "gm_calls": self._gm.calls, "gm_input_tokens": self._gm.input_tokens,
                 "gm_cached_tokens": self._gm.cached_tokens, "gm_output_tokens": self._gm.output_tokens}
        (self._out_dir / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")


def narration_error(acts: list[dict]) -> dict:
    """Score the GM's rulings against the engine shadow.

    An accepted act is **infeasible** if the engine rejected it or its skill failed, and **compressed** if the GM
    declared it finished while the engine skill was still running. Once an agent has had either, its later acts start
    from a state the engine no longer shares, so infeasible acts are also split into **first** (the agent still matched
    the engine) and **downstream**. A **false rejection** is an act the GM rejected that the engine's dry run found
    feasible. **Duration ratio** is the median of GM ticks over the engine's estimate for accepted acts that run over
    time."""
    gm_errors = sum(bool(act.get("gm", {}).get("error")) for act in acts)
    acts = [act for act in acts if not act.get("gm", {}).get("error")]  # no ruling was made: neither kind of error
    accepted = [act for act in acts if act.get("gm", {}).get("accepted") and "engine" in act]
    rejected = [act for act in acts if "gm" in act and not act["gm"]["accepted"]]
    infeasible: dict[str, int] = {}
    compressed: dict[str, int] = {}
    false_rejections: dict[str, int] = {}
    first_divergence: dict[str, int] = {}
    first = false_first = 0
    for act in acts:  # in game-time order
        matched = act["agent"] not in first_divergence
        if "gm" in act and not act["gm"]["accepted"]:
            if (act.get("engine_check") or {}).get("feasible"):
                false_rejections[act["tool"]] = false_rejections.get(act["tool"], 0) + 1
                false_first += matched
            continue
        if "engine" not in act:
            continue
        verdict, final = act["engine"], act.get("engine_final", "")
        failed = verdict not in {"started", "done", "carried on"} or final.startswith("failed")
        if failed:
            reason = verdict if verdict not in {"started", "done", "carried on"} else f"skill {final}"
            infeasible[reason] = infeasible.get(reason, 0) + 1
            first += matched
        elif final == "running":
            compressed[act["tool"]] = compressed.get(act["tool"], 0) + 1
        if (failed or final == "running") and matched:
            first_divergence[act["agent"]] = act["gameTime"]
    ratios = sorted(act["gm"]["duration_ticks"] / act["engine_check"]["ticks"] for act in accepted
                    if (act.get("engine_check") or {}).get("ticks"))
    by_count = lambda counts: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))  # noqa: E731
    count = sum(infeasible.values())
    return {"accepted": len(accepted), "infeasible": count,
            "infeasible_rate": count / len(accepted) if accepted else 0.0,
            "infeasible_first": first, "infeasible_downstream": count - first,
            "infeasible_by_reason": by_count(infeasible), "compressed": sum(compressed.values()),
            "compressed_by_tool": by_count(compressed), "first_divergence": first_divergence,
            "rejected": len(rejected), "false_rejections": sum(false_rejections.values()),
            "false_rejections_first": false_first, "false_rejections_by_tool": by_count(false_rejections),
            "duration_ratio_median": ratios[len(ratios) // 2] if ratios else None, "gm_errors": gm_errors}


def narrated_outcomes(acts: list[dict], manifest: dict, end: int | None = None) -> dict:
    """What the GM's accepted acts made true in the narrated story, and when (game time plus the GM's duration).

    The bridge is complete once every `bridge_span_*` section was accepted, the port once `port_repair` was; the fair
    is ready when a lantern was placed on one of the bell's four horizontal neighbours (the engine's `block_beside`
    rule) and opens then, or at the start of its `from_day` if later, unless that is after `end`; `builder_paid`
    totals the emeralds handed to the builder."""
    predicates = manifest["success_predicates"]
    ready_query = predicates.get("fair_ready", predicates["fair_open"])
    fair = next(query["block_beside"] for query in ready_query["all"] if "block_beside" in query)
    opens_from = predicates["fair_open"].get("from_day", 0) * 24000
    x, y, z = fair["pos"]
    beside = {(x + 1, y, z), (x - 1, y, z), (x, y, z + 1), (x, y, z - 1)}
    lantern = fair["block"].removeprefix("minecraft:")
    spans = sorted(name for name in manifest["schematics"] if name.startswith("bridge_span_"))
    built: dict[str, int] = {}
    fair_ready, paid = None, 0
    for act in sorted((act for act in acts if act.get("gm", {}).get("accepted")), key=lambda act: act["gameTime"]):
        done, args = act["gameTime"] + act["gm"].get("duration_ticks", 0), act.get("args", {})
        if act["tool"] == "build_section":
            built.setdefault(args.get("schematic"), done)
        elif (act["tool"] == "place" and fair_ready is None and str(args.get("item", "")).removeprefix("minecraft:")
              == lantern and (args.get("x"), args.get("y"), args.get("z")) in beside):
            fair_ready = done
        elif act["tool"] == "give" and args.get("target") == "builder" and "emerald" in str(args.get("item", "")):
            paid += int(args.get("count", 0))
    fair_open = None if fair_ready is None else max(fair_ready, opens_from)
    if end is not None and fair_open is not None and fair_open > end:
        fair_open = None
    return {"bridge_complete": max(built[name] for name in spans) if all(name in built for name in spans) else None,
            "port_repaired": built.get("port_repair"), "fair_ready": fair_ready, "fair_open": fair_open,
            "builder_paid": paid}


def outcome_divergence(narrated: dict, shadow: dict) -> dict:
    """For the outcomes both sides judge (bridge complete, fair ready, fair open): whether the narrated story and the
    engine shadow disagree, and the narrated time."""
    return {name: {"narrated": narrated[name] is not None, "narrated_tick": narrated[name],
                   "engine": bool(shadow.get(name)), "diverged": (narrated[name] is not None) != bool(shadow.get(name))}
            for name in ("bridge_complete", "fair_ready", "fair_open") if name in narrated}
