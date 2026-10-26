"""The decision log: every agent decision with the digest of the exact prompt that produced it.

A run appends each decision to `decisions.jsonl` in its orchestrator directory. The log has three uses:

- **Prefix replay (E3).** A run given its own earlier log as `source` reuses every decision without calling a model.
  The world is deterministic, so it reproduces the same prompts and the same world-hash trace.
- **Gated branches (L1).** A counterfactual branch gets the factual run's log as `source`. At each decision point, if
  the agent's prompt digest equals the factual one, the agent would have seen exactly the same input, and the factual
  decision is reused. Only agents whose prompts diverged call the model. This couples the branch to the factual run
  (like common random numbers) and makes branches cheap.
- **Analysis.** Each entry keeps the user prompt, so narration and investigation studies can see what an agent knew.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .llm import ToolCall, Turn, arguments_json


def prompt_digest(model: str, system: str, user: str, tools: list[dict], history: Sequence[Turn] = ()) -> str:
    """Return the SHA-256 of everything that determines a model's input: model id, system, earlier turns, user prompt
    and tools. A prompt with no earlier turns hashes as it did before conversations existed, so old logs still
    replay."""
    content = {"model": model, "system": system, "user": user, "tools": tools}
    if history:
        content["history"] = [[turn.user, turn.call.name if turn.call else None,
                               arguments_json(turn.call) if turn.call else None, turn.result] for turn in history]
    payload = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Decision:
    """One agent decision at one game time.

    `source` is `model` when a model produced it, or `replay` when it was reused from a source log.
    `tool` and `args` are null when the model returned no usable tool call. `outcome` is the action's immediate result
    in the world (e.g. `started`, `done`, `not possible: no_path`); how a started skill ends is in the agent's memory.
    `input_tokens` counts uncached input and `cached_tokens` the input served from a prompt cache; `seconds` is the
    model call's wall time. All three are 0 for a reused decision.
    """

    game_time: int
    agent: str
    digest: str
    interrupted: bool
    source: str
    tool: str | None
    args: dict | None
    model: str
    input_tokens: int
    output_tokens: int
    user_prompt: str
    outcome: str = ""
    cached_tokens: int = 0
    seconds: float = 0.0

    @property
    def tool_call(self) -> ToolCall | None:
        """The decision as a tool call, or None if the model gave none."""
        return None if self.tool is None else ToolCall(self.tool, dict(self.args or {}))


class DecisionLog:
    """An append-only JSONL decision log, indexed by (agent, game time)."""

    def __init__(self, path: Path):
        """Open (and load, if present) the log at `path`."""
        self.path = Path(path)
        self._index: dict[tuple[str, int], Decision] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        decision = Decision(**json.loads(line))
                        self._index[(decision.agent, decision.game_time)] = decision

    def __len__(self) -> int:
        """Number of decisions in the log."""
        return len(self._index)

    def get(self, agent: str, game_time: int) -> Decision | None:
        """Return the decision `agent` made at `game_time`, if any."""
        return self._index.get((agent, game_time))

    def append(self, decision: Decision) -> None:
        """Record a decision in memory and on disk."""
        key = (decision.agent, decision.game_time)
        if key in self._index:
            raise ValueError(f"Duplicate decision for {key}")
        self._index[key] = decision
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True) + "\n")

    def decisions(self) -> list[Decision]:
        """Return all decisions ordered by game time, then agent name."""
        return sorted(self._index.values(), key=lambda decision: (decision.game_time, decision.agent))
