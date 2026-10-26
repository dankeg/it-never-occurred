"""The delegate's input in Kev's format (DELEGATE_SPEC §3): the state document as one text, and one question row.

Kev reads a state as plain text, while the state document `S(d)` (a dataset record's `state`) is a conversation: the
system prompt, the epoch's earlier turns (user message, chosen call, its result) and the current user message. The
flattening below is fixed once, so every model sees the same bytes:

    {system}

    {turn 1 user message}

    Your action: {tool} {sorted-key JSON arguments}
    Result: {result}

    ... (each earlier turn of the epoch)

    {current user message}

It is prefix-stable within an epoch: the text for a decision is a prefix of the text for the agent's next decision in
the same epoch (that decision appends the action, its result and the next user message), so a cached state is only
extended, never recomputed. It uses no chat-template tokens (Kev escapes `<|...|>` in user text, and Kev-4B reads plain
documents on a base model), and it writes past calls in the same sorted-key JSON as the option labels.
"""

from __future__ import annotations

QUESTION = "Which act does {agent} take next?"


def state_text(state: dict) -> str:
    """Flatten a record's state document (`system`, `history`, `user`) into the delegate's state text."""
    parts = [state["system"]]
    for turn in state["history"]:
        action = "(none)" if turn["tool"] is None else f"{turn['tool']} {turn['arguments']}"
        parts.append(f"{turn['user']}\n\nYour action: {action}\nResult: {turn['result']}")
    parts.append(state["user"])
    return "\n\n".join(parts)


def kev_record(record: dict, labels: list[str] | None = None) -> dict:
    """Kev's internal record for one dataset record: the state text and one question whose options are the candidate
    labels (or `labels`, e.g. renamed ones, in the same order). The question's label is the index of the first
    candidate matching the frontier's choice (0 when none matches; such records are not scored)."""
    options = labels if labels is not None else [candidate["label"] for candidate in record["candidates"]]
    return {"state": state_text(record["state"]),
            "questions": [{"instr": QUESTION.format(agent=record["agent"]), "options": options,
                           "label": record["frontier_index"] or 0}]}
