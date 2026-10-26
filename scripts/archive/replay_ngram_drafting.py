"""Offline replay: how much would model-free n-gram drafting speed up decoding of our agents' replies?

Each logged decision's reply is re-rendered as the chat template renders a tool call and tokenized. Decoding is
replayed token by token with lossless draft-and-verify: at each step a drafter proposes up to K tokens from
(a) the decision's own prompt and reply so far (prompt lookup), (b) earlier replies in the run (suffix store,
SuffixDecoding-style), or both. Draft tokens are accepted while they equal the logged token; the step then emits the
accepted tokens plus one. For sampled replies this equals the acceptance probability of a fixed draft in expectation.

Cost per step: one forward over p positions, priced with optiq's measurement on a Qwen3.6-35B-A3B derivative on an
M3 Max (p=2: 1.11, p=5: 1.60 single steps; +4% for capturing hybrid state). Speedup = tokens / total step cost.

Usage: `<mlx python> scripts/archive/replay_ngram_drafting.py runs/<run> [runs/<run> ...]` (tokenizer only; no
model load).

Archived (C5, 2026-10-18): a one-off that estimated n-gram drafting speed-ups from logged replies (systems track). Kept
for reference; run it with `PYTHONPATH=scripts`.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from mlx_lm.tokenizer_utils import load as load_tokenizer

ROOT = Path.home() / "MinecraftSim"
sys.path.insert(0, str(ROOT))
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS  # noqa: E402

K = 5
MIN_MATCH = 2
MAX_MATCH = 8


def step_cost(positions: int) -> float:
    """Cost of one verify pass over `positions` tokens, in single-step units (piecewise linear through optiq's data)."""
    if positions <= 1:
        return 1.0
    base = 1.11 if positions == 2 else 1.11 + (positions - 2) * (1.60 - 1.11) / 3
    return base * 1.04


class SuffixStore:
    """Continuations of every n-gram (n = MIN_MATCH..MAX_MATCH) seen in earlier replies, with counts."""

    def __init__(self):
        self.next = defaultdict(lambda: defaultdict(int))

    def add(self, tokens):
        for n in range(MIN_MATCH, MAX_MATCH + 1):
            for i in range(len(tokens) - n):
                self.next[tuple(tokens[i:i + n])][tokens[i + n]] += 1

    def draft(self, history):
        """Greedy chain of most frequent continuations from the longest matching suffix."""
        out = []
        context = list(history)
        for _ in range(K):
            for n in range(MAX_MATCH, MIN_MATCH - 1, -1):
                key = tuple(context[-n:])
                if len(key) == n and key in self.next:
                    token = max(self.next[key].items(), key=lambda kv: kv[1])[0]
                    out.append(token)
                    context.append(token)
                    break
            else:
                break
        return out


class PromptIndex:
    """Latest position after each n-gram (n = MIN_MATCH..MAX_MATCH) in the prompt plus the reply so far."""

    def __init__(self, context):
        self.sequence = []
        self.after = {}
        for token in context:
            self.push(token)

    def push(self, token):
        self.sequence.append(token)
        end = len(self.sequence)
        for n in range(MIN_MATCH, MAX_MATCH + 1):
            if end - 1 - n >= 0:
                # the n-gram ending just before this token now has a continuation starting at end - 1
                self.after[tuple(self.sequence[end - 1 - n:end - 1])] = end - 1

    def draft(self):
        """Continuation after the most recent earlier occurrence of the longest matching suffix."""
        for n in range(MAX_MATCH, MIN_MATCH - 1, -1):
            start = self.after.get(tuple(self.sequence[-n:]))
            if start is not None:
                return self.sequence[start:start + K]
        return []


def replay(reply, context, store, mode):
    """Decode `reply` with drafts from `mode`; return total step cost, accepted and drafted token counts."""
    index = PromptIndex(context) if mode in ("prompt", "both") else None
    i, cost, accepted_total, drafted_total = 0, 0.0, 0, 0
    while i < len(reply):
        draft = index.draft() if index is not None else []
        if not draft and mode in ("suffix", "both"):
            draft = store.draft(context[-MAX_MATCH:] + reply[:i])
        draft = draft[:K]
        accepted = 0
        while accepted < len(draft) and i + accepted < len(reply) and draft[accepted] == reply[i + accepted]:
            accepted += 1
        cost += step_cost(1 + len(draft))
        drafted_total += len(draft)
        accepted_total += accepted
        for token in reply[i:i + accepted + 1]:
            if index is not None:
                index.push(token)
        i += accepted + 1
    return cost, accepted_total, drafted_total


def main():
    tokenizer = load_tokenizer(Path.home() / "models/Qwen3.6-35B-A3B-4bit")
    manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
    systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
    tools = [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS]
    results = {}
    for run in sys.argv[1:]:
        rows = [json.loads(line) for line in open(ROOT / run / "orchestrator/decisions.jsonl") if line.strip()]
        rows = [r for r in rows if r["source"] == "model" and r["tool"]]
        rows.sort(key=lambda r: (r["game_time"], r["agent"]))
        stores = {mode: SuffixStore() for mode in ("suffix", "both")}
        totals = {mode: [0, 0.0, 0, 0] for mode in ("none", "prompt", "suffix", "both")}
        for row in rows:
            messages = [{"role": "system", "content": systems[row["agent"]]},
                        {"role": "user", "content": row["user_prompt"]}]
            context = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                    enable_thinking=False)
            rendered = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": row["tool"], "arguments": row["args"]}}]}],
                tools=tools, tokenize=False, enable_thinking=False)
            text = rendered[rendered.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n"):]
            text = text[:text.find("<|im_end|>") + len("<|im_end|>")]
            reply = tokenizer.encode(text, add_special_tokens=False)
            for mode in totals:
                store = stores.get(mode)
                cost, accepted, drafted = (float(len(reply)), 0, 0) if mode == "none" else \
                    replay(reply, context, store or SuffixStore(), mode)
                totals[mode][0] += len(reply)
                totals[mode][1] += cost
                totals[mode][2] += accepted
                totals[mode][3] += drafted
            for store in stores.values():
                store.add(reply)
        results[run] = {mode: {"tokens": t[0], "speedup": round(t[0] / t[1], 3),
                               "acceptedPerDrafted": round(t[2] / t[3], 3) if t[3] else None,
                               "acceptedShareOfTokens": round(t[2] / t[0], 3)}
                        for mode, t in totals.items()}
        results[run]["decisions"] = len(rows)
        results[run]["meanReplyTokens"] = round(totals["none"][0] / len(rows), 1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
