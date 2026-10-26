"""`mlx_lm.server` with a prompt-cache policy for agent conversations. Run it with the MLX environment's Python.

Agents send append-only conversations (see `orchestrator.agent`): each prompt extends the agent's previous one. On a
hybrid model such as Qwen3.6-35B-A3B, whose linear-attention state cannot be rolled back, a cached prefix is reusable
only if it ends exactly where a stored checkpoint ends. The stock server stores, per request, a checkpoint after the
system prompt, one after the conversation (before the assistant's think tail), and one after the generated reply. For
these conversations:

- The conversation checkpoint is what the agent's next prompt reuses, and it supersedes the agent's previous one,
  which is its prefix. `AgentPromptCache` drops the superseded checkpoint, so each agent holds one.
- The reply checkpoint is never a prefix of a later prompt, because the chat template renders earlier assistant turns
  without the think tail they were generated after. `AgentPromptCache` does not store it.

Reuse stays exact: only identical token prefixes are served from the cache.

Sampling is made random again. MLX keeps its random state per thread, and mlx-lm's compiled categorical sampler
(`mx.compile` with the random state as input and output) does not advance that state outside the main thread: in the
server's generation thread every draw was the same, so the "random" token was fixed at every step of every request and
only the temperature changed replies (reproduced in isolation: [10, 10, 10, ...] in a worker thread against [10, 44, 0,
...] in the main thread). The sampler is replaced by the same draw uncompiled, and each unseeded request reseeds from
the OS; a seeded request still reseeds with its own seed.

MLX's buffer cache (freed buffers kept for reuse) is capped at `BUFFER_CACHE_LIMIT`. Uncapped, it grows toward the
server's wired limit and pushes the Minecraft servers on the same machine into swap.

Usage: `<mlx python> scripts/mlx_agent_server.py --model <path> [mlx_lm.server options]` (see `local_model.py`).
"""

from __future__ import annotations

import secrets

import mlx.core as mx
import mlx_lm.sample_utils as sample_utils
import mlx_lm.server as server
from mlx_lm.models.cache import LRUPromptCache

BUFFER_CACHE_LIMIT = 1 << 30


class AgentPromptCache(LRUPromptCache):
    """An LRU prompt cache that keeps one conversation checkpoint per agent and skips reply checkpoints."""

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type: str = "assistant"):
        """Store a checkpoint, unless it is a reply checkpoint; a conversation checkpoint replaces its prefixes."""
        if cache_type == "assistant":
            return
        if cache_type == "user":
            for old_model, old in list(self._lru._lrus["user"]):
                if old_model == model and len(old) < len(tokens) and tokens[:len(old)] == old:
                    self._drop(old_model, old)
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)

    def _drop(self, model, tokens) -> None:
        """Remove one stored checkpoint and its byte count."""
        entry = self._trie.pop(model, tokens)
        self._lru.remove(model, tokens)
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes


def categorical_sampling(logits, temp):
    """mlx-lm's categorical draw without `mx.compile`, so the thread's random state advances with every draw."""
    return mx.random.categorical(logits * (1 / temp))


def fresh_randomness(tokenize):
    """Wrap the server's per-request tokenization (it runs in the generation thread) to reseed unseeded requests."""
    def tokenize_with_fresh_seed(self, tokenizer, request, args):
        if args.seed is None:
            mx.random.seed(secrets.randbits(32))
        return tokenize(self, tokenizer, request, args)
    return tokenize_with_fresh_seed


if __name__ == "__main__":
    mx.set_cache_limit(BUFFER_CACHE_LIMIT)
    server.LRUPromptCache = AgentPromptCache
    sample_utils.categorical_sampling = categorical_sampling  # make_sampler's samplers look it up at call time
    server.ResponseGenerator._tokenize = fresh_randomness(server.ResponseGenerator._tokenize)
    server.main()
