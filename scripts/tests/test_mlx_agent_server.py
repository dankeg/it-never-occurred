"""Unit tests for the local model server's prompt-cache policy, `AgentPromptCache` (incident M4).

`scripts/mlx_agent_server.py` imports MLX at the top. Where mlx-lm is installed, the tests run against its real
`LRUPromptCache`. Elsewhere the module is imported over minimal stand-in `mlx` and `mlx_lm` modules, whose
`LRUPromptCache` keeps mlx-lm 0.31.3's bookkeeping for caches that cannot be trimmed (the hybrid model's), and the
stand-ins are removed again right after. Cache entries are stand-ins in both cases: layers with a byte size.
"""

from collections import deque
import importlib
import importlib.util
import sys
import types
import unittest


class StandInLRUPromptCache:
    """mlx-lm 0.31.3's `LRUPromptCache` bookkeeping for non-trimmable caches, without MLX: checkpoints by model and
    tokens, an LRU order per checkpoint type, and byte counts (no size limits, no lookups)."""

    class Order:
        """The LRU order of stored checkpoints, one queue per checkpoint type."""

        def __init__(self):
            """Empty queues for the three checkpoint types."""
            self._ordering = ["assistant", "user", "system"]
            self._lrus = {kind: deque() for kind in self._ordering}

        def __len__(self) -> int:
            """The number of stored checkpoints."""
            return sum(len(queue) for queue in self._lrus.values())

        def push(self, model, tokens, cache_type: str) -> None:
            """Append a checkpoint to its type's queue."""
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model, tokens) -> None:
            """Remove a checkpoint from whichever queue holds it."""
            for kind in self._ordering:
                if (model, tokens) in self._lrus[kind]:
                    self._lrus[kind].remove((model, tokens))
                    return

    class Store:
        """Checkpoint entries by model and token sequence (the real class keeps them in a trie)."""

        def __init__(self):
            """No entries."""
            self._entries: dict = {}

        def add(self, model, tokens, entry):
            """Store an entry and return the one it replaced, if any."""
            previous = self._entries.get((model, tuple(tokens)))
            self._entries[(model, tuple(tokens))] = entry
            return previous

        def pop(self, model, tokens):
            """Remove and return an entry."""
            return self._entries.pop((model, tuple(tokens)))

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
        """Empty, with the real class's constructor arguments (limits are not enforced here)."""
        self._trie = self.Store()
        self._lru = self.Order()
        self._n_bytes = 0
        self._n_bytes_by_type = {kind: 0 for kind in self._lru._ordering}

    def __len__(self) -> int:
        """The number of stored checkpoints."""
        return len(self._lru)

    @property
    def nbytes(self) -> int:
        """The bytes held by stored checkpoints."""
        return self._n_bytes

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type: str = "assistant") -> None:
        """Store a checkpoint, replacing one with the same tokens, as the real class does."""
        entry = types.SimpleNamespace(prompt_cache=prompt_cache, nbytes=sum(c.nbytes for c in prompt_cache),
                                      cache_type=cache_type)
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        previous = self._trie.add(model, tokens, entry)
        if previous is not None:
            self._n_bytes -= previous.nbytes
            self._n_bytes_by_type[previous.cache_type] -= previous.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

    def stats_by_type(self) -> dict:
        """Checkpoints and bytes per checkpoint type."""
        return {kind: {"n_sequences": len(self._lru._lrus[kind]), "n_bytes": self._n_bytes_by_type[kind]}
                for kind in self._lru._ordering}


class Layer:
    """A stand-in cache layer: a byte size, and not trimmable (like the hybrid model's linear-attention state)."""

    def __init__(self, nbytes: int):
        """Hold `nbytes`."""
        self.nbytes = nbytes

    def is_trimmable(self) -> bool:
        """Hybrid-model caches cannot be rolled back."""
        return False


def agent_prompt_cache_class():
    """Import `scripts.mlx_agent_server` and return `AgentPromptCache`, over stand-in MLX modules without mlx-lm."""
    if importlib.util.find_spec("mlx_lm") is not None:
        return importlib.import_module("scripts.mlx_agent_server").AgentPromptCache
    names = ["mlx", "mlx.core", "mlx_lm", "mlx_lm.sample_utils", "mlx_lm.server", "mlx_lm.models",
             "mlx_lm.models.cache"]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    modules["mlx_lm.models.cache"].LRUPromptCache = StandInLRUPromptCache
    sys.modules.update(modules)
    try:
        return importlib.import_module("scripts.mlx_agent_server").AgentPromptCache
    finally:
        for name in [*names, "scripts.mlx_agent_server"]:
            sys.modules.pop(name, None)


class AgentPromptCacheTest(unittest.TestCase):
    """Each agent keeps one conversation checkpoint, and reply checkpoints are never stored."""

    @classmethod
    def setUpClass(cls):
        """Load the cache class once."""
        cls.cache_class = agent_prompt_cache_class()

    def cache(self):
        """A fresh, empty cache."""
        return self.cache_class(max_size=10)

    def test_reply_checkpoints_are_not_stored(self):
        """A checkpoint after a generated reply never prefixes a later prompt, so it takes no space."""
        cache = self.cache()
        cache.insert_cache("qwen", [1, 2, 3, 4], [Layer(100)], cache_type="assistant")
        self.assertEqual((len(cache), cache.nbytes), (0, 0))

    def test_conversation_checkpoint_replaces_its_prefix(self):
        """An agent's next conversation checkpoint supersedes its previous one, which is a prefix of it."""
        cache = self.cache()
        cache.insert_cache("qwen", [1, 2, 3], [Layer(100)], cache_type="user")
        cache.insert_cache("qwen", [1, 2, 3, 4, 5], [Layer(150)], cache_type="user")
        self.assertEqual(list(cache._lru._lrus["user"]), [("qwen", [1, 2, 3, 4, 5])])
        self.assertEqual(cache.nbytes, 150)
        self.assertEqual(cache.stats_by_type()["user"], {"n_sequences": 1, "n_bytes": 150})

    def test_other_agents_other_models_and_system_checkpoints_are_kept(self):
        """Only a same-model, strictly shorter conversation prefix is dropped; other checkpoints stay."""
        cache = self.cache()
        cache.insert_cache("qwen", [1], [Layer(10)], cache_type="system")
        cache.insert_cache("qwen", [1, 7, 7], [Layer(20)], cache_type="user")
        cache.insert_cache("other", [1, 2], [Layer(30)], cache_type="user")
        cache.insert_cache("qwen", [1, 2, 3, 4], [Layer(40)], cache_type="user")
        self.assertEqual(list(cache._lru._lrus["user"]), [("qwen", [1, 7, 7]), ("other", [1, 2]),
                                                          ("qwen", [1, 2, 3, 4])])
        self.assertEqual(cache.stats_by_type()["system"], {"n_sequences": 1, "n_bytes": 10})
        self.assertEqual(cache.nbytes, 100)


if __name__ == "__main__":
    unittest.main()
