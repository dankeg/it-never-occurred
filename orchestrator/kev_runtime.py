"""Kev-4B's decision computation, rebuilt on MLX from its published weights (DELEGATE_SPEC §3–4), without Kev's code.

`KevModel` loads `Qwen/Qwen3.5-4B-Base` from the local Hugging Face cache with mlx-lm, folds Kev-4B's rank-16 LoRA
adapter into the base weights (`W + (alpha / r) · B · A`, computed in fp32 and rounded once to the base's bf16, the
documented fold), and reads Kev's pointer head from `head.pt` with a restricted unpickler that accepts only the three
names a tensor dictionary needs (no torch, no code from the Kev repository). One question row is

    [<state>] state tokens [<q>] question tokens ([<opt>] option tokens [</opt>])* [<decide>]

with Kev's five markers being existing Qwen special tokens, user text escaped (`<|name|>` becomes `<¦name¦>`), and the
question's positions continuing from the state (the state is prefilled into a prompt cache, then the question row runs
on a copy of it). The head scores option `c` as `(W_k h_c + b_k) · (W_q h_decide + b_q) / √256` from the final hidden
states (after the last RMSNorm) at `</opt>_c` and `<decide>`, in fp32, and the probabilities are `softmax(s / T)` with
Kev-4B's temperature unless another is given.

Needs `mlx`, `mlx-lm` and `numpy` (the local-model environment); it is imported only by evaluation scripts run there.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import re
import zipfile

MARKERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
_ESCAPE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
PREFILL_CHUNK = 1024


class _HeadUnpickler(pickle.Unpickler):
    """Unpickles a torch `head.pt` dictionary of float32 tensors into numpy arrays, refusing any other global."""

    def __init__(self, archive: zipfile.ZipFile, prefix: str):
        """Read pickled data and tensor storages from the checkpoint archive."""
        super().__init__(archive.open(f"{prefix}/data.pkl"))
        self._archive, self._prefix = archive, prefix

    def find_class(self, module: str, name: str):
        """Only an ordered dictionary, torch's tensor rebuild and its float storage type are allowed."""
        if (module, name) == ("collections", "OrderedDict"):
            from collections import OrderedDict
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self._rebuild
        if (module, name) == ("torch", "FloatStorage"):
            return "float32"
        raise pickle.UnpicklingError(f"head.pt may not reference {module}.{name}")

    def persistent_load(self, pid):
        """A storage reference (`('storage', type, key, location, numel)`) becomes its raw float32 values."""
        import numpy as np
        kind, dtype, key, _location, count = pid
        if kind != "storage" or dtype != "float32":
            raise pickle.UnpicklingError(f"Unsupported storage {pid!r}")
        values = np.frombuffer(self._archive.read(f"{self._prefix}/data/{key}"), dtype="<f4")
        if values.size != count:
            raise pickle.UnpicklingError(f"Storage {key} has {values.size} values, expected {count}")
        return values

    @staticmethod
    def _rebuild(storage, offset, size, stride, *_rest):
        """A contiguous tensor view of a storage as a numpy array."""
        import numpy as np
        expected = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
        if tuple(stride) != expected:
            raise pickle.UnpicklingError("Only contiguous tensors are supported")
        return storage[offset:offset + int(np.prod(size))].reshape(size).copy()


def read_head(path: Path) -> dict:
    """The pointer head's weights (`q.weight`, `q.bias`, `k.weight`, `k.bias`) from a Kev `head.pt`."""
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        meta = _HeadUnpickler(archive, prefix).load()
    return dict(meta["head"]) if "head" in meta else dict(meta)


def escape(text: str) -> str:
    """User text that can never produce one of Kev's markers."""
    return _ESCAPE.sub(r"<¦\1¦>", text)


class KevModel:
    """Kev-4B on MLX: the base with the adapter folded in, Kev's question rows, and the pointer head."""

    def __init__(self, base: Path, adapter: Path, temperature: float = 2.406050072164233):
        """Load the base snapshot directory and fold in the adapter directory (with `head.pt`)."""
        import mlx.core as mx
        from mlx_lm.utils import load_model, load_tokenizer
        import numpy as np
        self._mx, self._np = mx, np
        self.lm = load_model(Path(base))[0]
        self.tokenizer = load_tokenizer(Path(base))._tokenizer
        self.text = self.lm.language_model.model
        self.folded = self._fold(Path(adapter))
        head = read_head(Path(adapter) / "head.pt")
        self._wq, self._bq = head["q.weight"], head["q.bias"]
        self._wk, self._bk = head["k.weight"], head["k.bias"]
        self.temperature = temperature
        self.markers = [self.tokenizer.convert_tokens_to_ids(token) for token in MARKERS]
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else 0

    def _fold(self, adapter: Path) -> int:
        """Add `(alpha / r) · B · A` to every adapted weight, in fp32 on the CPU, rounded once to the weight's dtype.
        PEFT names the wrapped text model `base_model.model.`; mlx-lm nests it as `language_model.model.`."""
        mx = self._mx
        from mlx.utils import tree_flatten
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("use_rslora") or config.get("trainable_token_indices"):
            raise ValueError("Only plain LoRA adapters are supported")
        scale = config["lora_alpha"] / config["r"]
        weights = mx.load(str(adapter / "adapter_model.safetensors"))
        params = dict(tree_flatten(self.lm.parameters()))
        merged = {}
        with mx.stream(mx.cpu):
            for name, a in weights.items():
                if not name.endswith(".lora_A.weight"):
                    continue
                stem = name[: -len(".lora_A.weight")]
                target = stem.replace("base_model.model.", "language_model.model.", 1) + ".weight"
                if target not in params:
                    raise ValueError(f"Adapter tensor {stem} has no base weight {target}")
                base = params[target]
                delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * scale
                merged[target] = (base.astype(mx.float32) + delta).astype(base.dtype)
                mx.eval(merged[target])
        self.lm.load_weights(list(merged.items()), strict=False)
        mx.eval(self.lm.parameters())
        return len(merged)

    def tokens(self, text: str) -> list[int]:
        """Token ids of escaped user text, without special tokens."""
        return self.tokenizer(escape(text), add_special_tokens=False)["input_ids"]

    def branch(self, instruction: str, options: list[str]) -> tuple[list[int], int, list[int]]:
        """A question's tokens after the state, the offset of `<decide>`, and the offsets of each `</opt>`."""
        _, question, opening, closing, decide = self.markers
        ids, ends = [question] + self.tokens(instruction), []
        for option in options:
            ids += [opening] + self.tokens(option) + [closing]
            ends.append(len(ids) - 1)
        ids.append(decide)
        return ids, len(ids) - 1, ends

    def state_ids(self, state: str) -> list[int]:
        """The state's tokens, beginning with `<state>`."""
        return [self.markers[0]] + self.tokens(state)

    def prefill(self, ids: list[int]) -> list:
        """The state run into a fresh prompt cache, `PREFILL_CHUNK` tokens per pass."""
        mx = self._mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.lm)
        for start in range(0, len(ids), PREFILL_CHUNK):
            self.text(mx.array([ids[start:start + PREFILL_CHUNK]], dtype=mx.int32), cache=cache)
            mx.eval([layer.state for layer in cache])
        return cache

    def scores(self, cache: list, rows: list[tuple[list[int], int, list[int]]]) -> list:
        """Raw pointer scores of each question row, run as one batch on copies of the state cache (unchanged)."""
        mx, np = self._mx, self._np
        length = max(len(ids) for ids, _, _ in rows)
        batch = [type(layer).merge([layer] * len(rows)) for layer in cache]
        padded = mx.array([ids + [self.pad_id] * (length - len(ids)) for ids, _, _ in rows], dtype=mx.int32)
        hidden = self.text(padded, cache=batch).astype(mx.float32)
        mx.eval(hidden)
        hidden = np.array(hidden)
        result = []
        for row, (_, decide, ends) in zip(hidden, rows):
            query = self._wq @ row[decide] + self._bq
            keys = row[ends] @ self._wk.T + self._bk
            result.append(keys @ query / math.sqrt(self._wq.shape[0]))
        return result

    def probabilities(self, scores, temperature: float | None = None) -> list[float]:
        """`softmax(scores / T)` (the model's temperature unless another is given)."""
        np = self._np
        z = np.asarray(scores, dtype=np.float64) / (self.temperature if temperature is None else temperature)
        z = np.exp(z - z.max())
        return (z / z.sum()).tolist()

    def ask(self, state: str, questions: list[tuple[str, list[str]]], temperature: float | None = None) -> list:
        """Probabilities over each question's options for one state (the state is prefilled once)."""
        cache = self.prefill(self.state_ids(state))
        rows = [self.branch(instruction, options) for instruction, options in questions]
        return [self.probabilities(score, temperature) for score in self.scores(cache, rows)]
