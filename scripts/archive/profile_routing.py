"""Profile mixture-of-experts routing on logged agent decisions (run with the MLX environment's Python).

Question: is expert routing on our workload concentrated enough for a "hot set" of resident experts (colibri-style
placement), freeing memory for more Minecraft servers? The plan's rule: if about 25% of experts per layer cover >= 95%
of routed tokens on held-out decisions, a hot set is worth building; if routing is spread out, drop the idea.

Method. Decisions are sampled from a logged run and split by game time: hot sets are chosen on the earlier half and
scored on the later half, as a deployed hot set would be. Each agent's system prompt and tools are prefilled once (a
server caches them) and recorded separately; each decision then prefills its user message and decodes its reply
greedily, and every routed token-slot (8 experts per token per layer) is counted per layer. Reported:

- held-out coverage of the top-h experts per layer chosen on the earlier half (h = 16 ... 128 of 256);
- experts needed to cover 95% / 99% of held-out slots per layer (the routing's own concentration);
- distinct experts one decision touches per layer, and how many of them a 25% hot set would miss.

Usage: `<mlx python> scripts/archive/profile_routing.py --run runs/llm_village_sabotage_day1a --decisions 300`
(the model server must be stopped: the model is loaded in this process).

Archived (C5, 2026-10-18): a one-off that profiled mixture-of-experts routing on logged decisions (systems track). Kept
for reference; run it with `PYTHONPATH=scripts`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS  # noqa: E402

DEFAULT_MODEL = Path.home() / "models/Qwen3.6-35B-A3B-4bit"
HOT_SIZES = (16, 32, 48, 64, 96, 128)


class RoutingRecorder:
    """Counts routed token-slots per MoE layer, by tagging every `SwitchGLU` call with its layer."""

    def __init__(self, model, num_experts: int):
        """Index the model's MoE layers and route every `SwitchGLU` call through `record`."""
        self.layers = {id(layer.mlp.switch_mlp): index for index, layer in enumerate(model.layers)
                       if hasattr(layer.mlp, "switch_mlp")}
        self.num_experts = num_experts
        self._pending: list[tuple[int, mx.array]] = []
        original = SwitchGLU.__call__
        recorder = self

        def recording_call(switch, x, indices):
            recorder._pending.append((recorder.layers[id(switch)], indices))
            return original(switch, x, indices)

        SwitchGLU.__call__ = recording_call

    def take(self) -> tuple[np.ndarray, np.ndarray]:
        """Slot counts since the last call, as (prefill, decode) arrays of shape [layers, experts]."""
        prefill = np.zeros((len(self.layers), self.num_experts), dtype=np.int64)
        decode = np.zeros_like(prefill)
        order = sorted(self.layers.values())
        row = {layer: position for position, layer in enumerate(order)}
        for layer, indices in self._pending:
            values = np.array(indices)
            target = decode if values.ndim >= 2 and values.shape[-2] == 1 else prefill
            target[row[layer]] += np.bincount(values.ravel(), minlength=self.num_experts)
        self._pending = []
        return prefill, decode


class RoutingProfile:
    """Runs the logged decisions through the model and summarizes routing concentration."""

    def __init__(self, args):
        """Load the model and the decisions to profile."""
        self._args = args
        mx.set_cache_limit(1 << 30)  # keep MLX's freed-buffer cache small; the Minecraft servers need the memory
        self.model, self.tokenizer = load(str(args.model))
        config = json.loads((args.model / "config.json").read_text())
        text = config.get("text_config", config)
        self.recorder = RoutingRecorder(self.model, text["num_experts"])
        manifest = ScenarioManifest.load(ROOT / f"scenario/village_{args.variant}.json")
        self.systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
        self.tools = [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS]
        rows = [json.loads(line) for line in (args.run / "orchestrator/decisions.jsonl").open() if line.strip()]
        rows = [row for row in rows if row["source"] == "model" and row["agent"] in self.systems]
        rows.sort(key=lambda row: (row["game_time"], row["agent"]))
        middle = rows[len(rows) // 2]["game_time"]
        rng = random.Random(args.seed)
        early = [row for row in rows if row["game_time"] < middle]
        late = [row for row in rows if row["game_time"] >= middle]
        half = args.decisions // 2
        self.splits = {"early": sorted(rng.sample(early, min(half, len(early))), key=lambda r: r["game_time"]),
                       "late": sorted(rng.sample(late, min(half, len(late))), key=lambda r: r["game_time"])}

    def _system_cache(self, agent: str):
        """Prefill an agent's system prompt and tools once; return (tokens, cache) and its slot counts."""
        messages = [{"role": "system", "content": self.systems[agent]}, {"role": "user", "content": ""}]
        full = self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=False,
                                                  enable_thinking=False)
        marker = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.systems[agent]}, {"role": "user", "content": "x"}],
            tools=self.tools, add_generation_prompt=False, enable_thinking=False)
        end = next(i for i, (a, b) in enumerate(zip(full, marker)) if a != b)
        tokens = full[:end]
        cache = make_prompt_cache(self.model)
        self.model(mx.array(tokens)[None], cache=cache)
        mx.eval([c.state for c in cache])
        prefill, _ = self.recorder.take()
        return tokens, cache, prefill

    def run(self) -> dict:
        """Profile every sampled decision and return the summary."""
        sampler = make_sampler(temp=0.0)
        system_slots = None
        systems: dict[str, tuple] = {}
        counts = {}
        per_decision = []
        started = time.monotonic()
        for split, rows in self.splits.items():
            prefill_total = None
            decode_total = None
            for number, row in enumerate(rows):
                if row["agent"] not in systems:
                    tokens, cache, slots = self._system_cache(row["agent"])
                    systems[row["agent"]] = (tokens, cache)
                    system_slots = slots if system_slots is None else system_slots + slots
                tokens, cache = systems[row["agent"]]
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.systems[row["agent"]]},
                     {"role": "user", "content": row["user_prompt"]}],
                    tools=self.tools, add_generation_prompt=True, enable_thinking=False)
                if prompt[:len(tokens)] != tokens:
                    raise RuntimeError(f"system prefix mismatch for {row['agent']}")
                output = 0
                for response in stream_generate(self.model, self.tokenizer, prompt[len(tokens):],
                                                 max_tokens=self._args.max_tokens, sampler=sampler,
                                                 prompt_cache=copy.deepcopy(cache)):
                    output = response.generation_tokens
                prefill, decode = self.recorder.take()
                prefill_total = prefill if prefill_total is None else prefill_total + prefill
                decode_total = decode if decode_total is None else decode_total + decode
                per_decision.append({"split": split, "agent": row["agent"], "gameTime": row["game_time"],
                                     "userTokens": len(prompt) - len(tokens), "outputTokens": output,
                                     "prefillDistinct": (prefill > 0).sum(axis=1).tolist(),
                                     "decodeDistinct": (decode > 0).sum(axis=1).tolist(),
                                     "touched": [np.flatnonzero(layer).tolist()
                                                 for layer in (prefill + decode)]})
                if number % 25 == 0:
                    print(f"{split} {number}/{len(rows)} ({time.monotonic() - started:.0f} s)", flush=True)
            counts[split] = {"prefill": prefill_total, "decode": decode_total}
        return self._summary(counts, system_slots, per_decision)

    def _summary(self, counts: dict, system_slots: np.ndarray, per_decision: list[dict]) -> dict:
        """Coverage of early-chosen hot sets on late decisions, concentration, and per-decision misses."""
        early = counts["early"]["prefill"] + counts["early"]["decode"]
        late = counts["late"]["prefill"] + counts["late"]["decode"]
        late_decode = counts["late"]["decode"]
        ranking = np.argsort(-early, axis=1)
        coverage, decode_coverage, hot_sets = {}, {}, {}
        for size in HOT_SIZES:
            hot = ranking[:, :size]
            hot_sets[size] = [set(layer.tolist()) for layer in hot]
            covered = np.take_along_axis(late, hot, axis=1).sum(axis=1) / late.sum(axis=1)
            covered_decode = np.take_along_axis(late_decode, hot, axis=1).sum(axis=1) / late_decode.sum(axis=1)
            coverage[size] = {"mean": float(covered.mean()), "min": float(covered.min())}
            decode_coverage[size] = {"mean": float(covered_decode.mean()), "min": float(covered_decode.min())}
        needed = {}
        for share in (0.9, 0.95, 0.99):
            ordered = -np.sort(-late, axis=1)
            cumulative = ordered.cumsum(axis=1) / ordered.sum(axis=1, keepdims=True)
            per_layer = (cumulative < share).sum(axis=1) + 1
            needed[share] = {"mean": float(per_layer.mean()), "max": int(per_layer.max())}
        quarter = self.recorder.num_experts // 4
        late_rows = [row for row in per_decision if row["split"] == "late"]
        misses = [sum(len(set(touched) - hot_sets[quarter][layer]) for layer, touched in enumerate(row["touched"]))
                  for row in late_rows]
        distinct = [float(np.mean(row["prefillDistinct"])) for row in late_rows]
        verdict = coverage[quarter]["mean"] >= 0.95
        return {
            "experts": self.recorder.num_experts, "moeLayers": len(self.recorder.layers),
            "decisions": {split: len(rows) for split, rows in self.splits.items()},
            "heldOutCoverageOfEarlyHotSet": coverage, "heldOutDecodeCoverage": decode_coverage,
            "expertsToCoverHeldOut": needed,
            "systemPromptExpertsTouchedPerLayer": float((system_slots > 0).sum(axis=1).mean()),
            "perDecision": {"userTokensMean": float(np.mean([r["userTokens"] for r in late_rows])),
                            "outputTokensMean": float(np.mean([r["outputTokens"] for r in late_rows])),
                            "distinctExpertsPerLayerPrefill": float(np.mean(distinct)),
                            "distinctExpertsPerLayerDecode": float(np.mean([np.mean(r["decodeDistinct"])
                                                                            for r in late_rows])),
                            "quarterHotSetMissesPerDecision": float(np.mean(misses))},
            "verdict": ("hot set worthwhile: 25% of experts cover >= 95% of held-out slots" if verdict else
                        "routing too spread out for a 25% hot set"),
        }


def main(argv: list[str] | None = None) -> int:
    """Profile routing and write the summary next to the run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a run directory with orchestrator/decisions.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--decisions", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/routing_profile.json")
    args = parser.parse_args(argv)
    summary = RoutingProfile(args).run()
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
