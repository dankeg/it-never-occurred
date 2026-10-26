"""Measure multi-token-prediction (MTP) speculative decoding on logged agent decisions, at batch size 1.

Run with the MLX environment's Python and the unpacked `mlx-optiq` package on the path (its MTP runtime is vendored
from MTPLX and rolls back the hybrid model's recurrent state on rejected drafts). The model directory needs the MTP
head as `optiq/mtp.safetensors` (mlx-lm ignores that folder).

For each sampled decision, each agent's system prompt and tools are prefilled once and copied, then the user message
is prefilled and the reply decoded with plain autoregressive decoding (depth 0) and with MTP drafts of each depth, at
the agents' sampling temperature. Reported per mode: decode tokens/s, draft acceptance, and the share of replies
that are a complete tool call (a check that speculation leaves the output distribution intact).

Usage: `<mlx python> scripts/archive/bench_mtp.py --optiq <unpacked wheel dir> --run runs/<run> --decisions 40`
(the model server must be stopped: the model is loaded in this process).

Archived (C5, 2026-10-18): a one-off that measured MTP speculative decoding on logged decisions (systems track). Kept
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "models/Qwen3.6-35B-A3B-4bit"


class MtpBench:
    """Loads the model with its MTP head and decodes logged decisions in each mode."""

    def __init__(self, args):
        """Load the model, attach the MTP head, and sample the decisions."""
        sys.path.insert(0, str(args.optiq))
        sys.path.insert(0, str(ROOT))
        import mlx.core as mx
        from mlx_lm import load
        from optiq.runtime.engine import OptiqEngine
        from orchestrator.scenario import ScenarioManifest
        from orchestrator.tools import TOOL_DEFINITIONS

        mx.set_cache_limit(1 << 30)
        self._args = args
        self._mx = mx
        self.model, self.tokenizer = load(str(args.model))
        self.engine = OptiqEngine.from_loaded(self.model, self.tokenizer, str(args.model))
        if not self.engine.has_mtp:
            raise RuntimeError(f"No MTP head found under {args.model}/optiq/")
        manifest = ScenarioManifest.load(ROOT / f"scenario/village_{args.variant}.json")
        self.systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
        self.tools = [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS]
        rows = [json.loads(line) for line in (args.run / "orchestrator/decisions.jsonl").open() if line.strip()]
        rows = [row for row in rows if row["source"] == "model" and row["agent"] in self.systems]
        self.rows = random.Random(args.seed).sample(rows, min(args.decisions, len(rows)))
        self._prefixes: dict[str, tuple[list[int], list]] = {}

    def _template(self, agent: str, user: str | None) -> list[int]:
        """Prompt tokens for an agent's system prompt and tools, plus a user message (or an empty one)."""
        messages = [{"role": "system", "content": self.systems[agent]}, {"role": "user", "content": user or ""}]
        return self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=user is not None,
                                                  enable_thinking=False)

    def _prefix(self, agent: str) -> tuple[list[int], list]:
        """The agent's system-and-tools tokens and a prefilled cache for them (computed once)."""
        if agent not in self._prefixes:
            empty, marked = self._template(agent, None), self._template(agent, "x")
            end = next(i for i, (a, b) in enumerate(zip(empty, marked)) if a != b)
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(self.model)
            self.model(self._mx.array(empty[:end])[None], cache=cache)
            self._mx.eval([c.state for c in cache])
            self._prefixes[agent] = (empty[:end], cache)
        return self._prefixes[agent]

    def _decode(self, row: dict, depth: int) -> dict:
        """Decode one decision's reply in one mode; return its token count, decode time and draft counts."""
        tokens, cache = self._prefix(row["agent"])
        prompt = self._template(row["agent"], row["user_prompt"])
        if prompt[:len(tokens)] != tokens:
            raise RuntimeError(f"system prefix mismatch for {row['agent']}")
        text, events, first = "", 0, None
        for event in self.engine.generate_stream(prompt="", prompt_ids=prompt[len(tokens):],
                                                 cache=copy.deepcopy(cache), max_tokens=self._args.max_tokens,
                                                 depth=depth, temperature=self._args.temperature,
                                                 chat_template=False):
            text += event["text"]
            events += 1
            if first is None:
                first = time.perf_counter()  # the first token comes from prefill; decoding starts here
        stats = self.engine._last_stats if depth else None  # the engine keeps draft counts only for MTP
        return {"tokens": events - 1, "seconds": time.perf_counter() - first,
                "attempted": stats.drafts_attempted if stats else 0, "accepted": stats.drafts_accepted if stats else 0,
                "toolCall": "<tool_call>" in text and "</tool_call>" in text}

    def run(self) -> dict:
        """Decode every sampled decision in every mode (modes interleaved per decision) and summarize."""
        modes = [0, *self._args.depths]
        results = {depth: [] for depth in modes}
        started = time.monotonic()
        self._decode(self.rows[0], 0)  # warm-up: kernel compilation and the first prefix
        for number, row in enumerate(self.rows):
            for depth in modes:
                results[depth].append(self._decode(row, depth))
            if number % 10 == 0:
                print(f"{number}/{len(self.rows)} ({time.monotonic() - started:.0f} s)", flush=True)
        plain_rate = sum(r["tokens"] for r in results[0]) / sum(r["seconds"] for r in results[0])
        summary = {"decisions": len(self.rows), "temperature": self._args.temperature, "modes": {}}
        for depth, rows in results.items():
            rate = sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows)
            attempted = sum(r["attempted"] for r in rows)
            summary["modes"][f"depth{depth}"] = {
                "decodeTokensPerSecond": round(rate, 1), "speedup": round(rate / plain_rate, 3),
                "meanOutputTokens": round(sum(r["tokens"] for r in rows) / len(rows), 1),
                "acceptance": round(sum(r["accepted"] for r in rows) / attempted, 3) if attempted else None,
                "toolCallShare": round(sum(r["toolCall"] for r in rows) / len(rows), 3)}
        return summary


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write its summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optiq", type=Path, required=True, help="directory holding the `optiq` package")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mtp_bench.json")
    args = parser.parse_args(argv)
    summary = MtpBench(args).run()
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
