"""Check concurrent (batched) requests against a running local model server: tool-call validity and throughput.

Parallel study branches send model requests at the same time, which `mlx_lm.server` decodes as one batch. This sends the
same logged decision prompts once one at a time and once `--concurrency` at a time, and reports for each pass the share
of replies with a valid tool call (and how many needed the client's one retry) and decisions per minute. A batched pass
that loses valid tool calls is broken; one that keeps them is worth as much as its throughput gain.

Usage: `python3.11 scripts/archive/bench_batching.py --run runs/<run> --model <model path>
[--requests 24 --concurrency 4]`

Archived (C5, 2026-10-18): a one-off that showed batching is broken in mlx-lm 0.31.3 for this model (M2); the server now
serves one request at a time. Kept for reference; run it with `PYTHONPATH=scripts`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orchestrator.llm import OpenAICompatibleClient  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS  # noqa: E402


class BatchingBench:
    """Sends logged prompts serially and concurrently to one server and compares the passes."""

    def __init__(self, args):
        """Sample the prompts and build the client (thinking off, as in agent runs)."""
        manifest = ScenarioManifest.load(ROOT / f"scenario/village_{args.variant}.json")
        systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
        rows = [json.loads(line) for line in (args.run / "orchestrator/decisions.jsonl").open() if line.strip()]
        rows = [row for row in rows if row["source"] == "model" and row["agent"] in systems]
        self.prompts = [(systems[row["agent"]], row["user_prompt"])
                        for row in random.Random(args.seed).sample(rows, min(args.requests, len(rows)))]
        self.client = OpenAICompatibleClient(str(args.model), args.url, api_key="", temperature=0.7, max_tokens=384,
                                             extra={"chat_template_kwargs": {"enable_thinking": False}}, timeout=900)
        self.concurrency = args.concurrency

    def _pass(self, workers: int) -> dict:
        """Send every prompt with `workers` requests in flight; return validity and throughput."""
        started, retries = time.monotonic(), self.client.retries
        with ThreadPoolExecutor(max_workers=workers) as pool:
            responses = list(pool.map(lambda prompt: self.client.complete(*prompt, TOOL_DEFINITIONS), self.prompts))
        seconds = time.monotonic() - started
        valid = sum(response.tool_call is not None for response in responses)
        return {"concurrency": workers, "requests": len(responses), "validToolCalls": valid,
                "retriedForMissingToolCall": self.client.retries - retries,
                "seconds": round(seconds, 1), "decisionsPerMinute": round(60 * len(responses) / seconds, 1),
                "invalidSamples": [response.text[:200] for response in responses if response.tool_call is None][:3]}

    def run(self) -> dict:
        """A serial pass, then a concurrent pass (a single warm-up request first)."""
        self.client.complete(*self.prompts[0], TOOL_DEFINITIONS)
        serial, batched = self._pass(1), self._pass(self.concurrency)
        return {"serial": serial, "batched": batched,
                "throughputGain": round(batched["decisionsPerMinute"] / serial["decisionsPerMinute"], 2)}


def main(argv: list[str] | None = None) -> int:
    """Run both passes and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True, help="the model path the server was started with")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(BatchingBench(args).run(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
