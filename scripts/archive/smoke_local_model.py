"""Smoke test for a local agent model (e.g. Qwen3.6-35B-A3B): latency, tool-call validity, prompt caching.

It replays real decision prompts from a finished village run (the system prompt rebuilt from the scenario manifest,
the user prompt from the decision log) against the local server, unseeded:
- three decisions per agent in sequence, to measure reuse of each agent's cached system prefix;
- all six agents at once, to measure batching;
- for every call, whether it is one valid tool call.

It prints a short report. Village behaviour itself is tested by running `run_village.py --provider mlx-local`.

Archived (C5): latency, tool validity and conversation-cache checks are now gated by `scripts/test_it12_local_model.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.llm import OpenAICompatibleClient  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402
from orchestrator.tools import TOOL_DEFINITIONS, TOOL_NAMES, ToolTranslator  # noqa: E402

from scripts.local_model import DEFAULT_MODEL, MlxServer  # noqa: E402


class SmokeTest:
    """Times and checks local-model decisions on real prompts from a finished run."""

    def __init__(self, client: OpenAICompatibleClient, run: Path, manifest: ScenarioManifest):
        """Use prompts from `run`'s decision log, with system prompts rebuilt from `manifest`."""
        self._client = client
        self._systems = {agent.name: agent.system_prompt() for agent in manifest.agents()}
        self._prompts: dict[str, list[str]] = {}
        for line in (run / "orchestrator/decisions.jsonl").open(encoding="utf-8"):
            decision = json.loads(line)
            self._prompts.setdefault(decision["agent"], []).append(decision["user_prompt"])

    def call(self, agent: str, index: int) -> dict:
        """One decision on `agent`'s `index`-th logged prompt: timing, tool call, validity."""
        start = time.monotonic()
        response = self._client.complete(self._systems[agent], self._prompts[agent][index], TOOL_DEFINITIONS)
        call = response.tool_call
        return {"agent": agent, "seconds": round(time.monotonic() - start, 2), "valid": self.valid(agent, call),
                "tool": getattr(call, "name", None), "args": getattr(call, "args", None),
                "input": response.input_tokens, "cached": response.cache_read_tokens, "output": response.output_tokens}

    @staticmethod
    def valid(agent: str, call) -> str:
        """'ok', or why the tool call is unusable (no call, unknown tool, or arguments the translator rejects)."""
        if call is None:
            return "no tool call"
        if call.name not in TOOL_NAMES:
            return f"unknown tool {call.name}"
        try:
            ToolTranslator().translate(agent, call)
        except (ValueError, KeyError, TypeError) as failure:
            return f"bad arguments: {failure}"
        return "ok"

    def run(self) -> dict:
        """Three sequential decisions per agent (prefix reuse), then all agents at once (batching);
        return the report."""
        report = {"sequential": [], "concurrent": []}
        for agent in self._systems:
            for index in range(3):
                report["sequential"].append(self.call(agent, index))
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self._systems)) as pool:
            report["concurrent"] = list(pool.map(lambda agent: self.call(agent, 3), self._systems))
        report["concurrent_wall_seconds"] = round(time.monotonic() - start, 2)
        return report


def main(argv: list[str] | None = None) -> int:
    """Start the local server, run the smoke test, print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "runs/m5_village_sabotage")
    parser.add_argument("--variant", default="sabotage")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    manifest = ScenarioManifest.load(Path(__file__).resolve().parents[2] / "scenario" / f"village_{args.variant}.json")
    log = Path(__file__).resolve().parents[2] / "runs/mlx_server.log"
    with MlxServer(args.model, log=log) as server:
        client = OpenAICompatibleClient(str(args.model), server.url, api_key="", temperature=0.7, max_tokens=512,
                                        extra={"chat_template_kwargs": {"enable_thinking": False}}, timeout=600.0)
        report = SmokeTest(client, args.run, manifest).run()
    for result in report["sequential"] + report["concurrent"]:
        print(f"  {result['agent']:>12} {result['seconds']:6.2f}s {result['valid']:<14} {result['tool']} "
              f"{json.dumps(result['args'])[:110]}")
    calls = report["sequential"] + report["concurrent"]
    valid = sum(1 for result in calls if result["valid"] == "ok")
    first = [r["seconds"] for r in report["sequential"][0::3]]
    later = [r["seconds"] for i, r in enumerate(report["sequential"]) if i % 3]
    print(f"valid tool calls: {valid}/{len(calls)}; first decision per agent mean {sum(first) / len(first):.1f}s, "
          f"later decisions mean {sum(later) / len(later):.1f}s; "
          f"6 agents at once: {report['concurrent_wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
