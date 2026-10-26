"""Run the village crisis with LLM agents (M5 live; the base for M7/M8 runs).

This spends money: every agent decision that isn't replayed is a model call. `--max-calls` is a hard budget (the run
stops and reports when it would be exceeded), and the report prints token usage and the estimated cost at the given
per-million-token prices. Requires an API key for the chosen provider (e.g. ANTHROPIC_API_KEY).

`--checkpoint-every N` saves world and agent snapshots every N ticks (a multiple of 1,200). `--resume NAME` keeps an
unfinished run aside and restarts NAME from its latest checkpoint, reusing its logged decisions where prompts match.
Supply the same model and end-time options as the interrupted run; `--replay` then checks the resumed continuation.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import harness
    from .test_m5_village import VARIANTS, M5VillageExperiment, decision_trace
    from .village_session import Checkpoints
else:
    import harness
    from test_m5_village import VARIANTS, M5VillageExperiment, decision_trace
    from village_session import Checkpoints

from orchestrator.chatgpt import ChatGPTAuth, ChatGPTPlanClient  # noqa: E402  (test_m5_village sets sys.path)
from orchestrator.decisions import DecisionLog  # noqa: E402
from orchestrator.llm import AnthropicClient, OpenAICompatibleClient  # noqa: E402
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS  # noqa: E402


LOCAL_URL = "http://127.0.0.1:8080/v1"


def model_client(args):
    """Build the model client named on the command line.

    `mlx-local` is a local `mlx_lm.server` (e.g. Qwen3.6-35B-A3B): no key, thinking off, short tool-call outputs.
    `chatgpt-plan` uses the signed-in user's ChatGPT plan (run `scripts/chatgpt_login.py` once); `--model` is a slug
    from that plan's catalog.
    """
    if args.provider == "chatgpt-plan":
        return ChatGPTPlanClient(args.model, ChatGPTAuth.shared(), reasoning_effort=args.reasoning_effort)
    if args.provider == "anthropic":
        return AnthropicClient(args.model, temperature=args.temperature)
    if args.provider == "mlx-local":
        # Qwen's recommended non-thinking sampling: same judged quality as untruncated sampling, but no retries.
        return OpenAICompatibleClient(args.model, args.base_url or LOCAL_URL, api_key="", temperature=args.temperature,
                                      max_tokens=384, extra={"chat_template_kwargs": {"enable_thinking": False},
                                                             "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
                                      timeout=600.0)
    return OpenAICompatibleClient(args.model, args.base_url, temperature=args.temperature)


def cost(stats: dict, args) -> float:
    """Estimated spend in dollars: uncached input, cache writes (1.25x) and reads (0.1x), and output."""
    return (stats["input_tokens"] * args.price_in + stats["cache_write_tokens"] * args.price_in * 1.25
            + stats["cache_read_tokens"] * args.price_in * 0.1 + stats["output_tokens"] * args.price_out) / 1e6


def prepare_resume(directory: Path, end_tick: int) -> Path:
    """Refuse finished or uncheckpointed runs; keep an unfinished run aside and return its relocated checkpoint."""
    if (directory / "m5_report.json").exists():
        raise ValueError(f"Cannot resume finished run {directory}: m5_report.json exists")
    checkpoint = Checkpoints.latest(directory, end_tick)
    if __package__:
        from .run_study import set_aside
    else:
        from run_study import set_aside
    return set_aside(directory) / checkpoint.relative_to(directory)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the village with LLM agents, and print the report."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=VARIANTS,
                        default="sabotage")
    parser.add_argument("--provider", choices=("anthropic", "openai-compatible", "mlx-local", "chatgpt-plan"),
                        default="anthropic")
    parser.add_argument("--reasoning-effort", default=None, help="for --provider chatgpt-plan, e.g. low")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None, help="for --provider openai-compatible")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=None, help="end at this game time instead of after --days")
    parser.add_argument("--max-calls", type=int, required=True, help="hard budget of model calls")
    parser.add_argument("--price-in", type=float, default=0.0, help="dollars per million input tokens")
    parser.add_argument("--price-out", type=float, default=0.0, help="dollars per million output tokens")
    parser.add_argument("--name", default=None, help="run directory name (default: llm_village_<variant>)")
    parser.add_argument("--resume", default=None, help="restart this unfinished run from its latest checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="save a checkpoint every N ticks")
    parser.add_argument("--epoch-turns", type=int, default=8,
                        help="decisions per append-only prompt epoch (1: every prompt carries the full memory)")
    parser.add_argument("--replay", action="store_true",
                        help="then replay the run from its decision log with no model and check it is identical")
    args = parser.parse_args(argv)
    if args.resume is not None and args.name not in (None, args.resume):
        parser.error("--name must match --resume when both are supplied")
    checkpoints = Checkpoints(args.checkpoint_every) if args.checkpoint_every is not None else None
    experiment = M5VillageExperiment(args)
    name = args.resume or args.name or f"llm_village_{args.variant}"
    end = args.ticks if args.ticks is not None else args.days * 24_000
    resume_from = prepare_resume(args.runs_dir / name, end) if args.resume is not None else None
    source = (DecisionLog(resume_from.parent.parent / "orchestrator" / "decisions.jsonl")
              if resume_from is not None else None)
    # A local model or the user's plan judges reports at no extra cost.
    judge = model_client(args) if args.provider in ("mlx-local", "chatgpt-plan") else None
    runtime, tracker, report = experiment.run_once(name, model_client(args), source, end_tick=end,
                                                   max_model_calls=args.max_calls,
                                                   model_concurrency=1 if args.provider == "mlx-local" else None,
                                                   min_gap=THINKING_TICKS, rest=NIGHT_REST,
                                                   epoch_turns=args.epoch_turns, judge_client=judge,
                                                   checkpoints=checkpoints, resume_from=resume_from)
    report["estimated_cost_usd"] = round(cost(report["stats"], args), 4)
    print(json.dumps({key: report[key] for key in ("end", "stopped", "stats", "first_true", "welfare", "outcomes",
                                                   "estimated_cost_usd")}, indent=1))
    if not args.replay:
        return 0
    replay, _, _ = experiment.run_once(f"{name}_replay", None, runtime.log, end_tick=report["end"],
                                       min_gap=THINKING_TICKS, rest=NIGHT_REST, epoch_turns=args.epoch_turns,
                                       resume_from=resume_from)
    same = (replay.stats.model_calls == 0 and decision_trace(replay.log) == decision_trace(runtime.log)
            and replay.hashes == runtime.hashes)
    print(f"{'PASS' if same else 'FAIL'} replay of {name}: {replay.stats.replayed} decisions reused, "
          f"{replay.stats.model_calls} model calls, {len(replay.hashes)} hashes")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
