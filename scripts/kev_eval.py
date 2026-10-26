"""Kev-4B on the delegate's decisions, through our own MLX loader (`orchestrator.kev_runtime`; no Kev code runs).

Run with the local-model environment's Python (it has `mlx`, `mlx-lm` and `numpy`). Three modes:

- `--parity`: the published reference outputs in `scripts/tests/fixtures_delegate/kev_reference.json` (Kev's Space
  preset with fp32 raw probabilities, and the README example in bf16 with the calibrated temperature): the loader must
  reproduce each case's input-token count exactly and every probability within the case's tolerance, with no change of
  the most likely option (cases marked informational are reported, not gated). Run this first; the other modes are only
  meaningful if it passes.
- `--check2` (Phase 0 check 2): latency and peak memory at two state lengths, about 7k tokens (the record whose state is
  nearest 7,000 tokens) and 12k tokens (the longest state with its history turns repeated until it reaches 12,288
  tokens; a synthetic length): the state prefill, one question on the cached state, and MLX's and the process's peaks.
- Default, zero-shot: on every representable decision record (`scripts/build_decision_dataset.py`), Kev's choice agrees
  when its candidate matches the frontier's under the formal Match; overall, on non-speech decisions, by tool and by
  state length (<= 8k, 8-12k); expected calibration error of p(choice) over 15 bins. The option-renaming check (typed
  decision models follow option names over definitions) asks the same question three more times on the cached state:
  descriptions only, names only (tool and arguments), and names permuted among the options (each option keeps its own
  description but shows another option's name); for the last, how often Kev picks the option whose description is the
  frontier's choice versus the one showing its name.

Usage: `<python> scripts/kev_eval.py --out <prefix> [--parity | --check2 --records <jsonl> | --records <jsonl>]`.
Writes `<prefix>.json` (and `<prefix>.jsonl`, one line per record, for the zero-shot mode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.candidates import same_choice  # noqa: E402
from orchestrator.delegate_input import QUESTION, state_text  # noqa: E402
from orchestrator.llm import ToolCall  # noqa: E402

GIB = 1024 ** 3
BINS = 15
HUB = Path.home() / ".cache/huggingface/hub"
BASE = HUB / "models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"
REFERENCE = ROOT / "scripts/tests/fixtures_delegate/kev_reference.json"


def split_label(label: str) -> tuple[str, str]:
    """A candidate label's name part (`tool {arguments}`) and its description."""
    name, _, description = label.partition(" | ")
    return name, description


def renamed(labels: list[str], seed: int) -> tuple[list[str], list[int]]:
    """Labels whose names are permuted among the options (each keeps its own description), and the permutation: option
    `i` shows the name of option `order[i]`. No option keeps its own name when there are at least two."""
    order = list(range(len(labels)))
    if len(order) > 1:
        generator = random.Random(seed)
        while any(i == j for i, j in enumerate(order)):
            generator.shuffle(order)
    parts = [split_label(label) for label in labels]
    return [f"{parts[j][0]} | {parts[i][1]}" for i, j in enumerate(order)], order


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = BINS) -> float:
    """Expected calibration error with equal-width confidence bins."""
    total, error = len(confidences), 0.0
    for b in range(bins):
        members = [k for k, p in enumerate(confidences) if b / bins < p <= (b + 1) / bins or (b == 0 and p == 0)]
        if members:
            accuracy = sum(correct[k] for k in members) / len(members)
            mean = sum(confidences[k] for k in members) / len(members)
            error += len(members) / total * abs(accuracy - mean)
    return error


def summary(rows: list[dict]) -> dict:
    """Agreement overall, non-speech, by tool and by state length; ECE; the renaming check; seconds."""
    def share(items, key="agree"):
        return sum(row[key] for row in items) / len(items) if items else None
    non_speech = [row for row in rows if not row["free_text"]]
    tools = sorted({row["tool"] for row in rows})
    return {"representable_decisions": len(rows), "agreement": share(rows),
            "non_speech_decisions": len(non_speech), "non_speech_agreement": share(non_speech),
            "by_tool": {tool: {"decisions": len([r for r in rows if r["tool"] == tool]),
                               "agreement": share([r for r in rows if r["tool"] == tool])} for tool in tools},
            "by_length": {"<=8k": share([r for r in rows if r["state_tokens"] <= 8192]),
                          "8-12k": share([r for r in rows if 8192 < r["state_tokens"] <= 12288])},
            "ece_15_bins": expected_calibration_error([r["p_choice"] for r in rows], [r["agree"] for r in rows]),
            "mean_p_choice": statistics.fmean(r["p_choice"] for r in rows) if rows else None,
            "renaming": {"descriptions_only_agreement": share(rows, "descriptions_only_agree"),
                         "names_only_agreement": share(rows, "names_only_agree"),
                         "permuted_follows_description": share(rows, "permuted_follows_description"),
                         "permuted_follows_name": share(rows, "permuted_follows_name")},
            "seconds_median": statistics.median(r["seconds"] for r in rows) if rows else None}


class KevEvaluation:
    """Loads Kev-4B with our loader and runs one evaluation mode."""

    def __init__(self, args):
        """Load the model (base, folded adapter, head), recording load time and memory."""
        import mlx.core as mx
        from orchestrator.kev_runtime import KevModel
        self._mx, self._args = mx, args
        adapter = (next((HUB / "models--jaredpalmer--kev-4b/snapshots").iterdir()) if args.adapter is None
                   else Path(args.adapter))
        mx.reset_peak_memory()
        started = time.monotonic()
        self.model = KevModel(Path(args.base), adapter)
        self.load = {"seconds": round(time.monotonic() - started, 1), "folded_tensors": self.model.folded,
                     "mlx_peak_gib": mx.get_peak_memory() / GIB,
                     "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB}
        self.records = ([json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()
                         if line.strip()] if args.records else [])

    def parity(self) -> dict:
        """Each published case: input tokens (exact), max |dp| against the tolerance, and argmax agreement."""
        cases = json.loads(REFERENCE.read_text(encoding="utf-8"))["cases"]
        results = {}
        for case in cases:
            questions = [(q["instruction"], q["options"]) for q in case["questions"]]
            probabilities = self.model.ask(case["state"], questions, temperature=case["temperature"])
            tokens = len(self.model.state_ids(case["state"])) + sum(
                len(self.model.branch(*question)[0]) for question in questions)
            deltas = [abs(p - r) for got, q in zip(probabilities, case["questions"])
                      for p, r in zip(got, q["reference"])]
            flips = sum(max(range(len(got)), key=got.__getitem__)
                        != max(range(len(q["reference"])), key=q["reference"].__getitem__)
                        for got, q in zip(probabilities, case["questions"]))
            results[case["name"]] = {"input_tokens": tokens, "expected_input_tokens": case["input_tokens"],
                                     "max_abs_dp": max(deltas), "tolerance": case["tolerance"],
                                     "argmax_flips": flips,
                                     "probabilities": [[round(p, 4) for p in got] for got in probabilities],
                                     "passed": (tokens == case["input_tokens"] and max(deltas) <= case["tolerance"]
                                                and flips == 0)}
        gated = [results[case["name"]]["passed"] for case in cases if not case.get("informational")]
        return {"load": self.load, "cases": results, "passed": all(gated)}

    def _question(self, record: dict, labels: list[str] | None = None):
        """The question row of a record (its labels, or replacements in the same order)."""
        options = labels if labels is not None else [candidate["label"] for candidate in record["candidates"]]
        return self.model.branch(QUESTION.format(agent=record["agent"]), options)

    def check2(self) -> dict:
        """Prefill and question latency, and peak memory, at about 7k and 12k state tokens."""
        mx = self._mx
        lengths = [(record, len(self.model.state_ids(state_text(record["state"])))) for record in self.records]
        near7k = min(lengths, key=lambda pair: abs(pair[1] - 7000))[0]
        long_record = json.loads(json.dumps(max(lengths, key=lambda pair: pair[1])[0]))
        history, turn = list(long_record["state"]["history"]), 0
        while len(self.model.state_ids(state_text(long_record["state"]))) < 12288 and history:
            long_record["state"]["history"].append(history[turn % len(history)])
            turn += 1
        results = {}
        for name, record in (("7k", near7k), ("12k", long_record)):
            ids = self.model.state_ids(state_text(record["state"]))
            row = self._question(record)
            mx.reset_peak_memory()
            started = time.monotonic()
            cache = self.model.prefill(ids)
            prefill = time.monotonic() - started
            started = time.monotonic()
            self.model.scores(cache, [row])
            question = time.monotonic() - started
            results[name] = {"state_tokens": len(ids), "options": len(record["candidates"]),
                             "question_tokens": len(row[0]), "prefill_seconds": round(prefill, 2),
                             "question_on_cached_state_seconds": round(question, 2),
                             "mlx_peak_gib": mx.get_peak_memory() / GIB,
                             "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB}
        return {"load": self.load, "lengths": results}

    def zero_shot(self) -> tuple[list[dict], dict]:
        """Score every representable record with plain labels and the three renaming variants."""
        rows = []
        for index, record in enumerate(self.records):
            if not record["representable"]:
                continue
            frontier = ToolCall(record["frontier"]["name"], record["frontier"]["args"])
            labels = [candidate["label"] for candidate in record["candidates"]]
            calls = [ToolCall(candidate["call"]["name"], candidate["call"]["args"])
                     for candidate in record["candidates"]]
            permuted, order = renamed(labels, seed=index)
            variants = [None, [split_label(label)[1] for label in labels],
                        [split_label(label)[0] for label in labels], permuted]
            ids = self.model.state_ids(state_text(record["state"]))
            started = time.monotonic()
            cache = self.model.prefill(ids)
            scores = self.model.scores(cache, [self._question(record, variant) for variant in variants])
            seconds = time.monotonic() - started
            picks = [max(range(len(score)), key=lambda k, s=score: s[k]) for score in scores]
            p = self.model.probabilities(scores[0])
            rows.append({"run": record["run"], "agent": record["agent"], "game_time": record["game_time"],
                         "tool": frontier.name, "free_text": record["free_text"], "options": len(labels),
                         "state_tokens": len(ids), "seconds": round(seconds, 2), "p_choice": p[picks[0]],
                         "agree": same_choice(calls[picks[0]], frontier),
                         "descriptions_only_agree": same_choice(calls[picks[1]], frontier),
                         "names_only_agree": same_choice(calls[picks[2]], frontier),
                         "permuted_follows_description": same_choice(calls[picks[3]], frontier),
                         "permuted_follows_name": same_choice(calls[order[picks[3]]], frontier)})
        return rows, summary(rows)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the model, run the chosen mode, and write the results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--adapter", default=None, help="Kev-4B snapshot directory (default: the cached one)")
    parser.add_argument("--records", default=None)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parity", action="store_true")
    mode.add_argument("--check2", action="store_true")
    args = parser.parse_args(argv)
    evaluation = KevEvaluation(args)
    if args.parity:
        result = evaluation.parity()
    elif args.check2:
        result = evaluation.check2()
    else:
        rows, result = evaluation.zero_shot()
        result["load"] = evaluation.load
        Path(str(args.out) + ".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    Path(str(args.out) + ".json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
