"""The model-cost ledger (W3): credits by day, purpose and model across every run, at the ChatGPT plan's rates.

Offline and $0. Three sources:
- agent decisions: each run's `orchestrator/decisions.jsonl` (model calls only; replayed and held decisions are free);
- narrated game masters: `narrated_report.json`'s `gm_tokens`, priced at the GM's model (the agents' unless the report
  names `gm_model`);
- offline model calls (judges, extractors, auditors): rows appended to `runs/ledger.jsonl` by `LedgerWriter`.

A run is dated by its first recorded start (`provenance.json`), else its decision log's modification time. Credits
use plan rates per million tokens (input, cached input, output); `input_tokens` in our logs exclude cached tokens.

Usage: `python3.11 scripts/cost_ledger.py [--since YYYY-MM-DD] [--cap CREDITS] [--by run]`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
RATES = {"gpt-6.1-sol": (50.0, 2.5, 250.0), "gpt-6-luna": (2.5, 0.25, 12.5)}
LEDGER = RUNS / "ledger.jsonl"


def credits(model: str | None, uncached: int, cached: int, output: int) -> float:
    """Plan credits for one call or a token total (0 for a local or unknown model)."""
    rate_in, rate_cached, rate_out = RATES.get(model, (0.0, 0.0, 0.0))
    return (uncached * rate_in + cached * rate_cached + output * rate_out) / 1e6


class LedgerWriter:
    """Appends one row per offline model call (a judge, extractor or auditor) to `runs/ledger.jsonl`."""

    def __init__(self, purpose: str, path: Path = LEDGER):
        """`purpose` names the tool, e.g. `auditor` or `investigation_judge`."""
        self._purpose = purpose
        self._path = path

    def record(self, run: str, response) -> None:
        """Log an `LLMResponse`'s tokens and credits against `run`."""
        row = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "run": run,
               "purpose": self._purpose, "model": response.model, "input_tokens": response.input_tokens,
               "cached_tokens": response.cache_read_tokens, "output_tokens": response.output_tokens}
        row["credits"] = round(credits(row["model"], row["input_tokens"], row["cached_tokens"],
                                       row["output_tokens"]), 4)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


class CostLedger:
    """Collects cost rows (date, run, purpose, model, credits) from every source under a runs directory."""

    def __init__(self, runs: Path = RUNS):
        """Keep the runs directory; rows are read by `rows()`."""
        self._runs = runs

    def _date(self, directory: Path) -> str:
        """The run's start date: its first provenance start, else the decision log's modification date."""
        provenance = directory / "provenance.json"
        if provenance.exists():
            starts = json.loads(provenance.read_text(encoding="utf-8")).get("starts") or []
            if starts and starts[-1].get("time"):
                return starts[-1]["time"][:10]
        log = directory / "orchestrator/decisions.jsonl"
        return datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d")

    def _agent_rows(self, directory: Path) -> list[dict]:
        """One row per model used by the run's agents (model calls only)."""
        totals = defaultdict(float)
        with (directory / "orchestrator/decisions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source", "model") != "model":
                    continue
                totals[row.get("model")] += credits(row.get("model"), row.get("input_tokens", 0),
                                                    row.get("cached_tokens", 0), row.get("output_tokens", 0))
        date = self._date(directory)
        return [{"date": date, "run": directory.name, "purpose": "agents", "model": model, "credits": value}
                for model, value in totals.items() if value]

    def _gm_rows(self, directory: Path, agent_model: str | None) -> list[dict]:
        """The narrated game master's tokens, if the run logged them."""
        report = json.loads((directory / "narrated_report.json").read_text(encoding="utf-8"))
        tokens = report.get("gm_tokens")
        if not tokens:
            return []
        model = report.get("gm_model") or agent_model
        value = credits(model, tokens.get("input", 0), tokens.get("cached", 0), tokens.get("output", 0))
        return [{"date": self._date(directory), "run": directory.name, "purpose": "game_master", "model": model,
                 "credits": value}]

    def rows(self) -> list[dict]:
        """Every cost row: agents and game masters per run, plus the offline ledger's calls."""
        rows = []
        for log in sorted(self._runs.glob("*/orchestrator/decisions.jsonl")):
            directory = log.parent.parent
            agent_rows = self._agent_rows(directory)
            rows += agent_rows
            if (directory / "narrated_report.json").exists():
                rows += self._gm_rows(directory, agent_rows[0]["model"] if agent_rows else None)
        ledger = self._runs / LEDGER.name
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows.append({"date": row["time"][:10], "run": row["run"], "purpose": row["purpose"],
                                 "model": row["model"], "credits": row["credits"]})
        return rows


def main(argv: list[str] | None = None) -> int:
    """Print credits by day (and purpose and model), or by run, since a date, against an optional cap."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=None, help="only rows dated on or after YYYY-MM-DD")
    parser.add_argument("--cap", type=float, default=None, help="a credit cap to report the remainder against")
    parser.add_argument("--by", choices=("day", "run"), default="day")
    args = parser.parse_args(argv)
    rows = [row for row in CostLedger().rows() if args.since is None or row["date"] >= args.since]
    groups = defaultdict(float)
    for row in rows:
        key = (row["date"], row["purpose"], row["model"]) if args.by == "day" else (row["run"], row["purpose"],
                                                                                     row["model"])
        groups[key] += row["credits"]
    for key in sorted(groups):
        print(f"{' '.join(str(part) for part in key):70s} {groups[key]:9.2f}")
    total = sum(groups.values())
    print(f"{'total':70s} {total:9.2f}")
    if args.cap is not None:
        print(f"{'remaining under cap':70s} {args.cap - total:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
