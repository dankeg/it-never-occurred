"""Collect the M8 results (gates V-G1 to V-G5) from finished runs into `runs/m8_report.json` and print a summary.

Offline and $0: it reads what the M8 pipeline (`scripts/m8_pipeline.sh`) wrote and never starts a server or calls a
model. Missing runs are reported as missing, not skipped silently. Credits use each model's plan rates per million
tokens (input, cached input, output): Sol 50 / 2.5 / 250, Luna 2.5 / 0.25 / 12.5.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

RUNS = Path(__file__).resolve().parents[1] / "runs"
RATES = {"gpt-6.1-sol": (50.0, 2.5, 250.0), "gpt-6-luna": (2.5, 0.25, 12.5)}
VG1 = ("m8_vg1_sabotage_1", "m8_vg1_sabotage_2", "m8_vg1_accident_1")
STUDIES = ("road_closure", "no_saboteur", "honest_freight")
FAILED = re.compile(r"^(not possible|failed)")


class RunLog:
    """One run directory's decision log, read once: decisions, failure share and credits."""

    def __init__(self, directory: Path):
        """Read the run's decisions (empty when the run has none yet)."""
        self.directory = directory
        path = directory / "orchestrator/decisions.jsonl"
        self.decisions = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
                          if line.strip()] if path.exists() else []

    def summary(self) -> dict:
        """Decision count, share of failed actions, model calls, and credits at the plan rates."""
        called = [row for row in self.decisions if row.get("source", "model") == "model"]
        credits = 0.0
        for row in called:
            rate_in, rate_cached, rate_out = RATES.get(row.get("model"), (0.0, 0.0, 0.0))
            credits += (row.get("input_tokens", 0) * rate_in + row.get("cached_tokens", 0) * rate_cached
                        + row.get("output_tokens", 0) * rate_out) / 1e6
        failed = sum(bool(FAILED.match(str(row.get("outcome") or ""))) for row in self.decisions)
        return {"decisions": len(self.decisions), "model_calls": len(called),
                "failed_share": failed / len(self.decisions) if self.decisions else None,
                "credits": round(credits, 2)}


class M8Report:
    """Gather every gate's evidence into one dictionary."""

    def _json(self, path: Path) -> dict | None:
        """A JSON file's content, or None when it does not exist yet."""
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def vg1(self) -> dict:
        """Per seed: run summary, replay verdict (V-G2), outcomes, first-true ticks and per-run metrics (V-G3)."""
        result = {}
        for name in VG1:
            directory, output = RUNS / name, RUNS / f"{name}.out"
            report = self._json(directory / "m5_report.json")
            replay = next((line.strip() for line in output.read_text(encoding="utf-8").splitlines()
                           if re.match(r"^(PASS|FAIL) replay", line)), None) if output.exists() else None
            result[name] = {"run": RunLog(directory).summary(), "replay": replay,
                            "end": report and report.get("end"), "stopped": report and report.get("stopped"),
                            "outcomes": report and report.get("outcomes"),
                            "first_true": report and report.get("first_true"),
                            "welfare": report and report.get("welfare"),
                            "metrics": self._json(directory / "metrics.json")}
        return result

    def studies(self) -> dict:
        """Per study: its base, pairs finished, paired effects with 95% CIs, model calls, and the share saved by cone
        gating. road_closure forks at day 1 (base m8_luna); the other two at agent start (base m8b_luna)."""
        result = {}
        for study in STUDIES:
            report = self._json(RUNS / f"study_{study}_sabotage_report.json")
            result[study] = None if report is None else {
                key: report.get(key) for key in ("base", "branches", "pairsFinished", "stopped", "effects",
                                                 "model_calls", "decisions", "calls_saved_by_gating")}
        branches = (sorted(RUNS.glob("m8_luna_*")) + sorted(RUNS.glob("m8b_luna_*"))  # day-1 base, agent-start base
                    + sorted(RUNS.glob("study_*_sabotage_intervention_*")))
        result["credits"] = round(sum(RunLog(path).summary()["credits"] for path in branches if path.is_dir()), 2)
        return result

    def vg5(self) -> dict:
        """The narrated run: narration error, narrated against engine outcomes, calls, and agents' and GM's credits.

        The GM ran on the agents' model (`--gm-model` was not set), so its tokens are priced at that model's rates.
        """
        directory = RUNS / "m8_narrated_sol"
        report = self._json(directory / "narrated_report.json")
        run = RunLog(directory)
        summary = run.summary()
        if report is not None and report.get("gm_tokens"):
            tokens = report["gm_tokens"]
            model = next((row.get("model") for row in run.decisions if row.get("model")), None)
            rate_in, rate_cached, rate_out = RATES.get(model, (0.0, 0.0, 0.0))
            summary["gm_credits"] = round((tokens.get("input", 0) * rate_in + tokens.get("cached", 0) * rate_cached
                                           + tokens.get("output", 0) * rate_out) / 1e6, 2)
            summary["total_credits"] = round(summary["credits"] + summary["gm_credits"], 2)
        return {"run": summary, "report": None if report is None else {
            key: report.get(key) for key in ("start", "end", "stopped", "narration_error", "narrated_outcomes",
                                             "shadow_outcomes", "outcome_divergence", "agent_calls", "gm_calls",
                                             "acts", "gm_errors", "model_errors")}}

    def build(self) -> dict:
        """Every gate's evidence, written to `runs/m8_report.json`."""
        report = {"vg1": self.vg1(), "studies": self.studies(), "vg5": self.vg5()}
        (RUNS / "m8_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return report


def main() -> int:
    """Build the report and print a short per-gate summary."""
    report = M8Report().build()
    for name, run in report["vg1"].items():
        print(f"V-G1 {name}: {run['run']} replay={run['replay']} first_true={run['first_true']}")
    for study in STUDIES:
        entry = report["studies"][study]
        print(f"V-G4 {study}: " + ("missing" if entry is None else
                                    f"pairs {entry['pairsFinished']}, saved {entry['calls_saved_by_gating']}"))
    print(f"V-G4 credits: {report['studies']['credits']}")
    vg5 = report["vg5"]
    error = (vg5["report"] or {}).get("narration_error") or {}
    print(f"V-G5: {vg5['run']} infeasible_rate={error.get('infeasible_rate')} "
          f"divergence={(vg5['report'] or {}).get('outcome_divergence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
