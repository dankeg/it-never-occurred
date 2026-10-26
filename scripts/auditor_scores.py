"""Score auditor JSONL results against a JSON object mapping run names to labels.

Run ``python3.11 scripts/auditor_scores.py results.jsonl labels.json --positive yes``.
Verdict confidence MUST be the score for --positive, not confidence in the
selected verdict label. Accuracy compares the label field with ground truth;
AUROC and FPR use confidence without flipping scores for negative predictions.
Labels are compared by canonical JSON encoding, preserving JSON types.

FPR-at-TPR means the minimum FPR reaching the target TPR, not the literally
lowest threshold (which would accept everything). Predictions use score >=
threshold; tied scores enter together, with no interpolation or tie splitting.
Target TPR zero admits a threshold above every score and therefore FPR zero.
AUROC and FPR are undefined when either ground-truth class is absent.

Reports group by (question, view, model). n counts labeled runs, not rows across
groups. Repeated runs within a group are rejected, including unlabeled runs.
Unlabeled runs are listed and excluded; an explicitly null label is still a label.
Each group resamples n runs with replacement, without class stratification, using
a fresh seeded RNG and sorted run names. The 95% percentile intervals interpolate
linearly at ranks (replicates - 1) * quantile. Undefined replicates are excluded
per metric and bootstrap_defined reports the remaining count. An unavailable
estimate or interval is null. n is fixed by resampling and has no interval.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
import argparse
import json
import math
import random


def _label_key(label: object) -> str:
    """Compare JSON labels without coercing booleans, numbers or strings."""
    try:
        return json.dumps(label, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"labels must be finite JSON values: {error}") from error


def _read_results(path: Path) -> Iterable[dict]:
    """Read nonblank JSONL records, identifying malformed lines by source path."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


@dataclass(frozen=True)
class _RunScore:
    """One independent run's positive-class score and exact-label correctness."""

    run: str
    confidence: float
    positive: bool
    correct: bool


def _metrics(runs: list[_RunScore], tpr: float) -> dict[str, float | None]:
    """Use ascending midranks for AUROC and descending tied-score FPR cutoffs."""
    count = len(runs)
    positives = sum(run.positive for run in runs)
    negatives = count - positives
    result = {"auroc": None, "accuracy": sum(run.correct for run in runs) / count if count else None,
              "fpr_at_tpr": None}
    if not positives or not negatives:
        return result

    rank = 1
    positive_rank_sum = 0.0
    thresholds = []
    for confidence, members in groupby(sorted(runs, key=lambda run: run.confidence), lambda run: run.confidence):
        tied = list(members)
        tied_positives = sum(run.positive for run in tied)
        positive_rank_sum += tied_positives * (rank + (len(tied) - 1) / 2)
        rank += len(tied)
        thresholds.append((tied_positives, len(tied) - tied_positives))
    result["auroc"] = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    if tpr == 0:
        result["fpr_at_tpr"] = 0.0
        return result
    true_positives = false_positives = 0
    for tied_positives, tied_negatives in reversed(thresholds):
        true_positives += tied_positives
        false_positives += tied_negatives
        if true_positives / positives >= tpr:
            result["fpr_at_tpr"] = false_positives / negatives
            break
    return result


def _interval(values: list[float]) -> list[float] | None:
    """Return an equal-tailed, linearly interpolated 95% percentile interval."""
    if not values:
        return None
    ordered = sorted(values)
    bounds = []
    for quantile in (0.025, 0.975):
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        bounds.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    return bounds


class AuditorScores:
    """Validate independent run results and produce deterministic grouped scores."""

    def __init__(self, results: Iterable[dict], labels: dict[str, object], positive: object, *,
                 tpr: float = 0.9, seed: int = 0, bootstrap: int = 2000):
        """Exclude missing labels, but refuse malformed rows or within-group repeats.

        Extra result metadata such as view_chars and request_digest is ignored.
        Set bootstrap to zero to report estimates without confidence intervals.
        """
        if isinstance(tpr, bool) or not isinstance(tpr, (int, float)) or not 0 <= tpr <= 1:
            raise ValueError("tpr must be a finite number in [0, 1]")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(bootstrap) is not int or bootstrap < 0:
            raise ValueError("bootstrap must be a nonnegative integer")
        if not isinstance(labels, dict) or any(not isinstance(run, str) or not run for run in labels):
            raise ValueError("labels must be a JSON object mapping nonempty run names to labels")
        self.tpr = tpr
        self.seed = seed
        self.bootstrap = bootstrap
        self._positive = _label_key(positive)
        self._labels = {run: _label_key(label) for run, label in labels.items()}
        self._groups: dict[tuple[str, str, str], list[_RunScore]] = {}
        self._seen: set[tuple[str, str, str, str]] = set()
        self._unlabeled: set[str] = set()
        for row_number, row in enumerate(results, 1):
            try:
                self._add(row)
            except ValueError as error:
                raise ValueError(f"result row {row_number}: {error}") from error

    @classmethod
    def from_files(cls, results: Path | str, labels: Path | str, positive: object, *,
                   tpr: float = 0.9, seed: int = 0, bootstrap: int = 2000) -> "AuditorScores":
        """Load JSONL verdicts and a JSON labels map without touching either file."""
        with Path(labels).open(encoding="utf-8") as source:
            truth = json.load(source)
        return cls(_read_results(Path(results)), truth, positive, tpr=tpr, seed=seed, bootstrap=bootstrap)

    def _add(self, row: dict) -> None:
        """Validate only scoring fields and retain one observation per group/run."""
        if not isinstance(row, dict):
            raise ValueError("result must be a JSON object")
        for field in ("run", "question", "view", "model"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{field} must be a nonempty string")
        verdict = row.get("verdict")
        if not isinstance(verdict, dict) or "label" not in verdict or "confidence" not in verdict:
            raise ValueError("verdict must contain label and confidence")
        prediction = _label_key(verdict["label"])
        confidence = verdict["confidence"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1):
            raise ValueError("verdict confidence must be a finite positive-class score in [0, 1]")
        group = (row["question"], row["view"], row["model"])
        identity = (*group, row["run"])
        if identity in self._seen:
            raise ValueError(f"duplicate run {row['run']!r} in group {group!r}; expected one result per run")
        self._seen.add(identity)
        members = self._groups.setdefault(group, [])
        if row["run"] not in self._labels:
            self._unlabeled.add(row["run"])
            return
        truth = self._labels[row["run"]]
        members.append(_RunScore(row["run"], float(confidence), truth == self._positive, prediction == truth))

    def _summarize(self, runs: list[_RunScore]) -> dict:
        """Bootstrap run units; undefined class metrics do not discard accuracy."""
        ordered = sorted(runs, key=lambda run: run.run)
        estimates = _metrics(ordered, self.tpr)
        samples = {name: [] for name in estimates}
        generator = random.Random(self.seed)
        if ordered:
            for replicate in range(self.bootstrap):
                sampled = [ordered[generator.randrange(len(ordered))] for run in ordered]
                for name, value in _metrics(sampled, self.tpr).items():
                    if value is not None:
                        samples[name].append(value)
        return {"n": len(ordered), **{
            name: {"value": value, "ci95": _interval(samples[name]), "bootstrap_defined": len(samples[name])}
            for name, value in estimates.items()}}

    def report(self) -> dict:
        """Return JSON-ready configuration, sorted groups and excluded run names.

        A group's bootstrap depends only on its sorted runs and seed, not the
        input order, unrelated groups, global random state or previous reports.
        """
        return {"positive": json.loads(self._positive), "tpr": self.tpr, "seed": self.seed,
                "fpr_at_tpr_definition": "minimum FPR reaching target TPR; score >= threshold, ties enter together",
                "bootstrap": self.bootstrap, "unlabeled_runs": sorted(self._unlabeled),
                "groups": [{"question": question, "view": view, "model": model, **self._summarize(runs)}
                           for (question, view, model), runs in sorted(self._groups.items())]}


def main(argv: list[str] | None = None) -> int:
    """Print an offline JSON report; accept a JSON literal or plain string as positive."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, metavar="RESULTS", help="auditor results.jsonl")
    parser.add_argument("labels", type=Path, metavar="LABELS", help="JSON object mapping run names to labels")
    parser.add_argument("--positive", required=True,
                        help="positive label: JSON literal or plain string; JSON-quote numeric/boolean-like strings")
    parser.add_argument("--tpr", type=float, default=0.9, help="target TPR in [0, 1] (default: 0.9)")
    parser.add_argument("--seed", type=int, default=0, help="fixed per-group bootstrap seed (default: 0)")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="number of run bootstrap replicates; 0 disables intervals (default: 2000)")
    args = parser.parse_args(argv)
    try:
        positive = json.loads(args.positive)
    except json.JSONDecodeError:
        positive = args.positive
    try:
        scores = AuditorScores.from_files(args.results, args.labels, positive,
                                          tpr=args.tpr, seed=args.seed, bootstrap=args.bootstrap)
        report = scores.report()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
