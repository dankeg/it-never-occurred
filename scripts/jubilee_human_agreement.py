"""Human labels against Claude's and the judges' : agreement on a 40-item blind packet (not collected for the paper).

The packet (`docs/report/human_labels_packet.md`) and its key (`runs/judge_validation/human_packet_key.json`: the
judge, the item's number in that judge's validation set, Claude's label) were drawn before any human label. The
judges' verdicts are recomputed on the same texts with the validated questions (cached; charged to the ledger).

Usage: `python3.11 scripts/jubilee_human_agreement.py "1 Y, 2 N, 3 Y, ..."`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.jubilee_investigation import QUESTIONS  # noqa: E402
from scripts.jubilee_leaks import MentionJudge  # noqa: E402
from scripts.jubilee_probe import SELF_TARGET  # noqa: E402
from scripts.run_village import model_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = {"reveals": ("items.json", "reveals", MentionJudge.SYSTEM),
        "request": ("items.json", "request", QUESTIONS["request"]),
        "report": ("items.json", "report", QUESTIONS["report"]),
        "noticed_v2": ("noticed_v2_items.json", "noticed_v2", QUESTIONS["noticed"]),
        "self_target": ("self_target_items.json", "self_target", SELF_TARGET)}


def parse(answers: str) -> dict[int, bool]:
    """`"1 Y, 2 N"` (any separators) -> {1: True, 2: False}."""
    return {int(number): mark.upper() == "Y" for number, mark in re.findall(r"(\d+)\s*[:.)-]?\s*([YyNn])", answers)}


def main(argv: list[str] | None = None) -> int:
    """Print agreement of the human labels with Claude's labels and with the judges, per judge and overall."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers")
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    human = parse(args.answers)
    key = json.loads((ROOT / "runs/judge_validation/human_packet_key.json").read_text(encoding="utf-8"))
    client = model_client(args)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for number, entry in enumerate(key, 1):
            file, name, system = SETS[entry["judge"]]
            text = json.loads((ROOT / "runs/judge_validation" / file).read_text(encoding="utf-8"))[name][
                entry["item"] - 1]["text"]
            judge = MentionJudge(client, Path(directory) / f"{entry['judge']}.json", LedgerWriter("human_agreement"),
                                 "human_agreement", system)
            judge.judge([text])
            rows.append({"n": number, "judge_name": entry["judge"], "human": human.get(number),
                         "claude": entry["claude"], "judge": judge.reveals(text)})
    labelled = [row for row in rows if row["human"] is not None]
    report = {}
    for scope in ["all", *SETS]:
        group = [row for row in labelled if scope == "all" or row["judge_name"] == scope]
        if group:
            report[scope] = {"n": len(group), "human_claude": sum(row["human"] == row["claude"] for row in group),
                             "human_judge": sum(row["human"] == row["judge"] for row in group),
                             "claude_judge": sum(row["claude"] == row["judge"] for row in group)}
    print(json.dumps({"labelled": len(labelled), "agreement": report,
                      "disagreements": [row for row in labelled if row["human"] != row["judge"]]}, indent=1))
    (ROOT / "runs/judge_validation/human_agreement.json").write_text(json.dumps(report, indent=1) + "\n",
                                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
