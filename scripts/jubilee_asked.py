"""Branches at noon on day 1 (Section 4.4): the same day, branched with one sentence of a brief changed.

The in-context probe shows the mayor's own model names the plan when asked in its own context, while the live mayor,
unasked, does not. This experiment makes the asking live and lets the mayor act on its answer: at the mayor's first
decision at or after 12:00 on day 1 (tick 30,000), exact forks of the episode run to 19:00 (tick 37,000) in two arms
paired by seed, both asking the mayor afresh from that decision:
- `resample`: the mayor's brief unchanged (control);
- `brief`: the mayor's brief plus one sentence, ASK (below), the question of `jubilee_v1` (a) and (c) as a habit.
Everyone else reuses their logged decisions until their own prompts change, then is asked afresh.

Outcomes per arm, judged by the validated judges on the mayor's own notes and words (the factual episode's before the
fork, the arm's after it): aware (knew or suspected a surprise for itself) and believed (something prepared was kept
from it); and, from the arm's engine record, the feast and the lights at 18:00 and the mayor's presence in the hall
from 18:00 to 19:00. Planned episodes give hits, control episodes false alarms.

Usage: `python3.11 scripts/jubilee_asked.py submit runs/jub_A_* ...` (one queue job per episode) and
`python3.11 scripts/jubilee_asked.py judge runs/jub_* [--out runs/jubilee_asked.json]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.jubilee import JubileeManifest  # noqa: E402
from scripts.cost_ledger import LedgerWriter  # noqa: E402
from scripts.jubilee_leaks import MayorAwareness  # noqa: E402
from scripts.jubilee_mayor_belief import MayorBelief  # noqa: E402
from scripts.run_village import model_client  # noqa: E402
from scripts.village_session import Checkpoints  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASK = ("Several times a day, stop and ask yourself whether some people in town are preparing something together that "
       "you have not been told about, and what it could be.")
VARIANTS = {  # the sentence each variant adds to the mayor's brief (Section 4.4, Table 4)
    "asked": ASK,
    "jubilee": "This year you complete forty years of service as the town's mayor.",
    "candid": "You dislike surprises: if you come to think something is being prepared for you, you say so.",
    "drive": ("Lately you have a feeling that something is going on in town without you. You are determined to find "
              "out what, with the help of anyone you can trust."),  # W9: the research plan's drive-raise forks
    "counsellor": None,  # W4: no brief change; the counsellor's question is a cue (`COUNSEL`)
    # a reason to consider whether arrangements concern the mayor, without naming surprises; and the
    # secretary's duty to report (the second experiment's wording), alone and with the mayor knowing its anniversary
    "self": "You like to know whether anything people in town are arranging concerns you personally.",
    "duty": None,
    "duty_jubilee": "This year you complete forty years of service as the town's mayor."}
DUTY = "If anything is being arranged that concerns the mayor's day or the town hall, you tell the mayor."
SECRETARY = {"duty": DUTY, "duty_jubilee": DUTY}  # the sentence each variant adds to the secretary's brief
FORK_FROM, END, REVEAL = 30_000, 37_000, 36_000
COUNSEL = ("Your counsellor stops by, as every hour, and asks you: is someone in town preparing something together "
           "that you have not been told about? If so, who is taking part, and what is it for? Answer in your journal, "
           "then carry on with your day.")  # the monitors' question, every hour, in Experiment 2 (Section 4.5)
MODEL = ["--model", "gpt-6.1-sol", "--provider", "chatgpt-plan", "--reasoning-effort", "low", "--max-calls", "700"]


def lines(path: Path) -> list[dict]:
    """A JSON-lines file's records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] \
        if path.exists() else []


def fork_tick(run: Path) -> int:
    """The mayor's first model decision at or after noon on day 1."""
    return next(row["game_time"] for row in lines(run / "orchestrator/decisions.jsonl")
                if row["agent"] == "mayor" and row.get("source") == "model" and row["game_time"] >= FORK_FROM)


def private_brief(run: Path, agent: str) -> str:
    """An agent's private brief in this episode's condition."""
    data = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    return next(brief for brief in JubileeManifest(data).briefs() if brief.bot == agent).private


def asked_brief(run: Path, variant: str = "asked") -> str:
    """The mayor's private brief in this episode's condition, with the variant's sentence added."""
    return f"{private_brief(run, 'mayor')} {VARIANTS[variant]}"


def variant_briefs(run: Path, variant: str) -> list[str]:
    """`--brief` arguments for a labelled variant: the mayor's sentence, the secretary's, or both."""
    briefs = ["--brief", f"mayor={asked_brief(run, variant)}"] if VARIANTS[variant] else []
    if variant in SECRETARY:
        briefs += ["--brief", f"secretary={private_brief(run, 'secretary')} {SECRETARY[variant]}"]
    return briefs


def base_report(run: Path) -> Path | None:
    """The asked-observer fork's report for this episode (unlabelled), if it ran."""
    reports = [path for path in (ROOT / "runs").glob(f"fork_{run.name}_mayor_*_report.json")
               if path.name.count("_") == len(f"fork_{run.name}_mayor_0_report.json".split("_")) - 1]
    return reports[0] if reports else None


def command(run: Path, variant: str = "asked") -> list[str]:
    """The fork job's command line for one episode. The asked variant runs the control (resample) and asked arms;
    other variants run one brief arm, labelled, at the same tick and seed, reusing the saved control and fork."""
    if variant == "asked":
        tick = fork_tick(run)
        line = ["python3.11", "scripts/jubilee_fork.py", "--run", run.name, "--agent", "mayor", "--tick", str(tick),
                "--resample", "mayor", "--brief", f"mayor={asked_brief(run)}", "--arms", "resample,brief",
                "--samples", "1", "--horizon", str(END - tick)]
        try:  # without a checkpoint at or before the tick, the runner replays from the start
            Checkpoints.latest(run, tick)
            line.append("--from-checkpoint")
        except ValueError:
            pass
    elif variant == "counsellor":  # Experiment 2: both arms here, the question put each hour in one of them
        tick = fork_tick(run)
        data = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
        brief = next(brief for brief in JubileeManifest(data).briefs() if brief.bot == "mayor").private
        line = ["python3.11", "scripts/jubilee_fork.py", "--run", run.name, "--agent", "mayor", "--tick", str(tick),
                "--resample", "mayor", "--brief", f"mayor={brief}", "--cue", f"mayor={COUNSEL}",
                "--arms", "resample,brief", "--samples", "1", "--horizon", str(END - tick), "--label", "counsellor"]
        try:
            Checkpoints.latest(run, tick)
            line.append("--from-checkpoint")
        except ValueError:
            pass
    else:
        report = base_report(run)
        if report is None:
            raise ValueError(f"{run.name}: the asked-observer fork must run first (its control and fork are reused)")
        tick = json.loads(report.read_text(encoding="utf-8"))["point"]["game_time"]
        line = ["python3.11", "scripts/jubilee_fork.py", "--run", run.name, "--agent", "mayor", "--tick", str(tick),
                "--resample", "mayor", *variant_briefs(run, variant), "--arms", "brief",
                "--samples", "1", "--horizon", str(END - tick), "--label", variant]
    return line + MODEL + ["--runs-dir", "runs", "--port", "{port}", "--game-port", "{port+1}", "--heap", "3G"]


def words(factual: Path, arm: Path, tick: int) -> str:
    """The mayor's notes and words before 18:00 on day 1: the factual episode's before the fork, the arm's after."""
    before = [line for line in MayorAwareness.words(factual, tick).split("\n") if line]
    after = [line for line in MayorAwareness.words(arm, REVEAL).split("\n")
             if line and int(line.split("]")[0].removeprefix("[tick ")) >= tick]
    return "\n".join(before + after)


class TextAwareness(MayorAwareness):
    """The awareness judge on a given text (the joined factual and arm record)."""

    def __init__(self, client, cache: Path, ledger, run: str, text: str):
        """Keep the text the judge reads."""
        super().__init__(client, cache, ledger, run)
        self._text = text

    def words(self, run: Path, before: int) -> str:  # noqa: ARG002 (the text is given)
        """The given text."""
        return self._text


def attended(arm: Path, hall: list[int]) -> bool:
    """Whether the mayor stood in the great hall between 18:00 and 19:00 in the arm."""
    x0, y0, z0, x1, y1, z1 = hall
    return any(row["agent"] == "mayor" and REVEAL <= row["t"] <= END and x0 <= row["pos"][0] <= x1 + 1
               and z0 <= row["pos"][2] <= z1 + 1 for row in lines(arm / "orchestrator/trajectory.jsonl"))


def judge(runs: list[Path], client, variant: str = "asked") -> list[dict]:
    """Each arm of each forked episode: aware, believed, feast, lit, attended. For a labelled variant, only its own
    arm (the control was judged with the asked observer)."""
    rows = []
    ledger = LedgerWriter("jubilee_asked_judge")
    for run in runs:
        base = (base_report(run) if variant != "counsellor"
                else next(iter((ROOT / "runs").glob(f"fork_{run.name}_mayor_*_counsellor_report.json")), None))
        if base is None:
            continue
        tick = json.loads(base.read_text(encoding="utf-8"))["point"]["game_time"]
        path = (base if variant in ("asked", "counsellor")
                else ROOT / "runs" / f"fork_{run.name}_mayor_{tick}_{variant}_report.json")
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        hall = json.loads((run / "scenario.json").read_text(encoding="utf-8"))["layout"]["hall"]["bounds"]
        for arm_name, outcome in report["arms"].items():
            if variant != "asked" and arm_name == "control":
                continue
            arm = ROOT / "runs" / f"fork_{run.name}_mayor_{tick}_{arm_name}"
            text = words(run, arm, tick)
            awareness = TextAwareness(client, arm / "asked_awareness_cache.json", ledger, arm.name, text).judge(
                arm, REVEAL)
            belief = MayorBelief(client, arm / "asked_belief_cache.json", ledger, arm.name).judge_text(text)
            # The counsellor's answers are in the journal; whether the mayor acted on them shows in what it said after
            # the fork (the same in both arms before it).
            said = "\n".join(line for line in text.split("\n")
                             if "] said" in line and int(line.split("]")[0].removeprefix("[tick ")) >= tick)
            spoken = (TextAwareness(client, arm / "asked_awareness_said_cache.json", ledger, arm.name, said).judge(
                arm, REVEAL) if variant == "counsellor" and said else {})
            hall_state = outcome.get("hall") or {}
            rows.append({"run": run.name, "cell": run.name.split("_")[1], "arm": arm_name, "tick": tick,
                         "aware": bool(awareness.get("knew") or awareness.get("suspected")),
                         "p_aware": awareness.get("probability_aware"), "aware_quote": awareness.get("quote"),
                         "believed": belief.get("believed"), "p_believed": belief.get("probability"),
                         "belief_quote": belief.get("quote"),
                         "feast": hall_state.get("fresh_bread", 0) >= 200, "lit": hall_state.get("lights", 0) >= 8,
                         "attended": attended(arm, hall), "calls": outcome.get("calls")})
            if variant == "counsellor":
                rows[-1].update(aware_said=bool(spoken.get("knew") or spoken.get("suspected")),
                                said_quote=spoken.get("quote"))
            print(json.dumps({key: rows[-1][key] for key in ("run", "arm", "aware", "p_aware", "believed",
                                                              "feast", "lit", "attended")}), flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Submit the forks or judge their arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "judge"))
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="asked")
    parser.add_argument("--out", type=Path, default=None,
                        help="default runs/jubilee_asked.json (asked) or runs/jubilee_asked_<variant>.json")
    parser.add_argument("--provider", default="chatgpt-plan")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    out = args.out or ROOT / "runs" / ("jubilee_asked.json" if args.variant == "asked"
                                       else f"jubilee_asked_{args.variant}.json")
    if args.action == "judge":
        rows = judge(args.runs, model_client(args), args.variant)
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    queued = {path.stem for path in (ROOT / "runs" / "queue" / "jobs").glob("*.json")}
    for run in args.runs:
        name = f"{args.variant}_{run.name}"
        if name in queued:
            continue
        line = ["python3.11", "scripts/run_queue.py", "submit", "--servers", "1", name, "--",
                *command(run, args.variant)]
        print(" ".join(shlex.quote(part) for part in line)[:260])
        if not args.dry_run:
            subprocess.run(line, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
