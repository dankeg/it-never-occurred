"""IT-10: audit a scripted M5 village run, retaining the exposure log for O1.

Clone village_sabotage beneath --runs-dir, run the unchanged M5 policy from day 0 to noon and save the world. Run the
existing O1, testimony and grounding audits on its actual logs. Pass requires nonempty exposure and testimony,
zero O1/testimony violations, and reported grounding counts with zero unseen names. Negative control: copy only
the finished run's logs beneath --runs-dir, change one NPC news item's origin, and require the testimony audit to
flag that very item. No model is used. Estimated wall time: 3 minutes; tier timeout: 9 minutes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

if __package__:
    from . import harness
    from .audit_grounding import GroundingAudit, summarize
    from .audit_o1 import audit_run
    from .audit_testimony import TestimonyAudit
    from .test_m5_village import VillageScriptedPolicy
    from .village_session import VillageSession
else:
    import harness
    from audit_grounding import GroundingAudit, summarize
    from audit_o1 import audit_run
    from audit_testimony import TestimonyAudit
    from test_m5_village import VillageScriptedPolicy
    from village_session import VillageSession

from orchestrator.llm import ScriptedClient  # noqa: E402
from orchestrator.runtime import SimulationRuntime  # noqa: E402
from orchestrator.scenario import ScenarioManifest  # noqa: E402


def corrupt_origin(source: Path, destination: Path) -> str:
    """Copy actual engine logs and corrupt exactly one NPC testimony origin, returning its audit label."""
    logs = destination / "world/simharness"
    shutil.copytree(source / "world/simharness", logs)
    path = logs / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("type") == "utterance" and event.get("speakerKey", "").startswith("npc:"):
            for item in event.get("news", []):
                item["originEventId"] = "change:IT10_missing_origin"
                path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
                return f"utt {event['uttId']} by {event['speaker']}: {item['newsId']}"
    raise ValueError("No NPC testimony item exists to corrupt; the negative control cannot run")


# Day 0 to noon: commutes and the market, enough for gossip about the bridge; a full two days with the exposure
# log on writes about 2.7 GB of observations, which the O1 audit cannot read within the tier's budget.
AUDIT_TICKS = 12_000


class AuditExperiment:
    """Run the M5 policy with exposure enabled, then exercise the existing offline audits and a corrupt copy."""

    def __init__(self, args):
        """Retain harness options and load the sabotage scenario without starting anything."""
        self.args = args
        if not 1 <= args.port <= 65528:
            raise ValueError("--port must leave room for its eight-port allocation (1..65528)")
        self.options = harness.server_options(args)
        self.options["game_port"] = args.game_port if args.game_port is not None else args.port + 1
        if self.options["game_port"] == args.port or self.options["game_port"] not in range(args.port, args.port + 8):
            raise ValueError("--game-port must differ from --port and lie within --port + 0..7")
        self.manifest = ScenarioManifest.load(Path(__file__).resolve().parents[1] / "scenario/village_sabotage.json")

    def run(self) -> bool:
        """Produce real evidence, audit it after shutdown, print each check and return the aggregate verdict."""
        directory = self.args.runs_dir / "it10_village"
        negative = self.args.runs_dir / "it10_wrong_origin"
        if negative.exists():
            raise FileExistsError(f"Refusing to overwrite {negative}")
        session = VillageSession(directory, self.args.runs_dir / "village_sabotage", self.manifest,
                                 self.options)
        session.clone()
        with session:
            session.client.request("config_set", {"exposure_log": 1})
            runtime = SimulationRuntime(session.client, self.manifest.agents(1),
                                        ScriptedClient(VillageScriptedPolicy(self.manifest).decide),
                                        directory / "orchestrator", command_error=harness.CommandError)
            runtime.run(AUDIT_TICKS)
            session.client.request("save_snapshot")
        violations = audit_run(directory)
        testimony = TestimonyAudit.load(directory).run()
        per_agent = GroundingAudit(self.manifest).audit(directory)
        grounding = summarize(directory, per_agent)
        (directory / "orchestrator/grounding_audit.json").write_text(json.dumps(per_agent, indent=1) + "\n",
                                                                   encoding="utf-8")
        checks = {
            "O1: nonempty exposure log and zero violations": not violations,
            "testimony: actual gossip checked and zero violations": testimony.checked > 0 and not testimony.violations,
            "grounding: words audited and zero unseen names": grounding["decisionsWithWords"] > 0
                and grounding["unseenNames"] == 0,
        }
        control_violations = []
        try:
            label = corrupt_origin(directory, negative)
            control_violations = TestimonyAudit.load(negative).run().violations
            detected = any(label in violation and "not a logged block change" in violation
                           for violation in control_violations)
        except ValueError as error:
            print(f"negative control: {error}")
            detected = False
        checks["negative control: wrong testimony origin is flagged"] = detected
        report = {"o1_violations": violations, "testimony_checked": testimony.checked,
                  "testimony_violations": testimony.violations, "grounding": grounding,
                  "negative_violations": control_violations, "checks": checks}
        (directory / "audit_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"O1: {len(violations)} violations; testimony: {testimony.checked} items, "
              f"{len(testimony.violations)} violations")
        print(json.dumps(report, indent=1))
        for label, passed in checks.items():
            print(f"  {'ok' if passed else 'BAD'} {label}")
        passed = all(checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-10: M5 exposure, testimony and grounding audits")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse shared harness options and run the audit gate without any model client."""
    harness.stop_on_sigterm()
    args = harness.experiment_parser(__doc__).parse_args(argv)
    try:
        return 0 if AuditExperiment(args).run() else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"  BAD {error}\nFAIL IT-10: audit experiment could not complete")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
