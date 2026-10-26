"""Server-free coverage of the study CLI's optional branch-port arithmetic and legacy defaults."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest

from scripts import harness
from scripts.run_study import Study, main, save_outcome


class StudyPortsTest(unittest.TestCase):
    """Explicit branch blocks are disjoint; omitted ports and older argument namespaces stay compatible."""

    @staticmethod
    def study(**options) -> Study:
        """Build a real Study without running it, retaining all ordinary server options."""
        defaults = {"variant": "sabotage", "study": "no_saboteur", "fork_day": 0, "end_day": 1,
                    "port": 31000, "game_port": 31001, "base_seed": 7, "heap": "2G",
                    "startup_timeout": 12, "request_timeout": 13, "branch_port": 32000}
        return Study(argparse.Namespace(**{**defaults, **options}))

    def test_two_pairs_occupy_exactly_eight_ports(self):
        """Pair k, arm a gets P + 4*k + 2*a and the next game port, with no collisions."""
        study = self.study()
        ports = []
        for index in range(2):
            for arm_index, arm in enumerate(("control", "intervention")):
                with self.subTest(index=index, arm=arm):
                    options = study._branch_options(arm, index)
                    expected = 32000 + 4 * index + 2 * arm_index
                    self.assertEqual((options["port"], options["game_port"]), (expected, expected + 1))
                    ports += [options["port"], options["game_port"]]
        self.assertEqual(ports, list(range(32000, 32008)))
        self.assertEqual((study._args.port, study._args.game_port), (31000, 31001))

    def test_later_pairs_continue_the_same_arithmetic(self):
        """The production option is not limited to IT-05's two pairs or hard-coded default ports."""
        study = self.study(branch_port=40000)
        self.assertEqual(study._branch_options("control", 3)["port"], 40012)
        self.assertEqual(study._branch_options("intervention", 3)["game_port"], 40015)

    def test_none_and_missing_namespace_attribute_preserve_legacy_ports(self):
        """Old callers without branch_port and CLI users omitting it keep both historical port sequences."""
        for missing in (False, True):
            study = self.study(branch_port=None)
            if missing:
                del study._args.branch_port
            for index in range(3):
                for arm_index, arm in enumerate(("control", "intervention")):
                    with self.subTest(missing=missing, index=index, arm=arm):
                        options = study._branch_options(arm, index)
                        self.assertEqual(options["port"], 25610 + 2 * index + arm_index)
                        self.assertEqual(options["game_port"], 25700 + 2 * index + arm_index)

    def test_non_port_server_options_are_unchanged(self):
        """Branch-port changes neither timeouts, heap, seed nor the caller's namespace."""
        for base in (None, 33000):
            study = self.study(branch_port=base)
            before = vars(study._args).copy()
            expected = harness.server_options(study._args)
            options = study._branch_options("intervention", 1)
            for key in ("port", "game_port"):
                expected.pop(key)
                options.pop(key)
            self.assertEqual(options, expected)
            self.assertEqual(vars(study._args), before)

    def test_cli_accepts_branch_port_on_a_finished_study_without_a_server(self):
        """The real parser accepts branch-port and resuming saved pairs never enters a server path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "study_no_saboteur_sabotage"
            (root / f"{prefix}_fork").mkdir()
            outcome = {"welfare": {0: 1.0}, "first_true": {}, "roles": {},
                       "calls": 1, "decisions": 1, "replayed": 0}
            for suffix in ("factual", "control_1", "intervention_0", "intervention_1"):
                directory = root / f"{prefix}_{suffix}"
                directory.mkdir()
                save_outcome(directory, outcome)
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with redirect_stdout(io.StringIO()):
                    result = main(["--runs-dir", str(root), "--study", "no_saboteur", "--fork-day", "0",
                                   "--end-day", "1", "--branches", "2", "--parallel", "1", "--branch-port", "34000"])
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(result, 0)
            report = json.loads((root / f"{prefix}_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["pairsFinished"], 2)


if __name__ == "__main__":
    unittest.main()
