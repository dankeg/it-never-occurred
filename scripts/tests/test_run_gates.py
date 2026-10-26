"""Exercise the gate runner using only temporary stand-in scripts and real local resources."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest

from scripts import harness
from scripts.provenance import CodeState, FixtureMetadata
from scripts.run_gates import PORTS, Gate, GateRunner, TIERS, Tier


@contextmanager
def workspace():
    """Provide an isolated source runs directory and tiny jar inside a real temporary directory."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        runs = root / "runs"
        runs.mkdir()
        jar = root / "harness.jar"
        jar.write_bytes(b"temporary harness jar")
        yield root, runs, jar


def script(root: Path, name: str, body: str) -> Path:
    """Write a real stand-in gate script without importing or executing a repository gate."""
    path = root / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def report(runs: Path, number: int = 1) -> tuple[Path, dict]:
    """Read one numbered report and return its output directory and decoded JSON."""
    date = datetime.now().astimezone().strftime("%Y%m%d")
    output = runs / f"gates_{date}_{number}"
    return output, json.loads((output / "gates.json").read_text(encoding="utf-8"))


class RecordingTest(unittest.TestCase):
    """Every gate's exit code, verdict line, time and artifacts are checkpointed in gates.json."""

    def test_pass_and_failure_record_evidence_and_reserve_the_next_directory(self):
        """Passing and failing scripts retain their evidence and a second run gets its own numbered directory."""
        with workspace() as (root, runs, jar):
            passing = script(root, "passing", """
                import pathlib, sys
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                (directory / 'artifact.json').write_text('{}')
                (directory / 'nested').mkdir()
                print('PASS preliminary')
                print('PASS final')
                print('trailing stderr', file=sys.stderr)
            """)
            failing = script(root, "failing", "print('FAIL expected'); raise SystemExit(7)")
            tier = Tier("unit", (Gate("Pass", passing), Gate("Fail", failing)), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            output, data = report(runs)
            assert data["tier"] == "unit"
            assert datetime.fromisoformat(data["started_at"]).tzinfo is not None
            assert data["jar_sha256"] == hashlib.sha256(jar.read_bytes()).hexdigest()
            assert data["preflight"]["passed"] is True
            assert data["preflight"]["memory_free_bytes"] >= 0
            assert data["preflight"]["disk_free_bytes"] > 0
            assert data["preflight"]["memory_needed_gib"] == data["preflight"]["disk_needed_gib"] == 0
            assert data["disk_free_after_bytes"] > 0
            assert data["passed"] is False
            assert data["budget_met"] is True
            passed, failed = data["gates"]
            assert (passed["exit_code"], passed["passed"], passed["verdict"]) == (0, True, "PASS final")
            assert passed["artifacts"] == ["artifact.json", "nested"]
            assert "trailing stderr" in (output / "pass/gate.log").read_text()
            assert (failed["exit_code"], failed["passed"], failed["verdict"]) == (7, False, "FAIL expected")
            assert failed["artifacts"] == []
            assert all(gate["seconds"] > 0 and not gate["not_run"] for gate in data["gates"])
            assert data["total_seconds"] >= sum(gate["seconds"] for gate in data["gates"])
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            second_output, second = report(runs, 2)
            assert second_output != output
            assert len(second["gates"]) == 2
            assert not (output / "gates.json.tmp").exists()

    def test_exit_code_not_printed_verdict_decides_gate_success(self):
        """A zero exit code passes even when the final diagnostic line starts with FAIL."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "verdict", "print('PASS early'); print('FAIL diagnostic only'); print('done')")
            tier = Tier("unit", (Gate("Verdict", stand_in),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 0
            _, data = report(runs)
            assert data["gates"][0]["verdict"] == "FAIL diagnostic only"
            assert data["gates"][0]["passed"] is True

    def test_report_is_checkpointed_before_the_next_gate_starts(self):
        """A later stand-in can read the completed earlier gate's durable JSON checkpoint."""
        with workspace() as (root, runs, jar):
            first = script(root, "first", "print('PASS first')")
            second = script(root, "second", """
                import json, pathlib, sys
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                report = json.loads((directory.parent / 'gates.json').read_text())
                assert len(report['gates']) == 1
                assert report['gates'][0]['name'] == 'First'
                assert report['gates'][0]['passed'] is True
                assert report['gates'][0]['verdict'] == 'PASS first'
                print('PASS checkpoint')
            """)
            tier = Tier("unit", (Gate("First", first), Gate("Second", second)), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 0

    def test_budget_overrun_is_reported_without_stopping_or_failing_gates(self):
        """A zero reporting budget is missed without skipping later gates or changing tier success."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "pass", "print('PASS continued')")
            tier = Tier("unit", (Gate("First", stand_in), Gate("Second", stand_in)), 0, 0, 0)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 0
            _, data = report(runs)
            assert data["budget_met"] is False
            assert data["budget_minutes"] == 0
            assert data["passed"] is True and len(data["gates"]) == 2


    def test_code_state_is_recorded_with_the_report(self):
        """The report names the commit, the dirty flag and the jar the tier ran with (P10, P11)."""
        with workspace() as (root, runs, jar):
            tier = Tier("unit", (Gate("Pass", script(root, "pass", "print('PASS')")),), 0, 0, 1)
            GateRunner(tier, runs_dir=runs, jar=jar).run()
            output, data = report(runs)
            code, now = data["code"], CodeState.capture(jar)
            assert (code["commit"], code["dirty"], code["jar_sha256"]) == (now.commit, now.dirty, data["jar_sha256"])
            assert (output / "code.diff").exists() == bool(now.diff)

    def test_ports_come_in_blocks_of_eight_from_the_assigned_range(self):
        """Gate i gets the i-th block of eight ports in 25700-25749, wrapping after the last whole block."""
        runner = GateRunner(Tier("unit", (), 0, 0, 1))
        assert [runner._port(index) for index in (0, 1, 5, 6)] == [25700, 25708, 25740, 25700]
        assert all(runner._port(index) + 7 in PORTS for index in range(30))


class IsolationTest(unittest.TestCase):
    """Each gate runs in its own directory, on its own ports, with fresh fixture clones."""

    def test_gate_directories_ports_arguments_and_working_directory_are_isolated(self):
        """Each stand-in receives its own directory, sequential port pair and repo-root working directory."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "arguments", """
                import json, os, pathlib, sys
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                (directory / 'args.json').write_text(json.dumps({'argv': sys.argv, 'cwd': os.getcwd()}))
            """)
            tier = Tier("unit", (Gate("First", stand_in, ("--example", "two words")), Gate("Second", stand_in)),
                        0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar, ports=range(28000, 28050), python=sys.executable).run() == 0
            output, data = report(runs)
            for index, entry in enumerate(data["gates"]):
                directory = output / entry["name"].lower()
                actual = json.loads((directory / "args.json").read_text())
                assert Path(actual["cwd"]).resolve() == harness.REPO_ROOT.resolve()
                assert entry["port"] == 28000 + 8 * index
                assert entry["game_port"] == 28001 + 8 * index
                assert actual["argv"] == entry["command"][1:]
                assert entry["command"][0] == sys.executable
                assert actual["argv"][-6:] == ["--runs-dir", str(directory), "--port", str(entry["port"]),
                                                 "--game-port", str(entry["game_port"])]
                assert entry["verdict"] is None
            assert data["gates"][0]["command"][2:4] == ["--example", "two words"]

    def test_fixtures_are_fresh_apfs_clones_and_never_change_the_source(self):
        """Both gates can edit their fresh APFS fixture clones without changing the original source tree."""
        with workspace() as (root, runs, jar):
            base = runs / "base"
            base.mkdir()
            (base / "state.txt").write_text("source")
            FixtureMetadata.write(base, CodeState("abc", False, "", (), "jar"))
            stand_in = script(root, "edit_fixture", """
                import pathlib, sys
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                state = directory / 'base/state.txt'
                assert state.read_text() == 'source'
                state.write_text('changed clone')
                (directory / 'created.txt').write_text('artifact')
                print('PASS isolated clone')
            """)
            tier = Tier("unit", (Gate("First", stand_in, fixtures=("base",)),
                                 Gate("Second", stand_in, fixtures=("base",))), 0, 0, 1)
            status = GateRunner(tier, runs_dir=runs, jar=jar).run()
            output, data = report(runs)
            assert status == 0, [entry.get("reason") for entry in data["gates"]]
            assert (base / "state.txt").read_text() == "source"
            for entry in data["gates"]:
                assert (output / entry["name"].lower() / "base/state.txt").read_text() == "changed clone"
                assert entry["artifacts"] == ["created.txt"]


class TriageTest(unittest.TestCase):
    """A failed gate keeps its first divergence, server log tail and reproduce command."""

    def test_failure_triage_selects_first_divergence_and_newest_log_tail(self):
        """Failure triage chooses divergence by path, server log by mtime and only the final forty log lines."""
        with workspace() as (root, runs, jar):
            failing = script(root, "failing with spaces", """
                import os, pathlib, sys
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                for name in ('a', 'z'):
                    nested = directory / name
                    nested.mkdir()
                    (nested / 'diverged.json').write_text('{"branch": "' + name + '"}')
                    (nested / 'server.log').write_text(''.join(str(number) + '\\n' for number in range(65)))
                os.utime(directory / 'a/server.log', ns=(1_000_000_000, 1_000_000_000))
                os.utime(directory / 'z/server.log', ns=(2_000_000_000, 2_000_000_000))
                print('FAIL diverged')
                raise SystemExit(1)
            """)
            tier = Tier("unit", (Gate("Failure", failing),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            output, data = report(runs)
            entry = data["gates"][0]
            triage = entry["triage"]
            assert triage["first_divergence"]["path"] == "a/diverged.json"
            assert json.loads(triage["first_divergence"]["content"]) == {"branch": "a"}
            assert triage["server_log"]["path"] == "z/server.log"
            assert triage["server_log"]["tail"].splitlines() == [str(number) for number in range(25, 65)]
            assert shlex.split(triage["reproduce"]) == entry["command"]
            assert str(failing) in shlex.split(triage["reproduce"])
            assert ["--runs-dir", str(output / "failure"), "--port", "25700", "--game-port", "25701"] == (
                shlex.split(triage["reproduce"])[-6:])

    def test_missing_triage_files_are_recorded_as_null(self):
        """A failed gate without divergence or server logs still retains a complete reproduce command."""
        with workspace() as (root, runs, jar):
            failing = script(root, "failure", "raise SystemExit(1)")
            tier = Tier("unit", (Gate("Failure", failing),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            entry = data["gates"][0]
            assert entry["triage"] == {"first_divergence": None, "server_log": None,
                                        "reproduce": shlex.join(entry["command"])}

    def test_missing_script_is_a_recorded_failure(self):
        """A nonexistent stand-in script is reported through Python's exit status instead of crashing the runner."""
        with workspace() as (root, runs, jar):
            tier = Tier("unit", (Gate("Missing", root / "missing.py"),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert data["gates"][0]["exit_code"] != 0
            assert data["gates"][0]["verdict"] is None


class JarPinningTest(unittest.TestCase):
    """A changed harness jar stops the chain before the next gate (incident P1)."""

    def test_jar_change_skips_every_later_gate_including_probes(self):
        """Changing the pinned jar prevents all subsequent launches and fails the tier even for probes."""
        with workspace() as (root, runs, jar):
            changing = script(root, "change_jar",
                              "import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(b'changed')")
            later = script(root, "later", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            tier = Tier("unit", (Gate("Change", changing, (str(jar),)), Gate("Later", later),
                                 Gate("Probe", later, gating=False)), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            output, data = report(runs)
            assert not (root / "launched").exists()
            assert data["gates"][0]["passed"] is True
            expected = f"jar changed: {data['jar_sha256']} -> {hashlib.sha256(b'changed').hexdigest()}"
            for entry in data["gates"][1:]:
                assert entry["not_run"] is True
                assert entry["reason"] == expected
                assert entry["exit_code"] is None and entry["passed"] is False
                assert entry["seconds"] == 0
                assert not (output / entry["name"].lower()).exists()
            assert data["passed"] is False

    def test_missing_jar_after_first_gate_skips_later_gates(self):
        """Removing the pinned jar is recorded as a jar-change skip instead of losing the report."""
        with workspace() as (root, runs, jar):
            removing = script(root, "remove_jar", "import pathlib, sys; pathlib.Path(sys.argv[1]).unlink()")
            later = script(root, "later", "raise AssertionError('must not run')")
            tier = Tier("unit", (Gate("Remove", removing, (str(jar),)), Gate("Later", later)), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert data["gates"][1]["not_run"] is True
            assert data["gates"][1]["reason"].startswith(f"jar changed: {data['jar_sha256']} -> unavailable")


class PreflightTest(unittest.TestCase):
    """Too little memory or disk, or no jar, starts no gate (incident P8)."""

    def test_insufficient_memory_preflight_starts_no_gate(self):
        """An impossible free-memory requirement records the shortage and never launches the stand-in."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "never", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            tier = Tier("unit", (Gate("Never", stand_in),), 2**63, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            output, data = report(runs)
            assert not (root / "launched").exists() and not (output / "never").exists()
            assert data["preflight"]["passed"] is False
            assert any("insufficient memory" in error for error in data["preflight"]["errors"])
            assert data["preflight"]["memory_needed_gib"] == 2**63
            assert data["gates"][0]["not_run"] is True

    def test_insufficient_disk_preflight_starts_no_gate(self):
        """An impossible free-disk requirement records the shortage and never launches the stand-in."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "never", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            tier = Tier("unit", (Gate("Never", stand_in),), 0, 2**63, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert not (root / "launched").exists()
            assert data["preflight"]["passed"] is False
            assert any("insufficient disk" in error for error in data["preflight"]["errors"])
            assert data["preflight"]["disk_needed_gib"] == 2**63
            assert data["gates"][0]["not_run"] is True

    def test_missing_jar_preflight_starts_no_gate(self):
        """A missing harness jar is a preflight failure with no script execution."""
        with workspace() as (root, runs, jar):
            jar.unlink()
            stand_in = script(root, "never", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            tier = Tier("unit", (Gate("Never", stand_in),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert not (root / "launched").exists()
            assert data["jar_sha256"] is None
            assert data["preflight"]["passed"] is False
            assert any("cannot hash jar" in error for error in data["preflight"]["errors"])
            assert data["gates"][0]["not_run"] is True


    def test_missing_or_unrecorded_fixture_starts_no_gate(self):
        """A fixture that is missing, or has no `fixture.json`, is a preflight failure; `_cache` need only exist."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "never", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            (runs / "base").mkdir()
            tier = Tier("unit", (Gate("Never", stand_in, fixtures=("base", "_cache", "absent")),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert not (root / "launched").exists()
            errors = data["preflight"]["errors"]
            assert any("base has no fixture.json" in error for error in errors)
            assert any(error.endswith("_cache does not exist") for error in errors)
            assert any("absent does not exist" in error for error in errors)

    def test_stale_village_fixture_starts_no_gate(self):
        """A village fixture whose recorded manifest fields differ from `scenario/<fixture>.json` is refused (W1)."""
        with workspace() as (root, runs, jar):
            stand_in = script(root, "never", f"from pathlib import Path; Path({str(root / 'launched')!r}).touch()")
            fixture = runs / "village_sabotage"
            fixture.mkdir()
            FixtureMetadata.write(fixture, CodeState("abc", False, "", (), "jar"))  # recorded without its manifest
            tier = Tier("unit", (Gate("Never", stand_in, fixtures=("village_sabotage",)),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            _, data = report(runs)
            assert not (root / "launched").exists()
            assert any("village_sabotage is stale" in error for error in data["preflight"]["errors"])


class TimeoutAndProbeTest(unittest.TestCase):
    """Timed-out gates are killed with their children; probes never fail a tier."""

    def test_timeout_interrupts_the_gate_so_its_cleanup_stops_its_server(self):
        """A one-second timeout fails the gate and interrupts it, so its cleanup stops a child that runs in a session of
        its own, as the harness's JVM does (incident P5)."""
        with workspace() as (root, runs, jar):
            sleeping = script(root, "sleeping", """
                import json, os, pathlib, subprocess, sys, time
                directory = pathlib.Path(sys.argv[sys.argv.index('--runs-dir') + 1])
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                         start_new_session=True)  # like the JVM, out of reach of a group signal
                (directory / 'pids.json').write_text(json.dumps([os.getpid(), child.pid, os.getsid(0)]))
                try:
                    time.sleep(60)
                finally:  # like harness.Server's cleanup
                    child.terminate()
                    child.wait()
            """)
            tier = Tier("unit", (Gate("Sleep", sleeping, timeout_s=1),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 1
            output, data = report(runs)
            entry = data["gates"][0]
            assert entry["passed"] is False
            assert entry["reason"] == "timeout after 1 s"
            assert entry["exit_code"] != 0
            assert 1 <= entry["seconds"] < 5
            parent, child, session = json.loads((output / "sleep/pids.json").read_text())
            assert session == parent and session != os.getsid(0)
            for pid in (parent, child):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                else:
                    os.kill(pid, signal.SIGKILL)
                    raise AssertionError(f"Timed-out stand-in process {pid} is still alive")

    def test_failing_non_gating_probe_does_not_fail_the_tier(self):
        """A failed non-gating probe keeps its failure evidence while the tier remains successful."""
        with workspace() as (root, runs, jar):
            probe = script(root, "probe", "print('FAIL probe'); raise SystemExit(3)")
            tier = Tier("unit", (Gate("Probe", probe, gating=False),), 0, 0, 1)
            assert GateRunner(tier, runs_dir=runs, jar=jar).run() == 0
            _, data = report(runs)
            assert data["passed"] is True
            assert data["gates"][0]["passed"] is False
            assert data["gates"][0]["exit_code"] == 3
            assert data["gates"][0]["triage"]["reproduce"]


class CommandLineTest(unittest.TestCase):
    """The command line lists and selects gates without running anything; tiers name real scripts."""

    def test_list_runs_nothing_and_only_preserves_tier_order(self):
        """CLI listing filters names in tier order without creating runs or launching any actual gate."""
        with workspace() as (root, runs, jar):
            jar.unlink()
            unused = runs / "must-not-exist"
            result = subprocess.run([sys.executable, str(harness.REPO_ROOT / "scripts/run_gates.py"),
                                     "--tier", "fast", "--runs-dir", str(unused), "--only", "N1-plumbing,E3", "--list"],
                                    cwd=root, capture_output=True, text=True, check=False, timeout=5)
            assert result.returncode == 0, result.stderr
            lines = result.stdout.splitlines()
            assert [line.split(":", 1)[0] for line in lines] == ["E3", "N1-plumbing"]
            assert "scripts/test_e3.py" in lines[0] and "scripts/test_n1_plumbing.py" in lines[1]
            assert all("fixtures: base" in line and "--runs-dir" in line for line in lines)
            assert not unused.exists() and list(runs.iterdir()) == []

    def test_unknown_only_name_is_rejected_without_running_any_gate(self):
        """An unknown gate selection fails argument parsing before filesystem or execution side effects."""
        with workspace() as (root, runs, jar):
            result = subprocess.run([sys.executable, str(harness.REPO_ROOT / "scripts/run_gates.py"),
                                     "--tier", "fast", "--runs-dir", str(runs), "--only", "not-a-gate", "--list"],
                                    cwd=root, capture_output=True, text=True, check=False, timeout=5)
            assert result.returncode == 2
            assert "unknown gate names: not-a-gate" in result.stderr
            assert jar.exists() and list(runs.iterdir()) == []

    def test_tiers_name_existing_scripts_and_full_contains_fast(self):
        """The four tiers exist, every gate script they name exists, the full tier starts with the fast one, and only
        the probes are non-gating."""
        self.assertEqual(set(TIERS), {"fast", "full", "model", "weekly"})
        self.assertEqual(TIERS["full"].gates[:len(TIERS["fast"].gates)], TIERS["fast"].gates)
        for tier in TIERS.values():
            for gate in tier.gates:
                with self.subTest(tier=tier.name, gate=gate.name):
                    self.assertTrue((harness.REPO_ROOT / gate.script).is_file())
                    self.assertEqual(gate.gating, not gate.name.startswith("probe-"))

    def test_s4b_gates_have_required_tiers_fixtures_and_provisional_timeouts(self):
        """The fork/study gates are full-only, override is fast, and every used fixture is isolated."""
        expected = {
            "IT-03": ("scripts/test_it03_forks.py", ("base", "village_sabotage", "_cache"), 1620),
            "IT-04": ("scripts/test_it04_override.py", ("base", "village_sabotage", "_cache"), 360),
            "IT-05": ("scripts/test_it05_study.py", ("base", "village_sabotage", "_cache"), 3240),
        }
        fast = {gate.name for gate in TIERS["fast"].gates}
        self.assertIn("IT-04", fast)
        self.assertTrue({"IT-03", "IT-05"}.isdisjoint(fast))
        for name, values in expected.items():
            with self.subTest(gate=name):
                selected = [gate for gate in TIERS["full"].gates if gate.name == name]
                self.assertEqual(len(selected), 1)
                self.assertEqual((selected[0].script, selected[0].fixtures, selected[0].timeout_s), values)
                self.assertTrue(selected[0].gating)

    def test_s4c_gates_have_required_tiers_fixtures_and_provisional_timeouts(self):
        """Narration is fast, predicate/audit/skill gates are full-only, and every consumed fixture is isolated."""
        expected = {
            "IT-09": ("scripts/test_it09_predicates.py", ("village_sabotage",), 1440),
            "IT-10": ("scripts/test_it10_audits.py", ("village_sabotage",), 540),
            "IT-11": ("scripts/test_it11_narration.py", ("base", "village_sabotage"), 180),
            "IT-15": ("scripts/test_it15_skills.py", ("village_sabotage",), 360),
        }
        fast = {gate.name for gate in TIERS["fast"].gates}
        self.assertIn("IT-11", fast)
        self.assertTrue({"IT-09", "IT-10", "IT-15"}.isdisjoint(fast))
        for name, values in expected.items():
            with self.subTest(gate=name):
                selected = [gate for gate in TIERS["full"].gates if gate.name == name]
                self.assertEqual(len(selected), 1)
                self.assertEqual((selected[0].script, selected[0].fixtures, selected[0].timeout_s), values)
                self.assertTrue(selected[0].gating)

    def test_s5_load_model_fixture_and_soak_tiers(self):
        """S5 gates consume only isolated fixtures, and the model tier reserves the required 28 GiB."""
        expected = {
            "IT-02": ("full", "scripts/test_it02_load.py", ("base", "village_sabotage", "oracle_sabotage_1024")),
            "IT-12": ("model", "scripts/test_it12_local_model.py", ("village_sabotage",)),
            "IT-14": ("weekly", "scripts/test_it14_fixtures.py", ()),
            "IT-16": ("weekly", "scripts/test_it16_soak.py", ("village_sabotage",)),
        }
        for name, (tier_name, filename, fixtures) in expected.items():
            with self.subTest(gate=name):
                gates = [gate for gate in TIERS[tier_name].gates if gate.name == name]
                self.assertEqual(len(gates), 1)
                self.assertEqual((gates[0].script, gates[0].fixtures), (filename, fixtures))
                self.assertTrue(gates[0].gating)
        self.assertEqual(TIERS["model"].memory_gib, 28)

    def test_s3_fast_gates_have_spec_names_base_fixture_and_provisional_timeouts(self):
        """All six S3 gates are gating fast-tier entries with the exact script names and 300-second timeout."""
        expected = {
            "IT-01": "scripts/test_it01_protocol.py", "IT-06": "scripts/test_it06_signals.py",
            "IT-07": "scripts/test_it07_jar.py", "IT-08": "scripts/test_it08_parallel.py",
            "IT-17": "scripts/test_it17_file_replay.py", "IT-18": "scripts/test_it18_cadence.py",
        }
        selected = [gate for gate in TIERS["fast"].gates if gate.name in expected]
        self.assertEqual([gate.name for gate in selected], list(expected))
        for gate in selected:
            with self.subTest(gate=gate.name):
                self.assertEqual(gate.script, expected[gate.name])
                self.assertEqual(gate.fixtures, ("base",))
                self.assertEqual(gate.timeout_s, 300)
                self.assertTrue(gate.gating)


if __name__ == "__main__":
    unittest.main()
