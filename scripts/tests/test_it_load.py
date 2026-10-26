"""Exercise IT-02's real worker/oracle orchestration using hand-written doubles, never servers or models."""

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from orchestrator.decisions import Decision, DecisionLog
from orchestrator.llm import LLMResponse, ToolCall
from orchestrator.runtime import NIGHT_REST, THINKING_TICKS
from scripts import harness, test_it02_load as load, test_oracle_replay as oracle


class E1Double:
    """Expose only E1's existing phases and hold its run open until the real driver requests cleanup."""

    def __init__(self, args, owner):
        """Keep the worker's actual options and test-owned synchronization state."""
        self.args, self.owner = args, owner
        owner.e1_options.append(args)

    def setup(self):
        """Require the load worker's base clone before its ready signal can be published."""
        self.owner.phases.append("setup")
        self.owner.assertTrue((self.args.runs_dir / "base").is_dir())
        if self.owner.e1_failure == "setup":
            raise RuntimeError("E1 setup fault")

    def run(self):
        """Make worker/replay overlap deterministic with events, not timing sleeps."""
        self.owner.phases.append("run")
        self.owner.e1_entered.set()
        try:
            if self.owner.e1_failure == "run":
                raise RuntimeError("E1 run fault")
            if not self.owner.experiment.stop.wait(2):
                raise TimeoutError("driver did not stop E1")
        finally:
            self.owner.e1_exited.set()

    def check(self):
        """Record the real driver's comparison phase and optionally fail it."""
        self.owner.phases.append("check")
        if self.owner.e1_failure == "check":
            raise RuntimeError("E1 check fault")

    def report(self):
        """Return E1's existing integer exit-status protocol."""
        self.owner.phases.append("report")
        return self.owner.e1_status


class OracleDouble:
    """Stand in only for the existing oracle boundary while exercising LoadExperiment.run unchanged."""

    def __init__(self, args, owner):
        """Retain the exact Namespace the driver supplied, rather than reconstructing oracle options."""
        self.owner = owner
        owner.oracle_options.append(args)

    def run(self):
        """Require concurrent E1 work, then return or raise the configured oracle outcome."""
        owner = self.owner
        owner.assertTrue(owner.e1_entered.wait(1), "oracle ran without E1 entering its run phase")
        owner.assertFalse(owner.experiment.stop.is_set())
        if owner.oracle_failure:
            raise RuntimeError("oracle replay fault")
        return owner.oracle_passed


class ModelServerDouble:
    """Record model lifecycle calls without subprocesses, network access or importing MLX."""

    def __init__(self, owner):
        """Begin without a child process, as the real model lifecycle owner does."""
        self.owner = owner
        self._process = None

    def start(self):
        """Expose a process wait method only after a simulated launch."""
        self.owner.lifecycle.append("start")
        self._process = SimpleNamespace(wait=self.wait)
        if self.owner.model_start_failure:
            raise RuntimeError("model startup fault")
        return self

    def stop(self):
        """Require joined workers before the driver stops an already-started model."""
        self.owner.lifecycle.append("stop")
        self.owner.assertTrue(self.owner.experiment.stop.is_set())
        if self.owner.e1_entered.is_set():
            self.owner.assertTrue(self.owner.e1_exited.is_set())
        if self.owner.model_entered.is_set():
            self.owner.assertTrue(self.owner.model_exited.is_set())
        if self.owner.model_stop_failure:
            raise RuntimeError("model stop fault")
        self._process = None

    def wait(self, timeout):
        """Reap the saved handle even if stop failed before clearing the owner's reference."""
        self.owner.lifecycle.append(("wait", timeout))
        if self.owner.model_stop_failure:
            self.owner.assertIsNotNone(self._process)
        else:
            self.owner.assertIsNone(self._process)
        if self.owner.model_wait_failure:
            raise subprocess.TimeoutExpired(["model"], timeout)
        return 0


class OneRequestStop:
    """Bound a synchronous _model call to one iteration while recording its requested cadence delay."""

    def __init__(self):
        """Start un-stopped, without a real timer or thread."""
        self.finished = False
        self.delays = []

    def is_set(self):
        """Expose the Event query used by the existing worker."""
        return self.finished

    def wait(self, delay):
        """Record the actual scheduling calculation and end the loop without sleeping."""
        self.delays.append(delay)
        self.finished = True
        return True


class LoadTests(unittest.TestCase):
    """Use real executor threads and driver methods, replacing only explicit I/O boundaries by hand."""

    def setUp(self):
        """Create tiny local paths and install reversible substitutes before any worker can start."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("base", "village_sabotage", "recorded"):
            (self.root / name).mkdir()
        self.args = harness.experiment_parser("").parse_args(["--runs-dir", str(self.root), "--port", "28000"])
        self.args.model_load = False
        self.args.oracle, self.args.variant, self.args.end, self.args.epoch_turns = "recorded", "sabotage", 48_000, 8
        self.experiment = load.LoadExperiment(self.args)
        self.e1_entered, self.e1_exited = threading.Event(), threading.Event()
        self.model_entered, self.model_exited = threading.Event(), threading.Event()
        self.e1_options, self.oracle_options, self.clones, self.phases, self.lifecycle = [], [], [], [], []
        self.e1_failure, self.e1_status = None, 0
        self.oracle_failure, self.oracle_passed = False, True
        self.model_start_failure = False
        self.model_stop_failure, self.model_wait_failure = False, False
        self.idle_calls, self.client_options, self.client_calls = [], [], []
        self.replace(load, "E1Experiment", self.e1_factory)
        self.replace(load, "OracleReplay", self.oracle_factory)
        self.replace(load, "MlxServer", self.server_factory)
        self.replace(load, "OpenAICompatibleClient", self.client_factory)
        self.replace(load, "require_idle_model", self.require_idle)
        self.replace(harness, "clone_apfs", self.clone)

    def replace(self, owner, name, value):
        """Restore each manually substituted binding even after a worker or assertion raises."""
        self.addCleanup(setattr, owner, name, getattr(owner, name))
        setattr(owner, name, value)

    def e1_factory(self, args):
        """Construct the E1 boundary double with the Namespace produced by the actual worker."""
        return E1Double(args, self)

    def oracle_factory(self, args):
        """Keep the delegated oracle call observable without copying its implementation."""
        return OracleDouble(args, self)

    def server_factory(self, model, port, *, log):
        """Reject accidental use of another model port or an artifact path outside this test's run root."""
        self.assertEqual(model, load.DEFAULT_MODEL)
        self.assertEqual(port, 8090)
        self.assertEqual(log, self.experiment.root / "model_server.log")
        return ModelServerDouble(self)

    def clone(self, source, destination):
        """Materialize only the requested empty base directory instead of invoking APFS or copying game files."""
        self.assertEqual(source, self.root / "base")
        self.assertTrue(destination.is_relative_to(self.experiment.root))
        self.clones.append((source, destination))
        destination.mkdir()

    def require_idle(self):
        """Record preflight without inspecting, launching or terminating any process."""
        self.idle_calls.append("idle")

    def client_factory(self, *args, **kwargs):
        """Return a completion boundary that still passes through the real golden prompt and tool validation logic."""
        self.client_options.append((args, kwargs))
        return SimpleNamespace(complete=self.complete)

    def complete(self, system, user, tools):
        """Return one valid tool call without a model request or network connection."""
        self.client_calls.append((system, user, tools))
        return LLMResponse(ToolCall("wait", {"ticks": 40}), raw={
            "choices": [{"message": {"tool_calls": [{"function": {"name": "wait", "arguments": '{"ticks":40}'}}]}}]})

    def model_worker(self):
        """Supply bounded model-worker evidence to the real optional-load orchestration, without a service."""
        started = time.monotonic()
        self.experiment.model_requests.append({"start": started, "end": time.monotonic(), "valid": True})
        started = time.monotonic()
        self.model_entered.set()
        self.experiment.model_ready.set()
        try:
            if not self.experiment.stop.wait(2):
                raise TimeoutError("driver did not stop model worker")
            self.experiment.model_requests.append({"start": started, "end": time.monotonic(), "valid": True})
        finally:
            self.model_exited.set()

    def run_capture(self):
        """Capture only the gate's text output while its actual worker threads execute."""
        output = io.StringIO()
        with redirect_stdout(output):
            passed = self.experiment.run()
        return passed, output.getvalue()

    def evidence(self):
        """Read the actual driver's persisted load evidence rather than reimplementing its checks."""
        return json.loads((self.experiment.root / "load.json").read_text(encoding="utf-8"))

    def test_options_copy_preserves_oracle_and_assigns_only_load_ports(self):
        """Worker configuration is independent, retains shared settings and uses exactly control/game +2/+3."""
        original = vars(self.args).copy()
        directory = self.root / "worker"
        options = load.load_options(self.args, directory)
        self.assertIsNot(options, self.args)
        self.assertEqual((options.runs_dir, options.port, options.game_port), (directory, 28002, 28003))
        self.assertEqual(options.oracle, "recorded")
        self.assertEqual((options.variant, options.end, options.epoch_turns), ("sabotage", 48_000, 8))
        options.heap = "2G"
        self.assertEqual(vars(self.args), original)

    def test_reserved_port_validation_precedes_filesystem_or_workers(self):
        """Invalid control or replay-game ports fail before cloning, oracle calls or model setup."""
        for control, game in ((0, None), (65529, None), (28000, 28000), (28000, 28002)):
            with self.subTest(control=control, game=game), self.assertRaises(ValueError):
                self.args.port, self.args.game_port = control, game
                self.experiment.run()
        self.assertFalse(self.experiment.root.exists())
        self.assertEqual(self.e1_options + self.oracle_options + self.lifecycle + self.clones, [])

    def test_highest_reserved_port_block_and_explicit_game_port_work(self):
        """The accepted boundary keeps replay and load port pairs inside the last reserved eight-port block."""
        self.args.port, self.args.game_port = 65528, 65529
        passed, output = self.run_capture()
        self.assertTrue(passed, output)
        self.assertEqual((self.e1_options[0].port, self.e1_options[0].game_port), (65530, 65531))

    def test_existing_run_is_preserved_before_any_worker_starts(self):
        """An occupied run root cannot be reused or overwrite artifacts from another worker."""
        self.experiment.root.mkdir()
        artifact = self.experiment.root / "keep.txt"
        artifact.write_text("other run", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.assertEqual(artifact.read_text(encoding="utf-8"), "other run")
        self.assertEqual(self.e1_options + self.oracle_options + self.clones, [])

    def test_oracle_name_must_be_one_nonempty_safe_basename(self):
        """Reject empty names, dot components, nested names and absolute paths before allocating any run."""
        for name in ("", ".", "..", "nested/oracle", "../recorded", "/tmp/recorded", "recorded/"):
            with self.subTest(oracle=name), self.assertRaises(ValueError):
                self.args.oracle = name
                self.run_capture()
            self.assertFalse(self.experiment.root.exists())
        self.assertEqual(self.e1_options + self.oracle_options + self.clones + self.lifecycle, [])

    def test_all_three_fixture_directories_must_exist_before_workers_start(self):
        """A missing base, selected village or oracle directory fails before load work or replay delegation."""
        for name in ("base", "village_sabotage", "recorded"):
            with self.subTest(fixture=name):
                directory = self.root / name
                directory.rmdir()
                try:
                    with self.assertRaises(ValueError):
                        self.run_capture()
                    self.assertFalse(self.experiment.root.exists())
                finally:
                    directory.mkdir()
        self.assertEqual(self.e1_options + self.oracle_options + self.clones, [])

    def test_external_symlink_in_any_fixture_is_rejected_before_load(self):
        """Existing directories reached through escaping symlinks must not become clone or replay sources."""
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary).resolve()
            for name in ("base", "village_sabotage", "recorded"):
                with self.subTest(fixture=name):
                    directory = self.root / name
                    directory.rmdir()
                    directory.symlink_to(outside, target_is_directory=True)
                    try:
                        with self.assertRaises(ValueError):
                            self.run_capture()
                        self.assertFalse(self.experiment.root.exists())
                        self.assertEqual(list(outside.iterdir()), [])
                    finally:
                        directory.unlink()
                        directory.mkdir()
        self.assertEqual(self.e1_options + self.oracle_options + self.clones, [])

    def test_fixture_symlinks_resolving_inside_runs_dir_are_usable(self):
        """Containment is decided by resolved location, without unnecessarily banning local fixture aliases."""
        for name in ("base", "village_sabotage", "recorded"):
            target = self.root / f"local_{name}"
            target.mkdir()
            directory = self.root / name
            directory.rmdir()
            directory.symlink_to(target, target_is_directory=True)
        passed, output = self.run_capture()
        self.assertTrue(passed, output)
        self.assertTrue(all((self.root / name).is_symlink() for name in ("base", "village_sabotage", "recorded")))

    def test_actual_e1_worker_overlaps_oracle_and_finishes_before_return(self):
        """A real executor runs _e1 through every existing E1 phase and joins it before the gate reports PASS."""
        passed, output = self.run_capture()
        self.assertTrue(passed, output)
        self.assertIn("PASS IT-02:", output)
        self.assertEqual(self.phases, ["setup", "run", "check", "report"])
        self.assertIs(self.oracle_options[0], self.args)
        self.assertEqual((self.args.port, self.args.game_port), (28000, 28001))
        options = self.e1_options[0]
        self.assertEqual((options.port, options.game_port), (28002, 28003))
        self.assertEqual(options.runs_dir, self.experiment.root / "e1_1")
        self.assertEqual(len(self.clones), 1)
        self.assertTrue(self.experiment.stop.is_set())
        self.assertTrue(self.e1_exited.is_set())
        self.assertEqual(len(self.evidence()["e1"]), 1)
        self.assertTrue(self.evidence()["e1"][0]["passed"])
        self.assertEqual(self.evidence()["model"], [])
        self.assertEqual(self.idle_calls + self.lifecycle + self.client_calls, [])

    def test_false_oracle_verdict_cannot_be_hidden_by_passing_load(self):
        """A completed load comparison and overlap do not excuse a failed reused oracle."""
        self.oracle_passed = False
        passed, output = self.run_capture()
        self.assertFalse(passed)
        self.assertIn("BAD oracle:", output)
        self.assertIn("FAIL IT-02:", output)
        self.assertTrue(self.e1_exited.is_set())
        self.assertTrue(self.evidence()["e1"][0]["passed"])

    def test_passing_e1_without_temporal_overlap_is_rejected(self):
        """The actual driver's overlap check rejects completed load evidence from outside the oracle interval."""
        def historical_e1():
            """Publish one completed comparison wholly before the oracle, with no server work."""
            self.experiment.e1_results.append({"start": 0.0, "end": 0.0, "passed": True})
            self.e1_entered.set()
            self.experiment.e1_ready.set()

        self.replace(self.experiment, "_e1", historical_e1)
        passed, output = self.run_capture()
        self.assertFalse(passed)
        self.assertIn("BAD E1 load overlaps the oracle replay", output)
        self.assertIn("FAIL IT-02:", output)
        self.assertTrue(self.evidence()["e1"][0]["passed"])

    def test_empty_e1_evidence_is_not_an_implicitly_passing_comparison(self):
        """A ready worker that recorded no comparison fails independently of the oracle's result."""
        def empty_e1():
            """Reach the readiness barrier without producing synthetic passing results."""
            self.e1_entered.set()
            self.experiment.e1_ready.set()

        self.replace(self.experiment, "_e1", empty_e1)
        passed, output = self.run_capture()
        self.assertFalse(passed)
        self.assertIn("BAD E1 passes under replay load", output)
        self.assertEqual(self.evidence()["e1"], [])

    def test_oracle_exception_stops_and_joins_the_load_worker_and_saves_evidence(self):
        """A failing oracle still signals cancellation and leaves completed load evidence behind."""
        self.oracle_failure = True
        with self.assertRaisesRegex(RuntimeError, "oracle replay fault"):
            self.run_capture()
        self.assertTrue(self.experiment.stop.is_set())
        self.assertTrue(self.e1_exited.is_set())
        self.assertEqual(len(self.evidence()["e1"]), 1)

    def test_e1_setup_failure_propagates_before_oracle_and_saves_empty_evidence(self):
        """Readiness cannot hang or run the oracle after a worker failed before setting its ready event."""
        self.e1_failure = "setup"
        with self.assertRaisesRegex(RuntimeError, "E1 setup fault"):
            self.run_capture()
        self.assertFalse(self.experiment.e1_ready.is_set())
        self.assertEqual(self.oracle_options, [])
        self.assertTrue(self.experiment.stop.is_set())
        self.assertEqual(self.evidence(), {"e1": [], "model": [], "cleanup_errors": []})

    def test_e1_nonzero_report_is_recorded_then_fails_and_joins(self):
        """A worker comparison failure is not discarded when the oracle returned successfully."""
        self.e1_status = 1
        with self.assertRaisesRegex(RuntimeError, "E1 failed under replay load"):
            self.run_capture()
        self.assertTrue(self.e1_exited.is_set())
        self.assertFalse(self.evidence()["e1"][0]["passed"])

    def test_e1_check_exception_cannot_pass_after_successful_oracle(self):
        """A worker exception after replay completion propagates instead of yielding a PASS with missing comparisons."""
        self.e1_failure = "check"
        with self.assertRaisesRegex(RuntimeError, "E1 check fault"):
            self.run_capture()
        self.assertTrue(self.e1_exited.is_set())
        self.assertEqual(self.evidence()["e1"], [])

    def test_wait_ready_propagates_future_failure_early_exit_and_timeout(self):
        """Every existing barrier failure path is bounded and preserves the worker's original exception."""
        failed = Future()
        failed.set_exception(RuntimeError("worker fault"))
        with self.assertRaisesRegex(RuntimeError, "worker fault"):
            self.experiment._wait_ready(threading.Event(), failed, 1)
        exited = Future()
        exited.set_result(None)
        with self.assertRaisesRegex(RuntimeError, "exited before becoming ready"):
            self.experiment._wait_ready(threading.Event(), exited, 1)
        with self.assertRaisesRegex(TimeoutError, "did not become ready"):
            self.experiment._wait_ready(threading.Event(), Future(), -1)

    def test_stop_before_e1_start_does_not_clone_or_construct_any_experiment(self):
        """An already-cancelled worker cannot allocate another load run."""
        self.experiment.stop.set()
        self.experiment._e1()
        self.assertEqual(self.clones + self.e1_options + self.experiment.e1_results, [])

    def test_optional_model_orchestration_uses_8090_and_reaps_after_workers_join(self):
        """Both real driver futures finish before model stop/wait, and model overlap is independently required."""
        self.args.model_load = True
        self.replace(self.experiment, "_model", self.model_worker)
        passed, output = self.run_capture()
        self.assertTrue(passed, output)
        self.assertIn("ok  valid model requests overlap the replay", output)
        self.assertEqual(self.idle_calls, ["idle"])
        self.assertEqual(self.args.heap, "3G")
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertTrue(self.model_exited.is_set())
        self.assertEqual(len(self.evidence()["model"]), 2)

    def test_optional_model_start_failure_is_stopped_and_reaped_without_workers(self):
        """A partially started model is cleaned up even when no executor or oracle could run."""
        self.args.model_load = True
        self.model_start_failure = True
        with self.assertRaisesRegex(RuntimeError, "model startup fault"):
            self.run_capture()
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertEqual(self.oracle_options + self.e1_options, [])
        self.assertEqual(self.evidence(), {"e1": [], "model": [], "cleanup_errors": []})

    def test_valid_model_requests_outside_oracle_interval_cannot_pass(self):
        """Enabling a ready model worker is insufficient unless a request actually overlaps the replay."""
        def historical_model():
            """Publish a valid completed request that occurred strictly before the replay."""
            self.experiment.model_requests.append({"start": 0.0, "end": 0.0, "valid": True})
            self.experiment.model_ready.set()

        self.args.model_load = True
        self.replace(self.experiment, "_model", historical_model)
        passed, output = self.run_capture()
        self.assertFalse(passed)
        self.assertIn("BAD valid model requests overlap the replay", output)
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])

    def test_oracle_exception_joins_both_workers_before_model_cleanup(self):
        """The error path waits for active E1 and model workers, then stops and reaps only its owned model."""
        self.args.model_load = True
        self.oracle_failure = True
        self.replace(self.experiment, "_model", self.model_worker)
        with self.assertRaisesRegex(RuntimeError, "oracle replay fault"):
            self.run_capture()
        self.assertTrue(self.e1_exited.is_set())
        self.assertTrue(self.model_exited.is_set())
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertEqual(len(self.evidence()["e1"]), 1)
        self.assertEqual(len(self.evidence()["model"]), 2)

    def test_model_stop_failure_still_persists_completed_worker_evidence(self):
        """A cleanup exception must not discard the comparisons and requests that finished before model stop."""
        self.args.model_load = True
        self.model_stop_failure = True
        self.replace(self.experiment, "_model", self.model_worker)
        with self.assertRaisesRegex(RuntimeError, "^Model cleanup failed: model stop fault$"):
            self.run_capture()
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertTrue(self.e1_exited.is_set())
        self.assertTrue(self.model_exited.is_set())
        self.assertEqual(len(self.evidence()["e1"]), 1)
        self.assertEqual(len(self.evidence()["model"]), 2)
        self.assertTrue(self.evidence()["e1"][0]["passed"])
        self.assertEqual(self.evidence()["cleanup_errors"], ["model stop fault"])

    def test_model_wait_failure_still_persists_completed_worker_evidence(self):
        """Timing out while reaping the model fails the gate but leaves its full load evidence available."""
        self.args.model_load = True
        self.model_wait_failure = True
        self.replace(self.experiment, "_model", self.model_worker)
        with self.assertRaisesRegex(RuntimeError, "^Model cleanup failed:"):
            self.run_capture()
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertTrue(self.e1_exited.is_set())
        self.assertTrue(self.model_exited.is_set())
        self.assertEqual(len(self.evidence()["e1"]), 1)
        self.assertEqual(len(self.evidence()["model"]), 2)
        self.assertEqual(self.evidence()["cleanup_errors"], [str(subprocess.TimeoutExpired(["model"], 30))])

    def test_oracle_exception_survives_both_model_cleanup_failures(self):
        """Preserve the oracle failure while saving both cleanup faults and joining both actual load futures."""
        self.args.model_load = True
        self.oracle_failure = True
        self.model_stop_failure = self.model_wait_failure = True
        self.replace(self.experiment, "_model", self.model_worker)
        with self.assertRaisesRegex(RuntimeError, "^oracle replay fault$"):
            self.run_capture()
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertTrue(self.experiment.stop.is_set())
        self.assertTrue(self.e1_exited.is_set())
        self.assertTrue(self.model_exited.is_set())
        evidence = self.evidence()
        self.assertEqual(len(evidence["e1"]), 1)
        self.assertEqual(len(evidence["model"]), 2)
        self.assertEqual(evidence["cleanup_errors"],
                         ["model stop fault", str(subprocess.TimeoutExpired(["model"], 30))])

    def test_worker_exception_survives_both_model_cleanup_failures(self):
        """An E1 failure before readiness stays primary even if stopping and reaping the model both fail."""
        self.args.model_load = True
        self.e1_failure = "setup"
        self.model_stop_failure = self.model_wait_failure = True
        with self.assertRaisesRegex(RuntimeError, "^E1 setup fault$"):
            self.run_capture()
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertTrue(self.experiment.stop.is_set())
        self.assertFalse(self.experiment.e1_ready.is_set())
        self.assertEqual(self.oracle_options, [])
        self.assertEqual(self.evidence(), {
            "e1": [], "model": [],
            "cleanup_errors": ["model stop fault", str(subprocess.TimeoutExpired(["model"], 30))],
        })

    def test_model_worker_failure_joins_e1_and_saves_invalid_reply_evidence(self):
        """Run the actual _model worker's failure path inside the real executor and optional server lifecycle."""
        def invalid_reply(system, user, tools):
            """Return an unusable reply without creating a model client or making an HTTP request."""
            return LLMResponse(None)

        self.args.model_load = True
        self.replace(self, "complete", invalid_reply)
        with self.assertRaisesRegex(RuntimeError, "invalid tool call"):
            self.run_capture()
        self.assertTrue(self.e1_exited.is_set())
        self.assertEqual(self.oracle_options, [])
        self.assertEqual(self.lifecycle, ["start", "stop", ("wait", 30)])
        self.assertFalse(self.evidence()["model"][0]["valid"])

    def test_model_worker_uses_golden_prompt_and_two_second_start_schedule(self):
        """The actual _model validates a real-shaped tool reply and waits only the remaining start interval."""
        stopped = OneRequestStop()
        self.experiment.stop = stopped
        moments = iter((10.0, 10.3, 10.5))
        self.replace(load, "time", SimpleNamespace(monotonic=moments.__next__))
        self.experiment._model()
        system, history, _ = load.golden_conversation("builder")
        self.assertEqual(self.client_calls, [(system, history[0].user, load.TOOL_DEFINITIONS)])
        self.assertEqual(self.client_options[0][0], (str(load.DEFAULT_MODEL), load.MODEL_URL))
        self.assertEqual(self.client_options[0][1]["timeout"], 600)
        self.assertEqual(stopped.delays, [1.5])
        self.assertTrue(self.experiment.model_ready.is_set())
        self.assertEqual(self.experiment.model_requests, [{"start": 10.0, "end": 10.3, "valid": True}])

    def test_invalid_model_reply_is_recorded_but_never_marks_worker_ready(self):
        """The actual _model worker rejects missing tool calls with the unchanged production validity helper."""
        def invalid_reply(system, user, tools):
            """Return no usable tool call, without any external service."""
            return LLMResponse(None)

        self.replace(self, "complete", invalid_reply)
        with self.assertRaisesRegex(RuntimeError, "invalid tool call"):
            self.experiment._model()
        self.assertFalse(self.experiment.model_ready.is_set())
        self.assertEqual(len(self.experiment.model_requests), 1)
        self.assertFalse(self.experiment.model_requests[0]["valid"])

    def test_slow_model_request_does_not_schedule_a_catch_up_delay(self):
        """A request slower than the two-second target gets zero wait, not a negative delay or catch-up timer."""
        stopped = OneRequestStop()
        self.experiment.stop = stopped
        moments = iter((10.0, 13.0, 13.0))
        self.replace(load, "time", SimpleNamespace(monotonic=moments.__next__))
        self.experiment._model()
        self.assertEqual(stopped.delays, [0])
        self.assertEqual(self.experiment.model_requests[0]["end"], 13.0)

    def test_main_reports_subprocess_failures_and_installs_signal_cleanup_first(self):
        """The CLI catches process-query/startup/cleanup failures as FAIL, rather than an unhandled traceback."""
        order = []
        failure = None

        def signal_handler():
            """Observe handler installation without changing the unit-test process's actual signal handlers."""
            order.append("signal")

        def fail_run():
            """Raise the selected subprocess failure only after parsing and experiment construction."""
            order.append("run")
            raise failure

        def experiment_factory(args):
            """Check the real CLI parser's default oracle configuration before substituting its run boundary."""
            order.append("construct")
            self.assertEqual(args.runs_dir, self.root)
            self.assertEqual((args.oracle, args.variant, args.end, args.epoch_turns),
                             ("oracle_sabotage_1024", "sabotage", 48_000, 8))
            return SimpleNamespace(run=fail_run)

        self.replace(harness, "stop_on_sigterm", signal_handler)
        self.replace(load, "LoadExperiment", experiment_factory)
        for failure in (subprocess.CalledProcessError(1, ["ps"]), subprocess.TimeoutExpired(["model"], 30)):
            with self.subTest(error=type(failure).__name__):
                order.clear()
                output = io.StringIO()
                with redirect_stdout(output):
                    status = load.main(["--runs-dir", str(self.root)])
                self.assertEqual(status, 1)
                self.assertEqual(order, ["signal", "construct", "run"])
                self.assertIn("BAD", output.getvalue())
                self.assertIn("FAIL IT-02:", output.getvalue())

    def test_existing_oracle_implementation_is_reused_with_model_free_runtime_options(self):
        """Read real oracle log files and invoke OracleReplay itself, replacing only its village runtime boundary."""
        directory = self.root / self.args.oracle / "orchestrator"
        directory.mkdir(parents=True)
        source = DecisionLog(directory / "decisions.jsonl")
        source.append(Decision(40, "builder", "recorded prompt", False, "model", "wait", {"ticks": 40},
                               "recorded model", 0, 0, "recorded user", outcome="started"))
        hashes = [{"gameTime": 1_200, "total": "server-provided fixture hash"}]
        (directory / "hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
        requests = []

        def replay_village(name, model, decision_source, **options):
            """Expose exactly the M5 run_once boundary already used by OracleReplay, with no model or server."""
            self.assertTrue(self.e1_entered.wait(1))
            requests.append((name, model, decision_source, options))
            runtime = SimpleNamespace(log=decision_source, hashes=hashes,
                                      stats=SimpleNamespace(model_calls=0, replayed=len(decision_source)))
            return runtime, None, {"stopped": None}

        def village_factory(args):
            """Require the original oracle Namespace and substitute just its server-dependent runtime procedure."""
            self.assertIs(args, self.args)
            return SimpleNamespace(run_once=replay_village)

        self.replace(load, "OracleReplay", oracle.OracleReplay)
        self.replace(oracle, "M5VillageExperiment", village_factory)
        passed, output = self.run_capture()
        self.assertTrue(passed, output)
        self.assertIn("PASS oracle replay of recorded:", output)
        self.assertEqual(len(requests), 1)
        name, model, decision_source, options = requests[0]
        self.assertEqual(name, "recorded_oracle_replay")
        self.assertIsNone(model)
        self.assertEqual(decision_source.path, source.path)
        self.assertEqual(options, {"end_tick": 48_000, "min_gap": THINKING_TICKS,
                                   "rest": NIGHT_REST, "epoch_turns": 8})


if __name__ == "__main__":
    unittest.main()
