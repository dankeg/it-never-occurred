"""Server-free IT-14 tests: real metadata files, builder wiring, build order and cleanup without mock frameworks."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from scripts import build_village, harness, prepare_world, provenance, test_it14_fixtures as fixtures
from scripts.provenance import CodeState, FixtureMetadata


class FixtureWorkspaceTest(unittest.TestCase):
    """Provide disposable files and reversible attribute substitutions, never starting Minecraft or a model."""

    def setUp(self):
        """Create a tiny jar and shared arguments with all artifacts beneath a temporary run root."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.jar = self.root / "harness.jar"
        self.jar.write_bytes(b"fixture unit-test jar")
        self.state = CodeState("head-commit", False, "", (), provenance.file_sha256(self.jar))
        self.args = harness.experiment_parser("").parse_args([
            "--runs-dir", str(self.root), "--port", "28000", "--game-port", "28001", "--heap", "2G"])
        self.replace(harness, "MOD_JAR", self.jar)

    def replace(self, owner, name, value) -> None:
        """Restore every substituted attribute even when an assertion fails."""
        self.addCleanup(setattr, owner, name, getattr(owner, name))
        setattr(owner, name, value)

    def capture(self, jar, repo=None) -> CodeState:
        """Check the supplied jar exists and return a fixed clean code state for server-free build orchestration."""
        self.assertTrue(Path(jar).is_file())
        return self.state

    def write_metadata(self, directory: Path, manifest: Path | None = None) -> dict:
        """Write genuine fixture metadata with the existing writer and a synthetic harness response digest."""
        directory.mkdir(parents=True, exist_ok=True)
        return FixtureMetadata.write(directory, self.state, manifest=manifest, world_hash="a" * 64)


class FixtureEvidenceTest(FixtureWorkspaceTest):
    """Require complete matching metadata and diagnose stale manifests using the production metadata guard."""

    def test_base_records_code_and_world_without_manifest(self):
        """A complete freshly built base passes every metadata check."""
        directory = self.root / "base"
        self.write_metadata(directory)
        self.assertTrue(all(fixtures.metadata_checks(directory, self.state).values()))

    def test_missing_or_wrong_metadata_evidence_fails(self):
        """Nulls, a wrong commit or jar, dirty/error provenance, and adopted metadata cannot pass."""
        directory = self.root / "base"
        original = self.write_metadata(directory)
        cases = (("commit", None), ("commit", "other-head"), ("jar_sha256", None),
                 ("jar_sha256", "other-jar"), ("dirty", True), ("error", "git failed"))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                metadata = {**original, "code": {**original["code"], field: value}}
                (directory / FixtureMetadata.FILE).write_text(json.dumps(metadata), encoding="utf-8")
                self.assertFalse(all(fixtures.metadata_checks(directory, self.state).values()))
        for world_hash in (None, "", "a" * 63, "g" * 64, 123):
            with self.subTest(world_hash=world_hash):
                (directory / FixtureMetadata.FILE).write_text(
                    json.dumps({**original, "world_hash": world_hash}), encoding="utf-8")
                self.assertFalse(fixtures.metadata_checks(directory, self.state)["world hash recorded"])
        (directory / FixtureMetadata.FILE).write_text(json.dumps({**original, "adopted": True}), encoding="utf-8")
        self.assertFalse(fixtures.metadata_checks(directory, self.state)["built, not adopted"])

    def test_village_checks_manifest_name_and_all_world_fields(self):
        """The manifest reference and every manifest-field digest must match the emitted manifest."""
        self.replace(provenance, "REPO_ROOT", self.root)
        manifest = self.root / "scenario/village_accident.json"
        manifest.parent.mkdir()
        manifest.write_text('{"seed": 1, "layout": {"bell": [1, 2, 3]}}', encoding="utf-8")
        directory = self.root / "village_accident"
        self.write_metadata(directory, manifest)
        self.assertTrue(all(fixtures.metadata_checks(directory, self.state, manifest).values()))
        manifest.write_text('{"seed": 2, "layout": {"bell": [1, 2, 3]}}', encoding="utf-8")
        checks = fixtures.metadata_checks(directory, self.state, manifest)
        self.assertFalse(checks["manifest-field hashes match"])
        self.assertTrue(checks["manifest recorded"])
        path = directory / FixtureMetadata.FILE
        metadata = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**metadata, "manifest": "scenario/wrong.json"}), encoding="utf-8")
        self.assertFalse(fixtures.metadata_checks(directory, self.state, manifest)["manifest recorded"])

    def test_hash_equality_requires_evidence_and_rejects_divergence(self):
        """Equal real strings pass, whereas unequal, missing or empty world hashes do not."""
        first, second = self.root / "first", self.root / "second"
        self.write_metadata(first)
        metadata = self.write_metadata(second)
        self.assertTrue(fixtures.matching_worlds(first, second))
        for value in ("b" * 64, None, ""):
            with self.subTest(value=value):
                (second / FixtureMetadata.FILE).write_text(
                    json.dumps({**metadata, "world_hash": value}), encoding="utf-8")
                self.assertFalse(fixtures.matching_worlds(first, second))
        for directory in (first, second):
            (directory / FixtureMetadata.FILE).write_text('{}', encoding="utf-8")
        self.assertFalse(fixtures.matching_worlds(first, second))

    def test_malformed_metadata_is_an_error_not_a_pass(self):
        """Absent files, invalid JSON, non-object records and invalid code objects fail locally."""
        directory = self.root / "base"
        directory.mkdir()
        with self.assertRaises(FileNotFoundError):
            fixtures.metadata_checks(directory, self.state)
        for text in ("{", "[]", '{"code": null}'):
            with self.subTest(text=text):
                (directory / FixtureMetadata.FILE).write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    fixtures.metadata_checks(directory, self.state)

    def test_existing_stale_fixture_names_changed_fields_and_is_not_replaced(self):
        """Existing layout and container drift produces a failing check with the exact differing fields."""
        experiment = fixtures.FixtureExperiment(self.args)
        experiment.setup()
        committed = experiment._directory / "committed"
        committed.mkdir()
        for variant in fixtures.VARIANTS:
            (committed / f"village_{variant}.json").write_text('{"layout": {}, "containers": {}}', encoding="utf-8")
        manifest = committed / "village_accident.json"
        directory = self.root / "village_accident"
        self.write_metadata(directory, manifest)
        before = (directory / FixtureMetadata.FILE).read_bytes()
        manifest.write_text('{"layout": {"changed": true}, "containers": {"chest": 1}}', encoding="utf-8")
        experiment._existing_checks()
        failed = [label for label, passed in experiment._checks.items() if not passed]
        self.assertEqual(len(failed), 1)
        self.assertIn("changed in layout, containers since it was built", failed[0])
        self.assertEqual((directory / FixtureMetadata.FILE).read_bytes(), before)

    def test_empty_check_set_cannot_pass(self):
        """An experiment without any executed checks must return a FAIL verdict."""
        with redirect_stdout(io.StringIO()) as output:
            passed = fixtures.FixtureExperiment(self.args).check()
        self.assertFalse(passed)
        self.assertIn("FAIL IT-14:", output.getvalue())


class ServerFreeFixtureExperiment(fixtures.FixtureExperiment):
    """Exercise the actual ten-build driver using only files in place of git worktrees and Minecraft."""

    def __init__(self, args, state):
        """Retain a clean code state and record every simulated git or build operation."""
        super().__init__(args)
        self.state = state
        self.events = []
        self.failure = None
        self.changed_manifest = False

    def _git(self, *arguments):
        """Materialize a checkout or remove it, recording the exact git command requested by the real driver."""
        self.events.append(arguments)
        if arguments[:2] == ("worktree", "add"):
            (self._worktree / "scenario").mkdir(parents=True)
            for variant in fixtures.VARIANTS:
                (self._worktree / f"scenario/village_{variant}.json").write_text(
                    json.dumps({"variant": variant, "seed": 1, "layout": {}}), encoding="utf-8")
        elif arguments[:2] == ("worktree", "remove"):
            shutil.rmtree(self._worktree)
        else:
            raise AssertionError(arguments)
        return ""

    def _build(self, module, root, variant=None, pre=False):
        """Write metadata and a manifest instead of launching a child, optionally raising a deliberate failure. A
        pre-incident build keeps its manifest in the fixture; staging writes the checkout's scenario."""
        self.events.append((module, root, variant))
        if self.failure is not None:
            raise self.failure
        name = "base" if variant is None else f"village_{variant}{'_pre' if pre else ''}"
        directory = root / name
        directory.mkdir()
        manifest = None if variant is None else self._worktree / f"scenario/village_{variant}.json"
        if pre:
            (directory / "scenario.json").write_bytes(manifest.read_bytes())
            FixtureMetadata.write(directory, self.state, manifest=directory / "scenario.json", world_hash="b" * 64)
            return
        if self.changed_manifest and manifest is not None:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            manifest.write_text(json.dumps({**data, "changed": True}), encoding="utf-8")
        FixtureMetadata.write(directory, self.state, manifest=manifest, world_hash="a" * 64)


class FixtureDriverTest(FixtureWorkspaceTest):
    """Test fresh worktree lifetime, independent build roots and commands without executing a real gate."""

    def experiment(self) -> ServerFreeFixtureExperiment:
        """Construct a file-only experiment with an explicit captured code state."""
        self.replace(fixtures, "CodeState", SimpleNamespace(capture=self.capture))
        experiment = ServerFreeFixtureExperiment(self.args, self.state)
        experiment.setup()
        self.replace(provenance, "REPO_ROOT", experiment._worktree)
        return experiment

    def test_two_independent_builds_preserve_manifests_and_cleanup(self):
        """Prepare, then each variant's pre-incident build and staging, twice; metadata passes and no checkout
        survives the run."""
        experiment = self.experiment()
        experiment.run()
        events = [event for event in experiment.events
                  if event[0] in ("prepare_world", "build_village", "stage_incident")]
        self.assertEqual([(event[0], event[2]) for event in events],
                         [("prepare_world", None), ("build_village", "accident"), ("stage_incident", "accident"),
                          ("build_village", "sabotage"), ("stage_incident", "sabotage")] * 2)
        self.assertNotEqual(events[0][1], events[5][1])
        self.assertTrue(all(event[1].is_relative_to(self.root) for event in events))
        self.assertEqual(experiment.events[0], ("worktree", "add", "--detach", str(experiment._worktree), "HEAD"))
        self.assertEqual(experiment.events[-1], ("worktree", "remove", "--force", str(experiment._worktree)))
        self.assertFalse(experiment._worktree.exists())
        for root in experiment._roots:
            for variant in fixtures.VARIANTS:
                self.assertTrue((root / f"village_{variant}.json").is_file())
        with redirect_stdout(io.StringIO()) as output:
            self.assertTrue(experiment.check())
        self.assertIn("ok  negative control: changed layout reported stale", output.getvalue())
        self.assertIn("PASS IT-14:", output.getvalue())

    def test_worktree_removed_after_build_failure_and_interrupt(self):
        """Both ordinary builder errors and signal-style KeyboardInterrupt unwind through git worktree remove."""
        for failure in (harness.HarnessError("builder failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                experiment = self.experiment()
                experiment.failure = failure
                with self.assertRaises(type(failure)):
                    experiment.run()
                self.assertEqual(experiment.events[-1][:2], ("worktree", "remove"))
                self.assertFalse(experiment._worktree.exists())

    def test_changed_built_manifest_does_not_replace_committed_reference(self):
        """A builder rewriting its checkout's scenario cannot silently change the committed comparison baseline."""
        experiment = self.experiment()
        experiment.changed_manifest = True
        experiment.run()
        manifest_checks = {label: passed for label, passed in experiment._checks.items()
                           if "manifest equals committed HEAD" in label}
        self.assertEqual(len(manifest_checks), 4)
        self.assertFalse(any(manifest_checks.values()))

    def test_bootstrap_uses_pinned_jar_and_only_allocated_ports(self):
        """Commands import HEAD's builders, pass shared limits and do not require a jar inside the checkout."""
        experiment = self.experiment()
        for module, variant in (("prepare_world", None), ("build_village", "sabotage")):
            with self.subTest(module=module):
                command = experiment._command(module, experiment._roots[0], variant)
                compile(command[2], "<IT-14 bootstrap>", "exec")
                self.assertIn(f"from scripts import harness, {module}", command[2])
                self.assertEqual(command[3], str(experiment._jar))
                self.assertEqual(command[command.index("--port") + 1], "28000")
                self.assertEqual(command[command.index("--game-port") + 1], "28001")
                self.assertEqual(command[command.index("--heap") + 1], "2G")
                self.assertIn(str(experiment._roots[0]), command)
                self.assertEqual("--variant" in command, variant is not None)

    def test_child_bootstrap_runs_in_checkout_and_retains_failure_log(self):
        """Real tiny Python children verify jar binding and nonzero-exit diagnostics without importing any server."""
        experiment = fixtures.FixtureExperiment(self.args)
        experiment.setup()
        package = experiment._worktree / "scripts"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('', encoding="utf-8")
        (package / "harness.py").write_text('', encoding="utf-8")
        (package / "prepare_world.py").write_text(
            'from pathlib import Path\n'
            'def setup_server(name, root, *, mod_jar=Path("missing.jar")):\n'
            '    """Return only an existing jar; the bootstrap must replace the default."""\n'
            '    assert mod_jar.is_file()\n'
            '    return mod_jar\n'
            'def main():\n'
            '    """Exercise the setup binding without any server or model."""\n'
            '    print(setup_server("base", None))\n'
            '    return 0\n', encoding="utf-8")
        (package / "build_village.py").write_text(
            'from scripts import harness\n'
            'def main():\n'
            '    """Check the pinned jar and deliberately report a failed build."""\n'
            '    assert harness.MOD_JAR.is_file()\n'
            '    print("deliberate village failure")\n'
            '    return 4\n', encoding="utf-8")
        root = experiment._roots[0]
        root.mkdir()
        experiment._build("prepare_world", root)
        self.assertIn(str(experiment._jar), (root / "base.log").read_text(encoding="utf-8"))
        self.assertTrue(experiment._checks[f"{root.name}/base builder completed"])
        with self.assertRaisesRegex(harness.HarnessError, "village_accident build exited 4; see"):
            experiment._build("build_village", root, "accident")
        self.assertIn("deliberate village failure", (root / "village_accident.log").read_text(encoding="utf-8"))
        self.assertNotIn(f"{root.name}/village_accident builder completed", experiment._checks)

    def test_signal_handler_installed_before_parser(self):
        """Even a help-only invocation installs signal handling before parsing, and never starts a build."""
        events = []
        original_parser = harness.experiment_parser

        def stop_on_sigterm():
            """Record the first action in main without changing the test runner's signal handler."""
            events.append("signals")

        def parser(description):
            """Record parsing order and return the real shared parser."""
            events.append("parser")
            return original_parser(description)

        self.replace(harness, "stop_on_sigterm", stop_on_sigterm)
        self.replace(harness, "experiment_parser", parser)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            fixtures.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(events, ["signals", "parser"])


class BuilderClient:
    """A minimal deterministic protocol transcript for testing only the builders' end-of-build metadata wiring."""

    def __init__(self):
        """Start at tick zero with empty requests and no failed hash response."""
        self.time = 0
        self.requests = []
        self.bad_hash = False

    def request(self, command, args=None):
        """Answer gamerule queries, lockstep and ticks, and return existing-format synthetic harness hashes."""
        self.requests.append((command, args))
        if command == "command":
            values = dict(prepare_world.GAME_RULES)
            return {"value": int(values.get((args or {})["command"].removeprefix("gamerule "), 0))}
        if command == "step":
            self.time += args["n"]
        if command == "hash":
            return {"gameTime": self.time, **{key: "bad" if self.bad_hash else "a" * 64 for key in harness.HASH_KEYS}}
        return {"gameTime": self.time}


class BuilderServer:
    """A context-manager stand-in that records clean shutdown without creating any server process."""

    def __init__(self, client):
        """Retain a file-only client and record whether its context exited."""
        self.client = client
        self.stopped = False

    def __enter__(self):
        """Return the already-created stand-in without any startup side effects."""
        return self

    def __exit__(self, *_error):
        """Mark shutdown so tests can verify failed hash validation still unwinds cleanup."""
        self.stopped = True


class MetadataVillageBuilder(build_village.VillageBuilder):
    """Avoid world construction while retaining the real run, schematic, manifest and metadata writers."""

    def build(self, client):
        """Advance to a synthetic completed build and publish the same setup timestamp the real builder does."""
        client.request("step", {"n": 100})
        self.config.layout["setup"] = {"start_game_time": 0, "recording_start_game_time": 100}


class FixtureBuilderWiringTest(FixtureWorkspaceTest):
    """Run the real builder entry paths with tiny protocol stand-ins and the real FixtureMetadata writer."""

    def setUp(self):
        """Redirect only server setup and lifecycle boundaries; leave metadata and manifest serialization real."""
        super().setUp()
        self.client = BuilderClient()
        self.server = BuilderServer(self.client)
        self.replace(harness, "Server", self.make_server)
        self.replace(prepare_world, "CodeState", SimpleNamespace(capture=self.capture))
        self.replace(build_village, "CodeState", SimpleNamespace(capture=self.capture))
        self.replace(prepare_world, "setup_server", self.setup_base)
        self.replace(harness, "stop_on_sigterm", self.ignore_signal)
        self.replace(harness, "clone_apfs", self.clone_base)
        self.replace(build_village, "REPO_ROOT", self.root)
        self.replace(provenance, "REPO_ROOT", self.root)

    def capture(self, jar, repo=None) -> CodeState:
        """Require clean shutdown before capturing the code state used to publish fixture metadata."""
        self.assertTrue(self.server.stopped)
        return super().capture(jar, repo)

    def ignore_signal(self):
        """Avoid changing the unit runner's signal handler."""

    def setup_base(self, name, root, *, bootstrap_base):
        """Require explicit base bootstrapping and assemble an empty directory without downloads or Java."""
        self.assertIs(bootstrap_base, True)
        directory = root / name
        directory.mkdir()
        return directory

    def make_server(self, directory, **options):
        """Check the run root and shared port pair and return a process-free lifecycle stand-in."""
        self.assertTrue(directory.is_relative_to(self.root))
        self.assertEqual((options["port"], options["game_port"]), (28000, 28001))
        return self.server

    def clone_base(self, source, destination):
        """Refuse overwrites and create a tiny independent directory instead of an APFS clone."""
        self.assertEqual(source, self.root / "base")
        destination.mkdir()

    def test_prepare_writes_harness_region_hash_and_current_code_after_shutdown(self):
        """The base fixture records the real metadata schema and only hashes after its hundredth tick."""
        with redirect_stdout(io.StringIO()):
            status = prepare_world.main(["--runs-dir", str(self.root), "--port", "28000", "--game-port", "28001"])
        self.assertEqual(status, 0)
        self.assertTrue(self.server.stopped)
        self.assertEqual(self.client.requests[-1], ("hash", {"region": harness.REGION}))
        self.assertEqual(self.client.time, 100)
        self.assertTrue(all(fixtures.metadata_checks(self.root / "base", self.state).values()))

    def test_villages_write_the_emitted_manifest_fields_and_layout_region_hash(self):
        """Both variants use their own layout region and the manifest actually emitted by the builder."""
        for variant in fixtures.VARIANTS:
            with self.subTest(variant=variant):
                self.client.time = 0
                builder = MetadataVillageBuilder(build_village.ScenarioConfig.create(variant))
                directory = builder.run(self.args)
                manifest = self.root / f"scenario/village_{variant}.json"
                self.assertEqual(self.client.requests[-1], ("hash", {"region": builder.config.layout["region"]}))
                self.assertTrue(self.server.stopped)
                self.assertTrue(all(fixtures.metadata_checks(directory, self.state, manifest).values()))
                self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["variant"], variant)

    def test_failed_hash_validation_does_not_publish_success_metadata(self):
        """A malformed final harness response stops each builder before it writes fresh fixture metadata."""
        self.client.bad_hash = True
        with redirect_stdout(io.StringIO()), self.assertRaises(harness.ProtocolError):
            prepare_world.main(["--runs-dir", str(self.root), "--port", "28000", "--game-port", "28001"])
        self.assertFalse((self.root / "base/fixture.json").exists())
        builder = MetadataVillageBuilder(build_village.ScenarioConfig.create("accident"))
        with self.assertRaises(harness.ProtocolError):
            builder.run(self.args)
        self.assertFalse((self.root / "village_accident/fixture.json").exists())
        self.assertFalse((self.root / "scenario/village_accident.json").exists())
        self.assertTrue(self.server.stopped)


if __name__ == "__main__":
    unittest.main()
