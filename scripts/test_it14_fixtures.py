"""IT-14: reproduce base and both village fixtures from a fresh HEAD checkout (P2, W1).

Procedure: reserve a directory under --runs-dir, add a detached HEAD git worktree there, and pin a copy of the
already-built harness jar outside mod/. Run prepare_world, then for both variants build the pre-incident village and
stage its incident live (M7c), twice, in independent run roots, without borrowing an existing base or launcher
cache. Retain each built manifest and the committed manifests outside the checkout; always remove the checkout with
git worktree remove in finally.

Pass: each fixture's two world hashes match (base, both pre-incident and both staged villages); every fixture.json
records HEAD, the pinned jar, a clean checkout and a world hash; village metadata matches the manifest's world-field
hashes; every staged manifest equals its committed counterpart, so staging leaves the checkout clean.
Existing fixtures under --runs-dir are checked with FixtureMetadata.problems and stale fields are reported, never
silently adopted or replaced. Missing pre-existing fixtures are allowed: this gate builds them from scratch.
Negative control: change only layout in a copied manifest; FixtureMetadata.problems must name layout as stale.
Estimate: about an hour for ten sequential builds (staging searches and rehearses its incident), plus first-launch
downloads. A fresh world generates its force-loaded area before lockstep, so builds get BUILD_TIMEOUT per request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

if __package__:
    from . import harness
    from .provenance import CodeState, FixtureMetadata
else:
    import harness
    from provenance import CodeState, FixtureMetadata


VARIANTS = ("accident", "sabotage")
FIXTURES = ("base", *(f"village_{variant}{suffix}" for variant in VARIANTS for suffix in ("_pre", "")))
BUILD_TIMEOUT = 600.0  # seconds per request: a fresh world generates its force-loaded chunks before lockstep


def metadata_checks(directory: Path, state: CodeState, manifest: Path | None = None,
                    recorded: str | None = None) -> dict[str, bool]:
    """Compare a fresh fixture's recorded code and world/manifest evidence with the worktree build inputs; `recorded`
    is the manifest name the fixture should name (default `scenario/<manifest name>`)."""
    metadata = json.loads((directory / FixtureMetadata.FILE).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("code"), dict):
        raise ValueError(f"Invalid fixture metadata in {directory}")
    code = metadata["code"]
    world_hash = metadata.get("world_hash")
    checks = {
        "commit matches HEAD": state.commit is not None and code.get("commit") == state.commit,
        "jar matches pinned harness": state.jar_sha256 is not None and code.get("jar_sha256") == state.jar_sha256,
        "clean build provenance": code.get("dirty") is False and code.get("error") is None,
        "world hash recorded": (isinstance(world_hash, str) and len(world_hash) == 64
                                and all(character in "0123456789abcdef" for character in world_hash)),
        "built, not adopted": metadata.get("adopted") is False,
    }
    if manifest is not None:
        checks["manifest recorded"] = metadata.get("manifest") == (recorded or f"scenario/{manifest.name}")
        checks["manifest-field hashes match"] = (
            metadata.get("manifest_fields") == FixtureMetadata.field_hashes(manifest))
    else:
        checks["base has no manifest"] = metadata.get("manifest") is None and metadata.get("manifest_fields") is None
    return checks


def matching_worlds(first: Path, second: Path) -> bool:
    """Require two nonempty recorded world hashes to match, never accepting absent evidence as equality."""
    left = json.loads((first / FixtureMetadata.FILE).read_text(encoding="utf-8")).get("world_hash")
    right = json.loads((second / FixtureMetadata.FILE).read_text(encoding="utf-8")).get("world_hash")
    return isinstance(left, str) and bool(left) and left == right


class FixtureExperiment:
    """Own the fresh worktree, six builds, metadata comparisons and manifest-staleness negative control."""

    def __init__(self, args: argparse.Namespace):
        """Retain shared CLI options without creating directories, worktrees or processes."""
        self._args = args
        self._checks: dict[str, bool] = {}

    def _git(self, *arguments: str) -> str:
        """Run a bounded git command in the source repository, preserving useful failure diagnostics."""
        return subprocess.run(["git", "-C", str(harness.REPO_ROOT), *arguments], check=True,
                              capture_output=True, text=True, timeout=120).stdout

    def setup(self) -> None:
        """Reserve all artifacts under --runs-dir and copy the existing jar without writing anywhere in mod/."""
        if not 1 <= self._args.port <= 65528:
            raise ValueError("IT-14 requires --port in 1..65528")
        if self._args.game_port not in (None, self._args.port + 1):
            raise ValueError("IT-14 requires game port --port + 1")
        root = self._args.runs_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix="it14_fixtures_", dir=root))
        self._worktree = self._directory / "checkout"
        self._jar = self._directory / "simharness.jar"
        shutil.copy2(harness.MOD_JAR, self._jar)
        self._roots = [self._directory / f"build_{number}" for number in (1, 2)]

    def _command(self, module: str, root: Path, variant: str | None = None, pre: bool = False) -> list[str]:
        """Use the worktree's builders with the pinned jar and one sequential control/game port pair."""
        bootstrap = ("from pathlib import Path; import sys; from functools import partial; "
                     f"from scripts import harness, {module}; "
                     "harness.MOD_JAR = Path(sys.argv.pop(1)); ")
        if module == "prepare_world":
            bootstrap += "prepare_world.setup_server = partial(prepare_world.setup_server, mod_jar=harness.MOD_JAR); "
        bootstrap += f"raise SystemExit({module}.main())"
        command = [sys.executable, "-c", bootstrap, str(self._jar), "--runs-dir", str(root),
                   "--port", str(self._args.port), "--game-port", str(self._args.port + 1),
                   "--base-seed", str(self._args.base_seed), "--startup-timeout", str(self._args.startup_timeout),
                   "--request-timeout", str(max(self._args.request_timeout, BUILD_TIMEOUT))]
        if self._args.heap:
            command += ["--heap", self._args.heap]
        if variant is not None:
            command += ["--variant", variant]
        if pre:
            command += ["--pre-incident"]
        return command

    def _build(self, module: str, root: Path, variant: str | None = None, pre: bool = False) -> None:
        """Run a builder in HEAD, retaining its output and letting its signal handler stop its own JVM."""
        name = f"village_{variant}{'_pre' if pre else ''}" if variant is not None else "base"
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(self._command(module, root, variant, pre), cwd=self._worktree,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if status:
                raise harness.HarnessError(f"{name} build exited {status}; see {root / f'{name}.log'}")
        self._checks[f"{root.name}/{name} builder completed"] = True

    def _existing_checks(self) -> None:
        """Report existing fixtures' exact metadata problems, including the differing world-relevant fields."""
        for name in FIXTURES:
            directory = self._args.runs_dir / name
            if not directory.exists() and not directory.is_symlink():
                self._checks[f"existing {name}: absent; fresh builds used"] = True
                continue
            manifest = (None if name == "base" else directory / "scenario.json" if name.endswith("_pre")
                        else self._directory / "committed" / f"{name}.json")
            problems = FixtureMetadata.problems(directory, manifest)
            detail = "; ".join(problems) if problems else "metadata current"
            self._checks[f"existing {name}: {detail}"] = not problems

    def _negative_control(self) -> None:
        """Change a copied manifest's layout and require the existing metadata staleness guard to name it."""
        directory = self._directory / "negative_control"
        directory.mkdir()
        source = self._roots[0] / "village_accident"
        manifest = directory / "village_accident.json"
        data = json.loads((self._roots[0] / manifest.name).read_text(encoding="utf-8"))
        shutil.copy2(source / FixtureMetadata.FILE, directory / FixtureMetadata.FILE)
        manifest.write_text(json.dumps(data), encoding="utf-8")
        current = not FixtureMetadata.problems(directory, manifest)
        data["layout"] = {"it14_negative_control": True}
        manifest.write_text(json.dumps(data), encoding="utf-8")
        problems = FixtureMetadata.problems(directory, manifest)
        self._checks["negative control: changed layout reported stale"] = (
            current and len(problems) == 1 and "changed in layout since it was built" in problems[0])

    def run(self) -> None:
        """Build twice from HEAD and remove the worktree in finally, including on build failure or interruption."""
        added = False
        try:
            self._git("worktree", "add", "--detach", str(self._worktree), "HEAD")
            added = True
            state = CodeState.capture(self._jar, self._worktree)
            if state.error or not state.commit or not state.jar_sha256 or state.dirty:
                raise harness.HarnessError(f"Fresh HEAD worktree or pinned jar unavailable: {state.summary()}")
            committed = self._directory / "committed"
            committed.mkdir()
            for variant in VARIANTS:
                name = f"village_{variant}.json"
                shutil.copy2(self._worktree / "scenario" / name, committed / name)
            self._existing_checks()
            for root in self._roots:
                root.mkdir()
                self._build("prepare_world", root)
                self._check_metadata(root / "base", state)
                for variant in VARIANTS:
                    self._build("build_village", root, variant, pre=True)
                    pre = root / f"village_{variant}_pre"
                    self._check_metadata(pre, state, pre / "scenario.json", str((pre / "scenario.json").resolve()))
                    self._build("stage_incident", root, variant)
                    name = f"village_{variant}"
                    manifest = root / f"{name}.json"
                    shutil.copy2(self._worktree / "scenario" / manifest.name, manifest)
                    self._check_metadata(root / name, state, manifest)
                    self._checks[f"{root.name}/{name} manifest equals committed HEAD"] = (
                        json.loads(manifest.read_text(encoding="utf-8"))
                        == json.loads((committed / manifest.name).read_text(encoding="utf-8")))
            for name in FIXTURES:
                self._checks[f"{name} world hashes identical across both builds"] = matching_worlds(
                    self._roots[0] / name, self._roots[1] / name)
            self._negative_control()
        finally:
            if added:
                self._git("worktree", "remove", "--force", str(self._worktree))
                self._checks["temporary HEAD worktree removed"] = not self._worktree.exists()

    def _check_metadata(self, directory: Path, state: CodeState, manifest: Path | None = None,
                        recorded: str | None = None) -> None:
        """Label each metadata requirement with its build root and fixture to keep failures attributable."""
        for label, passed in metadata_checks(directory, state, manifest, recorded).items():
            self._checks[f"{directory.parent.name}/{directory.name}: {label}"] = passed

    def check(self) -> bool:
        """Print every recorded check and pass only when nonempty evidence satisfies all fixture invariants."""
        for label, passed in self._checks.items():
            print(f"  {'ok ' if passed else 'BAD'} {label}")
        passed = bool(self._checks) and all(self._checks.values())
        print(f"{'PASS' if passed else 'FAIL'} IT-14: fresh HEAD fixtures, metadata and committed manifests")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Install signal cleanup first, parse shared options, and run the fixture gate with a failing error verdict."""
    harness.stop_on_sigterm()
    experiment = FixtureExperiment(harness.experiment_parser(__doc__).parse_args(argv))
    try:
        experiment.setup()
        experiment.run()
    except (OSError, ValueError, harness.HarnessError, subprocess.SubprocessError) as error:
        experiment._checks[str(error)] = False
    return 0 if experiment.check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
