"""What code produced a run or a fixture (plan §2: incidents P1, P10, P11 and W1).

- `CodeState` is the code a process runs: the git commit, whether the tree was dirty (with the diff of tracked files and
  the names of untracked ones), and the SHA-256 of the harness jar.
- `RunProvenance` appends one `CodeState` per server start to a run directory's `provenance.json`, and saves a dirty
  tree's diff beside it, so a run can be replayed later with the prompts of the code that recorded it. A branch cloned
  from a fork inherits the fork's starts, so the file is the run's whole lineage.
- `FixtureMetadata` is a fixture's `fixture.json`: the code state that built it, its world hash if known, and a hash of
  each world-relevant field of the scenario manifest it was built from. A fixture whose manifest has since changed in
  one of those fields is stale (W1); `run_gates.py` refuses it.

Usage, to record the metadata of a fixture built before metadata existed (after checking it matches its manifest):
`python3.11 scripts/provenance.py adopt runs/village_sabotage --manifest scenario/village_sabotage.json`
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

if __package__:
    from .setup_server import REPO_ROOT
else:
    from setup_server import REPO_ROOT


def file_sha256(path: Path) -> str | None:
    """The SHA-256 of a file, streamed, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CodeState:
    """The code a process runs: git commit, dirty flag with the tracked diff and untracked names, and the jar's hash."""

    commit: str | None
    dirty: bool
    diff: str
    untracked: tuple[str, ...]
    jar_sha256: str | None
    error: str | None = None

    @classmethod
    def capture(cls, jar: Path, repo: Path = REPO_ROOT) -> CodeState:
        """Read the repository's state with git (an unreadable repository is recorded as an error, not raised)."""
        try:
            commit = cls._git(repo, "rev-parse", "HEAD").strip()
            status = cls._git(repo, "status", "--porcelain").splitlines()
            diff = cls._git(repo, "diff", "HEAD")
        except (OSError, subprocess.SubprocessError) as error:
            return cls(None, True, "", (), file_sha256(jar), f"git: {error}")
        untracked = tuple(line[3:] for line in status if line.startswith("?? "))
        return cls(commit, bool(status), diff, untracked, file_sha256(jar))

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        """One git command's standard output."""
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                              timeout=60).stdout

    def summary(self) -> dict:
        """The state without the diff text, for JSON records."""
        return {key: value for key, value in asdict(self).items() if key != "diff"}


class RunProvenance:
    """A run directory's `provenance.json`: one entry per server start, oldest first."""

    FILE = "provenance.json"

    @classmethod
    def record(cls, directory: Path, state: CodeState) -> dict:
        """Append this start's code state (saving a dirty diff as `provenance_<n>.diff`) and return the entry."""
        directory = Path(directory)
        starts = cls.starts(directory)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **state.summary(), "diff_file": None}
        if state.diff:
            entry["diff_file"] = f"provenance_{len(starts) + 1}.diff"
            (directory / entry["diff_file"]).write_text(state.diff, encoding="utf-8")
        (directory / cls.FILE).write_text(json.dumps({"starts": [*starts, entry]}, indent=1) + "\n",
                                          encoding="utf-8")
        return entry

    @classmethod
    def starts(cls, directory: Path) -> list[dict]:
        """The recorded starts of a run directory (empty for runs recorded before provenance existed)."""
        path = Path(directory) / cls.FILE
        return json.loads(path.read_text(encoding="utf-8"))["starts"] if path.exists() else []


class FixtureMetadata:
    """A fixture's `fixture.json`: the code that built it, its world hash, and its manifest's world-relevant fields."""

    FILE = "fixture.json"
    WORLD_FIELDS = ("seed", "layout", "places", "roads", "schematics", "containers", "documents", "evidence", "npcs",
                    "roles")

    @classmethod
    def read(cls, directory: Path) -> dict:
        """Report the inherited fixture.json identity, explicitly labeling fixtures that predate provenance."""
        path = Path(directory) / cls.FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "legacy_unversioned"}

    @classmethod
    def field_hashes(cls, manifest: Path) -> dict[str, str]:
        """SHA-256 of each world-relevant field of a scenario manifest (canonical JSON; absent fields are null)."""
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        return {field: hashlib.sha256(json.dumps(data.get(field), sort_keys=True).encode()).hexdigest()
                for field in cls.WORLD_FIELDS}

    @classmethod
    def write(cls, directory: Path, state: CodeState, *, manifest: Path | None = None, world_hash: str | None = None,
              adopted: bool = False) -> dict:
        """Write `fixture.json` for a fixture just built (or adopted: built earlier, checked by hand) and return it."""
        metadata = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code": state.summary(), "world_hash": world_hash,
                    "manifest": cls._name(manifest) if manifest else None,
                    "manifest_fields": cls.field_hashes(manifest) if manifest else None, "adopted": adopted}
        (Path(directory) / cls.FILE).write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
        return metadata

    @staticmethod
    def _name(path: Path) -> str:
        """A manifest's path relative to the repository when it is inside it, else absolute."""
        path = Path(path).resolve()
        return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)

    @classmethod
    def problems(cls, directory: Path, manifest: Path | None = None) -> list[str]:
        """Why a fixture cannot be used: missing, without metadata, or built from a manifest whose world-relevant
        fields have changed since (empty when it is current)."""
        directory = Path(directory)
        if not directory.is_dir():
            return [f"{directory} does not exist"]
        path = directory / cls.FILE
        if not path.exists():
            return [f"{directory} has no {cls.FILE}: rebuild it, or record it with `scripts/provenance.py adopt`"]
        if manifest is None:
            return []
        recorded = json.loads(path.read_text(encoding="utf-8")).get("manifest_fields") or {}
        changed = [field for field, digest in cls.field_hashes(manifest).items() if recorded.get(field) != digest]
        return [f"{directory.name} is stale: {Path(manifest).name} changed in {', '.join(changed)} since it was "
                f"built; rebuild it"] if changed else []


def main(argv: list[str] | None = None) -> int:
    """Record `fixture.json` for an existing fixture (`adopt`), or print a run's provenance (`show`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("adopt", "show"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=None, help="the scenario manifest the fixture was built from")
    parser.add_argument("--jar", type=Path, default=REPO_ROOT / "mod/build/libs/simharness-0.1.0.jar")
    args = parser.parse_args(argv)
    if args.action == "adopt":
        print(json.dumps(FixtureMetadata.write(args.directory, CodeState.capture(args.jar), manifest=args.manifest,
                                               adopted=True), indent=1))
    else:
        print(json.dumps(RunProvenance.starts(args.directory), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
