"""Assemble an isolated Fabric server directory without starting Java."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
MINECRAFT_VERSION = "26.3"
LOADER_VERSION = "0.19.5"
FABRIC_API_VERSION = "0.162.0+26.3"
BASE_SEED = 20261016
INSTALLERS_URL = "https://meta.fabricmc.net/v2/versions/installer"
SERVER_PROPERTIES = "\n".join(
    (
        "online-mode=false",
        f"level-seed={BASE_SEED}",
        r"level-type=minecraft\:flat",
        "spawn-monsters=false",
        "spawn-animals=false",
        "spawn-npcs=false",
        "pause-when-empty-seconds=0",
        "max-tick-time=-1",
        "view-distance=4",
        "simulation-distance=4",
        "sync-chunk-writes=true",
        "level-name=world",
        "server-ip=127.0.0.1",
    )
) + "\n"


def cached_launcher(cache_dir: Path) -> Path:
    """Return a validated cached launcher, downloading it atomically when absent.

    Invalid metadata or JARs raise ValueError; download and filesystem errors
    propagate after temporary files are removed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    launcher = cache_dir / "fabric-server-launch-26.3-0.19.5.jar"
    if launcher.exists():
        if not zipfile.is_zipfile(launcher):
            raise ValueError(f"Invalid cached launcher: {launcher}; remove it and retry")
        return launcher
    with urlopen(INSTALLERS_URL, timeout=30) as response:
        installers = json.loads(response.read(1024 * 1024))
    if not isinstance(installers, list) or not installers:
        raise ValueError("Fabric returned no installer versions")
    version = installers[0].get("version") if isinstance(installers[0], dict) else None
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version):
        raise ValueError(f"Invalid latest Fabric installer version: {version!r}")
    url = (
        f"https://meta.fabricmc.net/v2/versions/loader/{MINECRAFT_VERSION}/"
        f"{LOADER_VERSION}/{version}/server/jar"
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".jar", delete=False) as output:
            temporary_path = Path(output.name)
            with urlopen(url, timeout=30) as response:
                shutil.copyfileobj(response, output)
        if not zipfile.is_zipfile(temporary_path):
            raise ValueError("Fabric launcher download is not a JAR")
        os.replace(temporary_path, launcher)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return launcher


def fabric_api_jar(gradle_cache: Path) -> Path:
    """Return the pinned API JAR, raising FileNotFoundError unless exactly one exists."""
    version_dir = (
        gradle_cache / "caches/modules-2/files-2.1/net.fabricmc.fabric-api/fabric-api"
        / FABRIC_API_VERSION
    )
    matches = sorted(version_dir.glob(f"*/fabric-api-{FABRIC_API_VERSION}.jar"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one Fabric API JAR in {version_dir}: {matches}")
    return matches[0]


def setup_server(
    name: str,
    runs_dir: Path = RUNS_DIR,
    *,
    mod_jar: Path = REPO_ROOT / "mod/build/libs/simharness-0.1.0.jar",
    gradle_cache: Path | None = None,
    bootstrap_base: bool = False,
) -> Path:
    """Assemble and return an isolated run directory without starting a server.

    Validate the run name and dependencies before staging. Existing destinations
    are never overwritten; setup and filesystem failures propagate to the caller.
    Only prepare_world opts into bootstrapping the base without cached game files.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name == "_cache":
        raise ValueError(f"Invalid run name: {name!r}")
    if bootstrap_base and name != "base":
        raise ValueError("Only the base run may bootstrap game files; use scripts/prepare_world.py")
    runs_dir = Path(runs_dir).resolve()
    destination = runs_dir / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if not mod_jar.is_file():
        raise FileNotFoundError(f"Build the harness mod first: {mod_jar}")
    gradle_cache = Path(gradle_cache or os.environ.get("GRADLE_USER_HOME", Path.home() / ".gradle"))
    api_jar = fabric_api_jar(gradle_cache)
    runs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=runs_dir) as temporary:
        staging = Path(temporary) / "server"
        mods = staging / "mods"
        mods.mkdir(parents=True)
        _clone_game_files(runs_dir / "base", staging, bootstrap_base=bootstrap_base)
        launcher = cached_launcher(runs_dir / "_cache")
        shutil.copy2(launcher, staging / "fabric-server-launch.jar")
        shutil.copy2(mod_jar, mods / mod_jar.name)
        shutil.copy2(api_jar, mods / api_jar.name)
        (staging / "eula.txt").write_text("eula=true\n", encoding="utf-8")
        (staging / "server.properties").write_text(SERVER_PROPERTIES, encoding="utf-8")
        staging.rename(destination)
    return destination


def _clone_game_files(base: Path, staging: Path, *, bootstrap_base: bool = False) -> None:
    """Validate and APFS-clone downloaded game files, allowing only explicit base bootstrapping.

    Without this, every fresh run directory makes the Fabric launcher re-download the Minecraft server from Mojang,
    which adds a network dependency to every fork (a transient timeout failed E2-skills on 2026-10-16). Only the
    vanilla server download and libraries are cloned; processed mods are left for the launcher to rebuild.
    """
    if not base.exists() and bootstrap_base:
        return
    if not base.is_dir():
        raise FileNotFoundError(
            f"Missing prepared base server: {base}; run scripts/prepare_world.py --runs-dir {base.parent} first"
        )
    directories = (Path(".fabric/server"), Path("libraries"), Path("versions"))
    required_jars = (Path(f".fabric/server/{MINECRAFT_VERSION}-server.jar"),
                     Path(f"versions/{MINECRAFT_VERSION}/server-{MINECRAFT_VERSION}.jar"))
    missing = [str(relative) for relative in directories if not (base / relative).is_dir()]
    missing.extend(str(relative) for relative in required_jars if not (base / relative).is_file())
    if not any(path.is_file() for path in (base / "libraries").rglob("*.jar")):
        missing.append("libraries/**/*.jar")
    if missing:
        raise FileNotFoundError(
            f"Incomplete prepared base server {base}; missing game files: {', '.join(missing)}; "
            "rebuild it with scripts/prepare_world.py before creating runs"
        )
    for relative in directories:
        source = base / relative
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-c", "-R", str(source), str(target)], check=True)


def main(argv: list[str] | None = None) -> int:
    """Parse setup options, print the created directory, and return zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--gradle-cache", type=Path)
    parser.add_argument("--mod-jar", type=Path, default=REPO_ROOT / "mod/build/libs/simharness-0.1.0.jar")
    args = parser.parse_args(argv)
    print(setup_server(args.name, args.runs_dir, mod_jar=args.mod_jar, gradle_cache=args.gradle_cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
