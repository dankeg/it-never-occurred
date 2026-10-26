"""Stdlib control client, bounded JVM lifecycle, and shared experiment driver."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import re
import math
import os
from pathlib import Path
import platform
import plistlib
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time

if __package__:
    from . import scripted_routine as routine
    from .provenance import CodeState, RunProvenance
    from .setup_server import BASE_SEED, REPO_ROOT, RUNS_DIR
else:
    import scripted_routine as routine
    from provenance import CodeState, RunProvenance
    from setup_server import BASE_SEED, REPO_ROOT, RUNS_DIR


JAVA_HOME = Path.home() / ".local/jdks/jdk-25.0.4.1+1/Contents/Home"
MOD_JAR = REPO_ROOT / "mod/build/libs/simharness-0.1.0.jar"
FORCELOADED_CHUNKS = {"cx0": -10, "cz0": -10, "cx1": 9, "cz1": 9}  # matches prepare_world forceload (-160..159 blocks)
REGION = {"x0": -32, "z0": -32, "x1": 31, "z1": 31, "y0": -64, "y1": -40}
HASH_EVERY = 20
RUN_TICKS = 24_000
FORK_TICKS = 12_000
HASH_KEYS = ("total", "blocks", "blockEntities", "entities", "harness")
MIN_FREE_BYTES = 10 * 10**9  # refuse to start a server below this; a village run can write several GB


class HarnessError(RuntimeError):
    """A failed experiment invariant, filesystem operation, or server lifecycle."""


class ProtocolError(HarnessError):
    """An invalid response frame, envelope, or deterministic sample sequence."""


class CommandError(HarnessError):
    """A well-formed server rejection that leaves the connection usable."""


class Client:
    """Serialize newline-framed requests with bounded deadlines and no retries."""

    def __init__(self, port: int = 25599, *, timeout: float = 60.0):
        """Connect to the loopback port; reject nonpositive request timeouts."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._socket = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self._buffer = b""
        self._next_id = 1
        self._lock = threading.Lock()
        self._closed = False

    def request(self, command: str, args: dict | None = None) -> dict:
        """Return a command's result, closing on transport or protocol errors.

        Request IDs advance once per call. Server rejections raise CommandError
        without closing the connection; no failed request is retried.
        """
        with self._lock:
            if self._closed:
                raise ProtocolError("Control connection is closed")
            request_id = self._next_id
            self._next_id += 1
            payload = json.dumps({"id": request_id, "cmd": command, "args": args or {}}, allow_nan=False)
            try:
                deadline = time.monotonic() + self.timeout
                self._socket.settimeout(self.timeout)
                self._socket.sendall(payload.encode("utf-8") + b"\n")
                while b"\n" not in self._buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"{command}: request deadline exceeded")
                    self._socket.settimeout(remaining)
                    chunk = self._socket.recv(65536)
                    if not chunk:
                        raise ProtocolError(f"{command}: missing or incomplete response")
                    self._buffer += chunk
                    if len(self._buffer) > 8 * 1024 * 1024:
                        raise ProtocolError(f"{command}: oversized response")
                line, self._buffer = self._buffer.split(b"\n", 1)
                try:
                    response = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise ProtocolError(f"{command}: invalid JSON response") from error
                if (not isinstance(response, dict) or type(response.get("id")) is not int
                        or response["id"] != request_id or type(response.get("ok")) is not bool):
                    raise ProtocolError(f"{command}: invalid response envelope: {response!r}")
                if not response["ok"]:
                    raise CommandError(f"{command}: {response.get('error', 'unspecified server error')}")
                if not isinstance(response.get("result"), dict):
                    raise ProtocolError(f"{command}: result must be an object")
                return response["result"]
            except CommandError:
                raise
            except (OSError, HarnessError):
                self.close()
                raise

    def speak(self, name: str, text: str, volume: str = "talk", *, to: str | None = None) -> dict:
        """Start timed speech; return the server's skill acceptance without ticking."""
        args = {"name": name, "text": text, "volume": volume}
        if to is not None:
            args["to"] = to
        return self.request("speak", args)

    def say(self, name: str, text: str, *, to: str | None = None) -> dict:
        """Alias speak at talk volume, retaining optional directed addressing."""
        return self.speak(name, text, "talk", to=to)

    def ask(self, name: str, npc: str) -> dict:
        """Start the server's timed, in-range NPC news exchange without ticking."""
        return self.request("ask", {"name": name, "npc": npc})

    def close(self) -> None:
        """Close the connection once; subsequent requests fail locally."""
        if not self._closed:
            self._closed = True
            self._socket.close()

    def __enter__(self) -> Client:
        """Return this already-connected client for context-managed cleanup."""
        return self

    def __exit__(self, *_exc) -> None:
        """Close the client without suppressing the context's exception."""
        self.close()


def install_current_mod(directory: Path) -> None:
    """Atomically refresh only this run's harness JAR from the local build."""
    if not MOD_JAR.is_file():
        raise FileNotFoundError(f"Build the harness mod first: {MOD_JAR}")
    mods = Path(directory) / "mods"
    if not mods.is_dir() or mods.is_symlink():
        raise HarnessError(f"Run must have its own mods directory: {mods}")
    with tempfile.TemporaryDirectory(prefix=".simharness-install-", dir=mods) as temporary:
        staged = Path(temporary) / MOD_JAR.name
        shutil.copy2(MOD_JAR, staged)
        os.replace(staged, mods / MOD_JAR.name)


def record_provenance(directory: Path) -> None:
    """Append the running code's state (git commit, dirty diff, jar hash) to the run's `provenance.json` (P1, P10,
    P11)."""
    RunProvenance.record(directory, CodeState.capture(MOD_JAR))


def stop_on_sigterm() -> None:
    """Make SIGTERM raise `KeyboardInterrupt` in this process (C8), so `Server` context managers stop their JVMs, which
    run in sessions of their own that the signal does not reach (incident P5). Long-running scripts call it first."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)


class Server:
    """Own one bounded JVM lifecycle, its log, and its readiness-tested client.

    Startup uses the existing deterministic JVM properties and refreshes only
    the run's harness JAR. Context cleanup never hides an existing exception.
    """

    def __init__(self, directory: Path, *, port: int = 25599, base_seed: int = BASE_SEED,
                 startup_timeout: float = 180.0, request_timeout: float = 60.0,
                 shutdown_timeout: float = 60.0, game_port: int | None = None, heap: str | None = None):
        """Configure a run and positive lifecycle timeouts without starting it.

        `port` is the control socket. `game_port`, if given, rewrites the run's Minecraft `server-port`, so several
        servers can run side by side (each also needs its own control `port`). `heap` (e.g. "3G") caps the JVM heap,
        which matters when a local model shares the machine's memory.
        """
        if min(startup_timeout, request_timeout, shutdown_timeout) <= 0:
            raise ValueError("Server timeouts must be positive")
        self.directory = Path(directory).resolve()
        self.port = port
        self.base_seed = base_seed
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.shutdown_timeout = shutdown_timeout
        self.game_port = game_port
        self.heap = heap
        self.log_path = self.directory / "server.log"
        self.client: Client | None = None
        self.process: subprocess.Popen | None = None
        self._log = None

    def start(self) -> Server:
        """Start once and return this server after a valid ping, or clean up and raise."""
        if self.process is not None:
            raise HarnessError("Server instance cannot be started twice")
        java = JAVA_HOME / "bin/java"
        if not java.is_file() or not (self.directory / "fabric-server-launch.jar").is_file():
            raise FileNotFoundError(f"Missing {java} or launcher in {self.directory}")
        free = shutil.disk_usage(self.directory).free
        if free < MIN_FREE_BYTES:
            raise HarnessError(f"Only {free / 1e9:.1f} GB free; need {MIN_FREE_BYTES / 1e9:.0f} GB to start a server "
                               f"(prune finished runs: scripts/prune_runs.py --all-finished)")
        install_current_mod(self.directory)
        # Always set the game port: clones inherit server.properties, so a world built on a side port would
        # otherwise collide with another server. The default is Minecraft's own.
        properties = self.directory / "server.properties"
        if properties.exists():
            text = re.sub(r"(?m)^server-port=.*\n?", "", properties.read_text(encoding="utf-8")).rstrip("\n")
            properties.write_text(f"{text}\nserver-port={self.game_port or 25565}\n", encoding="utf-8")
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.port))
        self._log = self.log_path.open("ab", buffering=0)
        try:
            self._log.write(b"\n--- harness JVM start ---\n")
            self.process = subprocess.Popen(
                [str(java), *([f"-Xmx{self.heap}"] if self.heap else []), f"-Dsimharness.port={self.port}",
                 "-Dsimharness.reseed_every=20",
                 f"-Dsimharness.base_seed={self.base_seed}", "-jar", "fabric-server-launch.jar", "nogui"],
                cwd=self.directory, stdin=subprocess.DEVNULL, stdout=self._log,
                stderr=subprocess.STDOUT, env={**os.environ, "JAVA_HOME": str(JAVA_HOME)},
                start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise HarnessError(f"Server exited ({self.process.returncode}); see {self.log_path}")
                candidate = None
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    candidate = Client(self.port, timeout=min(1.0, remaining))
                    game_time(candidate.request("ping"))
                    if self.process.poll() is not None:
                        raise HarnessError(f"Server exited during readiness; see {self.log_path}")
                    candidate.timeout = self.request_timeout
                    self.client = candidate
                    try:
                        record_provenance(self.directory)
                    except OSError as error:  # not a readiness failure: stop the server
                        raise HarnessError(f"Cannot record provenance in {self.directory}: {error}") from error
                    return self
                except OSError:
                    if candidate is not None:
                        candidate.close()
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                except BaseException:
                    if candidate is not None:
                        candidate.close()
                    raise
            raise HarnessError(f"Server readiness timed out after {self.startup_timeout}s; see {self.log_path}")
        except BaseException:
            try:
                self._terminate()
            finally:
                self._close_handles()
            raise

    def _terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)

    def _close_handles(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def stop(self) -> None:
        """Request shutdown, bound the wait, and close handles even on failure."""
        try:
            if self.process is None:
                return
            if self.process.poll() is None:
                if self.client is not None:
                    self.client.timeout = min(self.request_timeout, self.shutdown_timeout)
                    try:
                        self.client.request("stop")
                    except (OSError, HarnessError):
                        pass
                try:
                    self.process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired as error:
                    self._terminate()
                    raise HarnessError(f"Forced shutdown after timeout; see {self.log_path}") from error
            if self.process.returncode != 0:
                raise HarnessError(f"Server exited ({self.process.returncode}); see {self.log_path}")
        finally:
            try:
                self._terminate()
            finally:
                self._close_handles()

    def kill(self) -> None:
        """Terminate the JVM now, from any thread, so a run blocked on this server fails and unwinds (C8: an interrupted
        multi-threaded script cannot stop its servers from the threads that use them)."""
        self._terminate()

    def __enter__(self) -> Server:
        """Start the server and return its ready lifecycle owner."""
        return self.start()

    def __exit__(self, exc_type, *_exc) -> None:
        """Stop the server, propagating cleanup errors only when no error is active."""
        try:
            self.stop()
        except (OSError, HarnessError):
            if exc_type is None:
                raise


def game_time(result: dict) -> int:
    """Return a nonnegative integer gameTime, or raise ProtocolError."""
    value = result.get("gameTime")
    if type(value) is not int or value < 0:
        raise ProtocolError(f"Invalid game time: {result!r}")
    return value


def enable_lockstep(client: Client, *, recording: bool = True, await_entities: bool = True) -> int:
    """Pause ticking, await loaded entities, set recording, then return game time. A fresh world has no force-loaded
    chunks yet, so `prepare_world` passes `await_entities=False` and awaits them after its forceload commands."""
    client.request("lockstep", {"enabled": True})
    if await_entities:
        # Non-ticking barrier: wait until every force-loaded chunk has its entities loaded and entity-ticking.
        client.request("await_entities", FORCELOADED_CHUNKS)
    client.request("recording", {"enabled": recording})
    return game_time(client.request("ping"))

def apply_commands(client: Client, commands: list[dict]) -> None:
    """Send commands in order, resolving the shared below-front target only once."""
    target = None
    for command in commands:
        args = dict(command.get("args", {}))
        if command.get("target") == routine.BELOW_FRONT:
            if target is None:
                target = routine.below_front(client.request("bot_state", {"name": routine.BOT_NAME}))
            args.update(target)
        client.request(command["cmd"], args)


def hash_components(value: dict, expected_time: int) -> dict:
    """Validate existing component digests and return them in their protocol order."""
    if not isinstance(value, dict) or game_time(value) != expected_time:
        raise ProtocolError(f"Hash game time differs from {expected_time}: {value!r}")
    for key in HASH_KEYS:
        digest = value.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ProtocolError(f"Invalid {key} hash: {digest!r}")
    return {"gameTime": expected_time, **{key: value[key] for key in HASH_KEYS}}


def step_hashes(result: dict, before: int, after: int) -> list[dict]:
    """Return validated samples for (before, after], rejecting missing or extra hashes."""
    if game_time(result) != after:
        raise ProtocolError(f"step advanced to {result.get('gameTime')}, expected {after}")
    records = result.get("hashes")
    expected_times = list(range((before // HASH_EVERY + 1) * HASH_EVERY, after + 1, HASH_EVERY))
    if not isinstance(records, list) or len(records) != len(expected_times):
        raise ProtocolError(f"step returned incomplete hashes for ({before}, {after}]")
    normalized = []
    for record, expected_time in zip(records, expected_times):
        if not isinstance(record, dict) or game_time(record) != expected_time:
            raise ProtocolError(f"Unexpected hash record at {expected_time}: {record!r}")
        normalized.append({"gameTime": expected_time,
                           "hash": hash_components(record.get("hash"), expected_time)})
    return normalized


def bot_position(client: Client) -> tuple[float, float, float]:
    """Read the routine bot's finite three-coordinate position or raise ProtocolError."""
    state = client.request("bot_state", {"name": routine.BOT_NAME})
    position = state.get("pos")
    if (not isinstance(position, list) or len(position) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in position)):
        raise ProtocolError(f"Invalid bot position: {state!r}")
    return tuple(position)


class RoutineRunner:
    """Drive a client's scripted inputs and samples using only server game time."""

    def __init__(self, client: Client):
        """Retain the connected client without issuing any commands."""
        self._client = client

    def run(self, end_game_time: int) -> list[dict]:
        """Return hashes through end_game_time, requiring horizontal bot movement.

        Batches end at the same input boundaries as the scripted routine. Invalid
        end times raise ValueError; bad samples or absent movement raise HarnessError.
        """
        client = self._client
        current = game_time(client.request("ping"))
        if type(end_game_time) is not int or end_game_time <= current:
            raise ValueError("end_game_time must be later than current game time")
        previous_position = bot_position(client)
        moved = False
        hashes = []
        while current < end_game_time:
            apply_commands(client, routine.actions_for(current))
            if current % routine.CYCLE_TICKS == 300:
                client.request("observe", {"name": routine.BOT_NAME})
            after = min(end_game_time, routine.next_action_time(current))
            result = client.request("step", {"n": after - current, "hash_every": HASH_EVERY, "region": REGION})
            hashes.extend(step_hashes(result, current, after))
            position = bot_position(client)
            moved |= math.hypot(position[0] - previous_position[0], position[2] - previous_position[2]) > 1e-6
            previous_position = position
            current = after
        if not moved:
            raise HarnessError("alice never moved horizontally according to bot_state")
        return hashes


def run_routine(client: Client, end_game_time: int) -> list[dict]:
    """Run the shared routine through end_game_time using the supplied client."""
    return RoutineRunner(client).run(end_game_time)


def _apfs_volume(path: Path) -> str:
    try:
        device = subprocess.run(["df", "-P", str(path)], check=True,
                                capture_output=True, text=True, timeout=15).stdout.splitlines()
        if len(device) != 2 or not device[1].split():
            raise HarnessError(f"Cannot identify the filesystem containing {path}")
        result = subprocess.run(["diskutil", "info", "-plist", device[1].split()[0]], check=True,
                                capture_output=True, timeout=15)
        info = plistlib.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise HarnessError(f"Cannot verify APFS volume for {path}: {detail}") from error
    if info.get("FilesystemType", "").lower() != "apfs" or not info.get("VolumeUUID"):
        raise HarnessError(f"APFS cloning required, but {path} is not a verified APFS volume")
    return info["VolumeUUID"]


class ApfsClone:
    """Own a fail-closed same-volume clone with atomic staging and no copy fallback."""

    def __init__(self, source: Path, destination: Path):
        """Retain paths; filesystem validation remains deferred until clone()."""
        self._source = source
        self._destination = destination

    def clone(self) -> None:
        """Clone with cp -c -R, rejecting unsupported volumes or existing destinations."""
        source = Path(self._source).resolve(strict=True)
        destination = Path(self._destination).absolute()
        if not source.is_dir():
            raise NotADirectoryError(source)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite clone destination {destination}")
        if platform.system() != "Darwin":
            raise HarnessError("APFS clones require macOS; no non-APFS copy fallback is allowed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _apfs_volume(source) != _apfs_volume(destination.parent):
            raise HarnessError("Source and destination must be on the same APFS volume")
        if source == destination or source in destination.resolve().parents:
            raise ValueError("Clone destination must not be inside its source")
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-clone-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "tree"
            try:
                subprocess.run(["/bin/cp", "-c", "-R", str(source), str(staging)],
                               check=True, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as error:
                detail = getattr(error, "stderr", None) or str(error)
                raise HarnessError(f"APFS clone failed: {detail}") from error
            if not staging.is_dir():
                raise HarnessError("APFS clone did not produce a directory")
            staging.rename(destination)


def clone_apfs(source: Path, destination: Path) -> None:
    """Clone a run on the same APFS volume with no ordinary-copy fallback."""
    ApfsClone(source, destination).clone()


def disk_usage(path: Path) -> str:
    """Return df's stripped output for path, propagating subprocess failures."""
    return subprocess.run(["df", "-k", str(path)], check=True, capture_output=True,
                          text=True, timeout=15).stdout.strip()


def write_hashes(path: Path, hashes: list[dict]) -> None:
    """Write the existing indented hash-array format with its trailing newline."""
    path.write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class HashComparison:
    """Compare existing ordered samples and retain the gate's exact diagnostic text."""

    def __init__(self, expected: list[dict], actual: list[dict], label: str):
        """Retain both sample sequences and their report label without normalizing them."""
        self._expected = expected
        self._actual = actual
        self._label = label

    def check(self) -> bool:
        """Print the first mismatch or PASS line and return whether samples match."""
        expected, actual, label = self._expected, self._actual, self._label
        for left, right in zip_longest(expected, actual):
            if left != right:
                times = [record["gameTime"] for record in (left, right) if record is not None]
                print(f"FAIL {label}: first mismatch at game time {min(times)}")
                print("A: " + json.dumps(left, sort_keys=True))
                print("B/C: " + json.dumps(right, sort_keys=True))
                return False
        print(f"PASS {label}: {len(expected)} matching hashes; movement verified via bot_state")
        return True


def compare_hashes(expected: list[dict], actual: list[dict], label: str) -> bool:
    """Compare samples and print the existing PASS/FAIL diagnostics for label."""
    return HashComparison(expected, actual, label).check()


def experiment_parser(description: str) -> argparse.ArgumentParser:
    """Return the shared gate parser with unchanged run, seed, port, and timeout defaults."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--port", type=int, default=25599)
    parser.add_argument("--game-port", type=int, default=None,
                        help="Minecraft server port (set it, with --port, to run beside another server)")
    parser.add_argument("--heap", default=None, help="JVM heap cap, e.g. 3G (use with a local model)")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser


def server_options(args: argparse.Namespace) -> dict:
    """Extract the existing Server keyword arguments from parsed gate options."""
    return {"port": args.port, "base_seed": args.base_seed, "game_port": args.game_port, "heap": args.heap,
            "startup_timeout": args.startup_timeout, "request_timeout": args.request_timeout}
