"""A local run queue: experiments run detached from the editor, within a server budget, with ports assigned.

Each job is a command line saved under `runs/queue/jobs/<name>.json` with the number of Minecraft servers it runs at
once. `serve` starts jobs in submission order while their servers fit the budget, gives each job its own block of 100
ports (`{port}` in the command is the block's first port, `{port+N}` the N-th after it), and writes a status file and
a log per job under `runs/queue/`. With `--detach` the server runs in its own session, so it outlives the shell that
started it. Our runners resume on a rerun (finished runs and branches are reused), so a failed job is resubmitted, not
repaired. A job starts only while the disk has `MIN_FREE_BYTES` free (a full disk otherwise fails every queued job in
turn); the budget can be changed while serving by writing a number to `runs/queue/max_servers`.

Usage:
  python3.11 scripts/run_queue.py submit NAME [--servers K] -- COMMAND ...
  python3.11 scripts/run_queue.py serve [--max-servers 5] [--detach]
  python3.11 scripts/run_queue.py status
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs" / "queue"
FIRST_PORT, BLOCK = 26_000, 100
MIN_FREE_BYTES = 15 * 10**9  # above the harness's 10 GB guard, so a started job does not fail on its own disk check
PORT = re.compile(r"\{port(?:\+(\d+))?\}")


def now() -> str:
    """The local time, for status files."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Job:
    """One queued command: its spec (`jobs/<name>.json`), status (`<name>.status.json`) and log (`<name>.log`)."""

    def __init__(self, name: str, queue: Path = QUEUE):
        """Bind the job's files under the queue directory."""
        self.name = name
        self.spec_path = queue / "jobs" / f"{name}.json"
        self.status_path = queue / f"{name}.status.json"
        self.log_path = queue / f"{name}.log"

    @property
    def spec(self) -> dict:
        """The submitted command, server count and submission time."""
        return json.loads(self.spec_path.read_text(encoding="utf-8"))

    @property
    def status(self) -> dict:
        """The job's state: queued, running, done or failed, with times, pid, ports and exit code."""
        if not self.status_path.exists():
            return {"state": "queued"}
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def write_status(self, **fields) -> None:
        """Merge fields into the status file."""
        self.status_path.write_text(json.dumps({**self.status, **fields}, indent=1) + "\n", encoding="utf-8")

    def command(self, base: int) -> list[str]:
        """The command with `{port}` and `{port+N}` replaced from the job's port block."""
        return [PORT.sub(lambda match: str(base + int(match[1] or 0)), part) for part in self.spec["command"]]


class RunQueue:
    """Starts queued jobs within a server budget and tracks the running ones until the queue is empty."""

    def __init__(self, max_servers: int, queue: Path = QUEUE, min_free: int = MIN_FREE_BYTES):
        """Keep the budget and the disk floor; jobs are read from the queue directory on every pass."""
        self._max_servers = max_servers
        self._queue = queue
        self._min_free = min_free
        self._running: dict[str, tuple[subprocess.Popen, int]] = {}  # name -> (process, port block)

    def jobs(self) -> list[Job]:
        """Every submitted job, oldest first."""
        jobs = [Job(path.stem, self._queue) for path in (self._queue / "jobs").glob("*.json")]
        return sorted(jobs, key=lambda job: job.spec["submitted"])

    def _free_block(self) -> int:
        """The lowest port block no running job holds."""
        held = {block for _, block in self._running.values()}
        return next(FIRST_PORT + BLOCK * index for index in range(100) if FIRST_PORT + BLOCK * index not in held)

    def _reap(self) -> None:
        """Record the exit of finished jobs."""
        for name, (process, _) in list(self._running.items()):
            code = process.poll()
            if code is not None:
                Job(name, self._queue).write_status(state="done" if code == 0 else "failed", exit=code, ended=now())
                del self._running[name]

    def _start(self, job: Job) -> None:
        """Start one job with its own port block, logging to its log file."""
        block = self._free_block()
        with job.log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(job.command(block), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
        self._running[job.name] = (process, block)
        job.write_status(state="running", pid=process.pid, ports=block, started=now(), command=job.command(block))

    def budget(self) -> int:
        """The server budget: `runs/queue/max_servers` if it holds a number, else the one given at start."""
        override = self._queue / "max_servers"
        try:
            return int(override.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return self._max_servers

    def disk_ok(self) -> bool:
        """Whether the disk holding the queue has room to start another job."""
        return shutil.disk_usage(self._queue).free >= self._min_free

    def serve(self, poll: float = 10.0) -> None:
        """Run until no job is queued or running. Jobs left `running` by a lost server are marked failed."""
        for job in self.jobs():
            if job.status["state"] == "running" and job.name not in self._running:
                job.write_status(state="failed", exit=None, ended=now(), note="the queue server stopped")
        while True:
            self._reap()
            used = sum(Job(name, self._queue).spec["servers"] for name in self._running)
            budget = self.budget()
            for job in self.jobs():
                if (job.status["state"] == "queued" and used + job.spec["servers"] <= budget
                        and self.disk_ok()):
                    self._start(job)
                    used += job.spec["servers"]
            if not self._running and not any(job.status["state"] == "queued" for job in self.jobs()):
                return
            time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    """Submit a job, serve the queue (optionally detached), or print every job's status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("name")
    submit.add_argument("--servers", type=int, default=1, help="Minecraft servers the job runs at once")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    serve = commands.add_parser("serve")
    serve.add_argument("--max-servers", type=int, default=5)
    serve.add_argument("--detach", action="store_true", help="run in a new session that outlives this shell")
    commands.add_parser("status")
    args = parser.parse_args(argv)
    (QUEUE / "jobs").mkdir(parents=True, exist_ok=True)
    if args.action == "submit":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        job = Job(args.name)
        if job.spec_path.exists():
            parser.error(f"job {args.name} exists; choose another name")
        job.spec_path.write_text(json.dumps({"command": command, "servers": args.servers, "submitted": now()},
                                            indent=1) + "\n", encoding="utf-8")
        print(f"queued {args.name}")
    elif args.action == "serve" and args.detach:
        with (QUEUE / "serve.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, __file__, "serve", "--max-servers", str(args.max_servers)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
        print(f"queue server pid {process.pid}")
    elif args.action == "serve":
        (QUEUE / "serve.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
        RunQueue(args.max_servers).serve()
    else:
        for job in RunQueue(0).jobs():
            status = job.status
            print(f"{job.name:40s} {status['state']:8s} servers={job.spec['servers']} ports={status.get('ports')} "
                  f"started={status.get('started')} ended={status.get('ended')} exit={status.get('exit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
