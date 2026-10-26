"""Run a local model server (MLX) for the agents, e.g. Qwen3.6-35B-A3B in 4-bit.

`MlxServer` starts `mlx_lm.server` from the repository's own `.venv-mlx` environment, with the agent-conversation
prompt-cache policy of `mlx_agent_server.py`, waits until it answers, and stops it cleanly. The server speaks the
OpenAI chat-completions API with tool calling, so the orchestrator's `OpenAICompatibleClient` (provider `mlx-local`)
talks to it directly.

Build the environment once from the pins in `pyproject.toml` (the `local-model` extra):
`python3.11 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm==0.31.3 mlx==0.32.0 transformers==5.15.0`

Usage: `python3.11 scripts/local_model.py` serves in the foreground until interrupted.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request

MLX_PYTHON = Path(__file__).resolve().parents[1] / ".venv-mlx/bin/python"  # built from the local-model pins (C2)
AGENT_SERVER = Path(__file__).resolve().with_name("mlx_agent_server.py")
DEFAULT_MODEL = Path.home() / "models/Qwen3.6-35B-A3B-4bit"


class MlxServer:
    """A local `mlx_lm.server` process: start, wait until ready, stop."""

    def __init__(self, model: Path = DEFAULT_MODEL, port: int = 8080, log: Path | None = None,
                 prompt_cache_size: int = 16, prompt_cache_bytes: str = "3GB"):
        """Configure the server. Up to `prompt_cache_size` checkpoints are kept (each agent needs its system prompt and
        its latest conversation), within `prompt_cache_bytes` of memory."""
        self.model = Path(model)
        self.port = port
        self.url = f"http://127.0.0.1:{port}/v1"
        self._log_path = log
        self._prompt_cache_size = prompt_cache_size
        self._prompt_cache_bytes = prompt_cache_bytes
        self._process: subprocess.Popen | None = None

    def start(self, timeout: float = 300.0) -> MlxServer:
        """Start the server and return once it lists its model; raise if it exits or times out."""
        if not self.model.is_dir():
            raise FileNotFoundError(f"Model not found: {self.model}")
        if not MLX_PYTHON.is_file():
            raise FileNotFoundError(f"No MLX environment at {MLX_PYTHON}: build it as this module's docstring says")
        log = self._log_path.open("ab") if self._log_path else subprocess.DEVNULL
        self._process = subprocess.Popen(
            [str(MLX_PYTHON), str(AGENT_SERVER), "--model", str(self.model), "--port", str(self.port),
             "--prompt-cache-size", str(self._prompt_cache_size), "--prompt-cache-bytes", self._prompt_cache_bytes,
             # One sequence at a time: mlx-lm 0.31.3's batched path corrupts this hybrid model's outputs when several
             # requests are in flight (bench_batching: 7 of 24 garbage replies at 4 in flight); queued requests wait.
             "--decode-concurrency", "1", "--prompt-concurrency", "1",
             "--chat-template-args",
             json.dumps({"enable_thinking": False}), "--log-level", "INFO"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"mlx_lm.server exited with {self._process.returncode}")
            try:
                with request.urlopen(f"{self.url}/models", timeout=2) as response:
                    json.loads(response.read())
                    return self
            except (error.URLError, TimeoutError, ConnectionError):
                time.sleep(1.0)
        self.stop()
        raise TimeoutError(f"mlx_lm.server not ready after {timeout}s")

    def stop(self) -> None:
        """Stop the server process if it is running."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def __enter__(self) -> MlxServer:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    """Serve the model in the foreground until interrupted or terminated (both stop the server process)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    log = Path(__file__).resolve().parents[1] / "runs/mlx_server.log"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # background jobs ignore Ctrl-C; TERM must also clean up
    with MlxServer(args.model, args.port, log=log) as server:
        print(f"serving {server.model.name} at {server.url} (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
