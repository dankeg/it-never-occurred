"""IT-19: a server stops cleanly soon after starting from the village (incident D10: shutdown hung at "Saving worlds").

Each trial clones `village_sabotage`, starts its server, turns lockstep and recording on, restores the bots, moves one
bot far outside the force-loaded area and steps a few ticks (20 to 120, so its chunk tickets start loads that are
still in flight), then stops the server.
A trial hangs if the stop takes more than 30 s: the server thread then spins in vanilla's shutdown loop
(`stopServer` -> `ServerChunkCache.tick` -> `ChunkMap.processUnloads`), waiting for chunk work that only main-thread
tasks can finish. Pass: no trial hangs. The trial count is `--trials` (default 20).
Revert check: before commit edc8492 (lockstep's `haveTime` hook ended every wait once the server was halted, also
inside `stopServer`) 9 of 18 stops hung; with it 0 of 24, each stop taking 1.3 to 3.7 s.
"""

from __future__ import annotations

import time

if __package__:
    from . import harness
else:
    import harness

STOP_LIMIT = 30.0
TICKS = (20, 40, 80, 120)
FAR = (480, 480)  # well outside the force-loaded 320 x 320 blocks around the origin


class ShutdownExperiment:
    """Runs start-restore-step-stop trials on fresh village clones and counts the stops that hang."""

    def __init__(self, args):
        """Keep parsed options."""
        self._args = args

    def trial(self, index: int) -> float | None:
        """One trial; returns the stop's wall time in seconds, or None if it hung (the server was then killed)."""
        directory = self._args.runs_dir / f"it19_trial_{index}"
        harness.clone_apfs(self._args.runs_dir / "village_sabotage", directory)
        server = harness.Server(directory, shutdown_timeout=STOP_LIMIT, **harness.server_options(self._args))
        server.start()
        try:
            harness.enable_lockstep(server.client, recording=True)
            server.client.request("restore_bots")
            # Move one bot far outside the force-loaded area, so its chunk tickets start loads that are still in flight
            # when the stop arrives (as when agents walk off after a fork restore).
            server.client.request("command", {"command": f"tp envoy {FAR[0]} -60 {FAR[1] + 16 * index}"})
            server.client.request("step", {"n": TICKS[index % len(TICKS)]})
        except BaseException:
            server.stop()
            raise
        started = time.monotonic()
        try:
            server.stop()
        except harness.HarnessError:
            return None
        return time.monotonic() - started

    def run(self) -> bool:
        """Run every trial, print each stop time, and return true if none hung."""
        times = []
        for index in range(self._args.trials):
            seconds = self.trial(index)
            times.append(seconds)
            print(f"  {'ok ' if seconds is not None else 'BAD'} trial {index}: {TICKS[index % len(TICKS)]} ticks, "
                  + (f"stopped in {seconds:.1f} s" if seconds is not None else f"hung (killed after {STOP_LIMIT:g} s)"),
                  flush=True)
        hangs = sum(seconds is None for seconds in times)
        passed = hangs == 0
        print(f"{'PASS' if passed else 'FAIL'} IT-19: {hangs} of {len(times)} stops hung")
        return passed


def main(argv: list[str] | None = None) -> int:
    """Parse options, run the trials, and return the exit status."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--trials", type=int, default=20)
    return 0 if ShutdownExperiment(parser.parse_args(argv)).run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
