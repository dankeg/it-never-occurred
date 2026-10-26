"""E2: clone the live, saved, lockstep-frozen world and resume in a fresh JVM."""

from __future__ import annotations

import time

if __package__:
    from . import harness, scripted_routine
    from .setup_server import setup_server
else:
    import harness
    import scripted_routine
    from setup_server import setup_server


class E2Experiment:
    """Fork a frozen live world and compare the unchanged absolute-time suffix."""

    def __init__(self, args):
        """Keep parsed CLI options for both server lifecycles."""
        self._args = args

    def setup(self) -> None:
        """Preflight both paths, create C's shell, then clone A from base."""
        args = self._args
        self._directory_a = args.runs_dir / "e2_a"
        self._directory_c = args.runs_dir / "e2_c"
        for directory in (self._directory_a, self._directory_c):
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        self._directory_c = setup_server("e2_c", args.runs_dir)
        harness.clone_apfs(args.runs_dir / "base", self._directory_a)

    def run(self) -> None:
        """Save and clone A while frozen, finish A, then restore and run C."""
        args = self._args
        directory_a, directory_c = self._directory_a, self._directory_c
        with harness.Server(directory_a, **harness.server_options(args)) as server:
            start = harness.enable_lockstep(server.client)
            snapshot_time = start + harness.FORK_TICKS
            end_time = start + harness.RUN_TICKS
            harness.apply_commands(server.client, scripted_routine.initial_commands())
            hashes_a = harness.run_routine(server.client, snapshot_time)
            before_usage = harness.disk_usage(args.runs_dir)
            started = time.perf_counter()
            server.client.request("save_snapshot")
            if harness.game_time(server.client.request("ping")) != snapshot_time:
                raise harness.HarnessError("A advanced during save_snapshot")
            harness.clone_apfs(directory_a / "world", directory_c / "world")
            latency = time.perf_counter() - started
            after_usage = harness.disk_usage(args.runs_dir)
            if harness.game_time(server.client.request("ping")) != snapshot_time:
                raise harness.HarnessError("A advanced while cloning its frozen world")
            print(f"Fork latency (save + APFS world clone): {latency:.6f}s")
            print(f"df -k before clone:\n{before_usage}")
            print(f"df -k after clone:\n{after_usage}")
            hashes_a.extend(harness.run_routine(server.client, end_time))
            harness.write_hashes(directory_a / "hashes.json", hashes_a)
        with harness.Server(directory_c, **harness.server_options(args)) as server:
            restored_time = harness.enable_lockstep(server.client)
            if restored_time != snapshot_time:
                raise harness.HarnessError(f"C started at {restored_time}, expected {snapshot_time}")
            server.client.request("restore_bots")
            hashes_c = harness.run_routine(server.client, end_time)
            harness.write_hashes(directory_c / "hashes.json", hashes_c)
        self._hashes_a, self._hashes_c = hashes_a, hashes_c
        self._snapshot_time = snapshot_time

    def check(self) -> None:
        """Compare only post-snapshot hashes, retaining the comparison output."""
        expected = [record for record in self._hashes_a if record["gameTime"] > self._snapshot_time]
        self._passed = harness.compare_hashes(expected, self._hashes_c, "E2")

    def report(self) -> int:
        """Return the exit status for the already-reported comparison."""
        return 0 if self._passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the E2 experiment phases."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = E2Experiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    experiment.check()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
