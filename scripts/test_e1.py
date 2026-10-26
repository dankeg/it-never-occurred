"""E1: two independently started 24,000-tick runs from APFS-cloned base."""

from __future__ import annotations

if __package__:
    from . import harness, scripted_routine
else:
    import harness
    import scripted_routine


class E1Experiment:
    """Compare independent runs without changing the absolute-time schedule."""

    def __init__(self, args):
        """Keep parsed CLI options for the experiment phases."""
        self._args = args

    def setup(self) -> None:
        """Reject existing destinations before cloning either fresh run."""
        args = self._args
        self._directories = [args.runs_dir / name for name in ("e1_a", "e1_b")]
        for directory in self._directories:
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {directory}")
        for directory in self._directories:
            harness.clone_apfs(args.runs_dir / "base", directory)
        self._results = []
        self._starts = []

    def run(self) -> None:
        """Run and persist both traces, rejecting unequal starting times."""
        for directory in self._directories:
            with harness.Server(directory, **harness.server_options(self._args)) as server:
                start = harness.enable_lockstep(server.client)
                self._starts.append(start)
                if len(self._starts) == 2 and self._starts[0] != start:
                    raise harness.HarnessError(f"E1 starting game times differ: {self._starts}")
                harness.apply_commands(server.client, scripted_routine.initial_commands())
                hashes = harness.run_routine(server.client, start + harness.RUN_TICKS)
                harness.write_hashes(directory / "hashes.json", hashes)
                self._results.append(hashes)

    def check(self) -> None:
        """Compare traces using the existing comparison and its output."""
        self._passed = harness.compare_hashes(*self._results, "E1")

    def report(self) -> int:
        """Return the exit status for the already-reported comparison."""
        return 0 if self._passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the E1 experiment phases."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = E1Experiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    experiment.check()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
