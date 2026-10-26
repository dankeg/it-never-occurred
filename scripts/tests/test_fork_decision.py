"""Server-free fork-arm validation and continuation tests using the existing scripted world."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from orchestrator.decisions import Decision, DecisionLog
from orchestrator.llm import ScriptedClient, ToolCall
from orchestrator.runtime import DecisionOverride, NIGHT_REST, SimulationRuntime, THINKING_TICKS
from orchestrator.scenario import ScenarioManifest
from scripts import fork_decision, harness
from scripts.fork_decision import DecisionFork, ForkPoint, parse_briefs
from scripts.tests.fake_world import FakeWorld
from scripts.village_session import CHECK_EVERY, VillageSession

ROOT = Path(__file__).resolve().parents[2]


def fork_args(root: Path, **options):
    """Build historical direct-call options, deliberately omitting the new optional attributes."""
    args = harness.experiment_parser("").parse_args(["--runs-dir", str(root)])
    defaults = dict(run="factual", variant="sabotage", agent="builder", tick=1240, horizon=80, epoch_turns=8,
                    swap='{"name": "move_to", "args": {"x": 3, "y": -60, "z": 1}}',
                    arms="control,swap,placebo", forced=False,
                    model="scripted", provider="anthropic", max_calls=None)
    for name, value in {**defaults, **options}.items():
        setattr(args, name, value)
    return args


@pytest.fixture
def logged_point(tmp_path):
    """Supply only the factual decision needed by constructor validation, without any world setup."""
    log = DecisionLog(tmp_path / "factual/orchestrator/decisions.jsonl")
    log.append(Decision(1240, "builder", "digest", False, "model", "wait", {"ticks": 40}, "factual", 0, 0, "prompt"))
    return tmp_path


@pytest.fixture
def village(tmp_path, monkeypatch):
    """Wire real sessions and runtimes to FakeWorld snapshots instead of APFS and server contexts."""
    manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
    initial = FakeWorld({role["name"]: role["spawn"] for role in manifest.roles},
                        containers={tuple(manifest.data["containers"]["records"]["pos"]): []})
    fixture = tmp_path / "village_sabotage"
    (fixture / "world").mkdir(parents=True)
    worlds, runtimes, prepared, sessions = {fixture: copy.deepcopy(initial)}, {}, {}, []
    request = FakeWorld.request

    def snapshot_request(world, command, args=None):
        """Accept the flush command while leaving the existing fake's runtime protocol unchanged."""
        return {} if command == "save_snapshot" else request(world, command, args)

    def clone(session):
        """Copy the source tree and its independent scripted world before entering the session."""
        shutil.copytree(session._source, session.directory)
        worlds[session.directory] = copy.deepcopy(worlds[session._source])

    def clone_world(source, destination):
        """Keep the saved fork's world state alongside the real orchestrator snapshot."""
        shutil.copytree(source, destination)
        worlds[destination.parent] = copy.deepcopy(worlds[source.parent])

    def enter(session):
        """Bind an already cloned FakeWorld without launching a server."""
        sessions.append(session)
        session.client = worlds[session.directory]
        session.start = session.client.game_time
        return session

    factory = DecisionFork._runtime

    def runtime(runner, client, agents, directory, model, override, sample_seed=None):
        """Record real runtime configuration and pre-continuation agent state for assertions."""
        result = factory(runner, client, agents, directory, model, override, sample_seed)
        runtimes[directory] = result
        prepared[directory] = [agent.to_json() for agent in agents]
        return result

    monkeypatch.setattr(FakeWorld, "request", snapshot_request)
    monkeypatch.setattr(VillageSession, "clone", clone)
    monkeypatch.setattr(VillageSession, "__enter__", enter)
    monkeypatch.setattr(VillageSession, "__exit__", lambda session, *exc: None)
    monkeypatch.setattr(harness, "clone_apfs", clone_world)
    monkeypatch.setattr(fork_decision, "setup_server", lambda name, root: (root / name).mkdir())
    monkeypatch.setattr(DecisionFork, "_runtime", runtime)
    monkeypatch.setattr(DecisionFork, "_model", lambda runner: ScriptedClient(
        lambda system, user: ToolCall("wait", {"ticks": 40}), model="continuation"))

    factual = tmp_path / "factual"
    (factual / "world").mkdir(parents=True)
    recorded = SimulationRuntime(initial, manifest.agents(8), ScriptedClient(
        lambda system, user: ToolCall("wait", {"ticks": 40}), model="factual"), factual / "orchestrator",
        command_error=harness.CommandError, hash_region=manifest.data["layout"]["region"], hash_every=CHECK_EVERY,
        min_gap=THINKING_TICKS, rest=NIGHT_REST)
    recorded.run(1200)
    recorded._ready_at["mayor"] = 1200
    checkpoint = factual / "checkpoints/1200"
    (checkpoint / "world").mkdir(parents=True)
    (checkpoint / "orchestrator").mkdir()
    recorded.save_state(checkpoint / "orchestrator/fork_state.json")
    (checkpoint / "orchestrator/first_true.json").write_text('{"bridge_complete": 800}', encoding="utf-8")
    worlds[checkpoint] = copy.deepcopy(initial)
    for name in ("900", "2400", "not-a-tick"):
        (factual / "checkpoints" / name).mkdir()
    recorded.run(1320)
    return SimpleNamespace(root=tmp_path, checkpoint=checkpoint, sessions=sessions, runtimes=runtimes,
                           prepared=prepared, manifest=manifest)


@pytest.mark.parametrize("entry, message", [
    ("builder", "AGENT=TEXT"), ("=secret", "AGENT=TEXT"), ("outsider=secret", "Unknown brief agent"),
    ("builder=", "nonempty"), ("builder=  \t", "nonempty"),
])
def test_brief_entries_are_validated(entry, message):
    """Malformed, unknown and empty replacements fail without needing a run or a model."""
    with pytest.raises(ValueError, match=message):
        parse_briefs([entry], ["builder", "mayor"])


def test_brief_text_preserves_equals_and_whitespace():
    """Split only the first equals sign and retain the exact nonempty replacement text."""
    assert parse_briefs(["builder= a=b ", "mayor=Help."], ["builder", "mayor"]) == {
        "builder": " a=b ", "mayor": "Help."}


@pytest.mark.parametrize("options, message", [
    ({"arms": "control,brief", "swap": None}, "requires at least one --brief"),
    ({"arms": "control,swap", "swap": None}, "--swap is required"),
    ({"arms": "control,swap_hold", "swap": None}, "--swap is required"),
    ({"samples": 0}, "--samples must be positive"),
    ({"samples": -2}, "--samples must be positive"),
    ({"brief": ["outsider=Help."]}, "Unknown brief agent"),
    ({"arms": "control,unknown"}, "Unknown arms"),
])
def test_invalid_options_fail_before_starting(logged_point, options, message):
    """Validate direct namespaces as well as CLI options before creating any arm directory."""
    with pytest.raises(ValueError, match=message):
        DecisionFork(fork_args(logged_point, **options))
    assert not list(logged_point.glob("fork_*"))


@pytest.mark.parametrize("arms", ["control", "placebo", "brief"])
def test_only_swap_arms_require_swap(logged_point, arms):
    """Control, placebo and brief can represent a decision point without a swapped call."""
    runner = DecisionFork(fork_args(logged_point, arms=arms, swap=None, brief=["builder=Help."]))
    assert runner.point.to_json()["swap"] is None


def test_resample_based_arms_share_their_resampling():
    """Resample, brief and mute arms ask the same agents afresh from t; only mute changes perception, nothing holds."""
    point = ForkPoint("factual", "builder", 1240, resample_agents=frozenset({"builder"}),
                      mute=frozenset({("mayor", "builder")}))
    resample, brief, mute = (point.override(arm, ["builder", "mayor"], 80) for arm in ("resample", "brief", "mute"))
    assert resample == brief == DecisionOverride("builder", 1240, resample_agents=frozenset({"builder"}))
    assert mute == DecisionOverride("builder", 1240, resample_agents=frozenset({"builder"}),
                                    mute=frozenset({("mayor", "builder")}))
    for override in (resample, brief, mute):
        assert override.resampling("builder", 1240) and override.resampling("builder", 1280)
        assert not override.resampling("builder", 1200) and not override.resampling("mayor", 1240)
        assert not override.holds("mayor", 1240)
    heard = {"heard": [{"speaker": "builder", "text": "x"}, {"speaker": "freight", "text": "y"}],
             "chatter": [{"speaker": "builder"}], "shout_seen": []}
    assert mute.perceived("mayor", heard, 1240) == {"heard": [{"speaker": "freight", "text": "y"}], "chatter": [],
                                                     "shout_seen": []}
    assert mute.perceived("mayor", heard, 1200) is heard and mute.perceived("builder", heard, 1240) is heard


@pytest.mark.parametrize("samples", [None, 1, 2])
def test_arm_names_seeds_reports_and_resume(village, samples):
    """Sample indices pair every non-control arm; omitted samples preserve historical unseeded requests."""
    options = {"samples": samples} if samples is not None else {}
    runner = DecisionFork(fork_args(village.root, arms="control,placebo,brief,swap,swap_hold",
                                    brief=["builder=Protect the bridge."], **options))
    report = runner.run()
    count = samples or 1
    expected = ["control"] + [f"{arm}_{sample}" if count > 1 else arm
                               for arm in ("placebo", "brief", "swap", "swap_hold")
                               for sample in range(1, count + 1)]
    assert list(report["arms"]) == expected
    assert report["stopped"] is None
    for name, outcome in report["arms"].items():
        assert outcome["arm"] == name
        assert "exposure_from" not in outcome
        runtime = village.runtimes[runner.directory(name)]
        seed = None
        if name != "control" and samples is not None:
            seed = int(name.rsplit("_", 1)[1]) if count > 1 else 1
        assert runtime._sample_seed == seed
        assert json.loads((runner.directory(name) / "outcome.json").read_text()) == outcome
    sessions = len(village.sessions)
    assert runner.run() == report
    assert len(village.sessions) == sessions
    assert json.loads((village.root / f"{runner._prefix}_report.json").read_text()) == report


def test_brief_changes_only_requested_agents_and_reuses_factual_model(village):
    """Unchanged agents replay even with a different continuation model; the changed brief is sampled normally."""
    runner = DecisionFork(fork_args(village.root, arms="control,brief", swap=None,
                                    brief=["builder=Protect the bridge."]))
    report = runner.run()
    _, original = SimulationRuntime.load_agents(runner.directory("fork") / "orchestrator/fork_state.json")
    originals = {agent.name: agent for agent in original}
    runtime = village.runtimes[runner.directory("brief")]
    for snapshot in village.prepared[runner.directory("brief")]:
        expected = originals[snapshot["brief"]["bot"]]
        if expected.name == "builder":
            expected.brief = replace(expected.brief, private="Protect the bridge.")
        assert snapshot == expected.to_json()
    for agent in runtime.agents:
        decision = runtime.log.get(agent.name, 1240)
        assert decision.source == ("model" if agent.name == "builder" else "replay")
        assert decision.model == ("continuation" if agent.name == "builder" else "factual")
    assert report["arms"]["brief"]["calls"] > 0
    assert report["arms"]["brief"]["override_applied"]  # the briefed point agent is resampled from t
    assert report["arms"]["control"]["calls"] == 0
    assert village.sessions[0]._fresh is True


def test_unchanged_brief_matches_the_resample_arm(village):
    """With identical brief text, the brief arm is the resample arm: only the point agent is asked afresh."""
    builder = next(agent for agent in village.manifest.agents(8) if agent.name == "builder")
    runner = DecisionFork(fork_args(village.root, arms="control,brief", swap=None,
                                    brief=[f"builder={builder.brief.private}"]))
    outcome = runner.run()["arms"]["brief"]
    runtime = village.runtimes[runner.directory("brief")]
    assert outcome["calls"] > 0 and outcome["override_applied"]
    assert all(decision.source == "replay" for decision in runtime.log.decisions() if decision.agent != "builder")
    assert all(decision.source == "model" for decision in runtime.log.decisions()
               if decision.agent == "builder" and decision.game_time >= 1240)


@pytest.mark.parametrize("exposure", [None, {"mayor": {"market_ledger": 20}}])
@pytest.mark.parametrize("tick", [1200, 1240])
def test_checkpoint_control_restores_state_timing_and_exposure(village, exposure, tick):
    """Use the latest numeric checkpoint no later than t, propagating incomplete-exposure provenance to all arms."""
    state = village.checkpoint / "orchestrator"
    (village.root / "factual/orchestrator/exposure.json").write_text(
        '{"builder": {"west_supports": 40}}', encoding="utf-8")
    if exposure is not None:
        (state / "exposure.json").write_text(json.dumps(exposure), encoding="utf-8")
    runner = DecisionFork(fork_args(village.root, tick=tick, from_checkpoint=True, samples=2,
                                    arms="control,brief,placebo", swap=None, brief=["builder=Protect the bridge."]))
    report = runner.run()
    control = village.sessions[0]
    assert control._source == village.checkpoint
    assert control._fresh is False
    assert control.start == 1200
    restored = json.loads((state / "fork_state.json").read_text())
    assert village.prepared[runner.directory("control")] == restored["agents"]
    assert village.runtimes[runner.directory("control")]._ready_at["mayor"] == 1200
    assert min(decision.game_time for decision in village.runtimes[runner.directory("control")].log.decisions()) == 1200
    for outcome in report["arms"].values():
        assert outcome["first_true"]["bridge_complete"] == 800
        assert outcome["information"]["builder"] == []
        assert outcome["information"]["mayor"] == ([] if exposure is None else ["market_ledger"])
        if exposure is None:
            assert outcome["exposure_from"] == 1200
        else:
            assert "exposure_from" not in outcome


def test_checkpoint_option_refuses_missing_eligible_checkpoint(logged_point):
    """A factual log alone cannot silently fall back to the fixture when a checkpoint was requested."""
    (logged_point / "factual/checkpoints/2400").mkdir(parents=True)
    with pytest.raises(ValueError, match="No checkpoint"):
        DecisionFork(fork_args(logged_point, from_checkpoint=True))


def test_budget_stop_does_not_start_more_samples_or_arms(village):
    """A stopped sample leaves the control report intact and prevents later pairs from starting."""
    runner = DecisionFork(fork_args(village.root, arms="placebo,brief", samples=2, max_calls=0, swap=None,
                                    brief=["builder=Protect the bridge."]))
    report = runner.run()
    assert list(report["arms"]) == ["control"]
    assert "budget" in report["stopped"].lower()
    assert not runner.directory("placebo_2").exists()
    assert not runner.directory("brief_1").exists()


def test_cli_accepts_checkpoint_repeated_briefs_and_explicit_single_sample(village, monkeypatch, capsys):
    """The CLI exposes the new options without requiring --swap for a brief/placebo comparison."""
    monkeypatch.setattr(harness, "stop_on_sigterm", lambda: None)
    assert fork_decision.main(["--runs-dir", str(village.root), "--run", "factual", "--agent", "builder",
                               "--tick", "1240", "--horizon", "80", "--arms", "control,brief,placebo",
                               "--brief", "builder=Protect the bridge.", "--brief", "mayor=Open the fair.",
                               "--from-checkpoint", "--samples", "1"]) == 0
    assert list(json.loads(capsys.readouterr().out)["arms"]) == ["control", "brief", "placebo"]
    runtime = next(runtime for directory, runtime in village.runtimes.items() if directory.name.endswith("_brief"))
    assert runtime._sample_seed == 1
    assert {agent.name: agent.brief.private for agent in runtime.agents}["mayor"] == "Open the fair."
