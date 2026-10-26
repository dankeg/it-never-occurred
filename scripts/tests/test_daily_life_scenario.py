"""Server-free daily-life manifests, opt-in provisioning and builder configuration."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from orchestrator.scenario import ScenarioManifest
from scripts.build_village import ScenarioConfig, VillageBuilder, main


ROOT = Path(__file__).resolve().parents[2]
LEGACY_KITS = {
    "mayor": {"minecraft:emerald": 64, "minecraft:lantern": 1},
    "builder": {"minecraft:iron_pickaxe": 1, "minecraft:iron_axe": 1,
                "minecraft:iron_shovel": 1, "minecraft:oak_planks": 12},
    "freight": {"minecraft:emerald": 24},
    "security": {"minecraft:cobblestone_wall": 64},
    "envoy": {"minecraft:emerald": 24},
    "investigator": {"minecraft:writable_book": 3},
}


class RecordingVillageClient:
    """Record real builder requests without a server or any running game time."""

    def __init__(self):
        """Start a protocol transcript at the prepared world's fixed game time."""
        self.requests = []

    def request(self, command, args=None):
        """Supply only the deterministic replies needed by the complete village build."""
        self.requests.append((command, args))
        if command == "config_get":
            return {"daily_hunger": True, "notable_blocks": ["minecraft:stone"]}
        if command in ("write_book", "write_sign"):
            return {"prov": f"document:{len(self.requests)}"}
        if command == "ping":
            return {"gameTime": 100}
        if command == "step":
            return {"gameTime": 100 + args["n"]}
        return {}


class DailyLifeScenarioTests(unittest.TestCase):
    """Keep default scenarios unchanged while exposing an explicit daily-life fixture."""

    def test_daily_life_variants_are_selectable_without_starting_servers(self):
        """Every village entry point accepts the generated fixture names before constructing its runner."""
        from scripts import build_decision_dataset, run_narrated_village, run_village, test_m5_village

        cases = ((run_village, "M5VillageExperiment", ["--model", "unused", "--max-calls", "1"]),
                 (run_narrated_village, "NarratedVillage", ["--scripted"]),
                 (test_m5_village, "M5VillageExperiment", []),
                 (build_decision_dataset, "DecisionDatasetBuilder", ["--run", "unused", "--states-only"]))
        for module, runner, options in cases:
            for variant in ("accident_daily_life", "sabotage_daily_life"):
                with self.subTest(entrypoint=module.__name__, variant=variant):
                    with mock.patch.object(module.harness, "stop_on_sigterm"), \
                            mock.patch.object(module, runner, side_effect=RuntimeError("parsed")) as constructor:
                        with self.assertRaisesRegex(RuntimeError, "^parsed$"):
                            module.main(["--variant", variant, *options])
                    self.assertEqual(constructor.call_args.args[0].variant, variant)

    def test_dataset_context_uses_enabled_scenario_tools(self):
        """Offline digest reconstruction uses eating tools only for an enabled manifest."""
        from scripts.build_decision_dataset import DecisionDatasetBuilder

        for variant, enabled in (("sabotage", False), ("sabotage_daily_life", True)):
            with self.subTest(variant=variant):
                args = SimpleNamespace(variant=variant, runs_dir=Path("unused"), run="unused",
                                       prompt_context=None, epoch_turns=8)
                with mock.patch.object(Path, "is_file", return_value=True), \
                        mock.patch("scripts.build_decision_dataset.DecisionLog") as log, \
                        mock.patch("scripts.build_decision_dataset.StateDocuments") as states:
                    log.return_value.decisions.return_value = []
                    DecisionDatasetBuilder(args)
                self.assertEqual(any(tool["name"] == "eat" for tool in states.call_args.args[2]), enabled)

    def test_default_off_preserves_exact_starting_kits(self):
        """Both incident variants and their pre-fixtures retain every legacy endowment."""
        for variant in ("accident", "sabotage"):
            for pre_incident in (False, True):
                with self.subTest(variant=variant, pre_incident=pre_incident):
                    config = ScenarioConfig.create(variant, 17, pre_incident=pre_incident)
                    explicit = ScenarioConfig.create(variant, 17, pre_incident=pre_incident, daily_hunger=False)
                    self.assertIs(config.daily_hunger, False)
                    self.assertIs(config.to_dict()["daily_hunger"], False)
                    self.assertEqual(config.to_dict(), explicit.to_dict())
                    self.assertEqual({role.name: role.kit for role in config.roles}, LEGACY_KITS)

    def test_enabled_adds_only_three_bread_per_role_deterministically(self):
        """Opt-in changes only the flag and food, including before live incident staging."""
        for variant in ("accident", "sabotage"):
            for pre_incident in (False, True):
                with self.subTest(variant=variant, pre_incident=pre_incident):
                    config = ScenarioConfig.create(variant, 17, pre_incident=pre_incident, daily_hunger=True)
                    original = ScenarioConfig.create(variant, 17, pre_incident=pre_incident)
                    self.assertIs(config.daily_hunger, True)
                    self.assertEqual({role.name: role.kit for role in config.roles},
                                     {name: {**kit, "minecraft:bread": 3} for name, kit in LEGACY_KITS.items()})
                    restored = config.to_dict()
                    restored["daily_hunger"] = False
                    for role in restored["roles"]:
                        role["kit"].pop("minecraft:bread")
                    self.assertEqual(restored, original.to_dict())
                    repeated = ScenarioConfig.create(variant, 17, pre_incident=pre_incident, daily_hunger=True)
                    self.assertEqual(config.to_dict(), repeated.to_dict())
                    self.assertEqual({role.name: role.kit for role in original.roles}, LEGACY_KITS)

    def test_written_manifests_include_flag_and_conditional_food(self):
        """Server-free manifest writing retains the configured flag and role inventories."""
        with tempfile.TemporaryDirectory() as directory:
            for daily_hunger in (False, True):
                with self.subTest(daily_hunger=daily_hunger):
                    config = ScenarioConfig.create("sabotage", 17, daily_hunger=daily_hunger)
                    path = Path(directory) / config.fixture_name / "scenario.json"
                    config.write(path)
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self.assertIs(data["daily_hunger"], daily_hunger)
                    self.assertEqual(data, json.loads(json.dumps(config.to_dict())))
                    self.assertEqual([role["kit"].get("minecraft:bread", 0) for role in data["roles"]],
                                     [3 if daily_hunger else 0] * 6)

    def test_checked_in_defaults_and_sprint_manifest(self):
        """Legacy manifests opt out; the sprint manifest is reproducible without fabricated setup IDs."""
        for variant in ("accident", "sabotage"):
            with self.subTest(variant=variant):
                data = ScenarioManifest.load(ROOT / "scenario" / f"village_{variant}.json").data
                self.assertIs(data["daily_hunger"], False)
                self.assertEqual({role["name"]: role["kit"] for role in data["roles"]}, LEGACY_KITS)
        config = ScenarioConfig.create("sabotage", daily_hunger=True)
        data = ScenarioManifest.load(ROOT / "scenario" / f"{config.fixture_name}.json").data
        self.assertIs(data["daily_hunger"], True)
        self.assertEqual(data, json.loads(json.dumps(config.to_dict())))
        self.assertNotIn("setup", data["layout"])

    def test_missing_and_false_flag_keep_role_briefs_identical(self):
        """An explicit off field cannot change any public, private or shared prompt text."""
        for variant in ("accident", "sabotage"):
            with self.subTest(variant=variant):
                data = ScenarioManifest.load(ROOT / "scenario" / f"village_{variant}.json").data
                legacy = copy.deepcopy(data)
                legacy.pop("daily_hunger")
                self.assertEqual(ScenarioManifest(data).briefs(), ScenarioManifest(legacy).briefs())
                current_agents = ScenarioManifest(data).agents(epoch_turns=8)
                legacy_agents = ScenarioManifest(legacy).agents(epoch_turns=8)
                self.assertEqual([agent.system_prompt().encode("utf-8") for agent in current_agents],
                                 [agent.system_prompt().encode("utf-8") for agent in legacy_agents])

    def test_fixture_names_do_not_overwrite_legacy_builds(self):
        """Daily-life and pre-incident output names leave existing fixture paths unchanged."""
        for variant in ("accident", "sabotage"):
            for daily_hunger in (False, True):
                for pre_incident in (False, True):
                    with self.subTest(variant=variant, daily_hunger=daily_hunger, pre_incident=pre_incident):
                        config = ScenarioConfig.create(variant, daily_hunger=daily_hunger, pre_incident=pre_incident)
                        expected = f"village_{variant}" + ("_daily_life" if daily_hunger else "")
                        self.assertEqual(config.fixture_name, expected + ("_pre" if pre_incident else ""))

    def test_cli_daily_hunger_is_explicit_and_default_off(self):
        """The opt-in CLI selects daily-life configuration without launching Minecraft."""
        for options, expected in (([], False), (["--daily-hunger"], True)):
            with self.subTest(options=options):
                with mock.patch("scripts.build_village.harness.stop_on_sigterm"), \
                        mock.patch.object(VillageBuilder, "run", autospec=True, return_value=Path("unused")) as run, \
                        mock.patch("builtins.print"):
                    self.assertEqual(main(["--variant", "sabotage", *options]), 0)
                self.assertIs(run.call_args.args[0].config.daily_hunger, expected)


class DailyLifeBuilderTests(unittest.TestCase):
    """Exercise the complete builder protocol with a deterministic recording client."""

    def test_build_applies_flag_and_only_enabled_roles_receive_food(self):
        """Configuration resets inherited hunger; only opt-in adds food give commands."""
        transcripts = {}
        for daily_hunger in (False, True):
            with self.subTest(daily_hunger=daily_hunger):
                config = ScenarioConfig.create("sabotage", 17, daily_hunger=daily_hunger)
                client = RecordingVillageClient()
                VillageBuilder(config).build(client)
                settings = [args for command, args in client.requests if command == "config_set"]
                self.assertEqual(len(settings), 1)
                self.assertIs(settings[0]["daily_hunger"], daily_hunger)
                self.assertEqual(settings[0]["notable_blocks"], sorted(config.layout["notable_additions"]))
                food = [args for command, args in client.requests
                        if command == "give" and args["item"] == "minecraft:bread"]
                self.assertEqual(food, [{"name": role.name, "item": "minecraft:bread", "count": 3}
                                        for role in config.roles] if daily_hunger else [])
                self.assertLess(next(index for index, (command, _) in enumerate(client.requests)
                                     if command == "config_set"),
                                next(index for index, (command, _) in enumerate(client.requests)
                                     if command == "spawn_bot"))
                transcripts[daily_hunger] = [(command, {**args, "daily_hunger": False}
                                             if command == "config_set" else args)
                                            for command, args in client.requests
                                            if not (command == "give" and args["item"] == "minecraft:bread")]
        self.assertEqual(transcripts[False], transcripts[True])


if __name__ == "__main__":
    unittest.main()
