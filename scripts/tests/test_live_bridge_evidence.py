"""Pure IT-09 coverage for real collapse provenance, witness sightings and live fixture reporting."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.scenario import ScenarioManifest
from scripts import harness
from scripts.build_village import ScenarioConfig
from scripts.provenance import CodeState, FixtureMetadata
from scripts.test_it09_predicates import PredicateExperiment, pickaxe_available
from scripts.tests.fake_world import FakeWorld


class WitnessWorld(FakeWorld):
    """Supply only physical fixture evidence and recorded testimony to the real IT-09 evidence procedure."""

    def __init__(self, manifest, states, testimony):
        """Retain independent world evidence, including the surviving live item after vanilla movement."""
        super().__init__({}, game_time=6000)
        self.states, self.testimony = deepcopy(states), deepcopy(testimony)
        self.entities = [{"type": "minecraft:villager", "uuid": state["uuid"]} for state in states.values()]
        tool = manifest["evidence"]["tool"]
        if tool:
            self.entities.append({"type": "minecraft:item", "uuid": tool["uuid"], "pos": "-8.1,-60,-3.4",
                                  "nbt": '{Item:{id:"minecraft:iron_pickaxe",count:1}}'})
        document = next(document for document in manifest["documents"] if document["id"] == "market_ledger")
        position = manifest["containers"][document["container"]]["pos"]
        self.containers[tuple(position)] = [{**document, "item": "minecraft:written_book"}]

    def request(self, command, args=None):
        """Answer evidence APIs without sockets, processes or any model and delegate ordinary reads."""
        if command == "save_snapshot":
            return {"gameTime": self.game_time}
        if command == "dump_entities":
            return {"entities": deepcopy(self.entities)}
        if command == "npc_state":
            return deepcopy(self.states[args["id"]])
        if command == "npc_testimony":
            return {"records": deepcopy(self.testimony[args["id"]])}
        return super().request(command, args)


class LiveBridgeEvidenceTests(unittest.TestCase):
    """Require collapse evidence, not a status shortcut or every witness seeing submerged supports."""

    def setUp(self):
        """Create exact protocol-shaped recorded removals, separate commuter sightings and one news holder."""
        self.manifest = json.loads(json.dumps(ScenarioConfig.create("sabotage", 17, pre_incident=True).to_dict()))
        self.events = [{"type": "block_change", "id": 101 + index, "gameTime": 1000 + index,
                        "pos": support["pos"], "from": support["block"], "to": "minecraft:air", "actor": "envoy"}
                       for index, support in enumerate(self.manifest["evidence"]["supports"])]
        supports = len(self.manifest["evidence"]["supports"])
        last = 100 + supports  # the last support change triggers the collapse
        deck = sorted(self.manifest["success_predicates"]["builder"]["all"][0]["blocks_present"],
                      key=lambda block: block["pos"])
        self.deck = [{"type": "block_change", "id": 201 + index, "gameTime": 1000 + supports - 1,
                      "pos": block["pos"], "from": block["block"], "to": "minecraft:air", "actor": "world",
                      "cause": "collapse", "causeEventId": last} for index, block in enumerate(deck)]
        self.drop = {"type": "drop", "id": 302, "gameTime": 1260, "bot": "envoy", "itemEntity": "tool-uuid",
                     "thrower": "envoy-uuid", "item": "minecraft:iron_pickaxe", "count": 1,
                     "pos": [-8.5, -60, -3.5]}
        collapse = {"type": "collapse", "id": 301, "gameTime": 1000 + supports - 1, "name": "west_bridge",
                    "causeEventId": last, "changes": [event["id"] for event in self.deck]}
        self.events += [*self.deck, collapse, self.drop]
        self.events.append({"type": "purchase", "id": 99, "gameTime": 900, "buyer": "envoy",
                            "item": "minecraft:iron_pickaxe", "count": 1, "cost": 8,
                            "shop": "container:" + ",".join(map(str, self.manifest["containers"]["stall_3"]["pos"]))})
        self.manifest["evidence"]["tool"].update(uuid="tool-uuid", thrower="envoy-uuid")
        self.manifest["evidence"]["tool"].pop("tag")
        self.news = {"newsId": "bridge-news-1", "originEventId": "change:201", "firsthand": True,
                     "kind": "bridge_or_port_change", "subject": "the bridge", "learnedAt": 1020,
                     "present": ["envoy"]}
        self.manifest["ground_truth"].update(
            status="staged_live", support_change_ids=[f"change:{101 + index}" for index in range(supports)],
            collapse_event_id=301, deck_change_ids=[f"change:{event['id']}" for event in self.deck],
            witness_interval={"since": 990, "until": 1250}, drop_event_id=302,
            witness_news_ids={"adult_01": [self.news["newsId"]]}, witness_news={"adult_01": [deepcopy(self.news)]})
        self.states = {witness: {"id": witness, "uuid": witness + "-uuid",
                                 "news": [deepcopy(self.news)] if witness == "adult_01" else []}
                       for witness in self.manifest["evidence"]["witnesses"]}
        envoy = {"name": "envoy", "prov": "entity:envoy-uuid", "pos": [-7.5, -60, 0.5], "losRay": 0}
        self.testimony = {witness: [{"npc": witness, "gameTime": 1020 if witness == "adult_01" else 1220,
                                    "entities": [deepcopy(envoy)], "changes":
                                    [{**self.deck[0], "prov": "change:201", "losRay": 0}]
                                    if witness == "adult_01" else []}] for witness in self.states}

    def _available(self, *, manifest=None, events=None, states=None, testimony=None):
        """Run the real IT-09 witness branch, reusing staging's pure validator rather than another checker."""
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            data = deepcopy(self.manifest if manifest is None else manifest)
            data["evidence"]["expected_chain"] = ["witness_observations"]
            experiment.manifest = ScenarioManifest(data)
            experiment.client = WitnessWorld(data, self.states if states is None else states,
                                               self.testimony if testimony is None else testimony)
            if events is not None:
                experiment.events.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                experiment._evidence()
            return next(check["actual"] for check in experiment.checks
                        if check["label"] in {"PASS witness_observations", "FAIL witness_observations"})

    def test_separate_commuters_and_only_one_firsthand_news_holder_pass(self):
        """Neither witness sees a submerged support, and the later witness need not see a deck change."""
        self.assertTrue(self._available())
        self.assertEqual(self.states["adult_02"]["news"], [])
        self.assertEqual(self.testimony["adult_02"][0]["changes"], [])
        self.assertFalse(self._available(events=[]))

    def test_each_sabotage_witness_needs_a_real_envoy_ray_inside_the_window(self):
        """Name-only claims, another entity and out-of-window sightings cannot pass."""
        for witness in self.states:
            for changes in ({"entities": []}, {"gameTime": 1251}, {"npc": "not-the-witness"},
                            {"entities": [{"name": "envoy", "prov": "entity:envoy-uuid"}]},
                            {"entities": [{"name": "envoy", "prov": "entity:wrong", "losRay": 0,
                                            "pos": [-7.5, -60, 0.5]}]}):
                with self.subTest(witness=witness, changes=changes):
                    testimony = deepcopy(self.testimony)
                    testimony[witness][0].update(changes)
                    self.assertFalse(self._available(testimony=testimony))

    def test_status_cannot_replace_complete_causal_recorded_deck_removals(self):
        """Missing, support-only, actor-mined, wrongly linked or incomplete collapse records fail."""
        for event_id in (101, 201, 301):
            with self.subTest(missing_event=event_id):
                self.assertFalse(self._available(events=[event for event in self.events if event["id"] != event_id]))
        for event_id, changes in ((201, {"actor": "envoy"}), (201, {"cause": "mine"}),
                                  (201, {"causeEventId": 101}), (201, {"to": "minecraft:water"}),
                                  (201, {"from": "minecraft:air"}), (301, {"changes": [201]}),
                                  (301, {"causeEventId": 999}), (301, {"name": "the port"})):
            with self.subTest(event_id=event_id, changes=changes):
                events = [{**event, **changes} if event["id"] == event_id else event for event in self.events]
                self.assertFalse(self._available(events=events))

    def test_news_requires_real_deck_origin_and_same_observation_present_names(self):
        """Even matching manifest claims cannot excuse hearsay, support news or unsupported present names."""
        for changes in ({"firsthand": False}, {"subject": "the port"}, {"kind": "construction"},
                        {"originEventId": "change:101"}, {"originEventId": "change:999"},
                        {"learnedAt": 1021}, {"present": []}, {"present": ["envoy", "mayor"]},
                        {"present": ["envoy", "envoy"]}):
            with self.subTest(changes=changes):
                manifest, states = deepcopy(self.manifest), deepcopy(self.states)
                states["adult_01"]["news"][0].update(changes)
                manifest["ground_truth"]["witness_news"]["adult_01"][0].update(changes)
                self.assertFalse(self._available(manifest=manifest, states=states))
        for changes in ({"changes": []}, {"changes": [{**self.deck[0], "prov": "change:201"}]},
                        {"changes": [{**self.deck[0], "id": 999, "prov": "change:999", "losRay": 0}]}):
            with self.subTest(record_changes=changes):
                testimony = deepcopy(self.testimony)
                testimony["adult_01"][0].update(changes)
                self.assertFalse(self._available(testimony=testimony))
        testimony = deepcopy(self.testimony)
        testimony["adult_01"].append({**deepcopy(testimony["adult_01"][0]), "gameTime": 1010})
        testimony["adult_01"][0]["entities"] = []
        self.assertFalse(self._available(testimony=testimony))

    def test_live_fixture_needs_an_explicit_observation_window(self):
        """Missing or incomplete staging bounds fail the gate, without inventing a different window."""
        for interval in ({}, {"since": 990}, {"until": 1250}):
            with self.subTest(interval=interval):
                manifest = deepcopy(self.manifest)
                manifest["ground_truth"]["witness_interval"] = interval
                self.assertFalse(self._available(manifest=manifest))

    def test_accident_requires_observed_collapse_news_with_no_present_agent(self):
        """The same event join admits actorless accidents but rejects an empty present claim hiding an agent."""
        self.manifest["variant"] = "accident"
        self.manifest["evidence"]["tool"] = None
        for event in self.events[:len(self.manifest["evidence"]["supports"])]:
            event["actor"] = "world"
        self.events = [event for event in self.events if event["type"] not in {"drop", "purchase"}]
        self.states["adult_01"]["news"][0]["present"] = []
        self.manifest["ground_truth"]["witness_news"]["adult_01"][0]["present"] = []
        self.assertFalse(self._available())
        for records in self.testimony.values():
            records[0]["entities"] = []
        self.assertTrue(self._available())
        self.states["adult_01"]["news"] = []
        self.assertFalse(self._available())

    def _experiment(self, directory):
        """Load the emitted live manifest and canonical provenance metadata through the real constructor."""
        fixture = directory / "village_sabotage"
        fixture.mkdir()
        manifest = fixture / "scenario.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        state = CodeState("commit", False, "", (), "jar-hash")
        metadata = FixtureMetadata.write(fixture, state, manifest=manifest, world_hash="world-hash")
        args = harness.experiment_parser("").parse_args(["--runs-dir", str(directory), "--port", "31000"])
        args.llm_judge_url = None
        args.llm_judge_model = "local"
        experiment = PredicateExperiment(args)
        self.assertEqual(experiment.manifest.data, self.manifest)
        self.assertEqual(experiment.fixture_metadata, json.loads(json.dumps(metadata)))
        experiment.events = directory / "events.jsonl"
        experiment.events.write_text("".join(json.dumps(event) + "\n" for event in self.events), encoding="utf-8")
        experiment.client = WitnessWorld(self.manifest, self.states, self.testimony)
        return experiment

    def test_live_it09_procedure_really_passes_and_missing_rays_fail_the_gate(self):
        """The real evidence loop passes a live fixture; status alone cannot convert missing testimony to XFAIL."""
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            output = io.StringIO()
            with redirect_stdout(output):
                experiment._evidence()
            self.assertTrue(all(check["passed"] for check in experiment.checks), experiment.checks)
            self.assertIn("PASS witness_observations", output.getvalue())
            self.assertNotIn("XFAIL", output.getvalue())
            experiment.client.testimony["adult_02"][0]["entities"] = []
            experiment.checks.clear()
            output = io.StringIO()
            with redirect_stdout(output):
                experiment._evidence()
            self.assertFalse(all(check["passed"] for check in experiment.checks))
            self.assertIn("FAIL witness_observations", output.getvalue())
            self.assertNotIn("XFAIL", output.getvalue())

    def test_live_pickaxe_joins_surviving_uuid_and_recorded_envoy_thrower(self):
        """A real dropped item moves and has no setup tag; another thrower or a missing drop still fails."""
        tool = self.manifest["evidence"]["tool"]
        entity = {"type": "minecraft:item", "uuid": tool["uuid"], "pos": "0,-61,0", "nbt": tool["item"]}
        self.assertTrue(pickaxe_available(tool, [entity], self.drop))
        self.assertFalse(pickaxe_available(tool, [entity]))
        self.assertFalse(pickaxe_available(tool, [{**entity, "uuid": "another"}], self.drop))
        for changes in ({"thrower": "mayor-uuid"}, {"bot": "mayor"}, {"itemEntity": "another"}, {"count": 2}):
            with self.subTest(changes=changes):
                self.assertFalse(pickaxe_available(tool, [entity], {**self.drop, **changes}))

    def test_only_older_fixture_uses_the_named_witness_expected_failure(self):
        """Setup-only evidence retains exactly its named W4 line, not an exemption for live fixtures."""
        self.manifest["ground_truth"]["status"] = "staged_before_recording"
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            output = io.StringIO()
            with redirect_stdout(output):
                experiment._evidence()
            self.assertTrue(all(check["passed"] for check in experiment.checks), experiment.checks)
            self.assertIn("XFAIL witness_observations (W4, older fixture)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
