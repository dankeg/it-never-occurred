"""Focused offline behavior tests for perception features and exposure tracking."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.candidates import Candidate
from orchestrator.llm import ToolCall
from orchestrator.perception import DecisionContext, EvidenceExposure, PerceptionFeatures, PerceptionTap, VillagerTracks
from orchestrator.scenario import ScenarioManifest

ROOT = Path(__file__).resolve().parents[2]


def observation(position=(-6, -60, 0), game_time=100, *, entities=(), blocks=(), heard=()):
    """Describe a small raw observation near the bridge with only the supplied facts."""
    return {"gameTime": game_time, "self": {"pos": list(position)}, "entities": list(entities),
            "blocks": list(blocks), "heard": list(heard)}


def villager_record(game_time, position=(-6, -60, 0), *, npc="adult_01", names=()):
    """Describe one villager observation, expressing the supplied feet position as an eye."""
    return {"gameTime": game_time, "npc": npc,
            "observerEye": [position[0], position[1] + 1.62, position[2]],
            "entities": [{"name": name} for name in names]}


class RecordingClient:
    """An in-memory harness client that records requests and returns queued responses."""

    def __init__(self, responses=(), error=None):
        """Store responses or an exception without connecting to a harness."""
        self.responses = iter(responses)
        self.error = error
        self.requests = []

    def request(self, command, args=None):
        """Record the exact arguments, then return a response or raise the configured error."""
        self.requests.append((command, args))
        if self.error is not None:
            raise self.error
        return next(self.responses)


class EvidenceExposureTest(unittest.TestCase):
    """Evidence is learned only from the specified sources and at their specified times."""

    def setUp(self):
        """Load real village evidence and start an empty exposure history."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.exposure = EvidenceExposure(self.manifest)

    def test_support_position_becomes_known_at_observation_time(self):
        """Seeing a support coordinate suffices, regardless of its observed block type."""
        self.exposure.observe("builder", observation(blocks=[{"pos": [-5, -63, 0]}]))
        self.assertEqual(self.exposure.items("builder", 100), frozenset())
        support = self.manifest.data["evidence"]["supports"][0]["pos"]
        self.exposure.observe("builder", observation(blocks=[{"pos": support, "block": "minecraft:oak_planks"}]))
        self.assertEqual(self.exposure.items("builder", 99), frozenset())
        self.assertEqual(self.exposure.items("builder", 100), frozenset({"west_supports"}))

    def test_testimony_requires_a_villager_and_both_words(self):
        """Only villager speech containing envoy and bridge counts, using the speech's own tick."""
        speaker = self.manifest.data["npcs"][-1]["name"]
        rejected = [{"speaker": "builder", "text": "envoy at the bridge", "gameTime": 10},
                    {"speaker": speaker, "text": "envoy at the market", "gameTime": 20},
                    {"speaker": speaker, "text": "the bridge collapsed", "gameTime": 30}]
        self.exposure.observe("investigator", observation(heard=rejected))
        self.assertEqual(self.exposure.items("investigator", 100), frozenset())
        self.exposure.observe("investigator", observation(heard=[
            {"speaker": speaker, "text": "The ENVOY was beside the BRIDGE", "gameTime": 40}]))
        self.assertEqual(self.exposure.items("investigator", 39), frozenset())
        self.assertEqual(self.exposure.items("investigator", 40), frozenset({"witness_observations"}))

    def test_ledger_entry_is_known_only_after_the_read_decision(self):
        """A case-insensitive full ledger entry becomes known one tick after reading it."""
        self.exposure.read("investigator", "an unrelated ledger page", 100)
        self.assertEqual(self.exposure.items("investigator", 101), frozenset())
        entry = self.manifest.data["evidence"]["ledger_entry"]
        self.exposure.read("investigator", f"Market records: {entry.upper()}.", 100)
        self.assertEqual(self.exposure.items("investigator", 100), frozenset())
        self.assertEqual(self.exposure.items("investigator", 101), frozenset({"market_ledger"}))

    def test_repeated_exposure_keeps_the_earliest_time(self):
        """Later and out-of-order observations cannot displace the earliest exposure."""
        support = self.manifest.data["evidence"]["supports"][0]["pos"]
        for game_time in (200, 400, 100):
            self.exposure.observe("builder", observation(game_time=game_time, blocks=[{"pos": support}]))
        self.assertEqual(self.exposure.to_json(), {"builder": {"west_supports": 100}})

    def test_json_round_trip_preserves_independent_agent_histories(self):
        """Restoration and exported dictionaries preserve times without sharing mutable histories."""
        known = {"builder": {"west_supports": 80}, "investigator": {"market_ledger": 101}}
        restored = EvidenceExposure(self.manifest, json.loads(json.dumps(known)))
        snapshot = restored.to_json()
        self.assertEqual(snapshot, known)
        self.assertEqual(restored.items("builder", 100), frozenset({"west_supports"}))
        self.assertEqual(restored.items("investigator", 100), frozenset())
        self.assertEqual(restored.items("unknown", 100), frozenset())
        snapshot["builder"]["west_supports"] = 999
        self.assertEqual(restored.to_json(), known)
        copied = EvidenceExposure(self.manifest, known)
        known["builder"]["west_supports"] = 999
        self.assertEqual(copied.items("builder", 80), frozenset({"west_supports"}))


class VillagerTracksTest(unittest.TestCase):
    """Logged sightings and feet positions use only the latest nonfuture record."""

    def test_position_uses_latest_record_at_or_before_the_tick(self):
        """Unsorted input is queried inclusively without interpolating or looking ahead."""
        tracks = VillagerTracks([villager_record(300, (6, -60, 0)), villager_record(100)])
        self.assertEqual(tracks.position("adult_01", 299), (-6.0, -60.0, 0.0))
        self.assertEqual(tracks.position("adult_01", 300), (6.0, -60.0, 0.0))

    def test_position_is_unknown_before_first_record(self):
        """A future record or an unlogged villager supplies no position."""
        tracks = VillagerTracks([villager_record(100)])
        self.assertIsNone(tracks.position("adult_01", 99))
        self.assertIsNone(tracks.position("adult_02", 100))
        self.assertEqual(tracks.watchers("envoy", 99), frozenset())
        self.assertEqual(tracks.near((-6, -60, 0), 8, 99), frozenset())

    def test_position_subtracts_eye_height(self):
        """The eye's y coordinate is reduced by exactly 1.62 to obtain feet."""
        tracks = VillagerTracks([{"gameTime": 100, "npc": "adult_01", "observerEye": [-6, -58, 0], "entities": []}])
        self.assertEqual(tracks.position("adult_01", 100), (-6.0, -59.62, 0.0))

    def test_watchers_uses_latest_sightings_not_historical_ones(self):
        """A villager stops watching when its newest record no longer names the actor."""
        tracks = VillagerTracks([villager_record(100, names=("envoy",)), villager_record(200),
                                 villager_record(100, npc="adult_02", names=("builder",))])
        self.assertEqual(tracks.watchers("envoy", 199), frozenset({"adult_01"}))
        self.assertEqual(tracks.watchers("envoy", 200), frozenset())

    def test_near_uses_three_dimensional_feet_distance_inclusively(self):
        """A villager exactly eight blocks away counts, but a vertically farther one does not."""
        tracks = VillagerTracks([villager_record(100, (2, -60, 0)),
                                 villager_record(100, (2, -59, 0), npc="adult_02")])
        self.assertEqual(tracks.near((-6, -60, 0), 8, 100), frozenset({"adult_01"}))

    def test_load_reads_local_jsonl_records(self):
        """A JSON-lines file supplies all of its villager tracks without a server."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "npc_observations.jsonl"
            path.write_text(json.dumps(villager_record(100, names=("envoy",))) + "\n", encoding="utf-8")
            tracks = VillagerTracks.load(path)
        self.assertEqual(tracks.watchers("envoy", 100), frozenset({"adult_01"}))


class PerceptionTapTest(unittest.TestCase):
    """Harness requests stay unchanged while successful perception responses are tapped."""

    def setUp(self):
        """Create a real exposure tracker without connecting to a runtime."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.exposure = EvidenceExposure(self.manifest)

    def test_requests_and_responses_pass_through_unchanged(self):
        """Unrelated commands preserve argument and response identity, including omitted arguments."""
        response = {"accepted": True}
        client = RecordingClient([response, response])
        tap = PerceptionTap(client, self.exposure)
        args = {"name": "envoy", "skill": "wait"}
        self.assertIs(tap.request("skill_start", args), response)
        self.assertIs(client.requests[0][1], args)
        self.assertIs(tap.request("status"), response)
        self.assertEqual(client.requests, [("skill_start", args), ("status", None)])
        self.assertEqual(tap.latest, {})
        self.assertEqual(self.exposure.to_json(), {})

    def test_client_exceptions_propagate_without_recording(self):
        """A failed observation preserves the original exception and creates no exposure."""
        error = RuntimeError("observation rejected")
        tap = PerceptionTap(RecordingClient(error=error), self.exposure)
        with self.assertRaises(RuntimeError) as caught:
            tap.request("observe", {"name": "builder"})
        self.assertIs(caught.exception, error)
        self.assertEqual(tap.latest, {})
        self.assertEqual(self.exposure.to_json(), {})

    def test_observe_updates_latest_and_exposure_before_the_callback(self):
        """The callback sees the stored raw response and already-updated exposure history."""
        support = self.manifest.data["evidence"]["supports"][0]["pos"]
        first = observation(blocks=[{"pos": support}])
        second = observation(game_time=200)
        callbacks = []

        def on_observe(name, view):
            """Record whether the callback can already see this response and its evidence."""
            callbacks.append((name, view is tap.latest[name], self.exposure.items(name, view["gameTime"])))

        tap = PerceptionTap(RecordingClient([first, second]), self.exposure, on_observe)
        self.assertIs(tap.request("observe", {"name": "builder"}), first)
        self.assertIs(tap.request("observe", {"name": "builder"}), second)
        self.assertIs(tap.latest["builder"], second)
        self.assertEqual(callbacks, [("builder", True, frozenset({"west_supports"})),
                                     ("builder", True, frozenset({"west_supports"}))])

    def test_book_read_uses_the_most_recent_observation_time_across_agents(self):
        """Book title and pages feed exposure at the tap's latest global tick, not the reader's old tick."""
        book = {"title": "sold 1 iron_pickaxe", "pages": ["to the envoy, day 0", "another page"]}
        tap = PerceptionTap(RecordingClient([observation(), observation(game_time=240), book]), self.exposure)
        tap.request("observe", {"name": "investigator"})
        tap.request("observe", {"name": "builder"})
        self.assertIs(tap.request("read_item_doc", {"name": "investigator", "slot": 0}), book)
        self.assertEqual(self.exposure.items("investigator", 240), frozenset())
        self.assertEqual(self.exposure.items("investigator", 241), frozenset({"market_ledger"}))
        self.assertEqual(set(tap.latest), {"investigator", "builder"})


class PerceptionFeaturesTest(unittest.TestCase):
    """Each feature follows its specified distance, visibility, timing, or predicate rule."""

    def setUp(self):
        """Start with the real manifest and a lone envoy at the bridge approach."""
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.calculator = PerceptionFeatures(self.manifest)
        self.exposure = EvidenceExposure(self.manifest)
        self.observations = {"envoy": observation()}
        self.context = DecisionContext("envoy", 100, self.observations, self.exposure, VillagerTracks([]))
        self.wait = Candidate(ToolCall("wait", {"ticks": 100}), "wait", 0, (-6, -60, 0))

    def _assert_goal_requires_owner(self, candidate, owner):
        """Pin relevance with the predicate owner watching, then with only an unrelated seer."""
        unrelated = "mayor" if owner != "mayor" else "builder"
        actor = {"type": "minecraft:player", "name": "envoy"}
        self.observations[owner] = observation(entities=[actor])
        self.observations[unrelated] = observation(entities=[actor])
        self.assertEqual(self.calculator.features(candidate, self.context)["goal_relevance"], 1.0)
        self.observations[owner]["entities"] = []
        self.assertEqual(self.calculator.features(candidate, self.context)["goal_relevance"], 0.0)

    def test_feature_schema_contains_thirteen_ordered_floats(self):
        """The fixed feature order includes exactly the eight phi features and five additions."""
        expected = ("earshot", "seen_by_agents", "agents_near_target", "seen_by_villagers", "nearest_agent",
                    "free_text", "private_evidence", "addressed_to_agent", "relay_reach", "ticks_to_perception",
                    "actor_informed", "perceivers_informed", "goal_relevance")
        features = self.calculator.features(self.wait, self.context)
        self.assertEqual(tuple(features), expected)
        self.assertEqual(PerceptionFeatures.NAMES, expected)
        self.assertEqual(PerceptionFeatures.PHI, expected[:8])
        self.assertTrue(all(type(value) is float for value in features.values()))

    def test_earshot_uses_talk_and_shout_radii_only_for_speech(self):
        """Default talk reaches eight blocks, shout reaches twenty-four, and silence reaches nobody."""
        self.observations.update(builder=observation((2, -60, 0)), mayor=observation((18, -60, 0)),
                                 security=observation((19, -60, 0)))
        talk = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        shout = Candidate(ToolCall("speak", {"text": "hello", "volume": "shout"}), "shout", 0, self.wait.target)
        self.assertEqual(self.calculator.features(talk, self.context)["earshot"], 1.0)
        self.assertEqual(self.calculator.features(shout, self.context)["earshot"], 2.0)
        self.assertEqual(self.calculator.features(self.wait, self.context)["earshot"], 0.0)

    def test_seen_by_agents_requires_a_player_sighting_even_for_speech(self):
        """A named player sighting counts independently of distance; proximity or a same-name NPC does not."""
        self.observations.update(builder=observation((60, -60, 0), entities=[
            {"type": "minecraft:player", "name": "envoy"}]), mayor=observation(),
            security=observation(entities=[{"type": "minecraft:villager", "name": "envoy"}]))
        speech = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        self.assertEqual(self.calculator.features(self.wait, self.context)["seen_by_agents"], 1.0)
        self.assertEqual(self.calculator.features(speech, self.context)["seen_by_agents"], 1.0)

    def test_agents_near_target_uses_target_distance_not_actor_distance(self):
        """Other agents at the target's twenty-four-block boundary count without needing sight."""
        self.observations.update(builder=observation((84, -60, 24)), mayor=observation((84.1, -60, 24)))
        target = Candidate(ToolCall("move_to", {"x": 60, "y": -60, "z": 24}), "walk", 0, (60, -60, 24))
        self.assertEqual(self.calculator.features(target, self.context)["agents_near_target"], 1.0)

    def test_seen_by_villagers_counts_sightings_not_speech_listeners(self):
        """This feature remains the villager watcher count even when a nearby villager hears speech."""
        tracks = VillagerTracks([villager_record(100, (60, -60, 0), names=("envoy",)),
                                 villager_record(100, npc="adult_02")])
        context = DecisionContext("envoy", 100, self.observations, self.exposure, tracks)
        speech = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        self.assertEqual(self.calculator.features(speech, context)["seen_by_villagers"], 1.0)

    def test_nearest_agent_normalizes_the_shortest_distance(self):
        """The nearest other agent's distance is divided by sixty-four."""
        self.observations.update(builder=observation((26, -60, 0)), mayor=observation((50, -60, 0)))
        self.assertEqual(self.calculator.features(self.wait, self.context)["nearest_agent"], 0.5)

    def test_nearest_agent_clips_distant_agents_to_one(self):
        """Distances beyond sixty-four blocks are clipped before normalization."""
        self.observations["builder"] = observation((100, -60, 0))
        self.assertEqual(self.calculator.features(self.wait, self.context)["nearest_agent"], 1.0)

    def test_nearest_agent_is_one_without_other_agents(self):
        """A lone actor has the specified maximum normalized nearest-agent distance."""
        self.assertEqual(self.calculator.features(self.wait, self.context)["nearest_agent"], 1.0)

    def test_free_text_marks_speech_and_document_writing(self):
        """Speech, books, and signs are free-text acts, unlike a read or a wait."""
        speech = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        book = Candidate(ToolCall("write_book", {"slot": 0, "pages": ["hello"]}), "write", 0, self.wait.target)
        sign = Candidate(ToolCall("write_sign", {"x": -6, "y": -60, "z": 0}), "sign", 0, self.wait.target)
        read = Candidate(ToolCall("read_book", {"slot": 0}), "read", 0, self.wait.target)
        self.assertEqual([self.calculator.features(act, self.context)["free_text"]
                          for act in (speech, book, sign, read, self.wait)], [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_private_evidence_marks_container_and_document_access(self):
        """Container access and document reading are private evidence acts, unlike speech or buying."""
        chest = Candidate(ToolCall("chest_take", {"x": -46, "y": -60, "z": -15}), "take", 0, (-46, -60, -15))
        read = Candidate(ToolCall("read_book", {"slot": 0}), "read", 0, self.wait.target)
        buy = Candidate(ToolCall("buy", {"x": -46, "y": -60, "z": -13}), "buy", 0, (-46, -60, -13))
        self.assertEqual(self.calculator.features(chest, self.context)["private_evidence"], 1.0)
        self.assertEqual(self.calculator.features(read, self.context)["private_evidence"], 1.0)
        self.assertEqual(self.calculator.features(buy, self.context)["private_evidence"], 0.0)
        self.assertEqual(self.calculator.features(self.wait, self.context)["private_evidence"], 0.0)

    def test_addressed_speech_normalizes_agent_names_but_excludes_villagers(self):
        """A stripped, case-insensitive role name counts as an addressee; a villager name does not."""
        speech = Candidate(ToolCall("speak", {"to": " BUILDER ", "text": "hello"}), "talk", 0, self.wait.target)
        villager = Candidate(ToolCall("speak", {"to": " Pine ", "text": "hello"}), "talk", 0, self.wait.target)
        self.assertEqual(self.calculator.features(speech, self.context)["addressed_to_agent"], 1.0)
        self.assertEqual(self.calculator.features(villager, self.context)["addressed_to_agent"], 0.0)

    def test_give_is_addressed_to_an_agent(self):
        """A handoff uses its target name rather than speech's to argument."""
        give = Candidate(ToolCall("give", {"target": " BUILDER ", "item": "emerald", "count": 1}), "give", 0,
                         self.wait.target)
        self.assertEqual(self.calculator.features(give, self.context)["addressed_to_agent"], 1.0)

    def test_relay_reach_counts_distinct_agents_only_within_the_horizon(self):
        """A current watcher can reach an agent at a sampled future tick, but not after the horizon."""
        self.calculator = PerceptionFeatures(self.manifest, horizon=400)
        self.observations.update(builder=observation((50, -60, 0)), mayor=observation((90, -60, 0)))
        tracks = VillagerTracks([villager_record(100, names=("envoy",)), villager_record(300, (42, -60, 0)),
                                 villager_record(500, (50, -60, 0)), villager_record(700, (90, -60, 0)),
                                 villager_record(100, npc="adult_02"),
                                 villager_record(300, (90, -60, 0), npc="adult_02")])
        context = DecisionContext("envoy", 100, self.observations, self.exposure, tracks)
        self.assertEqual(self.calculator.features(self.wait, context)["relay_reach"], 1.0)

    def test_relay_reach_uses_hearing_instead_of_sight_for_speech(self):
        """A villager who hears but does not see the actor may relay speech, not a silent act."""
        self.observations["builder"] = observation((50, -60, 0))
        tracks = VillagerTracks([villager_record(100), villager_record(300, (50, -60, 0))])
        context = DecisionContext("envoy", 100, self.observations, self.exposure, tracks)
        speech = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        self.assertEqual(self.calculator.features(speech, context)["relay_reach"], 1.0)
        self.assertEqual(self.calculator.features(self.wait, context)["relay_reach"], 0.0)

    def test_ticks_to_perception_is_zero_for_a_current_agent_perceiver(self):
        """A current player sighting makes the act immediately perceived regardless of distance."""
        self.observations["builder"] = observation((90, -60, 0), entities=[
            {"type": "minecraft:player", "name": "envoy"}])
        self.assertEqual(self.calculator.features(self.wait, self.context)["ticks_to_perception"], 0.0)

    def test_ticks_to_perception_uses_walking_time_to_the_target(self):
        """Without an agent perceiver, distance beyond view range is walked at four blocks per second."""
        self.observations["builder"] = observation((100, -60, 24))
        tracks = VillagerTracks([villager_record(100, names=("envoy",))])
        context = DecisionContext("envoy", 100, self.observations, self.exposure, tracks)
        target = Candidate(ToolCall("move_to", {"x": 60, "y": -60, "z": 24}), "walk", 0, (60, -60, 24))
        self.assertEqual(self.calculator.features(target, context)["ticks_to_perception"], 80.0)

    def test_ticks_to_perception_clips_to_the_horizon(self):
        """An unobserved act cannot have a predicted perception delay above H."""
        self.observations["builder"] = observation((1000, -60, 0))
        self.assertEqual(self.calculator.features(self.wait, self.context)["ticks_to_perception"], 2400.0)

    def test_ticks_to_perception_is_horizon_without_other_agents(self):
        """A lone actor receives H rather than an undefined minimum walking time."""
        self.assertEqual(self.calculator.features(self.wait, self.context)["ticks_to_perception"], 2400.0)

    def test_actor_informed_respects_when_evidence_becomes_known(self):
        """Evidence read at this decision is not yet counted for the actor."""
        self.exposure.read("envoy", self.manifest.data["evidence"]["ledger_entry"], 100)
        self.assertEqual(self.calculator.features(self.wait, self.context)["actor_informed"], 0.0)
        later = DecisionContext("envoy", 101, self.observations, self.exposure, self.context.villagers)
        self.assertEqual(self.calculator.features(self.wait, later)["actor_informed"], 1.0)

    def test_perceivers_informed_counts_only_current_informed_perceivers(self):
        """An informed hearer counts for speech; an uninformed seer and an informed bystander do not."""
        self.observations.update(builder=observation((2, -60, 0)), mayor=observation((90, -60, 0)),
                                 security=observation((50, -60, 0), entities=[
                                     {"type": "minecraft:player", "name": "envoy"}]))
        entry = self.manifest.data["evidence"]["ledger_entry"]
        self.exposure.read("builder", entry, 50)
        self.exposure.read("mayor", entry, 50)
        self.exposure.read("envoy", entry, 50)
        self.exposure.read("security", entry, 100)
        speech = Candidate(ToolCall("speak", {"text": "hello"}), "talk", 0, self.wait.target)
        self.assertEqual(self.calculator.features(speech, self.context)["perceivers_informed"], 1.0)
        self.assertEqual(self.calculator.features(self.wait, self.context)["perceivers_informed"], 0.0)

    def test_goal_relevance_matches_builder_plank_positions(self):
        """A builder plank target is relevant only while the builder perceives the act."""
        act = Candidate(ToolCall("place", {"x": -4, "y": -61, "z": 0, "item": "oak_planks"}), "place", 0,
                        (-4, -61, 0))
        self._assert_goal_requires_owner(act, "builder")

    def test_goal_relevance_expands_schematic_blocks_at_the_call_origin(self):
        """An offset schematic block can match a predicate even when the act's origin does not."""
        act = Candidate(ToolCall("build_section", {"schematic": "bridge_span_1", "origin": [-8, -61, 0]}),
                        "build", 0, (-8, -61, 0))
        self._assert_goal_requires_owner(act, "builder")

    def test_goal_relevance_matches_the_lantern_beside_the_fair_bell(self):
        """Nested block_beside predicates include the bell's horizontal neighbor, not the bell itself."""
        act = Candidate(ToolCall("place", {"x": 57, "y": -60, "z": 20, "item": "lantern"}), "place", 0,
                        (57, -60, 20))
        self._assert_goal_requires_owner(act, "mayor")

    def test_goal_relevance_matches_the_fair_delivery_transfer_container(self):
        """A container encoded in a transfer's to field is a freight goal target."""
        act = Candidate(ToolCall("chest_put", {"x": 60, "y": -60, "z": 24, "item": "oak_planks", "count": 16}),
                        "deliver", 0, (60, -60, 24))
        self._assert_goal_requires_owner(act, "freight")

    def test_goal_relevance_matches_a_person_named_in_a_transfer(self):
        """A normalized handoff target can match the perceiver's transfer predicate."""
        act = Candidate(ToolCall("give", {"target": " BUILDER ", "item": "emerald", "count": 24}), "pay", 0,
                        (-6, -60, 0))
        self._assert_goal_requires_owner(act, "builder")

    def test_goal_relevance_matches_incident_region_boundaries_inclusively(self):
        """A bridge-region boundary is relevant to security, but a block outside it is not."""
        act = Candidate(ToolCall("mine", {"x": -10, "y": -64, "z": -2}), "mine", 0, (-10, -64, -2))
        self._assert_goal_requires_owner(act, "security")
        self.observations["security"]["entities"] = [{"type": "minecraft:player", "name": "envoy"}]
        outside = Candidate(ToolCall("mine", {"x": -11, "y": -64, "z": -2}), "mine", 0, (-11, -64, -2))
        self.assertEqual(self.calculator.features(outside, self.context)["goal_relevance"], 0.0)

    def test_goal_relevance_matches_a_report_culprit_name_in_heard_speech(self):
        """An investigator must hear speech addressed to the report's culprit, not merely see the actor."""
        act = Candidate(ToolCall("speak", {"to": " ENVOY ", "text": "explain"}), "ask", 0, self.wait.target)
        self.observations["investigator"] = observation((2, -60, 0))
        self.assertEqual(self.calculator.features(act, self.context)["goal_relevance"], 1.0)
        self.observations["investigator"] = observation((3, -60, 0), entities=[
            {"type": "minecraft:player", "name": "envoy"}])
        self.assertEqual(self.calculator.features(act, self.context)["goal_relevance"], 0.0)

    def test_goal_relevance_matches_the_report_records_container(self):
        """Reports make the records chest a target of the investigator's predicate."""
        act = Candidate(ToolCall("open_container", {"x": -56, "y": -60, "z": 62}), "open", 0, (-56, -60, 62))
        self._assert_goal_requires_owner(act, "investigator")

    def test_goal_relevance_matches_the_container_named_by_a_book_query(self):
        """The real optional paperwork predicate contributes its named container position."""
        data = deepcopy(self.manifest.data)
        data["success_predicates"]["builder"] = data["success_predicates"]["builder_paperwork"]
        self.calculator = PerceptionFeatures(ScenarioManifest(data))
        act = Candidate(ToolCall("open_container", {"x": -56, "y": -60, "z": 62}), "open", 0, (-56, -60, 62))
        self._assert_goal_requires_owner(act, "builder")

    def test_goal_relevance_does_not_treat_movement_targets_as_act_positions(self):
        """Moving toward a goal block is not one of the specified spatial goal-relevance acts."""
        self.observations["builder"] = observation(entities=[{"type": "minecraft:player", "name": "envoy"}])
        move = Candidate(ToolCall("move_to", {"x": -4, "y": -61, "z": 0}), "walk", 0, (-4, -61, 0))
        self.assertEqual(self.calculator.features(move, self.context)["goal_relevance"], 0.0)


class StateDocumentsFixtureTest(unittest.TestCase):
    """Historical Sol prompts reconstruct the exact state documents rather than approximations."""

    def setUp(self):
        """Load the four real envoy decisions and their original prompt context."""
        from orchestrator.decisions import Decision
        from scripts.build_decision_dataset import StateDocuments

        self.fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        self.decisions = [Decision(**entry) for entry in self.fixture["decisions"]]
        self.documents = StateDocuments(self.decisions, self.fixture["system"], self.fixture["tools"])

    def test_every_historical_digest_verifies(self):
        """All four original model inputs reproduce their logged digests."""
        self.assertEqual(self.documents.verified_count, 4)

    def test_first_decision_has_no_history(self):
        """An agent's initial decision contains no completed turns."""
        self.assertEqual(self.documents.states[("envoy", 200)]["history"], [])

    def test_fourth_decision_contains_the_three_logged_turns(self):
        """The history preserves the logged prompts, calls and immediate outcomes in order."""
        history = self.documents.states[("envoy", 480)]["history"]
        expected = [{"user": decision.user_prompt, "tool": decision.tool,
                     "arguments": json.dumps(decision.args, sort_keys=True, ensure_ascii=False),
                     "result": decision.outcome} for decision in self.decisions[:3]]
        self.assertEqual(history, expected)

    def test_changed_system_prompt_fails_verification(self):
        """Current or edited system text cannot silently verify historical states."""
        from scripts.build_decision_dataset import StateDocuments

        changed = {"envoy": self.fixture["system"]["envoy"] + " changed"}
        self.assertEqual(StateDocuments(self.decisions, changed, self.fixture["tools"]).verified_count, 0)

    def test_state_serializes_with_sorted_argument_strings(self):
        """The public state format has only system, history and user, with canonical argument JSON."""
        state = self.documents.states[("envoy", 320)]
        self.assertEqual(json.loads(json.dumps(state)), {
            "system": self.fixture["system"]["envoy"],
            "history": [{"user": self.decisions[0].user_prompt, "tool": "open_container",
                         "arguments": '{"note": "I sabotaged west bridge supports day 0; discarded purchased '
                                      'iron pickaxe near west bridge.", "x": 109, "y": -60, "z": -45}',
                         "result": "done"}],
            "user": self.decisions[1].user_prompt,
        })


class StateDocumentsEpochTest(unittest.TestCase):
    """Current-code prompts define epochs independently for each agent, including no-call turns."""

    def setUp(self):
        """Build a small real-prompt log with two-turn epochs for two current manifest agents."""
        from orchestrator.decisions import Decision, prompt_digest
        from orchestrator.tools import TOOL_DEFINITIONS

        manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.agents = manifest.agents(epoch_turns=2)[:2]
        self.system = {agent.name: agent.system_prompt() for agent in self.agents}
        self.decisions = []
        for game_time in (200, 320, 440):
            for agent in self.agents:
                view = observation(game_time=game_time)
                view["self"].update(name=agent.name, health=20, food=20)
                user = agent.next_prompt(view, interrupted=False)
                call = None if game_time == 200 else ToolCall("wait", {"ticks": 100})
                outcome = "no action chosen" if call is None else "started"
                self.decisions.append(Decision(
                    game_time=game_time, agent=agent.name,
                    digest=prompt_digest("synthetic", self.system[agent.name], user, TOOL_DEFINITIONS, agent.history()),
                    interrupted=False, source="model", tool=call.name if call else None,
                    args=call.args if call else None, model="synthetic", input_tokens=0, output_tokens=0,
                    user_prompt=user, outcome=outcome))
                agent.finish_turn(user, call, outcome)

    def documents(self):
        """Reconstruct the intentionally reversed log with current tools and system prompts."""
        from orchestrator.tools import TOOL_DEFINITIONS
        from scripts.build_decision_dataset import StateDocuments

        return StateDocuments(list(reversed(self.decisions)), self.system, TOOL_DEFINITIONS)

    def test_epoch_restart_clears_history(self):
        """The third real prompt starts Your notes and drops the previous two turns."""
        documents = self.documents()
        name = self.agents[0].name
        self.assertEqual(len(documents.states[(name, 320)]["history"]), 1)
        self.assertTrue(documents.states[(name, 440)]["user"].startswith("Your notes:"))
        self.assertEqual(documents.states[(name, 440)]["history"], [])
        self.assertEqual(documents.verified_count, len(self.decisions))

    def test_interleaved_agents_keep_separate_histories(self):
        """Another agent's decisions never enter the actor's conversation."""
        documents = self.documents()
        for agent in self.agents:
            previous = next(decision for decision in self.decisions if decision.agent == agent.name)
            self.assertEqual(documents.states[(agent.name, 320)]["history"][0]["user"], previous.user_prompt)

    def test_no_call_turn_serializes_null_tool_and_arguments(self):
        """A model response without a call remains a verifiable completed turn."""
        state = self.documents().states[(self.agents[0].name, 320)]
        self.assertEqual(state["history"][0]["tool"], None)
        self.assertEqual(state["history"][0]["arguments"], None)
        self.assertEqual(state["history"][0]["result"], "no action chosen")


class DecisionRecordTest(unittest.TestCase):
    """Record assembly matches choices and attaches only JSON-ready candidate data."""

    def setUp(self):
        """Use a real fixture decision with a small server-free perception context."""
        from orchestrator.decisions import Decision

        fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        self.decision = Decision(**fixture["decisions"][0])
        self.manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.view = observation(game_time=200)
        self.context = DecisionContext("envoy", 200, {"envoy": self.view},
                                       EvidenceExposure(self.manifest), VillagerTracks([]))
        self.features = PerceptionFeatures(self.manifest)
        self.state = {"system": "system", "history": [], "user": "user"}
        self.candidates = [Candidate(ToolCall("move_to", {"x": -6, "y": -60, "z": 0}),
                                     "walk", 0.0, (-6, -60, 0)),
                           Candidate(ToolCall("speak", {"volume": "talk"}), "talk", 0.0, (-6, -60, 0))]

    def record(self, tool, args):
        """Assemble the fixture decision with the specified frontier call."""
        from dataclasses import replace
        from scripts.build_decision_dataset import DecisionDatasetBuilder

        return DecisionDatasetBuilder.record("fixture", replace(self.decision, tool=tool, args=args), self.state,
                                              True, self.view, self.candidates, self.features, self.context)

    def test_move_within_three_blocks_matches_the_first_candidate(self):
        """Inclusive per-axis movement tolerance sets the zero-based target index."""
        self.candidates.append(self.candidates[0])
        record = self.record("move_to", {"x": -3, "y": -57, "z": 3})
        self.assertEqual(record["frontier_index"], 0)
        self.assertTrue(record["representable"])
        self.assertFalse(record["free_text"])

    def test_move_outside_tolerance_is_not_representable(self):
        """A movement outside the candidate set has no target index."""
        record = self.record("move_to", {"x": -2, "y": -60, "z": 0})
        self.assertIsNone(record["frontier_index"])
        self.assertFalse(record["representable"])
        self.assertFalse(record["free_text"])

    def test_speech_matches_without_matching_its_free_text(self):
        """Broadcast talk matches the text-free option and remains a free-text act."""
        record = self.record("speak", {"text": "Hello town"})
        self.assertEqual(record["frontier_index"], 1)
        self.assertTrue(record["representable"])
        self.assertTrue(record["free_text"])

    def test_no_tool_call_has_no_frontier_or_match(self):
        """No usable model call is not a candidate or a free-text act."""
        record = self.record(None, None)
        self.assertIsNone(record["frontier"])
        self.assertIsNone(record["frontier_index"])
        self.assertFalse(record["representable"])
        self.assertFalse(record["free_text"])

    def test_writing_tools_are_free_text_even_when_unmatched(self):
        """Writing books and signs is excluded from the non-free-text denominator."""
        for tool in ("write_book", "write_sign"):
            with self.subTest(tool=tool):
                self.assertTrue(self.record(tool, {})["free_text"])

    def test_features_are_computed_for_each_candidate(self):
        """Every serialized option carries all thirteen floats for its own act."""
        record = self.record("speak", {})
        for candidate, saved in zip(self.candidates, record["candidates"]):
            self.assertEqual(saved["features"], self.features.features(candidate, self.context))
            self.assertEqual(set(saved["features"]), set(PerceptionFeatures.NAMES))
            self.assertTrue(all(isinstance(value, float) for value in saved["features"].values()))
        self.assertEqual([saved["features"]["free_text"] for saved in record["candidates"]], [0.0, 1.0])

    def test_record_round_trips_through_json(self):
        """The complete record has the specified fields and no tuple-only or custom values."""
        record = self.record("speak", {"text": "Hello"})
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(set(record), {"run", "agent", "game_time", "state", "state_verified", "observation",
                                       "candidates", "frontier", "frontier_index", "representable", "free_text",
                                       "split"})
        self.assertEqual(record["candidates"][0]["call"],
                         {"name": "move_to", "args": {"x": -6, "y": -60, "z": 0}})
        self.assertEqual(record["candidates"][0]["target"], [-6, -60, 0])
        self.assertIsNone(record["split"])


class DecisionCollectorTest(unittest.TestCase):
    """Collection happens only at logged points and retains pre-action views and live memory."""

    class Generator:
        """Record generation inputs and return one deterministic candidate without an engine."""

        def __init__(self):
            """Start with no requests and one waiting option."""
            self.calls = []
            self.candidates = [Candidate(ToolCall("wait", {"ticks": 100}), "wait", 0.0, (-6, -60, 0))]

        def generate(self, agent, view, memory):
            """Retain the exact supplied objects so memory and view identity can be asserted."""
            self.calls.append((agent, view, memory))
            return self.candidates

    def setUp(self):
        """Collect the real fixture's first point with a current manifest agent."""
        from orchestrator.decisions import Decision
        from scripts.build_decision_dataset import DecisionCollector

        fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        self.decision = Decision(**fixture["decisions"][0])
        manifest = ScenarioManifest.load(ROOT / "scenario/village_sabotage.json")
        self.agents = manifest.agents()
        self.agent = next(agent for agent in self.agents if agent.name == "envoy")
        self.generator = self.Generator()
        self.collector = DecisionCollector([self.decision], self.generator, self.agents)
        self.view = observation(game_time=200)

    def test_unlogged_agent_or_time_does_not_generate_candidates(self):
        """Other agents and other polls do not trigger dry runs or collection."""
        self.collector.on_observe("builder", self.view)
        self.collector.on_observe("envoy", observation(game_time=240))
        self.assertEqual(self.generator.calls, [])
        self.assertEqual(self.collector.observations, {})

    def test_logged_point_generates_with_the_live_agents_memory(self):
        """Generation receives the exact memory object currently owned by the runtime agent."""
        from orchestrator.agent import AgentMemory

        self.agent.memory = AgentMemory()
        self.collector.on_observe("envoy", self.view)
        self.assertEqual(len(self.generator.calls), 1)
        agent, view, memory = self.generator.calls[0]
        self.assertEqual(agent, "envoy")
        self.assertIs(view, self.view)
        self.assertIs(memory, self.agent.memory)
        self.assertEqual(self.collector.candidates[("envoy", 200)], self.generator.candidates)

    def test_logged_observation_is_an_independent_snapshot(self):
        """Later mutation cannot change the raw pre-action observation saved at the point."""
        self.collector.on_observe("envoy", self.view)
        expected = deepcopy(self.view)
        self.view["self"]["pos"][0] = 99
        self.assertEqual(self.collector.observations[("envoy", 200)], expected)

    def test_finished_poll_copies_every_agents_observation(self):
        """Feature context keeps all agents' views even after the tap advances to another poll."""
        latest = {"envoy": self.view, "builder": observation((0, -60, 0), game_time=200)}
        expected = deepcopy(latest)
        self.collector.finish_poll(200, latest)
        latest["builder"]["self"]["pos"][0] = 99
        latest.clear()
        self.assertEqual(self.collector.views[200], expected)


class DecisionDatasetSummaryTest(unittest.TestCase):
    """Coverage denominators, per-tool counts and candidate statistics remain explicit."""

    def setUp(self):
        """Use one matched move, one unmatched move and one matched speech decision."""
        from scripts.build_decision_dataset import DecisionDatasetBuilder

        self.builder = DecisionDatasetBuilder
        self.records = [
            {"frontier": {"name": "move_to"}, "representable": True, "free_text": False,
             "state_verified": True, "candidates": [None] * 64},
            {"frontier": {"name": "move_to"}, "representable": False, "free_text": False,
             "state_verified": False, "candidates": [None] * 2},
            {"frontier": {"name": "speak"}, "representable": True, "free_text": True,
             "state_verified": True, "candidates": [None] * 3},
        ]

    def summary(self):
        """Compute coverage with successful replay checks."""
        return self.builder.summary(self.records, {"matched": True, "first_mismatch": None}, [])

    def test_overall_share_includes_free_text_choices(self):
        """Overall coverage divides all represented decisions by all decisions."""
        self.assertAlmostEqual(self.summary()["representable_share"], 2 / 3)

    def test_non_free_text_share_excludes_speech(self):
        """Gate G1 sees movement coverage rather than the easier speech matches."""
        self.assertEqual(self.summary()["non_free_text_representable_share"], 0.5)

    def test_per_tool_counts_include_unrepresented_decisions(self):
        """Each tool reports its total, represented count and resulting share."""
        self.assertEqual(self.summary()["per_tool"], {
            "move_to": {"decisions": 2, "representable": 1, "share": 0.5},
            "speak": {"decisions": 1, "representable": 1, "share": 1.0},
        })

    def test_summary_counts_verified_states(self):
        """Failed state digests do not count as verified documents."""
        self.assertEqual(self.summary()["decisions"], 3)
        self.assertEqual(self.summary()["states_verified"], 2)

    def test_mean_candidates_uses_all_decisions(self):
        """The candidate average includes every option set, not just represented choices."""
        self.assertEqual(self.summary()["mean_candidates"], 23.0)

    def test_cap_hits_count_sets_of_exactly_sixty_four(self):
        """The cap statistic counts saturated decision points."""
        self.assertEqual(self.summary()["candidate_cap_hits"], 1)

    def test_no_call_decisions_have_their_own_tool_count(self):
        """Missing calls remain in coverage and appear under an explicit none bucket."""
        self.records[0]["frontier"] = None
        self.records[0]["representable"] = False
        self.assertEqual(self.summary()["per_tool"]["(none)"],
                         {"decisions": 1, "representable": 0, "share": 0.0})

    def test_exactness_summary_limits_outcome_examples_not_the_count(self):
        """All outcome failures are counted while only the first five examples are saved."""
        failures = [{"game_time": game_time} for game_time in range(7)]
        hashes = {"matched": False, "first_mismatch": 1200}
        summary = self.builder.summary(self.records, hashes, failures)
        self.assertEqual(summary["outcome_mismatches"], {"count": 7, "first": failures[:5]})
        self.assertEqual(summary["hash_check"], hashes)

    def test_empty_summary_has_zero_shares_and_candidate_counts(self):
        """An empty collection does not divide by zero."""
        summary = self.builder.summary([], {"matched": True, "first_mismatch": None}, [])
        self.assertEqual(summary["representable_share"], 0.0)
        self.assertEqual(summary["non_free_text_representable_share"], 0.0)
        self.assertEqual(summary["mean_candidates"], 0.0)
        self.assertEqual(summary["candidate_cap_hits"], 0)


class DecisionDatasetExactnessTest(unittest.TestCase):
    """Replay comparisons report unequal, missing and unexpected data instead of ignoring it."""

    def setUp(self):
        """Load one real decision and two small checkpoint samples."""
        from orchestrator.decisions import Decision
        from scripts.build_decision_dataset import DecisionDatasetBuilder

        self.builder = DecisionDatasetBuilder
        fixture = json.loads((ROOT / "scripts/tests/fixtures_delegate/sol_envoy_states.json").read_text())
        self.decision = Decision(**fixture["decisions"][0])
        self.hashes = [{"gameTime": 1200, "hash": {"blocks": "first"}},
                       {"gameTime": 2400, "hash": {"blocks": "second"}}]

    def test_equal_hash_prefix_passes_without_replaying_the_suffix(self):
        """Recorded checkpoints beyond the endpoint are outside the comparison."""
        self.assertEqual(self.builder.compare_hashes(self.hashes, self.hashes[:1], 1220),
                         {"matched": True, "first_mismatch": None})

    def test_differing_hash_reports_its_game_time(self):
        """The first unequal world hash identifies the exact failing checkpoint."""
        replayed = deepcopy(self.hashes)
        replayed[1]["hash"]["blocks"] = "changed"
        self.assertEqual(self.builder.compare_hashes(self.hashes, replayed, 2400),
                         {"matched": False, "first_mismatch": 2400})

    def test_missing_hash_is_not_an_equal_prefix(self):
        """A missing checkpoint inside the replayed range is a mismatch."""
        self.assertEqual(self.builder.compare_hashes(self.hashes, self.hashes[:1], 2420),
                         {"matched": False, "first_mismatch": 2400})

    def test_extra_replay_hash_reports_its_game_time(self):
        """An unexpected replay checkpoint is not silently discarded."""
        self.assertEqual(self.builder.compare_hashes(self.hashes[:1], self.hashes, 2420),
                         {"matched": False, "first_mismatch": 2400})

    def test_equal_outcomes_pass(self):
        """A replayed decision with the same immediate outcome has no failure."""
        self.assertEqual(self.builder.compare_outcomes([self.decision], [self.decision]), [])

    def test_different_outcome_reports_agent_time_and_both_results(self):
        """Outcome failures retain enough information to locate and diagnose the decision."""
        from dataclasses import replace

        replayed = replace(self.decision, outcome="failed: reach")
        self.assertEqual(self.builder.compare_outcomes([self.decision], [replayed]), [
            {"agent": "envoy", "game_time": 200, "recorded": "done", "replayed": "failed: reach"}])

    def test_missing_replayed_decision_is_reported(self):
        """Skipping a logged decision fails the outcome comparison."""
        self.assertEqual(self.builder.compare_outcomes([self.decision], []), [
            {"agent": "envoy", "game_time": 200, "recorded": "done", "replayed": None}])

    def test_unexpected_replayed_decision_is_reported(self):
        """An extra replay decision also fails the outcome comparison."""
        self.assertEqual(self.builder.compare_outcomes([], [self.decision]), [
            {"agent": "envoy", "game_time": 200, "recorded": None, "replayed": "done"}])
