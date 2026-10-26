"""Server-free auditor views built from small logs in the engine's actual format."""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from scripts.run_views import RunViews, main


def event(tick, event_type, **fields):
    """Build an engine event with the real tick and type field names."""
    return {"id": tick, "gameTime": tick, "type": event_type, **fields}


def decision(tick, tool, agent="builder", outcome="started", **args):
    """Build a decision with deliberately private prompt and note sentinels."""
    return {"game_time": tick, "agent": agent, "tool": tool, "outcome": outcome,
            "args": {"note": "PRIVATE NOTE", **args}, "user_prompt": "PRIVATE PROMPT"}


class RunViewsTests(unittest.TestCase):
    """Exercise speech, causal joins, sampled proximity and safe text output."""

    def setUp(self):
        """Create a run with two roles and no engine or model dependency."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self._json("scenario.json", {"roles": [{"name": name, "brief": "PRIVATE BRIEF"}
                                             for name in ("builder", "envoy", "mayor")]})

    def _json(self, filename, value):
        """Write one synthetic JSON artifact under the isolated run."""
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _jsonl(self, filename, rows):
        """Write actual JSONL, including a blank line to exercise optional whitespace."""
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n" + "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_speech_uses_actual_listeners_and_preserves_npc_identity(self):
        """Sort by logged time and never confuse an addressee or eye position with hearing."""
        self._jsonl("world/simharness/events.jsonl", [
            event(24000, "utterance", speaker="builder", speakerKey="builder", volume="shout",
                  to="envoy", text="No one heard this.", listeners=[], listenerEyes={"envoy": [1, 2, 3]}),
            event(6000, "utterance", speaker="Dahlia", speakerKey="npc:adult_19", volume="talk",
                  to="builder", text="I saw the bridge.", listeners=["envoy"], startedAt=5900),
            event(6000, "utterance", speaker="envoy", volume="whisper", text="Reply", listeners=["builder"]),
        ])
        speech = RunViews(self.directory).speech()
        self.assertEqual([row["tick"] for row in speech], [6000, 6000, 24000])
        self.assertEqual(speech[0]["speaker_key"], "npc:adult_19")
        self.assertEqual(speech[0]["listeners"], ["envoy"])
        self.assertEqual(speech[0]["to"], "builder")
        self.assertEqual(speech[0]["clock"], "day 0, 12:00")
        self.assertEqual(speech[-1]["day"], 1)
        self.assertEqual(speech[-1]["clock"], "day 1, 06:00")
        self.assertEqual(speech[-1]["listeners"], [])

    def _action_logs(self):
        """Write a skill spanning carried-on and failed decisions plus all required effects."""
        self._jsonl("orchestrator/decisions.jsonl", [
            decision(10, "build_section", schematic="bridge"),
            decision(15, "keep_going", outcome="carried on"),
            decision(20, "build_section", outcome="not possible: busy"),
            decision(25, "speak", agent="envoy", text="Hello"),
            decision(26, "ask", agent="envoy", npc="Dahlia"),
            decision(40, "wait", ticks=10),
            decision(60, "give", to="envoy", item="oak_planks", count=1),
            decision(70, "buy", item="bread", count=1),
            decision(80, "write_book", outcome="done", title="Report", pages=["Public evidence"]),
            decision(90, "ferry", to="east"),
        ])
        self._jsonl("world/simharness/events.jsonl", [
            event(5, "block_change", actor="envoy", pos=[0, 0, 0], **{"from": "planks", "to": "air"}),
            event(21, "block_change", actor="builder", pos=[1, 0, 0], **{"from": "air", "to": "planks"}),
            event(22, "block_change", actor="world", pos=[2, 0, 0], **{"from": "air", "to": "planks"}),
            event(27, "utterance", speaker="envoy", volume="talk", text="Hello", listeners=["builder"]),
            event(28, "skill_end", bot="envoy", skill="speak", skillId="envoy:1", outcome="succeeded"),
            event(30, "skill_end", bot="builder", skill="place_section", skillId="builder:1",
                  outcome="failed", reason="missing materials"),
            event(31, "block_change", actor="builder", pos=[3, 0, 0], **{"from": "air", "to": "planks"}),
            event(50, "skill_end", bot="builder", skill="wait", skillId="builder:2", outcome="succeeded"),
            event(61, "transfer", item="minecraft:oak_planks", count=1, **{"from": "builder", "to": "envoy"}),
            event(61, "skill_end", bot="builder", skill="give", skillId="builder:3", outcome="succeeded"),
            event(71, "purchase", buyer="builder", item="minecraft:bread", count=1, cost=2),
            event(71, "skill_end", bot="builder", skill="buy", skillId="builder:4", outcome="succeeded"),
            event(80, "doc_write", author="builder", title="Report", kind="book", docId="book:1"),
            event(95, "ferry_trip", rider="builder", startTick=90, endTick=95, **{"from": "west", "to": "east"}),
            event(95, "skill_end", bot="builder", skill="ferry", skillId="builder:5", outcome="succeeded"),
        ])

    def test_actions_join_skill_alias_and_keep_failed_attempts_separate(self):
        """An alias, failed attempt and keep-going poll must not steal a running skill's evidence."""
        self._action_logs()
        rows = RunViews(self.directory).actions()
        joined = next(row for row in rows if row["tick"] == 10)
        self.assertEqual(joined["skill_end"]["skillId"], "builder:1")
        self.assertEqual(joined["skill_end"]["outcome"], "failed")
        self.assertEqual([entry["id"] for entry in joined["effects"]], [21])
        self.assertEqual(joined["outcome"], "started")
        self.assertEqual(joined["args"], {"schematic": "bridge"})
        for tick in (15, 20):
            row = next(row for row in rows if row["tick"] == tick)
            self.assertIsNone(row["skill_end"])
            self.assertEqual(row["effects"], [])
        self.assertEqual(next(row for row in rows if row["tick"] == 31)["tool"], None)
        self.assertFalse(any(row["tool"] in {"speak", "ask"} for row in rows))
        self.assertFalse(any(row["agent"] == "world" for row in rows))

    def test_actions_include_all_effect_types_and_unmatched_engine_evidence(self):
        """Transfers, purchases, document writes, ferry trips and staging events all remain visible."""
        self._action_logs()
        rows = RunViews(self.directory).actions()
        for tick, kind in ((60, "transfer"), (70, "purchase"), (80, "doc_write"), (90, "ferry_trip")):
            row = next(row for row in rows if row["tick"] == tick)
            self.assertEqual([entry["type"] for entry in row["effects"]], [kind])
        orphan = next(row for row in rows if row["tick"] == 5)
        self.assertIsNone(orphan["tool"])
        self.assertEqual(orphan["effects"][0]["actor"], "envoy")
        receiving = next(row for row in rows if row["tick"] == 61)
        self.assertEqual(receiving["agent"], "envoy")
        self.assertIsNone(receiving["tool"])

    def test_sprint_actions_join_new_effects_and_render_hold_without_an_event(self):
        """Drop, use, eat and collect own their effects; hold needs no invented event or ending."""
        self._jsonl("orchestrator/decisions.jsonl", [
            decision(5, "hold", outcome="done", item="minecraft:bread"),
            decision(10, "drop", item="minecraft:bread", count=2),
            decision(12, "drop", outcome="not possible: busy"),
            decision(20, "use", pos=[1, 2, 3]),
            decision(30, "eat", item="minecraft:bread"),
            decision(40, "collect", item="minecraft:bread"),
        ])
        effects = [event(15, "drop", bot="builder", item="minecraft:bread", count=2, itemEntity="item:1"),
                   event(25, "block_use", actor="builder", skillId="builder:20", block="minecraft:oak_door"),
                   event(35, "eat", bot="builder", item="minecraft:bread", count=1, foodBefore=10, foodAfter=15),
                   event(45, "pickup", bot="builder", item="minecraft:bread", count=1, itemEntity="item:1")]
        endings = [event(tick + 5, "skill_end", bot="builder", skill=tool,
                         skillId=f"builder:{tick}", outcome="succeeded")
                   for tick, tool in ((10, "drop"), (20, "use"), (30, "eat"), (40, "collect"))]
        self._jsonl("world/simharness/events.jsonl", [
            *effects, *endings, event(35, "hunger", bot="builder", before=10, after=15, reason="eat")])
        views = RunViews(self.directory)
        rows = views.actions()
        self.assertEqual(len(rows), 6)
        for tick, effect in zip((10, 20, 30, 40), effects):
            with self.subTest(effect=effect["type"]):
                row = next(row for row in rows if row["tick"] == tick)
                self.assertEqual(row["effects"], [effect])
                self.assertEqual(row["skill_end"]["skillId"], f"builder:{tick}")
                self.assertIn(json.dumps(effect["type"]), views.render("actions"))
        hold = rows[0]
        self.assertEqual((hold["tool"], hold["args"], hold["skill_end"], hold["effects"]),
                         ("hold", {"item": "minecraft:bread"}, None, []))
        self.assertEqual(rows[2]["effects"], [])
        self.assertNotIn('"type": "hunger"', views.render("actions"))
        self.assertNotIn("PRIVATE", json.dumps(rows))
        self.assertIn('"tool": "hold"', views.render("actions", until=5))

    def test_sprint_effects_without_decisions_remain_public_evidence(self):
        """Recorded actor fields attribute unmatched effects without guessing a causing action."""
        effects = [event(1, "drop", bot="builder"), event(2, "block_use", actor="builder"),
                   event(3, "eat", bot="builder"), event(4, "pickup", bot="builder"),
                   event(5, "pickup", actor="builder")]
        self._jsonl("world/simharness/events.jsonl", effects)
        rows = RunViews(self.directory).actions()
        self.assertEqual([row["effects"][0] for row in rows], effects)
        self.assertTrue(all(row["agent"] == "builder" and row["tool"] is None for row in rows))

    def test_death_without_decisions_renders_all_engine_fields(self):
        """A death remains visible with its raw damage-source id and unrounded engine position."""
        death = event(13800, "death", bot="envoy", cause="drown", pos=[0.5, -61.25, 41.5])
        self._jsonl("world/simharness/events.jsonl", [death])
        views = RunViews(self.directory)
        self.assertEqual(views.actions(), [{"tick": 13800, "agent": "envoy", "tool": None, "args": {},
                                           "outcome": None, "skill_end": None, "effects": [death]}])
        text = views.render("actions")
        self.assertTrue(text.startswith("[tick 13800; day 0, 19:48] EFFECT "))
        self.assertEqual(json.loads(text.split(" EFFECT ", 1)[1]),
                         {"agent": "envoy", "action_tick": 13800, "event": death})
        self.assertEqual(views.render("both"), text)
        self.assertEqual(views.render("speech"), "")

    def test_deaths_are_standalone_evidence_with_agent_and_time_filters(self):
        """A bot's last action does not own its death, and view filters cannot leak another bot's death."""
        self._jsonl("orchestrator/decisions.jsonl", [decision(10, "move_to"),
                                                    decision(20, "speak", agent="envoy", text="Help!")])
        builder = event(30, "death", bot="builder", cause="fall", pos=[1.5, -62, 3.5])
        envoy = event(40, "death", bot="envoy", cause="drown", pos=[0.5, -61, 41.5])
        self._jsonl("world/simharness/events.jsonl", [envoy, builder])
        views = RunViews(self.directory)
        rows = views.actions()
        self.assertEqual([row["tick"] for row in rows], [10, 30, 40])
        self.assertEqual(rows[0]["effects"], [])
        self.assertTrue(all(row["tool"] is None and row["skill_end"] is None for row in rows[1:]))
        self.assertEqual([row["effects"] for row in rows[1:]], [[builder], [envoy]])
        self.assertNotIn('"type": "death"', views.render("actions", until=29))
        self.assertIn('"cause": "fall"', views.render("actions", agents=["builder"], until=30))
        self.assertNotIn('"cause": "drown"', views.render("actions", agents=["builder"]))
        self.assertEqual(views.render("actions", agents=["envoy"], until=39), "")
        self.assertIn('"cause": "drown"', views.render("actions", agents=["envoy"], until=40))

    def test_future_events_are_generic_and_nested_private_fields_never_escape(self):
        """Unknown events naming a role agent survive without joining unrelated decisions or exposing private payloads;
        events naming no role agent (engine bookkeeping) stay out of the actions view."""
        self._jsonl("orchestrator/decisions.jsonl", [
            decision(1, "future_tool", payload={"note": "PRIVATE NESTED NOTE", "items": [
                {"user_prompt": "PRIVATE PROMPT", "private_goals": ["PRIVATE GOAL"], "item": "bread"}]}),
        ])
        self._jsonl("world/simharness/events.jsonl", [
            event(2, "future_effect", actor="builder", notes=["PRIVATE NOTE"],
                  payload={"goal": "PRIVATE GOAL", "item": "bread"}),
            event(3, "future_weather", state="rain", evidence=[{"note": "PRIVATE NOTE", "level": 2}]),
            event(4, "skill_end", bot="builder", skill="future_tool", outcome="succeeded", note="PRIVATE NOTE"),
        ])
        views = RunViews(self.directory)
        rows = views.actions()
        self.assertEqual(rows[0]["effects"], [])
        self.assertEqual(rows[1]["effects"][0]["type"], "future_effect")
        self.assertIsNone(rows[1]["tool"])
        self.assertEqual(len(rows), 2)
        self.assertNotIn("PRIVATE", json.dumps(rows))
        self.assertNotIn("PRIVATE", views.render("both"))
        self.assertNotIn("future_weather", views.render("actions"))
        self.assertIn("PRIVATE", json.dumps(views.events))

    def _exposure_logs(self):
        """Record private, overheard and wordless exchanges, including NPC identity and missing evidence."""
        self._jsonl("world/simharness/events.jsonl", [
            event(10, "utterance", speaker="builder", volume="whisper", to="envoy", text="Quiet words",
                  listeners=["envoy"], whisperSeen={"npc:watcher": {"losRay": 0}, "mayor": {"losRay": 0}}),
            event(20, "utterance", speaker="envoy", volume="talk", to="builder", text="Reply",
                  listeners=["builder", "mayor", "npc:witness"],
                  attentionLost={"npc:watcher": [{"uttId": 1}]}),
            event(30, "utterance", speaker="builder", volume="whisper", to="envoy", text="Unheard",
                  listeners=[], listenerEyes={"envoy": [0, 0, 0], "mayor": [0, 0, 0]},
                  listenerStartEyes={"envoy": [0, 0, 0]}, speakerStartEye=[0, 0, 0], pos=[0, 0, 0]),
            event(40, "utterance", speaker="builder", volume="shout", text="Broadcast", listeners=["mayor"]),
            event(50, "utterance", speaker="mayor", to="mayor", listeners=["builder"]),
            event(60, "utterance", speaker="envoy", speakerKey="npc:imposter", to="builder", listeners=["builder"]),
            event(70, "utterance", speaker="builder", to="Dahlia", volume="talk", listeners=["npc:adult_19"],
                  exchange={"requester": "builder", "npc": "npc:adult_19", "text": "PRIVATE UNDELIVERED ANSWER"}),
            event(80, "utterance", speaker="envoy", to="mayor", volume="talk", listeners=["mayor"]),
        ])

    def test_speech_exposes_whisper_volume_and_only_recorded_wordless_recipients(self):
        """Hearing takes precedence over chatter, whose evidence keys are sorted and deduplicated."""
        self._jsonl("world/simharness/events.jsonl", [
            event(1, "utterance", speaker="builder", volume="whisper", listeners=["envoy"],
                  whisperSeen={"npc:watcher": {}, "envoy": {}, "mayor": {}},
                  attentionLost={"mayor": [], "npc:other": []}, listenerEyes={"nearby": [0, 0, 0]}),
        ])
        views = RunViews(self.directory)
        row = views.speech()[0]
        self.assertEqual(row["volume"], "whisper")
        self.assertEqual(row["listeners"], ["envoy"])
        self.assertEqual(row["chatter"], ["mayor", "npc:other", "npc:watcher"])
        self.assertIn('"volume": "whisper"', views.render("speech"))
        self.assertNotIn("nearby", views.render("speech"))

    def test_exposure_rows_use_actual_audiences_not_geometry_or_private_exchange_text(self):
        """Only canonical role speakers count, and recorded NPC addressees can be matched to listeners."""
        self._exposure_logs()
        rows = RunViews(self.directory).exposure()
        self.assertEqual([row["tick"] for row in rows], [10, 20, 30, 40, 50, 70, 80])
        self.assertEqual((rows[0]["speaker"], rows[0]["to"], rows[0]["volume"], rows[0]["clock"]),
                         ("builder", "envoy", "whisper", "day 0, 06:00"))
        self.assertTrue(rows[0]["addressee_heard"])
        self.assertEqual(rows[0]["chatter"], ["mayor", "npc:watcher"])
        self.assertEqual(rows[1]["listeners"], ["builder", "mayor", "npc:witness"])
        self.assertEqual(rows[1]["chatter"], ["npc:watcher"])
        self.assertFalse(rows[2]["addressee_heard"])
        self.assertEqual(rows[2]["chatter"], [])
        self.assertIsNone(rows[3]["addressee_heard"])
        self.assertEqual(rows[5]["to_key"], "npc:adult_19")
        self.assertTrue(rows[5]["addressee_heard"])
        self.assertNotIn("text", rows[0])
        self.assertNotIn("PRIVATE", json.dumps(rows))

    def test_exposure_pair_counts_are_bidirectional_exchange_counts_not_audience_sizes(self):
        """A role and villager can expose the same exchange, counted only once in the union."""
        self._exposure_logs()
        self.assertEqual(RunViews(self.directory).exposure_pairs(), [
            {"agents": ["builder", "envoy"], "tick": 30, "exchanges": 3,
             "heard_by_third_party": 1, "chatter_by_third_party": 2, "perceived_by_third_party": 2},
            {"agents": ["envoy", "mayor"], "tick": 80, "exchanges": 1,
             "heard_by_third_party": 0, "chatter_by_third_party": 0, "perceived_by_third_party": 0},
        ])

    def test_exposure_cli_filters_before_counting_pairs(self):
        """Speaker and inclusive tick filters cannot leak later or opposite-speaker exchanges into counts."""
        self._exposure_logs()
        views = RunViews(self.directory)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main([str(self.directory), "--kind", "exposure", "--agents", "builder",
                                   "--until", "30"]), 0)
        rendered = stdout.getvalue()
        self.assertEqual(rendered, views.render("exposure", ["builder"], until=30))
        self.assertEqual(rendered.count("EXPOSURE"), 2)
        self.assertEqual(rendered.count(" PAIR "), 1)
        self.assertIn('"exchanges": 2', rendered)
        self.assertIn('"heard_by_third_party": 0', rendered)
        self.assertIn('"chatter_by_third_party": 1', rendered)
        self.assertNotIn("PRIVATE", views.render("exposure"))
        self.assertNotIn("PRIVATE", views.render("speech"))
        self.assertEqual(views.render("exposure", agents=[]), "")
        self.assertEqual(views.render("exposure", until=0), "")

    def test_render_bytes_are_identical_across_process_hash_seeds(self):
        """Set iteration must not affect any view, including pair order and generic evidence."""
        self._exposure_logs()
        self._jsonl("orchestrator/decisions.jsonl", [decision(1, "hold", outcome="done", item="bread")])
        script = ("from scripts.run_views import RunViews; import sys; "
                  "views = RunViews(sys.argv[1]); "
                  "print(''.join(views.render(kind) for kind in ('speech', 'actions', 'both', 'exposure')), end='')")
        outputs = [subprocess.run([sys.executable, "-c", script, str(self.directory)], check=True,
                                  cwd=Path(__file__).resolve().parents[2], capture_output=True,
                                  env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "987")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn(b"PRIVATE", outputs[0])

    def test_new_skill_bounds_a_missing_end_and_keeps_orphan_endings(self):
        """Missing endings cannot cause an older same-agent action to claim a later skill."""
        self._jsonl("orchestrator/decisions.jsonl", [decision(10, "mine"), decision(20, "wait")])
        self._jsonl("world/simharness/events.jsonl", [
            event(25, "skill_end", bot="builder", skill="mine", skillId="builder:old", outcome="cancelled"),
            event(30, "skill_end", bot="builder", skill="wait", skillId="builder:new", outcome="succeeded"),
        ])
        rows = RunViews(self.directory).actions()
        self.assertIsNone(rows[0]["skill_end"])
        self.assertEqual(rows[1]["skill_end"]["skillId"], "builder:new")
        self.assertIsNone(rows[2]["tool"])

    def test_observations_use_the_saved_items_shape(self):
        """Reading observations does not assume a flattened self or heard structure."""
        row = {"gameTime": 200, "observer": "envoy", "kind": "observe", "items": [
            {"category": "self", "name": "envoy", "pos": [1, 2, 3], "inventory": [],
             "skill": {"skillId": None, "state": "idle"}}]}
        self._jsonl("world/simharness/observations.jsonl", [row])
        self.assertEqual(RunViews(self.directory).observations, [row])

    def test_meetings_are_pairwise_three_dimensional_inclusive_samples(self):
        """Close pairs split on separation, unknown positions and missing synchronous samples."""
        positions = {0: {"builder": [0, 0, 0], "envoy": [8, 0, 0], "npc:1": [0, 0, 0]},
                     10: {"builder": [0, 0, 0], "envoy": [0, 8, 0]},
                     20: {"builder": [0, 0, 0], "envoy": [0, 9, 0]},
                     30: {"builder": [0, 0, 0], "envoy": [1, 0, 0]},
                     40: {"builder": [0, 0, 0]},
                     50: {"builder": [0, 0, 0], "envoy": [1, 0, 0], "mayor": [0, 0, 0]},
                     60: {"builder": None, "envoy": [1, 0, 0], "mayor": [0, 0, 0]}}
        self._jsonl("orchestrator/trajectory.jsonl", [
            {"t": tick, "agent": agent, "pos": pos, "skill": None, "state": "idle"}
            for tick, snapshot in reversed(list(positions.items())) for agent, pos in snapshot.items()])
        views = RunViews(self.directory)
        meetings = views.meetings()
        pair = [row for row in meetings if row["agents"] == ["builder", "envoy"]]
        self.assertEqual([(row["start_tick"], row["end_tick"]) for row in pair], [(0, 10), (30, 30), (50, 50)])
        other_pair = next(row for row in meetings if row["agents"] == ["envoy", "mayor"])
        self.assertEqual((other_pair["start_tick"], other_pair["end_tick"]), (50, 60))
        self.assertTrue(all(len(row["agents"]) == 2 for row in meetings))
        self.assertEqual(views.meetings(0), [{"start_tick": 50, "end_tick": 50, "agents": ["builder", "mayor"]}])
        for radius in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                views.meetings(radius)

    def test_render_is_deterministic_public_and_globally_chronological(self):
        """Rendering excludes private inputs and orders future joined evidence at its actual time."""
        self._action_logs()
        views = RunViews(self.directory)
        rendered = views.render("both")
        self.assertEqual(rendered, RunViews(self.directory).render("both"))
        self.assertNotIn("PRIVATE", rendered)
        ticks = [int(value) for value in re.findall(r"^\[tick (\d+);", rendered, re.MULTILINE)]
        self.assertEqual(ticks, sorted(ticks))
        self.assertIn('"text": "Hello"', rendered)
        limited = views.render("both", agents=["builder"], until=21)
        self.assertIn('"id": 21', limited)
        self.assertNotIn('"agent": "envoy"', limited)
        self.assertNotIn("missing materials", limited)
        self.assertNotIn("SKILL_END", limited)
        self.assertNotIn("SPEECH", limited)
        self.assertEqual(views.render("both", agents=[]), "")
        self.assertEqual(views.render("both", until=-1), "")
        with self.assertRaises(ValueError):
            views.render("meetings")

    def test_render_speech_filter_accepts_canonical_npc_keys(self):
        """An NPC's stable key can select speech even though it is displayed under a name."""
        self._jsonl("world/simharness/events.jsonl", [
            event(1, "utterance", speaker="Dahlia", speakerKey="npc:adult_19", volume="talk",
                  text="News", listeners=["envoy"]),
        ])
        views = RunViews(self.directory)
        self.assertIn("News", views.render("speech", ["npc:adult_19"]))
        self.assertEqual(views.render("speech", ["envoy"]), "")

    def test_optional_logs_cli_and_read_only_input(self):
        """Empty runs work; CLI output changes only the explicitly chosen destination."""
        self.assertEqual(RunViews(self.directory).render("both"), "")
        self._action_logs()
        originals = {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()}
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir) / "view.txt"
            self.assertEqual(main([str(self.directory), "--out", str(output), "--kind", "actions",
                                   "--agents", "builder, envoy", "--until", "21"]), 0)
            self.assertEqual(output.read_text(), RunViews(self.directory).render("actions", ["builder", "envoy"], 21))
        self.assertEqual(originals, {path: path.read_bytes() for path in self.directory.rglob("*") if path.is_file()})
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main([str(self.directory), "--kind", "speech"]), 0)
        self.assertIn("SPEECH", stdout.getvalue())

    def test_invalid_run_and_malformed_json_are_reported(self):
        """Missing scenarios and corrupt logs fail explicitly instead of creating empty evidence."""
        with self.assertRaises(ValueError):
            RunViews(self.directory / "missing")
        self._jsonl("world/simharness/events.jsonl", [])
        (self.directory / "world/simharness/events.jsonl").write_text("{bad json}")
        with self.assertRaises(json.JSONDecodeError):
            RunViews(self.directory)
