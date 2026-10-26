"""Non-gating M3c sight, speech, and rumor probes in the built village.

Run explicitly to start Minecraft. Six independent village clones retain their
thirty villagers and six idle role agents, adding six navigating/dwelling bots.
Curves sample every 200 ticks for two full days after the day-1 phase-500 event.
Missing thresholds are null, not zero; speech reach excludes physical witnesses
from both counts and denominators. Rumors have no physical intervention.
Plausibility never determines exit status.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

if __package__:
    from . import audit_o1, harness, test_v1
    from .prune_runs import prune_run
else:
    import audit_o1
    import harness
    import test_v1
    from prune_runs import prune_run


SAMPLE_TICKS = 200
ROUTINE_TICKS = 20
DWELL_TICKS = 200
EVENT_TICK = test_v1.CLOSURE_TICK
END_TICK = EVENT_TICK + 2 * test_v1.DAY_TICKS
MAJOR_POS = [-12, -60, 55]
MINOR_POS = [112, -60, -48]
RUMOR_TEXT = "The well is poisoned! Don't drink!"
REALISTIC = {"speech_rate": 0.15, "talk_radius": 8.0, "shout_radius": 24.0,
             "shout_cooldown_ticks": 600, "attention_slots": 2, "relay_rate": 0.5, "news_ttl_days": 3}
NAIVE = {**REALISTIC, "speech_rate": 0.0, "talk_radius": 16.0, "shout_radius": 16.0,
         "shout_cooldown_ticks": 0, "attention_slots": 0}
# Road-following loop through the west town (straight legs along roads, so steering never meets a wall):
# market -> west spine -> school road -> spine -> town-hall road -> spine -> back to the market.
LOOP = ((-52.5, -60, -3.5), (-11.5, -60, -3.5), (-11.5, -60, 36.5), (-59.5, -60, 36.5), (-11.5, -60, 36.5),
        (-11.5, -60, 74.5), (-59.5, -60, 74.5), (-11.5, -60, 74.5), (-11.5, -60, -3.5))


class SpreadCurve:
    """Separate physical witnesses from speech-informed non-witnesses by population."""

    def __init__(self, event_tick: int):
        """Start an empty curve referenced to the intervention tick."""
        self._event_tick = event_tick
        self._samples = []

    @staticmethod
    def _counts(npcs: set[str], bots: set[str], npc_population: int, bot_population: int) -> dict:
        """Describe a channel with its actual eligible population and nullable fractions."""
        population = npc_population + bot_population
        return {"npcCount": len(npcs), "botCount": len(bots), "count": len(npcs) + len(bots),
                "npcPopulation": npc_population, "botPopulation": bot_population, "population": population,
                "npcFraction": len(npcs) / npc_population if npc_population else None,
                "botFraction": len(bots) / bot_population if bot_population else None,
                "fraction": (len(npcs) + len(bots)) / population if population else None}

    def sample(self, tick: int, npcs: set[str], bots: set[str], *,
               npc_witnesses: set[str] | None = None, bot_witnesses: set[str] | None = None) -> None:
        """Partition knowledge, using the remaining non-witnesses for speech fractions.

        Witnesses remain known even if absent from current news. Hearing followed
        by sight moves a listener into witnesses, never into both groups. Empty
        non-witness populations have null fractions and cannot reach thresholds.
        """
        if tick < self._event_tick or self._samples and tick <= self._samples[-1]["gameTime"]:
            raise ValueError("Spread samples must increase from the event tick")
        npc_witnesses, bot_witnesses = npc_witnesses or set(), bot_witnesses or set()
        npcs, bots = npcs | npc_witnesses, bots | bot_witnesses
        if len(npcs) > 30 or len(bots) > 6:
            raise ValueError("Spread sample exceeds the configured population")
        total = self._counts(npcs, bots, 30, 6)
        self._samples.append({"gameTime": tick, "elapsedTicks": tick - self._event_tick,
                              **{key: total[key] for key in ("npcCount", "botCount", "npcFraction",
                                                            "botFraction", "fraction")},
                              "witnesses": self._counts(npc_witnesses, bot_witnesses, 30, 6),
                              "informed_by_speech": self._counts(npcs - npc_witnesses, bots - bot_witnesses,
                                                                 30 - len(npc_witnesses), 6 - len(bot_witnesses)),
                              "total": total})

    def report(self) -> dict:
        """Return sampled threshold times and final reach without interpolation."""
        def thresholds(field, channel="total"):
            return {str(percent): next((sample["elapsedTicks"] for sample in self._samples
                                        if sample[channel][field] is not None
                                        and sample[channel][field] >= percent / 100), None) for percent in (25, 50, 75)}

        speech = self._samples[-1]["informed_by_speech"] if self._samples else {}
        return {"samples": copy.deepcopy(self._samples), "timeToPercentTicks": thresholds("fraction"),
                "npcTimeToPercentTicks": thresholds("npcFraction"),
                "botTimeToPercentTicks": thresholds("botFraction"),
                "speechTimeToPercentTicks": thresholds("fraction", "informed_by_speech"),
                "npcSpeechTimeToPercentTicks": thresholds("npcFraction", "informed_by_speech"),
                "botSpeechTimeToPercentTicks": thresholds("botFraction", "informed_by_speech"),
                "finalReach": self._samples[-1]["fraction"] if self._samples else None,
                "finalNpcReach": self._samples[-1]["npcFraction"] if self._samples else None,
                "finalBotReach": self._samples[-1]["botFraction"] if self._samples else None,
                "finalSpeechReach": speech.get("fraction"), "finalNpcSpeechReach": speech.get("npcFraction"),
                "finalBotSpeechReach": speech.get("botFraction")}


class _RoutineBot:
    """Navigate a shared physical loop, dwelling at each stop to receive gossip."""

    def __init__(self, index: int, cooldown: int):
        """Wrap six bots' starting waypoints around the loop without consuming RNG."""
        self._name = f"comm_{index}"
        self._waypoint = index % len(LOOP)
        self._dwelling_until = 0
        self._next_shout = EVENT_TICK
        self._cooldown = cooldown
        self._shouts = 0

    @property
    def name(self) -> str:
        """Return the bot's stable control identity."""
        return self._name

    @property
    def shouts(self) -> int:
        """Return the number of accepted adversarial utterances."""
        return self._shouts

    def spawn(self, client: harness.Client) -> None:
        """Create the bot at its first standing waypoint."""
        horizontal, height, vertical = LOOP[self._waypoint]
        client.request("spawn_bot", {"name": self.name, "x": horizontal, "y": height, "z": vertical,
                                     "yaw": 0.0, "pitch": 0.0})
        self._waypoint = (self._waypoint + 1) % len(LOOP)

    @staticmethod
    def heading(position: tuple, target: tuple) -> tuple[float, float]:
        """Return horizontal distance and Minecraft yaw for one physical leg."""
        horizontal, vertical = target[0] - position[0], target[2] - position[2]
        return math.hypot(horizontal, vertical), math.degrees(math.atan2(-horizontal, vertical))

    def _input(self, client: harness.Client, forward: float = 0.0, yaw: float = 0.0) -> None:
        client.request("bot_input", {"name": self.name, "forward": forward, "strafe": 0.0,
                                      "jump": False, "sprint": False, "sneak": False, "yaw": yaw, "pitch": 0.0})

    def _steer(self, client: harness.Client, target: tuple) -> float:
        position = test_v1.vector(client.request("bot_state", {"name": self.name}).get("pos"), "routine bot position")
        distance, yaw = self.heading(position, target)
        self._input(client, 1.0 if distance > 0.75 else 0.0, yaw)
        return distance

    @staticmethod
    def _travel_delay(distance: float) -> int:
        return max(1, min(ROUTINE_TICKS, int(distance / 0.22)))

    def witness(self, client: harness.Client, tick: int, position: list[int]) -> int:
        """Walk to a nearby lookout before the event, then dwell through it."""
        horizontal, height, vertical = position
        distance = self._steer(client, (horizontal - 5.5, height, vertical + 8.5))
        self._dwelling_until = 0
        return tick + (self._travel_delay(distance) if distance > 0.75 else ROUTINE_TICKS)

    def drive(self, client: harness.Client, tick: int, shout_text: str | None) -> int:
        """Advance navigation or timed speech and return the next service tick.

        Persistent physical inputs follow a loop whose legs avoid the closure.
        Timed speech stops walking; instantaneous speech restores walking inputs
        immediately, without replanning A* on every naive shout. Zero-cost naive
        speech is attempted once per game tick, never in an infinite loop.
        """
        status = client.request("skill_status", {"name": self.name})
        if shout_text is not None and tick >= self._next_shout:
            if status.get("state") != "running" or status.get("skill") != "speak":
                payload = {"name": self.name, "skill": "speak", "args": {"text": shout_text, "volume": "shout"}}
                dry = client.request("skill_start", {**payload, "dry_run": True})
                if dry.get("feasible") is True:
                    duration = dry["estimate"]["ticks"]
                    test_v1.check(type(duration) is int and duration >= 0, "Invalid speech duration")
                    self._input(client)
                    accepted = client.speak(self.name, shout_text, "shout")
                    test_v1.check(accepted.get("accepted") is True, "Feasible adversarial shout was rejected")
                    self._shouts += 1
                    self._next_shout = tick + max(1, duration, self._cooldown)
                    if duration:
                        return min(tick + duration, self._next_shout)
                    status = {"state": "succeeded"}
                else:
                    test_v1.check(dry.get("reason") in {"rate_limited", "busy"}, f"Unexpected shout rejection: {dry}")
                    return tick + 1
            else:
                return tick + 1
        if status.get("state") == "running":
            self._input(client)
            return min(tick + ROUTINE_TICKS, self._next_shout) if shout_text else tick + ROUTINE_TICKS
        delay = ROUTINE_TICKS
        if tick < self._dwelling_until:
            self._input(client)
        else:
            distance = self._steer(client, LOOP[self._waypoint])
            if distance <= 0.75:
                self._waypoint = (self._waypoint + 1) % len(LOOP)
                self._dwelling_until = tick + DWELL_TICKS
            else:
                delay = self._travel_delay(distance)
        return min(tick + delay, self._next_shout) if shout_text else tick + delay


class CommunicationProbe:
    """Own isolated communication scenarios, knowledge curves, and one report."""

    def __init__(self, args):
        """Retain CLI settings without creating directories or starting servers."""
        self._args = args
        self._source = args.runs_dir / f"village_{args.variant}"
        self._directory = args.runs_dir / "probe_comm"
        self._report = {"gating": False, "sampleTicks": SAMPLE_TICKS, "eventTick": EVENT_TICK,
                        "durationTicks": 2 * test_v1.DAY_TICKS, "npcs": 30, "bots": 6,
                        "naiveOpportunityTicks": 1,
                        "knowledge": "Physical witnesses come only from their own admitted change observations. "
                                     "NPCs keep current matching news; bots retain delivered matching speech. "
                                     "Speech-informed excludes every physical witness; total is their union. "
                                     "Rumors require exact text in full heard speech or relayed shout news; "
                                     "the source is not pre-counted and firsthand shout news is still speech.",
                        "thresholds": "First 200-tick sample reaching threshold; null means not reached. "
                                      "Speech fractions use the non-witness population at each sample; "
                                      "an empty population has a null fraction.",
                        "scenarios": {}, "errors": []}

    def setup(self) -> None:
        """Read the village roster and create a new report directory without a server."""
        manifest = harness.REPO_ROOT / "scenario" / f"village_{self._args.variant}.json"
        self._npc_ids = [npc["id"] for npc in json.loads(manifest.read_text(encoding="utf-8"))["npcs"]]
        self._directory.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def matches(value: dict, event: dict) -> bool:
        """Match subjects or claims, never treating a shout's location as its subject."""
        if any(CommunicationProbe.matches(news, event) for news in value.get("news", [])):
            return True
        if event.get("kind") == "rumor":
            return value.get("text") == event["text"] and value.get("kind", "shout") == "shout"
        origin = value.get("originEventId")
        if origin is not None and str(origin) in (str(event["originEventId"]), f"change:{event['originEventId']}"):
            return True
        subject = None if value.get("kind") == "shout" else value.get("subject")
        if isinstance(subject, dict):
            subject = subject.get("pos")
        if subject == event["pos"] or value.get("kind") != "shout" and value.get("pos") == event["pos"]:
            return True
        for candidate in (subject, value.get("text")):
            if isinstance(candidate, str):
                compact = "".join(candidate.split())
                if json.dumps(event["pos"], separators=(",", ":")) in compact:
                    return True
        return False

    @staticmethod
    def _npc_witnesses(directory: Path, event: dict) -> dict:
        """Stream the closed sight log once, retaining first observation times, not event times."""
        if event["kind"] == "rumor":
            return {}
        witnesses = {}
        with (directory / "world/simharness/npc_observations.jsonl").open(encoding="utf-8") as records:
            for line in records:
                record = json.loads(line)
                for item in record.get("changes", []):
                    if item.get("prov") == f"change:{event['originEventId']}":
                        witnesses.setdefault(record["npc"], {"gameTime": record["gameTime"], "item": item})
        return witnesses

    def _intervene(self, client: harness.Client, directory: Path, minor: bool) -> dict:
        """Track one exact origin from a full-width road wall or an off-road block."""
        position = MINOR_POS if minor else MAJOR_POS
        if minor:
            horizontal, height, vertical = position
            test_v1.command(client, f"setblock {horizontal} {height} {vertical} minecraft:oak_planks")
        else:
            test_v1.command(client, "fill -13 -60 55 -10 -60 55 minecraft:cobblestone_wall")
        client.request("save_snapshot")
        changes = [event for event in audit_o1.read_jsonl(directory / "world/simharness/events.jsonl")
                   if event.get("type") == "block_change" and event.get("gameTime") == EVENT_TICK
                   and event.get("pos") == position]
        test_v1.check(len(changes) == 1, "Intervention lacks one exact block-change origin")
        return {"kind": "construction" if minor else "road_blocked", "pos": list(position),
                "originEventId": changes[0]["id"], "gameTime": EVENT_TICK}

    def _scenario(self, label: str, *, minor: bool = False, adversarial: bool = False,
                  naive: bool = False, rumor: bool = False) -> dict:
        """Restore one isolated village clone and measure the existing six-bot probe."""
        directory = self._directory / label
        harness.clone_apfs(self._source, directory)
        config = NAIVE if naive else REALISTIC
        bots = [_RoutineBot(index, config["shout_cooldown_ticks"]) for index in range(6)]
        curve = SpreadCurve(EVENT_TICK)
        known_bots, witnessed = set(), set()
        evidence, npc_evidence = {}, {}
        samples = []
        event = None
        with harness.Server(directory, **harness.server_options(self._args)) as server:
            client = server.client
            tick = harness.enable_lockstep(client, recording=False)
            test_v1.check(tick < 1000, "Built village must precede day-0 departures")
            client.request("restore_bots")
            client.request("config_set", {**config, "exposure_log": 0})  # the probe never reads the exposure log
            for bot in bots:
                bot.spawn(client)
            client.request("recording", {"enabled": True})
            service_ticks = {bot.name: tick for bot in bots}
            next_sample = EVENT_TICK
            while True:
                if tick == EVENT_TICK and event is None:
                    event = ({"kind": "rumor", "text": RUMOR_TEXT, "speaker": bots[0].name, "gameTime": EVENT_TICK}
                             if rumor else self._intervene(client, directory, minor))
                for bot in bots:
                    if tick >= service_ticks[bot.name] or tick == next_sample:
                        observation = client.request("observe", {"name": bot.name, "include_terrain": False})
                        if event is not None:
                            for item in observation.get("changes", []) if not rumor else []:
                                if item.get("prov") == f"change:{event['originEventId']}":
                                    witnessed.add(bot.name)
                                    known_bots.add(bot.name)
                                    evidence.setdefault(bot.name, {"gameTime": tick, "channel": "sight", "item": item})
                            for item in observation.get("heard", []):
                                if self.matches(item, event):
                                    known_bots.add(bot.name)
                                    evidence.setdefault(bot.name, {"gameTime": tick, "channel": "heard", "item": item})
                        if tick < END_TICK:
                            shout = None
                            if (adversarial and event is not None and bot is bots[0]
                                    and (rumor or bot.name in known_bots)):
                                shout = (RUMOR_TEXT if rumor else
                                         f"road_blocked at {json.dumps(event['pos'], separators=(',', ':'))}")
                            witness = bots[3] if minor else bots[0]
                            if not rumor and bot is witness and EVENT_TICK - 1000 <= tick < EVENT_TICK:
                                service_ticks[bot.name] = bot.witness(client, tick, MINOR_POS if minor else MAJOR_POS)
                            else:
                                service_ticks[bot.name] = bot.drive(client, tick, shout)
                if event is not None and tick == next_sample:
                    informed = set()
                    for npc in self._npc_ids:
                        state = client.request("npc_state", {"id": npc})
                        news = state.get("news")
                        test_v1.check(isinstance(news, list), "npc_state must expose its news list")
                        matching = [item for item in news if self.matches(item, event)]
                        if matching:
                            informed.add(npc)
                            npc_evidence.setdefault(npc, {"gameTime": tick, "item": matching[0]})
                    samples.append((tick, informed, set(known_bots), set(witnessed)))
                    next_sample += SAMPLE_TICKS
                if tick == END_TICK:
                    client.request("save_snapshot")
                    break
                following = min(END_TICK, next_sample, *service_ticks.values())
                if event is None:
                    following = min(following, EVENT_TICK)
                test_v1.check(following > tick, "Probe scheduler did not advance")
                tick = harness.game_time(client.request("step", {"n": following - tick}))
                test_v1.check(tick == following, "Probe step returned the wrong tick")
        npc_witnesses = self._npc_witnesses(directory, event)
        prune_run(directory)  # the measurements are extracted; free the scenario's world (several GB)
        for tick, informed_npcs, informed_bots, bot_witnesses in samples:
            seen_npcs = {npc for npc, record in npc_witnesses.items() if record["gameTime"] <= tick}
            curve.sample(tick, informed_npcs, informed_bots, npc_witnesses=seen_npcs, bot_witnesses=bot_witnesses)
        return {"config": dict(config), "event": event, "curve": curve.report(),
                "shouts": bots[0].shouts, "botEvidence": evidence, "npcEvidence": npc_evidence,
                "botSightWitnesses": sorted(witnessed), "npcSightWitnesses": sorted(npc_witnesses),
                "npcSightEvidence": npc_witnesses}

    def run(self) -> None:
        """Run all scenarios independently, retaining operational errors for repair."""
        cases = (("major", {}), ("minor", {"minor": True}),
                 ("adversarial_realistic", {"adversarial": True}),
                 ("adversarial_naive", {"adversarial": True, "naive": True}),
                 ("rumor_realistic", {"adversarial": True, "rumor": True}),
                 ("rumor_naive", {"adversarial": True, "naive": True, "rumor": True}))
        for label, options in cases:
            try:
                self._report["scenarios"][label] = self._scenario(label, **options)
            except test_v1.ERRORS as error:
                message = f"{label}: {type(error).__name__}: {error}"
                self._report["errors"].append(message)
                print(f"ERROR communication probe: {message}")

    @staticmethod
    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        """Return a finite ratio, or null for unreached/zero denominators."""
        return numerator / denominator if numerator is not None and denominator else None

    def _comparison(self, realistic: dict, naive: dict) -> dict:
        """Compare final reach and every sampled threshold without inventing missing times."""
        times = {percent: self.ratio(realistic["timeToPercentTicks"][percent], naive["timeToPercentTicks"][percent])
                 for percent in ("25", "50", "75")}
        return {"reachRatioRealisticOverNaive": self.ratio(realistic["finalReach"], naive["finalReach"]),
                "timeTo50RatioRealisticOverNaive": times["50"], "timeToPercentRatioRealisticOverNaive": times}

    @staticmethod
    def _less_traction(realistic: dict, naive: dict) -> bool:
        """Require actual shouts and naive reach before describing reduced realistic traction."""
        realistic_curve, naive_curve = realistic["curve"], naive["curve"]
        realistic_time = realistic_curve["timeToPercentTicks"]["50"]
        naive_time = naive_curve["timeToPercentTicks"]["50"]
        return realistic["shouts"] > 0 and naive["shouts"] > 0 and naive_curve["finalReach"] > 0 and (
            realistic_curve["finalReach"] <= 0.75 * naive_curve["finalReach"]
            or naive_time is not None and (realistic_time is None or realistic_time >= 2 * max(1, naive_time)))

    def report(self) -> int:
        """Persist curves and print descriptive plausibility, never a realism gate."""
        scenarios = self._report["scenarios"]
        curves = {label: result["curve"] for label, result in scenarios.items()}
        if {"major", "minor", "adversarial_realistic", "adversarial_naive"} <= scenarios.keys():
            realistic, naive = curves["adversarial_realistic"], curves["adversarial_naive"]
            speech_informed = curves["major"]["samples"][-1]["informed_by_speech"]["npcCount"] > 0
            self._report["comparison"] = {**self._comparison(realistic, naive),
                "realisticReachGainOverPassive": realistic["finalReach"] - curves["major"]["finalReach"]}
            self._report["plausibility"] = {
                "majorReachesMostNpcsViaGossip": curves["major"]["finalNpcReach"] >= 0.75 and speech_informed,
                "majorHasSpeechInformedNpcs": speech_informed,
                "minorStaysLocal": 0 < curves["minor"]["finalReach"] <= 0.25,
                "realisticAdversaryHasFarLessReachOrSlowerSpread":
                    self._less_traction(scenarios["adversarial_realistic"], scenarios["adversarial_naive"])}
            self._report["plausibilityCriteria"] = {
                "major": "At least 75% of NPCs, including speech-informed non-witnesses",
                "majorSpeech": "At least one speech-informed NPC non-witness after two days",
                "minor": "Nonzero reach of at most 25%",
                "adversary": "Both shouters active and naive reach nonzero; reach at most 75% of naive, "
                             "or time-to-50 at least twice naive (or unreached)"}
            print("Communication comparison: " + json.dumps(self._report["comparison"], sort_keys=True))
        if {"rumor_realistic", "rumor_naive"} <= scenarios.keys():
            self._report["rumorComparison"] = self._comparison(curves["rumor_realistic"], curves["rumor_naive"])
            self._report.setdefault("plausibility", {})["realisticRumorHasFarLessReachOrSlowerSpread"] = (
                self._less_traction(scenarios["rumor_realistic"], scenarios["rumor_naive"]))
            self._report.setdefault("plausibilityCriteria", {})["rumor"] = (
                "Both shouters active and naive reach nonzero; two-day speech-only reach at most 75% of naive, "
                "or time-to-50 at least twice naive (or unreached)")
            print("Rumor comparison: " + json.dumps(self._report["rumorComparison"], sort_keys=True))
        for label, plausible in self._report.get("plausibility", {}).items():
            print(f"PLAUSIBILITY (non-gating) {label}: {'observed' if plausible else 'not observed'}")
        for label, result in scenarios.items():
            curve = result["curve"]
            print(f"{label}: t25/t50/t75={curve['timeToPercentTicks']}; final reach={curve['finalReach']:.3f}; "
                  f"speech t25/t50/t75={curve['speechTimeToPercentTicks']}; "
                  f"final speech reach={curve['finalSpeechReach']}")
        destination = self._directory / "report.json"
        destination.write_text(json.dumps(self._report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Communication report: {destination}")
        return int(bool(self._report["errors"]))


def main(argv: list[str] | None = None) -> int:
    """Select a built village and run the explicitly requested communication probe."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    probe = CommunicationProbe(parser.parse_args(argv))
    probe.setup()
    probe.run()
    return probe.report()


if __name__ == "__main__":
    raise SystemExit(main())
