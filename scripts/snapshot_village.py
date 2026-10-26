"""Capture a viewable snapshot of a village run: the map, every villager and agent, and the run's history.

It works on a disposable clone of a finished run, so the run itself is never touched. The clone is advanced
`--advance` ticks (agents idle) so the snapshot can show the town mid-routine. The script writes `snapshot.json`
(top-down map, people, agents with their latest prompt, timeline), which `village_view.html` renders.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

if __package__:
    from . import harness
else:
    import harness

REGION = (-128, 127)
LAYERS = (-57, -58, -59, -60, -61, -62)  # top-down: the first non-air block wins


class VillageSnapshot:
    """Reads a run's world (through a clone) and its logs into one JSON document for the viewer."""

    def __init__(self, run: Path, manifest: Path):
        """Point at a finished run directory and its scenario manifest."""
        self._run = Path(run)
        self._manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))

    def capture(self, args) -> dict:
        """Clone the run, advance it, and read the map and everyone's state; then read the run's logs."""
        scratch = args.runs_dir / f"{self._run.name}_snapshot"
        if scratch.exists():
            shutil.rmtree(scratch)
        harness.clone_apfs(self._run, scratch)
        try:
            with harness.Server(scratch, **harness.server_options(args)) as server:
                client = server.client
                start = harness.enable_lockstep(client, recording=False)
                client.request("restore_bots")
                if args.advance:
                    client.request("step", {"n": args.advance})
                now = harness.game_time(client.request("ping"))
                snapshot = {"gameTime": now, "runEnd": start, "map": self._map(client),
                            "npcs": self._npcs(client), "agents": self._agents(client)}
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        snapshot.update(self._history())
        snapshot["places"] = [{"name": p["name"], "pos": p["pos"], "kind": p["kind"]} for p in self._manifest["places"]]
        snapshot["variant"] = self._manifest["variant"]
        return snapshot

    def map_layers(self, client) -> dict:
        """Every block in the top layers (y = -57 down to -62) of the region,
        as a shared palette and one grid per layer.

        A viewer applies logged block changes to these layers and shows the first non-air block per column.
        """
        low, high = REGION
        columns = [(x, z) for z in range(low, high + 1) for x in range(low, high + 1)]
        names, layers = {}, []
        for y in LAYERS:
            grid = []
            for begin in range(0, len(columns), 8192):
                batch = columns[begin:begin + 8192]
                blocks = client.request("blocks", {"positions": [[x, y, z] for x, z in batch]})["blocks"]
                grid.extend(names.setdefault(block, len(names)) for block in blocks)
            layers.append({"y": y, "grid": grid})
        palette = sorted(names, key=names.get)
        return {"origin": low, "size": high - low + 1, "palette": palette, "layers": layers}

    def capture_map(self, args) -> dict:
        """Capture the base map layers from a clone of the run (the built village, before anything happens)."""
        scratch = args.runs_dir / f"{self._run.name}_mapcapture"
        if scratch.exists():
            shutil.rmtree(scratch)
        harness.clone_apfs(self._run, scratch)
        try:
            with harness.Server(scratch, **harness.server_options(args)) as server:
                harness.enable_lockstep(server.client, recording=False)
                result = self.map_layers(server.client)
                result["gameTime"] = harness.game_time(server.client.request("ping"))
                return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _map(self, client) -> dict:
        """Top-down block per column (first non-air block from the top layers), as a palette and index grid."""
        low, high = REGION
        columns = [(x, z) for z in range(low, high + 1) for x in range(low, high + 1)]
        top: dict[tuple[int, int], str] = {}
        for y in LAYERS:
            pending = [column for column in columns if column not in top]
            for begin in range(0, len(pending), 8192):
                batch = pending[begin:begin + 8192]
                blocks = client.request("blocks", {"positions": [[x, y, z] for x, z in batch]})["blocks"]
                for column, block in zip(batch, blocks):
                    if block != "minecraft:air":
                        top[column] = block
        palette = sorted(set(top.values()) | {"minecraft:air"})
        index = {name: i for i, name in enumerate(palette)}
        grid = [index[top.get(column, "minecraft:air")] for column in columns]
        return {"origin": low, "size": high - low + 1, "palette": palette, "grid": grid}

    def _npcs(self, client) -> list[dict]:
        """Every villager's position, activity, trip, and the news it knows."""
        result = []
        for npc in self._manifest["npcs"]:
            state = client.request("npc_state", {"id": npc["id"]})
            trip = state.get("trip") or {}
            result.append({"id": npc["id"], "name": state["name"], "role": npc["role"], "pos": state["pos"],
                           "activity": state["activity"], "home": npc["home"], "work": npc.get("work"),
                           "school": npc.get("school"), "to": trip.get("to"),
                           "news": [{"kind": n["kind"], "subject": n["subject"], "chain": n["chain"],
                                     "firsthand": n["firsthand"]} for n in state.get("news", [])]})
        return result

    def _agents(self, client) -> list[dict]:
        """Every role agent's position, inventory and current skill."""
        result = []
        for role in self._manifest["roles"]:
            state = client.request("bot_state", {"name": role["name"]})
            skill = client.request("skill_status", {"name": role["name"]})
            result.append({"name": role["name"], "pos": state["pos"], "public_goal": role["public_goal"],
                           "private_goal": role["private_goal"],
                           "inventory": [{"item": i["item"].removeprefix("minecraft:"), "count": i["count"]}
                                         for i in state["inventory"]],
                           "skill": {"name": skill.get("skill"), "state": skill.get("state"),
                                     "reason": skill.get("reason")}})
        return result

    def _history(self) -> dict:
        """Decisions per agent (with the latest prompt), speech, transfers, and the run report."""
        decisions = [json.loads(line) for line in (self._run / "orchestrator/decisions.jsonl").open(encoding="utf-8")]
        by_agent: dict[str, list] = {}
        latest_prompt = {}
        for decision in decisions:
            by_agent.setdefault(decision["agent"], []).append(
                {"t": decision["game_time"], "tool": decision["tool"],
                 "args": {k: v for k, v in (decision["args"] or {}).items() if k != "note"},
                 "outcome": decision["outcome"], "note": (decision["args"] or {}).get("note")})
            latest_prompt[decision["agent"]] = {"t": decision["game_time"], "prompt": decision["user_prompt"]}
        speech, transfers, changes = [], [], Counter()
        for line in (self._run / "world/simharness/events.jsonl").open(encoding="utf-8"):
            event = json.loads(line)
            if event["type"] == "utterance":
                speech.append({"t": event["gameTime"], "speaker": event["speaker"], "text": event["text"],
                               "volume": event.get("volume"), "heardBy": len(event.get("listeners", [])),
                               "news": len(event.get("news", []))})
            elif event["type"] in {"transfer", "purchase", "ferry_trip"}:
                transfers.append({"t": event["gameTime"], **{k: event.get(k) for k in
                                                             ("type", "from", "to", "item", "count", "buyer", "cost")}})
            elif event["type"] == "block_change":
                changes[event.get("actor", "world")] += 1
        report_path = self._run / "m5_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        return {"decisions": by_agent, "latestPrompt": latest_prompt, "speech": speech, "transfers": transfers,
                "blockChangesByActor": dict(changes),
                "report": {key: report.get(key) for key in ("first_true", "welfare", "outcomes", "stats")}}


def main(argv: list[str] | None = None) -> int:
    """Capture a snapshot of `--run` and write it next to the viewer as `snapshot.json`."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    parser = harness.experiment_parser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", choices=("accident", "sabotage"), default="sabotage")
    parser.add_argument("--advance", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-only", action="store_true",
                        help="write only the base map layers (for the viewer: runs/village_<variant>/viewer_map.json)")
    args = parser.parse_args(argv)
    manifest = Path(__file__).resolve().parents[1] / "scenario" / f"village_{args.variant}.json"
    if args.map_only:
        layers = VillageSnapshot(args.run, manifest).capture_map(args)
        args.out.write_text(json.dumps(layers, separators=(",", ":")), encoding="utf-8")
        print(f"map layers at tick {layers['gameTime']}: {len(layers['palette'])} block types, "
              f"{len(layers['layers'])} layers")
        return 0
    snapshot = VillageSnapshot(args.run, manifest).capture(args)
    args.out.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    print(f"snapshot at tick {snapshot['gameTime']}: {len(snapshot['npcs'])} villagers, "
          f"{len(snapshot['agents'])} agents, "
          f"{sum(len(v) for v in snapshot['decisions'].values())} decisions, {len(snapshot['speech'])} utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
