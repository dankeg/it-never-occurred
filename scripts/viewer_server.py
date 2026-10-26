"""Serve recorded village runs and the offline viewer, without starting Minecraft.

Logs are streamed into small, per-source caches. Complete appended lines are read once; prompts stay in the
decision log and are fetched by byte offset only when requested. The HTTP server binds to loopback only.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"decisions": "orchestrator/decisions.jsonl", "trajectory": "orchestrator/trajectory.jsonl",
           "events": "world/simharness/events.jsonl", "welfare": "world/simharness/welfare.jsonl",
           "npcs": "world/simharness/npc_observations.jsonl"}
PRIMARY = ("decisions", "trajectory", "events")
EVENT_TYPES = {"utterance", "transfer", "purchase", "ferry_trip", "skill_end"}
POSITION = re.compile(r"You are (.+?) at \(\s*([-+\d.eE]+),\s*([-+\d.eE]+),\s*([-+\d.eE]+)\s*\)\.")


class RunData:
    """Parse one run on demand, preserving compact derived records across processes and live appends."""

    def __init__(self, run_dir: Path, *, scenario_dir: Path | None = None, cache_dir: Path | None = None,
                 variant: str | None = None):
        """Use `cache_dir` as this run's cache, defaulting to its own `.viewer_cache` directory."""
        self.path = Path(run_dir).resolve()
        self.variant = variant or ("accident" if "accident" in self.path.name.lower() else "sabotage")
        if self.variant not in {"accident", "sabotage"}:
            raise ValueError("variant must be accident or sabotage")
        self._scenario_dir = Path(scenario_dir) if scenario_dir is not None else ROOT / "scenario"
        self._cache_dir = Path(cache_dir) if cache_dir is not None else self.path / ".viewer_cache"
        self._sources: dict[str, dict] = {}
        self._ranges: dict[str, tuple[list, list]] = {}
        self._documents: dict[Path, tuple[list, dict]] = {}
        self._lock = threading.RLock()
        self._warnings: set[str] = set()

    @staticmethod
    def _empty() -> dict:
        """Return an empty source cursor and its compact, JSON-serializable derived data."""
        return {"version": 1, "signature": None, "offset": 0, "start": None, "end": None, "invalid": 0,
                "data": {"records": [], "agents": {}, "npcs": {}, "trips": {}, "changes": []}}

    @staticmethod
    def _signature(path: Path) -> list | None:
        """Identify a file by size, nanosecond mtime and filesystem identity, without reading or hashing it."""
        try:
            status = path.stat()
        except FileNotFoundError:
            return None
        return [status.st_size, status.st_mtime_ns, status.st_dev, status.st_ino]

    @staticmethod
    def _tick(value) -> int | None:
        """Accept finite numeric game times, leaving missing or malformed times unknown."""
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return int(value)
        return None

    @staticmethod
    def _point(value) -> bool:
        """Whether a value is a finite three-coordinate position."""
        return isinstance(value, list) and len(value) == 3 and all(
            isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool) and math.isfinite(coordinate)
            for coordinate in value)

    def _read_source(self, key: str) -> dict:
        """Load a source's cursor, then consume only new complete JSONL lines up to the observed file size."""
        source = self.path / SOURCES[key]
        signature = self._signature(source)
        state = self._sources.get(key)
        cache = self._cache_dir / f"{key}.json"
        if state is None:
            try:
                state = json.loads(cache.read_text(encoding="utf-8"))
                if (state.get("version") != 1 or not isinstance(state.get("data"), dict)
                        or not isinstance(state.get("offset"), int) or state["offset"] < 0
                        or not {"records", "agents", "npcs", "trips", "changes"} <= state["data"].keys()):
                    state = self._empty()
            except (OSError, ValueError, TypeError, AttributeError):
                state = self._empty()
        previous = state["signature"]
        if signature is None:
            state = self._empty()
        elif signature != previous:
            append = (isinstance(previous, list) and len(previous) == 4 and signature[2:] == previous[2:]
                      and signature[0] > previous[0] and state["offset"] <= signature[0])
            if not append:
                state = self._empty()
            with source.open("rb") as handle:
                handle.seek(state["offset"])
                while handle.tell() < signature[0]:
                    offset = handle.tell()
                    line = handle.readline(signature[0] - offset)
                    if not line.endswith(b"\n"):
                        break
                    state["offset"] = handle.tell()
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("record must be an object")
                    except (ValueError, UnicodeDecodeError):
                        state["invalid"] += 1
                        continue
                    self._consume(key, state, record, offset)
            self._sort(state["data"])
            state["signature"] = signature
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(state, handle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                temporary.replace(cache)
            except OSError as error:
                self._warnings.add(f"Cache unavailable: {error.strerror}; using memory instead.")
        self._sources[key] = state
        return state

    @staticmethod
    def _sort(data: dict) -> None:
        """Keep time-based queries correct even when records arrive out of game-time order."""
        data["records"].sort(key=lambda record: record["t"])
        data["changes"].sort(key=lambda change: change[0])
        for group in ("agents", "npcs", "trips"):
            for points in data[group].values():
                points.sort(key=lambda point: point[0])

    def _consume(self, key: str, state: dict, record: dict, offset: int) -> None:
        """Project one source record into the data needed for replay, never retaining full observations or prompts."""
        tick = self._tick(record.get("t", record.get("game_time", record.get("gameTime"))))
        if tick is not None:
            state["start"] = tick if state["start"] is None else min(state["start"], tick)
            state["end"] = tick if state["end"] is None else max(state["end"], tick)
        data = state["data"]
        if key == "decisions" and tick is not None and isinstance(record.get("agent"), str):
            decision = {name: value for name, value in record.items() if name != "user_prompt"}
            decision.update(t=tick, type="decision", _offset=offset)
            data["records"].append(decision)
            prompt = record.get("user_prompt") or ""
            position = POSITION.search(prompt)
            if position and position[1] == record["agent"]:
                coordinates = [float(position[index]) for index in (2, 3, 4)]
                if self._point(coordinates):
                    activity = re.search(r"Current activity: (\S+)", prompt)
                    last = re.search(r"Last activity: (\S+) (\S+)", prompt)
                    skill = activity[1] if activity else last[1] if last else None
                    status = "running" if activity else last[2].rstrip(".") if last else "idle"
                    data["agents"].setdefault(record["agent"], []).append(
                        [tick, coordinates[0], coordinates[2], skill, status])
        elif key == "trajectory" and tick is not None and self._point(record.get("pos")):
            agent = record.get("agent")
            if isinstance(agent, str):
                data["agents"].setdefault(agent, []).append(
                    [tick, record["pos"][0], record["pos"][2], record.get("skill"), record.get("state", "idle")])
        elif key == "npcs" and tick is not None and self._point(record.get("observerEye")):
            npc = str(record.get("npc") or record.get("observer") or "").removeprefix("npc:")
            if npc:
                points = data["npcs"].setdefault(npc, [])
                if not points or tick - points[-1][0] >= 100:
                    eye = record["observerEye"]
                    data["npcs"][npc].append([tick, eye[0], eye[2]])
        elif key == "welfare" and record.get("npc") is not None:
            depart, arrive = self._tick(record.get("departTick")), self._tick(record.get("arriveTick"))
            if depart is not None:
                npc = str(record["npc"]).removeprefix("npc:")
                data["trips"].setdefault(npc, []).append(
                    [depart, arrive, record.get("activity", "home"), record.get("to"), record.get("onTime"),
                     record.get("outcome"), record.get("from")])
                state["start"] = depart if state["start"] is None else min(state["start"], depart)
                if arrive is not None:
                    state["end"] = arrive if state["end"] is None else max(state["end"], arrive)
        elif key == "events" and tick is not None:
            if record.get("type") == "block_change" and self._point(record.get("pos")):
                position = record["pos"]
                data["changes"].append([tick, *position, record.get("to", "minecraft:air"),
                                        record.get("actor", "world")])
            elif record.get("type") in EVENT_TYPES:
                event = {name: value for name, value in record.items()
                         if name not in {"listenerEyes", "listenerStartEyes", "attentionLost"}}
                event["t"] = tick
                data["records"].append(event)

    def _refresh(self, keys=SOURCES) -> None:
        """Refresh selected logs under the caller's run lock."""
        for key in keys:
            self._read_source(key)

    def _document(self, path: Path) -> dict:
        """Read a small optional map, manifest or report only when its size or mtime changes."""
        signature = self._signature(path)
        if signature is None:
            return {}
        previous = self._documents.get(path)
        if previous is not None and previous[0] == signature:
            return previous[1]
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("expected a JSON object")
        except (OSError, ValueError) as error:
            self._warnings.add(f"Cannot read {path.name}: {error}")
            return {}
        self._documents[path] = (signature, document)
        return document

    def _edge_bounds(self, key: str) -> list:
        """Read the first and last complete records of chronological NPC/welfare logs for cheap run discovery."""
        path = self.path / SOURCES[key]
        signature = self._signature(path)
        if signature is None:
            return []
        parsed = self._sources.get(key)
        if parsed is not None and parsed["signature"] == signature:
            return [value for value in (parsed["start"], parsed["end"]) if value is not None]
        previous = self._ranges.get(key)
        if previous is not None and previous[0] == signature:
            return previous[1]

        def times(line: bytes) -> list:
            """Extract a record's available time bounds without retaining its observation body."""
            try:
                record = json.loads(line)
                return [tick for field in ("t", "game_time", "gameTime", "departTick", "arriveTick")
                        if (tick := self._tick(record.get(field))) is not None]
            except (ValueError, AttributeError, UnicodeDecodeError):
                return []

        bounds = []
        with path.open("rb") as handle:
            while handle.tell() < signature[0]:
                line = handle.readline(signature[0] - handle.tell())
                if not line.endswith(b"\n"):
                    break
                if values := times(line):
                    bounds.extend(values)
                    break
            window = 8192
            while signature[0]:
                begin = max(0, signature[0] - window)
                handle.seek(begin)
                lines = handle.read(signature[0] - begin).split(b"\n")[:-1]
                if begin:
                    lines = lines[1:]
                values = next((values for line in reversed(lines) if (values := times(line))), [])
                if values or not begin:
                    bounds.extend(values)
                    break
                window *= 2
        self._ranges[key] = (signature, bounds)
        return bounds

    def _summary(self) -> dict:
        """Describe time bounds and freshness without streaming the large NPC log just for the run picker."""
        bounds = [self._sources[key] for key in PRIMARY
                  if key in self._sources and self._sources[key]["start"] is not None]
        ticks = [tick for state in bounds for tick in (state["start"], state["end"]) if tick is not None]
        ticks.extend(self._edge_bounds("npcs"))
        ticks.extend(self._edge_bounds("welfare"))
        signatures = [signature for relative in SOURCES.values()
                      if (signature := self._signature(self.path / relative)) is not None]
        modified = max((signature[1] / 1_000_000_000 for signature in signatures), default=0)
        return {"name": self.path.name, "variant": self.variant,
                "start": min(ticks, default=0), "end": max(ticks, default=0),
                "modified": modified, "live": time.time() - modified < 60}

    def summary(self) -> dict:
        """List this run using its primary logs, without parsing NPC observations just for the picker."""
        with self._lock:
            self._refresh(PRIMARY)
            return self._summary()

    def map(self) -> dict:
        """Return the selected village's base layers, or raise FileNotFoundError if no capture exists."""
        with self._lock:
            path = self.path.parent / f"village_{self.variant}" / "viewer_map.json"
            document = self._document(path)
            if not {"origin", "size", "palette", "layers"} <= document.keys():
                raise FileNotFoundError(f"No base map: {path.name} for village_{self.variant}")
            return document

    def _changes(self) -> list:
        """Filter changes to columns and heights present in the base map, retaining actors internally."""
        try:
            base = self.map()
        except FileNotFoundError:
            return []
        origin, size = base["origin"], base["size"]
        heights = {layer["y"] for layer in base["layers"]}
        return [change for change in self._sources["events"]["data"]["changes"]
                if change[2] in heights and origin <= change[1] < origin + size
                and origin <= change[3] < origin + size]

    def changes(self) -> list:
        """Return sorted `[t, x, y, z, block]` changes within the captured map layers."""
        with self._lock:
            self._refresh(("events",))
            return [change[:5] for change in self._changes()]

    def tracks(self) -> dict:
        """Return agent samples, 100-tick NPC samples, and trips with optional welfare status columns."""
        with self._lock:
            self._refresh(("decisions", "trajectory", "npcs", "welfare"))
            agents = dict(self._sources["decisions"]["data"]["agents"])
            agents.update(self._sources["trajectory"]["data"]["agents"])
            return {"agents": agents, "npcs": self._sources["npcs"]["data"]["npcs"],
                    "trips": self._sources["welfare"]["data"]["trips"]}

    def events(self, start: int | None = None, end: int | None = None) -> list:
        """Return decisions and retained world events in an inclusive window, never including prompts."""
        with self._lock:
            self._refresh(("decisions", "events"))
            result = []
            for key in ("decisions", "events"):
                records = self._sources[key]["data"]["records"]
                lower = 0 if start is None else bisect_left(records, start, key=lambda record: record["t"])
                upper = len(records) if end is None else bisect_right(records, end, key=lambda record: record["t"])
                result.extend({name: value for name, value in record.items() if name != "_offset"}
                              for record in records[lower:upper])
            return sorted(result, key=lambda record: record["t"])

    def prompt(self, agent: str, tick: int) -> dict | None:
        """Seek to the latest decision at or before `tick`, reading its prompt only on demand, with the agent's
        earlier turns in the same prompt epoch (the conversation the model saw before this message): each turn's
        message, the action chosen and its immediate result. An epoch starts at a full-memory prompt."""
        with self._lock:
            self._refresh(("decisions",))
            records = self._sources["decisions"]["data"]["records"]
            upper = bisect_right(records, tick, key=lambda record: record["t"])
            mine = [decision for decision in records[:upper] if decision["agent"] == agent]
            if not mine:
                return None
            with (self.path / SOURCES["decisions"]).open("rb") as handle:
                def read(decision):
                    handle.seek(decision["_offset"])
                    return json.loads(handle.readline())
                record = read(mine[-1])
                history = []
                if not record.get("user_prompt", "").startswith("Your notes:"):
                    for decision in reversed(mine[:-1]):
                        earlier = read(decision)
                        history.append({"t": earlier["game_time"], "message": earlier.get("user_prompt", ""),
                                        "tool": earlier.get("tool"), "args": earlier.get("args"),
                                        "result": earlier.get("outcome", "")})
                        if earlier.get("user_prompt", "").startswith("Your notes:"):
                            break
            return {"prompt": record.get("user_prompt", ""), "history": list(reversed(history)),
                    "decision": {name: value for name, value in mine[-1].items() if name != "_offset"}}

    def meta(self) -> dict:
        """Return scenario people and goals, time bounds, optional reports, and agent-change highlights."""
        with self._lock:
            self._refresh()
            manifest = self._document(self._scenario_dir / f"village_{self.variant}.json")
            agents = [dict(role) for role in manifest.get("roles", [])]
            known = {role["name"] for role in agents}
            names = set(self._sources["decisions"]["data"]["agents"]) | set(
                self._sources["trajectory"]["data"]["agents"]) | {
                    record["agent"] for record in self._sources["decisions"]["data"]["records"]}
            agents.extend({"name": name, "public_goal": "", "private_goal": ""} for name in sorted(names - known))
            report = self._document(self.path / "m5_report.json")
            selected = {key: report[key] for key in ("first_true", "welfare", "outcomes", "stats", "end", "stopped")
                        if key in report}
            stats = self._document(self.path / "orchestrator/stats.json")
            if stats:
                selected.setdefault("stats", stats)
            for path in self.path.parent.glob("study_*_report.json"):
                prefix = path.stem.removesuffix("_report")
                if self.path.name == prefix or self.path.name.startswith(prefix + "_"):
                    selected["study_report"] = self._document(path)
                    break
            warnings = set(self._warnings)
            if not manifest:
                warnings.add(f"Missing scenario/village_{self.variant}.json; showing recorded people only.")
            if not self._signature(self.path / SOURCES["trajectory"]):
                warnings.add("Agent positions and activity use decision prompts; "
                             "movement between decisions is unknown.")
            if not self._signature(self.path / SOURCES["npcs"]):
                warnings.add("No NPC observations; villager positions are unavailable.")
            for key, state in self._sources.items():
                if state["invalid"]:
                    warnings.add(f"Skipped {state['invalid']} malformed complete lines in {SOURCES[key]}.")
            people = [dict(npc, id=str(npc["id"]).removeprefix("npc:")) for npc in manifest.get("npcs", [])]
            npc_ids = {npc["id"] for npc in people}
            recorded_ids = set(self._sources["npcs"]["data"]["npcs"]) | set(
                self._sources["welfare"]["data"]["trips"])
            people.extend({"id": npc, "name": npc, "role": "unknown"} for npc in sorted(recorded_ids - npc_ids))
            return {**self._summary(), "agents": agents, "villagers": people, "places": manifest.get("places", []),
                    "report": selected, "warnings": sorted(warnings),
                    "agent_changes": [[*change[:4], change[5]] for change in self._changes()
                                      if change[5] in known | names]}


class RunIndex:
    """Discover readable run directories and reuse their in-memory parsers between HTTP requests."""

    def __init__(self, runs_dir: Path, *, cache_dir: Path | None = None, scenario_dir: Path | None = None):
        """Use an optional cache root with a separate child directory for each run."""
        self.path = Path(runs_dir).resolve()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._scenario_dir = scenario_dir
        self._runs: dict[tuple[str, str | None], RunData] = {}
        self._lock = threading.Lock()

    def run(self, name: str, variant: str | None = None) -> RunData:
        """Resolve one immediate run child, rejecting traversal and directory symlinks outside the runs root."""
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise ValueError("invalid run name")
        path = (self.path / name).resolve()
        if path.parent != self.path or not path.is_dir() or not any(
                (path / SOURCES[key]).is_file() for key in ("decisions", "events")):
            raise FileNotFoundError(f"Run not found: {name}")
        with self._lock:
            key = (name, variant)
            if key not in self._runs:
                cache = self._cache_dir / name if self._cache_dir is not None else None
                run = RunData(path, cache_dir=cache, scenario_dir=self._scenario_dir, variant=variant)
                existing = next((item for (run_name, _), item in self._runs.items() if run_name == name), None)
                if existing is not None:
                    run._sources = existing._sources
                    run._ranges = existing._ranges
                    run._documents = existing._documents
                    run._lock = existing._lock
                self._runs[key] = run
            return self._runs[key]

    def runs(self) -> list:
        """List runs newest-modified first, marking logs modified in the last 60 seconds as live."""
        if not self.path.is_dir():
            return []
        result = []
        for path in self.path.iterdir():
            if path.is_dir():
                try:
                    result.append(self.run(path.name).summary())
                except (FileNotFoundError, PermissionError):
                    continue
        return sorted(result, key=lambda run: (run["modified"], run["name"]), reverse=True)


class ViewerServer(ThreadingHTTPServer):
    """A loopback HTTP server carrying the shared run index and an explicit static asset allowlist."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], index: RunIndex):
        """Attach the data index before accepting viewer requests."""
        self.index = index
        super().__init__(address, ViewerHandler)


class ViewerHandler(BaseHTTPRequestHandler):
    """Serve JSON APIs and only the three viewer assets, never arbitrary run or repository files."""

    def do_GET(self) -> None:
        """Dispatch a GET request, returning JSON errors for invalid or missing API resources."""
        try:
            url = urlsplit(self.path)
            if url.path.startswith("/api/"):
                self._json(self._api(unquote(url.path), parse_qs(url.query)))
                return
            assets = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                      "/viewer.js": ("viewer.js", "text/javascript"), "/viewer.css": ("viewer.css", "text/css")}
            if url.path not in assets:
                raise FileNotFoundError("Resource not found")
            filename, mime = assets[url.path]
            self._send((ROOT / "viewer" / filename).read_bytes(), mime + "; charset=utf-8")
        except (ValueError, TypeError) as error:
            self._json({"error": str(error)}, 400)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)
        except OSError:
            self._json({"error": "Cannot read the requested run data"}, 500)

    def _api(self, path: str, query: dict) -> dict | list | None:
        """Resolve an API resource and validate variant, time-window and prompt parameters."""
        if path == "/api/runs":
            return self.server.index.runs()
        parts = path.split("/")
        if len(parts) != 5 or parts[1:3] != ["api", "runs"]:
            raise FileNotFoundError("API endpoint not found")
        name, endpoint = parts[3:]
        variant = query.get("variant", [None])[0]
        if variant is not None and variant not in {"accident", "sabotage"}:
            raise ValueError("variant must be accident or sabotage")
        run = self.server.index.run(name, variant)
        if endpoint in {"meta", "map", "changes", "tracks"}:
            return getattr(run, endpoint)()
        if endpoint == "events":
            start = int(query["from"][0]) if "from" in query else None
            end = int(query["to"][0]) if "to" in query else None
            if start is not None and end is not None and start > end:
                raise ValueError("from must not exceed to")
            return run.events(start, end)
        if endpoint == "prompt":
            if not query.get("agent") or "t" not in query:
                raise ValueError("prompt requires agent and t")
            return run.prompt(query["agent"][0], int(query["t"][0]))
        raise FileNotFoundError("API endpoint not found")

    def _json(self, value, status: int = 200) -> None:
        """Encode a JSON response without caching live results in the browser."""
        self._send(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(),
                   "application/json; charset=utf-8", status)

    def _send(self, body: bytes, mime: str, status: int = 200) -> None:
        """Write a length-delimited response with explicit content type and safe local caching headers."""
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(argv: list[str] | None = None) -> int:
    """Start the local viewer with configurable run and cache roots, without accessing a running game."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path,
                        help="cache root; each run gets a child directory (default: <run>/.viewer_cache)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        parser.error(f"runs directory does not exist: {args.runs_dir}")
    server = ViewerServer(("127.0.0.1", args.port), RunIndex(args.runs_dir, cache_dir=args.cache_dir))
    print(f"Viewer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
