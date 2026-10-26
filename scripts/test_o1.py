"""O1: physical observation, provenance, documents, speech, and container ACLs."""

from __future__ import annotations

from pathlib import Path

if __package__:
    from . import audit_o1, harness
else:
    import audit_o1
    import harness


CHEST = [0, -60, 6]
SIGN = [0, -60, 7]
VISIBLE = [2, -60, 1]
HIDDEN = [0, -60, 6]


def check(condition: bool, message: str) -> None:
    """Raise a harness failure when an O1 assertion does not hold."""
    if not condition:
        raise harness.HarnessError(message)


def command(client: harness.Client, value: str) -> dict:
    """Send an unchanged server command and return its response."""
    return client.request("command", {"command": value})


def teleport(client: harness.Client, name: str, position: tuple) -> None:
    """Teleport a named bot to the fixture position with zero rotation."""
    command(client, f"tp {name} {' '.join(str(value) for value in position)} 0 0")


def wall(client: harness.Client, block: str) -> None:
    """Fill the occluding wall with the requested Minecraft block."""
    command(client, f"fill -3 -61 3 3 -57 3 minecraft:{block}")


def observe(client: harness.Client, name: str = "alice") -> dict:
    """Request the named bot's agent-facing observation."""
    return client.request("observe", {"name": name})


def at(items: list[dict], position: list[int]) -> list[dict]:
    """Select observation items at exactly the given block position."""
    return [item for item in items if item.get("pos") == position]


def events(client: harness.Client, directory: Path) -> list[dict]:
    """Flush the snapshot before reading the run's provenance events."""
    client.request("save_snapshot")
    return audit_o1.read_jsonl(directory / "world/simharness/events.jsonl")


def setup(client: harness.Client) -> None:
    """Build the O1 fixtures and spawn all bots before enabling recording."""
    client.request("config_set", {"view_radius": 24.0, "terrain_radius": 12,
                                  "terrain_step": 2, "reach": 4.5, "talk_radius": 8.0})
    command(client, "fill -8 -61 -8 24 -61 24 minecraft:stone")
    command(client, "setblock 0 -60 6 minecraft:chest")
    command(client, "setblock 0 -60 7 minecraft:oak_sign")
    wall(client, "stone")
    for name, position in (("alice", (0.5, -60, 0.5)), ("bob", (0.5, -60, 9.5)),
                           ("carol", (20.5, -60, 0.5))):
        client.request("spawn_bot", {"name": name, "x": position[0], "y": position[1],
                                      "z": position[2], "yaw": 0.0, "pitch": 0.0})
    client.request("recording", {"enabled": True})


def hidden_chest(client: harness.Client, _directory: Path) -> None:
    """Check that a wall hides the chest and removing it exposes the chest."""
    check(not at(observe(client)["blocks"], CHEST), "chest leaked through the wall")
    wall(client, "air")
    client.request("step", {"n": 1})
    check(bool(at(observe(client)["blocks"], CHEST)), "unobstructed chest is missing")


def document_reach(client: harness.Client, _directory: Path) -> None:
    """Check sign text against document reach while its block remains visible."""
    command(client, "setblock 0 -60 6 minecraft:air")
    client.request("write_sign", {"name": "bob", "pos": SIGN, "front": ["SECRET", "", "", ""]})
    far = observe(client)
    sign_blocks = at(far["blocks"], SIGN)
    check(bool(sign_blocks), "distant sign is not visible as a notable block")
    check(not at(far["documents"], SIGN), "distant sign text leaked beyond reach")
    check(all(not {"text", "pages", "front", "back"}.intersection(block) for block in sign_blocks),
          "notable block entry contains document text")
    teleport(client, "alice", (0.5, -60, 5.0))
    nearby = at(observe(client)["documents"], SIGN)
    check(len(nearby) == 1 and nearby[0].get("text", {}).get("front") == ["SECRET", "", "", ""],
          "nearby sign does not expose SECRET")
    check(nearby[0].get("text", {}).get("back") == ["", "", "", ""], "sign back text is missing")
    teleport(client, "alice", (0.5, -60, 0.5))


def speech_radius(client: harness.Client, _directory: Path) -> None:
    """Check timed talk at five/twelve blocks and exactly-once consumption."""
    teleport(client, "bob", (5.5, -60, 0.5))
    teleport(client, "carol", (12.5, -60, 0.5))
    dry = client.request("skill_start", {"name": "alice", "skill": "speak",
                                         "args": {"text": "hello", "volume": "talk"}, "dry_run": True})
    duration = dry.get("estimate", {}).get("ticks")
    check(dry.get("feasible") is True and type(duration) is int and duration > 0,
          "talk dry-run must estimate a positive duration")
    check(client.say("alice", "hello").get("accepted") is True, "talk skill was rejected")
    check(not observe(client, "bob")["heard"], "bob heard speech before it completed")
    if duration > 1:
        client.request("step", {"n": duration - 1})
        check(not observe(client, "bob")["heard"], "bob heard speech before its estimated end")
    client.request("step", {"n": 1})
    heard = observe(client, "bob")["heard"]
    check(len(heard) == 1 and heard[0].get("text") == "hello" and
          heard[0].get("speaker") == "alice" and heard[0].get("prov", "").startswith("utt:"),
          "bob did not hear the expected utterance")
    check(not observe(client, "carol")["heard"], "carol heard speech from outside the radius")
    check(not observe(client, "bob")["heard"], "bob heard the same utterance twice")
    check(not observe(client)["heard"], "alice heard herself")


def entity_occlusion(client: harness.Client, _directory: Path) -> None:
    """Check that entity visibility follows physical wall occlusion."""
    wall(client, "stone")
    teleport(client, "bob", (0.5, -60, 6.5))
    bob_uuid = client.request("bot_state", {"name": "bob"}).get("uuid")

    def is_bob(entity):
        """Identify Bob in the entity projection used by this visibility case."""
        return entity.get("name") == "bob" or (bob_uuid is not None and entity.get("uuid") == bob_uuid)

    check(not any(is_bob(entity) for entity in observe(client)["entities"]), "bob leaked through the wall")
    wall(client, "air")
    check(any(is_bob(entity) for entity in observe(client)["entities"]), "unobstructed bob is missing")


def visible_changes(client: harness.Client, directory: Path) -> None:
    """Check visible changes and resolve their actor-free provenance once."""
    wall(client, "stone")
    command(client, "setblock 2 -60 1 minecraft:stone")
    command(client, "setblock 0 -60 6 minecraft:stone")
    observe(client)
    client.request("step", {"n": 1})
    for position, nearby in ((VISIBLE, (2.5, -60, 0.5)), (HIDDEN, (0.5, -60, 5.5))):
        teleport(client, "bob", nearby)
        client.request("bot_break", {"name": "bob", **dict(zip(("x", "y", "z"), position))})
    changes = observe(client)["changes"]
    check(len(changes) == 1 and changes[0].get("pos") == VISIBLE, "changes did not report exactly the visible break")
    check("actor" not in changes[0], "agent-facing change leaks the actor")
    provenance = changes[0].get("prov", "")
    check(provenance.startswith("change:"), "visible change lacks change provenance")
    matching = [event for event in events(client, directory)
                if event.get("type") == "block_change" and f"change:{event['id']}" == provenance]
    check(len(matching) == 1 and matching[0].get("actor") == "bob" and matching[0].get("pos") == VISIBLE
          and matching[0].get("from") == "minecraft:stone" and matching[0].get("to") == "minecraft:air",
          "visible break provenance does not resolve to bob's block_change")
    check(not observe(client)["changes"], "changes were delivered twice")


def books(client: harness.Client, directory: Path) -> None:
    """Check book read/write content and its document-write provenance."""
    client.request("give", {"name": "alice", "item": "minecraft:writable_book", "count": 1})
    written = client.request("write_book", {"name": "alice", "slot": 0, "title": "O1 notes",
                                           "pages": ["SECRET", "page two"]})
    read = client.request("read_item_doc", {"name": "alice", "slot": 0})
    check(read.get("prov") == written.get("prov") and isinstance(read.get("prov"), str)
          and read["prov"].startswith("doc:"), "book read/write provenance mismatch")
    check(read.get("title") == "O1 notes" and read.get("author") == "alice"
          and read.get("pages") == ["SECRET", "page two"], "book content differs from write_book")
    matching = [event for event in events(client, directory)
                if event.get("type") == "doc_write" and event.get("prov") == read["prov"]]
    check(len(matching) == 1 and matching[0].get("author") == "alice", "book doc_write event is missing")


def container_acl(client: harness.Client, _directory: Path) -> None:
    """Check denied, permitted, and master-key access to one chest version."""
    command(client, "setblock 0 -60 6 minecraft:chest")
    teleport(client, "alice", (0.5, -60, 5.0))
    teleport(client, "bob", (1.5, -60, 6.5))
    client.request("acl_set", {"pos": CHEST, "allowed": ["bob"]})
    try:
        client.request("open_container", {"name": "alice", "pos": CHEST})
    except harness.CommandError as error:
        check(any(word in str(error).lower() for word in ("acl", "permitted", "denied", "allow", "access")),
              f"container failed for a reason other than ACL: {error}")
    else:
        raise harness.HarnessError("ACL allowed alice to open bob's chest")
    allowed = client.request("open_container", {"name": "bob", "pos": CHEST})
    check(isinstance(allowed.get("items"), list) and allowed.get("prov", "").startswith("container:0,-60,6@"),
          "bob could not open the ACL-permitted chest")
    client.request("bot_flag", {"name": "alice", "master_key": True})
    master = client.request("open_container", {"name": "alice", "pos": CHEST})
    check(master.get("prov") == allowed["prov"] and master.get("items") == allowed["items"],
          "master key did not open the same unchanged container version")


TESTS = (("T1 hidden chest", hidden_chest), ("T2 document reach", document_reach),
         ("T3 speech radius", speech_radius), ("T4 entity occlusion", entity_occlusion),
         ("T5 changes", visible_changes), ("T6 books", books), ("T7 ACL", container_acl))


class O1Experiment:
    """Run the ordered observation cases and offline audit without extra ticks."""

    def __init__(self, args):
        """Keep parsed CLI options for the observation experiment."""
        self._args = args

    def setup(self) -> None:
        """Reject an existing run and clone the untouched base."""
        self._directory = self._args.runs_dir / "o1"
        if self._directory.exists() or self._directory.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {self._directory}")
        harness.clone_apfs(self._args.runs_dir / "base", self._directory)
        self._passed = True
        self._completed = set()

    def run(self) -> None:
        """Run cases in order, reporting each failure without skipping later cases."""
        directory = self._directory
        try:
            with harness.Server(directory, **harness.server_options(self._args)) as server:
                harness.enable_lockstep(server.client, recording=False)
                setup(server.client)
                for label, test in TESTS:
                    try:
                        test(server.client, directory)
                    except (harness.HarnessError, OSError, ValueError, KeyError, TypeError) as error:
                        print(f"FAIL {label}: {error}")
                        self._passed = False
                    else:
                        print(f"PASS {label}")
                    self._completed.add(label)
                server.client.request("save_snapshot")
        except (harness.HarnessError, OSError, ValueError) as error:
            print(f"FAIL O1 setup/shutdown: {error}")
            self._passed = False

    def check(self) -> None:
        """Report unrun cases and audit the persisted logs after server shutdown."""
        for label, _test in TESTS:
            if label not in self._completed:
                print(f"FAIL {label}: setup or server lifecycle failed")
        if audit_o1.main([str(self._directory)]) != 0:
            self._passed = False

    def report(self) -> int:
        """Print the final gate verdict and return its original exit status."""
        print(f"{'PASS' if self._passed else 'FAIL'} O1")
        return 0 if self._passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the O1 experiment phases."""
    harness.stop_on_sigterm()  # C8: SIGTERM stops this run's servers cleanly
    experiment = O1Experiment(harness.experiment_parser(__doc__).parse_args(argv))
    experiment.setup()
    experiment.run()
    experiment.check()
    return experiment.report()


if __name__ == "__main__":
    raise SystemExit(main())
