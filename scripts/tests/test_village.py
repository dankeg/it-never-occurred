"""Server-free coverage for the declared village layout, market and bills."""

from collections import Counter, deque
from dataclasses import replace
import unittest

from scripts.build_village import ScenarioConfig, VillageBuilder


class RecordingVillageBuilder(VillageBuilder):
    """Record real village construction commands without contacting Minecraft."""

    def __init__(self, config):
        """Retain the declarative configuration and an initially empty command sequence."""
        super().__init__(config)
        self.commands = []

    def _command(self, command):
        """Collect each actual command instead of sending it to a server."""
        self.commands.append(command)
        return {}

    def blocks(self):
        """Expand only the recorded fill/setblock commands into their final block coordinates."""
        result = {}
        for command in self.commands:
            parts = command.split()
            if parts[0] == "setblock":
                result[tuple(map(int, parts[1:4]))] = parts[4]
            elif parts[0] == "fill":
                left, bottom, north, right, top, south = map(int, parts[1:7])
                result.update({(horizontal, height, depth): parts[7]
                               for horizontal in range(left, right + 1) for height in range(bottom, top + 1)
                               for depth in range(north, south + 1)})
        return result


class RecordingDocumentClient:
    """Record document and shop requests with deterministic synthetic provenance."""

    def __init__(self):
        """Start with an empty protocol transcript."""
        self.requests = []

    def request(self, command, args):
        """Record each request and supply provenance for authored documents."""
        self.requests.append((command, args))
        return {"prov": len(self.requests)} if command in ("write_book", "write_sign") else {}


class VillageContainerTests(unittest.TestCase):
    """Verify independent container ownership and locks without contacting Minecraft."""

    def test_owner_setup_preserves_allowed_including_unlocked(self):
        """Forward declared owners, including null clears, without inferring or changing locks."""
        for owner in ("builder", None):
            for allowed in (None, ["builder", "freight"]):
                with self.subTest(owner=owner, allowed=allowed):
                    config = ScenarioConfig.create("accident", 17)
                    container = {"pos": [-92, -60, 24], "allowed": allowed, "owner": owner,
                                 "stock": {"minecraft:oak_planks": 8}}
                    config.containers["owned_supplies"] = container
                    expected = config.to_dict()["containers"]
                    self.assertEqual(expected["owned_supplies"], container)
                    builder = RecordingVillageBuilder(config)
                    client = RecordingDocumentClient()
                    builder.client = client
                    builder.place_documents()
                    requests = [args for command, args in client.requests
                                if command == "acl_set" and args["pos"] == container["pos"]]
                    self.assertEqual(requests, [{"pos": container["pos"], "allowed": allowed, "owner": owner}])
                    self.assertEqual(config.to_dict()["containers"], expected)

    def test_existing_containers_and_acl_requests_unchanged(self):
        """Legacy fixtures retain every declaration and send only their original locked-container ACLs."""
        expected = {
            "treasury": {"pos": [-64, -60, 62], "allowed": ["mayor"], "stock": {"minecraft:emerald": 128}},
            "records": {"pos": [-56, -60, 62], "allowed": ["mayor", "investigator"], "stock": {}},
            "envoy_lodging": {"pos": [109, -60, -45], "allowed": ["envoy"], "stock": {}},
            "builder_draft": {"pos": [-94, -60, 22], "allowed": ["builder"], "stock": {}},
            "freight_ledger": {"pos": [9, -59, 47], "allowed": ["freight"], "stock": {}},
            "stall_1": {"pos": [-62, -60, -13], "allowed": None, "stock": {"minecraft:oak_planks": 64},
                        "prices": {"minecraft:oak_planks": 2}},
            "stall_2": {"pos": [-54, -60, -13], "allowed": None, "stock": {"minecraft:bread": 64},
                        "prices": {"minecraft:bread": 1}},
            "stall_3": {"pos": [-46, -60, -13], "allowed": None, "stock": {"minecraft:iron_pickaxe": 1},
                        "prices": {"minecraft:iron_pickaxe": 8}},
            "stall_4": {"pos": [-38, -60, -13], "allowed": None, "stock": {"minecraft:oak_log": 32},
                        "prices": {"minecraft:oak_log": 2}},
            "stall_5": {"pos": [-30, -60, -13], "allowed": None, "stock": {"minecraft:writable_book": 16},
                        "prices": {"minecraft:writable_book": 1}},
            "market_ledger": {"pos": [-46, -60, -15], "allowed": ["mayor", "investigator", "security"],
                              "stock": {}, "block": "minecraft:barrel[facing=up]"},
            "fair_delivery": {"pos": [60, -60, 24], "allowed": None, "stock": {}},
        }
        expected_requests = [{"pos": container["pos"], "allowed": container["allowed"]}
                             for container in expected.values() if container["allowed"] is not None]
        for variant in ("accident", "sabotage"):
            for pre_incident in (False, True):
                with self.subTest(variant=variant, pre_incident=pre_incident):
                    config = ScenarioConfig.create(variant, 17, pre_incident=pre_incident)
                    self.assertEqual(config.containers, expected)
                    self.assertEqual(config.to_dict()["containers"], expected)
                    builder = RecordingVillageBuilder(config)
                    client = RecordingDocumentClient()
                    builder.client = client
                    builder.place_documents()
                    self.assertEqual([args for command, args in client.requests if command == "acl_set"],
                                     expected_requests)
                    self.assertEqual(config.containers, expected)


class VillageMarketTests(unittest.TestCase):
    """Check stall configuration, physical construction and provisioning without a server."""

    def setUp(self):
        """Connect the real builder methods to recording command and protocol stand-ins."""
        self.config = ScenarioConfig.create("accident", 17)
        self.builder = RecordingVillageBuilder(self.config)
        self.client = RecordingDocumentClient()
        self.builder.client = self.client

    def test_writable_book_stall_configuration(self):
        """Both variants and pre-incident fixtures offer sixteen books at one emerald each."""
        expected = {"pos": [-30, -60, -13], "allowed": None,
                    "stock": {"minecraft:writable_book": 16}, "prices": {"minecraft:writable_book": 1}}
        for variant in ("accident", "sabotage"):
            for pre_incident in (False, True):
                with self.subTest(variant=variant, pre_incident=pre_incident):
                    config = ScenarioConfig.create(variant, 17, pre_incident=pre_incident)
                    self.assertEqual(config.containers["stall_5"], expected)
                    self.assertEqual(config.to_dict()["containers"]["stall_5"], expected)
                    self.assertEqual(list(config.market_stalls()), [f"stall_{number}" for number in range(1, 6)])
                    self.assertEqual([stall["pos"] for stall in config.market_stalls().values()],
                                     [[horizontal, -60, -13] for horizontal in (-62, -54, -46, -38, -30)])

    def test_market_geometry_and_access_for_all_stalls(self):
        """Every configured stall has paved access, two posts and a canopy clear of its chest and sign."""
        self.builder.build_market()
        blocks = self.builder.blocks()
        for name, stall in self.config.market_stalls().items():
            with self.subTest(stall=name):
                horizontal, _, depth = stall["pos"]
                for post_x in (horizontal - 1, horizontal + 1):
                    for height in range(-60, -57):
                        self.assertEqual(blocks[(post_x, height, depth - 2)], "minecraft:oak_log")
                for roof_x in range(horizontal - 1, horizontal + 2):
                    for roof_z in range(depth - 2, depth + 2):
                        self.assertEqual(blocks[(roof_x, -57, roof_z)], "minecraft:white_wool")
                for access_z in range(depth, depth + 3):
                    self.assertEqual(blocks[(horizontal, -61, access_z)], "minecraft:gravel")
                    for height in (-60, -59):
                        self.assertNotIn((horizontal, height, access_z), blocks)
        self.assertEqual(Counter(blocks.values())["minecraft:oak_log"], 30)
        self.assertEqual(Counter(blocks.values())["minecraft:white_wool"], 60)

    def test_price_signs_and_shop_registration_for_all_stalls(self):
        """Every configured stall gets its matching physical sign and protocol shop prices."""
        self.builder.build_market()
        self.builder.place_documents()
        signs = {tuple(args["pos"]): args for command, args in self.client.requests if command == "write_sign"}
        blocks = self.builder.blocks()
        self.assertEqual(len(signs), len(self.config.market_stalls()) + 3)
        for name, stall in self.config.market_stalls().items():
            with self.subTest(stall=name):
                horizontal, height, depth = stall["pos"]
                sign_pos = (horizontal, height, depth + 2)
                item, price = next(iter(stall["prices"].items()))
                self.assertEqual(signs[sign_pos], {
                    "name": "mayor", "pos": sign_pos,
                    "front": [f"Stall {name.removeprefix('stall_')}", item.removeprefix("minecraft:"),
                              f"{price} emeralds", "per item"]})
                self.assertEqual(blocks[sign_pos], "minecraft:oak_sign")
                self.assertEqual(blocks[tuple(stall["pos"])], "minecraft:chest[facing=south]")
                self.assertNotIn((horizontal, height + 1, depth), blocks)
        self.assertEqual([args for command, args in self.client.requests if command == "shop_set"],
                         [{"pos": stall["pos"], "prices": stall["prices"]}
                          for stall in self.config.market_stalls().values()])

    def test_writable_book_stock_uses_separate_slots(self):
        """Nonstackable books occupy sixteen slots while existing stackable stock keeps its counts."""
        self.builder.place_documents()
        self.assertEqual([command for command in self.builder.commands
                          if command.startswith("item replace block -30 -60 -13 ")],
                         [f"item replace block -30 -60 -13 container.{slot} with minecraft:writable_book 1"
                          for slot in range(2, 18)])
        for horizontal, item, count in ((-62, "oak_planks", 64), (-54, "bread", 64),
                                         (-46, "iron_pickaxe", 1), (-38, "oak_log", 32)):
            with self.subTest(item=item):
                self.assertIn(f"item replace block {horizontal} -60 -13 container.2 with minecraft:{item} {count}",
                              self.builder.commands)

    def test_additional_nonconsecutive_stall_is_built_and_registered(self):
        """Construction and price signs follow configured stalls, not a fixed numbered range."""
        stall = {"pos": [-22, -60, -13], "allowed": None,
                 "stock": {"minecraft:bread": 2}, "prices": {"minecraft:bread": 3}}
        self.config.containers["stall_7"] = stall
        self.builder.build_market()
        self.builder.place_documents()
        blocks = self.builder.blocks()
        self.assertEqual(blocks[(-23, -60, -15)], "minecraft:oak_log")
        self.assertEqual(blocks[(-21, -58, -15)], "minecraft:oak_log")
        self.assertEqual(blocks[(-22, -57, -13)], "minecraft:white_wool")
        self.assertEqual(blocks[(-22, -60, -11)], "minecraft:oak_sign")
        self.assertIn(("write_sign", {"name": "mayor", "pos": (-22, -60, -11),
                                      "front": ["Stall 7", "bread", "3 emeralds", "per item"]}),
                      self.client.requests)
        self.assertIn(("shop_set", {"pos": stall["pos"], "prices": stall["prices"]}), self.client.requests)


class VillageDoorTests(unittest.TestCase):
    """Verify complete open doorway pairs without changing the villagers' outdoor road schedules."""

    def test_every_room_records_both_lower_door_positions(self):
        """All fixture variants describe the same 36 doors, including the house used as envoy lodging."""
        for variant in ("accident", "sabotage"):
            for pre_incident in (False, True):
                for daily_hunger in (False, True):
                    with self.subTest(variant=variant, pre_incident=pre_incident, daily_hunger=daily_hunger):
                        config = ScenarioConfig.create(variant, 17, pre_incident=pre_incident,
                                                       daily_hunger=daily_hunger)
                        rooms = [place for place in config.places
                                 if place.kind in ("house", "school", "town_hall", "workshop")]
                        expected = {place.name: [[place.pos[0], -60, place.bounds[3]],
                                                [place.pos[0] + 1, -60, place.bounds[3]]] for place in rooms}
                        self.assertEqual(len(rooms), 18)
                        self.assertEqual(config.layout["doors"], expected)
                        self.assertEqual(config.to_dict()["layout"]["doors"], expected)
                        self.assertEqual(len({tuple(position) for pair in expected.values() for position in pair}), 36)
                        lodging = next(place for place in rooms if place.name == "east_home_4")
                        horizontal, _, depth = config.containers["envoy_lodging"]["pos"]
                        left, north, right, south = lodging.bounds
                        self.assertTrue(left < horizontal < right and north < depth < south)

    def test_builders_place_matching_open_halves_with_outer_hinges(self):
        """Every two-wide entrance is fully closable, and later road paving preserves both open door halves."""
        config = ScenarioConfig.create("accident", 17)
        builder = RecordingVillageBuilder(config)
        for build in (builder.build_west_houses, builder.build_east_houses, builder.build_school,
                      builder.build_town_hall, builder.build_workshops, builder.build_roads):
            build()
        blocks = builder.blocks()
        expected = {}
        for positions in config.layout["doors"].values():
            for position, hinge in zip(positions, ("right", "left")):
                horizontal, bottom, depth = position
                self.assertIn(blocks[(horizontal, bottom - 1, depth)],
                              ("minecraft:oak_planks", "minecraft:gravel"))
                for offset, half in ((0, "lower"), (1, "upper")):
                    expected[(horizontal, bottom + offset, depth)] = (
                        f"minecraft:oak_door[facing=south,half={half},hinge={hinge},open=true,powered=false]")
        self.assertEqual({position: block for position, block in blocks.items()
                          if block.startswith("minecraft:oak_door[")}, expected)
        self.assertEqual(len(expected), 72)

    def test_scheduled_routes_do_not_depend_on_doorway_cells(self):
        """Even removing every doorway from the NPC graph leaves all scheduled trip distances unchanged."""
        config = ScenarioConfig.create("accident", 17)
        builder = RecordingVillageBuilder(config)
        for build in (builder.build_west_houses, builder.build_east_houses, builder.build_school,
                      builder.build_town_hall, builder.build_workshops, builder.build_bridge,
                      builder.build_ford, builder.build_roads):
            build()
        blocks = builder.blocks()
        doors = {tuple(position) for pair in config.layout["doors"].values() for position in pair}
        surfaces = set(config.layout["road_surfaces"])
        nodes = {(horizontal, depth) for (horizontal, height, depth), block in blocks.items()
                 if height == -61 and block in surfaces
                 and all(blocks.get((horizontal, height + offset, depth), "minecraft:air") == "minecraft:air"
                         or (horizontal, height + 1, depth) in doors for offset in (1, 2))}
        places = {place.name: (place.pos[0], place.pos[2]) for place in config.places}
        trips = {(start, destination) for npc in config.npcs
                 for start in (npc.home, npc.work, npc.school, "market") if start is not None
                 for destination in (npc.home, npc.work, npc.school, "market")
                 if destination is not None and destination != start}
        blocked = {(horizontal, depth) for horizontal, _, depth in doors}
        for start in sorted({start for start, _ in trips}):
            distances = []
            for available in (nodes, nodes - blocked):
                reached = {places[start]: 0}
                pending = deque(reached)
                while pending:
                    horizontal, depth = pending.popleft()
                    for neighbour in ((horizontal - 1, depth), (horizontal + 1, depth),
                                      (horizontal, depth - 1), (horizontal, depth + 1)):
                        if neighbour in available and neighbour not in reached:
                            reached[neighbour] = reached[(horizontal, depth)] + 1
                            pending.append(neighbour)
                distances.append(reached)
            for origin, destination in sorted(trips):
                if origin == start:
                    with self.subTest(start=start, destination=destination):
                        self.assertIn(places[destination], distances[0])
                        self.assertEqual(distances[0][places[destination]], distances[1].get(places[destination]))


class VillageLayoutTests(unittest.TestCase):
    """Verify layout arithmetic and recorded build commands without Minecraft."""

    def test_pre_incident_builds_all_five_sections_and_intact_supports(self):
        """Both pre-fixtures fill the declared gap exactly and keep purchase stock, funds and witnesses intact."""
        for variant in ("accident", "sabotage"):
            with self.subTest(variant=variant):
                config = ScenarioConfig.create(variant, 17, pre_incident=True)
                original = ScenarioConfig.create(variant, 17)
                builder = RecordingVillageBuilder(config)
                builder.build_bridge()
                builder.place_evidence()
                blocks = builder.blocks()
                left, bottom, north, right, top, south = config.layout["bridge"]["gap"]
                gap = {(horizontal, height, depth) for horizontal in range(left, right + 1)
                       for height in range(bottom, top + 1) for depth in range(north, south + 1)}
                deck = {tuple(block["pos"]): block["block"]
                        for block in config.success_predicates["builder"]["all"][0]["blocks_present"]}
                self.assertEqual(set(deck), gap)
                self.assertEqual({position: blocks[position] for position in gap}, deck)
                self.assertEqual(sum(name.startswith("bridge_span_") for name in config.schematics), 5)
                supports = ScenarioConfig.create("accident", 17).evidence.supports
                self.assertEqual(config.evidence.supports, supports)
                self.assertTrue(all(blocks[tuple(support["pos"])] == support["block"] for support in supports))
                self.assertEqual(config.containers["stall_3"]["stock"]["minecraft:iron_pickaxe"], 1)
                self.assertGreaterEqual(next(role.kit["minecraft:emerald"] for role in config.roles
                                             if role.name == "envoy"), 8)
                self.assertFalse(any(command.startswith("summon") for command in builder.commands))
                self.assertEqual(config.npcs, original.npcs)
                self.assertEqual(config.roads, original.roads)
                self.assertEqual(config.npc_payload(), original.npc_payload())
                nodes = {(position[0], position[2]) for position in blocks if position[1] == -61}
                self.assertTrue(all((horizontal, 0) in nodes for horizontal in range(-8, 9)))

    def test_only_pre_incident_builds_spans_in_sorted_deterministic_order(self):
        """Legacy fixtures retain the broken bridge; dictionary and block order cannot alter the pre-build."""
        config = ScenarioConfig.create("sabotage", 17, pre_incident=True)
        shuffled = replace(config, schematics={name: {**schematic, "blocks": list(reversed(schematic["blocks"]))}
                                               for name, schematic in reversed(list(config.schematics.items()))})
        builders = [RecordingVillageBuilder(candidate) for candidate in (config, shuffled)]
        for builder in builders:
            builder.build_bridge()
        self.assertEqual(builders[0].commands, builders[1].commands)
        deck = {tuple(block["pos"]) for block in config.success_predicates["builder"]["all"][0]["blocks_present"]}
        placements = [tuple(map(int, command.split()[1:4])) for command in builders[0].commands
                      if command.startswith("setblock") and tuple(map(int, command.split()[1:4])) in deck]
        self.assertEqual(placements, sorted(deck))
        for variant in ("accident", "sabotage"):
            with self.subTest(variant=variant):
                builder = RecordingVillageBuilder(ScenarioConfig.create(variant, 17))
                builder.build_bridge()
                builder.place_evidence()
                self.assertFalse(deck & builder.blocks().keys())

    def test_layout_routes_population_and_bills(self):
        """Check the ford detour, per-bank place connectivity and exact plank bills."""
        config = ScenarioConfig.create("sabotage", 17)
        self.assertEqual(config.to_dict(), ScenarioConfig.create("sabotage", 17).to_dict())
        places = {place.name: place for place in config.places}
        self.assertEqual(sum(place.kind == "house" for place in config.places), 14)
        self.assertEqual(Counter(npc.role for npc in config.npcs), {"adult": 22, "child": 8})
        self.assertEqual(sum(npc.role == "adult" and places[npc.home].bank != places[npc.work].bank
                             for npc in config.npcs), 8)
        nodes = set()
        for road in config.roads:
            left, north, right, south = road["bounds"]
            self.assertTrue(-128 <= left <= right <= 127 and -128 <= north <= south <= 127)
            nodes.update((horizontal, depth) for horizontal in range(left, right + 1)
                         for depth in range(north, south + 1))
        for key in ("west_abutment", "east_abutment"):
            left, _, north, right, _, south = config.layout["bridge"][key]
            nodes.update((horizontal, depth) for horizontal in range(left, right + 1)
                         for depth in range(north, south + 1))
        nodes.update((horizontal, -120) for horizontal in range(-7, 8))
        nodes.update(config.pedestrian_cells())
        start, target = (-8, 0), (8, 0)
        distances = {start: 0}
        pending = deque([start])
        while pending:
            horizontal, depth = pending.popleft()
            for neighbour in ((horizontal - 1, depth), (horizontal + 1, depth),
                              (horizontal, depth - 1), (horizontal, depth + 1)):
                if neighbour in nodes and neighbour not in distances:
                    distances[neighbour] = distances[(horizontal, depth)] + 1
                    pending.append(neighbour)
        self.assertEqual(distances[target], config.layout["ford"]["ford_route_edges"])
        self.assertGreater(distances[target], 10 * config.layout["ford"]["bridge_route_edges"])
        self.assertTrue(all((place.pos[0], place.pos[2]) in distances for place in config.places))
        for bank, boundary in (("west", -7), ("east", 7)):
            bank_nodes = {node for node in nodes if (node[0] <= boundary if bank == "west" else node[0] >= boundary)}
            visited = set()
            pending = deque([(-8, 0) if bank == "west" else (8, 0)])
            while pending:
                current = pending.popleft()
                if current in visited:
                    continue
                visited.add(current)
                horizontal, depth = current
                pending.extend(neighbour for neighbour in ((horizontal - 1, depth), (horizontal + 1, depth),
                                                          (horizontal, depth - 1), (horizontal, depth + 1))
                               if neighbour in bank_nodes and neighbour not in visited)
            self.assertEqual(visited, bank_nodes)
        deck = set()
        for number, count in enumerate((6, 6, 6, 4, 4), 1):
            section = config.schematics[f"bridge_span_{number}"]
            self.assertEqual(section["bill"], {"minecraft:oak_planks": count})
            self.assertEqual(Counter(block["block"] for block in section["blocks"]), section["bill"])
            origin_x, origin_y, origin_z = section["origin"]
            deck.update((origin_x + block["dx"], origin_y + block["dy"], origin_z + block["dz"])
                        for block in section["blocks"])
        self.assertEqual(deck, {(horizontal, -61, depth) for horizontal in range(-6, 7) for depth in (0, 1)})
        self.assertEqual(config.schematics["port_repair"]["bill"], {"minecraft:oak_planks": 9})


if __name__ == "__main__":
    unittest.main()
