package org.minecraftsim.harness;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.UUID;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.Container;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.storage.LevelResource;
import net.minecraft.world.phys.Vec3;
import org.minecraftsim.harness.RoadGraph.Node;

/**
 * Owns scripted NPC schedules, road movement, daily metrics, and testimony logs.
 * Server-thread ticks use overworld game time and sorted NPC IDs, moving before
 * scheduling and observing after road updates. Schedules use deterministic hashes,
 * not RNG. Private schedule, movement, and metrics methods share the persisted data
 * model; its fields and Gson layout are also part of harness hashing.
 */
public final class Villagers implements AutoCloseable {
	private static final Gson GSON = new GsonBuilder().serializeNulls().create();
	private final MinecraftServer server;
	private final HarnessState harness;
	private final Path directory;
	private final Data data;
	private final List<VillagerPurchases.Arrival> marketArrivals = new ArrayList<>();
	private BufferedWriter welfare;
	private BufferedWriter observations;

	private Villagers(MinecraftServer server, HarnessState harness, Data data) {
		this.server = server;
		this.harness = harness;
		this.data = data;
		directory = server.getWorldPath(LevelResource.ROOT).resolve("simharness");
	}

	/**
	 * Validates roads and NPC definitions, spawns their bodies, and initializes today.
	 * Must run on the server thread; validated NPC bodies are spawned in sorted ID order.
	 * @param server the owning server
	 * @param harness the state used for canonical IDs, events, and observations
	 * @param args road, place, NPC, and optional speed, surface, news-kind and landmark arguments
	 * @return the initialized NPC controller
	 * @throws IllegalArgumentException for invalid configuration or road locations
	 * @throws IllegalStateException if an NPC body cannot be spawned
	 */
	public static Villagers setup(MinecraftServer server, HarnessState harness, JsonObject args) {
		Data data = new Data();
		data.baseSeed = Determinism.baseSeed(server);
		data.speed = args.has("speed") ? args.get("speed").getAsDouble() : 0.12;
		data.schedule = args.has("schedule") ? Schedule.from(args.getAsJsonObject("schedule")) : new Schedule();
		if (!Double.isFinite(data.speed) || data.speed <= 0) {
			throw new IllegalArgumentException("speed must be finite and positive");
		}
		Set<String> surfaces = new TreeSet<>(Set.of("minecraft:dirt_path", "minecraft:gravel", "minecraft:stone_bricks"));
		if (args.has("surfaces")) {
			surfaces.clear();
			for (JsonElement value : args.getAsJsonArray("surfaces")) {
				Identifier id = Identifier.parse(value.getAsString());
				if (!BuiltInRegistries.BLOCK.containsKey(id)) {
					throw new IllegalArgumentException("Unknown road surface: " + id);
				}
				surfaces.add(id.toString());
			}
		}
		data.graph = new RoadGraph(server.overworld(), Region.fromJson(args.getAsJsonObject("region")), surfaces);
		for (var place : args.getAsJsonObject("places").entrySet()) {
			JsonObject coordinates = new JsonObject();
			coordinates.add("pos", place.getValue());
			BlockPos position = ControlCommands.position(coordinates);
			Node node = new Node(position.getX(), position.getY(), position.getZ());
			if (!data.graph.contains(node)) {
				throw new IllegalArgumentException("Place must be a passable road node: " + place.getKey());
			}
			data.places.put(place.getKey(), node);
		}
		for (JsonElement value : args.getAsJsonArray("npcs")) {
			JsonObject definition = value.getAsJsonObject();
			Npc npc = new Npc();
			npc.id = definition.get("id").getAsString();
			npc.name = definition.get("name").getAsString();
			npc.home = definition.get("home").getAsString();
			npc.role = definition.get("role").getAsString();
			npc.work = definition.has("work") ? definition.get("work").getAsString() : null;
			npc.school = definition.has("school") ? definition.get("school").getAsString() : null;
			if (definition.has("market_days")) {
				for (JsonElement day : definition.getAsJsonArray("market_days")) {
					int marketDay = day.getAsBigDecimal().intValueExact();
					if (marketDay < 0 || marketDay > 6) {
						throw new IllegalArgumentException("market_days entries must be 0..6");
					}
					npc.marketDays.add(marketDay);
				}
			}
			if (npc.id.isEmpty() || data.npcs.containsKey(npc.id) || !Set.of("adult", "child").contains(npc.role)) {
				throw new IllegalArgumentException("NPC IDs must be unique and nonempty; role must be adult or child");
			}
			for (String place : new String[]{npc.home, npc.work, npc.school, npc.marketDays.isEmpty() ? null : "market"}) {
				if (place != null && !data.places.containsKey(place)) {
					throw new IllegalArgumentException("Unknown place: " + place);
				}
			}
			npc.uuid = Determinism.npcUuid(data.baseSeed, npc.id);
			if (server.overworld().getEntity(npc.uuid) != null) {
				throw new IllegalArgumentException("NPC UUID already exists: " + npc.id);
			}
			npc.place = npc.home;
			npc.pos = vector(data.places.get(npc.home).feet());
			npc.lastObservationTick = server.overworld().getGameTime();
			npc.observationCursor = harness.counter("eventCounter");
			data.npcs.put(npc.id, npc);
		}
		VillagerNews.validate(args);
		Villagers villagers = new Villagers(server, harness, data);
		for (Npc npc : data.npcs.values()) {
			Villager body = new Villager(EntityTypes.VILLAGER, server.overworld());
			body.setUUID(npc.uuid);
			body.setVillagerDataFinalized(true);
			body.setNoAi(true);
			body.setPermanentlyInvulnerable(true);
			body.setPersistenceRequired();
			body.setSilent(true);
			body.setCustomName(Component.literal(npc.name));
			villagers.position(npc, body);
			if (!server.overworld().addFreshEntity(body)) {
				throw new IllegalStateException("Could not spawn NPC: " + npc.id);
			}
		}
		villagers.day(server.overworld().getGameTime() / 24_000);
		harness.news().configure(args);
		return villagers;
	}

	/**
	 * Restores the saved graph and NPC state and repositions existing bodies.
	 * @param server the owning server, accessed on its server thread
	 * @param harness the restored harness state
	 * @param saved the persisted NPC data tree with its unchanged Gson field layout
	 * @return the restored controller without advancing schedules or movement
	 * @throws RuntimeException if saved data or a required NPC body is invalid
	 */
	public static Villagers restore(MinecraftServer server, HarnessState harness, JsonElement saved) {
		Data data = GSON.fromJson(saved, Data.class);
		if (data.schedule == null) {
			data.schedule = new Schedule();
		}
		data.graph.restored();
		Villagers villagers = new Villagers(server, harness, data);
		for (Npc npc : data.npcs.values()) {
			villagers.position(npc, villagers.body(npc));
		}
		return villagers;
	}

	/**
	 * Returns the persisted NPC data without transient server references or log writers.
	 * @return the Gson data tree used by persistence and harness hashing
	 */
	public JsonElement serialization() {
		return GSON.toJsonTree(data);
	}

	/**
	 * Looks up the stable harness identity of a registered NPC body.
	 * @param entity the entity whose UUID is compared with saved NPC UUIDs
	 * @return {@code npc:<id>}, or {@code null} for an unregistered entity
	 */
	public String id(Entity entity) {
		for (Npc npc : data.npcs.values()) {
			if (npc.uuid.equals(entity.getUUID())) {
				return "npc:" + npc.id;
			}
		}
		return null;
	}

	/** Returns the loaded NPC bodies in stable {@code npc:<rawId>} order without changing state. */
	TreeMap<String, Entity> bodies() {
		TreeMap<String, Entity> result = new TreeMap<>();
		for (Npc npc : data.npcs.values()) {
			result.put("npc:" + npc.id, body(npc));
		}
		return result;
	}

	/** Copies existing named road-place anchors for opt-in news, including legacy fixture restores. */
	JsonObject places() {
		JsonObject places = new JsonObject();
		data.places.forEach((name, node) -> places.add(name, HarnessState.vector(node.feet())));
		return places;
	}

	/**
	 * Returns the place where this villager is dwelling (home, work, market or school), or null while travelling.
	 * Gossip happens only between villagers dwelling at the same place.
	 */
	String dwellingPlace(String key) {
		return dwelling(key) ? require(key).place : null;
	}

	/** Tests for a stationary scheduled-place dwell, excluding travel and stranded failures. */
	boolean dwelling(String key) {
		Npc npc = require(key);
		return npc.trip == null && Set.of("home", "work", "market", "school").contains(npc.activity)
			&& point(npc.pos).equals(data.places.get(npc.place).feet());
	}

	/**
	 * Checks registration without creating harness state or consuming RNG.
	 * @param entity the entity to inspect
	 * @return whether it is a registered NPC in a server level
	 */
	public static boolean isNpc(Entity entity) {
		if (!(entity.level() instanceof net.minecraft.server.level.ServerLevel level)) {
			return false;
		}
		HarnessState state = HarnessState.existing(level.getServer());
		return state != null && state.villagers() != null && state.villagers().id(entity) != null;
	}

	/**
	 * Marks a road location dirty for the existing next-tick graph update.
	 * @param position the changed block, supplied on the server thread
	 */
	public void blockChanged(BlockPos position) {
		data.graph.changed(position);
	}

	/**
	 * Runs movement, schedules, body placement, road reroutes, optional arrival-ordered purchases, then observations.
	 * Called once per completed server tick; all phase times use overworld game time.
	 */
	public void tick() {
		long tick = server.overworld().getGameTime();
		Day today = day(tick / 24_000);
		for (Npc npc : data.npcs.values()) {
			if (npc.trip != null) {
				move(npc, today, tick);
			}
			schedule(npc, today, tick);
			position(npc, body(npc));
		}
		if (data.graph.update(server.overworld())) {
			for (Npc npc : data.npcs.values()) {
				if (npc.trip != null) {
					route(npc, tick);
				}
			}
		}
		marketArrivals.sort(Comparator.naturalOrder());
		for (VillagerPurchases.Arrival arrival : marketArrivals) {
			purchase(arrival.npc(), arrival.tick() / 24_000);
		}
		marketArrivals.clear();
		for (Npc npc : data.npcs.values()) {
			if (tick - npc.lastObservationTick >= 20) {
				observe(npc, tick);
			}
		}
	}

	private Day day(long number) {
		Day day = data.days.get(number);
		if (day != null) {
			return day;
		}
		if (data.days.containsKey(number - 1)) {
			JsonObject event = harness.event("npc_day");
			event.add("metrics", dailyJson(data.days.get(number - 1)));
			append(false, event);
		}
		day = new Day();
		day.day = number;
		data.days.put(number, day);
		for (Npc npc : data.npcs.values()) {
			npc.day = number;
			npc.offset = Determinism.npcOffset(data.baseSeed, npc.id, number, npc.role.equals("adult") ? 400 : 200);
			npc.departed = false;
			npc.evening = false;
			Daily record = new Daily();
			record.npc = npc.id;
			record.workOnTime = npc.role.equals("adult") && npc.work != null ? false : null;
			record.workStatus = record.workOnTime == null ? "not_scheduled" : "pending";
			record.marketVisited = marketDay(npc, number) ? false : null;
			record.marketStatus = record.marketVisited == null ? "not_scheduled" : "pending";
			record.schoolStatus = npc.role.equals("child") && npc.school != null ? "pending" : "not_scheduled";
			day.npcs.put(npc.id, record);
			if (npc.role.equals("adult")) {
				if (npc.work != null) {
					firstPath(record, npc.home, npc.work);
				}
				String workplace = npc.work == null ? npc.home : npc.work;
				if (marketDay(npc, number)) {
					firstPath(record, workplace, "market");
					firstPath(record, "market", npc.home);
				}
				firstPath(record, workplace, npc.home);
			} else if (npc.school != null) {
				firstPath(record, npc.home, npc.school);
				firstPath(record, npc.school, npc.home);
			}
		}
		return day;
	}

	private boolean marketDay(Npc npc, long day) {
		return npc.role.equals("adult") && npc.marketDays.contains((int) (day % 7));
	}

	private void firstPath(Daily record, String from, String to) {
		record.firstPaths.put(from + "\n" + to, data.graph.path(data.places.get(from), data.places.get(to)));
	}

	private void schedule(Npc npc, Day today, long tick) {
		if (npc.trip != null) {
			return;
		}
		long phase = tick % 24_000;
		Schedule times = data.schedule;
		if (npc.role.equals("adult")) {
			if (!npc.departed && phase >= times.workDepart + npc.offset) {
				npc.departed = true;
				if (npc.work != null) {
					depart(npc, npc.work, "work", tick);
				}
			}
			if (npc.trip == null && !npc.evening && phase >= times.adultEvening) {
				npc.evening = true;
				if (marketDay(npc, today.day)) {
					depart(npc, "market", "market", tick);
				} else {
					home(npc, tick);
				}
			}
		} else {
			if (!npc.departed && phase >= times.childDepart + npc.offset) {
				npc.departed = true;
				if (npc.school != null) {
					depart(npc, npc.school, "school", tick);
				}
			}
			if (npc.trip == null && !npc.evening && phase >= times.childEvening) {
				npc.evening = true;
				home(npc, tick);
			}
		}
	}

	private void home(Npc npc, long tick) {
		if (!point(npc.pos).equals(data.places.get(npc.home).feet())) {
			depart(npc, npc.home, "home", tick);
		} else {
			npc.activity = "home";
		}
	}

	private void depart(Npc npc, String destination, String activity, long tick) {
		Trip trip = new Trip();
		trip.day = tick / 24_000;
		trip.npc = npc.id;
		trip.from = npc.place;
		trip.to = destination;
		trip.departTick = tick;
		trip.activity = activity;
		npc.trip = trip;
		npc.activity = "travel_" + activity;
		route(npc, tick);
	}

	private Node snap(Npc npc) {
		if (npc.path.isEmpty()) {
			return data.places.get(npc.place);
		}
		int index = Math.min(npc.index, npc.path.size() - 1);
		return java.util.stream.Stream.of(npc.path.get(Math.max(0, index - 1)), npc.path.get(index))
			.distinct().filter(data.graph::contains)
			.min(Comparator.comparingDouble((Node node) -> node.feet().distanceToSqr(point(npc.pos))).thenComparing(Comparator.naturalOrder()))
			.orElse(npc.path.get(index));
	}

	private void route(Npc npc, long tick) {
		Trip trip = npc.trip;
		Node start = snap(npc);
		List<Node> path = data.graph.path(start, data.places.get(trip.to));
		int previousStart = npc.path.indexOf(start);
		if (!trip.segments.isEmpty() && previousStart >= 0 && !path.isEmpty()
				&& path.equals(npc.path.subList(previousStart, npc.path.size()))) {
			return;
		}
		if (!path.isEmpty()) {
			npc.path = path;
			npc.index = 0;
		}
		Segment segment = new Segment();
		segment.tick = tick;
		segment.position = npc.pos.clone();
		segment.travelled = trip.travelled;
		segment.movingTicks = trip.ticks;
		segment.path = path;
		segment.snapDistance = path.isEmpty() ? 0 : point(npc.pos).distanceTo(start.feet());
		trip.segments.add(segment);
		trip.path = path;
		trip.pathLength = Math.max(0, path.size() - 1);
		Daily record = data.days.get(trip.day).npcs.get(npc.id);
		String key = trip.from + "\n" + trip.to;
		record.firstPaths.putIfAbsent(key, data.graph.path(data.places.get(trip.from), data.places.get(trip.to)));
		if (trip.segments.size() > 1
				|| !record.firstPaths.get(key).equals(data.graph.path(data.places.get(trip.from), data.places.get(trip.to)))) {
			data.days.get(trip.day).affectedNpcs.add(npc.id);
		}
		if (path.isEmpty()) {
			finish(npc, tick, false);
		} else if (path.size() == 1 && segment.snapDistance == 0) {
			finish(npc, tick, true);
		}
	}

	private void move(Npc npc, Day today, long tick) {
		Trip trip = npc.trip;
		trip.ticks++;
		today.npcs.get(npc.id).commuteTicks++;
		double budget = data.speed;
		while (npc.index < npc.path.size()) {
			Vec3 current = point(npc.pos);
			Vec3 target = npc.path.get(npc.index).feet();
			double distance = current.distanceTo(target);
			if (distance > 1.0e-9) {
				Vec3 direction = target.subtract(current).scale(1 / distance);
				npc.yaw = (float) Math.toDegrees(Math.atan2(-direction.x, direction.z));
				if (distance > budget + 1.0e-9) {
					npc.pos = vector(current.add(direction.scale(budget)));
					trip.travelled += budget;
					return;
				}
			}
			npc.pos = vector(target);
			trip.travelled += distance;
			budget = Math.max(0, budget - distance);
			npc.index++;
		}
		finish(npc, tick, true);
	}

	private void finish(Npc npc, long tick, boolean arrived) {
		Trip trip = npc.trip;
		Daily record = data.days.get(trip.day).npcs.get(npc.id);
		trip.arriveTick = arrived ? tick : null;
		trip.outcome = arrived ? "arrived" : "unreachable";
		trip.onTime = arrived;
		long phase = tick - trip.day * 24_000;
		if (trip.activity.equals("work")) {
			trip.onTime = arrived && phase <= data.schedule.workDeadline;
			record.workOnTime = trip.onTime;
			record.workStatus = !arrived ? "unreachable" : trip.onTime ? "on_time" : "late";
		} else if (trip.activity.equals("school")) {
			trip.onTime = arrived && phase <= data.schedule.schoolDeadline;
			record.schoolStatus = !arrived || phase > data.schedule.schoolAbsent ? "absent" : trip.onTime ? "on_time" : "late";
			if (record.schoolStatus.equals("absent")) {
				trip.outcome = "absent";
			}
		} else if (trip.activity.equals("market")) {
			trip.onTime = arrived && phase <= data.schedule.marketDeadline;
			record.marketVisited = trip.onTime;
			record.marketStatus = !arrived ? "unreachable" : trip.onTime ? "visited" : "missed";
			if (arrived && !trip.onTime) {
				trip.outcome = "missed";
			}
		}
		if (arrived) {
			npc.place = trip.to;
		}
		npc.activity = arrived ? trip.activity : "stranded";
		npc.trip = null;
		data.days.get(trip.day).trips.add(trip);
		JsonObject event = tripJson(trip);
		harness.event("npc_trip").entrySet().forEach(entry -> event.add(entry.getKey(), entry.getValue()));
		append(false, event);
		if (arrived && npc.role.equals("adult") && trip.activity.equals("market")
				&& trip.to.equals("market") && !harness.config().villagerPurchases().isEmpty()) {
			marketArrivals.add(new VillagerPurchases.Arrival(tick, npc.id));
		}
		if (trip.activity.equals("market") || trip.activity.equals("school") && trip.outcome.equals("absent")) {
			home(npc, tick);
		}
	}

	/** Buys each configured market good in configuration order, without adding any persisted NPC state. */
	private void purchase(String npc, long today) {
		for (JsonElement value : harness.config().villagerPurchases()) {
			JsonObject entry = value.getAsJsonObject();
			JsonArray stall = entry.getAsJsonArray("stall");
			BlockPos position = new BlockPos(stall.get(0).getAsInt(), stall.get(1).getAsInt(), stall.get(2).getAsInt());
			String itemId = entry.get("item").getAsString();
			int count = entry.get("count").getAsInt();
			JsonObject shops = harness.map("shops");
			JsonObject prices = shops == null ? null : shops.getAsJsonObject(HarnessState.key(position));
			VillagerPurchases.Outcome outcome = VillagerPurchases.Outcome.SOLD_OUT;
			int price = 0;
			if (prices != null && prices.has(itemId)
					&& server.overworld().getBlockEntity(position) instanceof Container shop) {
				price = prices.get(itemId).getAsInt();
				Item item = BuiltInRegistries.ITEM.getValue(Identifier.parse(itemId));
				outcome = VillagerPurchases.purchase(shop, item, count, price, today, harness.config().freshDays());
			}
			if (outcome == VillagerPurchases.Outcome.PURCHASED) {
				JsonObject event = harness.event("purchase");
				event.addProperty("buyer", npc);
				event.addProperty("shop", "container:" + HarnessState.key(position));
				event.addProperty("item", itemId);
				event.addProperty("count", count);
				event.addProperty("cost", Math.multiplyExact(price, count));
				harness.logEvent(event);
			} else if (outcome == VillagerPurchases.Outcome.SOLD_OUT) {
				JsonObject event = harness.event("sold_out");
				event.addProperty("npc", npc);
				event.add("stall", stall.deepCopy());
				event.addProperty("item", itemId);
				harness.logEvent(event);
			}
		}
	}

	private Villager body(Npc npc) {
		Entity entity = server.overworld().getEntity(npc.uuid);
		if (!(entity instanceof Villager body)) {
			throw new IllegalStateException("NPC entity is not loaded: " + npc.id + "; await_entities before restore_bots");
		}
		return body;
	}

	private void position(Npc npc, Villager body) {
		body.noPhysics = true;
		body.setNoGravity(true);
		body.setPos(point(npc.pos));
		body.setDeltaMovement(Vec3.ZERO);
		body.setYRot(npc.yaw);
		body.setYHeadRot(npc.yaw);
		body.setYBodyRot(npc.yaw);
		body.setXRot(0);
	}

	private static double[] vector(Vec3 position) {
		return new double[]{position.x, position.y, position.z};
	}

	private static Vec3 point(double[] position) {
		return new Vec3(position[0], position[1], position[2]);
	}

	private static JsonArray pathJson(List<Node> path) {
		JsonArray result = new JsonArray();
		path.forEach(node -> result.add(HarnessState.position(node.block())));
		return result;
	}

	private static JsonObject tripJson(Trip trip) {
		JsonObject result = GSON.toJsonTree(trip).getAsJsonObject();
		result.add("path", pathJson(trip.path));
		for (int index = 0; index < trip.segments.size(); index++) {
			result.getAsJsonArray("segments").get(index).getAsJsonObject().add("path", pathJson(trip.segments.get(index).path));
		}
		return result;
	}

	/**
	 * Builds the public NPC state without advancing movement or schedules.
	 * @param id the bare NPC ID or its {@code npc:} identity
	 * @return the existing state response, including current route, trip and live salient news
	 * @throws IllegalArgumentException if the NPC is unknown
	 */
	public JsonObject state(String id) {
		Npc npc = require(id);
		JsonObject result = new JsonObject();
		result.addProperty("id", npc.id);
		result.addProperty("npcId", "npc:" + npc.id);
		result.addProperty("name", npc.name);
		result.addProperty("uuid", npc.uuid.toString());
		result.add("pos", GSON.toJsonTree(npc.pos));
		result.addProperty("activity", npc.activity);
		result.add("path", pathJson(npc.path));
		result.addProperty("index", npc.index);
		result.addProperty("offset", npc.offset);
		result.addProperty("graphVersion", data.graph.version());
		result.add("trip", npc.trip == null ? com.google.gson.JsonNull.INSTANCE : tripJson(npc.trip));
		result.add("news", harness.news().items("npc:" + npc.id, Integer.MAX_VALUE));
		return result;
	}

	private Npc require(String id) {
		Npc npc = data.npcs.get(id);
		if (npc == null && id.startsWith("npc:")) {
			npc = data.npcs.get(id.substring(4));
		}
		if (npc == null) {
			throw new IllegalArgumentException("Unknown NPC: " + id);
		}
		return npc;
	}

	/**
	 * Builds daily metrics in saved day and NPC order without modifying them.
	 * @param args optional integral {@code day} filter; omission selects all saved days
	 * @return the existing {@code days} response with trip and aggregate metrics
	 * @throws ArithmeticException if a supplied day is not an exact long integer
	 */
	public JsonObject metrics(JsonObject args) {
		JsonObject result = new JsonObject();
		JsonArray days = new JsonArray();
		Long requested = args.has("day") ? args.get("day").getAsBigDecimal().longValueExact() : null;
		for (Day day : data.days.values()) {
			if (requested == null || day.day == requested) {
				days.add(dailyJson(day));
			}
		}
		result.add("days", days);
		return result;
	}

	private JsonObject dailyJson(Day day) {
		JsonObject result = new JsonObject();
		result.addProperty("day", day.day);
		JsonArray records = new JsonArray();
		long total = 0;
		long maximum = 0;
		int scheduled = 0;
		int late = 0;
		int markets = 0;
		int missed = 0;
		int schools = 0;
		int absent = 0;
		for (Daily record : day.npcs.values()) {
			JsonObject item = GSON.toJsonTree(record).getAsJsonObject();
			item.remove("firstPaths");
			records.add(item);
			total += record.commuteTicks;
			maximum = Math.max(maximum, record.commuteTicks);
			if (record.workOnTime != null) {
				scheduled++;
				if (record.workStatus.equals("late")) {
					late++;
				}
			}
			if (record.marketVisited != null) {
				markets++;
				if (record.marketStatus.equals("missed") || record.marketStatus.equals("unreachable")) {
					missed++;
				}
			}
			if (!record.schoolStatus.equals("not_scheduled")) {
				schools++;
				scheduled++;
				if (record.schoolStatus.equals("late")) {
					late++;
				} else if (record.schoolStatus.equals("absent")) {
					absent++;
				}
			}
		}
		result.add("npcs", records);
		JsonArray trips = new JsonArray();
		day.trips.forEach(trip -> trips.add(tripJson(trip)));
		result.add("trips", trips);
		result.addProperty("meanCommute", day.npcs.isEmpty() ? 0 : (double) total / day.npcs.size());
		result.addProperty("maxCommute", maximum);
		result.addProperty("lateShare", scheduled == 0 ? 0 : (double) late / scheduled);
		result.addProperty("missedMarketShare", markets == 0 ? 0 : (double) missed / markets);
		result.addProperty("absentShare", schools == 0 ? 0 : (double) absent / schools);
		result.add("affectedNpcs", GSON.toJsonTree(day.affectedNpcs));
		return result;
	}

	private void observe(Npc npc, long tick) {
		Villager observer = body(npc);
		Vec3 eye = observer.getEyePosition();
		double radius = harness.config().number("view_radius");
		JsonArray entities = new JsonArray();
		List<Entity> candidates = new ArrayList<>();
		server.overworld().getAllEntities().forEach(entity -> {
			if (entity != observer && !entity.isRemoved() && (entity instanceof SimBot || id(entity) != null)
					&& eye.distanceTo(entity.position()) <= radius) {
				candidates.add(entity);
			}
		});
		candidates.sort(Comparator.comparing(Entity::getUUID));
		for (Entity entity : candidates) {
			Sight.Ray ray = Sight.entity(observer, entity);
			if (ray == null) {
				continue;
			}
			JsonObject item = new JsonObject();
			item.addProperty("prov", "entity:" + entity.getUUID());
			item.addProperty("name", entity instanceof SimBot bot ? bot.getGameProfile().name() : entity.getName().getString());
			if (id(entity) != null) {
				item.addProperty("npcId", id(entity));
			}
			item.add("pos", HarnessState.vector(entity.position()));
			JsonObject load = VisibleLoads.seen(entity, harness.config());
			if (load != null) {
				item.add("load", load);
			}
			if (harness.config().nightSight()) {
				Vec3 center = entity.getBoundingBox().getCenter();
				item.addProperty("outside", harness.news().outside(center)
					|| server.overworld().canSeeSky(BlockPos.containing(center)));
			}
			ray.audit(item);
			entities.add(item);
		}
		JsonArray changes = new JsonArray();
		for (JsonElement value : harness.changes()) {
			JsonObject change = value.getAsJsonObject();
			if (change.get("id").getAsLong() <= npc.observationCursor) {
				continue;
			}
			JsonObject coordinates = new JsonObject();
			coordinates.add("pos", change.get("pos"));
			BlockPos position = ControlCommands.position(coordinates);
			if (eye.distanceTo(Vec3.atCenterOf(position)) > radius) {
				continue;
			}
			Sight.Ray ray = Sight.block(observer, position);
			if (ray != null) {
				JsonObject item = change.deepCopy();
				item.addProperty("prov", "change:" + change.get("id").getAsLong());
				ray.audit(item);
				changes.add(item);
			}
		}
		JsonObject record = harness.event("npc_observation");
		record.addProperty("prov", "npc_observation:" + record.get("id").getAsLong());
		record.addProperty("npc", npc.id);
		record.addProperty("observer", "npc:" + npc.id);
		record.add("observerEye", HarnessState.vector(eye));
		record.add("entities", entities);
		if (harness.config().nightSight()) {
			record.addProperty("dayTime", server.overworld().getOverworldClockTime());
		}
		record.add("changes", changes);
		append(true, record);
		harness.news().observe("npc:" + npc.id, record);
		npc.lastObservationTick = tick;
		npc.observationCursor = harness.counter("eventCounter");
	}

	/**
	 * Flushes logs and reads an NPC's testimony in file order over an inclusive interval.
	 * @param args the NPC {@code id} and optional integral {@code since}/{@code until}
	 * @return the matching records, or an empty list when the log does not exist
	 * @throws IllegalArgumentException if the NPC is unknown
	 * @throws IOException if logs cannot be flushed or read
	 */
	public JsonObject testimony(JsonObject args) throws IOException {
		Npc npc = require(args.get("id").getAsString());
		long since = args.has("since") ? args.get("since").getAsBigDecimal().longValueExact() : Long.MIN_VALUE;
		long until = args.has("until") ? args.get("until").getAsBigDecimal().longValueExact() : Long.MAX_VALUE;
		flush();
		JsonArray records = new JsonArray();
		Path path = directory.resolve("npc_observations.jsonl");
		if (Files.exists(path)) {
			try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
				for (String line; (line = reader.readLine()) != null;) {
					JsonObject record = JsonParser.parseString(line).getAsJsonObject();
					long tick = record.get("gameTime").getAsLong();
					if (record.get("npc").getAsString().equals(npc.id) && tick >= since && tick <= until) {
						records.add(record);
					}
				}
			}
		}
		JsonObject result = new JsonObject();
		result.add("records", records);
		return result;
	}

	private void append(boolean observation, JsonObject record) {
		try {
			BufferedWriter writer = observation ? observations : welfare;
			if (writer == null) {
				Files.createDirectories(directory);
				writer = Files.newBufferedWriter(directory.resolve(observation ? "npc_observations.jsonl" : "welfare.jsonl"),
					StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
				if (observation) {
					observations = writer;
				} else {
					welfare = writer;
				}
			}
			writer.write(HarnessState.canonical(record));
			writer.newLine();
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
	}

	/**
	 * Flushes open welfare and observation logs without creating missing logs.
	 * @throws IOException if either writer cannot be flushed
	 */
	public void flush() throws IOException {
		if (welfare != null) {
			welfare.flush();
		}
		if (observations != null) {
			observations.flush();
		}
	}

	/**
	 * Closes open welfare and observation writers in that order.
	 * @throws IOException if either writer cannot be closed
	 */
	@Override
	public void close() throws IOException {
		if (welfare != null) {
			welfare.close();
		}
		if (observations != null) {
			observations.close();
		}
	}

	/**
	 * Daily phase times (ticks into the day) for departures, deadlines and evenings. The defaults fit the small V1
	 * town; larger scenarios pass a {@code schedule} object to {@code npc_setup}, calibrated to their commutes.
	 */
	private static final class Schedule {
		long workDepart = 1_000;
		long workDeadline = 3_000;
		long childDepart = 1_500;
		long schoolDeadline = 2_500;
		long schoolAbsent = 4_000;
		long childEvening = 8_000;
		long adultEvening = 9_000;
		long marketDeadline = 11_000;

		/** Reads overrides for any of the fields, keeping defaults for the rest, and checks each lies within a day. */
		static Schedule from(JsonObject json) {
			JsonObject merged = GSON.toJsonTree(new Schedule()).getAsJsonObject();
			for (var entry : json.entrySet()) {
				if (!merged.has(entry.getKey())) {
					throw new IllegalArgumentException("Unknown schedule field: " + entry.getKey());
				}
				merged.add(entry.getKey(), entry.getValue());
			}
			Schedule schedule = GSON.fromJson(merged, Schedule.class);
			for (long value : new long[]{schedule.workDepart, schedule.workDeadline, schedule.childDepart, schedule.schoolDeadline,
					schedule.schoolAbsent, schedule.childEvening, schedule.adultEvening, schedule.marketDeadline}) {
				if (value < 0 || value >= 24_000) {
					throw new IllegalArgumentException("Schedule times must lie within a day");
				}
			}
			return schedule;
		}
	}

	private static final class Data {
		long baseSeed;
		double speed;
		Schedule schedule;
		RoadGraph graph;
		TreeMap<String, Node> places = new TreeMap<>();
		TreeMap<String, Npc> npcs = new TreeMap<>();
		TreeMap<Long, Day> days = new TreeMap<>();
	}

	private static final class Npc {
		String id;
		String name;
		String role;
		String home;
		String work;
		String school;
		Set<Integer> marketDays = new TreeSet<>();
		UUID uuid;
		double[] pos;
		float yaw;
		String place;
		String activity = "home";
		List<Node> path = List.of();
		int index;
		long day;
		int offset;
		boolean departed;
		boolean evening;
		Trip trip;
		long lastObservationTick;
		long observationCursor;
	}

	private static final class Day {
		long day;
		TreeMap<String, Daily> npcs = new TreeMap<>();
		List<Trip> trips = new ArrayList<>();
		Set<String> affectedNpcs = new TreeSet<>();
	}

	private static final class Daily {
		String npc;
		long commuteTicks;
		Boolean workOnTime;
		String workStatus = "not_scheduled";
		Boolean marketVisited;
		String marketStatus;
		String schoolStatus;
		TreeMap<String, List<Node>> firstPaths = new TreeMap<>();
	}

	private static final class Trip {
		long day;
		String npc;
		String from;
		String to;
		String activity;
		long departTick;
		Long arriveTick;
		long ticks;
		int pathLength;
		Boolean onTime;
		String outcome;
		double travelled;
		List<Node> path = List.of();
		List<Segment> segments = new ArrayList<>();
	}

	private static final class Segment {
		long tick;
		double[] position;
		double travelled;
		long movingTicks;
		double snapDistance;
		List<Node> path;
	}
}
