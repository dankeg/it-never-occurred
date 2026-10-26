package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import net.minecraft.core.BlockPos;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.AbstractFurnaceBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.phys.Vec3;

/**
 * Derives news from admitted observations and runs deterministic, dwell-only gossip.
 * All scenario, knowledge, successful shares and current encounters live beneath
 * {@code HarnessState.map("communication")}; this service holds no simulation state.
 */
final class VillagerNews {
	private static final long DAY_TICKS = 24_000L;
	private static final Map<String, Double> SALIENCE = Map.of(
		"road_blocked", 0.9, "bridge_or_port_change", 0.8, "shout", 0.6, "construction", 0.2,
		"load", 0.7, "night", 0.6, "sold_out", 0.7, "decoration", 0.5, "change", 0.8);
	private static final Set<String> DEFAULT_KINDS = Set.of(
		"road_blocked", "bridge_or_port_change", "shout", "construction");
	private static final Set<String> DEFAULT_SURFACES = Set.of(
		"minecraft:dirt_path", "minecraft:gravel", "minecraft:stone_bricks");
	private final MinecraftServer server;
	private final HarnessState harness;

	/** Binds the service without accessing state that the harness has not initialized yet. */
	VillagerNews(MinecraftServer server, HarnessState harness) {
		this.server = server;
		this.harness = harness;
	}

	/** Validates the news-specific setup before any NPC bodies are spawned. */
	static void validate(JsonObject args) {
		scenario(args, false);
	}

	/**
	 * Saves validated scenario news settings after the NPC bodies and schedules are ready.
	 * Does not look up the controller, which the harness assigns only after setup returns.
	 */
	void configure(JsonObject args) {
		JsonObject scenario = scenario(args, harness.config().nightSight()
			|| !harness.config().visibleLoads().isEmpty() || purchasesEnabled(harness.config().json())
			|| harness.config().truthfulNews() || harness.config().askAbout());
		scenario.addProperty("base_seed", Determinism.baseSeed(server));
		state().add("scenario", scenario);
		section("news");
		section("shared");
		section("encounters");
	}

	/**
	 * Admits first-hand news only from entities and changes in an already appended sight record.
	 * Classification precedes the enabled-kind filter: disabling a specific kind suppresses
	 * it rather than relabeling the same source as lower-salience construction.
	 */
	void observe(String npcKey, JsonObject observationRecord) {
		JsonObject scenario = runtimeScenario();
		if (scenario == null) {
			return;
		}
		String witness = harness.communication().name(npcKey);
		for (JsonElement value : observedPeople(scenario, witness, observationRecord,
				harness.config().nightSight(), harness.config().visibleLoads())) {
			remember(npcKey, value.getAsJsonObject());
		}
		if (!observationRecord.has("changes")) {
			return;
		}
		for (JsonElement value : observationRecord.getAsJsonArray("changes")) {
			JsonObject change = value.getAsJsonObject();
			BlockPos position = ControlCommands.position(change);
			JsonObject news = blockItem(scenario, witness, observationRecord, change,
				harness.config().nightSight() && emitting(change, position), roadBlocked(change, position, scenario),
				harness.config().truthfulNews());
			if (enabled(news.get("kind").getAsString())) {
				remember(npcKey, news);
			}
		}
	}

	/** Learns a configured purchase failure only after its complete engine event has been logged. */
	void soldOut(JsonObject event) {
		JsonObject scenario = runtimeScenario();
		if (scenario == null || !event.get("type").getAsString().equals("sold_out") || !enabled("sold_out")) {
			return;
		}
		String npcKey = event.get("npc").getAsString();
		if (!npcKey.startsWith("npc:")) {
			npcKey = "npc:" + npcKey;
		}
		remember(npcKey, soldOutItem(scenario, harness.communication().name(npcKey), event));
	}

	/** Treats named building bounds as indoors, without mistaking tree cover for a building. */
	boolean outside(Vec3 position) {
		JsonObject scenario = state().getAsJsonObject("scenario");
		return scenario == null || outside(position, scenario);
	}

	/** Tests the target's precise center against inclusive block bounds, independently of sky or light levels. */
	static boolean outside(Vec3 position, JsonObject scenario) {
		return landmark(BlockPos.containing(position), scenario) == null;
	}

	/** Derives person news from admitted bot sightings, never from inventories or unseen world participants. */
	static JsonArray observedPeople(JsonObject scenario, String witness, JsonObject observation,
			boolean nightSight, Map<String, Integer> visibleLoads) {
		JsonArray result = new JsonArray();
		boolean loads = enabled("load", scenario, nightSight, !visibleLoads.isEmpty(), false);
		boolean night = enabled("night", scenario, nightSight, !visibleLoads.isEmpty(), false)
			&& observation.has("dayTime") && NightSight.isNight(observation.get("dayTime").getAsLong());
		if ((!loads && !night) || !observation.has("entities")) {
			return result;
		}
		for (JsonElement value : observation.getAsJsonArray("entities")) {
			JsonObject entity = value.getAsJsonObject();
			if (!namedPerson(entity)) {
				continue;
			}
			if (loads && entity.has("load")) {
				JsonObject load = entity.getAsJsonObject("load");
				if (visibleLoads.containsKey(load.get("item").getAsString())
						&& load.get("count_band").getAsString().equals("sack")) {
					JsonObject news = personItem("load", scenario, witness, observation, entity);
					news.add("item", load.get("item").deepCopy());
					news.addProperty("count_band", "sack");
					news.addProperty("text", entity.get("name").getAsString() + " was carrying a sack of "
						+ itemName(load.get("item").getAsString()) + location(news) + ".");
					result.add(news);
				}
			}
			if (night && entity.has("outside") && entity.get("outside").getAsBoolean()) {
				JsonObject news = personItem("night", scenario, witness, observation, entity);
				news.addProperty("text", entity.get("name").getAsString() + " was outside at night" + location(news) + ".");
				result.add(news);
			}
		}
		return result;
	}

	/** Keeps exact sight provenance while coalescing repeated reports of the same person, goods, day and place. */
	private static JsonObject personItem(String kind, JsonObject scenario, String witness,
			JsonObject observation, JsonObject entity) {
		String origin = observation.get("prov").getAsString();
		long tick = observation.get("gameTime").getAsLong();
		JsonObject news = item(origin, kind, entity.get("name"), witness, tick, tick);
		String person = entity.get(entity.has("prov") ? "prov" : "name").getAsString();
		locate(news, entity.getAsJsonArray("pos"), scenario);
		news.addProperty("newsId", "sighting:" + Math.floorDiv(tick, DAY_TICKS) + ":" + kind + ":" + person
			+ (kind.equals("load") ? ":" + entity.getAsJsonObject("load").get("item").getAsString() : "")
			+ ":" + (news.has("place") ? news.get("place").getAsString() : "outside"));
		return news;
	}

	/** Names a stall's place, not its shopper, and anchors first-hand knowledge to the logged event. */
	static JsonObject soldOutItem(JsonObject scenario, String witness, JsonObject event) {
		JsonArray position = event.getAsJsonArray("stall");
		String place = place(position, scenario);
		String subject = place == null ? "the stall at " + position : place;
		long tick = event.get("gameTime").getAsLong();
		JsonObject news = item("sold_out:" + event.get("id").getAsLong(), "sold_out",
			new JsonPrimitive(subject), witness, tick, tick);
		locate(news, position, scenario);
		news.add("item", event.get("item").deepCopy());
		news.addProperty("text", "The stall at " + (place == null ? position.toString() : place) + " had sold out of "
			+ itemName(event.get("item").getAsString()) + ".");
		return news;
	}

	/** Only an admitted change to an emitting block inside named bounds becomes decoration news. */
	static JsonObject decorationItem(JsonObject scenario, String witness, JsonObject observation,
			JsonObject change, boolean emitting) {
		String name = emitting ? landmark(ControlCommands.position(change), scenario) : null;
		if (name == null) {
			return null;
		}
		JsonObject news = observedItem("decoration", new JsonPrimitive(name), witness, observation, change);
		news.add("pos", change.get("pos").deepCopy());
		news.addProperty("place", name);
		news.addProperty("text", name + " was being decorated with lights.");
		return news;
	}

	/** Classifies an admitted block change, leaving legacy item bytes untouched unless truthful news is enabled. */
	static JsonObject blockItem(JsonObject scenario, String witness, JsonObject observation, JsonObject change,
			boolean emitting, boolean blocked, boolean truthfulNews) {
		String replacement = truthfulNews ? blockName(change.get("to").getAsString()) : "";
		boolean oven = truthfulNews && oven(replacement);
		JsonObject decoration = !oven ? decorationItem(scenario, witness, observation, change, emitting) : null;
		if (decoration != null) {
			return decoration;
		}
		BlockPos position = ControlCommands.position(change);
		String landmark = landmark(position, scenario);
		String action = oven ? ovenAction(change) : null;
		String kind = blocked ? "road_blocked" : truthfulNews && (landmark != null || action != null) ? "change"
			: landmark == null ? "construction" : "bridge_or_port_change";
		JsonElement subject = !blocked && landmark != null ? new JsonPrimitive(landmark) : HarnessState.position(position);
		JsonObject news = observedItem(kind, subject, witness, observation, change);
		if (kind.equals("change")) {
			String previous = blockName(change.get("from").getAsString());
			boolean removed = air(replacement);
			news.addProperty("block", removed ? previous : replacement);
			news.addProperty("action", action != null ? action : removed ? "removed" : air(previous) ? "placed" : "changed");
			locate(news, change.getAsJsonArray("pos"), scenario);
		}
		return news;
	}

	/** Removes namespace and properties without consulting registry state or the later world. */
	private static String blockName(String state) {
		int properties = state.indexOf('[');
		String block = properties < 0 ? state : state.substring(0, properties);
		return block.substring(block.indexOf(':') + 1);
	}

	private static boolean air(String block) {
		return Set.of("air", "cave_air", "void_air").contains(block);
	}

	private static boolean oven(String block) {
		return Set.of("smoker", "furnace", "blast_furnace").contains(block);
	}

	/** Captures oven light transitions at occurrence time before property-free block IDs lose that evidence. */
	static void recordOven(JsonObject change, BlockState from, BlockState to, boolean truthfulNews) {
		if (truthfulNews && to.getBlock() instanceof AbstractFurnaceBlock) {
			change.addProperty("fromLit", from.hasProperty(BlockStateProperties.LIT) && from.getValue(BlockStateProperties.LIT));
			change.addProperty("toLit", to.getValue(BlockStateProperties.LIT));
		}
	}

	/** Reports only admitted lit-property transitions; other oven changes retain their physical action. */
	private static String ovenAction(JsonObject change) {
		String before = change.get("from").getAsString();
		String after = change.get("to").getAsString();
		boolean wasLit = change.has("fromLit") ? change.get("fromLit").getAsBoolean()
			: before.replace('[', ',').replace(']', ',').contains(",lit=true,");
		boolean lit = change.has("toLit") ? change.get("toLit").getAsBoolean()
			: after.replace('[', ',').replace(']', ',').contains(",lit=true,");
		return lit && (!wasLit || !blockName(before).equals(blockName(after))) ? "lit"
			: wasLit && !lit && blockName(before).equals(blockName(after)) ? "unlit" : null;
	}

	/** Reads only the loaded, currently admitted replacement state; asynchronous light levels are never consulted. */
	private boolean emitting(JsonObject change, BlockPos position) {
		if (!server.overworld().hasChunkAt(position)) {
			return false;
		}
		BlockState block = server.overworld().getBlockState(position);
		return HarnessState.blockId(block).equals(change.get("to").getAsString()) && block.getLightEmission() > 0;
	}

	/** Applies the same admitted, named-bot predicate to person news and block-scene presence. */
	private static boolean namedPerson(JsonObject entity) {
		return !entity.has("npcId") && entity.has("name") && !entity.get("name").getAsString().isBlank()
			&& entity.has("losRay") && !entity.get("losRay").isJsonNull() && entity.get("losRay").getAsInt() >= 0;
	}

	/** Copies an observed position and, when available, its named building or nearest configured place. */
	private static void locate(JsonObject news, JsonArray position, JsonObject scenario) {
		news.add("pos", position.deepCopy());
		String name = place(position, scenario);
		if (name != null) {
			news.addProperty("place", name);
		}
	}

	private static String place(JsonArray position, JsonObject scenario) {
		Vec3 feet = new Vec3(position.get(0).getAsDouble(), position.get(1).getAsDouble(), position.get(2).getAsDouble());
		String building = landmark(BlockPos.containing(feet), scenario);
		if (building != null || !scenario.has("places")) {
			return building;
		}
		String nearest = null;
		double distance = Double.POSITIVE_INFINITY;
		for (var entry : new TreeMap<>(scenario.getAsJsonObject("places").asMap()).entrySet()) {
			JsonArray anchor = entry.getValue().getAsJsonArray();
			double candidate = feet.distanceToSqr(anchor.get(0).getAsDouble(), anchor.get(1).getAsDouble(),
				anchor.get(2).getAsDouble());
			if (candidate < distance) {
				distance = candidate;
				nearest = entry.getKey();
			}
		}
		return nearest;
	}

	private static String itemName(String item) {
		return item.substring(item.indexOf(':') + 1).replace('_', ' ');
	}

	private static String location(JsonObject news) {
		return news.has("place") ? " at " + news.get("place").getAsString() : "";
	}

	/** Supplies old saved setups with live place anchors without mutating any persisted scenario or hash input. */
	private JsonObject runtimeScenario() {
		JsonObject scenario = state().getAsJsonObject("scenario");
		if (scenario != null && !scenario.has("places") && harness.villagers() != null
				&& (harness.config().nightSight() || !harness.config().visibleLoads().isEmpty()
					|| purchasesEnabled(harness.config().json()) || harness.config().truthfulNews() || harness.config().askAbout())) {
			return withPlaces(scenario, harness.villagers().places());
		}
		return scenario;
	}

	/** Returns a detached, augmented view only when a legacy scenario lacks its own place anchors. */
	static JsonObject withPlaces(JsonObject scenario, JsonObject places) {
		JsonObject result = scenario.deepCopy();
		if (!result.has("places")) {
			result.add("places", places.deepCopy());
		}
		return result;
	}

	/**
	 * Derives block news from its exact sight record and tick, not from the world's later participants. Villagers logs
	 * only bots and NPCs, distinguished by npcId; present names need LOS and feet within eight blocks of block center.
	 */
	static JsonObject observedItem(String kind, JsonElement subject, String witnessName,
			JsonObject observationRecord, JsonObject change) {
		JsonObject news = item("change:" + change.get("id").getAsLong(), kind, subject, witnessName,
			observationRecord.get("gameTime").getAsLong(), change.get("gameTime").getAsLong());
		Set<String> names = new TreeSet<>();
		Vec3 center = Vec3.atCenterOf(ControlCommands.position(change));
		for (JsonElement value : observationRecord.getAsJsonArray("entities")) {
			JsonObject entity = value.getAsJsonObject();
			if (!namedPerson(entity)) {
				continue;
			}
			JsonArray position = entity.getAsJsonArray("pos");
			Vec3 feet = new Vec3(position.get(0).getAsDouble(), position.get(1).getAsDouble(), position.get(2).getAsDouble());
			if (feet.distanceTo(center) <= 8.0) {
				names.add(entity.get("name").getAsString());
			}
		}
		JsonArray present = new JsonArray();
		names.forEach(present::add);
		news.add("present", present);
		return news;
	}

	/** Returns a detached relay while retaining the original presence list verbatim. */
	static JsonObject relay(JsonObject source, String speaker, long learnedAt) {
		JsonObject news = source.deepCopy();
		appendSpeaker(news, speaker);
		news.addProperty("firsthand", false);
		news.addProperty("learnedAt", learnedAt);
		return news;
	}

	/**
	 * Learns delivered relay metadata and direct, content-bearing shouts, never mere chatter.
	 * Shout subjects use the delivered speaker's feet position; their text remains a claim,
	 * not privileged attribution to an observed block change.
	 */
	void heard(String listenerKey, JsonObject heardItem) {
		if (!heardItem.has("text") || heardItem.get("text").getAsString().isBlank()) {
			return;
		}
		long learnedAt = time();
		if (heardItem.has("news")) {
			for (JsonElement value : heardItem.getAsJsonArray("news")) {
				JsonObject news = value.getAsJsonObject();
				if (!enabled(news.get("kind").getAsString())) {
					continue;
				}
				remember(listenerKey, relay(news, heardItem.get("speaker").getAsString(), learnedAt));
			}
		}
		if (enabled("shout") && heardItem.has("volume") && heardItem.get("volume").getAsString().equals("shout")) {
			JsonArray speakerPosition = heardItem.getAsJsonArray("speakerPos");
			JsonArray subject = new JsonArray();
			for (JsonElement coordinate : speakerPosition) {
				subject.add((int) Math.floor(coordinate.getAsDouble()));
			}
			String origin = "utt:" + heardItem.get("uttId").getAsLong();
			JsonObject news = item(origin, "shout", subject, harness.communication().name(listenerKey),
				learnedAt, heardItem.get("startedAt").getAsLong());
			news.add("text", heardItem.get("text").deepCopy());
			remember(listenerKey, news);
		}
	}

	/** Returns detached live items, hiding disabled opt-in kinds, by fixed salience then stable news ID. */
	JsonArray items(String npcKey, int limit) {
		JsonArray result = new JsonArray();
		JsonObject news = state().getAsJsonObject("news");
		if (limit <= 0 || news == null || !news.has(npcKey)) {
			return result;
		}
		List<JsonObject> ordered = new ArrayList<>();
		for (JsonElement value : news.getAsJsonObject(npcKey).asMap().values()) {
			JsonObject item = value.getAsJsonObject();
			String kind = item.get("kind").getAsString();
			if (live(item, time()) && (DEFAULT_KINDS.contains(kind) || enabled(kind))) {
				ordered.add(item);
			}
		}
		ordered.sort(Comparator.comparingDouble((JsonObject item) -> salience(item)).reversed()
			.thenComparing(item -> item.get("newsId").getAsString()));
		for (int index = 0; index < Math.min(limit, ordered.size()); index++) {
			result.add(ordered.get(index).deepCopy());
		}
		return result;
	}

	/** Answers a targeted question only from this villager's live first- and second-hand memory. */
	JsonArray items(String npcKey, int limit, String about) {
		return matching(items(npcKey, Integer.MAX_VALUE), about, runtimeScenario(), limit);
	}

	/** Selects detached reports by occurrence time, with stable source IDs breaking ties rather than salience. */
	static JsonArray matching(JsonArray news, String about, JsonObject scenario, int limit) {
		List<JsonObject> ordered = new ArrayList<>();
		for (JsonElement value : news) {
			JsonObject item = value.getAsJsonObject();
			if (matches(item, about, scenario)) {
				ordered.add(item);
			}
		}
		ordered.sort(Comparator.comparingLong((JsonObject item) -> originatedAt(item)).reversed()
			.thenComparing(item -> item.get("newsId").getAsString()));
		JsonArray result = new JsonArray();
		for (int index = 0; index < Math.min(limit, ordered.size()); index++) {
			result.add(ordered.get(index).deepCopy());
		}
		return result;
	}

	/** Matches names exactly, ignoring case, never unrelated text or another participant's private knowledge. */
	private static boolean matches(JsonObject item, String about, JsonObject scenario) {
		for (String field : List.of("subject", "place", "landmark")) {
			JsonElement value = item.get(field);
			if (value != null && value.isJsonPrimitive() && value.getAsString().equalsIgnoreCase(about)) {
				return true;
			}
		}
		if (item.has("present")) {
			for (JsonElement person : item.getAsJsonArray("present")) {
				if (person.getAsString().equalsIgnoreCase(about)) {
					return true;
				}
			}
		}
		JsonElement position = item.has("pos") ? item.get("pos") : item.get("subject");
		if (scenario != null && position != null && position.isJsonArray()) {
			String name = place(position.getAsJsonArray(), scenario);
			return name != null && name.equalsIgnoreCase(about);
		}
		return false;
	}

	/** Old saved items predate occurrence timestamps, so their original learning tick is the available fallback. */
	static long originatedAt(JsonObject news) {
		return news.get(news.has("originatedAt") ? "originatedAt" : "learnedAt").getAsLong();
	}

	/**
	 * Expires old knowledge and updates directed encounters without ever delaying NPC schedules.
	 * Busy speakers may wait to try; once a trial is made, successful or not, that directed
	 * encounter is spent until either participant leaves dwell or their shared talk radius.
	 */
	void tick() {
		forget();
		JsonObject scenario = state().getAsJsonObject("scenario");
		Villagers villagers = harness.villagers();
		if (scenario == null || villagers == null) {
			return;
		}
		Communication communication = harness.communication();
		TreeMap<String, Entity> participants = new TreeMap<>(communication.participants());
		Set<String> dwelling = new TreeSet<>();
		participants.keySet().stream().filter(communication::dwelling).forEach(dwelling::add);
		JsonObject previous = section("encounters");
		JsonObject encounters = new JsonObject();
		double radius = harness.config().number("talk_radius");
		long tick = time();
		for (var speaker : villagers.bodies().entrySet()) {
			String npcKey = speaker.getKey();
			if (!dwelling.contains(npcKey)) {
				continue;
			}
			JsonObject partners = new JsonObject();
			JsonObject priorPartners = previous.getAsJsonObject(npcKey);
			for (var partner : participants.entrySet()) {
				String partnerKey = partner.getKey();
				if (npcKey.equals(partnerKey) || !dwelling.contains(partnerKey)
						|| speaker.getValue().level() != partner.getValue().level()
						|| speaker.getValue().getEyePosition().distanceToSqr(partner.getValue().getEyePosition()) > radius * radius
						|| !samePlace(npcKey, partnerKey)) {
					continue;
				}
				JsonObject encounter = priorPartners == null ? null : priorPartners.getAsJsonObject(partnerKey);
				if (encounter == null) {
					encounter = new JsonObject();
					encounter.addProperty("startedAt", tick);
				}
				partners.add(partnerKey, encounter);
				if (encounter.has("attemptedAt") || communication.busy(npcKey)) {
					continue;
				}
				JsonObject news = unshared(npcKey, partnerKey);
				if (news == null) {
					continue;
				}
				String newsId = news.get("newsId").getAsString();
				encounter.addProperty("attemptedAt", tick);
				encounter.addProperty("newsId", newsId);
				double draw = Determinism.relayDraw(scenario.get("base_seed").getAsLong(), npcKey.substring(4),
					partnerKey, newsId, tick / DAY_TICKS);
				if (draw < salience(news) * harness.config().number("relay_rate")) {
					appendSpeaker(news, communication.name(npcKey));
					JsonArray relayed = new JsonArray();
					relayed.add(news);
					if (communication.startNpc(npcKey, summary(news), relayed, communication.name(partnerKey)) != null) {
						object(object(section("shared"), npcKey), partnerKey).addProperty(newsId, tick);
					}
				}
			}
			if (!partners.isEmpty()) {
				encounters.add(npcKey, partners);
			}
		}
		state().add("encounters", encounters);
	}

	/**
	 * Villagers gossip only with people at the same place: the same household, workplace, school or market. A
	 * villager and a bot gossip whenever both are dwelling within talk radius, since bots have no scheduled places.
	 * Without this, neighbours in separate houses within talk radius chattered through walls and news saturated a
	 * small town within seconds.
	 */
	private boolean samePlace(String npcKey, String partnerKey) {
		if (!partnerKey.startsWith("npc:")) {
			return true;
		}
		String place = harness.villagers().dwellingPlace(npcKey);
		return place != null && place.equals(harness.villagers().dwellingPlace(partnerKey));
	}

	/** Preserves the legacy default scenario, adding place anchors only for opted-in mechanics. */
	static JsonObject scenario(JsonObject args, boolean includePlaces) {
		JsonObject result = new JsonObject();
		Set<String> kinds = new TreeSet<>(DEFAULT_KINDS);
		if (args.has("news_kinds")) {
			kinds.clear();
			for (JsonElement value : args.getAsJsonArray("news_kinds")) {
				String kind = value.getAsString();
				if (!SALIENCE.containsKey(kind)) {
					throw new IllegalArgumentException("Unknown news kind: " + kind);
				}
				kinds.add(kind);
			}
		}
		JsonArray enabled = new JsonArray();
		kinds.forEach(enabled::add);
		result.add("news_kinds", enabled);
		if (includePlaces && args.has("news_kinds") && kinds.equals(DEFAULT_KINDS)) {
			result.addProperty("news_kinds_explicit", true);
		}
		result.add("region", args.getAsJsonObject("region").deepCopy());
		JsonArray surfaces = new JsonArray();
		if (args.has("surfaces")) {
			Set<String> names = new TreeSet<>();
			args.getAsJsonArray("surfaces").forEach(value -> names.add(value.getAsString()));
			names.forEach(surfaces::add);
		} else {
			new TreeSet<>(DEFAULT_SURFACES).forEach(surfaces::add);
		}
		result.add("surfaces", surfaces);
		TreeMap<String, JsonObject> landmarks = new TreeMap<>();
		if (args.has("landmarks")) {
			for (JsonElement value : args.getAsJsonArray("landmarks")) {
				JsonObject landmark = value.getAsJsonObject();
				String name = landmark.get("name").getAsString();
				JsonObject region = landmark.getAsJsonObject("region");
				JsonArray minimum = region.getAsJsonArray("min");
				JsonArray maximum = region.getAsJsonArray("max");
				if (name.isBlank() || landmarks.containsKey(name) || minimum.size() != 3 || maximum.size() != 3) {
					throw new IllegalArgumentException("Landmarks need unique nonempty names and three-coordinate min/max bounds");
				}
				for (int index = 0; index < 3; index++) {
					int lower = minimum.get(index).getAsBigDecimal().intValueExact();
					int upper = maximum.get(index).getAsBigDecimal().intValueExact();
					if (lower > upper) {
						throw new IllegalArgumentException("Landmark minimum bounds must not exceed maximum bounds");
					}
				}
				JsonObject normalized = new JsonObject();
				normalized.addProperty("name", name);
				JsonObject bounds = new JsonObject();
				bounds.add("min", minimum.deepCopy());
				bounds.add("max", maximum.deepCopy());
				normalized.add("region", bounds);
				landmarks.put(name, normalized);
			}
		}
		JsonArray regions = new JsonArray();
		landmarks.values().forEach(regions::add);
		result.add("landmarks", regions);
		if (includePlaces && args.has("places")) {
			result.add("places", args.getAsJsonObject("places").deepCopy());
		}
		return result;
	}

	private boolean roadBlocked(JsonObject change, BlockPos position, JsonObject scenario) {
		String block = change.get("to").getAsString();
		if (!block.equals("minecraft:barrier") && !block.equals("minecraft:iron_bars")
				&& !block.endsWith("_fence") && !block.endsWith("_wall")) {
			return false;
		}
		BlockPos surface = position.below();
		if (!Region.fromJson(scenario.getAsJsonObject("region")).contains(surface.getX(), surface.getY(), surface.getZ())
				|| !server.overworld().hasChunkAt(surface)) {
			return false;
		}
		String below = HarnessState.blockId(server.overworld().getBlockState(surface));
		for (JsonElement value : scenario.getAsJsonArray("surfaces")) {
			if (value.getAsString().equals(below)) {
				return true;
			}
		}
		return false;
	}

	private static String landmark(BlockPos position, JsonObject scenario) {
		int[] coordinates = {position.getX(), position.getY(), position.getZ()};
		for (JsonElement value : scenario.getAsJsonArray("landmarks")) {
			JsonObject landmark = value.getAsJsonObject();
			JsonObject region = landmark.getAsJsonObject("region");
			boolean inside = true;
			for (int index = 0; index < 3; index++) {
				inside &= coordinates[index] >= region.getAsJsonArray("min").get(index).getAsInt()
					&& coordinates[index] <= region.getAsJsonArray("max").get(index).getAsInt();
			}
			if (inside) {
				return landmark.get("name").getAsString();
			}
		}
		return null;
	}

	private static JsonObject item(String origin, String kind, JsonElement subject, String witnessName,
			long learnedAt, long originatedAt) {
		JsonObject result = new JsonObject();
		result.addProperty("newsId", origin);
		result.addProperty("kind", kind);
		result.add("subject", subject.deepCopy());
		result.addProperty("originEventId", origin);
		JsonArray chain = new JsonArray();
		chain.add(witnessName);
		result.add("chain", chain);
		result.addProperty("firsthand", true);
		result.addProperty("learnedAt", learnedAt);
		result.addProperty("originatedAt", originatedAt);
		result.addProperty("salience", SALIENCE.get(kind));
		return result;
	}

	/** Rehearing a live source never refreshes its age or overwrites a first-hand witness. */
	private void remember(String listenerKey, JsonObject news) {
		if (!live(news, time())) {
			return;
		}
		JsonObject known = object(section("news"), listenerKey);
		String newsId = news.get("newsId").getAsString();
		JsonObject previous = known.getAsJsonObject(newsId);
		if (previous != null && live(previous, time())) {
			if (previous.get("firsthand").getAsBoolean() || !news.get("firsthand").getAsBoolean()) {
				return;
			}
			news.addProperty("learnedAt", Math.min(previous.get("learnedAt").getAsLong(), news.get("learnedAt").getAsLong()));
		}
		known.add(newsId, news);
	}

	private JsonObject unshared(String npcKey, String partnerKey) {
		JsonObject shared = section("shared").getAsJsonObject(npcKey);
		JsonObject partner = shared == null ? null : shared.getAsJsonObject(partnerKey);
		for (JsonElement value : items(npcKey, Integer.MAX_VALUE)) {
			JsonObject news = value.getAsJsonObject();
			if (partner == null || !partner.has(news.get("newsId").getAsString())) {
				return news;
			}
		}
		return null;
	}

	private void forget() {
		long tick = time();
		JsonObject news = section("news");
		for (JsonElement value : news.asMap().values()) {
			value.getAsJsonObject().entrySet().removeIf(entry -> !live(entry.getValue().getAsJsonObject(), tick));
		}
		news.entrySet().removeIf(entry -> entry.getValue().getAsJsonObject().isEmpty());
		JsonObject shared = section("shared");
		for (var speaker : shared.entrySet()) {
			JsonObject known = news.getAsJsonObject(speaker.getKey());
			JsonObject partners = speaker.getValue().getAsJsonObject();
			for (JsonElement value : partners.asMap().values()) {
				value.getAsJsonObject().entrySet().removeIf(entry -> known == null || !known.has(entry.getKey()));
			}
			partners.entrySet().removeIf(entry -> entry.getValue().getAsJsonObject().isEmpty());
		}
		shared.entrySet().removeIf(entry -> entry.getValue().getAsJsonObject().isEmpty());
	}

	private boolean enabled(String kind) {
		return enabled(kind, state().getAsJsonObject("scenario"), harness.config().nightSight(),
			!harness.config().visibleLoads().isEmpty(), kind.equals("sold_out") && purchasesEnabled(harness.config().json()),
			harness.config().truthfulNews());
	}

	/** Applies feature gates without replacing the four stored defaults or weakening an explicit kind filter. */
	static boolean enabled(String kind, JsonObject scenario, boolean nightSight, boolean visibleLoads,
			boolean villagerPurchases) {
		return enabled(kind, scenario, nightSight, visibleLoads, villagerPurchases, false);
	}

	/** New change reports require their opt-in gate as well as any explicit scenario kind filter. */
	static boolean enabled(String kind, JsonObject scenario, boolean nightSight, boolean visibleLoads,
			boolean villagerPurchases, boolean truthfulNews) {
		boolean admitted = switch (kind) {
			case "change" -> truthfulNews;
			case "load" -> visibleLoads;
			case "night", "decoration" -> nightSight;
			case "sold_out" -> villagerPurchases;
			default -> DEFAULT_KINDS.contains(kind);
		};
		if (!admitted) {
			return false;
		}
		if (scenario == null) {
			return true;
		}
		Set<String> kinds = new TreeSet<>();
		for (JsonElement value : scenario.getAsJsonArray("news_kinds")) {
			if (value.getAsString().equals(kind)) {
				return true;
			}
			kinds.add(value.getAsString());
		}
		return !DEFAULT_KINDS.contains(kind) && !scenario.has("news_kinds_explicit") && kinds.equals(DEFAULT_KINDS);
	}

	/** Purchase failures are news only while scripted market purchases are opted in. */
	static boolean purchasesEnabled(JsonObject config) {
		return config.has("villager_purchases") && !config.getAsJsonArray("villager_purchases").isEmpty();
	}

	private boolean live(JsonObject news, long tick) {
		long origin = news.get(news.has("originatedAt") ? "originatedAt" : "learnedAt").getAsLong();
		long ttl = harness.config().integer("news_ttl_days") * DAY_TICKS;
		return tick - origin <= ttl && tick - news.get("learnedAt").getAsLong() <= ttl;
	}

	private static double salience(JsonObject news) {
		return SALIENCE.get(news.get("kind").getAsString());
	}

	private static void appendSpeaker(JsonObject news, String speaker) {
		JsonArray chain = news.getAsJsonArray("chain");
		if (chain.isEmpty() || !chain.get(chain.size() - 1).getAsString().equals(speaker)) {
			chain.add(speaker);
		}
	}

	private static String summary(JsonObject news) {
		if (news.get("kind").getAsString().equals("change")) {
			JsonArray items = new JsonArray();
			items.add(news);
			return Communication.answer(items, true);
		}
		if (news.has("text")) {
			return news.get("text").getAsString();
		}
		JsonElement subject = news.get("subject");
		return news.get("kind").getAsString().replace('_', ' ') + " at "
			+ (subject.isJsonPrimitive() ? subject.getAsString() : subject.toString()) + ".";
	}

	private JsonObject state() {
		return harness.map("communication");
	}

	private JsonObject section(String key) {
		return object(state(), key);
	}

	private static JsonObject object(JsonObject parent, String key) {
		if (!parent.has(key)) {
			parent.add(key, new JsonObject());
		}
		return parent.getAsJsonObject(key);
	}

	private long time() {
		return server.overworld().getGameTime();
	}
}
