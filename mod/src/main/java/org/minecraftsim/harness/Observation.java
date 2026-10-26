package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.phys.Vec3;

/**
 * Builds bot observations, proximity speech queues and exposure audits on the server thread.
 * Sight probes use loaded chunks only; responses retain deterministic distance/coordinate/identity
 * ordering and game-time cursors, and observing consumes heard queues only after writing the audit.
 */
public final class Observation {
	private final MinecraftServer server;
	private final BotManager bots;
	private final Documents documents;

	/**
	 * Shares existing server, bot and document services without performing an observation.
	 * @param server owning server
	 * @param bots manager supplying speech listeners
	 * @param documents service resolving document content and provenance
	 */
	public Observation(MinecraftServer server, BotManager bots, Documents documents) {
		this.server = server;
		this.bots = bots;
		this.documents = documents;
	}

	/**
	 * Collects admitted self/entity/block/document/change/heard data and advances the observer cursor.
	 * @param observer overworld bot to observe from on the server thread
	 * @param includeTerrain whether to include the configured terrain probes
	 * @return ordered protocol observation, including an empty terrain array when disabled
	 * @throws IllegalArgumentException if the bot is outside the overworld or document access is invalid
	 * @throws java.io.UncheckedIOException if provenance or exposure logging fails
	 */
	public JsonObject observe(SimBot observer, boolean includeTerrain) {
		requireOverworld(observer);
		HarnessState state = HarnessState.get(server);
		SimConfig config = state.config();
		Vec3 eye = observer.getEyePosition();
		String name = observer.getGameProfile().name();
		JsonObject response = new JsonObject();
		response.addProperty("gameTime", server.overworld().getGameTime());
		JsonObject self = self(observer);
		response.add("self", self);
		JsonArray exposures = new JsonArray();
		JsonObject intrinsic = self.deepCopy();
		intrinsic.addProperty("category", "self");
		intrinsic.addProperty("ownedBy", name);
		intrinsic.addProperty("distance", 0);
		intrinsic.addProperty("inRadius", true);
		intrinsic.addProperty("reachOk", true);
		intrinsic.add("losRay", JsonNull.INSTANCE);
		exposures.add(intrinsic);
		documents.appendInventoryExposures(observer, self.getAsJsonArray("inventory"), exposures);
		List<JsonObject> entities = new ArrayList<>();
		List<Entity> candidates = new ArrayList<>();
		observer.level().getAllEntities().forEach(candidates::add);
		candidates.sort(Comparator.comparingDouble((Entity entity) -> eye.distanceTo(entity.position()))
			.thenComparing(entity -> entity.getUUID().toString()));
		for (Entity entity : candidates) {
			if (entity == observer || entity.isRemoved()) {
				continue;
			}
			double distance = eye.distanceTo(entity.position());
			if (distance > config.number("view_radius")) {
				continue;
			}
			Sight.Ray ray = Sight.entity(observer, entity);
			if (ray == null) {
				continue;
			}
			JsonObject item = new JsonObject();
			String uuid = entity.getUUID().toString();
			item.addProperty("prov", "entity:" + uuid);
			item.addProperty("uuid", uuid);
			item.addProperty("type", entity.typeHolder().getRegisteredName());
			if (entity instanceof SimBot bot) {
				item.addProperty("name", bot.getGameProfile().name());
				item.addProperty("heldItem", bot.getMainHandItem().isEmpty() ? null
					: BuiltInRegistries.ITEM.getKey(bot.getMainHandItem().getItem()).toString());
				JsonObject activity = bot.skills().activity();
				if (activity != null && activity.has("pos")) {
					BlockPos target = position(activity.getAsJsonArray("pos"));
					if (eye.distanceTo(Vec3.atCenterOf(target)) > config.number("view_radius") || Sight.block(observer, target) == null) {
						activity.remove("pos");
					}
				}
				item.add("activity", activity == null ? JsonNull.INSTANCE : activity);
			} else if (entity.hasCustomName()) {
				item.addProperty("name", entity.getCustomName().getString());
			}
			if (state.villagers() != null) {
				String npcId = state.villagers().id(entity);
				if (npcId != null) {
					item.addProperty("npcId", npcId);
				}
			}
			JsonObject load = VisibleLoads.seen(entity, config);
			if (load != null) {
				item.add("load", load);
			}
			item.add("pos", HarnessState.vector(entity.position()));
			item.addProperty("distance", distance);
			state.origin(item.get("prov").getAsString(), "entity", item);
			entities.add(item);
			JsonObject audit = admitted(item, "entity", config.number("view_radius"), ray, false);
			audit.add("targetEye", HarnessState.vector(entity.getEyePosition()));
			audit.add("targetCenter", HarnessState.vector(entity.getBoundingBox().getCenter()));
			exposures.add(audit);
		}
		List<JsonObject> blocks = new ArrayList<>();
		List<JsonObject> readable = new ArrayList<>();
		double radius = Math.max(config.number("view_radius"), config.number("reach"));
		Set<String> notable = config.notable();
		ServerLevel level = observer.level();
		int lowY = Math.max(level.getMinY(), (int) Math.floor(eye.y - radius));
		int highY = Math.min(level.getMaxY(), (int) Math.floor(eye.y + radius));
		for (int x = (int) Math.floor(eye.x - radius); x <= Math.floor(eye.x + radius); x++) {
			for (int z = (int) Math.floor(eye.z - radius); z <= Math.floor(eye.z + radius); z++) {
				if (!level.hasChunk(Math.floorDiv(x, 16), Math.floorDiv(z, 16))) {
					continue;
				}
				for (int y = lowY; y <= highY; y++) {
					BlockPos position = new BlockPos(x, y, z);
					double distance = eye.distanceTo(Vec3.atCenterOf(position));
					if (distance > radius) {
						continue;
					}
					var block = level.getBlockState(position);
					if (block.isAir()) {
						continue;
					}
					String blockId = HarnessState.blockId(block);
					boolean reportBlock = distance <= config.number("view_radius") && notable.contains(blockId);
					boolean readDocument = distance <= config.number("reach") && block.hasBlockEntity();
					if (!reportBlock && !readDocument) {
						continue;
					}
					Sight.Ray ray = Sight.block(observer, position);
					if (ray == null) {
						continue;
					}
					if (reportBlock) {
						String key = HarnessState.key(position);
						JsonElement change = state.map("latestChanges").get(key);
						JsonObject item = new JsonObject();
						item.addProperty("prov", "block:" + key + "@" + (change == null ? "base" : change.getAsString()));
						item.add("pos", HarnessState.position(position));
						item.addProperty("block", blockId);
						item.addProperty("distance", distance);
						describeOpenState(item, block);
						if (change == null) {
							state.origin(item.get("prov").getAsString(), "block", item);
						}
						blocks.add(item);
						exposures.add(admitted(item, "block", config.number("view_radius"), ray, false));
					}
					if (readDocument) {
						JsonObject item = documents.document(observer, position);
						if (item != null) {
							item.addProperty("distance", distance);
							readable.add(item);
							exposures.add(admitted(item, "document", config.number("reach"), ray, true));
						}
					}
				}
			}
		}
		List<JsonObject> terrain = includeTerrain ? terrain(observer, exposures) : new ArrayList<>();
		for (String category : List.of("heard", "chatter", "shout_seen")) {
			List<JsonObject> items = new ArrayList<>();
			for (JsonElement queued : state.communication().queued(category, name)) {
				JsonObject utterance = queued.getAsJsonObject();
				JsonObject item = new JsonObject();
				for (String key : List.of("prov", "speaker", "gameTime", "volume", "text", "distance", "to",
						"firsthand", "chain", "originEventId", "newsId", "kind", "subject", "news", "sense", "pos")) {
					if (utterance.has(key) && (!category.equals("shout_seen") || List.of("prov", "speaker", "gameTime").contains(key))) {
						item.add(key, utterance.get(key).deepCopy());
					}
				}
				items.add(item);
				JsonObject audit = utterance.deepCopy();
				audit.addProperty("category", category);
				audit.addProperty("inRadius", utterance.get("inStart").getAsBoolean() || utterance.get("inEnd").getAsBoolean());
				if (!audit.has("losRay")) {
					audit.add("losRay", JsonNull.INSTANCE);
				}
				exposures.add(audit);
			}
			response.add(category, ordered(items));
		}
		List<JsonObject> changes = new ArrayList<>();
		long cursor = state.map("observeCursor").has(name) ? state.map("observeCursor").get(name).getAsLong() : 0;
		for (JsonElement element : state.changes()) {
			JsonObject event = element.getAsJsonObject();
			if (event.get("id").getAsLong() <= cursor) {
				continue;
			}
			BlockPos position = position(event.getAsJsonArray("pos"));
			double distance = eye.distanceTo(Vec3.atCenterOf(position));
			if (distance > config.number("view_radius")) {
				continue;
			}
			Sight.Ray ray = Sight.block(observer, position);
			if (ray == null) {
				continue;
			}
			JsonObject item = new JsonObject();
			item.addProperty("prov", "change:" + event.get("id").getAsLong());
			for (String key : new String[]{"pos", "from", "to", "gameTime"}) {
				item.add(key, event.get(key).deepCopy());
			}
			JsonObject audit = item.deepCopy();
			audit.addProperty("distance", distance);
			exposures.add(admitted(audit, "change", config.number("view_radius"), ray, false));
			item.addProperty("distance", distance);
			changes.add(item);
		}
		List<JsonObject> structures = new ArrayList<>();
		for (Structures.Seen seen : state.structures().seen(observer, config.number("view_radius"))) {
			JsonObject item = seen.item();
			state.origin(item.get("prov").getAsString(), "structure", item);
			JsonObject audit = item.deepCopy();
			audit.add("pos", HarnessState.position(seen.target()));
			audit.addProperty("distance", seen.targetDistance());
			exposures.add(admitted(audit, "structure", config.number("view_radius"), seen.ray(), false));
			structures.add(item);
		}
		response.add("entities", ordered(entities));
		response.add("blocks", ordered(blocks));
		response.add("structures", ordered(structures));
		JsonArray orderedTerrain = ordered(terrain);
		orderedTerrain.forEach(element -> element.getAsJsonObject().remove("distance"));
		response.add("terrain", orderedTerrain);
		response.add("documents", ordered(readable));
		JsonArray orderedChanges = ordered(changes);
		orderedChanges.forEach(element -> element.getAsJsonObject().remove("distance"));
		response.add("changes", orderedChanges);
		JsonArray seenActions = new JsonArray();
		for (JsonElement element : state.seenActions().queued(name)) {
			JsonObject audit = element.getAsJsonObject();
			JsonObject item = new JsonObject();
			for (String key : List.of("prov", "gameTime", "actor", "action", "pos", "target", "item")) {
				if (audit.has(key)) {
					item.add(key, audit.get(key).deepCopy());
				}
			}
			seenActions.add(item);
			exposures.add(audit.deepCopy());
		}
		response.add("seen_actions", seenActions);
		state.exposure(name, eye, "observe", ordered(exposures.asList().stream().map(JsonElement::getAsJsonObject).toList()));
		state.seenActions().clear(name);
		for (String category : List.of("heard", "chatter", "shout_seen")) {
			state.communication().clear(category, name);
		}
		state.map("lastObserve").addProperty(name, level.getGameTime());
		state.map("observeCursor").addProperty(name, state.counter("eventCounter"));
		return response;
	}

	/** Appends the visible open state of doors, trapdoors and gates without describing unseen blocks. */
	static void describeOpenState(JsonObject item, BlockState block) {
		if (block.getBlock() instanceof DoorBlock || block.getBlock() instanceof TrapDoorBlock
				|| block.getBlock() instanceof FenceGateBlock) {
			JsonObject properties = new JsonObject();
			properties.addProperty("open", block.getValue(BlockStateProperties.OPEN));
			item.add("properties", properties);
		}
	}

	private JsonObject self(SimBot bot) {
		JsonObject self = new JsonObject();
		self.addProperty("name", bot.getGameProfile().name());
		self.add("pos", HarnessState.vector(bot.position()));
		self.addProperty("yaw", bot.getYRot());
		self.addProperty("pitch", bot.getXRot());
		self.addProperty("health", bot.getHealth());
		self.addProperty("food", bot.getFoodData().getFoodLevel());
		if (bot.isInWater()) {
			self.addProperty("in_water", true);
		}
		if (HarnessState.get(server).config().dailyHunger()) {
			self.addProperty("hungry", bot.getFoodData().getFoodLevel() < 6);
		}
		self.addProperty("selectedSlot", bot.getInventory().getSelectedSlot());
		self.add("skill", bot.skills().status());
		JsonArray inventory = new JsonArray();
		for (int slot = 0; slot < bot.getInventory().getContainerSize(); slot++) {
			var stack = bot.getInventory().getItem(slot);
			if (!stack.isEmpty()) {
				inventory.add(documents.inventoryItem(bot, slot, stack));
			}
		}
		self.add("inventory", inventory);
		return self;
	}

	private List<JsonObject> terrain(SimBot observer, JsonArray exposures) {
		List<JsonObject> result = new ArrayList<>();
		SimConfig config = HarnessState.get(server).config();
		int radius = config.integer("terrain_radius");
		int step = config.integer("terrain_step");
		Vec3 eye = observer.getEyePosition();
		ServerLevel level = observer.level();
		int minX = Math.floorDiv((int) Math.floor(eye.x - radius), step) * step;
		int minZ = Math.floorDiv((int) Math.floor(eye.z - radius), step) * step;
		for (int x = minX; x <= eye.x + radius; x += step) {
			for (int z = minZ; z <= eye.z + radius; z += step) {
				double distance = Math.hypot(x + 0.5 - eye.x, z + 0.5 - eye.z);
				if (distance > radius || !level.hasChunk(Math.floorDiv(x, 16), Math.floorDiv(z, 16))) {
					continue;
				}
				for (int y = Math.min(level.getMaxY(), (int) Math.floor(eye.y + 4)); y >= level.getMinY(); y--) {
					BlockPos position = new BlockPos(x, y, z);
					var block = level.getBlockState(position);
					if (block.isAir()) {
						continue;
					}
					Sight.Ray ray = Sight.block(observer, position);
					if (ray != null) {
						JsonObject item = new JsonObject();
						item.addProperty("x", x);
						item.addProperty("z", z);
						item.addProperty("y", y);
						item.addProperty("block", HarnessState.blockId(block));
						item.addProperty("distance", distance);
						result.add(item);
						JsonObject audit = item.deepCopy();
						audit.add("pos", HarnessState.position(position));
						exposures.add(admitted(audit, "terrain", radius, ray, false));
					}
					break;
				}
			}
		}
		return result;
	}

	/**
	 * Starts the timed talk-volume speak skill through the legacy alias.
	 * @param speaker overworld speaker on the server thread
	 * @param text speech text, at most 256 UTF-16 code units
	 * @return skill acceptance and utterance ID; delivery happens only at completion
	 * @throws IllegalArgumentException if the speaker is outside the overworld or text is too long
	 * @throws java.io.UncheckedIOException if utterance logging fails
	 */
	public JsonObject say(SimBot speaker, String text) {
		requireOverworld(speaker);
		JsonObject args = new JsonObject();
		args.addProperty("text", text);
		args.addProperty("volume", "talk");
		return speaker.skills().start("speak", args, false);
	}

	/**
	 * Copies an already admitted item and attaches its established sight/radius audit fields.
	 * @param item response item to copy
	 * @param category exposure category
	 * @param radius admission radius used by the caller
	 * @param ray successful sight evidence
	 * @param reach whether to include a successful reach flag
	 * @return annotated deep copy; this helper does not repeat admission checks
	 */
	public static JsonObject admitted(JsonObject item, String category, double radius, Sight.Ray ray, boolean reach) {
		JsonObject audit = item.deepCopy();
		audit.addProperty("category", category);
		audit.addProperty("radius", radius);
		audit.addProperty("inRadius", true);
		if (reach) {
			audit.addProperty("reachOk", true);
		}
		ray.audit(audit);
		return audit;
	}

	private static JsonArray ordered(List<JsonObject> items) {
		List<JsonObject> sorted = new ArrayList<>(items);
		sorted.sort(Comparator.comparingDouble((JsonObject item) -> item.has("distance") ? item.get("distance").getAsDouble() : 0)
			.thenComparingDouble(item -> coordinate(item, 0))
			.thenComparingDouble(item -> coordinate(item, 1))
			.thenComparingDouble(item -> coordinate(item, 2))
			.thenComparing(item -> item.has("uuid") ? item.get("uuid").getAsString() : item.has("prov") ? item.get("prov").getAsString() : ""));
		JsonArray result = new JsonArray();
		sorted.forEach(result::add);
		return result;
	}

	private static double coordinate(JsonObject item, int axis) {
		if (item.has("uuid")) {
			return 0;
		}
		if (item.has("pos")) {
			return item.getAsJsonArray("pos").get(axis).getAsDouble();
		}
		String key = new String[]{"x", "y", "z"}[axis];
		return item.has(key) ? item.get(key).getAsDouble() : 0;
	}

	private static BlockPos position(JsonArray position) {
		return new BlockPos(position.get(0).getAsInt(), position.get(1).getAsInt(), position.get(2).getAsInt());
	}

	private void requireOverworld(SimBot bot) {
		if (bot.level() != server.overworld()) {
			throw new IllegalArgumentException("Observation requires a bot in the overworld");
		}
	}
}
