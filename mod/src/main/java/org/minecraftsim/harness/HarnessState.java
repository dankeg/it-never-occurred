package org.minecraftsim.harness;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.function.Supplier;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.storage.LevelResource;
import net.minecraft.world.phys.Vec3;

/**
 * Owns per-server canonical harness data, configuration, communication, ferry, NPCs and audit writers.
 * Simulation callers use the server thread; counters and timestamps follow game time,
 * canonical object keys are sorted, and array order is retained for hashing and persistence.
 */
public final class HarnessState implements AutoCloseable {
	private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().serializeNulls().create();
	private static final Map<MinecraftServer, HarnessState> STATES = new WeakHashMap<>();
	private static final ThreadLocal<String> ACTOR = new ThreadLocal<>();
	private final MinecraftServer server;
	private final Path directory;
	private JsonObject data = new JsonObject();
	private SimConfig config = new SimConfig();
	private BufferedWriter events;
	private BufferedWriter observations;
	private Villagers villagers;
	private final Communication communication = new Communication(this);
	private final SeenActions seenActions = new SeenActions(this);
	private final Ferry ferry = new Ferry(this);
	private final Structures structures = new Structures(this);
	private final BlockCollapse collapse = new BlockCollapse(this);
	private final DailyHunger hunger = new DailyHunger(this);
	private final VillagerNews news;
	private final NightSight nightSight;

	private HarnessState(MinecraftServer server) {
		this.server = server;
		news = new VillagerNews(server, this);
		nightSight = new NightSight(server);
		directory = server.getWorldPath(LevelResource.ROOT).resolve("simharness");
		data.addProperty("eventCounter", 0L);
		data.addProperty("utteranceCounter", 0L);
		data.addProperty("recording", false);
		for (String key : new String[]{"heard", "lastObserve", "observeCursor", "latestChanges", "docVersions", "containers", "acls", "botFlags", "origins", "externalDocs"}) {
			data.add(key, new JsonObject());
		}
		data.add("changes", new JsonArray());
		data.add("communication", new JsonObject());
	}

	/** Returns the server-thread speech controller backed by canonical harness state. */
	Communication communication() {
		return communication;
	}

	/** Returns completion-time sight evidence backed by optional canonical harness state. */
	SeenActions seenActions() {
		return seenActions;
	}

	/** Returns the ferry controller backed by optional canonical harness state. */
	Ferry ferry() {
		return ferry;
	}

	/** Returns the known-structures controller backed by optional canonical harness state. */
	Structures structures() {
		return structures;
	}

	/** Returns the support-collapse controller without creating optional fork state. */
	BlockCollapse collapse() {
		return collapse;
	}

	/** Returns the opt-in daily food controller without creating saved state when disabled. */
	DailyHunger hunger() {
		return hunger;
	}

	/** Returns the shared transient loaded-block emission cache, never serialized or hashed. */
	NightSight nightSight() {
		return nightSight;
	}

	/** Invalidates opt-in emission reads after vanilla edits even when event recording and NPCs are disabled. */
	public void invalidateNightSight() {
		if (config.nightSight()) {
			nightSight.invalidate();
		}
	}

	/** Whether vanilla hooks must retain support transitions for a configured collapse rule. */
	public boolean collapseConfigured() {
		return collapse.configured();
	}

	/** Returns the server-thread news controller backed by canonical harness state. */
	VillagerNews news() {
		return news;
	}

	/** Returns the owning server without creating or advancing simulation state. */
	MinecraftServer server() {
		return server;
	}

	/**
	 * Looks up or initializes the state associated with a server.
	 * @param server owning server, accessed on its main thread
	 * @return existing or newly initialized state
	 */
	public static HarnessState get(MinecraftServer server) {
		return STATES.computeIfAbsent(server, HarnessState::new);
	}

	/**
	 * Looks up state without initializing it during vanilla hooks.
	 * @param server owning server
	 * @return current state, or null before initialization or after removal
	 */
	public static HarnessState existing(MinecraftServer server) {
		return STATES.get(server);
	}

	/**
	 * Removes and closes a server's state when present.
	 * @param server server that has stopped
	 * @throws UncheckedIOException if closing audit writers fails
	 */
	public static void remove(MinecraftServer server) {
		HarnessState state = STATES.remove(server);
		if (state != null) {
			state.close();
		}
	}

	/**
	 * Provides the current validated configuration object.
	 * @return live configuration, replaced on load
	 */
	public SimConfig config() {
		return config;
	}

	/**
	 * Reads the currently configured NPC simulation.
	 * @return live NPC simulation, or null before setup
	 */
	public Villagers villagers() {
		return villagers;
	}

	/**
	 * Initializes NPCs once; repeated setup requests are ignored without revalidation.
	 * @param args NPC scenario setup arguments
	 * @return protocol object indicating whether setup was ignored
	 * @throws IllegalArgumentException if first-time scenario validation fails
	 */
	public JsonObject setupVillagers(JsonObject args) {
		boolean ignored = villagers != null;
		if (!ignored) {
			villagers = Villagers.setup(server, this, args);
		}
		JsonObject result = new JsonObject();
		result.addProperty("ignored", ignored);
		return result;
	}

	/** Collapses unsupported sections before NPC sight sampling, then advances ferry, speech and gossip. */
	public void tickVillagers() {
		hunger.tick();
		collapse.tick();
		ferry.tick();
		if (villagers != null) {
			villagers.tick();
		}
		communication.tick();
		if (villagers != null) {
			news.tick();
		}
	}

	/**
	 * Provides a live named JSON object in canonical state.
	 * @param key existing object-valued state key
	 * @return mutable stored object, or null for an absent key
	 * @throws ClassCastException if the stored value is not an object
	 */
	public JsonObject map(String key) {
		return data.getAsJsonObject(key);
	}

	/**
	 * Adds an empty named map on first use, for optional features (e.g. shops) whose absence keeps older state hashes.
	 * @param key map name, which must not already exist
	 */
	public void putMap(String key) {
		if (data.has(key)) {
			throw new IllegalStateException("State map already exists: " + key);
		}
		data.add(key, new JsonObject());
	}

	/**
	 * Removes an optional named map, restoring the state an unused feature leaves.
	 * @param key map name; absent keys are ignored
	 */
	public void removeMap(String key) {
		data.remove(key);
	}

	/**
	 * Provides the ordered block-change history used by observations.
	 * @return live mutable change array
	 */
	public JsonArray changes() {
		return data.getAsJsonArray("changes");
	}

	/**
	 * Reads the saved exposure-recording flag.
	 * @return current recording-mode flag
	 */
	public boolean recording() {
		return data.get("recording").getAsBoolean();
	}

	/**
	 * Updates the canonical recording flag without opening audit files.
	 * @param enabled requested recording mode
	 */
	public void recording(boolean enabled) {
		data.addProperty("recording", enabled);
	}

	/**
	 * Reads a saved numeric counter without incrementing it.
	 * @param key existing counter key
	 * @return current counter value
	 * @throws RuntimeException if the value is absent or not convertible
	 */
	public long counter(String key) {
		return data.get(key).getAsLong();
	}

	/**
	 * Increments and stores a counter using exact long arithmetic.
	 * @param key existing counter key
	 * @return updated counter value
	 * @throws ArithmeticException on long overflow, before storing a replacement
	 */
	public long next(String key) {
		long value = Math.incrementExact(counter(key));
		data.addProperty(key, value);
		return value;
	}

	/**
	 * Resolves or creates the saved heard-utterance queue for one bot.
	 * @param name bot name
	 * @return live mutable utterance array in delivery order
	 */
	public JsonArray heard(String name) {
		JsonObject queues = map("heard");
		if (!queues.has(name)) {
			queues.add(name, new JsonArray());
		}
		return queues.getAsJsonArray(name);
	}

	/**
	 * Runs an action in a thread-local provenance scope, restoring the prior scope in finally.
	 * @param <T> action result type
	 * @param actor bot name or system actor label
	 * @param action work to execute synchronously
	 * @return action result, unchanged
	 * @throws RuntimeException if the action fails; the previous actor is still restored
	 */
	public static <T> T acting(String actor, Supplier<T> action) {
		String previous = ACTOR.get();
		ACTOR.set(actor);
		try {
			return action.get();
		} finally {
			if (previous == null) {
				ACTOR.remove();
			} else {
				ACTOR.set(previous);
			}
		}
	}

	/**
	 * Allocates and logs a block-change event, then notifies configured NPCs.
	 * @param position changed block position
	 * @param from previous block state
	 * @param to replacement block state
	 * @throws UncheckedIOException if writing the event fails
	 */
	public void blockChange(BlockPos position, BlockState from, BlockState to) {
		nightSight.invalidate();
		JsonObject event = event("block_change");
		event.add("pos", position(position));
		event.addProperty("from", blockId(from));
		event.addProperty("to", blockId(to));
		VillagerNews.recordOven(event, from, to, config.truthfulNews());
		event.addProperty("actor", ACTOR.get() == null ? "world" : ACTOR.get());
		event.addProperty("cause", ACTOR.get() == null ? "world" : ACTOR.get().equals("system") ? "command" : "bot_action");
		if (collapse.removingCause() != null) {
			event.addProperty("cause", "collapse");
			event.addProperty("causeEventId", collapse.removingCause());
		}
		map("latestChanges").addProperty(key(position), event.get("id").getAsLong());
		changes().add(event.deepCopy());
		logEvent(event);
		collapse.changed(position, from, to, event.get("id").getAsLong());
		if (villagers != null) {
			villagers.blockChanged(position);
		}
	}

	/**
	 * Allocates an event header with the next ID and current overworld game time.
	 * @param type event type label
	 * @return new event object, not yet logged
	 * @throws ArithmeticException if the event counter overflows
	 */
	public JsonObject event(String type) {
		JsonObject event = new JsonObject();
		event.addProperty("id", next("eventCounter"));
		event.addProperty("gameTime", server.overworld().getGameTime());
		event.addProperty("type", type);
		return event;
	}

	/**
	 * Logs an origin once per provenance key without allowing details to replace header fields.
	 * @param prov stable provenance key
	 * @param kind origin kind
	 * @param details additional JSON fields copied into the origin event
	 * @throws UncheckedIOException if writing the first origin event fails
	 */
	public void origin(String prov, String kind, JsonObject details) {
		if (!map("origins").has(prov)) {
			JsonObject event = event("origin");
			event.addProperty("prov", prov);
			event.addProperty("kind", kind);
			details.entrySet().stream().filter(entry -> !java.util.Set.of("id", "gameTime", "type", "kind", "prov").contains(entry.getKey()))
				.forEach(entry -> event.add(entry.getKey(), entry.getValue().deepCopy()));
			map("origins").addProperty(prov, event.get("id").getAsLong());
			logEvent(event);
		}
	}

	/**
	 * Appends canonical event JSON to events.jsonl, lazily opening its writer.
	 * @param event completed event record
	 * @throws UncheckedIOException if opening or writing the audit file fails
	 */
	public void logEvent(JsonObject event) {
		try {
			if (events == null) {
				Files.createDirectories(directory);
				events = Files.newBufferedWriter(directory.resolve("events.jsonl"), StandardCharsets.UTF_8,
					StandardOpenOption.CREATE, StandardOpenOption.APPEND);
			}
			events.write(canonical(event));
			events.newLine();
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
		if (event.has("type") && event.get("type").getAsString().equals("sold_out")) {
			news.soldOut(event);
		}
	}

	/**
	 * Appends an exposure record in observations.jsonl (skipped when config exposure_log is 0, e.g. for long runs whose
	 * decision logs already keep each agent's rendered view); callers decide whether recording is enabled.
	 * @param observer observing bot name
	 * @param eye observer eye position at exposure
	 * @param kind exposure category
	 * @param items ordered exposed-item audit records
	 * @throws UncheckedIOException if opening or writing the audit file fails
	 */
	public void exposure(String observer, Vec3 eye, String kind, JsonArray items) {
		if (config.integer("exposure_log") == 0) {
			return;
		}
		JsonObject record = new JsonObject();
		record.addProperty("gameTime", server.overworld().getGameTime());
		record.addProperty("observer", observer);
		record.add("observerEye", vector(eye));
		record.addProperty("kind", kind);
		record.add("items", items);
		try {
			if (observations == null) {
				Files.createDirectories(directory);
				observations = Files.newBufferedWriter(directory.resolve("observations.jsonl"), StandardCharsets.UTF_8,
					StandardOpenOption.CREATE, StandardOpenOption.APPEND);
			}
			observations.write(canonical(record));
			observations.newLine();
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
	}

	/**
	 * Canonicalizes state with config, name-ordered skills, automatic water jumps and optional NPC state.
	 * @return exact JSON text used for persistence and the harness hash component
	 */
	public String serialization() {
		JsonObject saved = data.deepCopy();
		saved.add("config", config.json());
		JsonObject skills = new JsonObject();
		JsonObject afloat = new JsonObject();
		server.getPlayerList().getPlayers().stream().filter(player -> player instanceof SimBot)
			.map(player -> (SimBot) player).sorted(java.util.Comparator.comparing(bot -> bot.getGameProfile().name()))
			.forEach(bot -> {
				skills.add(bot.getGameProfile().name(), JsonParser.parseString(bot.skills().serialization()));
				if (bot.afloatJump()) {
					afloat.addProperty(bot.getGameProfile().name(), true);
				}
			});
		saved.add("skills", skills);
		if (!afloat.isEmpty()) {
			saved.add("afloat", afloat);
		}
		if (villagers != null) {
			saved.add("npcs", villagers.serialization());
		}
		return canonical(saved);
	}

	/**
	 * Flushes audit streams and atomically replaces harness_state.json with canonical state.
	 * @throws IOException if flushing, writing or atomically moving the save fails
	 */
	public void save() throws IOException {
		Files.createDirectories(directory);
		flush();
		Path temporary = directory.resolve("harness_state.json.tmp");
		Files.writeString(temporary, serialization(), StandardCharsets.UTF_8);
		Files.move(temporary, directory.resolve("harness_state.json"), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
	}

	/**
	 * Flushes writers and restores an existing state file, leaving state unchanged if absent.
	 * Saved skills are restored separately by BotManager, not by this method.
	 * @throws IOException if flushing, reading or closing the old NPC simulation fails
	 * @throws RuntimeException if saved JSON, configuration or NPC state is invalid
	 */
	public void load() throws IOException {
		flush();
		Path path = directory.resolve("harness_state.json");
		if (Files.exists(path)) {
			JsonObject loaded = JsonParser.parseString(Files.readString(path, StandardCharsets.UTF_8)).getAsJsonObject();
			SimConfig restored = new SimConfig();
			restored.update(loaded.getAsJsonObject("config"));
			loaded.remove("config");
			loaded.remove("skills");
			loaded.remove("afloat");
			JsonElement npcs = loaded.remove("npcs");
			data = loaded;
			if (!data.has("communication")) {
				data.add("communication", new JsonObject());
			}
			config = restored;
			if (villagers != null) {
				villagers.close();
			}
			villagers = npcs == null ? null : Villagers.restore(server, this, npcs);
		}
	}

	/**
	 * Flushes the event, observation and villager logs so readers (viewers, live analysis) see every completed step.
	 * Only file buffers are touched; simulation state is unchanged.
	 * @throws UncheckedIOException if a log cannot be flushed
	 */
	public void flushLogs() {
		try {
			flush();
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
	}

	private void flush() throws IOException {
		if (villagers != null) {
			villagers.flush();
		}
		if (events != null) {
			events.flush();
		}
		if (observations != null) {
			observations.flush();
		}
	}

	/**
	 * Closes NPC and harness audit writers in their established order.
	 * @throws UncheckedIOException if a writer cannot be closed
	 */
	@Override
	public void close() {
		try {
			if (villagers != null) {
				villagers.close();
			}
			if (events != null) {
				events.close();
			}
			if (observations != null) {
				observations.close();
			}
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
	}

	/**
	 * Sorts object keys recursively while retaining array order and configured Gson encoding.
	 * @param value nonnull JSON element to encode
	 * @return canonical JSON text without HTML escaping
	 */
	public static String canonical(JsonElement value) {
		if (value.isJsonObject()) {
			JsonObject sorted = new JsonObject();
			value.getAsJsonObject().keySet().stream().sorted().forEach(key ->
				sorted.add(key, JsonParser.parseString(canonical(value.getAsJsonObject().get(key)))));
			return GSON.toJson(sorted);
		}
		if (value.isJsonArray()) {
			JsonArray ordered = new JsonArray();
			value.getAsJsonArray().forEach(element -> ordered.add(JsonParser.parseString(canonical(element))));
			return GSON.toJson(ordered);
		}
		return GSON.toJson(value);
	}

	/**
	 * Hashes UTF-8 text without consuming simulation randomness.
	 * @param value text to hash
	 * @return lowercase hexadecimal SHA-256 digest
	 * @throws IllegalStateException if SHA-256 is unavailable
	 */
	public static String sha256(String value) {
		try {
			return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));
		} catch (NoSuchAlgorithmException failure) {
			throw new IllegalStateException(failure);
		}
	}

	/**
	 * Resolves a block state's registry identifier, independent of its properties.
	 * @param block block state
	 * @return namespaced registry identifier
	 */
	public static String blockId(BlockState block) {
		return BuiltInRegistries.BLOCK.getKey(block.getBlock()).toString();
	}

	/**
	 * Encodes an ACL/change-map position in the established comma-separated form.
	 * @param position block coordinate
	 * @return x,y,z key with no spaces
	 */
	public static String key(BlockPos position) {
		return position.getX() + "," + position.getY() + "," + position.getZ();
	}

	/**
	 * Encodes integer block coordinates in x/y/z order.
	 * @param position block coordinate
	 * @return new three-element JSON array
	 */
	public static JsonArray position(BlockPos position) {
		JsonArray result = new JsonArray();
		result.add(position.getX());
		result.add(position.getY());
		result.add(position.getZ());
		return result;
	}

	/**
	 * Encodes a vector without rounding in x/y/z order.
	 * @param vector vector to encode
	 * @return new three-element JSON array
	 */
	public static JsonArray vector(Vec3 vector) {
		JsonArray result = new JsonArray();
		result.add(vector.x);
		result.add(vector.y);
		result.add(vector.z);
		return result;
	}
}
