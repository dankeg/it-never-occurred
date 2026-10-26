package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.authlib.GameProfile;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;
import net.minecraft.core.UUIDUtil;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.ListTag;
import net.minecraft.nbt.NbtAccounter;
import net.minecraft.nbt.NbtIo;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.util.ProblemReporter;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.storage.LevelResource;
import net.minecraft.world.level.storage.TagValueInput;
import net.minecraft.world.level.storage.TagValueOutput;
import net.minecraft.world.phys.Vec3;

/**
 * Owns fake-player lifecycles and bot snapshots on the server main thread.
 * Names order iteration and restoration; bots.dat and physics/input encodings remain stable.
 */
public final class BotManager implements AutoCloseable {
	private final MinecraftServer server;
	private final Map<String, SimBot> bots = new TreeMap<>();
	private final ArrayList<NullConnection> connections = new ArrayList<>();

	/**
	 * Creates an empty manager without spawning players.
	 * @param server server owning these bots and their world save
	 */
	public BotManager(MinecraftServer server) {
		this.server = server;
	}

	/**
	 * Takes an immutable, name-ordered snapshot of managed bot references.
	 * @return current bots in name order
	 */
	public java.util.Collection<SimBot> all() {
		return java.util.List.copyOf(bots.values());
	}

	/**
	 * Resolves an existing managed bot without creating it.
	 * @param name exact bot name
	 * @return managed bot
	 * @throws IllegalArgumentException if the name is unknown
	 */
	public SimBot get(String name) {
		SimBot bot = bots.get(name);
		if (bot == null) {
			throw new IllegalArgumentException("Unknown bot: " + name);
		}
		return bot;
	}

	private SimBot create(String name) {
		if (!name.matches("[A-Za-z0-9_]{1,16}")) {
			throw new IllegalArgumentException("Bot names must be 1-16 ASCII letters, digits or underscores");
		}
		if (bots.containsKey(name) || server.getPlayerList().getPlayerByName(name) != null) {
			throw new IllegalArgumentException("Player already exists: " + name);
		}
		GameProfile profile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(name), name);
		return new SimBot(server, server.overworld(), profile);
	}

	private void place(SimBot bot) {
		NullConnection connection = new NullConnection();
		connections.add(connection);
		server.getPlayerList().placeNewPlayer(connection, bot, CommonListenerCookie.createInitial(bot.getGameProfile(), false));
		bots.put(bot.getGameProfile().name(), bot);
	}

	/**
	 * Creates, positions and joins a survival bot using the supplied protocol fields.
	 * @param args name, x/y/z and yaw/pitch fields
	 * @return joined bot's protocol state
	 * @throws IllegalArgumentException if the name, position or view angles are invalid or occupied
	 * @throws RuntimeException if required fields cannot be read
	 */
	public JsonObject spawn(JsonObject args) {
		SimBot bot = create(args.get("name").getAsString());
		double x = args.get("x").getAsDouble();
		double y = args.get("y").getAsDouble();
		double z = args.get("z").getAsDouble();
		float yaw = args.get("yaw").getAsFloat();
		float pitch = args.get("pitch").getAsFloat();
		if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(z)) {
			throw new IllegalArgumentException("Bot coordinates must be finite");
		}
		bot.setInput(BotInput.idle(yaw, pitch));
		bot.snapTo(x, y, z, yaw, pitch);
		place(bot);
		bot.setGameMode(GameType.SURVIVAL);
		return state(bot);
	}

	/**
	 * Serializes visible bot state with inventory entries in ascending slot order.
	 * @param bot bot to inspect on the server thread
	 * @return newly allocated protocol state object
	 */
	public JsonObject state(SimBot bot) {
		HarnessState harness = HarnessState.existing(server);
		JsonObject result = new JsonObject();
		result.addProperty("name", bot.getGameProfile().name());
		result.add("pos", vector(bot.position()));
		result.add("deltaMovement", vector(bot.getDeltaMovement()));
		result.addProperty("yRot", bot.getYRot());
		result.addProperty("xRot", bot.getXRot());
		result.addProperty("onGround", bot.onGround());
		result.addProperty("health", bot.getHealth());
		result.addProperty("food", bot.getFoodData().getFoodLevel());
		result.addProperty("selectedSlot", bot.getInventory().getSelectedSlot());
		JsonArray inventory = new JsonArray();
		for (int slot = 0; slot < bot.getInventory().getContainerSize(); slot++) {
			var stack = bot.getInventory().getItem(slot);
			if (!stack.isEmpty()) {
				JsonObject item = new JsonObject();
				item.addProperty("slot", slot);
				item.addProperty("item", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
				item.addProperty("count", stack.getCount());
				if (harness != null) {
					Freshness.describe(stack, harness.config(), Math.floorDiv(bot.level().getGameTime(), 24_000L), item);
				}
				inventory.add(item);
			}
		}
		result.add("inventory", inventory);
		return result;
	}

	private static JsonArray vector(Vec3 vector) {
		JsonArray result = new JsonArray();
		result.add(vector.x);
		result.add(vector.y);
		result.add(vector.z);
		return result;
	}

	private Path snapshotPath() {
		return server.getWorldPath(LevelResource.ROOT).resolve("simharness/bots.dat");
	}

	/**
	 * Atomically replaces bots.dat, then saves canonical harness state.
	 * @throws IOException if player serialization or snapshot/harness file operations fail
	 */
	public void save() throws IOException {
		ListTag savedBots = new ListTag();
		for (Map.Entry<String, SimBot> entry : bots.entrySet()) {
			SimBot bot = entry.getValue();
			CompoundTag saved = new CompoundTag();
			saved.putString("name", entry.getKey());
			ProblemReporter.Collector reporter = new ProblemReporter.Collector();
			TagValueOutput output = TagValueOutput.createWithContext(reporter, server.registryAccess());
			bot.saveWithoutId(output);
			if (!reporter.isEmpty()) {
				throw new IOException("Cannot save full bot data for " + entry.getKey() + ": " + reporter.getReport());
			}
			saved.put("player", output.buildResult());
			saved.put("input", bot.input().save());
			saved.put("physics", bot.savePhysics());
			saved.putByteArray("skills", bot.skills().serialization().getBytes(StandardCharsets.UTF_8));
			savedBots.add(saved);
		}
		CompoundTag root = new CompoundTag();
		root.putLong("gameTime", server.overworld().getGameTime());
		root.put("bots", savedBots);
		Path target = snapshotPath();
		Files.createDirectories(target.getParent());
		Path temporary = Files.createTempFile(target.getParent(), "bots-", ".dat");
		try {
			NbtIo.writeCompressed(root, temporary);
			Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
		} finally {
			Files.deleteIfExists(temporary);
		}
		HarnessState.get(server).save();
	}

	private void loadPlayer(SimBot bot, CompoundTag data) throws IOException {
		ProblemReporter.Collector reporter = new ProblemReporter.Collector();
		bot.load(TagValueInput.create(reporter, server.registryAccess(), data));
		if (!reporter.isEmpty()) {
			throw new IOException("Cannot load full bot data for " + bot.getGameProfile().name() + ": " + reporter.getReport());
		}
	}

	/**
	 * Loads harness state and restores name-ordered bots at the saved world game time.
	 * Player data is reapplied after joining, before physics and skill restoration.
	 * @return current game time and restored bot state array
	 * @throws IllegalStateException if lockstep is not idle, bots exist or snapshot time differs
	 * @throws IOException if the snapshot cannot be read or full player data cannot be loaded
	 * @throws RuntimeException if saved bot fields are invalid
	 */
	public JsonObject restore() throws IOException {
		if (!Lockstep.INSTANCE.isEnabled() || Lockstep.INSTANCE.pendingTicks() != 0 || !bots.isEmpty()) {
			throw new IllegalStateException("restore_bots requires idle lockstep and no existing bots");
		}
		HarnessState.get(server).load();
		Determinism.reseedForCommand(server);
		CompoundTag root = NbtIo.readCompressed(snapshotPath(), NbtAccounter.unlimitedHeap());
		if (root.getLongOr("gameTime", -1) != server.overworld().getGameTime()) {
			throw new IllegalStateException("Bot snapshot game time does not match the world");
		}
		ArrayList<CompoundTag> savedBots = new ArrayList<>();
		for (var tag : root.getListOrEmpty("bots")) {
			if (!(tag instanceof CompoundTag saved)) {
				throw new IOException("Invalid bot entry");
			}
			savedBots.add(saved);
		}
		savedBots.sort(Comparator.comparing(saved -> saved.getStringOr("name", "")));
		JsonArray states = new JsonArray();
		for (CompoundTag saved : savedBots) {
			SimBot bot = create(saved.getStringOr("name", ""));
			bot.tickCount = saved.getCompoundOrEmpty("physics").getIntOr("tickCount", 0);
			CompoundTag player = saved.getCompound("player").orElseThrow(() -> new IOException("Missing player data"));
			loadPlayer(bot, player);
			place(bot);
			loadPlayer(bot, player);
			bot.setInput(BotInput.load(saved.getCompoundOrEmpty("input")));
			bot.loadPhysics(saved.getCompoundOrEmpty("physics"));
			bot.skills().restore(saved.getByteArray("skills").map(bytes -> new String(bytes, StandardCharsets.UTF_8))
				.orElseGet(() -> saved.getStringOr("skills", "{}")));
			bot.connection.resetPosition();
			bot.level().getChunkSource().move(bot);
			states.add(state(bot));
		}
		Determinism.afterRestore(server);
		JsonObject result = new JsonObject();
		result.addProperty("gameTime", server.overworld().getGameTime());
		result.add("bots", states);
		return result;
	}

	/** Releases fake connections and forgets managed bots without saving a snapshot. */
	@Override
	public void close() {
		connections.forEach(NullConnection::close);
		connections.clear();
		bots.clear();
	}
}
