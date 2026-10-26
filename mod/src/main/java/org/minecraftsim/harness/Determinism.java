package org.minecraftsim.harness;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.WeakHashMap;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.RandomSource;
import net.minecraft.world.RandomSequences;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.Level;
import org.minecraftsim.harness.mixin.EntityRandomAccessor;
import org.minecraftsim.harness.mixin.LevelRandomAccessor;
import org.minecraftsim.harness.mixin.ServerRandomAccessor;

/**
 * Defines stable identities, RNG checkpoints and entity-creation streams.
 * Pure identity helpers draw no RNG; mutable creation state is server-thread-only
 * and keyed by game time, dimension and creation/command index, never wall time.
 */
public final class Determinism {
	/** Server-thread creation state; weak keys do not retain stopped servers. */
	private static final Map<MinecraftServer, CreationState> CREATION_STATES = new WeakHashMap<>();

	private Determinism() {
	}

	/**
	 * Derives the canonical positive entity ID without drawing randomness.
	 * @param uuid entity identity
	 * @return stable nonzero 31-bit ID
	 */
	public static int idFor(UUID uuid) {
		int id = (int) (mix(uuid.getMostSignificantBits(), uuid.getLeastSignificantBits()) & 0x7fffffffL);
		return id == 0 ? 1 : id;
	}

	/**
	 * Reads the server-thread creation stream's current base seed.
	 * @param server owning server
	 * @return configured or most recently reseeded base seed
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static long baseSeed(MinecraftServer server) {
		return creationState(server).baseSeed;
	}

	/**
	 * Derives a stable NPC UUID in the NPC identity namespace without RNG draws.
	 * @param baseSeed scenario seed
	 * @param id scenario NPC identifier
	 * @return deterministic NPC UUID
	 */
	public static UUID npcUuid(long baseSeed, String id) {
		long identity = stableHash("npc:" + id);
		return new UUID(mix(baseSeed, identity, 0), mix(baseSeed, identity, 1));
	}

	/**
	 * Derives a daily schedule offset without sharing an RNG stream.
	 * @param baseSeed scenario seed
	 * @param id scenario NPC identifier
	 * @param day game-time day index
	 * @param bound positive exclusive offset bound
	 * @return deterministic offset from zero inclusive to bound exclusive
	 * @throws ArithmeticException if bound is zero
	 */
	public static int npcOffset(long baseSeed, String id, long day, int bound) {
		return (int) Math.floorMod(mix(baseSeed, stableHash("npc:" + id), day), bound);
	}

	/**
	 * Resets creation counters and reseeds level, entity, server and named-sequence RNGs.
	 * Seeds incorporate current overworld game time in the established composition.
	 * @param server owning server, accessed on its main thread
	 * @param baseSeed new base seed
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static void reseed(MinecraftServer server, long baseSeed) {
		CreationState creation = creationState(server);
		creation.baseSeed = baseSeed;
		creation.counters.clear();
		long gameTime = server.overworld().getGameTime();
		for (ServerLevel level : server.getAllLevels()) {
			long dimension = stableHash(level.dimension().identifier().toString());
			LevelRandomAccessor access = (LevelRandomAccessor) level;
			access.simharness$getRandom().setSeed(mix(baseSeed, gameTime, dimension));
			access.simharness$getSoundSeedGenerator().setSeed(mix(baseSeed, gameTime, dimension, stableHash("sound")));
			access.simharness$setRandValue((int) mix(baseSeed, gameTime, dimension, stableHash("block_position")));
			for (Entity entity : level.getAllEntities()) {
				UUID uuid = entity.getUUID();
				((EntityRandomAccessor) entity).simharness$getRandom()
					.setSeed(mix(baseSeed, gameTime, uuid.getMostSignificantBits(), uuid.getLeastSignificantBits()));
			}
		}
		((ServerRandomAccessor) server).simharness$getRandom().setSeed(mix(baseSeed, gameTime, stableHash("server")));
		RandomSequences sequences = server.getRandomSequences();
		sequences.forAllSequences((id, sequence) -> sequence.random()
			.setSeed(mix(baseSeed, gameTime, stableHash("sequence"), stableHash(id.toString()))));
		sequences.setDirty();
	}

	/**
	 * Reseeds shared level/server RNGs for the next command index at this game time.
	 * The command index resets only when game time changes.
	 * @param server owning server, accessed on its main thread
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static void reseedForCommand(MinecraftServer server) {
		CreationState creation = creationState(server);
		long gameTime = server.overworld().getGameTime();
		if (creation.commandGameTime != gameTime) {
			creation.commandGameTime = gameTime;
			creation.commandIndex = 0;
		}
		long index = creation.commandIndex++;
		long baseSeed = creation.baseSeed;
		for (ServerLevel level : server.getAllLevels()) {
			long dimension = stableHash(level.dimension().identifier().toString());
			LevelRandomAccessor access = (LevelRandomAccessor) level;
			access.simharness$getRandom().setSeed(commandSeed(baseSeed, gameTime, dimension, index));
		}
		((ServerRandomAccessor) server).simharness$getRandom().setSeed(mix(baseSeed, gameTime, stableHash("server_command"), index));
	}

	/** Derives the level command seed with the established input order and namespace. */
	static long commandSeed(long baseSeed, long gameTime, long dimension, long index) {
		return mix(baseSeed, gameTime, dimension, stableHash("command"), index);
	}

	/**
	 * Begins tick-scoped creation indexing and applies any configured game-time reseed.
	 * @param server server about to tick on its main thread
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static void beforeTick(MinecraftServer server) {
		CreationState creation = creationState(server);
		creation.counters.clear();
		creation.duringTick = true;
		reseedCheckpoint(server);
	}

	/**
	 * Restores the between-ticks creation scope after loading bots.
	 * @param server server performing the restore on its main thread
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static void afterRestore(MinecraftServer server) {
		afterTick(server);
	}

	/**
	 * Clears creation counters and leaves tick scope without drawing randomness.
	 * @param server server that finished ticking on its main thread
	 * @throws IllegalStateException if called off the server main thread
	 */
	public static void afterTick(MinecraftServer server) {
		CreationState creation = creationState(server);
		creation.counters.clear();
		creation.duringTick = false;
	}

	/**
	 * Creates an indexed entity RNG only for server-main-thread lockstep construction.
	 * Other construction contexts retain vanilla random-source creation.
	 * @param level entity's destination level
	 * @return new deterministic or vanilla random source for that construction
	 * @throws ArithmeticException if the per-scope creation index overflows
	 */
	public static RandomSource createEntityRandom(Level level) {
		if (!(level instanceof ServerLevel serverLevel)) {
			return RandomSource.create();
		}
		MinecraftServer server = serverLevel.getServer();
		if (!server.isSameThread() || !Lockstep.INSTANCE.isEnabled()) {
			return RandomSource.create();
		}
		CreationState creation = creationState(server);
		String dimension = level.dimension().identifier().toString();
		long gameTime = level.getGameTime();
		SpawnScope scope = new SpawnScope(dimension, gameTime);
		long index = creation.counters.getOrDefault(scope, 0L);
		creation.counters.put(scope, Math.incrementExact(index));
		return RandomSource.create(mix(creation.baseSeed, gameTime, stableHash(dimension),
			stableHash("entity_creation"), creation.duringTick ? 1L : 0L, index));
	}

	/** Checkpoint policy: only configured overworld game-time multiples trigger reseeding. */
	private static void reseedCheckpoint(MinecraftServer server) {
		long interval = Long.getLong("simharness.reseed_every", 0L);
		if (interval > 0 && server.overworld().getGameTime() % interval == 0) {
			reseed(server, Long.getLong("simharness.base_seed", 0L));
		}
	}

	private static CreationState creationState(MinecraftServer server) {
		if (!server.isSameThread()) {
			throw new IllegalStateException("Determinism must run on the server main thread");
		}
		return CREATION_STATES.computeIfAbsent(server, ignored -> new CreationState());
	}

	/** Stable encodings: UTF-8 hashing and fixed integer mixing never draw from game RNGs. */
	private static long stableHash(String value) {
		long hash = 0xcbf29ce484222325L;
		for (byte octet : value.getBytes(StandardCharsets.UTF_8)) {
			hash = (hash ^ (octet & 0xffL)) * 0x100000001b3L;
		}
		return hash;
	}

	/** Draws a uniform relay decision using only the specified encounter key and SplitMix64. */
	static double relayDraw(long baseSeed, String npcId, String partnerKey, String newsId, long day) {
		return (mix(baseSeed, stableHash(npcId), stableHash(partnerKey), stableHash(newsId), day) >>> 11) * 0x1.0p-53;
	}

	private static long mix(long... values) {
		long mixed = 0x9e3779b97f4a7c15L;
		for (long value : values) {
			mixed += value + 0x9e3779b97f4a7c15L;
			mixed = (mixed ^ (mixed >>> 30)) * 0xbf58476d1ce4e5b9L;
			mixed = (mixed ^ (mixed >>> 27)) * 0x94d049bb133111ebL;
			mixed ^= mixed >>> 31;
		}
		return mixed;
	}

	private static final class CreationState {
		private long baseSeed = Long.getLong("simharness.base_seed", 0L);
		private boolean duringTick;
		private final Map<SpawnScope, Long> counters = new HashMap<>();
		private long commandGameTime = Long.MIN_VALUE;
		private long commandIndex;
	}

	private record SpawnScope(String dimension, long gameTime) {
	}
}
