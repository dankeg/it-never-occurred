package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.ChunkPos;

/**
 * Non-ticking readiness barrier. After a (re)start, vanilla loads entity sections asynchronously and integrates them
 * only inside a world tick, so a restored run would otherwise skip entities on its first tick (E2 diagnosis,
 * 2026-10-16: restored item entities were one tick behind). This waits, without advancing game time, until every chunk
 * in the rectangle has its entities loaded and is entity-ticking.
 */
public final class Readiness {
	private Readiness() {
	}

	/**
	 * Processes pending entity loads until every selected chunk is entity-ticking.
	 * Managed waiting permits server tasks but must not advance overworld game time.
	 * @param server server executing the barrier on its main thread
	 * @param args inclusive cx0/cz0/cx1/cz1 chunk bounds
	 * @return game time, selected chunk count and elapsed wall-clock wait milliseconds
	 * @throws IllegalStateException if game time advances during the barrier
	 * @throws RuntimeException if required bounds cannot be read
	 */
	public static JsonObject awaitEntities(MinecraftServer server, JsonObject args) {
		int cx0 = args.get("cx0").getAsInt();
		int cz0 = args.get("cz0").getAsInt();
		int cx1 = args.get("cx1").getAsInt();
		int cz1 = args.get("cz1").getAsInt();
		ServerLevel level = server.overworld();
		long before = level.getGameTime();
		List<ChunkPos> chunks = new ArrayList<>();
		for (int x = cx0; x <= cx1; x++) {
			for (int z = cz0; z <= cz1; z++) {
				chunks.add(new ChunkPos(x, z));
			}
		}
		long started = System.nanoTime();
		server.managedBlock(() -> {
			((org.minecraftsim.harness.mixin.ServerLevelEntityAccessor) level).simharness$getEntityManager().processPendingLoads();
			for (ChunkPos chunk : chunks) {
				if (!level.areEntitiesActuallyLoadedAndTicking(chunk)) {
					return false;
				}
			}
			return true;
		});
		if (level.getGameTime() != before) {
			throw new IllegalStateException("Game time advanced during await_entities");
		}
		JsonObject result = new JsonObject();
		result.addProperty("gameTime", level.getGameTime());
		result.addProperty("chunks", chunks.size());
		result.addProperty("waitedMs", (System.nanoTime() - started) / 1_000_000.0);
		return result;
	}
}
