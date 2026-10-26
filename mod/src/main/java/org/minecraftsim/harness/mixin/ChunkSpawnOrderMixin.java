package org.minecraftsim.harness.mixin;

import java.util.Comparator;
import java.util.List;
import net.minecraft.server.level.ServerChunkCache;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.chunk.LevelChunk;
import org.minecraftsim.harness.Lockstep;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * Orders the per-tick spawning-chunk list by chunk position instead of shuffling it with the level RNG.
 *
 * <p>Vanilla shuffles the chunks near any player every tick, drawing one level-RNG value per chunk. Bots are players,
 * and chunks near a bot that lie outside the force-loaded area load asynchronously, so the number of draws, and thus
 * every later level-RNG value (e.g. item-drop motion), would depend on chunk-loading timing. Mob spawning is disabled,
 * so the order has no other effect. Applies only in lockstep.
 */
@Mixin(value = ServerChunkCache.class, remap = false)
public abstract class ChunkSpawnOrderMixin {
	@Redirect(method = "tickChunks(Lnet/minecraft/util/profiling/ProfilerFiller;)V",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/util/Util;shuffle(Ljava/util/List;Lnet/minecraft/util/RandomSource;)V"))
	private void simharness$deterministicSpawnOrder(List<LevelChunk> chunks, RandomSource random) {
		if (Lockstep.INSTANCE.isEnabled()) {
			chunks.sort(Comparator.comparingLong(chunk -> chunk.getPos().pack()));
		} else {
			net.minecraft.util.Util.shuffle(chunks, random);
		}
	}
}
