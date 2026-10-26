package org.minecraftsim.harness.mixin;

import net.minecraft.server.MinecraftServer;
import net.minecraft.util.RandomSource;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Exposes the existing server RNG for checkpoint reseeding without adding RNG draws. */
@Mixin(value = MinecraftServer.class, remap = false)
public interface ServerRandomAccessor {
	/**
	 * Reads the server-owned random source.
	 * @return live server RNG
	 */
	@Accessor("random")
	RandomSource simharness$getRandom();
}
