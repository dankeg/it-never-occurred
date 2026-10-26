package org.minecraftsim.harness.mixin;

import net.minecraft.util.RandomSource;
import net.minecraft.world.level.Level;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Bridges level RNG state for deterministic checkpoint reseeding on the server thread. */
@Mixin(value = Level.class, remap = false)
public interface LevelRandomAccessor {
	/**
	 * Reads the shared level random source without consuming it.
	 * @return live level RNG
	 */
	@Accessor("random")
	RandomSource simharness$getRandom();

	/**
	 * Reads the level sound-seed stream without consuming it.
	 * @return live sound-seed RNG
	 */
	@Accessor("soundSeedGenerator")
	RandomSource simharness$getSoundSeedGenerator();

	/**
	 * Replaces the block-position sampling state with its deterministic checkpoint value.
	 * @param value replacement vanilla randValue
	 */
	@Accessor("randValue")
	void simharness$setRandValue(int value);
}
