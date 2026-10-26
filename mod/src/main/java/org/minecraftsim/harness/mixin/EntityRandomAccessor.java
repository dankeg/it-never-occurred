package org.minecraftsim.harness.mixin;

import net.minecraft.util.RandomSource;
import net.minecraft.world.entity.Entity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Exposes the existing entity RNG for deterministic reseeding, without drawing from it. */
@Mixin(value = Entity.class, remap = false)
public interface EntityRandomAccessor {
	/**
	 * Reads the entity-owned random source.
	 * @return live entity RNG, not a new stream
	 */
	@Accessor("random")
	RandomSource simharness$getRandom();
}
