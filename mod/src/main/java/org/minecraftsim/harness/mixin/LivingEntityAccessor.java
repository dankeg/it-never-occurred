package org.minecraftsim.harness.mixin;

import net.minecraft.world.entity.LivingEntity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Bridges the jump cooldown omitted from vanilla player persistence for exact bot restoration. */
@Mixin(value = LivingEntity.class, remap = false)
public interface LivingEntityAccessor {
	/**
	 * Reads the current jump cooldown without modifying it.
	 * @return vanilla noJumpDelay counter
	 */
	@Accessor("noJumpDelay")
	int simharness$getNoJumpDelay();

	/**
	 * Restores the jump cooldown on the server thread.
	 * @param delay saved noJumpDelay counter
	 */
	@Accessor("noJumpDelay")
	void simharness$setNoJumpDelay(int delay);
}
