package org.minecraftsim.harness.mixin;

import net.minecraft.world.entity.LivingEntity;
import org.minecraftsim.harness.Villagers;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Disables vanilla pushing only for harness NPCs, preserving their kinematic movement invariant. */
@Mixin(value = LivingEntity.class, remap = false)
public abstract class NpcCollisionMixin {
	@Inject(method = "isPushable", at = @At("HEAD"), cancellable = true)
	private void simharness$excludeKinematicBodies(CallbackInfoReturnable<Boolean> callback) {
		if (Villagers.isNpc((LivingEntity) (Object) this)) {
			callback.setReturnValue(false);
		}
	}
}
