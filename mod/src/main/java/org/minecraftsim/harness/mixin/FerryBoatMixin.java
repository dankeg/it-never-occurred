package org.minecraftsim.harness.mixin;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.vehicle.boat.AbstractBoat;
import org.minecraftsim.harness.Ferry;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Suppresses vanilla ticking and pushing only for the harness's registered ferry. */
@Mixin(value = AbstractBoat.class, remap = false)
public abstract class FerryBoatMixin {
	@Inject(method = "tick", at = @At("HEAD"), cancellable = true)
	private void simharness$kinematicTick(CallbackInfo callback) {
		if (Ferry.isBoat((AbstractBoat) (Object) this)) {
			callback.cancel();
		}
	}

	@Inject(method = "isPushable", at = @At("HEAD"), cancellable = true)
	private void simharness$excludeKinematicBoat(CallbackInfoReturnable<Boolean> callback) {
		if (Ferry.isBoat((AbstractBoat) (Object) this)) {
			callback.setReturnValue(false);
		}
	}

	@Inject(method = "push(Lnet/minecraft/world/entity/Entity;)V", at = @At("HEAD"), cancellable = true)
	private void simharness$preventPushing(Entity other, CallbackInfo callback) {
		if (Ferry.isBoat((AbstractBoat) (Object) this)) {
			callback.cancel();
		}
	}
}
