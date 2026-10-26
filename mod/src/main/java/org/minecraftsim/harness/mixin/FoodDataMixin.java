package org.minecraftsim.harness.mixin;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.food.FoodData;
import org.minecraftsim.harness.HarnessState;
import org.minecraftsim.harness.SimBot;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Leaves vanilla food untouched unless a survival bot uses the harness's fixed daily hunger schedule. */
@Mixin(value = FoodData.class, remap = false)
public abstract class FoodDataMixin {
	@Shadow
	private float exhaustionLevel;

	@Shadow
	public abstract int getFoodLevel();

	/** Discards vanilla exhaustion costs and stops the starvation/healing timer at zero food. */
	@Inject(method = "tick", at = @At("HEAD"), cancellable = true)
	private void simharness$dailyHunger(ServerPlayer player, CallbackInfo callback) {
		if (!(player instanceof SimBot) || player.hasInfiniteMaterials()) {
			return;
		}
		HarnessState state = HarnessState.existing(player.level().getServer());
		if (state != null && state.config().dailyHunger()) {
			exhaustionLevel = 0;
			if (getFoodLevel() == 0) {
				callback.cancel();
			}
		}
	}
}
