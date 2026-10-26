package org.minecraftsim.harness.mixin;

import net.minecraft.world.entity.npc.villager.Villager;
import org.minecraftsim.harness.Villagers;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Cancels vanilla villager ticks only for harness NPCs, whose schedule advances in the harness hook. */
@Mixin(value = Villager.class, remap = false)
public abstract class VillagerMixin {
	@Inject(method = "tick", at = @At("HEAD"), cancellable = true)
	private void simharness$kinematicTick(CallbackInfo callback) {
		if (Villagers.isNpc((Villager) (Object) this)) {
			callback.cancel();
		}
	}
}
