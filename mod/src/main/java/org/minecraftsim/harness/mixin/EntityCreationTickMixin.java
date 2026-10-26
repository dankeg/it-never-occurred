package org.minecraftsim.harness.mixin;

import java.util.function.BooleanSupplier;
import net.minecraft.server.MinecraftServer;
import org.minecraftsim.harness.Determinism;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Ends deterministic entity-creation tick scope at the existing vanilla server tick tail. */
@Mixin(value = MinecraftServer.class, remap = false)
public abstract class EntityCreationTickMixin {
	@Inject(method = "tickServer", at = @At("TAIL"))
	private void simharness$finishCreationScope(BooleanSupplier haveTime, CallbackInfo callback) {
		Determinism.afterTick((MinecraftServer) (Object) this);
	}
}
