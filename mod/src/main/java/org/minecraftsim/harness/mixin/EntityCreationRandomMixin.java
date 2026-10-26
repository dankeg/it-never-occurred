package org.minecraftsim.harness.mixin;

import net.minecraft.util.RandomSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.level.Level;
import org.minecraftsim.harness.Determinism;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/** Delegates entity-constructor RNG creation to the lockstep-aware deterministic stream policy. */
@Mixin(value = Entity.class, remap = false)
public abstract class EntityCreationRandomMixin {
	@Redirect(method = "<init>", at = @At(value = "INVOKE", target = "Lnet/minecraft/util/RandomSource;create()Lnet/minecraft/util/RandomSource;"))
	private RandomSource simharness$createRandom(EntityType<?> type, Level level) {
		return Determinism.createEntityRandom(level);
	}
}
