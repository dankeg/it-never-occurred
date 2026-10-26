package org.minecraftsim.harness.mixin;

import java.util.UUID;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.entity.EntityInLevelCallback;
import net.minecraft.world.level.storage.ValueInput;
import net.minecraft.world.level.storage.ValueOutput;
import org.minecraftsim.harness.Determinism;
import org.minecraftsim.harness.Lockstep;
import org.minecraftsim.harness.SimHarness;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Persists server entity tick counts and assigns UUID-derived IDs in lockstep mode.
 * Registered entities retain their IDs; pre-registration collisions use ascending IDs with wraparound.
 */
@Mixin(value = Entity.class, remap = false)
public abstract class EntityCanonMixin {
	@Shadow
	private EntityInLevelCallback levelCallback;

	@Unique
	private boolean simharness$warnedRegisteredUuidChange;

	@Inject(method = "<init>(Lnet/minecraft/world/entity/EntityType;Lnet/minecraft/world/level/Level;)V", at = @At("TAIL"))
	private void simharness$canonicalCreationId(EntityType<?> type, Level level, CallbackInfo callback) {
		simharness$assignCanonicalId();
	}

	@Inject(method = "setUUID(Ljava/util/UUID;)V", at = @At("TAIL"))
	private void simharness$canonicalUuidId(UUID uuid, CallbackInfo callback) {
		simharness$assignCanonicalId();
	}

	@Inject(method = "saveWithoutId(Lnet/minecraft/world/level/storage/ValueOutput;)V", at = @At("TAIL"))
	private void simharness$saveTickCount(ValueOutput output, CallbackInfo callback) {
		Entity entity = (Entity) (Object) this;
		if (entity.level() instanceof ServerLevel) {
			output.putInt("simharness_tickCount", entity.tickCount);
		}
	}

	@Inject(method = "load(Lnet/minecraft/world/level/storage/ValueInput;)V", at = @At("TAIL"))
	private void simharness$loadTickCountAndId(ValueInput input, CallbackInfo callback) {
		Entity entity = (Entity) (Object) this;
		if (entity.level() instanceof ServerLevel) {
			entity.tickCount = input.getIntOr("simharness_tickCount", entity.tickCount);
			simharness$assignCanonicalId();
		}
	}

	@Unique
	private void simharness$assignCanonicalId() {
		Entity entity = (Entity) (Object) this;
		if (!Lockstep.INSTANCE.isEnabled() || !(entity.level() instanceof ServerLevel level)) {
			return;
		}
		if (this.levelCallback != EntityInLevelCallback.NULL) {
			if (!this.simharness$warnedRegisteredUuidChange) {
				this.simharness$warnedRegisteredUuidChange = true;
				SimHarness.LOGGER.warn("Not reassigning canonical ID for registered entity {} (ID {})", entity.getUUID(), entity.getId());
			}
			return;
		}
		int id = Determinism.idFor(entity.getUUID());
		while (level.getChunkSource().hasEntityWithId(id)) {
			id = id == Integer.MAX_VALUE ? 1 : id + 1;
		}
		entity.setId(id);
	}
}
