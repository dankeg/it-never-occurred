package org.minecraftsim.harness.mixin;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import java.util.Arrays;
import java.util.Comparator;
import java.util.function.Consumer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.entity.EntityTickList;
import org.minecraftsim.harness.Lockstep;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Orders server-level entity ticks by canonical integer ID only while lockstep is enabled.
 * Keeps vanilla's single-iteration guard and clears iteration state even when a consumer fails.
 */
@Mixin(value = EntityTickList.class, remap = false)
public abstract class EntityTickListMixin {
	@Shadow
	private Int2ObjectMap<Entity> active;

	@Shadow
	private Int2ObjectMap<Entity> iterated;

	@Inject(method = "forEach", at = @At("HEAD"), cancellable = true)
	private void simharness$canonicalTickOrder(Consumer<Entity> output, CallbackInfo callback) {
		if (!Lockstep.INSTANCE.isEnabled() || this.active.isEmpty()
				|| !(this.active.values().iterator().next().level() instanceof ServerLevel)) {
			return;
		}
		if (this.iterated != null) {
			throw new UnsupportedOperationException("Only one concurrent iteration supported");
		}
		this.iterated = this.active;
		try {
			Entity[] entities = this.active.values().toArray(Entity[]::new);
			Arrays.sort(entities, Comparator.comparingInt(Entity::getId));
			for (Entity entity : entities) {
				output.accept(entity);
			}
		} finally {
			this.iterated = null;
		}
		callback.cancel();
	}
}
