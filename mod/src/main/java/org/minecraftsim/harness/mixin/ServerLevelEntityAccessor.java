package org.minecraftsim.harness.mixin;

import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.entity.PersistentEntitySectionManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Exposes pending entity-load integration for the readiness barrier without ticking the world. */
@Mixin(value = ServerLevel.class, remap = false)
public interface ServerLevelEntityAccessor {
	/**
	 * Reads the level-owned entity section manager.
	 * @return live manager whose pending loads are processed on the server thread
	 */
	@Accessor("entityManager")
	PersistentEntitySectionManager<Entity> simharness$getEntityManager();
}
