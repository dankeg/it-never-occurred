package org.minecraftsim.harness;

import net.minecraft.SharedConstants;
import net.minecraft.core.component.DataComponentInitializers;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;

/** Boots the game's registries once per test JVM and binds item components, as a server's data load does. */
final class TestRegistries {
	private static boolean ready;

	private TestRegistries() {
	}

	/** Bootstraps registries and binds every item's default components, so tests can create item stacks. */
	static synchronized void bootstrap() {
		if (ready) {
			return;
		}
		SharedConstants.tryDetectVersion();
		Bootstrap.bootStrap();
		BuiltInRegistries.DATA_COMPONENT_INITIALIZERS.build(VanillaRegistries.createWorldLookup())
			.forEach(DataComponentInitializers.PendingComponents::apply);
		ready = true;
	}
}
