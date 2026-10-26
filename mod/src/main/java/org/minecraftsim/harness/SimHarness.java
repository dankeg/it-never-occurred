package org.minecraftsim.harness;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Connects Fabric lifecycle events to harness state, bots and the loopback control socket.
 * Startup/shutdown hooks preserve server-thread initialization and step-abort ordering.
 */
public class SimHarness implements ModInitializer {
	/** Stable Fabric mod identifier, also used as the logger name. */
	public static final String MOD_ID = "simharness";
	/** Harness diagnostics; logging does not participate in canonical state. */
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);
	private ControlSocket control;
	private BotManager bots;

	/**
	 * Registers lifecycle hooks once when Fabric initializes the mod.
	 * Socket startup failures become IllegalStateException in the server-started hook.
	 */
	@Override
	public void onInitialize() {
		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			HarnessState.get(server);
			bots = new BotManager(server);
			try {
				control = new ControlSocket(server, bots);
			} catch (java.io.IOException failure) {
				throw new IllegalStateException("Unable to start SimHarness control socket", failure);
			}
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			Lockstep.INSTANCE.abort(new IllegalStateException("Server stopping"));
			if (control != null) {
				control.close();
			}
		});
		ServerLifecycleEvents.SERVER_STOPPED.register(server -> {
			if (bots != null) {
				bots.close();
			}
			HarnessState.remove(server);
		});
		LOGGER.info("SimHarness loaded");
	}
}
