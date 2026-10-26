package org.minecraftsim.harness.mixin;

import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Bridges fake-player connection fields preserved verbatim in physics snapshots. */
@Mixin(value = ServerGamePacketListenerImpl.class, remap = false)
public interface GameListenerAccessor {
	/**
	 * Reads the client-loaded timeout without advancing it.
	 * @return current vanilla timeout counter
	 */
	@Accessor("clientLoadedTimeoutTimer")
	int simharness$getClientLoadedTimeoutTimer();

	/**
	 * Restores the saved timeout counter on the server thread.
	 * @param timer saved timeout counter
	 */
	@Accessor("clientLoadedTimeoutTimer")
	void simharness$setClientLoadedTimeoutTimer(int timer);

	/**
	 * Reads the vanilla respawn-wait flag.
	 * @return whether the listener is waiting for respawn
	 */
	@Accessor("waitingForRespawn")
	boolean simharness$isWaitingForRespawn();

	/**
	 * Restores the saved respawn-wait flag on the server thread.
	 * @param waiting saved respawn-wait flag
	 */
	@Accessor("waitingForRespawn")
	void simharness$setWaitingForRespawn(boolean waiting);
}
