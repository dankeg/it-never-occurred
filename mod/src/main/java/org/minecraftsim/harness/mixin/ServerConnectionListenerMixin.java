package org.minecraftsim.harness.mixin;

import java.util.Comparator;
import java.util.List;
import net.minecraft.network.Connection;
import net.minecraft.server.network.ServerConnectionListener;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.minecraftsim.harness.Lockstep;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Orders player connections by profile name after pending joins, only during lockstep ticks. */
@Mixin(value = ServerConnectionListener.class, remap = false)
public abstract class ServerConnectionListenerMixin {
	@Shadow
	@Final
	private List<Connection> connections;

	@Inject(method = "tick", at = @At(value = "INVOKE", target = "Lnet/minecraft/server/network/ServerConnectionListener;addPendingConnections()V", shift = At.Shift.AFTER))
	private void simharness$canonicalPlayerTickOrder(CallbackInfo callback) {
		if (Lockstep.INSTANCE.isEnabled()) {
			this.connections.sort(Comparator.comparing(connection ->
				connection.getPacketListener() instanceof ServerGamePacketListenerImpl listener
					? listener.player.getGameProfile().name() : ""));
		}
	}
}
