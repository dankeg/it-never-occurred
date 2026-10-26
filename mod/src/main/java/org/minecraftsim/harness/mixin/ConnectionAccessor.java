package org.minecraftsim.harness.mixin;

import io.netty.channel.Channel;
import net.minecraft.network.Connection;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Supplies the embedded fake-player channel without changing vanilla transport logic. */
@Mixin(value = Connection.class, remap = false)
public interface ConnectionAccessor {
	/**
	 * Installs the backing channel during fake-connection construction.
	 * @param channel embedded channel owned by the fake connection
	 */
	@Accessor("channel")
	void simharness$setChannel(Channel channel);
}
