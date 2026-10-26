package org.minecraftsim.harness;

import io.netty.channel.ChannelFutureListener;
import io.netty.channel.embedded.EmbeddedChannel;
import net.minecraft.network.Connection;
import net.minecraft.network.PacketListener;
import net.minecraft.network.ProtocolInfo;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import org.minecraftsim.harness.mixin.ConnectionAccessor;

// Adapted from Carpet's FakeClientConnection (MIT), copyright gnembon.
/**
 * In-memory fake-player connection that discards networking without invoking send callbacks.
 * It supplies vanilla's required channel but adds no packet-driven simulation work.
 */
public final class NullConnection extends Connection {
	private final EmbeddedChannel channel = new EmbeddedChannel();

	/** Creates a serverbound connection backed by one embedded channel. */
	public NullConnection() {
		super(PacketFlow.SERVERBOUND);
		((ConnectionAccessor) (Object) this).simharness$setChannel(channel);
	}

	/**
	 * Discards an outgoing packet without scheduling a listener.
	 * @param packet ignored packet
	 * @param listener ignored completion listener
	 * @param flush ignored flush request
	 */
	@Override
	public void send(Packet<?> packet, ChannelFutureListener listener, boolean flush) {
	}

	/** Ignores transport read-only requests for the embedded connection. */
	@Override
	public void setReadOnly() {
	}

	/** Suppresses network disconnection handling for a server-owned fake player. */
	@Override
	public void handleDisconnection() {
	}

	/**
	 * Skips the fake client's handshake listener installation.
	 * @param listener ignored handshake listener
	 */
	@Override
	public void setListenerForServerboundHandshake(PacketListener listener) {
	}

	/**
	 * Skips protocol/channel setup that requires a real client.
	 * @param <T> inbound listener type
	 * @param protocol ignored inbound protocol
	 * @param listener ignored inbound listener
	 */
	@Override
	public <T extends PacketListener> void setupInboundProtocol(ProtocolInfo<T> protocol, T listener) {
	}

	/** Finishes the embedded channel and releases its retained messages. */
	public void close() {
		channel.finishAndReleaseAll();
	}
}
