package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.concurrent.CompletableFuture;
import net.minecraft.server.MinecraftServer;

/**
 * Owns the server-thread lockstep gate and one outstanding step request.
 * Hash checkpoints use overworld game time, never wall time or the server tick counter.
 */
public final class Lockstep {
	/** Server-thread gate shared by the control dispatcher and vanilla tick hooks. */
	public static final Lockstep INSTANCE = new Lockstep();
	private boolean enabled = true;
	private long pendingTicks;
	private CompletableFuture<JsonObject> stepFuture;
	private JsonArray hashes;
	private long hashEvery;
	private Region region;

	private Lockstep() {
	}

	/**
	 * Reads the mode without changing the gate; callers use the server thread.
	 * @return whether lockstep gating is enabled
	 */
	public boolean isEnabled() {
		return enabled;
	}

	/**
	 * Reads the remaining tick budget without consuming it.
	 * @return number of requested ticks not yet admitted by the server-thread gate
	 */
	public long pendingTicks() {
		return pendingTicks;
	}

	/**
	 * Consumes the tick already admitted by the server-thread gate.
	 * The caller must first observe enabled lockstep and a nonzero pending budget.
	 */
	public void consumePendingTick() {
		pendingTicks--;
	}

	/**
	 * Changes the gate mode on the server thread without altering queued work.
	 * @param enabled requested lockstep mode
	 * @throws IllegalStateException if a step request is outstanding
	 */
	public void setEnabled(boolean enabled) {
		if (stepFuture != null) {
			throw new IllegalStateException("A step is in progress");
		}
		this.enabled = enabled;
	}

	/**
	 * Queues an exact tick budget; a zero-tick request completes immediately.
	 * @param server server whose overworld supplies result and checkpoint times
	 * @param ticks nonnegative number of ticks to admit
	 * @param interval checkpoint interval, or zero to disable hashing
	 * @param hashRegion region required when hashing is enabled
	 * @return the request future, failed by {@link #abort(Throwable)} on tick failure
	 * @throws IllegalStateException if lockstep is disabled or another request is outstanding
	 * @throws IllegalArgumentException if the tick budget or hash configuration is invalid
	 */
	public CompletableFuture<JsonObject> step(MinecraftServer server, long ticks, long interval, Region hashRegion) {
		if (!enabled || stepFuture != null) {
			throw new IllegalStateException("step requires idle lockstep mode");
		}
		if (ticks < 0 || interval < 0 || (interval > 0 && hashRegion == null)) {
			throw new IllegalArgumentException("n must be nonnegative; hashing requires a positive interval and region");
		}
		hashes = new JsonArray();
		hashEvery = interval;
		region = hashRegion;
		pendingTicks = ticks;
		stepFuture = new CompletableFuture<>();
		CompletableFuture<JsonObject> result = stepFuture;
		if (ticks == 0) {
			finish(server);
		}
		return result;
	}

	/**
	 * Records a due game-time checkpoint and completes an exhausted request.
	 * Checkpoint failures abort the future rather than escaping this server-thread hook.
	 * @param server server that just completed a tick
	 */
	public void afterTick(MinecraftServer server) {
		if (stepFuture == null) {
			return;
		}
		try {
			long gameTime = server.overworld().getGameTime();
			if (hashEvery > 0 && gameTime % hashEvery == 0) {
				JsonObject record = new JsonObject();
				record.addProperty("gameTime", gameTime);
				record.add("hash", StateHash.compute(server, region));
				hashes.add(record);
			}
			if (pendingTicks == 0) {
				finish(server);
			}
		} catch (Exception failure) {
			abort(failure);
		}
	}

	private void finish(MinecraftServer server) {
		HarnessState state = HarnessState.existing(server);
		if (state != null) {
			state.flushLogs();
		}
		JsonObject result = new JsonObject();
		result.addProperty("gameTime", server.overworld().getGameTime());
		result.add("hashes", hashes);
		CompletableFuture<JsonObject> completed = stepFuture;
		stepFuture = null;
		completed.complete(result);
	}

	/**
	 * Clears the tick budget and fails any outstanding request on the server thread.
	 * @param failure cause supplied to the request future
	 */
	public void abort(Throwable failure) {
		pendingTicks = 0;
		if (stepFuture != null) {
			CompletableFuture<JsonObject> failed = stepFuture;
			stepFuture = null;
			failed.completeExceptionally(failure);
		}
	}
}
