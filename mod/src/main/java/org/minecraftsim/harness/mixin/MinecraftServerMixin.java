package org.minecraftsim.harness.mixin;

import java.util.function.BooleanSupplier;
import net.minecraft.server.MinecraftServer;
import org.minecraftsim.harness.Determinism;
import org.minecraftsim.harness.HarnessState;
import org.minecraftsim.harness.Lockstep;
import net.minecraft.util.Util;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Applies the lockstep budget at vanilla tick boundaries and delegates deterministic/NPC hooks.
 * Idle gating never advances game time; wall-clock deadline repair does not enter hashes or schedules.
 */
@Mixin(value = MinecraftServer.class, remap = false)
public abstract class MinecraftServerMixin {
	@Shadow
	private long nextTickTimeNanos;

	@Shadow
	private volatile boolean running;

	@Shadow
	private volatile boolean isSaving;

	@Inject(method = "processPacketsAndTick", at = @At("HEAD"), cancellable = true)
	private void simharness$gate(boolean sprinting, CallbackInfo callback) {
		Lockstep lockstep = Lockstep.INSTANCE;
		if (lockstep.isEnabled()) {
			if (lockstep.pendingTicks() == 0) {
				callback.cancel();
			} else {
				lockstep.consumePendingTick();
			}
		}
	}

	@ModifyArg(method = "processPacketsAndTick", at = @At(value = "INVOKE", target = "Lnet/minecraft/server/MinecraftServer;tickServer(Ljava/util/function/BooleanSupplier;)V"), index = 0)
	private BooleanSupplier simharness$noSpareTime(BooleanSupplier original) {
		return Lockstep.INSTANCE.isEnabled() ? () -> false : original;
	}

	@Inject(method = "waitUntilNextTick", at = @At("HEAD"), cancellable = true)
	private void simharness$runQueuedTicks(CallbackInfo callback) {
		if (Lockstep.INSTANCE.isEnabled() && Lockstep.INSTANCE.pendingTicks() > 0) {
			callback.cancel();
		}
	}

	// Wake the idle loop as soon as a step request has queued ticks: waitUntilNextTick blocks on !haveTime(),
	// so without this every step request pays up to one 50 ms tick interval of latency.
	@Inject(method = "haveTime", at = @At("HEAD"), cancellable = true)
	private void simharness$noWaitWhenPending(CallbackInfoReturnable<Boolean> callback) {
		// Also end waits at once after halt(), so shutdown is not delayed until a stale deadline; but not inside
		// stopServer (isSaving): its loop waits for the chunk maps to have no work, and only the tasks that
		// waitUntilNextTick runs while haveTime() holds can finish chunk loads still in flight. Ending that wait at once
		// starved them, so the loop spun forever at "Saving worlds" (incident D10).
		if (Lockstep.INSTANCE.isEnabled() && (Lockstep.INSTANCE.pendingTicks() > 0 || !this.running && !this.isSaving)) {
			callback.setReturnValue(false);
		}
	}

	@Inject(method = "autoSave", at = @At("HEAD"), cancellable = true)
	private void simharness$disableAutosave(CallbackInfo callback) {
		if (Lockstep.INSTANCE.isEnabled()) {
			callback.cancel();
		}
	}

	@Inject(method = "tickServer", at = @At("HEAD"))
	private void simharness$checkpoint(BooleanSupplier haveTime, CallbackInfo callback) {
		Determinism.beforeTick((MinecraftServer) (Object) this);
	}

	@Inject(method = "tickServer", at = @At("TAIL"))
	private void simharness$finishTick(BooleanSupplier haveTime, CallbackInfo callback) {
		if (Lockstep.INSTANCE.isEnabled()) {
			// The vanilla loop adds one tick interval to its deadline per iteration. Bursts of back-to-back lockstep ticks
			// would push the deadline far into the future (2,000 ticks in ~1 s puts it ~100 s ahead), so halt() goes
			// unnoticed and task gating (shouldRun -> haveTime) drifts. Re-anchor the deadline to real time after every tick.
			this.nextTickTimeNanos = Util.getNanos() + ((MinecraftServer) (Object) this).tickRateManager().nanosecondsPerTick();
		}
		MinecraftServer server = (MinecraftServer) (Object) this;
		HarnessState state = HarnessState.existing(server);
		if (state != null) {
			state.tickVillagers();
		}
		Lockstep.INSTANCE.afterTick(server);
	}
}
