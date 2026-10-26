package org.minecraftsim.harness.mixin;

import java.util.ArrayDeque;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import org.minecraftsim.harness.HarnessState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Captures successful overworld block changes for recording and NPC invalidation.
 * Tracking is server-main-thread-only and uses a stack to retain nested setBlock ordering.
 */
@Mixin(value = Level.class, remap = false)
public abstract class LevelSetBlockMixin {
	@Unique
	private final ArrayDeque<BlockState[]> simharness$changes = new ArrayDeque<>();

	@Unique
	private HarnessState simharness$state() {
		if ((Object) this instanceof ServerLevel level && level == level.getServer().overworld() && level.getServer().isSameThread()) {
			return HarnessState.existing(level.getServer());
		}
		return null;
	}

	@Inject(method = "setBlock(Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;II)Z", at = @At("HEAD"))
	private void simharness$begin(BlockPos position, BlockState block, int flags, int limit, CallbackInfoReturnable<Boolean> callback) {
		if (simharness$state() != null) {
			simharness$changes.push(new BlockState[1]);
		}
	}

	@Redirect(method = "setBlock(Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;II)Z",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/world/level/chunk/LevelChunk;setBlockState(Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;I)Lnet/minecraft/world/level/block/state/BlockState;"))
	private BlockState simharness$capture(LevelChunk chunk, BlockPos position, BlockState block, int flags) {
		BlockState[] pending = simharness$state() == null ? null : simharness$changes.peek();
		BlockState previous = chunk.setBlockState(position, block, flags);
		if (pending != null) {
			pending[0] = previous;
		}
		return previous;
	}

	@Inject(method = "setBlock(Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;II)Z", at = @At("RETURN"))
	private void simharness$finish(BlockPos position, BlockState block, int flags, int limit, CallbackInfoReturnable<Boolean> callback) {
		HarnessState state = simharness$state();
		if (state != null) {
			BlockState[] pending = simharness$changes.pop();
			if (callback.getReturnValueZ() && pending[0] != null) {
				state.invalidateNightSight();
			}
			if ((state.recording() || state.villagers() != null || state.collapseConfigured())
					&& callback.getReturnValueZ() && pending[0] != null) {
				state.blockChange(position, pending[0], ((Level) (Object) this).getBlockState(position));
			}
		}
	}
}
