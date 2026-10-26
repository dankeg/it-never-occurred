package org.minecraftsim.harness.mixin;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityReference;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import org.minecraftsim.harness.ItemPickup;
import org.minecraftsim.harness.CarryLimits;
import org.minecraftsim.harness.HarnessState;
import org.minecraftsim.harness.SimBot;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Records actual vanilla pickup deltas, including partial insertions that return false, without changing them. */
@Mixin(value = ItemEntity.class, remap = false)
public abstract class ItemEntityMixin {
	@Shadow
	private EntityReference<Entity> thrower;

	/**
	 * A role agent takes back an item it dropped only deliberately, with `collect`. The drop skill leaves the item at the
	 * dropper's feet (vanilla throws it forward with random spread), so walking over or standing on it would otherwise
	 * undo the drop as soon as the pickup delay ends. Everyone else picks it up as in vanilla.
	 */
	@Inject(method = "playerTouch", at = @At("HEAD"), cancellable = true)
	private void simharness$ownDrop(Player player, CallbackInfo callback) {
		if (player instanceof SimBot bot && thrower != null && bot.getUUID().equals(thrower.getUUID())
				&& !bot.skills().collecting()) {
			callback.cancel();
		}
	}

	/** Wraps only the inventory insertion; vanilla retains its return value, pickup checks and subsequent effects. */
	@Redirect(method = "playerTouch", at = @At(value = "INVOKE",
		target = "Lnet/minecraft/world/entity/player/Inventory;add(Lnet/minecraft/world/item/ItemStack;)Z"))
	private boolean simharness$pickup(Inventory inventory, ItemStack stack, Player player) {
		if (player instanceof SimBot bot) {
			HarnessState state = HarnessState.existing(bot.level().getServer());
			if (state != null && CarryLimits.check(inventory, stack.getItem(), stack.getCount(), state.config()) != null) {
				return false;
			}
		}
		Item item = stack.getItem();
		int before = stack.getCount();
		boolean result = inventory.add(stack);
		int count = before - stack.getCount();
		if (count > 0 && player instanceof SimBot bot) {
			ItemEntity entity = (ItemEntity) (Object) this;
			new ItemPickup(item, count, entity.getUUID(), entity.position(), thrower == null ? null : thrower.getUUID()).record(bot);
		}
		return result;
	}
}
