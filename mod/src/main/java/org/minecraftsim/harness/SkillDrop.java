package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.List;
import net.minecraft.world.Container;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;

/** A single dropped stack and staged inventory, or a failure that leaves the real inventory unchanged. */
record SkillDrop(String reason, ItemStack stack, List<ItemStack> slots) {

	/**
	 * Stages exactly the requested count from the first component-compatible group large enough to supply it.
	 * All slots, including equipment, follow the same ascending order as ordinary inventory transfers.
	 * Different components cannot share one item entity, and its stack must fit vanilla's persistent count codec.
	 * @param inventory source inventory, read only during planning
	 * @param item requested item identity
	 * @param count positive requested amount
	 * @return an independent stack and slot copies, or missing_items, stack_too_large or incompatible_items
	 * @throws IllegalArgumentException if the count is not positive
	 */
	static SkillDrop plan(Container inventory, Item item, int count) {
		if (count < 1) {
			throw new IllegalArgumentException("count must be positive");
		}
		if (SkillInventory.count(inventory, item) < count) {
			return failed("missing_items");
		}
		if (count > Item.ABSOLUTE_MAX_STACK_SIZE) {
			return failed("stack_too_large");
		}
		for (int candidate = 0; candidate < inventory.getContainerSize(); candidate++) {
			ItemStack exemplar = inventory.getItem(candidate);
			if (!exemplar.is(item)) {
				continue;
			}
			List<ItemStack> slots = new ArrayList<>();
			int remaining = count;
			for (int slot = 0; slot < inventory.getContainerSize(); slot++) {
				ItemStack stack = inventory.getItem(slot).copy();
				if (remaining > 0 && ItemStack.isSameItemSameComponents(exemplar, stack)) {
					int amount = Math.min(remaining, stack.getCount());
					stack.shrink(amount);
					remaining -= amount;
				}
				slots.add(stack);
			}
			if (remaining == 0) {
				return new SkillDrop(null, exemplar.copyWithCount(count), slots);
			}
		}
		return failed("incompatible_items");
	}

	/** Returns an empty staging result without allocating an entity or changing the source inventory. */
	private static SkillDrop failed(String reason) {
		return new SkillDrop(reason, ItemStack.EMPTY, List.of());
	}
}
