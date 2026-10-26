package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.List;
import net.minecraft.world.Container;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;

/**
 * Stages and commits inventory transfers using deterministic ascending slot order.
 * Simulation uses copied stacks; real containers change only through explicit commit paths.
 */
public final class SkillInventory {
	private static final int HOTBAR_SIZE = 9;

	private SkillInventory() {
	}

	/**
	 * Selects a matching hotbar stack, or swaps the first inventory match into the selected hotbar slot.
	 * The selected stack takes priority, then ascending hotbar and inventory slots. Dry runs never mutate stacks.
	 * @param inventory inventory whose complete stacks and components must be preserved
	 * @param item item identity to hold
	 * @param selected current hotbar selection
	 * @param commit whether to perform a required swap
	 * @return resulting hotbar selection, or -1 when the item is absent
	 */
	public static int hold(Container inventory, Item item, int selected, boolean commit) {
		if (inventory.getItem(selected).is(item)) {
			return selected;
		}
		for (int slot = 0; slot < inventory.getContainerSize(); slot++) {
			ItemStack stack = inventory.getItem(slot);
			if (stack.isEmpty() || !stack.is(item)) {
				continue;
			}
			if (slot < HOTBAR_SIZE) {
				return slot;
			}
			if (commit) {
				ItemStack previous = inventory.getItem(selected);
				inventory.setItem(selected, stack);
				inventory.setItem(slot, previous);
				inventory.setChanged();
			}
			return selected;
		}
		return -1;
	}

	/**
	 * Determines insertion capacity, excluding player equipment slots.
	 * @param container inventory or ordinary container
	 * @return number of slots considered for insertion
	 */
	public static int size(Container container) {
		return container instanceof Inventory inventory ? inventory.getNonEquipmentItems().size() : container.getContainerSize();
	}

	/**
	 * Copies all insertion slots without changing their stack components.
	 * @param container source container
	 * @return mutable slot list holding independent stack copies
	 */
	public static List<ItemStack> copy(Container container) {
		return copy(container, size(container));
	}

	/** Copies every reported slot when a transaction may consume ingredients held in equipment slots. */
	public static List<ItemStack> copyAll(Container container) {
		return copy(container, container.getContainerSize());
	}

	private static List<ItemStack> copy(Container container, int size) {
		List<ItemStack> result = new ArrayList<>();
		for (int slot = 0; slot < size; slot++) {
			result.add(container.getItem(slot).copy());
		}
		return result;
	}

	/**
	 * Counts matching items across the container's full reported size.
	 * @param container source container, including equipment when reported by its size
	 * @param item item identity to count
	 * @return total matching stack count
	 */
	public static int count(Container container, Item item) {
		int count = 0;
		for (int slot = 0; slot < container.getContainerSize(); slot++) {
			ItemStack stack = container.getItem(slot);
			if (stack.is(item)) {
				count += stack.getCount();
			}
		}
		return count;
	}

	/**
	 * Stages an insertion in slot order using container placement and stack-size rules.
	 * Failure may leave the staging list partially filled; neither container nor offered stack is changed.
	 * @param container placement policy and stack-limit source
	 * @param slots mutable staged slot contents
	 * @param offered stack whose item/components/count are offered
	 * @return whether every offered item fit in the staging list
	 */
	public static boolean insert(Container container, List<ItemStack> slots, ItemStack offered) {
		int remaining = offered.getCount();
		for (int slot = 0; slot < slots.size() && remaining > 0; slot++) {
			ItemStack current = slots.get(slot);
			if (!container.canPlaceItem(slot, offered) || !current.isEmpty() && !ItemStack.isSameItemSameComponents(current, offered)) {
				continue;
			}
			int available = container.getMaxStackSize(offered) - current.getCount();
			int amount = Math.min(remaining, Math.max(0, available));
			if (amount > 0) {
				if (current.isEmpty()) {
					slots.set(slot, offered.copyWithCount(amount));
				} else {
					current.grow(amount);
				}
				remaining -= amount;
			}
		}
		return remaining == 0;
	}

	/**
	 * Writes the staged slots in order, then marks the container changed.
	 * @param container destination container
	 * @param slots replacement contents for the leading slots
	 */
	public static void commit(Container container, List<ItemStack> slots) {
		for (int slot = 0; slot < slots.size(); slot++) {
			container.setItem(slot, slots.get(slot));
		}
		container.setChanged();
	}

	/**
	 * Simulates a permission-aware transfer and optionally commits only after complete success.
	 * @param from source container
	 * @param to destination container
	 * @param item requested item identity
	 * @param count number of items requested by the already validated caller
	 * @param commit whether to apply the successful staged source and destination contents
	 * @return null on success, otherwise inventory_full or missing_items without container mutation
	 */
	public static String transfer(Container from, Container to, Item item, int count, boolean commit) {
		List<ItemStack> source = copy(from, from.getContainerSize());
		List<ItemStack> destination = copy(to);
		int remaining = count;
		for (int slot = 0; slot < source.size() && remaining > 0; slot++) {
			ItemStack stack = source.get(slot);
			if (!stack.is(item) || !from.canTakeItem(to, slot, stack)) {
				continue;
			}
			int amount = Math.min(remaining, stack.getCount());
			if (!insert(to, destination, stack.copyWithCount(amount))) {
				return "inventory_full";
			}
			stack.shrink(amount);
			remaining -= amount;
		}
		if (remaining > 0) {
			return "missing_items";
		}
		if (commit) {
			commit(from, source);
			commit(to, destination);
		}
		return null;
	}

	/** Applies an optional destination-bot load policy before staging a complete transfer. */
	public static String transfer(Container from, Container to, Item item, int count, SimConfig config, boolean commit) {
		if (config != null && CarryLimits.check(to, item, count, config) != null) {
			return "too_heavy";
		}
		return transfer(from, to, item, count, commit);
	}
}
