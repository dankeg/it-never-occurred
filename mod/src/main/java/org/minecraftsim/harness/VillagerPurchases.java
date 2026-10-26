package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import net.minecraft.world.Container;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;

/** Stages exact-count market sales and their payment without RNG, NPC inventories, or server dependencies. */
final class VillagerPurchases {
	private VillagerPurchases() {
	}

	/** A completed market trip, ordered by arrival tick and then the raw scripted villager ID. */
	record Arrival(long tick, String npc) implements Comparable<Arrival> {
		@Override
		public int compareTo(Arrival other) {
			int byTick = Long.compare(tick, other.tick);
			return byTick != 0 ? byTick : npc.compareTo(other.npc);
		}
	}

	/** Only unavailable stock makes sold-out news; a chest unable to hold payment stays unchanged. */
	enum Outcome {
		PURCHASED, SOLD_OUT, NO_PAYMENT_ROOM
	}

	/**
	 * Removes freshest eligible goods first, breaking equal-day ties by slot, and deposits the full emerald price.
	 * Stock removal and payment share a copied inventory and commit together only on complete success.
	 * @param shop stall container and its placement/stack-size policy
	 * @param item configured item identity
	 * @param count positive exact quantity requested
	 * @param price nonnegative emerald price per unit
	 * @param today arrival's game day
	 * @param freshDays nullable freshness limit; null disables mould checks
	 * @return the sale result, with no changes on either failure
	 */
	static Outcome purchase(Container shop, Item item, int count, int price, long today, Integer freshDays) {
		if (count <= 0 || price < 0) {
			throw new IllegalArgumentException("Purchase count must be positive and price nonnegative");
		}
		List<ItemStack> slots = SkillInventory.copy(shop);
		List<Integer> candidates = new ArrayList<>();
		for (int slot = 0; slot < slots.size(); slot++) {
			ItemStack stack = slots.get(slot);
			if (stack.is(item) && shop.canTakeItem(shop, slot, stack)
					&& (freshDays == null || !Freshness.mouldy(stack, today, freshDays))) {
				candidates.add(slot);
			}
		}
		candidates.sort(Comparator.comparingLong((Integer slot) -> Freshness.bakedDay(slots.get(slot)))
			.reversed().thenComparingInt(Integer::intValue));
		int remaining = count;
		for (int slot : candidates) {
			ItemStack stack = slots.get(slot);
			int amount = Math.min(remaining, stack.getCount());
			stack.shrink(amount);
			remaining -= amount;
			if (remaining == 0) {
				break;
			}
		}
		if (remaining != 0) {
			return Outcome.SOLD_OUT;
		}
		long cost = (long) price * count;
		if (cost > Integer.MAX_VALUE || cost > 0
				&& !SkillInventory.insert(shop, slots, new ItemStack(Items.EMERALD, (int) cost))) {
			return Outcome.NO_PAYMENT_ROOM;
		}
		SkillInventory.commit(shop, slots);
		return Outcome.PURCHASED;
	}
}
