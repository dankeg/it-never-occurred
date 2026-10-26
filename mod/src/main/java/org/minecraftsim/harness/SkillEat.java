package org.minecraftsim.harness;

import net.minecraft.core.component.DataComponents;
import net.minecraft.world.Container;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.food.FoodProperties;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;

/** A single food item selected without mutation, or a feasibility failure. */
record SkillEat(String reason, int slot, FoodProperties food) {
	/** Checks actual stack components in ascending slot order, without consuming food or changing hunger. */
	static SkillEat plan(Container inventory, Item item) {
		return plan(inventory, item, null, 0);
	}

	/** Applies the optional freshness rule to the same first matching food stack used by legacy eating. */
	static SkillEat plan(Container inventory, Item item, Integer freshDays, long today) {
		boolean found = false;
		for (int slot = 0; slot < inventory.getContainerSize(); slot++) {
			ItemStack stack = inventory.getItem(slot);
			if (!stack.isEmpty() && stack.is(item)) {
				found = true;
				FoodProperties food = stack.get(DataComponents.FOOD);
				if (food != null) {
					if (freshDays != null && Freshness.mouldy(stack, today, freshDays)) {
						return new SkillEat("mouldy", -1, null);
					}
					return new SkillEat(null, slot, food);
				}
			}
		}
		return new SkillEat(found ? "not_food" : "missing_items", -1, null);
	}

	/** Rechecks the current inventory and consumes exactly one item, restoring vanilla nutrition and saturation. */
	static SkillEat consume(Container inventory, Item item, FoodData foodData) {
		return consume(inventory, item, foodData, null, 0);
	}

	/** Rechecks freshness at completion, so a meal crossing the day boundary cannot eat mouldy bread. */
	static SkillEat consume(Container inventory, Item item, FoodData foodData, Integer freshDays, long today) {
		SkillEat checked = plan(inventory, item, freshDays, today);
		if (checked.reason() == null) {
			inventory.removeItem(checked.slot(), 1);
			inventory.setChanged();
			foodData.eat(checked.food());
		}
		return checked;
	}
}
