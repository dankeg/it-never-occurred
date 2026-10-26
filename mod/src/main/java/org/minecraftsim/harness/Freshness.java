package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import net.minecraft.core.component.DataComponents;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.CustomData;

/** Owns bread's day stamp, strict mould boundary and opt-in observation metadata. */
public final class Freshness {
	public static final String BAKED_DAY = "simharness_baked_day";

	private Freshness() {
	}

	/** Reads the integer baking day; bread without a stamp was baked on day zero. */
	public static int bakedDay(ItemStack stack) {
		return stack.getOrDefault(DataComponents.CUSTOM_DATA, CustomData.EMPTY).copyTag().getIntOr(BAKED_DAY, 0);
	}

	/** Tests the strict age boundary, allowing stamps before day zero. */
	public static boolean mouldy(ItemStack stack, long today, int freshDays) {
		return stack.is(Items.BREAD) && today - bakedDay(stack) > freshDays;
	}

	/** Stamps bread without discarding unrelated custom data or other stack components. */
	public static void stamp(ItemStack stack, long day) {
		if (stack.is(Items.BREAD)) {
			int baked = Math.toIntExact(day);
			CustomData.update(DataComponents.CUSTOM_DATA, stack, tag -> tag.putInt(BAKED_DAY, baked));
		}
	}

	/** Adds per-stack freshness only when the lifetime rule is enabled. */
	public static void describe(ItemStack stack, SimConfig config, long today, JsonObject result) {
		Integer days = config.freshDays();
		if (days != null && stack.is(Items.BREAD)) {
			result.addProperty("baked_day", bakedDay(stack));
			result.addProperty("mouldy", mouldy(stack, today, days));
		}
	}
}
