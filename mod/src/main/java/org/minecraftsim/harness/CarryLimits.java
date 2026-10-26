package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.List;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.Container;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;

/** Checks complete gains before mutation and provides concrete capacity hints for the agent. */
public final class CarryLimits {
	private CarryLimits() {
	}

	/** Checks an exact incoming amount against all inventory slots, including equipment. */
	public static JsonObject check(Container inventory, Item item, int count, SimConfig config) {
		return check(item, SkillInventory.count(inventory, item), count, config);
	}

	/** Checks a simulated recipe's net gains, including remainders and multi-output recipes. */
	public static JsonObject check(List<ItemStack> before, List<ItemStack> after, SimConfig config) {
		for (String identifier : config.carryLimits().keySet()) {
			Item item = BuiltInRegistries.ITEM.getValue(net.minecraft.resources.Identifier.parse(identifier));
			int held = before.stream().filter(stack -> stack.is(item)).mapToInt(ItemStack::getCount).sum();
			int resulting = after.stream().filter(stack -> stack.is(item)).mapToInt(ItemStack::getCount).sum();
			JsonObject failure = check(item, held, resulting - held, config);
			if (failure != null) {
				return failure;
			}
		}
		return null;
	}

	/** Returns a detached failure payload or null; removals never fail an incoming-load rule. */
	private static JsonObject check(Item item, int held, int count, SimConfig config) {
		String identifier = BuiltInRegistries.ITEM.getKey(item).toString();
		JsonObject limits = config.carryLimits();
		if (count <= 0 || !limits.has(identifier) || (long) held + count <= limits.get(identifier).getAsInt()) {
			return null;
		}
		int limit = limits.get(identifier).getAsInt();
		JsonObject failure = new JsonObject();
		failure.addProperty("reason", "too_heavy");
		failure.addProperty("item", identifier);
		failure.addProperty("limit", limit);
		failure.addProperty("have", held);
		failure.addProperty("remaining", Math.max(0, limit - held));
		failure.addProperty("hint", "you can carry " + limit + " " + identifier.replace("minecraft:", "")
			+ " at most; you have " + held + "; you can take " + Math.max(0, limit - held) + " more");
		return failure;
	}
}
