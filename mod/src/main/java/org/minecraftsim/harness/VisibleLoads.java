package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.Map;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.Container;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.entity.player.Player;

/** Reports one deterministic public load band, never inventory quantities or slots. */
final class VisibleLoads {
	private VisibleLoads() {
	}

	/** Describes a person's first qualifying item in registry-name order, or no load. */
	static JsonObject seen(Entity person, SimConfig config) {
		Container inventory = person instanceof Player player ? player.getInventory()
			: person instanceof Villager villager ? villager.getInventory() : null;
		return inventory == null ? null : seen(inventory, config.visibleLoads());
	}

	/** Sums all inventory slots while exposing only the configured item and sack band. */
	static JsonObject seen(Container inventory, Map<String, Integer> thresholds) {
		for (var threshold : new java.util.TreeMap<>(thresholds).entrySet()) {
			long count = 0;
			for (int slot = 0; slot < inventory.getContainerSize(); slot++) {
				var stack = inventory.getItem(slot);
				if (!stack.isEmpty() && BuiltInRegistries.ITEM.getKey(stack.getItem()).toString().equals(threshold.getKey())) {
					count += stack.getCount();
				}
			}
			if (count >= threshold.getValue()) {
				JsonObject result = new JsonObject();
				result.addProperty("item", threshold.getKey());
				result.addProperty("count_band", "sack");
				return result;
			}
		}
		return null;
	}
}
