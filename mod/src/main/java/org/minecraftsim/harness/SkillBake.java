package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.List;
import net.minecraft.world.Container;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;

/** A complete, component-preserving wheat-to-bread transaction, committed only after the oven timer. */
record SkillBake(String reason, long ticks, List<ItemStack> inventory, JsonObject details) {
	/** Stages the whole batch without taking wheat, accepting partial output or mutating inventory. */
	static SkillBake plan(Container inventory, int count, SimConfig config, long day) {
		if (count < 1 || count > 32) {
			throw new IllegalArgumentException("count must be 1..32");
		}
		JsonObject failure = CarryLimits.check(inventory, Items.BREAD, count, config);
		if (failure != null) {
			return new SkillBake("too_heavy", 0, List.of(), failure);
		}
		int needed = count * 3;
		if (SkillInventory.count(inventory, Items.WHEAT) < needed) {
			return new SkillBake("not_enough_wheat", 0, List.of(), new JsonObject());
		}
		List<ItemStack> staged = SkillInventory.copyAll(inventory);
		for (ItemStack stack : staged) {
			if (stack.is(Items.WHEAT)) {
				int used = Math.min(needed, stack.getCount());
				stack.shrink(used);
				needed -= used;
			}
		}
		ItemStack bread = new ItemStack(Items.BREAD, count);
		Freshness.stamp(bread, day);
		if (needed > 0 || !SkillInventory.insert(inventory, staged.subList(0, SkillInventory.size(inventory)), bread)) {
			return new SkillBake(needed > 0 ? "not_enough_wheat" : "inventory_full", 0, List.of(), new JsonObject());
		}
		return new SkillBake(null, (long) count * config.integer("bake_ticks"), staged, new JsonObject());
	}

	/** Writes the prechecked batch once, leaving cancelled or failed plans untouched. */
	void commit(Container destination) {
		if (reason != null) {
			throw new IllegalStateException("Cannot commit a failed bake");
		}
		SkillInventory.commit(destination, inventory);
	}

	/** Toggles only an oven's light; cancellation must never overwrite a replacement block. */
	static BlockState light(BlockState current, boolean lit) {
		return current.is(Blocks.SMOKER) ? current.setValue(BlockStateProperties.LIT, lit) : current;
	}
}
