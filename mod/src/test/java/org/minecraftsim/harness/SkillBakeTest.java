package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Exercises exact baking transactions, whole-batch timing and saved in-flight oven state. */
final class SkillBakeTest {
	/** Initializes vanilla components once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Four loaves use twelve wheat, take 160 ticks and retain the completion day component. */
	@Test
	void wholeBatchUsesExactWheatDurationAndStamp() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.WHEAT, 7), new ItemStack(Items.WHEAT, 8), ItemStack.EMPTY);
		SkillBake planned = SkillBake.plan(inventory, 4, config(), 3);
		assertNull(planned.reason());
		assertEquals(160, planned.ticks());
		assertEquals(15, SkillInventory.count(inventory, Items.WHEAT));
		assertEquals(0, SkillInventory.count(inventory, Items.BREAD));
		planned.commit(inventory);
		assertEquals(3, SkillInventory.count(inventory, Items.WHEAT));
		assertEquals(4, SkillInventory.count(inventory, Items.BREAD));
		assertEquals(3, Freshness.bakedDay(inventory.getItem(0)));
	}

	/** Cancelled staging consumes nothing; wheat and carrying failures cannot commit a partial batch. */
	@Test
	void failuresAndUncommittedPlansLeaveInputsUntouched() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.WHEAT, 12), new ItemStack(Items.BREAD, 30), ItemStack.EMPTY);
		SkillBake heavy = SkillBake.plan(inventory, 4, config(), 1);
		assertEquals("too_heavy", heavy.reason());
		assertThrows(IllegalStateException.class, () -> heavy.commit(inventory));
		assertEquals(12, SkillInventory.count(inventory, Items.WHEAT));
		assertEquals(30, SkillInventory.count(inventory, Items.BREAD));
		assertNull(SkillBake.plan(inventory, 2, config(), 1).reason());
		assertEquals(12, SkillInventory.count(inventory, Items.WHEAT));
		SimpleContainer shortWheat = new SimpleContainer(new ItemStack(Items.WHEAT, 11), ItemStack.EMPTY);
		assertEquals("not_enough_wheat", SkillBake.plan(shortWheat, 4, config(), 1).reason());
		assertEquals(11, SkillInventory.count(shortWheat, Items.WHEAT));
		assertThrows(IllegalArgumentException.class, () -> SkillBake.plan(inventory, 0, config(), 1));
		assertThrows(IllegalArgumentException.class, () -> SkillBake.plan(inventory, 33, config(), 1));
	}

	/** Checkpoints and forks retain timer, oven position and exact serialization without touching a world. */
	@Test
	void activeBakingRoundTripsItsInternalProgress() {
		Skills skills = new Skills(null);
		JsonObject saved = JsonParser.parseString("""
			{"sequence":1,"active":{"skillId":"baker:1","skill":"bake","args":{"count":4},"state":"running",
			"startedAt":100,"progress":{"ticks":80},"internal":{"elapsed":80,"estimate":160,"pos":[1,2,3]}},"last":null}
			""").getAsJsonObject();
		skills.restore(saved.toString());
		assertEquals(saved, JsonParser.parseString(skills.serialization()));
		assertFalse(skills.status().has("internal"));
		assertEquals("bake", skills.activity().get("skill").getAsString());
		assertEquals(JsonParser.parseString("[1,2,3]"), skills.activity().get("pos"));
		Skills fork = new Skills(null);
		fork.restore(skills.serialization());
		assertEquals(skills.serialization(), fork.serialization());
	}

	/** Missing bake_ticks keeps ovens off while its effective numeric default remains forty. */
	@Test
	void ovensAreExplicitlyOptIn() {
		SimConfig config = new SimConfig();
		assertFalse(config.bakingEnabled());
		assertEquals(40, config.integer("bake_ticks"));
		config.update(JsonParser.parseString("{\"bake_ticks\":40}").getAsJsonObject());
		assertTrue(config.bakingEnabled());
	}

	/** The same light transition used at start, completion and cancellation preserves unrelated block states. */
	@Test
	void ovenLightsAndUnlightsWithoutReplacingOtherBlocks() {
		BlockState oven = Blocks.SMOKER.defaultBlockState();
		assertFalse(oven.getValue(BlockStateProperties.LIT));
		BlockState lit = SkillBake.light(oven, true);
		assertTrue(lit.getValue(BlockStateProperties.LIT));
		assertTrue(lit.getLightEmission() > 0);
		assertEquals(oven, SkillBake.light(lit, false));
		assertEquals(Blocks.STONE.defaultBlockState(), SkillBake.light(Blocks.STONE.defaultBlockState(), false));
	}

	/** Configures the opt-in oven and the scenario's maximum bread load. */
	private static SimConfig config() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"bake_ticks\":40,\"carry_limits\":{\"minecraft:bread\":32}}").getAsJsonObject());
		return config;
	}
}
