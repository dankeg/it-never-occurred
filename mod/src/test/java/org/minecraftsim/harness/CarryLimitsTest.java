package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.List;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins all-stack capacities, atomic transfers and exact observation-side capacity hints. */
final class CarryLimitsTest {
	/** Initializes vanilla item identity once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Incoming chest, give and purchase transfers reject the whole request without moving source or destination. */
	@Test
	void transfersNeverSilentlyTakeAPartialLoad() {
		SimConfig config = limits();
		for (boolean commit : new boolean[]{false, true}) {
			SimpleContainer source = new SimpleContainer(new ItemStack(Items.BREAD, 8));
			SimpleContainer destination = new SimpleContainer(new ItemStack(Items.BREAD, 20), new ItemStack(Items.BREAD, 10));
			assertEquals("too_heavy", SkillInventory.transfer(source, destination, Items.BREAD, 3, config, commit));
			assertEquals(8, SkillInventory.count(source, Items.BREAD));
			assertEquals(30, SkillInventory.count(destination, Items.BREAD));
			assertNull(SkillInventory.transfer(source, destination, Items.BREAD, 2, config, commit));
			assertEquals(commit ? 32 : 30, SkillInventory.count(destination, Items.BREAD));
		}
	}

	/** Collect and vanilla pickup use the same whole-entity check and exact remaining-capacity payload. */
	@Test
	void pickupAndCollectUseAllStacksAndConcreteHints() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.BREAD, 16), new ItemStack(Items.BREAD, 14));
		JsonObject failure = CarryLimits.check(inventory, Items.BREAD, 64, limits());
		assertEquals("too_heavy", failure.get("reason").getAsString());
		assertEquals(32, failure.get("limit").getAsInt());
		assertEquals(30, failure.get("have").getAsInt());
		assertEquals(2, failure.get("remaining").getAsInt());
		assertTrue(failure.get("hint").getAsString().contains("you can carry 32 bread at most; you have 30"));
		assertEquals(30, SkillInventory.count(inventory, Items.BREAD));
		assertNull(CarryLimits.check(inventory, Items.EMERALD, Integer.MAX_VALUE, limits()));
		assertNull(CarryLimits.check(inventory, Items.BREAD, Integer.MAX_VALUE, new SimConfig()));
	}

	/** Crafting checks net recipe output and remainders rather than incorrectly limiting consumed ingredients. */
	@Test
	void stagedRecipeGainsAndRemovalsRespectCapacity() {
		List<ItemStack> before = List.of(new ItemStack(Items.WHEAT, 96), new ItemStack(Items.BREAD, 30));
		List<ItemStack> after = List.of(new ItemStack(Items.WHEAT, 87), new ItemStack(Items.BREAD, 33));
		assertEquals("too_heavy", CarryLimits.check(before, after, limits()).get("reason").getAsString());
		assertNull(CarryLimits.check(after, before, limits()));
		assertNull(CarryLimits.check(before, after, new SimConfig()));
	}

	/** Builds the Jubilee's two independent opt-in limits. */
	private static SimConfig limits() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"carry_limits\":{\"minecraft:bread\":32,\"minecraft:wheat\":96}}").getAsJsonObject());
		return config;
	}
}
