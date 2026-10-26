package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.core.component.DataComponents;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.CustomData;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins the shared freshness rule, native component stacking and opt-in projections. */
final class FreshnessTest {
	/** Initializes vanilla stack components once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Bread remains fresh on its baking day and at the exact lifetime boundary. */
	@Test
	void strictBoundaryNegativeDaysAndUnstampedBread() {
		ItemStack bread = new ItemStack(Items.BREAD);
		assertEquals(0, Freshness.bakedDay(bread));
		assertFalse(Freshness.mouldy(bread, 0, 2));
		assertFalse(Freshness.mouldy(bread, 2, 2));
		assertTrue(Freshness.mouldy(bread, 3, 2));
		Freshness.stamp(bread, -3);
		assertEquals(-3, Freshness.bakedDay(bread));
		assertFalse(Freshness.mouldy(bread, -1, 2));
		assertTrue(Freshness.mouldy(bread, 0, 2));
		assertFalse(Freshness.mouldy(new ItemStack(Items.WHEAT), 100, 0));
	}

	/** Stamping keeps unrelated data and different baking days stay in distinct native stacks. */
	@Test
	void stampsPreserveComponentsAndDoNotStackAcrossDays() {
		ItemStack first = new ItemStack(Items.BREAD, 4);
		CustomData.update(DataComponents.CUSTOM_DATA, first, tag -> tag.putString("origin", "bakery"));
		Freshness.stamp(first, 1);
		ItemStack second = first.copy();
		Freshness.stamp(second, 2);
		assertEquals("bakery", second.get(DataComponents.CUSTOM_DATA).copyTag().getString("origin").orElseThrow());
		assertFalse(ItemStack.isSameItemSameComponents(first, second));
		assertTrue(ItemStack.isSameItemSameComponents(first, first.copy()));
		SimpleContainer source = new SimpleContainer(first, second);
		SimpleContainer destination = new SimpleContainer(3);
		assertNull(SkillInventory.transfer(source, destination, Items.BREAD, 8, true));
		assertEquals(1, Freshness.bakedDay(destination.getItem(0)));
		assertEquals(2, Freshness.bakedDay(destination.getItem(1)));
	}

	/** Inventory/container projections remain unchanged when disabled, including already stamped stock. */
	@Test
	void metadataAppearsOnlyWhenFreshnessIsEnabled() {
		SimConfig config = new SimConfig();
		ItemStack bread = new ItemStack(Items.BREAD);
		Freshness.stamp(bread, -4);
		JsonObject item = new JsonObject();
		Freshness.describe(bread, config, 0, item);
		assertEquals(new JsonObject(), item);
		config.update(JsonParser.parseString("{\"fresh_days\":2}").getAsJsonObject());
		Freshness.describe(bread, config, 0, item);
		assertEquals(JsonParser.parseString("{\"baked_day\":-4,\"mouldy\":true}"), item);
	}

	/** Both planning and completion refuse mouldy bread without changing hunger or consuming any loaf. */
	@Test
	void eatingRechecksAgeAndLegacyEatingIgnoresIt() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.BREAD, 2));
		FoodData food = new FoodData();
		food.setFoodLevel(4);
		assertNull(SkillEat.plan(inventory, Items.BREAD, 2, 2).reason());
		assertEquals("mouldy", SkillEat.consume(inventory, Items.BREAD, food, 2, 3).reason());
		assertEquals(2, inventory.getItem(0).getCount());
		assertEquals(4, food.getFoodLevel());
		assertNull(SkillEat.consume(inventory, Items.BREAD, food).reason());
		assertEquals(1, inventory.getItem(0).getCount());
	}
}
