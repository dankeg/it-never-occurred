package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.core.component.DataComponents;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.food.FoodProperties;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Exercises real inventory components and vanilla food restoration without a server or random consumption effects. */
final class SkillEatTest {
	/** Initializes actual vanilla food components once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Feasibility reads the first matching slot without consuming items or adding food. */
	@Test
	void planningDoesNotMutateInventory() {
		ItemStack bread = new ItemStack(Items.BREAD, 3);
		SimpleContainer inventory = new SimpleContainer(ItemStack.EMPTY, bread);
		SkillEat planned = SkillEat.plan(inventory, Items.BREAD);
		assertNull(planned.reason());
		assertEquals(1, planned.slot());
		assertEquals(5, planned.food().nutrition());
		assertSame(bread, inventory.getItem(1));
		assertEquals(3, bread.getCount());
	}

	/** A completed meal consumes exactly one item and uses vanilla nutrition and saturation. */
	@Test
	void breadRestoresVanillaFoodAndSaturation() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.BREAD, 2), new ItemStack(Items.BREAD, 3));
		FoodData food = hungry();
		SkillEat eaten = SkillEat.consume(inventory, Items.BREAD, food);
		assertNull(eaten.reason());
		assertEquals(9, food.getFoodLevel());
		assertEquals(6.0F, food.getSaturationLevel(), 0.0001F);
		assertEquals(1, inventory.getItem(0).getCount());
		assertEquals(3, inventory.getItem(1).getCount());
	}

	/** Actual stack components, not a hardcoded food table, determine restoration and vanilla caps it at twenty. */
	@Test
	void honorsStackFoodComponentsAndVanillaCap() {
		ItemStack bread = new ItemStack(Items.BREAD, 2);
		bread.set(DataComponents.FOOD, new FoodProperties(3, 1.5F, false));
		SimpleContainer inventory = new SimpleContainer(bread);
		FoodData food = hungry();
		SkillEat.consume(inventory, Items.BREAD, food);
		assertEquals(7, food.getFoodLevel());
		assertEquals(1.5F, food.getSaturationLevel(), 0.0001F);
		food.setFoodLevel(19);
		SkillEat.consume(inventory, Items.BREAD, food);
		assertEquals(20, food.getFoodLevel());
		assertTrue(inventory.getItem(0).isEmpty());
	}

	/** A nonfood component override cannot hide a later edible stack of the same item. */
	@Test
	void skipsMatchingStacksWithoutFoodComponents() {
		ItemStack inedible = new ItemStack(Items.BREAD, 1);
		inedible.remove(DataComponents.FOOD);
		SimpleContainer inventory = new SimpleContainer(inedible, new ItemStack(Items.BREAD, 1));
		FoodData food = hungry();
		assertEquals(1, SkillEat.consume(inventory, Items.BREAD, food).slot());
		assertSame(inedible, inventory.getItem(0));
		assertTrue(inventory.getItem(1).isEmpty());
		assertEquals(9, food.getFoodLevel());
		assertEquals("not_food", SkillEat.plan(inventory, Items.BREAD).reason());
	}

	/** Missing items and held nonfood fail without consuming anything or changing food. */
	@Test
	void failuresLeaveFoodAndInventoryUnchanged() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.STONE, 2));
		FoodData food = hungry();
		assertEquals("missing_items", SkillEat.consume(inventory, Items.BREAD, food).reason());
		assertEquals("not_food", SkillEat.consume(inventory, Items.STONE, food).reason());
		assertEquals(2, inventory.getItem(0).getCount());
		assertEquals(4, food.getFoodLevel());
		assertEquals(0.0F, food.getSaturationLevel());
	}

	/** Completion rechecks stock, so a removed meal cannot restore food after a successful dry run. */
	@Test
	void removedFoodCannotBeConsumedLater() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.BREAD, 1));
		FoodData food = hungry();
		assertNull(SkillEat.plan(inventory, Items.BREAD).reason());
		inventory.removeItem(0, 1);
		assertEquals("missing_items", SkillEat.consume(inventory, Items.BREAD, food).reason());
		assertEquals(4, food.getFoodLevel());
	}

	/** Provides a deterministic low-food state with no saturation to obscure the restoration check. */
	private static FoodData hungry() {
		FoodData food = new FoodData();
		food.setFoodLevel(4);
		food.setSaturation(0);
		return food;
	}
}
