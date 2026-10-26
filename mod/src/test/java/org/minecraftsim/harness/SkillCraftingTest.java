package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.ItemStackTemplate;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.crafting.CraftingBookCategory;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.item.crafting.Recipe;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.ShapelessRecipe;
import net.minecraft.world.item.crafting.ShapedRecipe;
import net.minecraft.world.item.crafting.ShapedRecipePattern;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins recipe feasibility, exact material bills and rejection reasons using world-independent vanilla recipes. */
final class SkillCraftingTest {
	/** Initializes registries once for vanilla ingredients, recipe displays and output stacks. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Exactly enough ingredients produce the requested crafts, timing and material bill without consumption. */
	@Test
	void recipeAcceptsExactlyEnoughMaterialsAndReturnsTheExactBill() {
		List<ItemStack> inventory = inventory(4);
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 8,
			10, false, inventory, 64, null);
		assertEquals(JsonParser.parseString("""
			{"ticks":20,"crafts":2,"outputPerCraft":4,"needsTable":false,
			 "materials":{"minecraft:oak_planks":4},"recipe":"test:sticks"}
			"""), result);
		assertTrue(inventory.getFirst().is(Items.OAK_PLANKS));
		assertEquals(4, inventory.getFirst().getCount());
		for (int slot = 1; slot < inventory.size(); slot++) {
			assertTrue(inventory.get(slot).isEmpty());
		}
	}

	/** Being one ingredient short rejects the whole multi-craft plan with missing_items and no partial bill. */
	@Test
	void recipeRejectsOneItemShortWithTheExactMissingItemsReason() {
		List<ItemStack> inventory = inventory(3);
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 8,
			10, false, inventory, 64, null);
		assertEquals(JsonParser.parseString("""
			{"reason":"missing_items","ticks":0,"materials":{},"recipe":"test:sticks",
			 "crafts":0,"outputPerCraft":0,"needsTable":false}
			"""), result);
		assertTrue(inventory.getFirst().is(Items.OAK_PLANKS));
		assertEquals(3, inventory.getFirst().getCount());
		for (int slot = 1; slot < inventory.size(); slot++) {
			assertTrue(inventory.get(slot).isEmpty());
		}
	}

	/** Ingredients spread across slots are allocated once each and counted under the same material key. */
	@Test
	void recipeCombinesIngredientsAcrossInventorySlots() {
		List<ItemStack> inventory = inventory(1);
		inventory.set(1, new ItemStack(Items.OAK_PLANKS, 1));
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 4,
			10, false, inventory, 64, null);
		assertEquals(JsonParser.parseString("""
			{"ticks":10,"crafts":1,"outputPerCraft":4,"needsTable":false,
			 "materials":{"minecraft:oak_planks":2},"recipe":"test:sticks"}
			"""), result);
		assertEquals(1, inventory.get(0).getCount());
		assertEquals(1, inventory.get(1).getCount());
	}

	/** Enough ingredients still fail with inventory_full when no slot can hold the output. */
	@Test
	void recipeRejectsOutputThatCannotFitWithTheExactInventoryFullReason() {
		List<ItemStack> inventory = inventory(3);
		for (int slot = 1; slot < inventory.size(); slot++) {
			inventory.set(slot, new ItemStack(Items.STONE, 64));
		}
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 4,
			10, false, inventory, 64, null);
		assertEquals(JsonParser.parseString("""
			{"reason":"inventory_full","ticks":0,"materials":{},"recipe":"test:sticks",
			 "crafts":0,"outputPerCraft":0,"needsTable":false}
			"""), result);
		assertEquals(3, inventory.getFirst().getCount());
		for (int slot = 1; slot < inventory.size(); slot++) {
			assertTrue(inventory.get(slot).is(Items.STONE));
			assertEquals(64, inventory.get(slot).getCount());
		}
	}

	/** Equivalent matching recipes are selected by identifier rather than collection insertion order. */
	@Test
	void equivalentRecipesAreSelectedInIdentifierOrder() {
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:z_sticks"), stickRecipe("test:a_sticks")),
			Items.STICK, 4, 10, false, inventory(2), 64, null);
		assertEquals(JsonParser.parseString("""
			{"ticks":10,"crafts":1,"outputPerCraft":4,"needsTable":false,
			 "materials":{"minecraft:oak_planks":2},"recipe":"test:a_sticks"}
			"""), result);
	}

	private static List<ItemStack> inventory(int planks) {
		List<ItemStack> inventory = new ArrayList<>(Collections.nCopies(Inventory.INVENTORY_SIZE, ItemStack.EMPTY));
		inventory.set(0, new ItemStack(Items.OAK_PLANKS, planks));
		return inventory;
	}

	/** Recipe simulation refuses an over-limit batch before any ingredient can be consumed. */
	@Test
	void wholeCraftOutputRespectsCarryingLimits() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"carry_limits\":{\"minecraft:stick\":6}}").getAsJsonObject());
		List<ItemStack> inventory = inventory(4);
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 8,
			10, false, inventory, 64, null, config, 0);
		assertEquals("too_heavy", result.get("reason").getAsString());
		assertEquals(4, inventory.getFirst().getCount());
		assertEquals(0, result.get("have").getAsInt());
		assertEquals(6, result.get("remaining").getAsInt());
	}

	/** The load rule is checked before ordinary slot capacity and includes recipe-size rounding. */
	@Test
	void craftingRejectsOverweightOutputEvenWhenSlotsAreFull() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"carry_limits\":{\"minecraft:stick\":3}}").getAsJsonObject());
		List<ItemStack> inventory = inventory(4);
		for (int slot = 1; slot < inventory.size(); slot++) {
			inventory.set(slot, new ItemStack(Items.STONE, 64));
		}
		JsonObject result = SkillCrafting.planRecipes(List.of(stickRecipe("test:sticks")), Items.STICK, 1,
			10, false, inventory, 64, null, config, 0);
		assertEquals("too_heavy", result.get("reason").getAsString());
		assertEquals(0, result.get("have").getAsInt());
		assertEquals(4, inventory.getFirst().getCount());
	}

	/** Vanilla-sized shaped furnace and smoker recipes and the nine-wheat hay conversion need no special cases. */
	@Test
	void ovenRecipeChainAndHayConversionAreCraftable() {
		Recipe.CommonInfo common = new Recipe.CommonInfo(true);
		CraftingRecipe.CraftingBookInfo book = new CraftingRecipe.CraftingBookInfo(CraftingBookCategory.MISC, "");
		ShapedRecipe furnace = new ShapedRecipe(common, book,
			ShapedRecipePattern.of(Map.of('#', Ingredient.of(Items.COBBLESTONE)), "###", "# #", "###"),
			new ItemStackTemplate(Items.FURNACE));
		ShapedRecipe smoker = new ShapedRecipe(common, book,
			ShapedRecipePattern.of(Map.of('#', Ingredient.of(Items.OAK_LOG), 'F', Ingredient.of(Items.FURNACE)), " # ", "#F#", " # "),
			new ItemStackTemplate(Items.SMOKER));
		ShapelessRecipe wheat = new ShapelessRecipe(common, book, new ItemStackTemplate(Items.WHEAT, 9), List.of(Ingredient.of(Items.HAY_BLOCK)));
		List<ItemStack> inventory = inventory(0);
		inventory.set(0, new ItemStack(Items.COBBLESTONE, 8));
		JsonObject furnacePlan = SkillCrafting.planRecipes(List.of(holder("test:furnace", furnace)), Items.FURNACE, 1,
			10, true, inventory, 64, null);
		assertTrue(!furnacePlan.has("reason"), furnacePlan.toString());
		assertTrue(furnacePlan.get("needsTable").getAsBoolean());
		inventory.set(0, new ItemStack(Items.FURNACE));
		inventory.set(1, new ItemStack(Items.OAK_LOG, 4));
		JsonObject smokerPlan = SkillCrafting.planRecipes(List.of(holder("test:smoker", smoker)), Items.SMOKER, 1,
			10, true, inventory, 64, null);
		assertTrue(!smokerPlan.has("reason"), smokerPlan.toString());
		inventory = inventory(0);
		inventory.set(0, new ItemStack(Items.HAY_BLOCK));
		JsonObject wheatPlan = SkillCrafting.planRecipes(List.of(holder("test:wheat", wheat)), Items.WHEAT, 9,
			10, false, inventory, 64, null);
		assertTrue(!wheatPlan.has("reason"), wheatPlan.toString());
		assertEquals(9, wheatPlan.get("outputPerCraft").getAsInt());
	}

	/** Wraps a vanilla crafting recipe with a stable registry identifier. */
	private static RecipeHolder<?> holder(String id, CraftingRecipe recipe) {
		return new RecipeHolder<>(ResourceKey.create(Registries.RECIPE, Identifier.parse(id)), recipe);
	}

	private static RecipeHolder<?> stickRecipe(String id) {
		ShapelessRecipe recipe = new ShapelessRecipe(new Recipe.CommonInfo(true),
			new CraftingRecipe.CraftingBookInfo(CraftingBookCategory.MISC, ""), new ItemStackTemplate(Items.STICK, 4),
			List.of(Ingredient.of(Items.OAK_PLANKS), Ingredient.of(Items.OAK_PLANKS)));
		return new RecipeHolder<>(ResourceKey.create(Registries.RECIPE, Identifier.parse(id)), recipe);
	}
}
