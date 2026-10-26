package org.minecraftsim.harness;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.serialization.JsonOps;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import net.minecraft.core.RegistryAccess;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.RegistryOps;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.ItemStackTemplate;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.crafting.BannerDuplicateRecipe;
import net.minecraft.world.item.crafting.BookCloningRecipe;
import net.minecraft.world.item.crafting.CraftingInput;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.DecoratedPotRecipe;
import net.minecraft.world.item.crafting.FireworkRocketRecipe;
import net.minecraft.world.item.crafting.FireworkStarFadeRecipe;
import net.minecraft.world.item.crafting.FireworkStarRecipe;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.item.crafting.MapExtendingRecipe;
import net.minecraft.world.item.crafting.PlacementInfo;
import net.minecraft.world.item.crafting.Recipe;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.RepairItemRecipe;
import net.minecraft.world.item.crafting.ShapedRecipe;
import net.minecraft.world.item.crafting.ShapelessRecipe;
import net.minecraft.world.item.crafting.ShieldDecorationRecipe;
import net.minecraft.world.item.crafting.TransmuteRecipe;
import net.minecraft.world.item.crafting.display.RecipeDisplay;
import net.minecraft.world.item.crafting.display.ShapedCraftingRecipeDisplay;
import net.minecraft.world.item.crafting.display.SlotDisplay;
import net.minecraft.world.level.Level;

/**
 * Plans and commits component-aware crafting transactions on the server thread.
 * Recipe IDs and inventory slots have fixed search order; planning and execution
 * share transaction feasibility and stage copied inventories before committing.
 */
public final class SkillCrafting {
	private SkillCrafting() {
	}

	/**
	 * Simulates enough crafts for the requested count without modifying bot inventory.
	 * @param bot bot providing inventory and server recipes
	 * @param args output item identifier and optional positive count, defaulting to one
	 * @param craftTicks nonnegative duration per craft
	 * @param tableAvailable whether the caller has admitted a crafting table
	 * @return plan with timing/materials/recipe data, or a reason-bearing rejection object
	 * @throws IllegalArgumentException if item/count arguments or craftTicks are invalid
	 * @throws ArithmeticException if exact material accounting overflows
	 */
	public static JsonObject plan(SimBot bot, JsonObject args, int craftTicks, boolean tableAvailable) {
		if (args == null || !args.has("item") || !args.get("item").isJsonPrimitive()
				|| !args.getAsJsonPrimitive("item").isString()) {
			throw new IllegalArgumentException("item must be an item identifier string");
		}
		Item item = item(args.get("item").getAsString());
		int count = args.has("count") ? positiveInteger(args.get("count")) : 1;
		if (craftTicks < 0) {
			throw new IllegalArgumentException("craftTicks must be nonnegative");
		}
		Inventory inventory = bot.getInventory();
		return planRecipes(bot.level().recipeAccess().getRecipes(), item, count, craftTicks, tableAvailable,
			copyInventory(inventory), inventory.getMaxStackSize(), bot.level(), HarnessState.get(bot.level().getServer()).config(),
			Math.floorDiv(bot.level().getGameTime(), 24_000L));
	}

	/**
	 * Rechecks and commits one recipe transaction, including output and remainder capacity.
	 * @param bot bot whose inventory will change only after transaction success
	 * @param recipeId planned recipe's registry identifier
	 * @param item expected output item identifier
	 * @return produced count on success, otherwise the established reason-bearing failure object
	 * @throws IllegalArgumentException if either identifier is invalid or the output item is unknown
	 */
	public static JsonObject craft(SimBot bot, String recipeId, String item) {
		Item output = item(item);
		Identifier identifier = Identifier.tryParse(recipeId == null ? "" : recipeId);
		if (identifier == null) {
			throw new IllegalArgumentException("recipeId must be a recipe identifier string");
		}
		RecipeHolder<?> holder = bot.level().recipeAccess().byKey(ResourceKey.create(Registries.RECIPE, identifier)).orElse(null);
		if (holder == null || !(holder.value() instanceof CraftingRecipe recipe)) {
			return failure("no_recipe");
		}
		Inventory inventory = bot.getInventory();
		List<ItemStack> original = copyInventory(inventory);
		if (outputMatch(recipe, output, original, bot.level()) == 0) {
			return failure("no_recipe");
		}
		SimConfig config = HarnessState.get(bot.level().getServer()).config();
		Transaction transaction = transaction(recipe, output, original, inventory.getMaxStackSize(), bot.level(), config,
			Math.floorDiv(bot.level().getGameTime(), 24_000L));
		if (transaction.reason() != null) {
			if (transaction.reason().equals("too_heavy")) {
				return CarryLimits.check(original, transaction.inventory(), config);
			}
			return failure(transaction.reason());
		}
		JsonObject heavy = CarryLimits.check(original, transaction.inventory(), config);
		if (heavy != null) {
			return heavy;
		}
		for (int slot = 0; slot < transaction.inventory().size(); slot++) {
			ItemStack replacement = transaction.inventory().get(slot);
			if (!ItemStack.matches(inventory.getItem(slot), replacement)) {
				inventory.setItem(slot, replacement);
			}
		}
		inventory.setChanged();
		JsonObject result = new JsonObject();
		result.addProperty("produced", transaction.produced());
		return result;
	}

	static JsonObject planRecipes(Collection<RecipeHolder<?>> recipes, Item item, int count,
			int craftTicks, boolean tableAvailable, List<ItemStack> inventory, int maxStackSize, Level level) {
		return planRecipes(recipes, item, count, craftTicks, tableAvailable, inventory, maxStackSize, level, null, 0);
	}

	/** Simulates every intermediate net gain under the optional goods rules before any real craft starts. */
	static JsonObject planRecipes(Collection<RecipeHolder<?>> recipes, Item item, int count,
			int craftTicks, boolean tableAvailable, List<ItemStack> inventory, int maxStackSize, Level level, SimConfig config, long day) {
		JsonObject rejected = null;
		for (RecipeHolder<?> holder : recipes.stream().sorted(Comparator.comparing(recipe -> recipe.id().identifier().toString())).toList()) {
			if (!(holder.value() instanceof CraftingRecipe recipe)) {
				continue;
			}
			int outputMatch = outputMatch(recipe, item, inventory, level);
			if (outputMatch == 0) {
				continue;
			}
			JsonObject candidate = simulate(recipe, item, count, craftTicks, tableAvailable, inventory, maxStackSize, level, config, day);
			candidate.addProperty("recipe", holder.id().identifier().toString());
			if (!candidate.has("reason") || candidate.get("reason").getAsString().equals("needs_crafting_table")) {
				return candidate;
			}
			if (outputMatch < 0) {
				continue;
			}
			if (rejected == null || priority(candidate.get("reason").getAsString()) > priority(rejected.get("reason").getAsString())) {
				rejected = candidate;
			}
		}
		return rejected != null ? rejected : planFailure("no_recipe");
	}

	private static JsonObject simulate(CraftingRecipe recipe, Item item, int count, int craftTicks, boolean tableAvailable,
			List<ItemStack> original, int maxStackSize, Level level, SimConfig config, long day) {
		List<ItemStack> inventory = copyStacks(original);
		Map<String, Long> materials = new TreeMap<>();
		long produced = 0;
		int crafts = 0;
		int outputPerCraft = 0;
		boolean needsTable = false;
		while (produced < count) {
			Transaction transaction = transaction(recipe, item, inventory, maxStackSize, level, config, day);
			if (transaction.reason() != null) {
				JsonObject failure = planFailure(transaction.reason());
				if (transaction.reason().equals("too_heavy")) {
					JsonObject heavy = CarryLimits.check(original, transaction.inventory(), config);
					if (heavy != null) {
						heavy.entrySet().forEach(entry -> failure.add(entry.getKey(), entry.getValue()));
					}
				}
				return failure;
			}
			JsonObject heavy = config == null ? null : CarryLimits.check(original, transaction.inventory(), config);
			if (heavy != null) {
				JsonObject failure = planFailure("too_heavy");
				heavy.entrySet().forEach(entry -> failure.add(entry.getKey(), entry.getValue()));
				return failure;
			}
			if (outputPerCraft == 0) {
				outputPerCraft = transaction.produced();
			}
			int repetitions = sameInventory(inventory, transaction.inventory())
				? (int) ((count - produced + transaction.produced() - 1L) / transaction.produced()) : 1;
			produced += (long) transaction.produced() * repetitions;
			crafts += repetitions;
			needsTable |= transaction.table();
			transaction.materials().forEach((ingredient, amount) ->
				materials.merge(ingredient, Math.multiplyExact(amount, repetitions), Math::addExact));
			inventory = transaction.inventory();
		}
		JsonObject result = new JsonObject();
		result.addProperty("ticks", (long) crafts * craftTicks);
		result.addProperty("crafts", crafts);
		result.addProperty("outputPerCraft", outputPerCraft);
		result.addProperty("needsTable", needsTable);
		JsonObject bill = new JsonObject();
		materials.forEach(bill::addProperty);
		result.add("materials", bill);
		if (needsTable && !tableAvailable) {
			result.addProperty("reason", "needs_crafting_table");
		}
		return result;
	}

	private static Transaction transaction(CraftingRecipe recipe, Item item, List<ItemStack> original, int maxStackSize, Level level,
			SimConfig config, long day) {
		List<Layout> layouts = layouts(recipe, item, level);
		if (layouts.isEmpty()) {
			return rejected("no_recipe");
		}
		for (Layout layout : layouts) {
			int[] remaining = original.stream().mapToInt(ItemStack::getCount).toArray();
			int[] selected = new int[layout.ingredients().size()];
			boolean[][] accepted = new boolean[selected.length][original.size()];
			for (int ingredient = 0; ingredient < selected.length; ingredient++) {
				for (int slot = 0; slot < original.size(); slot++) {
					accepted[ingredient][slot] = !original.get(slot).isEmpty()
						&& layout.ingredients().get(ingredient).test(original.get(slot).copyWithCount(1));
				}
			}
			CraftingInput input = allocate(recipe, item, layout, original, remaining, selected, accepted, 0, level);
			if (input == null) {
				continue;
			}
			ItemStack output = recipe.assemble(input).copy();
			if (config != null && config.freshDays() != null) {
				Freshness.stamp(output, day);
			}
			List<ItemStack> remainders = copyStacks(recipe.getRemainingItems(input));
			if (output.isEmpty() || !output.is(item) || remainders.size() != input.size()) {
				return rejected("no_recipe");
			}
			List<ItemStack> inventory = copyStacks(original);
			Map<String, Long> materials = new TreeMap<>();
			for (int slot : selected) {
				ItemStack ingredient = inventory.get(slot);
				materials.merge(BuiltInRegistries.ITEM.getKey(ingredient.getItem()).toString(), 1L, Math::addExact);
				ingredient.shrink(1);
				if (ingredient.isEmpty()) {
					inventory.set(slot, ItemStack.EMPTY);
				}
			}
			if (config != null) {
				List<ItemStack> projected = new ArrayList<>(inventory);
				projected.add(output);
				projected.addAll(remainders);
				if (CarryLimits.check(original, projected, config) != null) {
					return new Transaction("too_heavy", projected, materials, output.getCount(), layout.width() > 2 || layout.height() > 2);
				}
			}
			if (!insert(inventory, output, maxStackSize)) {
				return rejected("inventory_full");
			}
			for (ItemStack remainder : remainders) {
				if (!insert(inventory, remainder, maxStackSize)) {
					return rejected("inventory_full");
				}
			}
			return new Transaction(null, inventory, materials, output.getCount(), layout.width() > 2 || layout.height() > 2);
		}
		return rejected(outputMatch(recipe, item, original, level) == 0 ? "no_recipe" : "missing_items");
	}

	private static CraftingInput allocate(CraftingRecipe recipe, Item item, Layout layout, List<ItemStack> inventory,
			int[] remaining, int[] selected, boolean[][] accepted, int ingredient, Level level) {
		if (ingredient == selected.length) {
			List<ItemStack> grid = new ArrayList<>();
			for (int index : layout.cells()) {
				grid.add(index == PlacementInfo.EMPTY_SLOT ? ItemStack.EMPTY : inventory.get(selected[index]).copyWithCount(1));
			}
			CraftingInput input = CraftingInput.of(layout.width(), layout.height(), grid);
			return recipe.matches(input, level) && recipe.assemble(input).is(item) ? input : null;
		}
		int[] owners = new int[selected.length];
		Arrays.fill(owners, -1);
		for (int pending = ingredient; pending < selected.length; pending++) {
			if (!assign(pending, accepted, remaining, owners, new boolean[inventory.size()])) {
				return null;
			}
		}
		List<ItemStack> tried = new ArrayList<>();
		for (int slot = 0; slot < inventory.size(); slot++) {
			ItemStack stack = inventory.get(slot);
			if (remaining[slot] <= 0 || !accepted[ingredient][slot]
					|| tried.stream().anyMatch(previous -> ItemStack.isSameItemSameComponents(previous, stack))) {
				continue;
			}
			tried.add(stack);
			remaining[slot]--;
			selected[ingredient] = slot;
			CraftingInput input = allocate(recipe, item, layout, inventory, remaining, selected, accepted, ingredient + 1, level);
			remaining[slot]++;
			if (input != null) {
				return input;
			}
		}
		return null;
	}

	private static boolean assign(int ingredient, boolean[][] accepted, int[] capacity, int[] owners, boolean[] visited) {
		for (int slot = 0; slot < capacity.length; slot++) {
			if (capacity[slot] <= 0 || visited[slot] || !accepted[ingredient][slot]) {
				continue;
			}
			visited[slot] = true;
			int occupied = 0;
			for (int owner : owners) {
				if (owner == slot) {
					occupied++;
				}
			}
			if (occupied < capacity[slot]) {
				owners[ingredient] = slot;
				return true;
			}
			for (int previous = 0; previous < owners.length; previous++) {
				if (owners[previous] == slot && assign(previous, accepted, capacity, owners, visited)) {
					owners[ingredient] = slot;
					return true;
				}
			}
		}
		return false;
	}

	private static boolean insert(List<ItemStack> inventory, ItemStack stack, int maxStackSize) {
		int remaining = stack.getCount();
		for (int slot = 0; slot < Inventory.INVENTORY_SIZE && remaining > 0; slot++) {
			ItemStack existing = inventory.get(slot);
			int limit = Math.min(maxStackSize, stack.getMaxStackSize());
			if (existing.isEmpty()) {
				int amount = Math.min(remaining, limit);
				inventory.set(slot, stack.copyWithCount(amount));
				remaining -= amount;
			} else if (existing.isStackable() && ItemStack.isSameItemSameComponents(existing, stack)) {
				int amount = Math.min(remaining, Math.max(0, limit - existing.getCount()));
				existing.grow(amount);
				remaining -= amount;
			}
		}
		return remaining == 0;
	}

	private static List<Layout> layouts(CraftingRecipe recipe, Item item, Level level) {
		PlacementInfo placement = recipe.placementInfo();
		if (placement.isImpossibleToPlace() || placement.ingredients().isEmpty()) {
			return specialLayouts(recipe, item, level);
		}
		int[] cells = placement.slotsToIngredientIndex().toIntArray();
		boolean[] present = new boolean[placement.ingredients().size()];
		for (int index : cells) {
			if (index == PlacementInfo.EMPTY_SLOT) {
				continue;
			}
			if (index < 0 || index >= present.length || present[index]) {
				return List.of();
			}
			present[index] = true;
		}
		for (boolean ingredient : present) {
			if (!ingredient) {
				return List.of();
			}
		}
		if (recipe instanceof ShapedRecipe shaped) {
			int width = shaped.getWidth();
			int height = shaped.getHeight();
			return width >= 1 && width <= 3 && height >= 1 && height <= 3 && cells.length == width * height
				? List.of(new Layout(width, height, placement.ingredients(), cells)) : List.of();
		}
		for (RecipeDisplay display : recipe.display()) {
			if (display instanceof ShapedCraftingRecipeDisplay shaped) {
				return shaped.width() >= 1 && shaped.width() <= 3 && shaped.height() >= 1 && shaped.height() <= 3
						&& cells.length == shaped.width() * shaped.height()
					? List.of(new Layout(shaped.width(), shaped.height(), placement.ingredients(), cells)) : List.of();
			}
		}
		TreeSet<Integer> counts = new TreeSet<>();
		if (recipe instanceof TransmuteRecipe) {
			for (int count = 2; count <= placement.ingredients().size(); count++) {
				counts.add(count);
			}
		} else {
			counts.add(placement.ingredients().size());
		}
		List<Layout> layouts = new ArrayList<>();
		for (int count : counts) {
			if (count < 1 || count > 9 || count > cells.length || count > placement.ingredients().size()) {
				return List.of();
			}
			for (int index = 0; index < count; index++) {
				if (cells[index] != index) {
					return List.of();
				}
			}
			int width = count <= 4 ? Math.min(count, 2) : 3;
			int height = (count + width - 1) / width;
			int[] grid = new int[width * height];
			Arrays.fill(grid, PlacementInfo.EMPTY_SLOT);
			System.arraycopy(cells, 0, grid, 0, count);
			layouts.add(new Layout(width, height, placement.ingredients().subList(0, count), grid));
		}
		return layouts;
	}

	private static List<Layout> specialLayouts(CraftingRecipe recipe, Item item, Level level) {
		if (recipe instanceof RepairItemRecipe) {
			return List.of(new Layout(2, 1, List.of(Ingredient.of(item), Ingredient.of(item)), new int[]{0, 1}));
		}
		JsonObject data = recipeData(recipe, level);
		if (data == null) {
			return List.of();
		}
		if (recipe instanceof DecoratedPotRecipe) {
			List<Ingredient> ingredients = ingredients(data, level, "back", "left", "right", "front");
			return ingredients.size() == 4
				? List.of(new Layout(3, 3, ingredients, new int[]{-1, 0, -1, 1, -1, 2, -1, 3, -1})) : List.of();
		}
		if (recipe instanceof MapExtendingRecipe) {
			List<Ingredient> ingredients = ingredients(data, level,
				"material", "material", "material", "material", "map", "material", "material", "material", "material");
			return ingredients.size() == 9
				? List.of(new Layout(3, 3, ingredients, new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8})) : List.of();
		}
		String[] fields;
		if (recipe instanceof BannerDuplicateRecipe) {
			fields = new String[]{"banner", "banner"};
		} else if (recipe instanceof BookCloningRecipe) {
			fields = new String[]{"source", "material"};
		} else if (recipe instanceof ShieldDecorationRecipe) {
			fields = new String[]{"banner", "target"};
		} else if (recipe instanceof FireworkRocketRecipe) {
			fields = new String[]{"shell", "fuel"};
		} else if (recipe instanceof FireworkStarRecipe) {
			fields = new String[]{"fuel", "dye"};
		} else if (recipe instanceof FireworkStarFadeRecipe) {
			fields = new String[]{"target", "dye"};
		} else {
			return List.of();
		}
		List<Ingredient> ingredients = ingredients(data, level, fields);
		return ingredients.size() == 2 ? List.of(new Layout(2, 1, ingredients, new int[]{0, 1})) : List.of();
	}

	private static List<Ingredient> ingredients(JsonObject data, Level level, String... fields) {
		List<Ingredient> ingredients = new ArrayList<>();
		for (String field : fields) {
			if (!data.has(field)) {
				return List.of();
			}
			Ingredient ingredient = Ingredient.CODEC.parse(serialization(level), data.get(field)).result().orElse(null);
			if (ingredient == null) {
				return List.of();
			}
			ingredients.add(ingredient);
		}
		return ingredients;
	}

	private static RegistryOps<JsonElement> serialization(Level level) {
		return RegistryOps.create(JsonOps.INSTANCE,
			level == null ? RegistryAccess.fromRegistryOfRegistries(BuiltInRegistries.REGISTRY) : level.registryAccess());
	}

	private static JsonObject recipeData(CraftingRecipe recipe, Level level) {
		JsonElement data = Recipe.DIRECT_CODEC.encodeStart(serialization(level), recipe).result().orElse(null);
		return data != null && data.isJsonObject() ? data.getAsJsonObject() : null;
	}

	private static int outputMatch(CraftingRecipe recipe, Item item, List<ItemStack> inventory, Level level) {
		if (recipe instanceof RepairItemRecipe) {
			if (repairable(item.getDefaultInstance())) {
				return 1;
			}
			for (ItemStack stack : inventory) {
				if (stack.is(item) && repairable(stack)) {
					return 1;
				}
			}
			return 0;
		}
		if (!(recipe instanceof ShapedRecipe) && !(recipe instanceof ShapelessRecipe) && !(recipe instanceof TransmuteRecipe)) {
			JsonObject data = recipeData(recipe, level);
			if (data != null && data.has("result")) {
				ItemStackTemplate result = ItemStackTemplate.CODEC.parse(serialization(level), data.get("result")).result().orElse(null);
				if (result != null) {
					return result.item().value() == item ? 1 : 0;
				}
			}
		}
		List<RecipeDisplay> displays = recipe.display();
		int result = displays.isEmpty() ? -1 : 0;
		for (RecipeDisplay display : displays) {
			int match = outputMatch(display.result(), item);
			if (match == 1) {
				return 1;
			}
			if (match < 0) {
				result = -1;
			}
		}
		return result;
	}

	private static boolean repairable(ItemStack stack) {
		return stack.has(DataComponents.MAX_DAMAGE) && stack.has(DataComponents.DAMAGE);
	}

	private static int outputMatch(SlotDisplay display, Item item) {
		if (display instanceof SlotDisplay.ItemStackSlotDisplay stack) {
			return stack.stack().item().value() == item ? 1 : 0;
		}
		if (display instanceof SlotDisplay.ItemSlotDisplay stack) {
			return stack.item().value() == item ? 1 : 0;
		}
		if (display instanceof SlotDisplay.Composite composite) {
			int result = 0;
			for (SlotDisplay child : composite.contents()) {
				int match = outputMatch(child, item);
				if (match == 1) {
					return 1;
				}
				if (match < 0) {
					result = -1;
				}
			}
			return result;
		}
		if (display instanceof SlotDisplay.WithAnyPotion potion) {
			return outputMatch(potion.display(), item);
		}
		if (display instanceof SlotDisplay.WithRemainder remainder) {
			return outputMatch(remainder.input(), item);
		}
		if (display instanceof SlotDisplay.OnlyWithComponent component) {
			return outputMatch(component.source(), item);
		}
		return -1;
	}

	private static Item item(String value) {
		Identifier identifier = Identifier.tryParse(value == null ? "" : value);
		if (identifier == null || !BuiltInRegistries.ITEM.containsKey(identifier)) {
			throw new IllegalArgumentException("Unknown item identifier: " + value);
		}
		Item item = BuiltInRegistries.ITEM.getValue(identifier);
		if (item == Items.AIR) {
			throw new IllegalArgumentException("item must not be air");
		}
		return item;
	}

	private static int positiveInteger(JsonElement value) {
		if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("count must be a finite positive integer");
		}
		try {
			int count = value.getAsBigDecimal().intValueExact();
			if (count > 0) {
				return count;
			}
		} catch (ArithmeticException | NumberFormatException invalid) {
			throw new IllegalArgumentException("count must be a finite positive integer", invalid);
		}
		throw new IllegalArgumentException("count must be a finite positive integer");
	}

	private static List<ItemStack> copyInventory(Inventory inventory) {
		List<ItemStack> result = new ArrayList<>(inventory.getContainerSize());
		for (int slot = 0; slot < inventory.getContainerSize(); slot++) {
			result.add(inventory.getItem(slot).copy());
		}
		return result;
	}

	private static List<ItemStack> copyStacks(List<ItemStack> stacks) {
		List<ItemStack> result = new ArrayList<>(stacks.size());
		stacks.forEach(stack -> result.add(stack.copy()));
		return result;
	}

	private static boolean sameInventory(List<ItemStack> first, List<ItemStack> second) {
		for (int slot = 0; slot < first.size(); slot++) {
			if (!ItemStack.matches(first.get(slot), second.get(slot))) {
				return false;
			}
		}
		return true;
	}

	private static int priority(String reason) {
		return switch (reason) {
			case "too_heavy" -> 4;
			case "needs_crafting_table" -> 3;
			case "inventory_full" -> 2;
			case "missing_items" -> 1;
			default -> 0;
		};
	}

	private static JsonObject failure(String reason) {
		JsonObject result = new JsonObject();
		result.addProperty("reason", reason);
		return result;
	}

	private static JsonObject planFailure(String reason) {
		JsonObject result = failure(reason);
		result.addProperty("ticks", 0L);
		result.add("materials", new JsonObject());
		result.addProperty("recipe", "");
		result.addProperty("crafts", 0);
		result.addProperty("outputPerCraft", 0);
		result.addProperty("needsTable", false);
		return result;
	}

	private static Transaction rejected(String reason) {
		return new Transaction(reason, List.of(), Map.of(), 0, false);
	}

	private record Layout(int width, int height, List<Ingredient> ingredients, int[] cells) {
	}

	private record Transaction(String reason, List<ItemStack> inventory, Map<String, Long> materials, int produced, boolean table) {
	}
}
