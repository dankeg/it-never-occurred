package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.BedBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.ButtonBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.LeverBlock;
import net.minecraft.world.level.block.SignBlock;
import net.minecraft.world.level.block.TrapDoorBlock;

/**
 * Owns validated simulation settings and their stable JSON representation.
 * Updates replace a validated copy atomically; notable block names remain sorted.
 */
public final class SimConfig {
	private static final Map<String, Integer> DAILY_DEFAULTS = Map.of(
		"eat_ticks", 32, "hunger_interval_ticks", 6_000, "hunger_points", 2, "bake_ticks", 40);
	private JsonObject values;

	/** Initializes fixed defaults and the sorted registry-derived notable-block set. */
	public SimConfig() {
		values = new JsonObject();
		values.addProperty("view_radius", 24.0);
		values.addProperty("terrain_radius", 12);
		values.addProperty("terrain_step", 2);
		values.addProperty("reach", 4.5);
		values.addProperty("speech_radius", 8.0);
		values.addProperty("speech_rate", 0.15);
		values.addProperty("talk_radius", 8.0);
		values.addProperty("shout_radius", 24.0);
		values.addProperty("shout_cooldown_ticks", 600);
		values.addProperty("attention_slots", 2);
		values.addProperty("relay_rate", 0.5);
		values.addProperty("news_ttl_days", 3);
		values.addProperty("exposure_log", 1);
		values.addProperty("craft_ticks", 10);
		values.addProperty("place_ticks", 4);
		values.addProperty("container_ticks", 5);
		values.addProperty("give_ticks", 5);
		values.addProperty("use_ticks", 5);
		values.addProperty("drop_ticks", 10);
		Set<String> names = new TreeSet<>(Set.of("minecraft:chest", "minecraft:trapped_chest", "minecraft:barrel",
			"minecraft:lectern", "minecraft:crafting_table", "minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker",
			"minecraft:barrier", "minecraft:bell", "minecraft:composter", "minecraft:anvil", "minecraft:smithing_table",
			"minecraft:ladder", "minecraft:scaffolding"));
		BuiltInRegistries.BLOCK.forEach(block -> {
			if (block instanceof SignBlock || block instanceof DoorBlock || block instanceof BedBlock || block instanceof TrapDoorBlock
					|| block instanceof FenceGateBlock || block instanceof LeverBlock || block instanceof ButtonBlock) {
				names.add(BuiltInRegistries.BLOCK.getKey(block).toString());
			}
		});
		JsonArray notable = new JsonArray();
		names.forEach(notable::add);
		values.add("notable_blocks", notable);
	}

	/**
	 * Takes a copy that callers may edit without changing configuration.
	 * @return deep copy of the current settings
	 */
	public JsonObject json() {
		return values.deepCopy();
	}

	/**
	 * Reads a numeric setting using Gson's double conversion.
	 * @param key existing numeric setting name
	 * @return configured value
	 * @throws RuntimeException if the key is absent or not convertible
	 */
	public double number(String key) {
		return values.has(key) ? values.get(key).getAsDouble() : DAILY_DEFAULTS.get(key).doubleValue();
	}

	/**
	 * Reads a numeric setting using Gson's integer conversion.
	 * @param key existing numeric setting name
	 * @return configured integer value
	 * @throws RuntimeException if the key is absent or not convertible
	 */
	public int integer(String key) {
		return values.has(key) ? values.get(key).getAsInt() : DAILY_DEFAULTS.get(key);
	}

	/** Returns the opt-in hunger rule without adding default fields to legacy saved state. */
	public boolean dailyHunger() {
		return values.has("daily_hunger") && values.get("daily_hunger").getAsBoolean();
	}

	/** Returns the opt-in night rule without changing legacy saved configuration. */
	public boolean nightSight() {
		return values.has("night_sight") && values.get("night_sight").getAsBoolean();
	}

	/** Returns the opt-in truthful news rule without changing legacy saved configuration. */
	public boolean truthfulNews() {
		return values.has("truthful_news") && values.get("truthful_news").getAsBoolean();
	}

	/** Returns the opt-in question filtering rule without changing legacy saved configuration. */
	public boolean askAbout() {
		return values.has("ask_about") && values.get("ask_about").getAsBoolean();
	}

	/** Copies the sorted item thresholds; an absent or empty map disables visible loads. */
	public Map<String, Integer> visibleLoads() {
		Map<String, Integer> result = new TreeMap<>();
		if (values.has("visible_loads")) {
			values.getAsJsonObject("visible_loads").entrySet()
				.forEach(entry -> result.put(entry.getKey(), entry.getValue().getAsInt()));
		}
		return result;
	}

	/** Returns the opt-in bread lifetime without changing legacy configuration JSON. */
	public Integer freshDays() {
		return values.has("fresh_days") ? values.get("fresh_days").getAsInt() : null;
	}

	/** An explicit bake_ticks setting enables ovens; its effective duration otherwise defaults to forty. */
	public boolean bakingEnabled() {
		return values.has("bake_ticks");
	}

	/** Enables owned-shop transfers only in explicitly configured goods worlds. */
	public boolean goodsEnabled() {
		return values.has("carry_limits") || values.has("fresh_days") || values.has("bake_ticks") || values.has("villager_purchases");
	}

	/** Copies the per-item carrying limits, empty in legacy worlds. */
	public JsonObject carryLimits() {
		return values.has("carry_limits") ? values.getAsJsonObject("carry_limits").deepCopy() : new JsonObject();
	}

	/** Copies the ordered market purchase definitions, empty in legacy worlds. */
	public JsonArray villagerPurchases() {
		return values.has("villager_purchases") ? values.getAsJsonArray("villager_purchases").deepCopy() : new JsonArray();
	}

	/**
	 * Copies the notable-block identifiers without exposing mutable configuration.
	 * @return new sorted set of configured block identifiers
	 */
	public Set<String> notable() {
		Set<String> result = new TreeSet<>();
		values.getAsJsonArray("notable_blocks").forEach(value -> result.add(value.getAsString()));
		return result;
	}

	/**
	 * Validates all supplied settings before replacing the current configuration.
	 * @param updates subset of existing configuration keys and replacement values
	 * @throws IllegalArgumentException if a key, block identifier or numeric value is invalid
	 * @throws ArithmeticException if an integer setting is fractional or out of range
	 * @throws RuntimeException if a supplied JSON value has the wrong shape
	 */
	public void update(JsonObject updates) {
		JsonObject candidate = values.deepCopy();
		for (var entry : updates.entrySet()) {
			String key = entry.getKey();
			JsonElement value = entry.getValue();
			if (!candidate.has(key) && !DAILY_DEFAULTS.containsKey(key) && !Set.of("daily_hunger", "carry_limits",
					"fresh_days", "villager_purchases", "night_sight", "visible_loads", "truthful_news", "ask_about").contains(key)) {
				throw new IllegalArgumentException("Unknown configuration key: " + key);
			}
			if (key.equals("carry_limits")) {
				JsonObject limits = new JsonObject();
				for (var limit : new java.util.TreeMap<>(value.getAsJsonObject().asMap()).entrySet()) {
					Identifier item = Identifier.parse(limit.getKey());
					if (!BuiltInRegistries.ITEM.containsKey(item)) {
						throw new IllegalArgumentException("Unknown item: " + item);
					}
					limits.addProperty(item.toString(), nonnegativeInteger(limit.getValue(), key));
				}
				if (limits.isEmpty()) {
					candidate.remove(key);
				} else {
					candidate.add(key, limits);
				}
			} else if (key.equals("fresh_days")) {
				if (value.isJsonNull()) {
					candidate.remove(key);
				} else {
					candidate.addProperty(key, nonnegativeInteger(value, key));
				}
			} else if (key.equals("villager_purchases")) {
				JsonArray purchases = new JsonArray();
				for (JsonElement element : value.getAsJsonArray()) {
					JsonObject purchase = element.getAsJsonObject();
					JsonArray stall = purchase.getAsJsonArray("stall");
					if (stall.size() != 3) {
						throw new IllegalArgumentException("stall must have three integer coordinates");
					}
					for (JsonElement coordinate : stall) {
						if (!coordinate.isJsonPrimitive() || !coordinate.getAsJsonPrimitive().isNumber()) {
							throw new IllegalArgumentException("stall must have three integer coordinates");
						}
						coordinate.getAsBigDecimal().intValueExact();
					}
					Identifier item = Identifier.parse(purchase.get("item").getAsString());
					int count = nonnegativeInteger(purchase.get("count"), key);
					if (!BuiltInRegistries.ITEM.containsKey(item) || count == 0) {
						throw new IllegalArgumentException("Purchases require a known item and positive count");
					}
					JsonObject purchaseEntry = new JsonObject();
					purchaseEntry.add("stall", stall.deepCopy());
					purchaseEntry.addProperty("item", item.toString());
					purchaseEntry.addProperty("count", count);
					purchases.add(purchaseEntry);
				}
				if (purchases.isEmpty()) {
					candidate.remove(key);
				} else {
					candidate.add(key, purchases);
				}
			} else if (key.equals("daily_hunger") || key.equals("night_sight")
					|| key.equals("truthful_news") || key.equals("ask_about")) {
				if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) {
					throw new IllegalArgumentException(key + " must be a boolean");
				}
				if (value.getAsBoolean()) {
					candidate.addProperty(key, true);
				} else {
					candidate.remove(key);
				}
			} else if (key.equals("visible_loads")) {
				if (!value.isJsonObject()) {
					throw new IllegalArgumentException("visible_loads must be an object");
				}
				Map<String, Integer> thresholds = new TreeMap<>();
				for (var threshold : value.getAsJsonObject().entrySet()) {
					Identifier id = Identifier.parse(threshold.getKey());
					JsonElement amount = threshold.getValue();
					if (!BuiltInRegistries.ITEM.containsKey(id) || !amount.isJsonPrimitive()
							|| !amount.getAsJsonPrimitive().isNumber() || !Double.isFinite(amount.getAsDouble())
							|| amount.getAsDouble() <= 0) {
						throw new IllegalArgumentException("Visible loads need known items and positive integer thresholds");
					}
					if (thresholds.put(id.toString(), amount.getAsBigDecimal().intValueExact()) != null) {
						throw new IllegalArgumentException("Duplicate visible load item: " + id);
					}
				}
				if (thresholds.isEmpty()) {
					candidate.remove(key);
				} else {
					JsonObject normalized = new JsonObject();
					thresholds.forEach(normalized::addProperty);
					candidate.add(key, normalized);
				}
			} else if (key.equals("notable_blocks")) {
				Set<String> names = new TreeSet<>();
				for (JsonElement element : value.getAsJsonArray()) {
					Identifier id = Identifier.parse(element.getAsString());
					if (!BuiltInRegistries.BLOCK.containsKey(id)) {
						throw new IllegalArgumentException("Unknown block: " + id);
					}
					names.add(id.toString());
				}
				JsonArray notable = new JsonArray();
				names.forEach(notable::add);
				candidate.add(key, notable);
			} else {
				if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()
						|| !Double.isFinite(value.getAsDouble()) || value.getAsDouble() < 0) {
					throw new IllegalArgumentException(key + " must be finite and nonnegative");
				}
				if ((key.equals("relay_rate") || key.equals("exposure_log")) && value.getAsDouble() > 1) {
					throw new IllegalArgumentException(key + " must be at most one");
				}
				if (key.equals("terrain_radius") || key.equals("terrain_step") || key.endsWith("_ticks")
						|| key.equals("attention_slots") || key.equals("news_ttl_days") || key.equals("exposure_log")
						|| key.equals("hunger_points")) {
					int integer = value.getAsBigDecimal().intValueExact();
					if ((key.equals("terrain_step") || key.equals("hunger_interval_ticks")) && integer == 0) {
						throw new IllegalArgumentException(key + " must be positive");
					}
					if (!key.equals("bake_ticks") && DAILY_DEFAULTS.containsKey(key) && integer == DAILY_DEFAULTS.get(key)) {
						candidate.remove(key);
					} else {
						candidate.addProperty(key, integer);
					}
				} else {
					candidate.addProperty(key, value.getAsDouble());
				}
				if (key.equals("speech_radius") || key.equals("talk_radius")) {
					candidate.addProperty(key.equals("speech_radius") ? "talk_radius" : "speech_radius", value.getAsDouble());
				}
			}
		}
		values = candidate;
	}

	/** Validates whole nonnegative counts without Gson's permissive string and fractional conversions. */
	private static int nonnegativeInteger(JsonElement value, String key) {
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException(key + " must contain nonnegative integers");
		}
		int number = value.getAsBigDecimal().intValueExact();
		if (number < 0) {
			throw new IllegalArgumentException(key + " must contain nonnegative integers");
		}
		return number;
	}
}
