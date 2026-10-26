package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins atomic shared-contract updates and omission of disabled goods settings from old hashes. */
final class GoodsConfigTest {
	/** Initializes item registry validation once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Disabled effective defaults and empty reset values produce the exact old configuration tree. */
	@Test
	void disabledDefaultsDoNotChangeLegacyJson() {
		SimConfig config = new SimConfig();
		JsonObject before = config.json();
		assertNull(config.freshDays());
		assertTrue(config.carryLimits().isEmpty());
		assertTrue(config.villagerPurchases().isEmpty());
		assertFalse(config.goodsEnabled());
		config.update(json("{\"fresh_days\":null,\"carry_limits\":{},\"villager_purchases\":[],\"night_sight\":false,\"visible_loads\":{}}"));
		assertEquals(before, config.json());
	}

	/** One update accepts every worker's shared keys and saved configuration restores those exact values. */
	@Test
	void allSharedKeysCanBeSetAndRestoredTogether() {
		SimConfig config = new SimConfig();
		config.update(json("""
			{"carry_limits":{"minecraft:bread":32,"minecraft:wheat":96},"fresh_days":2,"bake_ticks":40,
			"villager_purchases":[{"stall":[1,2,-3],"item":"minecraft:bread","count":2}],
			"night_sight":true,"visible_loads":{"minecraft:bread":9}}
			"""));
		assertTrue(config.goodsEnabled());
		assertEquals(2, config.freshDays());
		assertEquals(32, config.carryLimits().get("minecraft:bread").getAsInt());
		SimConfig restored = new SimConfig();
		restored.update(config.json());
		assertEquals(config.json(), restored.json());
	}

	/** Invalid item IDs, count shapes and fractional values never partially apply a mixed update. */
	@Test
	void invalidGoodsSettingsRejectAtomically() {
		for (String invalid : new String[]{
			"{\"carry_limits\":{\"minecraft:bread\":-1}}", "{\"carry_limits\":{\"minecraft:bread\":1.5}}",
			"{\"carry_limits\":{\"minecraft:bread\":\"32\"}}", "{\"carry_limits\":{\"test:absent\":32}}",
			"{\"fresh_days\":-1}", "{\"fresh_days\":1.5}", "{\"fresh_days\":true}",
			"{\"villager_purchases\":[{\"stall\":[1,2,3],\"item\":\"minecraft:bread\",\"count\":0}]}",
			"{\"villager_purchases\":[{\"stall\":[1,2],\"item\":\"minecraft:bread\",\"count\":1}]}",
			"{\"villager_purchases\":[{\"stall\":[\"1\",2,3],\"item\":\"minecraft:bread\",\"count\":1}]}",
			"{\"villager_purchases\":[{\"stall\":[1,2.5,3],\"item\":\"minecraft:bread\",\"count\":1}]}"}) {
			SimConfig config = new SimConfig();
			JsonObject before = config.json();
			JsonObject update = json(invalid);
			update.addProperty("reach", 5);
			assertThrows(RuntimeException.class, () -> config.update(update), invalid);
			assertEquals(before, config.json());
		}
	}

	/** Parses a test configuration object without mutating shared fixtures. */
	private static JsonObject json(String value) {
		return JsonParser.parseString(value).getAsJsonObject();
	}
}
