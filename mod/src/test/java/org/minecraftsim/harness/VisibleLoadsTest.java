package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Map;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins opt-in configuration, summed inventory thresholds and count-free public bands. */
final class VisibleLoadsTest {
	/** Initializes item components before constructing inventory stacks. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Split stacks count together, without revealing a slot or exact count. */
	@Test
	void sumsStacksAndReportsOnlyASack() {
		SimpleContainer inventory = new SimpleContainer(4);
		inventory.setItem(0, new ItemStack(Items.BREAD, 4));
		inventory.setItem(2, new ItemStack(Items.BREAD, 4));
		assertNull(VisibleLoads.seen(inventory, Map.of("minecraft:bread", 9)));
		inventory.setItem(3, new ItemStack(Items.BREAD, 1));
		assertEquals(JsonParser.parseString("{\"item\":\"minecraft:bread\",\"count_band\":\"sack\"}"),
			VisibleLoads.seen(inventory, Map.of("minecraft:bread", 9)));
		assertNull(VisibleLoads.seen(inventory, Map.of()));
	}

	/** Multiple qualifying items use a stable item-name order rather than map insertion order. */
	@Test
	void selectsTheFirstQualifyingItemDeterministically() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.WHEAT, 10), new ItemStack(Items.BREAD, 10));
		assertEquals("minecraft:bread", VisibleLoads.seen(inventory, Map.of("minecraft:wheat", 9, "minecraft:bread", 9))
			.get("item").getAsString());
	}

	/** Explicit false/empty overrides restore exactly the legacy serialized defaults. */
	@Test
	void defaultsAndDisabledOverridesLeaveLegacyConfigurationUntouched() {
		SimConfig config = new SimConfig();
		JsonObject legacy = config.json();
		assertFalse(config.nightSight());
		assertTrue(config.visibleLoads().isEmpty());
		config.update(JsonParser.parseString("{\"night_sight\":true,\"visible_loads\":{\"bread\":9}}").getAsJsonObject());
		assertTrue(config.nightSight());
		assertEquals(Map.of("minecraft:bread", 9), config.visibleLoads());
		config.visibleLoads().clear();
		assertEquals(1, config.visibleLoads().size());
		config.update(JsonParser.parseString("{\"night_sight\":false,\"visible_loads\":{}}").getAsJsonObject());
		assertEquals(legacy, config.json());
	}

	/** Invalid thresholds or booleans reject the entire configuration update. */
	@Test
	void validatesOverridesAtomically() {
		for (String invalid : new String[]{"{\"night_sight\":1}", "{\"night_sight\":\"true\"}",
				"{\"visible_loads\":[]}", "{\"visible_loads\":{\"bread\":0}}",
				"{\"visible_loads\":{\"bread\":-1}}", "{\"visible_loads\":{\"bread\":\"9\"}}",
				"{\"visible_loads\":{\"missing_item\":9}}", "{\"visible_loads\":{\"bread\":1.5}}",
				"{\"visible_loads\":{\"bread\":2147483648}}", "{\"visible_loads\":{\"bread\":9,\"minecraft:bread\":9}}"}) {
			SimConfig config = new SimConfig();
			JsonObject before = config.json();
			assertThrows(RuntimeException.class, () -> config.update(JsonParser.parseString(invalid).getAsJsonObject()), invalid);
			assertEquals(before, config.json());
		}
	}
}
