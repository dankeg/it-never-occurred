package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.block.BedBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.ButtonBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.LeverBlock;
import net.minecraft.world.level.block.SignBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins configuration defaults, opt-in flags, atomic overrides, radius aliases and protocol error messages. */
final class SimConfigTest {
	/** Initializes registries once for registry-derived notable block defaults. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Every numeric setting starts with its established protocol default. */
	@Test
	void numericDefaultsMatchTheProtocol() {
		JsonObject actual = new SimConfig().json();
		actual.remove("notable_blocks");
		assertEquals(JsonParser.parseString("""
			{"view_radius":24.0,"terrain_radius":12,"terrain_step":2,"reach":4.5,
			 "speech_radius":8.0,"speech_rate":0.15,"talk_radius":8.0,"shout_radius":24.0,
			 "shout_cooldown_ticks":600,"attention_slots":2,"relay_rate":0.5,"news_ttl_days":3,
			 "exposure_log":1,"craft_ticks":10,"place_ticks":4,"container_ticks":5,"give_ticks":5,"use_ticks":5,"drop_ticks":10}
			"""), actual);
	}

	/** Witness rules default to false without adding fields or changing serialized legacy configuration. */
	@Test
	void witnessFlagsDefaultToFalseWithoutAddingLegacyJsonFields() {
		SimConfig config = new SimConfig();
		String legacyJson = config.json().toString();
		assertFalse(config.truthfulNews());
		assertFalse(config.askAbout());
		assertFalse(config.json().has("truthful_news"));
		assertFalse(config.json().has("ask_about"));
		config.update(new JsonObject());
		assertEquals(legacyJson, config.json().toString());
		config.update(JsonParser.parseString("{\"truthful_news\":false,\"ask_about\":false}").getAsJsonObject());
		assertFalse(config.truthfulNews());
		assertFalse(config.askAbout());
		assertEquals(legacyJson, config.json().toString());
	}

	/** Each witness rule toggles independently, and disabling both restores byte-identical legacy JSON. */
	@Test
	void witnessFlagsToggleIndependentlyAndRestoreLegacyJson() {
		SimConfig config = new SimConfig();
		String legacyJson = config.json().toString();
		config.update(JsonParser.parseString("{\"truthful_news\":true}").getAsJsonObject());
		assertTrue(config.truthfulNews());
		assertFalse(config.askAbout());
		assertEquals(JsonParser.parseString("true"), config.json().get("truthful_news"));
		assertFalse(config.json().has("ask_about"));
		config.update(JsonParser.parseString("{\"ask_about\":true}").getAsJsonObject());
		assertTrue(config.truthfulNews());
		assertTrue(config.askAbout());
		assertEquals(JsonParser.parseString("true"), config.json().get("ask_about"));
		config.update(JsonParser.parseString("{\"truthful_news\":false}").getAsJsonObject());
		assertFalse(config.truthfulNews());
		assertTrue(config.askAbout());
		assertFalse(config.json().has("truthful_news"));
		config.update(JsonParser.parseString("{\"ask_about\":false}").getAsJsonObject());
		assertFalse(config.truthfulNews());
		assertFalse(config.askAbout());
		assertEquals(legacyJson, config.json().toString());
	}

	/** Nonboolean witness values reject earlier numeric and boolean changes in either initial flag state. */
	@Test
	void witnessFlagsRejectNonBooleanValuesAtomically() {
		for (boolean enabled : List.of(false, true)) {
			SimConfig config = new SimConfig();
			if (enabled) {
				config.update(JsonParser.parseString("{\"truthful_news\":true,\"ask_about\":true}").getAsJsonObject());
			}
			String before = config.json().toString();
			for (String key : List.of("truthful_news", "ask_about")) {
				String otherKey = key.equals("truthful_news") ? "ask_about" : "truthful_news";
				for (String invalid : List.of("null", "0", "1", "\"true\"", "\"false\"", "[]", "[true]", "{}")) {
					JsonObject updates = new JsonObject();
					updates.addProperty("view_radius", 100);
					updates.addProperty(otherKey, !enabled);
					updates.add(key, JsonParser.parseString(invalid));
					IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
						() -> config.update(updates), key + " = " + invalid);
					assertEquals(key + " must be a boolean", failure.getMessage());
					assertEquals(enabled, config.truthfulNews());
					assertEquals(enabled, config.askAbout());
					assertEquals(before, config.json().toString());
				}
			}
		}
	}

	/** Valid witness changes remain unapplied when a later setting fails validation. */
	@Test
	void witnessFlagUpdatesAreAtomicWhenALaterSettingIsInvalid() {
		SimConfig config = new SimConfig();
		String before = config.json().toString();
		JsonObject updates = JsonParser.parseString("{\"truthful_news\":true,\"ask_about\":true,\"reach\":-1}").getAsJsonObject();
		assertThrows(IllegalArgumentException.class, () -> config.update(updates));
		assertFalse(config.truthfulNews());
		assertFalse(config.askAbout());
		assertEquals(before, config.json().toString());
	}

	/** Notable defaults include fixed interactables and registry-derived blocks in unique sorted order. */
	@Test
	void notableDefaultsIncludeFixedAndRegistryDerivedBlocksInSortedOrder() {
		SimConfig config = new SimConfig();
		List<String> actual = new ArrayList<>();
		config.json().getAsJsonArray("notable_blocks").forEach(value -> actual.add(value.getAsString()));
		Set<String> expected = new TreeSet<>(Set.of("minecraft:chest", "minecraft:trapped_chest", "minecraft:barrel",
			"minecraft:lectern", "minecraft:crafting_table", "minecraft:furnace", "minecraft:blast_furnace", "minecraft:smoker",
			"minecraft:barrier", "minecraft:bell", "minecraft:composter", "minecraft:anvil", "minecraft:smithing_table",
			"minecraft:ladder", "minecraft:scaffolding"));
		BuiltInRegistries.BLOCK.forEach(block -> {
			if (block instanceof SignBlock || block instanceof DoorBlock || block instanceof BedBlock || block instanceof TrapDoorBlock
					|| block instanceof FenceGateBlock || block instanceof LeverBlock || block instanceof ButtonBlock) {
				expected.add(BuiltInRegistries.BLOCK.getKey(block).toString());
			}
		});
		assertEquals(new ArrayList<>(expected), actual);
		assertEquals(expected, config.notable());
		assertTrue(config.notable().containsAll(Set.of("minecraft:oak_sign", "minecraft:oak_door", "minecraft:red_bed", "minecraft:oak_trapdoor")));
	}

	/** Updates replace supplied values while retaining untouched configuration settings. */
	@Test
	void updateOverridesNumericSettingsAndPreservesOtherDefaults() {
		SimConfig config = new SimConfig();
		JsonObject updates = JsonParser.parseString("""
			{"view_radius":32.5,"terrain_radius":6,"terrain_step":3,"reach":5.0,"speech_rate":0.25,
			 "shout_radius":30.0,"shout_cooldown_ticks":12,"attention_slots":1,"relay_rate":0.25,
			 "news_ttl_days":4,"exposure_log":0,"craft_ticks":2,"place_ticks":3,"container_ticks":4,"give_ticks":6}
			""").getAsJsonObject();
		config.update(updates);
		for (var entry : updates.entrySet()) {
			assertEquals(entry.getValue(), config.json().get(entry.getKey()), entry.getKey());
		}
		assertEquals(8.0, config.number("talk_radius"), 0.0);
		assertEquals(8.0, config.number("speech_radius"), 0.0);
	}

	/** Drop duration accepts nonnegative whole ticks and rejects invalid overrides atomically. */
	@Test
	void dropDurationUsesTheOrdinaryTickSettingValidation() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"drop_ticks\":7}").getAsJsonObject());
		assertEquals(7, config.integer("drop_ticks"));
		config.update(JsonParser.parseString("{\"drop_ticks\":0}").getAsJsonObject());
		assertEquals(0, config.integer("drop_ticks"));
		JsonObject before = config.json();
		assertThrows(IllegalArgumentException.class,
			() -> config.update(JsonParser.parseString("{\"drop_ticks\":-1}").getAsJsonObject()));
		assertThrows(ArithmeticException.class,
			() -> config.update(JsonParser.parseString("{\"drop_ticks\":1.5}").getAsJsonObject()));
		assertEquals(before, config.json());
	}

	/** Use duration defaults to five ticks and shares atomic nonnegative whole-tick validation. */
	@Test
	void useDurationUsesTheOrdinaryTickSettingValidation() {
		SimConfig config = new SimConfig();
		assertEquals(5, config.integer("use_ticks"));
		config.update(JsonParser.parseString("{\"use_ticks\":0}").getAsJsonObject());
		assertEquals(0, config.integer("use_ticks"));
		JsonObject before = config.json();
		assertThrows(IllegalArgumentException.class,
			() -> config.update(JsonParser.parseString("{\"use_ticks\":-1}").getAsJsonObject()));
		assertThrows(ArithmeticException.class,
			() -> config.update(JsonParser.parseString("{\"use_ticks\":1.5}").getAsJsonObject()));
		assertEquals(before, config.json());
	}

	/** Setting either radius alias updates both settings to exactly the supplied value. */
	@Test
	void speechAndTalkRadiusStayInSyncWhicheverIsSet() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"speech_radius\":3.25}").getAsJsonObject());
		assertEquals(3.25, config.number("speech_radius"), 0.0);
		assertEquals(3.25, config.number("talk_radius"), 0.0);
		config.update(JsonParser.parseString("{\"talk_radius\":5.5}").getAsJsonObject());
		assertEquals(5.5, config.number("speech_radius"), 0.0);
		assertEquals(5.5, config.number("talk_radius"), 0.0);
	}

	/** Notable-block overrides validate identifiers and retain only sorted unique entries. */
	@Test
	void updateSortsAndDeduplicatesNotableBlockOverrides() {
		SimConfig config = new SimConfig();
		config.update(JsonParser.parseString("{\"notable_blocks\":[\"minecraft:stone\",\"minecraft:oak_sign\",\"minecraft:stone\"]}")
			.getAsJsonObject());
		assertEquals(JsonParser.parseString("[\"minecraft:oak_sign\",\"minecraft:stone\"]"), config.json().get("notable_blocks"));
		assertEquals(Set.of("minecraft:oak_sign", "minecraft:stone"), config.notable());
	}

	/** Unknown keys return the exact protocol error and leave even earlier candidate overrides unapplied. */
	@Test
	void unknownKeysRejectTheWholeUpdateWithTheProtocolError() {
		SimConfig config = new SimConfig();
		JsonObject before = config.json();
		JsonObject updates = JsonParser.parseString("{\"view_radius\":100,\"unknown_setting\":1}").getAsJsonObject();
		IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> config.update(updates));
		assertEquals("Unknown configuration key: unknown_setting", failure.getMessage());
		assertEquals(before, config.json());
	}
}
