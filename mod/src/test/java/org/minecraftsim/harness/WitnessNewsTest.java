package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Locale;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins truthful, question-specific witness news and legacy bytes using admitted JSON rather than a server. */
final class WitnessNewsTest {
	/** Initializes real oven states once without starting a world or server. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Landmark changes use the replacement block, except removals, and discard namespaces and properties. */
	@Test
	void truthfulLandmarkChangesNormalizeBlocksAndClassifyActions() {
		String[][] cases = {
			{"minecraft:air", "minecraft:oak_planks[waterlogged=false]", "oak_planks", "placed"},
			{"minecraft:oak_stairs[facing=north]", "minecraft:air", "oak_stairs", "removed"},
			{"minecraft:oak_stairs[facing=north]", "minecraft:oak_stairs[facing=south]", "oak_stairs", "changed"},
			{"minecraft:stone", "example:polished_andesite[axis=y]", "polished_andesite", "changed"}
		};
		for (String[] example : cases) {
			JsonObject change = change(example[0], example[1]);
			JsonObject item = file(change, false, false, true);
			assertChange(item, example[2], example[3]);
			assertEquals("change:7", item.get("newsId").getAsString());
			assertEquals("change:7", item.get("originEventId").getAsString());
			assertEquals(100L, item.get("originatedAt").getAsLong());
			assertEquals(110L, item.get("learnedAt").getAsLong());
			assertEquals(array("[\"Witness\"]"), item.get("chain"));
			assertEquals(array("[\"baker\",\"freight\"]"), item.get("present"));
			assertTrue(item.get("firsthand").getAsBoolean());
		}
	}

	/** All three oven blocks distinguish placement, removal, other changes and both light transitions. */
	@Test
	void smokersFurnacesAndBlastFurnacesAreNotLightingDecorations() {
		for (String block : new String[]{"smoker", "furnace", "blast_furnace"}) {
			String cold = "minecraft:" + block + "[facing=north,lit=false]";
			String hot = "minecraft:" + block + "[facing=north,lit=true]";
			assertChange(file(change("minecraft:air", cold), false, false, true), block, "placed");
			assertChange(file(change(hot, "minecraft:air"), false, false, true), block, "removed");
			assertChange(file(change(cold, "minecraft:" + block + "[facing=south,lit=false]"),
				false, false, true), block, "changed");
			JsonObject lit = file(change(cold, hot), true, false, true);
			JsonObject unlit = file(change(hot, cold), false, false, true);
			assertChange(lit, block, "lit");
			assertChange(unlit, block, "unlit");
			assertEquals("The oven at the bakery was lit. baker and freight were there.",
				Communication.answer(news(lit), true));
			assertEquals("The oven at the bakery was unlit. baker and freight were there.",
				Communication.answer(news(unlit), true));
		}
	}

	/** Registry-only event ids retain admitted oven state transitions, but neither disabled nor nonoven metadata grows. */
	@Test
	void realOvenStatesRecordLitTransitionsWithoutChangingLegacyEvents() {
		Block[] ovens = {Blocks.SMOKER, Blocks.FURNACE, Blocks.BLAST_FURNACE};
		String[] names = {"smoker", "furnace", "blast_furnace"};
		for (int index = 0; index < ovens.length; index++) {
			String block = names[index];
			String registryId = "minecraft:" + block;
			BlockState cold = ovens[index].defaultBlockState().setValue(BlockStateProperties.LIT, false);
			BlockState hot = cold.setValue(BlockStateProperties.LIT, true);
			JsonObject lit = change(registryId, registryId);
			JsonObject expectedLit = lit.deepCopy();
			expectedLit.addProperty("fromLit", false);
			expectedLit.addProperty("toLit", true);
			VillagerNews.recordOven(lit, cold, hot, true);
			assertEquals(expectedLit, lit);
			assertChange(file(lit, true, false, true), block, "lit");
			JsonObject unlit = change(registryId, registryId);
			JsonObject expectedUnlit = unlit.deepCopy();
			expectedUnlit.addProperty("fromLit", true);
			expectedUnlit.addProperty("toLit", false);
			VillagerNews.recordOven(unlit, hot, cold, true);
			assertEquals(expectedUnlit, unlit);
			assertChange(file(unlit, false, false, true), block, "unlit");
			JsonObject disabled = change(registryId, registryId);
			String legacyBytes = disabled.toString();
			VillagerNews.recordOven(disabled, cold, hot, false);
			assertEquals(legacyBytes, disabled.toString());
			VillagerNews.recordOven(disabled, hot, cold, false);
			assertEquals(legacyBytes, disabled.toString());
			JsonObject placed = change("minecraft:air", registryId);
			VillagerNews.recordOven(placed, Blocks.AIR.defaultBlockState(), hot, true);
			assertFalse(placed.get("fromLit").getAsBoolean());
			assertTrue(placed.get("toLit").getAsBoolean());
			assertChange(file(placed, true, false, true), block, "lit");
		}
		BlockState darkLamp = Blocks.REDSTONE_LAMP.defaultBlockState().setValue(BlockStateProperties.LIT, false);
		BlockState litLamp = darkLamp.setValue(BlockStateProperties.LIT, true);
		JsonObject nonoven = change("minecraft:redstone_lamp", "minecraft:redstone_lamp");
		String nonovenBytes = nonoven.toString();
		VillagerNews.recordOven(nonoven, darkLamp, litLamp, true);
		assertEquals(nonovenBytes, nonoven.toString());
	}

	/** Ordinary lanterns retain the exact existing decoration item and answer in either truthful mode. */
	@Test
	void lanternDecorationIsUnchangedByTruthfulNews() {
		JsonObject change = change("minecraft:air", "minecraft:lantern[hanging=false,waterlogged=false]");
		JsonObject legacy = file(change, true, false, false);
		JsonObject truthful = file(change, true, false, true);
		assertEquals(legacy, truthful);
		assertEquals("decoration", truthful.get("kind").getAsString());
		assertEquals("bakery", truthful.get("place").getAsString());
		assertEquals(array("[1,2,3]"), truthful.get("pos"));
		assertEquals(0.5, truthful.get("salience").getAsDouble());
		assertEquals("bakery was being decorated with lights. baker and freight were there.",
			Communication.answer(news(truthful), true));
		assertEquals(Communication.answer(news(legacy)), Communication.answer(news(truthful), true));
	}

	/** Turning truthful news off preserves the old landmark item's complete serialized JSON and answer. */
	@Test
	void disabledTruthfulNewsHasGoldenLandmarkBytesAndAnswer() {
		JsonObject item = file(change("minecraft:air", "minecraft:stone_bricks"), false, false, false);
		assertEquals("{\"newsId\":\"change:7\",\"kind\":\"bridge_or_port_change\",\"subject\":\"bakery\","
			+ "\"originEventId\":\"change:7\",\"chain\":[\"Witness\"],\"firsthand\":true,\"learnedAt\":110,"
			+ "\"originatedAt\":100,\"salience\":0.8,\"present\":[\"baker\",\"freight\"]}", item.toString());
		assertEquals("bridge_or_port_change at \"bakery\". baker and freight were there.",
			Communication.answer(news(item), false));
		assertEquals(Communication.answer(news(item)), Communication.answer(news(item), false));
		assertFalse(item.has("block"));
		assertFalse(item.has("action"));
		JsonObject originalFormat = change("minecraft:air", "minecraft:stone_bricks");
		originalFormat.remove("from");
		originalFormat.remove("to");
		assertEquals(item.toString(), file(originalFormat, false, false, false).toString());
	}

	/** Disabled truthful news even preserves the former misleading oven-decoration classification verbatim. */
	@Test
	void disabledTruthfulNewsHasGoldenOvenDecorationBytesAndAnswer() {
		JsonObject item = file(change("minecraft:smoker[facing=north,lit=false]",
			"minecraft:smoker[facing=north,lit=true]"), true, false, false);
		assertEquals("{\"newsId\":\"change:7\",\"kind\":\"decoration\",\"subject\":\"bakery\","
			+ "\"originEventId\":\"change:7\",\"chain\":[\"Witness\"],\"firsthand\":true,\"learnedAt\":110,"
			+ "\"originatedAt\":100,\"salience\":0.5,\"present\":[\"baker\",\"freight\"],\"pos\":[1,2,3],"
			+ "\"place\":\"bakery\",\"text\":\"bakery was being decorated with lights.\"}", item.toString());
		assertEquals("bakery was being decorated with lights. baker and freight were there.",
			Communication.answer(news(item), false));
	}

	/** Road precedence and anonymous construction keep their old positional subjects and salience. */
	@Test
	void disabledTruthfulNewsHasGoldenRoadAndConstructionBytes() {
		JsonObject blocked = file(change("minecraft:air", "minecraft:barrier"), false, true, false);
		assertEquals("{\"newsId\":\"change:7\",\"kind\":\"road_blocked\",\"subject\":[1,2,3],"
			+ "\"originEventId\":\"change:7\",\"chain\":[\"Witness\"],\"firsthand\":true,\"learnedAt\":110,"
			+ "\"originatedAt\":100,\"salience\":0.9,\"present\":[\"baker\",\"freight\"]}", blocked.toString());
		assertEquals("road_blocked at [1,2,3]. baker and freight were there.",
			Communication.answer(news(blocked), false));
		JsonObject outside = change("minecraft:air", "minecraft:stone_bricks");
		outside.add("pos", array("[30,2,30]"));
		JsonObject construction = file(outside, false, false, false);
		assertEquals("{\"newsId\":\"change:7\",\"kind\":\"construction\",\"subject\":[30,2,30],"
			+ "\"originEventId\":\"change:7\",\"chain\":[\"Witness\"],\"firsthand\":true,\"learnedAt\":110,"
			+ "\"originatedAt\":100,\"salience\":0.2,\"present\":[]}", construction.toString());
		assertEquals("construction at [30,2,30]", Communication.answer(news(construction), false));
	}

	/** The already-gated emission flag remains authoritative rather than treating every lantern as decoration. */
	@Test
	void aClosedEmissionGateRetainsLegacyLandmarkClassification() {
		JsonObject item = file(change("minecraft:air", "minecraft:lantern[hanging=false]"), false, false, false);
		assertEquals("bridge_or_port_change", item.get("kind").getAsString());
		assertFalse(item.has("place"));
	}

	/** Change reports use the specified human sentences, simple plurals and the existing presence suffix. */
	@Test
	void truthfulAnswersRenderPlacedRemovedAndUnknownChangesNaturally() {
		JsonObject placed = report("placed", "town hall", "lantern", "placed", 100L);
		placed.add("present", array("[\"baker\",\"freight\"]"));
		assertEquals("Someone put up lanterns at the town hall. baker and freight were there.",
			Communication.answer(news(placed), true));
		assertEquals("Something was taken down at the bakery.", Communication.answer(news(
			report("removed", "bakery", "stone_bricks", "removed", 100L)), true));
		JsonObject unknown = report("changed", "the bridge", "unknown", "changed", 100L);
		assertEquals("There was work on the bridge.", Communication.answer(news(unknown), true));
		unknown.remove("block");
		assertEquals("There was work on the bridge.", Communication.answer(news(unknown), true));
		assertEquals("I have no news.", Communication.answer(new JsonArray(), true));
	}

	/** Truthful rendering leaves already-natural load, night, sold-out and decoration sentences alone. */
	@Test
	void existingNaturalKindsRetainTheirAnswers() {
		JsonArray items = array("[{\"kind\":\"load\",\"subject\":\"freight\",\"item\":\"minecraft:bread\","
			+ "\"count_band\":\"sack\",\"place\":\"market\"},{\"kind\":\"night\",\"subject\":\"baker\",\"place\":\"dock\"},"
			+ "{\"kind\":\"sold_out\",\"subject\":\"market\",\"item\":\"minecraft:baked_potato\"},"
			+ "{\"kind\":\"decoration\",\"subject\":\"town hall\",\"present\":[\"builder\"]}]");
		String expected = "freight was carrying a sack of bread at market; baker was outside at night at dock; "
			+ "The stall at market had sold out of baked potato; town hall was being decorated with lights. "
			+ "builder was there.";
		assertEquals(expected, Communication.answer(items, false));
		assertEquals(expected, Communication.answer(items, true));
		for (JsonElement value : items) {
			JsonArray single = news(value.getAsJsonObject());
			assertEquals(Communication.answer(single), Communication.answer(single, true));
		}
	}

	/** Implicit legacy defaults admit change only through its own opt-in, never through unrelated news mechanics. */
	@Test
	void changeKindRequiresTruthfulNewsEvenWhenOtherMechanicsAreEnabled() {
		JsonObject scenario = scenario();
		JsonObject saved = scenario.deepCopy();
		assertFalse(VillagerNews.enabled("change", scenario, false, false, false));
		assertFalse(VillagerNews.enabled("change", scenario, false, false, false, false));
		assertFalse(VillagerNews.enabled("change", scenario, true, true, true, false));
		assertTrue(VillagerNews.enabled("change", scenario, false, false, false, true));
		assertTrue(VillagerNews.enabled("change", scenario, true, true, true, true));
		assertEquals(saved, scenario);
	}

	/** Explicit filters, including the old four-kind list, suppress change despite an enabled truthful mechanic. */
	@Test
	void explicitExcludedKindsSuppressTruthfulChangeNews() {
		for (String filter : new String[]{"[]", "[\"construction\"]",
				"[\"bridge_or_port_change\",\"construction\",\"road_blocked\",\"shout\"]"}) {
			JsonObject args = scenario();
			args.add("news_kinds", array(filter));
			JsonObject filtered = VillagerNews.scenario(args, true);
			assertFalse(VillagerNews.enabled("change", filtered, false, false, false, true), filter);
			assertFalse(VillagerNews.enabled("change", filtered, true, true, true, true), filter);
		}
	}

	/** Scenario validation accepts an explicitly requested change kind without bypassing its feature gate. */
	@Test
	void explicitlyListedChangeIsValidatedAndStillRequiresTruthfulNews() {
		JsonObject args = scenario();
		args.add("news_kinds", array("[\"construction\",\"change\"]"));
		JsonObject saved = args.deepCopy();
		VillagerNews.validate(args);
		JsonObject filtered = VillagerNews.scenario(args, true);
		assertEquals(array("[\"change\",\"construction\"]"), filtered.get("news_kinds"));
		assertTrue(VillagerNews.enabled("change", filtered, false, false, false, true));
		assertFalse(VillagerNews.enabled("change", filtered, true, true, true, false));
		assertEquals(saved, args);
	}

	/** Names match complete subjects, present names and explicit place/landmark aliases, ignoring case. */
	@Test
	void matchingUsesCaseInsensitiveNamesAndAliasesRatherThanTextFragments() {
		JsonObject subject = report("subject", "BaKeR", "smoker", "lit", 100L);
		subject.add("present", array("[\"baker\"]"));
		JsonObject present = report("present", "bakery", "smoker", "lit", 90L);
		present.add("present", array("[\"freight\",\"BAKER\"]"));
		JsonObject alias = report("alias", "town hall", "lantern", "placed", 80L);
		alias.addProperty("place", "Civic Square");
		alias.addProperty("landmark", "Council Hall");
		JsonObject mentioned = report("mentioned", "dock", "lantern", "placed", 110L);
		mentioned.addProperty("text", "baker spoke about Civic Square and Council Hall.");
		JsonArray items = news(mentioned, alias, present, subject);
		assertEquals(array("[\"subject\",\"present\"]"), ids(VillagerNews.matching(items, "bAkEr", null, 3)));
		for (String name : new String[]{"TOWN HALL", "cIVIC sQUARE", "council HALL"}) {
			assertEquals(array("[\"alias\"]"), ids(VillagerNews.matching(items, name, null, 3)));
		}
		assertTrue(VillagerNews.matching(items, "bake", null, 3).isEmpty());
		assertTrue(VillagerNews.matching(items, "unknown", null, 3).isEmpty());
	}

	/** Both legacy positional subjects and explicit positions resolve through named bounds and place anchors. */
	@Test
	void matchingResolvesPositionsOnlyWhenAScenarioIsAvailable() {
		JsonObject positionalSubject = report("subject", "unused", "stone_bricks", "changed", 100L);
		positionalSubject.add("subject", array("[1,2,3]"));
		JsonObject boundedPosition = report("pos", "construction site", "stone_bricks", "changed", 90L);
		boundedPosition.add("pos", array("[2,4,4]"));
		JsonObject anchoredPosition = report("market", "unloading", "stone_bricks", "changed", 80L);
		anchoredPosition.add("pos", array("[20,2,3]"));
		JsonArray items = news(anchoredPosition, boundedPosition, positionalSubject);
		assertEquals(array("[\"subject\",\"pos\"]"),
			ids(VillagerNews.matching(items, "BAKERY", scenario(), 3)));
		assertEquals(array("[\"market\"]"), ids(VillagerNews.matching(items, "MaRkEt", scenario(), 3)));
		assertTrue(VillagerNews.matching(items, "bakery", null, 3).isEmpty());
		assertTrue(VillagerNews.matching(items, "market", null, 3).isEmpty());
	}

	/** Origins, not salience or last learning time, control recency; tied ids sort lexically before truncation. */
	@Test
	void matchingOrdersOriginsAndTiesBeforeApplyingTheThreeItemLimit() {
		JsonObject older = report("older", "bakery", "lantern", "placed", 80L);
		older.addProperty("salience", 0.9);
		older.addProperty("learnedAt", 9_999L);
		JsonObject tiedSecond = report("change:2", "bakery", "lantern", "placed", 110L);
		JsonObject newest = report("change:20", "bakery", "lantern", "placed", 120L);
		JsonObject tiedFirst = report("change:10", "bakery", "lantern", "placed", 110L);
		JsonObject unrelated = report("unrelated", "dock", "lantern", "placed", 130L);
		JsonArray items = news(older, tiedSecond, unrelated, newest, tiedFirst);
		assertEquals(array("[\"change:20\",\"change:10\",\"change:2\"]"),
			ids(VillagerNews.matching(items, "BAKERY", null, 3)));
		assertEquals(array("[\"change:20\",\"change:10\",\"change:2\",\"older\"]"),
			ids(VillagerNews.matching(items, "bakery", null, 10)));
		assertEquals(array("[\"change:20\"]"), ids(VillagerNews.matching(items, "bakery", null, 1)));
		assertTrue(VillagerNews.matching(items, "bakery", null, 0).isEmpty());
	}

	/** Filtering never sorts or decorates stored memory in place, and every returned nested value is detached. */
	@Test
	void matchingIsRepeatableAndReturnsDetachedItemsWithoutMutatingItsInputs() {
		JsonObject first = report("first", "bakery", "smoker", "lit", 100L);
		first.add("present", array("[\"baker\"]"));
		first.add("pos", array("[1,2,3]"));
		JsonArray items = news(report("older", "bakery", "lantern", "placed", 90L), first);
		JsonObject scenario = scenario();
		JsonArray savedItems = items.deepCopy();
		JsonObject savedScenario = scenario.deepCopy();
		JsonArray selected = VillagerNews.matching(items, "bakery", scenario, 3);
		JsonArray repeated = VillagerNews.matching(items, "bakery", scenario, 3);
		assertEquals(selected.toString(), repeated.toString());
		assertEquals(savedItems, items);
		assertEquals(savedScenario, scenario);
		JsonObject selectedFirst = selected.get(0).getAsJsonObject();
		selectedFirst.addProperty("subject", "fabricated");
		selectedFirst.getAsJsonArray("present").add("invented");
		selectedFirst.getAsJsonArray("chain").add("invented");
		selectedFirst.getAsJsonArray("pos").add(99);
		selected.remove(1);
		assertEquals(savedItems, items);
		assertEquals(savedScenario, scenario);
		assertEquals(array("[\"first\",\"older\"]"), ids(repeated));
		assertEquals(savedItems.get(1), repeated.get(0));
	}

	/** Minecraft noon, evening and night start at ticks 6000, 12000 and 16000, not wall-clock hours. */
	@Test
	void timeWordsChangeAtEachMinecraftDayBoundary() {
		long now = 1_000_000L;
		long[] phases = {0L, 5_999L, 6_000L, 11_999L, 12_000L, 15_999L, 16_000L, 23_999L, 24_000L};
		String[] expected = {"this morning", "this morning", "this afternoon", "this afternoon",
			"this evening", "this evening", "last night", "last night", "this morning"};
		for (int index = 0; index < phases.length; index++) {
			assertEquals(expected[index], Communication.when(now, now, phases[index]), "dayTime=" + phases[index]);
		}
	}

	/** Dawn retains the immediately preceding night but labels older nights and yesterday's daytime yesterday. */
	@Test
	void priorNightAndYesterdayRespectTheDawnBoundary() {
		long now = 60_000L;
		assertEquals("last night", Communication.when(now - 1L, now, 0L));
		assertEquals("last night", Communication.when(now - 8_000L, now, 0L));
		assertEquals("yesterday", Communication.when(now - 8_001L, now, 0L));
		assertEquals("yesterday", Communication.when(now - 24_000L, now, 0L));
		assertEquals("yesterday", Communication.when(now - 24_001L, now, 0L));
		assertEquals("yesterday", Communication.when(now - 32_000L, now, 0L));
		assertEquals("last night", Communication.when(now - 1L, now, 24_000L));
		assertEquals("yesterday", Communication.when(now - 24_001L, now, 24_000L));
	}

	/** Game time dates origins, while the current day-time phase supplies the world's independent clock offset. */
	@Test
	void originTimesUseTheCurrentDayTimeOffsetRatherThanGameTimeModulo() {
		long now = 100_000L;
		for (long dayTime : new long[]{12_000L, 60_000L}) {
			assertEquals("this morning", Communication.when(now - 12_000L, now, dayTime));
			assertEquals("this morning", Communication.when(now - 6_001L, now, dayTime));
			assertEquals("this afternoon", Communication.when(now - 6_000L, now, dayTime));
			assertEquals("this afternoon", Communication.when(now - 1L, now, dayTime));
			assertEquals("this evening", Communication.when(now, now, dayTime));
			assertEquals("last night", Communication.when(now - 12_001L, now, dayTime));
			assertEquals("yesterday", Communication.when(now - 36_001L, now, dayTime));
		}
		assertEquals("this evening", Communication.when(now - 1L, now, 16_000L));
		assertEquals("last night", Communication.when(now, now, 16_000L));
	}

	/** About answers describe the original event's time and reserve heard wording for second-hand memory. */
	@Test
	void aboutAnswersRenderNaturalFactsAndOriginalTimeWithSecondhandAttribution() {
		JsonObject item = report("oven", "bakery", "smoker", "lit", 45_000L);
		item.addProperty("learnedAt", 50_000L);
		item.add("present", array("[\"baker\",\"freight\"]"));
		JsonArray items = news(item);
		String firsthand = Communication.answerAbout(items, "bakery", 50_000L, 7_000L);
		assertEquals("The oven at the bakery was lit this morning. baker and freight were there.", firsthand);
		item.addProperty("firsthand", false);
		item.add("chain", array("[\"Witness\",\"Messenger\"]"));
		String secondhand = Communication.answerAbout(items, "bakery", 50_000L, 7_000L);
		assertEquals("I heard that the oven at the bakery was lit this morning. baker and freight were there.",
			secondhand);
		item.add("present", new JsonArray());
		assertEquals("I heard that the oven at the bakery was lit this morning.",
			Communication.answerAbout(items, "bakery", 50_000L, 7_000L));
	}

	/** Each report keeps its own clock and relay status instead of attributing every fact to a messenger. */
	@Test
	void aboutAnswersKeepFirsthandAndSecondhandReportsSeparate() {
		JsonObject firsthand = report("first", "bakery", "smoker", "lit", 45_000L);
		JsonObject secondhand = report("second", "bridge", "unknown", "changed", 49_500L);
		secondhand.addProperty("firsthand", false);
		String answer = Communication.answerAbout(news(firsthand, secondhand), "baker", 50_000L, 7_000L);
		assertTrue(answer.contains("this morning"), answer);
		assertTrue(answer.contains("this afternoon"), answer);
		assertTrue(answer.toLowerCase(Locale.ROOT).contains("oven at the bakery"), answer);
		assertTrue(answer.toLowerCase(Locale.ROOT).contains("work on the bridge"), answer);
		assertFalse(answer.startsWith("I heard that "), answer);
		assertEquals(1, answer.split("I heard that ", -1).length - 1);
	}

	/** Empty filtered memory uses the exact no-match sentence and preserves the name as it was asked. */
	@Test
	void noMatchPreservesTheQuestionInTheExactSentence() {
		assertEquals("I haven't seen or heard anything about BaKeR.",
			Communication.answerAbout(new JsonArray(), "BaKeR", 50_000L, 7_000L));
		assertEquals("I haven't seen or heard anything about town hall.",
			Communication.answerAbout(new JsonArray(), "town hall", 50_000L, 7_000L));
	}

	/** A disabled ask-about gate ignores even malformed argument values without validating or mutating them. */
	@Test
	void disabledAboutIgnoresAllArgumentTypes() {
		for (String value : new String[]{"null", "true", "17", "[]", "{}", "\"\"", "\"   \"", "\"BaKeR\""}) {
			JsonObject args = json("{\"npc\":\"Witness\",\"about\":" + value + "}");
			JsonObject saved = args.deepCopy();
			assertNull(Communication.about(args, false));
			assertEquals(saved, args);
		}
		assertNull(Communication.about(json("{\"npc\":17,\"about\":{}}"), false));
		assertNull(Communication.about(json("{\"about\":[]}"), false));
		assertNull(Communication.about(new JsonObject(), false));
	}

	/** Enabled, absent about retains ordinary ask, while a valid question preserves its spelling and input. */
	@Test
	void enabledAboutIsOptionalAndReturnsTheValidatedQuestion() {
		assertNull(Communication.about(new JsonObject(), true));
		assertNull(Communication.about(json("{\"npc\":\"Witness\"}"), true));
		assertNull(Communication.about(json("{\"about\":\"BaKeR\"}"), true));
		JsonObject args = json("{\"npc\":\"Witness\",\"about\":\"BaKeR\"}");
		JsonObject saved = args.deepCopy();
		assertEquals("BaKeR", Communication.about(args, true));
		assertEquals("BaKeR", Communication.about(args, true));
		assertEquals(saved, args);
	}

	/** Enabled about rejects coercible nonstrings and blank names instead of silently changing the question. */
	@Test
	void enabledAboutRejectsInvalidTypesAndBlankNames() {
		for (String value : new String[]{"null", "true", "17", "[]", "{}", "\"\"", "\" \\t\\n \""}) {
			JsonObject args = json("{\"npc\":\"Witness\",\"about\":" + value + "}");
			JsonObject saved = args.deepCopy();
			assertThrows(IllegalArgumentException.class, () -> Communication.about(args, true), value);
			assertEquals(saved, args);
		}
	}

	/** Exchange/event metadata records the exact question and supplied source-id order, not memory order or facts. */
	@Test
	void recordAboutStoresExactQuestionAndOrderedIdsWithoutChangingOtherMetadata() {
		JsonArray selected = news(report("change:20", "bakery", "smoker", "lit", 120L),
			report("change:10", "bakery", "smoker", "unlit", 110L),
			report("change:2", "bakery", "lantern", "placed", 110L));
		JsonArray saved = selected.deepCopy();
		JsonObject target = json("{\"type\":\"ask\",\"speaker\":\"mayor\",\"exchange\":{\"npc\":\"Witness\"},"
			+ "\"about\":\"stale\",\"newsIds\":[\"old\"]}");
		String about = "  BaKeR at the Town Hall  ";
		Communication.recordAbout(target, about, selected);
		assertEquals(json("{\"type\":\"ask\",\"speaker\":\"mayor\",\"exchange\":{\"npc\":\"Witness\"},"
			+ "\"about\":\"  BaKeR at the Town Hall  \",\"newsIds\":[\"change:20\",\"change:10\",\"change:2\"]}"), target);
		assertEquals(saved, selected);
		String recorded = target.toString();
		Communication.recordAbout(target, about, selected);
		assertEquals(recorded, target.toString());
	}

	/** Separate event/exchange targets and the selected memory cannot alias any stored source-id array. */
	@Test
	void recordAboutDetachesIdsFromMemoryAndFromOtherTargets() {
		JsonArray selected = news(report("first", "bakery", "smoker", "lit", 100L),
			report("second", "bakery", "smoker", "unlit", 90L));
		JsonArray saved = selected.deepCopy();
		JsonObject event = new JsonObject();
		JsonObject exchange = new JsonObject();
		Communication.recordAbout(event, "BaKeR", selected);
		Communication.recordAbout(exchange, "BaKeR", selected);
		event.getAsJsonArray("newsIds").remove(0);
		event.getAsJsonArray("newsIds").add("fabricated");
		assertEquals(saved, selected);
		assertEquals(array("[\"first\",\"second\"]"), exchange.get("newsIds"));
		selected.get(0).getAsJsonObject().addProperty("newsId", "changed-memory");
		selected.remove(1);
		assertEquals(array("[\"second\",\"fabricated\"]"), event.get("newsIds"));
		assertEquals(array("[\"first\",\"second\"]"), exchange.get("newsIds"));
		assertEquals("BaKeR", event.get("about").getAsString());
		assertEquals("BaKeR", exchange.get("about").getAsString());
	}

	/** No-match metadata explicitly replaces stale ids with an empty array while retaining the exact question. */
	@Test
	void recordAboutRecordsAnExplicitEmptyIdArrayForNoMatches() {
		JsonObject target = json("{\"type\":\"ask\",\"about\":\"old question\",\"newsIds\":[\"old-source\"]}");
		JsonArray empty = new JsonArray();
		Communication.recordAbout(target, "ToWn HaLl", empty);
		assertEquals(json("{\"type\":\"ask\",\"about\":\"ToWn HaLl\",\"newsIds\":[]}"), target);
		assertTrue(empty.isEmpty());
		Communication.recordAbout(target, "ToWn HaLl", empty);
		assertEquals(new JsonArray(), target.get("newsIds"));
		assertEquals("I haven't seen or heard anything about ToWn HaLl.",
			Communication.answerAbout(empty, target.get("about").getAsString(), 110L, 6_000L));
	}

	/** Repeating the complete filing, selection and answer path is byte-stable and does not alter source JSON. */
	@Test
	void repeatedWitnessInputsProduceIdenticalItemsAndAnswersWithoutMutation() {
		JsonObject scenario = scenario();
		JsonObject change = change("minecraft:smoker[facing=north,lit=false]",
			"minecraft:smoker[facing=north,lit=true]");
		JsonObject observation = observation(change);
		JsonObject savedScenario = scenario.deepCopy();
		JsonObject savedObservation = observation.deepCopy();
		JsonObject savedChange = change.deepCopy();
		JsonObject first = VillagerNews.blockItem(scenario, "Witness", observation, change, true, false, true);
		JsonObject second = VillagerNews.blockItem(scenario, "Witness", observation, change, true, false, true);
		assertEquals(first.toString(), second.toString());
		JsonArray memory = news(first);
		JsonArray savedMemory = memory.deepCopy();
		JsonArray selected = VillagerNews.matching(memory, "BAKER", scenario, 3);
		assertEquals(array("[\"change:7\"]"), ids(selected));
		String answer = Communication.answerAbout(selected, "BAKER", 110L, 6_000L);
		assertEquals(answer, Communication.answerAbout(selected, "BAKER", 110L, 6_000L));
		assertEquals(Communication.answer(selected, true), Communication.answer(selected, true));
		assertEquals(savedScenario, scenario);
		assertEquals(savedObservation, observation);
		assertEquals(savedChange, change);
		assertEquals(savedMemory, memory);
		first.getAsJsonArray("present").add("invented");
		first.getAsJsonArray("chain").add("invented");
		if (first.has("pos")) {
			first.getAsJsonArray("pos").add(99);
		}
		assertEquals(savedObservation, observation);
		assertEquals(savedChange, change);
		assertEquals(savedMemory.get(0), selected.get(0));
	}

	/** Checks the shared, landmark-specific classification contract for every truthful block example. */
	private static void assertChange(JsonObject item, String block, String action) {
		assertNotNull(item);
		assertEquals("change", item.get("kind").getAsString());
		assertEquals("bakery", item.get("subject").getAsString());
		assertEquals(block, item.get("block").getAsString());
		assertEquals(action, item.get("action").getAsString());
		assertEquals(0.8, item.get("salience").getAsDouble());
	}

	/** Files one admitted change with explicit emission and road decisions, without world access. */
	private static JsonObject file(JsonObject change, boolean emitting, boolean blocked, boolean truthfulNews) {
		return VillagerNews.blockItem(scenario(), "Witness", observation(change), change, emitting, blocked, truthfulNews);
	}

	/** A named bakery and a separate market anchor make positional place resolution unambiguous. */
	private static JsonObject scenario() {
		return VillagerNews.scenario(json("{\"region\":{\"min\":[-40,0,-40],\"max\":[40,20,40]},"
			+ "\"places\":{\"bakery\":[1,2,3],\"market\":[20,2,3]},"
			+ "\"landmarks\":[{\"name\":\"bakery\",\"region\":{\"min\":[0,1,2],\"max\":[2,4,4]}}]}"), true);
	}

	/** Events and their later observations deliberately use distinct times to catch provenance drift. */
	private static JsonObject change(String from, String to) {
		JsonObject change = json("{\"id\":7,\"pos\":[1,2,3],\"gameTime\":100,\"losRay\":0}");
		change.addProperty("from", from);
		change.addProperty("to", to);
		return change;
	}

	/** Only the two nearby, admitted bots contribute presence; duplicate and NPC names do not. */
	private static JsonObject observation(JsonObject change) {
		JsonObject observation = json("{\"prov\":\"npc_observation:8\",\"observer\":\"npc:witness\",\"gameTime\":110,"
			+ "\"entities\":[{\"name\":\"freight\",\"pos\":[1.5,2.5,3.5],\"losRay\":0},"
			+ "{\"name\":\"baker\",\"pos\":[2.5,2.5,3.5],\"losRay\":1},"
			+ "{\"name\":\"freight\",\"pos\":[1.5,2.5,3.5],\"losRay\":2},"
			+ "{\"name\":\"villager\",\"npcId\":\"adult_01\",\"pos\":[1.5,2.5,3.5],\"losRay\":0}]}");
		observation.add("changes", news(change));
		return observation;
	}

	/** Supplies complete memory metadata but no cached text, exercising natural rendering from structured facts. */
	private static JsonObject report(String id, String subject, String block, String action, long originatedAt) {
		JsonObject item = json("{\"kind\":\"change\",\"firsthand\":true,\"chain\":[\"Witness\"],\"present\":[],\"salience\":0.8}");
		item.addProperty("newsId", id);
		item.addProperty("originEventId", id);
		item.addProperty("subject", subject);
		item.addProperty("block", block);
		item.addProperty("action", action);
		item.addProperty("originatedAt", originatedAt);
		item.addProperty("learnedAt", originatedAt);
		return item;
	}

	/** Keeps the caller's objects as memory references so accidental source mutation is observable. */
	private static JsonArray news(JsonObject... items) {
		JsonArray news = new JsonArray();
		for (JsonObject item : items) {
			news.add(item);
		}
		return news;
	}

	/** Compares both selection membership and order without coupling assertions to unrelated optional fields. */
	private static JsonArray ids(JsonArray items) {
		JsonArray ids = new JsonArray();
		for (JsonElement value : items) {
			ids.add(value.getAsJsonObject().get("newsId").getAsString());
		}
		return ids;
	}

	/** Parses a small deterministic object fixture without shared mutable state. */
	private static JsonObject json(String text) {
		return JsonParser.parseString(text).getAsJsonObject();
	}

	/** Parses an expected JSON array or an explicit ordered fixture. */
	private static JsonArray array(String text) {
		return JsonParser.parseString(text).getAsJsonArray();
	}
}
