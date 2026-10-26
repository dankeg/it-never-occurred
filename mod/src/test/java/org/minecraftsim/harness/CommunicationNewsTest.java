package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

/** Pins ask presence wording and relay delivery using ordinary JSON values, without a server. */
final class CommunicationNewsTest {
	/** Old and empty presence do not alter the existing answer, including shout text and empty news. */
	@Test
	void absentPresenceRetainsExistingAnswers() {
		assertEquals("I have no news.", Communication.answer(new JsonArray()));
		JsonArray news = news();
		assertEquals("bridge_or_port_change at \"the bridge\"", Communication.answer(news));
		news.get(0).getAsJsonObject().add("present", new JsonArray());
		assertEquals("bridge_or_port_change at \"the bridge\"", Communication.answer(news));
		news.get(0).getAsJsonObject().addProperty("text", "A claim.");
		assertEquals("bridge_or_port_change at \"the bridge\": A claim.", Communication.answer(news));
	}

	/** Single and plural presence end each report with was/were there, using the recorded names in order. */
	@Test
	void answersEndPresenceClaimsWithSingularOrPluralWording() {
		JsonArray news = news();
		JsonObject item = news.get(0).getAsJsonObject();
		item.add("present", JsonParser.parseString("[\"envoy\"]"));
		assertEquals("bridge_or_port_change at \"the bridge\". envoy was there.", Communication.answer(news));
		item.add("present", JsonParser.parseString("[\"alice\",\"envoy\"]"));
		assertEquals("bridge_or_port_change at \"the bridge\". alice and envoy were there.", Communication.answer(news));
		item.add("present", JsonParser.parseString("[\"alice\",\"bob\",\"envoy\"]"));
		assertEquals("bridge_or_port_change at \"the bridge\". alice, bob and envoy were there.", Communication.answer(news));
		item.addProperty("text", "The deck fell.");
		assertEquals("bridge_or_port_change at \"the bridge\": The deck fell. alice, bob and envoy were there.",
			Communication.answer(news));
	}

	/** Separate reports keep their own presence instead of attributing every name to every changed block. */
	@Test
	void multipleReportsDoNotMixPresence() {
		JsonArray news = news();
		JsonObject first = news.get(0).getAsJsonObject();
		first.add("present", JsonParser.parseString("[\"alice\"]"));
		JsonObject second = first.deepCopy();
		second.addProperty("subject", "the port");
		second.add("present", JsonParser.parseString("[\"envoy\"]"));
		news.add(second);
		assertEquals("bridge_or_port_change at \"the bridge\". alice was there.; "
			+ "bridge_or_port_change at \"the port\". envoy was there.", Communication.answer(news));
	}

	/** The delivered news and convenience fields retain detached copies of presence. */
	@Test
	void deliveryPreservesPresenceAndLeavesTheUtteranceUnchanged() {
		JsonObject utterance = new JsonObject();
		utterance.addProperty("speaker", "Witness");
		utterance.add("news", news());
		JsonObject source = utterance.getAsJsonArray("news").get(0).getAsJsonObject();
		source.add("present", JsonParser.parseString("[\"envoy\"]"));
		JsonObject saved = utterance.deepCopy();
		JsonObject delivered = new JsonObject();
		Communication.relayed(utterance, delivered, 200L);
		JsonObject relay = delivered.getAsJsonArray("news").get(0).getAsJsonObject();
		assertEquals(saved, utterance);
		assertEquals(source.get("present"), relay.get("present"));
		assertEquals(source.get("present"), delivered.get("present"));
		assertEquals(200L, relay.get("learnedAt").getAsLong());
		assertFalse(relay.get("firsthand").getAsBoolean());
		relay.getAsJsonArray("present").add("fabricated");
		assertEquals(saved, utterance);
		assertEquals(source.get("present"), delivered.get("present"));
	}

	/** Delivery from a legacy save preserves missing optional fields and handles no-news utterances. */
	@Test
	void deliveryDoesNotInventPresenceForLegacyNews() {
		JsonObject utterance = new JsonObject();
		utterance.addProperty("speaker", "Witness");
		utterance.add("news", news());
		JsonObject delivered = new JsonObject();
		Communication.relayed(utterance, delivered, 200L);
		assertFalse(delivered.has("present"));
		assertFalse(delivered.has("observation"));
		utterance.add("news", new JsonArray());
		delivered = new JsonObject();
		Communication.relayed(utterance, delivered, 200L);
		assertEquals(new JsonObject(), delivered);
	}

	/** New kinds render as natural first-hand facts even when the optional summary text is absent. */
	@Test
	void newKindsHaveNaturalFallbackWording() {
		JsonArray news = JsonParser.parseString("["
			+ "{\"kind\":\"load\",\"subject\":\"envoy\",\"item\":\"minecraft:bread\",\"count_band\":\"sack\",\"place\":\"market\"},"
			+ "{\"kind\":\"night\",\"subject\":\"alice\",\"place\":\"dock\"},"
			+ "{\"kind\":\"sold_out\",\"subject\":\"market\",\"item\":\"minecraft:baked_potato\"},"
			+ "{\"kind\":\"decoration\",\"subject\":\"town hall\",\"present\":[\"builder\"]}]").getAsJsonArray();
		assertEquals("envoy was carrying a sack of bread at market; alice was outside at night at dock; "
			+ "The stall at market had sold out of baked potato; town hall was being decorated with lights. "
			+ "builder was there.", Communication.answer(news));
		news.get(0).getAsJsonObject().remove("place");
		news.get(1).getAsJsonObject().remove("place");
		assertEquals("envoy was carrying a sack of bread; alice was outside at night; "
			+ "The stall at market had sold out of baked potato; town hall was being decorated with lights. "
			+ "builder was there.", Communication.answer(news));
	}

	/** Natural new-kind summaries coexist with byte-identical legacy reports in the same answer. */
	@Test
	void newSummariesDoNotChangeLegacyWording() {
		JsonArray news = news();
		JsonObject load = JsonParser.parseString("{\"kind\":\"load\",\"subject\":\"envoy\","
			+ "\"item\":\"minecraft:bread\",\"text\":\"envoy was carrying a sack of bread at market.\"}").getAsJsonObject();
		news.add(load);
		JsonObject saved = load.deepCopy();
		assertEquals("bridge_or_port_change at \"the bridge\"; envoy was carrying a sack of bread at market.",
			Communication.answer(news));
		assertEquals(saved, load);
	}

	/** Supplies an old-format first-hand block item so optional presence is exercised explicitly. */
	private static JsonArray news() {
		return JsonParser.parseString("""
			[{"newsId":"change:7","originEventId":"change:7","kind":"bridge_or_port_change",
			  "subject":"the bridge","firsthand":true,"chain":["Witness"],"learnedAt":110}]
			""").getAsJsonArray();
	}
}
