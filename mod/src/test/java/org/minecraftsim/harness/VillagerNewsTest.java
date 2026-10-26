package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.util.Map;
import java.util.Set;
import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.Test;

/** Pins observation-grounded news, opt-in gates and detached relay metadata without a world or mock objects. */
final class VillagerNewsTest {
	/** Only bots seen with LOS within eight blocks of this block's center contribute sorted, unique names. */
	@Test
	void presenceUsesTheSameObservationAndChangedBlock() {
		JsonObject observation = observation();
		JsonObject change = observation.getAsJsonArray("changes").get(0).getAsJsonObject();
		JsonObject news = VillagerNews.observedItem("bridge_or_port_change", new JsonPrimitive("the bridge"),
			"Witness", observation, change);
		assertEquals(JsonParser.parseString("[\"alice\",\"envoy\"]"), news.get("present"));
		change.add("pos", JsonParser.parseString("[30,2,3]"));
		assertTrue(VillagerNews.observedItem("construction", change.get("pos"), "Witness", observation, change)
			.getAsJsonArray("present").isEmpty());
	}

	/** Presence uses three-dimensional, inclusive distance rather than observer distance or a horizontal box. */
	@Test
	void presenceIncludesTheExactBoundaryButNotTheNextRepresentableDistance() {
		JsonObject observation = observation();
		JsonObject change = observation.getAsJsonArray("changes").get(0).getAsJsonObject();
		JsonObject entity = JsonParser.parseString("{\"name\":\"envoy\",\"pos\":[1.5,10.5,3.5],\"losRay\":0}")
			.getAsJsonObject();
		JsonArray entities = new JsonArray();
		entities.add(entity);
		observation.add("entities", entities);
		assertEquals(JsonParser.parseString("[\"envoy\"]"), VillagerNews.observedItem("construction",
			change.get("pos"), "Witness", observation, change).get("present"));
		entity.getAsJsonArray("pos").set(1, new JsonPrimitive(Math.nextUp(10.5)));
		assertTrue(VillagerNews.observedItem("construction", change.get("pos"), "Witness", observation, change)
			.getAsJsonArray("present").isEmpty());
	}

	/** Initial learning time comes from the observation, while event time and detached presence remain unchanged. */
	@Test
	void firstHandLearningTimeUsesTheSourceObservationAndEvent() {
		JsonObject observation = observation();
		JsonObject change = observation.getAsJsonArray("changes").get(0).getAsJsonObject();
		JsonObject news = VillagerNews.observedItem("bridge_or_port_change", new JsonPrimitive("the bridge"),
			"Witness", observation, change);
		assertEquals("change:7", news.get("newsId").getAsString());
		assertEquals("change:7", news.get("originEventId").getAsString());
		assertTrue(news.get("firsthand").getAsBoolean());
		assertEquals(100L, news.get("originatedAt").getAsLong());
		assertEquals(110L, news.get("learnedAt").getAsLong());
		assertEquals(JsonParser.parseString("[\"Witness\"]"), news.get("chain"));
		assertFalse(news.has("observation"));
		observation.addProperty("gameTime", 999L);
		observation.getAsJsonArray("entities").get(0).getAsJsonObject().addProperty("name", "changed");
		assertEquals(110L, news.get("learnedAt").getAsLong());
		assertEquals(JsonParser.parseString("[\"alice\",\"envoy\"]"), news.get("present"));
	}

	/** Neither NPC sightings nor an empty accident scene become attributed agents. */
	@Test
	void absentAgentsProduceAnExplicitEmptyPresenceList() {
		JsonObject observation = observation();
		JsonObject change = observation.getAsJsonArray("changes").get(0).getAsJsonObject();
		observation.add("entities", JsonParser.parseString("[{\"name\":\"envoy\",\"npcId\":\"adult_01\","
			+ "\"pos\":[1.5,2.5,3.5],\"losRay\":0}]"));
		assertTrue(VillagerNews.observedItem("construction", change.get("pos"), "Witness", observation, change)
			.getAsJsonArray("present").isEmpty());
		observation.add("entities", new JsonArray());
		assertTrue(VillagerNews.observedItem("construction", change.get("pos"), "Witness", observation, change)
			.getAsJsonArray("present").isEmpty());
	}

	/** Multiple deliveries preserve presence while extending chains and replacing only learning time/status. */
	@Test
	void relaysPreservePresenceWithoutAliasingTheSource() {
		JsonObject observation = observation();
		JsonObject source = VillagerNews.observedItem("construction", new JsonPrimitive("the bridge"), "Witness",
			observation, observation.getAsJsonArray("changes").get(0).getAsJsonObject());
		JsonObject saved = source.deepCopy();
		JsonObject first = VillagerNews.relay(source, "Witness", 200L);
		JsonObject second = VillagerNews.relay(first, "Neighbour", 300L);
		assertEquals(saved, source);
		assertEquals(source.get("present"), second.get("present"));
		assertEquals(source.get("originatedAt"), second.get("originatedAt"));
		assertEquals(JsonParser.parseString("[\"Witness\",\"Neighbour\"]"), second.get("chain"));
		assertFalse(second.get("firsthand").getAsBoolean());
		assertEquals(300L, second.get("learnedAt").getAsLong());
		second.getAsJsonArray("present").add("fabricated");
		assertEquals(saved, source);
		assertEquals(source.get("present"), first.get("present"));
	}

	/** Relaying an old save does not invent presence. */
	@Test
	void legacyRelaysKeepAbsentPresenceAbsent() {
		JsonObject source = JsonParser.parseString("{\"chain\":[\"Witness\"],\"firsthand\":true,\"learnedAt\":110}")
			.getAsJsonObject();
		JsonObject relay = VillagerNews.relay(source, "Witness", 200L);
		assertFalse(relay.has("present"));
		assertFalse(relay.has("observation"));
		assertTrue(source.get("firsthand").getAsBoolean());
	}

	/** Opting out retains the exact legacy scenario fields, ordering and four sorted default kinds. */
	@Test
	void defaultScenarioRemainsByteIdentical() {
		JsonObject args = setup();
		args.remove("landmarks");
		JsonObject scenario = VillagerNews.scenario(args, false);
		assertEquals("{\"news_kinds\":[\"bridge_or_port_change\",\"construction\",\"road_blocked\",\"shout\"],"
			+ "\"region\":{\"min\":[-20,0,-20],\"max\":[20,20,20]},"
			+ "\"surfaces\":[\"minecraft:dirt_path\",\"minecraft:gravel\",\"minecraft:stone_bricks\"],\"landmarks\":[]}",
			scenario.toString());
		assertFalse(scenario.has("places"));
		assertFalse(scenario.has("news_kinds_explicit"));
		assertEquals(scenario.get("news_kinds"), VillagerNews.scenario(args, true).get("news_kinds"));
		assertEquals(args.get("places"), VillagerNews.scenario(args, true).get("places"));
		args.add("news_kinds", scenario.get("news_kinds").deepCopy());
		assertEquals(scenario.toString(), VillagerNews.scenario(args, false).toString());
	}

	/** New kinds require their own mechanic but remain available beside unchanged implicit legacy defaults. */
	@Test
	void newKindsFollowTheirMechanicGates() {
		JsonObject scenario = newsScenario();
		for (String kind : Set.of("load", "night", "decoration", "sold_out")) {
			assertFalse(VillagerNews.enabled(kind, scenario, false, false, false));
			assertTrue(VillagerNews.enabled(kind, scenario, true, true, true));
		}
		assertTrue(VillagerNews.enabled("load", scenario, false, true, false));
		assertFalse(VillagerNews.enabled("load", scenario, true, false, true));
		assertTrue(VillagerNews.enabled("night", scenario, true, false, false));
		assertTrue(VillagerNews.enabled("decoration", scenario, true, false, false));
		assertFalse(VillagerNews.enabled("night", scenario, false, true, true));
		assertFalse(VillagerNews.enabled("decoration", scenario, false, true, true));
		assertTrue(VillagerNews.enabled("sold_out", scenario, false, false, true));
		assertFalse(VillagerNews.enabled("sold_out", scenario, true, true, false));
		assertTrue(VillagerNews.enabled("construction", scenario, false, false, false));
		assertFalse(VillagerNews.enabled("unknown", scenario, true, true, true));
	}

	/** Explicit empty, legacy-default and extended filters suppress excluded kinds instead of enabling defaults. */
	@Test
	void explicitKindsRemainAuthoritative() {
		JsonObject args = setup();
		for (String filter : new String[] {"[]", "[\"construction\"]",
				"[\"bridge_or_port_change\",\"construction\",\"road_blocked\",\"shout\"]"}) {
			args.add("news_kinds", JsonParser.parseString(filter));
			JsonObject scenario = VillagerNews.scenario(args, true);
			for (String kind : Set.of("load", "night", "decoration", "sold_out")) {
				assertFalse(VillagerNews.enabled(kind, scenario, true, true, true));
			}
		}
		args.add("news_kinds", JsonParser.parseString("[\"load\",\"sold_out\"]"));
		JsonObject scenario = VillagerNews.scenario(args, true);
		assertTrue(VillagerNews.enabled("load", scenario, true, true, true));
		assertTrue(VillagerNews.enabled("sold_out", scenario, true, true, true));
		assertFalse(VillagerNews.enabled("night", scenario, true, true, true));
		assertFalse(VillagerNews.enabled("construction", scenario, true, true, true));
		assertFalse(VillagerNews.enabled("load", scenario, true, false, true));
		args.add("news_kinds", JsonParser.parseString("[\"not_news\"]"));
		assertThrows(IllegalArgumentException.class, () -> VillagerNews.validate(args));
	}

	/** Load reports contain only the admitted band, named carrier, observed location and exact sight provenance. */
	@Test
	void loadNewsUsesAdmittedSightingsWithoutExactInventoryCounts() {
		JsonObject observation = personObservation();
		JsonObject news = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			false, Map.of("minecraft:bread", 9)).get(0).getAsJsonObject();
		assertEquals("load", news.get("kind").getAsString());
		assertEquals("envoy", news.get("subject").getAsString());
		assertEquals("minecraft:bread", news.get("item").getAsString());
		assertEquals("sack", news.get("count_band").getAsString());
		assertEquals(JsonParser.parseString("[1.5,2.5,3.5]"), news.get("pos"));
		assertEquals("town hall", news.get("place").getAsString());
		assertEquals("sighting:0:load:entity:envoy:minecraft:bread:town hall", news.get("newsId").getAsString());
		assertEquals("npc_observation:8", news.get("originEventId").getAsString());
		assertEquals(110L, news.get("originatedAt").getAsLong());
		assertEquals(110L, news.get("learnedAt").getAsLong());
		assertEquals(JsonParser.parseString("[\"Witness\"]"), news.get("chain"));
		assertTrue(news.get("firsthand").getAsBoolean());
		assertFalse(news.has("count"));
		assertFalse(news.has("inventory"));
		assertEquals("envoy was carrying a sack of bread at town hall.", news.get("text").getAsString());
		JsonArray answer = new JsonArray();
		answer.add(news);
		assertTrue(Communication.answer(answer).contains(news.get("text").getAsString()));
		observation.getAsJsonArray("entities").get(0).getAsJsonObject().getAsJsonArray("pos").set(0, new JsonPrimitive(99));
		assertEquals(JsonParser.parseString("[1.5,2.5,3.5]"), news.get("pos"));
	}

	/** NPCs, anonymous or occluded people and missing or unconfigured loads never become person news. */
	@Test
	void personNewsRejectsUnadmittedPeopleAndDisabledLoads() {
		JsonObject observation = personObservation();
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, false, Map.of()).isEmpty());
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			false, Map.of("minecraft:wheat", 9)).isEmpty());
		JsonObject person = observation.getAsJsonArray("entities").get(0).getAsJsonObject();
		for (String mutation : new String[] {"{\"npcId\":\"adult_01\"}", "{\"name\":\" \"}",
				"{\"losRay\":null}", "{\"losRay\":-1}"}) {
			JsonObject invalid = person.deepCopy();
			JsonParser.parseString(mutation).getAsJsonObject().entrySet()
				.forEach(entry -> invalid.add(entry.getKey(), entry.getValue()));
			JsonArray entities = new JsonArray();
			entities.add(invalid);
			observation.add("entities", entities);
			assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation,
				true, Map.of("minecraft:bread", 9)).isEmpty());
		}
		person.remove("losRay");
		JsonArray entities = new JsonArray();
		entities.add(person);
		observation.add("entities", entities);
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			true, Map.of("minecraft:bread", 9)).isEmpty());
		person.addProperty("losRay", 0);
		person.remove("load");
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, false,
			Map.of("minecraft:bread", 9)).isEmpty());
	}

	/** Night news uses the shared day-time phase rather than elapsed game time or an invented outside flag. */
	@Test
	void nightNewsRequiresOutsideAndTheExactNightPhase() {
		JsonObject observation = personObservation();
		for (long phase : new long[] {12_999L, 13_000L, 22_999L, 23_000L, 23_001L}) {
			for (long day : new long[] {0L, 24_000L}) {
				observation.addProperty("dayTime", phase + day);
				JsonArray news = VillagerNews.observedPeople(newsScenario(), "Witness", observation, true, Map.of());
				assertEquals(phase >= 13_000L && phase < 23_000L ? 1 : 0, news.size());
			}
		}
		observation.addProperty("dayTime", 13_000L);
		JsonObject news = VillagerNews.observedPeople(newsScenario(), "Witness", observation, true, Map.of())
			.get(0).getAsJsonObject();
		assertEquals("night", news.get("kind").getAsString());
		assertEquals("envoy", news.get("subject").getAsString());
		assertEquals("envoy was outside at night at town hall.", news.get("text").getAsString());
		assertEquals(110L, news.get("originatedAt").getAsLong());
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, false, Map.of()).isEmpty());
		JsonObject person = observation.getAsJsonArray("entities").get(0).getAsJsonObject();
		person.addProperty("outside", false);
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, true, Map.of()).isEmpty());
		person.remove("outside");
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, true, Map.of()).isEmpty());
		person.addProperty("outside", true);
		observation.remove("dayTime");
		observation.addProperty("gameTime", 14_000L);
		assertTrue(VillagerNews.observedPeople(newsScenario(), "Witness", observation, true, Map.of()).isEmpty());
	}

	/** Stable person IDs distinguish kinds and people while relays retain detached facts and original provenance. */
	@Test
	void personIdsAndRelaysRetainDeterministicProvenance() {
		JsonObject observation = personObservation();
		JsonArray first = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			true, Map.of("minecraft:bread", 9));
		assertEquals(first, VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			true, Map.of("minecraft:bread", 9)));
		assertNotEquals(first.get(0).getAsJsonObject().get("newsId"), first.get(1).getAsJsonObject().get("newsId"));
		JsonObject other = observation.getAsJsonArray("entities").get(0).getAsJsonObject().deepCopy();
		other.addProperty("name", "alice");
		other.addProperty("prov", "entity:alice");
		observation.getAsJsonArray("entities").add(other);
		JsonArray both = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			true, Map.of("minecraft:bread", 9));
		assertEquals(4, both.size());
		assertNotEquals(both.get(0).getAsJsonObject().get("newsId"), both.get(2).getAsJsonObject().get("newsId"));
		JsonObject source = first.get(0).getAsJsonObject();
		JsonObject relay = VillagerNews.relay(source, "Neighbour", 300L);
		for (String key : Set.of("pos", "place", "item", "count_band", "subject", "originEventId", "originatedAt")) {
			assertEquals(source.get(key), relay.get(key));
		}
		assertEquals(JsonParser.parseString("[\"Witness\",\"Neighbour\"]"), relay.get("chain"));
		assertFalse(relay.get("firsthand").getAsBoolean());
		relay.getAsJsonArray("pos").set(0, new JsonPrimitive(99));
		assertEquals(1.5, source.getAsJsonArray("pos").get(0).getAsDouble());
	}

	/** Repeated twenty-tick sightings coalesce within a day and place while keeping their real source records. */
	@Test
	void recurringSightingsHaveStableDailyIdsWithoutLosingObservationProvenance() {
		JsonObject observation = personObservation();
		JsonObject first = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			false, Map.of("minecraft:bread", 9)).get(0).getAsJsonObject();
		observation.addProperty("prov", "npc_observation:9");
		observation.addProperty("gameTime", 130L);
		JsonObject repeated = VillagerNews.observedPeople(newsScenario(), "Neighbour", observation,
			false, Map.of("minecraft:bread", 9)).get(0).getAsJsonObject();
		assertEquals(first.get("newsId"), repeated.get("newsId"));
		assertEquals("npc_observation:8", first.get("originEventId").getAsString());
		assertEquals("npc_observation:9", repeated.get("originEventId").getAsString());
		assertEquals(110L, first.get("originatedAt").getAsLong());
		assertEquals(130L, repeated.get("originatedAt").getAsLong());
		observation.addProperty("gameTime", 24_110L);
		JsonObject tomorrow = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			false, Map.of("minecraft:bread", 9)).get(0).getAsJsonObject();
		assertNotEquals(first.get("newsId"), tomorrow.get("newsId"));
		observation.addProperty("gameTime", 110L);
		observation.getAsJsonArray("entities").get(0).getAsJsonObject().add("pos", JsonParser.parseString("[10,2,3]"));
		JsonObject elsewhere = VillagerNews.observedPeople(newsScenario(), "Witness", observation,
			false, Map.of("minecraft:bread", 9)).get(0).getAsJsonObject();
		assertNotEquals(first.get("newsId"), elsewhere.get("newsId"));
		assertEquals("market", elsewhere.get("place").getAsString());
	}

	/** Legacy fixture anchors are supplied ephemerally, leaving the original scenario bytes and caller data intact. */
	@Test
	void legacyScenarioPlaceFallbackDoesNotChangeSavedJson() {
		JsonObject scenario = VillagerNews.scenario(setup(), false);
		String saved = scenario.toString();
		JsonObject anchors = setup().getAsJsonObject("places");
		JsonObject runtime = VillagerNews.withPlaces(scenario, anchors);
		assertEquals(saved, scenario.toString());
		assertEquals(anchors, runtime.get("places"));
		anchors.add("market", JsonParser.parseString("[99,2,3]"));
		assertEquals(JsonParser.parseString("[10,2,3]"), runtime.getAsJsonObject("places").get("market"));
		assertEquals(runtime, VillagerNews.withPlaces(runtime, anchors));
	}

	/** Sold-out facts require opted-in scripted purchases and name the stall's place instead of its shopper. */
	@Test
	void soldOutNewsNamesTheStallAndLoggedEngineEvent() {
		assertFalse(VillagerNews.purchasesEnabled(new JsonObject()));
		assertFalse(VillagerNews.purchasesEnabled(JsonParser.parseString("{\"villager_purchases\":[]}").getAsJsonObject()));
		assertTrue(VillagerNews.purchasesEnabled(JsonParser.parseString("{\"villager_purchases\":["
			+ "{\"stall\":[10,2,3],\"item\":\"minecraft:bread\",\"count\":1}]}").getAsJsonObject()));
		JsonObject event = JsonParser.parseString("{\"id\":42,\"type\":\"sold_out\",\"gameTime\":200,"
			+ "\"npc\":\"adult_01\",\"stall\":[10,2,3],\"item\":\"minecraft:bread\"}").getAsJsonObject();
		JsonObject news = VillagerNews.soldOutItem(newsScenario(), "Witness", event);
		assertEquals("sold_out", news.get("kind").getAsString());
		assertEquals("market", news.get("subject").getAsString());
		assertEquals("market", news.get("place").getAsString());
		assertEquals(event.get("stall"), news.get("pos"));
		assertEquals("minecraft:bread", news.get("item").getAsString());
		assertEquals("sold_out:42", news.get("newsId").getAsString());
		assertEquals("sold_out:42", news.get("originEventId").getAsString());
		assertEquals(200L, news.get("originatedAt").getAsLong());
		assertEquals(200L, news.get("learnedAt").getAsLong());
		assertTrue(news.get("firsthand").getAsBoolean());
		assertEquals("The stall at market had sold out of bread.", news.get("text").getAsString());
		assertEquals(news, VillagerNews.soldOutItem(newsScenario(), "Witness", event));
		event.getAsJsonArray("stall").set(0, new JsonPrimitive(99));
		assertEquals(10, news.getAsJsonArray("pos").get(0).getAsInt());
	}

	/** Decorations require emitting replacements inside inclusive named bounds, not nearby place anchors. */
	@Test
	void decorationNewsRequiresAnEmittingChangeInsideNamedBounds() {
		JsonObject observation = observation();
		JsonObject change = observation.getAsJsonArray("changes").get(0).getAsJsonObject();
		assertNull(VillagerNews.decorationItem(newsScenario(), "Witness", observation, change, false));
		JsonObject news = VillagerNews.decorationItem(newsScenario(), "Witness", observation, change, true);
		assertEquals("decoration", news.get("kind").getAsString());
		assertEquals("town hall", news.get("subject").getAsString());
		assertEquals("town hall", news.get("place").getAsString());
		assertEquals(change.get("pos"), news.get("pos"));
		assertEquals("change:7", news.get("originEventId").getAsString());
		assertEquals(100L, news.get("originatedAt").getAsLong());
		assertEquals(110L, news.get("learnedAt").getAsLong());
		assertEquals(JsonParser.parseString("[\"alice\",\"envoy\"]"), news.get("present"));
		assertEquals("town hall was being decorated with lights.", news.get("text").getAsString());
		change.add("pos", JsonParser.parseString("[2,4,4]"));
		assertEquals("town hall", VillagerNews.decorationItem(newsScenario(), "Witness", observation, change, true)
			.get("subject").getAsString());
		change.add("pos", JsonParser.parseString("[3,4,4]"));
		assertNull(VillagerNews.decorationItem(newsScenario(), "Witness", observation, change, true));
		change.add("pos", JsonParser.parseString("[10,2,3]"));
		assertNull(VillagerNews.decorationItem(newsScenario(), "Witness", observation, change, true));
	}

	/** Named bounds define indoors without considering tree cover or treating nearby place anchors as roofs. */
	@Test
	void outsideUsesNamedBuildingBoundsRatherThanSkyVisibility() {
		JsonObject scenario = newsScenario();
		assertFalse(VillagerNews.outside(new Vec3(0, 1, 2), scenario));
		assertFalse(VillagerNews.outside(new Vec3(2.999, 4.999, 4.999), scenario));
		assertTrue(VillagerNews.outside(new Vec3(-0.001, 2, 3), scenario));
		assertTrue(VillagerNews.outside(new Vec3(3, 2, 3), scenario));
		assertTrue(VillagerNews.outside(new Vec3(1, 5, 3), scenario));
		assertTrue(VillagerNews.outside(new Vec3(10, 2, 3), scenario));
		scenario.add("landmarks", new JsonArray());
		assertTrue(VillagerNews.outside(new Vec3(1, 2, 3), scenario));
	}

	/** The new scenario retains named bounds and positional places without sharing mutable setup JSON. */
	private static JsonObject newsScenario() {
		return VillagerNews.scenario(setup(), true);
	}

	/** Provides one named building and a separate market anchor so their place semantics remain distinct. */
	private static JsonObject setup() {
		return JsonParser.parseString("{\"region\":{\"min\":[-20,0,-20],\"max\":[20,20,20]},"
			+ "\"places\":{\"market\":[10,2,3],\"town hall\":[0,2,3]},"
			+ "\"landmarks\":[{\"name\":\"town hall\",\"region\":{\"min\":[0,1,2],\"max\":[2,4,4]}}]}")
			.getAsJsonObject();
	}

	/** Supplies a named, admitted bot with a visible sack and outside flag at night, without exact counts. */
	private static JsonObject personObservation() {
		return JsonParser.parseString("{\"prov\":\"npc_observation:8\",\"gameTime\":110,\"dayTime\":13000,"
			+ "\"entities\":[{\"prov\":\"entity:envoy\",\"name\":\"envoy\",\"pos\":[1.5,2.5,3.5],\"losRay\":0,"
			+ "\"outside\":true,\"load\":{\"item\":\"minecraft:bread\",\"count_band\":\"sack\"}}]}")
			.getAsJsonObject();
	}

	/** Mirrors the Villagers sight schema: only bots lack npcId; admitted sight records carry losRay. */
	private static JsonObject observation() {
		return JsonParser.parseString("""
			{"prov":"npc_observation:8","observer":"npc:witness","gameTime":110,
			 "changes":[{"id":7,"pos":[1,2,3],"gameTime":100,"losRay":0}],
			 "entities":[
			  {"name":"envoy","pos":[1.5,2.5,3.5],"losRay":0},
			  {"name":"alice","pos":[9.5,2.5,3.5],"losRay":1},
			  {"name":"envoy","pos":[1.5,2.5,3.5],"losRay":2},
			  {"name":"villager","npcId":"adult_01","pos":[1.5,2.5,3.5],"losRay":0},
			  {"name":"distant","pos":[9.5,10.5,3.5],"losRay":0},
			  {"name":"hidden","pos":[1.5,2.5,3.5]},
			  {"name":"null-ray","pos":[1.5,2.5,3.5],"losRay":null},
			  {"name":"invalid-ray","pos":[1.5,2.5,3.5],"losRay":-1},
			  {"name":" ","pos":[1.5,2.5,3.5],"losRay":0}]}
			""").getAsJsonObject();
	}
}
