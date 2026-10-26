package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Set;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.Test;

/** Verifies count-free completion facts and the allowlisted public activity projection without a server. */
final class SeenActionsTest {
	/** Transfer evidence has only item identity, actor, action, position, time and provenance. */
	@Test
	void completionFactsOmitCountsAndPrivateArguments() {
		JsonObject transfer = json("{\"id\":7,\"gameTime\":20,\"item\":\"minecraft:iron_pickaxe\",\"count\":12,\"note\":\"secret\"}");
		for (String action : new String[]{"chest_take", "chest_put"}) {
			JsonObject fact = SeenActions.fact("builder", action, new BlockPos(-8, -61, 0), transfer);
			assertEquals(Set.of("prov", "gameTime", "actor", "action", "pos", "item"), fact.keySet());
			assertEquals("transfer:7", fact.get("prov").getAsString());
			assertEquals("builder", fact.get("actor").getAsString());
			assertEquals(action, fact.get("action").getAsString());
			assertEquals("minecraft:iron_pickaxe", fact.get("item").getAsString());
			assertEquals(json("{\"pos\":[-8,-61,0]}").get("pos"), fact.get("pos"));
		}
		assertThrows(IllegalArgumentException.class,
			() -> SeenActions.fact("builder", "mine", BlockPos.ZERO, transfer));
	}

	/** Hand-overs reveal their recipient, never a container position or private event details. */
	@Test
	void giveFactsContainOnlyTheItemAndParticipants() {
		JsonObject event = json("{\"id\":8,\"gameTime\":21,\"from\":\"envoy\",\"to\":\"freight\","
			+ "\"item\":\"minecraft:iron_pickaxe\",\"count\":12,\"note\":\"secret\",\"args\":{\"count\":12}}");
		JsonObject fact = SeenActions.fact("envoy", "give", null, event);
		assertEquals(json("{\"prov\":\"transfer:8\",\"gameTime\":21,\"actor\":\"envoy\",\"action\":\"give\","
			+ "\"target\":\"freight\",\"item\":\"minecraft:iron_pickaxe\"}"), fact);
	}

	/** Drop and pickup witnesses refer to their own events while retaining exact non-block coordinates. */
	@Test
	void itemEntityFactsOmitCountsThrowersAndPrivateArguments() {
		JsonObject event = json("{\"id\":9,\"gameTime\":22,\"bot\":\"security\",\"pos\":[1.25,-60.5,2.75],"
			+ "\"item\":\"minecraft:iron_pickaxe\",\"count\":3,\"thrower\":\"secret\",\"prov\":\"entity:secret\","
			+ "\"itemEntity\":\"secret\",\"note\":\"secret\",\"args\":{\"count\":3}}");
		for (String action : new String[]{"drop", "pickup"}) {
			JsonObject fact = SeenActions.fact("security", action, null, event);
			assertEquals(Set.of("prov", "gameTime", "actor", "action", "pos", "item"), fact.keySet());
			assertEquals(action + ":9", fact.get("prov").getAsString());
			assertEquals(action, fact.get("action").getAsString());
			assertEquals("security", fact.get("actor").getAsString());
			assertEquals(event.get("pos"), fact.get("pos"));
			fact.getAsJsonArray("pos").set(0, new com.google.gson.JsonPrimitive(100));
			assertEquals(1.25, event.getAsJsonArray("pos").get(0).getAsDouble());
		}
	}

	/** Only physically targeted skill names expose a position, never counts, notes or arbitrary arguments. */
	@Test
	void activityExposesOnlyThePublicNameAndBlockTarget() {
		for (String skill : new String[]{"mine", "place", "chest_take", "chest_put", "buy"}) {
			Skills skills = restored(skill, "{\"pos\":[-8,-61,0],\"item\":\"minecraft:diamond\",\"count\":64,\"note\":\"secret\"}");
			JsonObject activity = skills.activity();
			assertEquals(Set.of("skill", "pos"), activity.keySet());
			assertEquals(skill, activity.get("skill").getAsString());
			activity.remove("pos");
			assertFalse(skills.activity().get("pos").isJsonNull());
		}
		assertEquals(json("{\"skill\":\"mine\",\"pos\":[-8,-61,0]}"),
			restored("mine", "{\"x\":-8,\"y\":-61,\"z\":0}").activity());
	}

	/** Movement destinations, speech text, recipients and construction plans are not observable arguments. */
	@Test
	void activityNeverExposesInvisibleIntentAndIdleIsNull() {
		for (String skill : new String[]{"move_to", "give", "craft", "wait", "collect", "ferry"}) {
			assertEquals(json("{\"skill\":\"" + skill + "\"}"),
				restored(skill, "{\"x\":100,\"y\":0,\"z\":100,\"text\":\"secret\",\"target\":\"mayor\"}").activity());
		}
		assertEquals(json("{\"skill\":\"build_section\"}"), restored("place_section", "{}").activity());
		assertNull(restored("speak", "{\"text\":\"secret\"}").activity());
		assertNull(restored("ask", "{\"target\":\"npc:1\"}").activity());
		Skills idle = new Skills(null);
		idle.restore("{\"last\":{\"skill\":\"mine\",\"state\":\"succeeded\"}}");
		assertNull(idle.activity());
	}

	/** Restores just enough active state to inspect the public projection without accessing the world. */
	private static Skills restored(String skill, String args) {
		Skills skills = new Skills(null);
		skills.restore("{\"active\":{\"skill\":\"" + skill + "\",\"args\":" + args + "}}");
		return skills;
	}

	/** Parses a compact JSON fixture. */
	private static JsonObject json(String source) {
		return JsonParser.parseString(source).getAsJsonObject();
	}
}
