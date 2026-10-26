package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import org.junit.jupiter.api.Test;

/** Pins independent locks and ownership and the provenance metadata on successful owned-container takes. */
final class ContainerOwnershipTest {
	/** Owned shops permit only their explicit owner to stock and collect takings. */
	@Test
	void stallStockTransfersRequireTheNamedOwner() {
		assertTrue(Documents.ownsStall(new JsonPrimitive("baker"), "baker"));
		assertFalse(Documents.ownsStall(new JsonPrimitive("baker"), "visitor"));
		assertFalse(Documents.ownsStall(null, "baker"));
		assertFalse(Documents.ownsStall(JsonNull.INSTANCE, "baker"));
	}
	/** A null lock permits nonowners, while an allowed list still enforces explicit membership. */
	@Test
	void ownershipDoesNotLockAnOtherwiseOpenContainer() {
		assertTrue(Documents.permitted(Documents.allowed(JsonNull.INSTANCE), "visitor"));
		assertTrue(Documents.permitted(null, "visitor"));
		assertTrue(Documents.permitted(new JsonPrimitive("*"), "visitor"));
		JsonElement allowed = Documents.allowed(JsonParser.parseString("[\"owner\",\"owner\",\"alice\"]"));
		assertEquals(JsonParser.parseString("[\"alice\",\"owner\"]"), allowed);
		assertTrue(Documents.permitted(allowed, "owner"));
		assertFalse(Documents.permitted(allowed, "visitor"));
		assertFalse(Documents.permitted(JsonParser.parseString("[]"), "owner"));
	}

	/** Taking from another person's container remains a transfer, explicitly identifying the owner and taker. */
	@Test
	void nonownerTakingIsLoggedWithoutDenyingTheTransfer() {
		JsonObject event = JsonParser.parseString("{\"type\":\"transfer\",\"from\":\"container:1,2,3\",\"to\":\"alice\"}")
			.getAsJsonObject();
		Documents.taken(event, new JsonPrimitive("builder"), "alice");
		assertEquals("transfer", event.get("type").getAsString());
		assertEquals("builder", event.get("owner").getAsString());
		assertTrue(event.get("taken_by_non_owner").getAsBoolean());
		assertEquals("alice", event.get("to").getAsString());
		Documents.taken(event, new JsonPrimitive("builder"), "builder");
		assertFalse(event.get("taken_by_non_owner").getAsBoolean());
	}

	/** Unowned takes keep their existing event shape, including when ownership was explicitly cleared. */
	@Test
	void unownedTransfersRemainUnchanged() {
		JsonObject event = JsonParser.parseString("{\"type\":\"transfer\",\"count\":2}").getAsJsonObject();
		JsonObject before = event.deepCopy();
		Documents.taken(event, null, "alice");
		Documents.taken(event, JsonNull.INSTANCE, "alice");
		assertEquals(before, event);
	}

	/** Owners and lock entries reject malformed names instead of silently changing policy. */
	@Test
	void policyValidationPreservesTheSourceAndRejectsMalformedValues() {
		assertNull(Documents.owner(null));
		assertNull(Documents.owner(JsonNull.INSTANCE));
		assertEquals("builder", Documents.owner(new JsonPrimitive("builder")));
		JsonElement source = JsonParser.parseString("[\"builder\",\"alice\",\"builder\"]");
		JsonElement before = source.deepCopy();
		Documents.allowed(source);
		assertEquals(before, source);
		for (String invalid : new String[]{"1", "true", "{}", "[]", "\"\"", "\"invalid name\""}) {
			assertThrows(IllegalArgumentException.class, () -> Documents.owner(JsonParser.parseString(invalid)));
		}
		for (String invalid : new String[]{"1", "true", "{}", "\"alice\"", "[null]", "[\"invalid name\"]"}) {
			assertThrows(IllegalArgumentException.class, () -> Documents.allowed(JsonParser.parseString(invalid)));
		}
		assertThrows(IllegalArgumentException.class, () -> Documents.allowed(null));
	}
}
