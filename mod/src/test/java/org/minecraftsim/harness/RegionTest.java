package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

/** Pins inclusive block bounds, exclusive point boundaries and exact region JSON validation. */
final class RegionTest {
	/** Minimum and maximum blocks are included while each maximum-plus-one point is excluded. */
	@Test
	void containmentIncludesBlockBoundsButExcludesMaximumPlusOne() {
		Region region = new Region(-2, -3, 4, 5, -1, 6);
		assertTrue(region.contains(-2, -1, -3));
		assertTrue(region.contains(4, 6, 5));
		assertTrue(region.contains(Math.nextDown(5.0), Math.nextDown(7.0), Math.nextDown(6.0)));
		assertFalse(region.contains(Math.nextDown(-2.0), 0, 0));
		assertFalse(region.contains(0, Math.nextDown(-1.0), 0));
		assertFalse(region.contains(0, 0, Math.nextDown(-3.0)));
		assertFalse(region.contains(5, 0, 0));
		assertFalse(region.contains(0, 7, 0));
		assertFalse(region.contains(0, 0, 6));
	}

	/** Maximum-plus-one containment remains correct at the largest integer block coordinate. */
	@Test
	void maximumIntegerBoundsDoNotOverflowContainment() {
		Region region = new Region(Integer.MAX_VALUE, 0, Integer.MAX_VALUE, 0, 0, 0);
		assertTrue(region.contains(Integer.MAX_VALUE + 0.5, 0.5, 0.5));
		assertFalse(region.contains((double) Integer.MAX_VALUE + 1.0, 0.5, 0.5));
	}

	/** Region JSON reads all six exact integer coordinates in protocol order. */
	@Test
	void fromJsonPreservesExactCoordinatesInProtocolOrder() {
		JsonObject json = bounds();
		json.addProperty("x0", -2.0);
		assertEquals(new Region(-2, -3, 4, 5, -1, 6), Region.fromJson(json));
	}

	/** Missing or nonnumeric coordinates are rejected with their precise field error. */
	@Test
	void fromJsonRejectsMissingAndNonnumericCoordinates() {
		assertEquals("region is required", assertThrows(IllegalArgumentException.class, () -> Region.fromJson(null)).getMessage());
		for (String key : new String[]{"x0", "z0", "x1", "z1", "y0", "y1"}) {
			JsonObject missing = bounds();
			missing.remove(key);
			assertEquals("region." + key + " must be an integer",
				assertThrows(IllegalArgumentException.class, () -> Region.fromJson(missing)).getMessage());
			for (String invalid : new String[]{"null", "\"1\"", "true", "[]", "{}"}) {
				JsonObject json = bounds();
				json.add(key, JsonParser.parseString(invalid));
				assertEquals("region." + key + " must be an integer",
					assertThrows(IllegalArgumentException.class, () -> Region.fromJson(json)).getMessage());
			}
		}
	}

	/** Fractional and out-of-range coordinates are rejected rather than rounded or truncated. */
	@Test
	void fromJsonRejectsFractionsAndIntegerOverflow() {
		for (String key : new String[]{"x0", "z0", "x1", "z1", "y0", "y1"}) {
			for (String invalid : new String[]{"1.5", "2147483648", "-2147483649"}) {
				JsonObject json = bounds();
				json.add(key, JsonParser.parseString(invalid));
				assertEquals("region." + key + " must be a 32-bit integer",
					assertThrows(IllegalArgumentException.class, () -> Region.fromJson(json)).getMessage());
			}
		}
	}

	/** Reversed bounds on any axis are rejected without silently reordering coordinates. */
	@Test
	void fromJsonRejectsReversedBoundsOnEveryAxis() {
		for (String key : new String[]{"x0", "z0", "y0"}) {
			JsonObject json = bounds();
			json.addProperty(key, 100);
			assertEquals("Region minimum bounds must not exceed maximum bounds",
				assertThrows(IllegalArgumentException.class, () -> Region.fromJson(json)).getMessage());
		}
		assertEquals(new Region(0, 0, 0, 0, 0, 0),
			Region.fromJson(JsonParser.parseString("{\"x0\":0,\"z0\":0,\"x1\":0,\"z1\":0,\"y0\":0,\"y1\":0}").getAsJsonObject()));
	}

	private static JsonObject bounds() {
		return JsonParser.parseString("{\"x0\":-2,\"z0\":-3,\"x1\":4,\"z1\":5,\"y0\":-1,\"y1\":6}").getAsJsonObject();
	}
}
