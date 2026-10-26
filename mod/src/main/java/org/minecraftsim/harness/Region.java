package org.minecraftsim.harness;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/**
 * Inclusive integer block bounds used by canonical hashing; consumes no RNG.
 * @param x0 minimum x coordinate
 * @param z0 minimum z coordinate
 * @param x1 maximum x coordinate
 * @param z1 maximum z coordinate
 * @param y0 minimum y coordinate
 * @param y1 maximum y coordinate
 */
public record Region(int x0, int z0, int x1, int z1, int y0, int y1) {
	/**
	 * Rejects reversed bounds without reordering coordinates.
	 * @param x0 minimum x coordinate
	 * @param z0 minimum z coordinate
	 * @param x1 maximum x coordinate
	 * @param z1 maximum z coordinate
	 * @param y0 minimum y coordinate
	 * @param y1 maximum y coordinate
	 * @throws IllegalArgumentException if any minimum exceeds its maximum
	 */
	public Region {
		if (x0 > x1 || y0 > y1 || z0 > z1) {
			throw new IllegalArgumentException("Region minimum bounds must not exceed maximum bounds");
		}
	}

	/**
	 * Reads six required exact 32-bit integer bounds in protocol order.
	 * @param json region object
	 * @return validated bounds
	 * @throws IllegalArgumentException if the object or any bounds are invalid
	 */
	public static Region fromJson(JsonObject json) {
		if (json == null) {
			throw new IllegalArgumentException("region is required");
		}
		return new Region(coordinate(json, "x0"), coordinate(json, "z0"), coordinate(json, "x1"),
			coordinate(json, "z1"), coordinate(json, "y0"), coordinate(json, "y1"));
	}

	/**
	 * Tests a point against the block volume, excluding each maximum plus one.
	 * @param x point x coordinate
	 * @param y point y coordinate
	 * @param z point z coordinate
	 * @return whether the point lies within all three half-open coordinate intervals
	 */
	public boolean contains(double x, double y, double z) {
		return x >= x0 && x < (double) x1 + 1.0
			&& y >= y0 && y < (double) y1 + 1.0
			&& z >= z0 && z < (double) z1 + 1.0;
	}

	private static int coordinate(JsonObject json, String key) {
		JsonElement value = json.get(key);
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("region." + key + " must be an integer");
		}
		try {
			return value.getAsBigDecimal().intValueExact();
		} catch (ArithmeticException | NumberFormatException exception) {
			throw new IllegalArgumentException("region." + key + " must be a 32-bit integer", exception);
		}
	}
}
