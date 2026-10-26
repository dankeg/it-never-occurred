package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins opt-in scheduling and legacy configuration bytes without constructing a server or using wall time. */
final class DailyHungerTest {
	/** Initializes the registry-derived configuration defaults once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Missing and explicitly disabled flags retain both the food level and the original configuration bytes. */
	@Test
	void disabledRuleLeavesLegacyStateUnchanged() {
		SimConfig config = new SimConfig();
		String before = HarnessState.canonical(config.json());
		assertFalse(config.dailyHunger());
		for (long time : new long[]{0, 5_999, 6_000, 12_000, 24_000, Long.MAX_VALUE}) {
			assertEquals(20, DailyHunger.foodAfter(time, 20, config));
		}
		config.update(json("{\"daily_hunger\":false,\"eat_ticks\":32,\"hunger_interval_ticks\":6000,\"hunger_points\":2}"));
		assertFalse(config.dailyHunger());
		assertEquals(before, HarnessState.canonical(config.json()));
		config.update(json("{\"daily_hunger\":true}"));
		assertTrue(config.dailyHunger());
		config.update(json("{\"daily_hunger\":false}"));
		assertEquals(before, HarnessState.canonical(config.json()));
	}

	/** A day costs eight points at four absolute game-time boundaries, never at startup or neighboring ticks. */
	@Test
	void defaultScheduleUsesOnlyExactGameTimeBoundaries() {
		SimConfig config = enabled();
		int food = 20;
		for (long time = 0; time <= 24_000; time++) {
			food = DailyHunger.foodAfter(time, food, config);
			assertEquals(20 - 2 * (int) (time / 6_000), food, "game time " + time);
		}
	}

	/** Depletion clamps at zero regardless of the cost and never introduces starvation damage or negative food. */
	@Test
	void emptyFoodRemainsAtZero() {
		SimConfig config = enabled();
		assertEquals(0, DailyHunger.foodAfter(6_000, 1, config));
		assertEquals(0, DailyHunger.foodAfter(12_000, 0, config));
		assertEquals(1, DailyHunger.foodAfter(6_001, 1, config));
	}

	/** Schedule decisions need no transient cursor, so restored configuration yields the same next boundary. */
	@Test
	void customScheduleSurvivesConfigurationRestore() {
		SimConfig config = enabled();
		config.update(json("{\"hunger_interval_ticks\":100,\"hunger_points\":3,\"eat_ticks\":8}"));
		SimConfig restored = new SimConfig();
		restored.update(config.json());
		assertEquals(8, restored.integer("eat_ticks"));
		assertEquals(20, DailyHunger.foodAfter(99, 20, restored));
		assertEquals(17, DailyHunger.foodAfter(100, 20, restored));
		assertEquals(17, DailyHunger.foodAfter(101, 17, restored));
		assertEquals(14, DailyHunger.foodAfter(200, 17, restored));
	}

	/** Boolean flags and whole nonnegative costs validate atomically, while interval zero is invalid. */
	@Test
	void invalidSettingsDoNotPartiallyEnableHunger() {
		SimConfig config = new SimConfig();
		JsonObject before = config.json();
		for (String invalid : new String[]{"{\"daily_hunger\":1}", "{\"daily_hunger\":\"true\"}",
				"{\"daily_hunger\":true,\"hunger_interval_ticks\":0}", "{\"hunger_points\":-1}", "{\"eat_ticks\":-1}"}) {
			assertThrows(IllegalArgumentException.class, () -> config.update(json(invalid)));
			assertEquals(before, config.json());
		}
		assertThrows(ArithmeticException.class, () -> config.update(json("{\"hunger_points\":1.5}")));
		assertThrows(ArithmeticException.class, () -> config.update(json("{\"eat_ticks\":1.5}")));
		assertEquals(before, config.json());
		assertEquals(32, config.integer("eat_ticks"));
		assertEquals(32.0, config.number("eat_ticks"));
		assertEquals(6_000.0, config.number("hunger_interval_ticks"));
		config.update(json("{\"eat_ticks\":0,\"hunger_points\":0}"));
		assertEquals(0, config.integer("eat_ticks"));
	}

	/** Creates the smallest enabled configuration without changing the established defaults. */
	private static SimConfig enabled() {
		SimConfig config = new SimConfig();
		config.update(json("{\"daily_hunger\":true}"));
		return config;
	}

	/** Parses a small configuration fixture without retaining mutable shared JSON. */
	private static JsonObject json(String text) {
		return JsonParser.parseString(text).getAsJsonObject();
	}
}
