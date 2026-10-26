package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.UUID;
import net.minecraft.util.RandomSource;
import org.junit.jupiter.api.Test;

/** Pins deterministic command draws and the input-only identity, offset and relay functions. */
final class DeterminismTest {
	/** Equal base seeds, game times, dimensions and command indices reproduce every draw. */
	@Test
	void sameCommandInputsProduceTheSameDraws() {
		long seed = Determinism.commandSeed(1234L, 24000L, 42L, 0L);
		assertEquals(-9120340891209288651L, seed);
		long[] expected = draws(seed);
		draws(Determinism.commandSeed(999L, 1L, 2L, 3L));
		assertArrayEquals(expected, draws(Determinism.commandSeed(1234L, 24000L, 42L, 0L)));
	}

	/** A different command index changes the command seed and resulting draw sequence. */
	@Test
	void differentCommandIndicesProduceDifferentDraws() {
		long first = Determinism.commandSeed(1234L, 24000L, 42L, 0L);
		long second = Determinism.commandSeed(1234L, 24000L, 42L, 1L);
		assertEquals(-4620368794762148779L, second);
		assertFalse(Arrays.equals(draws(first), draws(second)));
		assertNotEquals(first, Determinism.commandSeed(1235L, 24000L, 42L, 0L));
		assertNotEquals(first, Determinism.commandSeed(1234L, 24001L, 42L, 0L));
		assertNotEquals(first, Determinism.commandSeed(1234L, 24000L, 43L, 0L));
	}

	/** Entity IDs are stable positive integers determined only by their UUIDs. */
	@Test
	void entityIdsAreStablePositiveFunctionsOfTheirUuids() {
		UUID identity = UUID.fromString("12345678-1234-5678-9abc-def012345678");
		assertEquals(941870485, Determinism.idFor(identity));
		Determinism.idFor(new UUID(0L, 0L));
		assertEquals(941870485, Determinism.idFor(identity));
		for (long index = -10; index <= 10; index++) {
			int actual = Determinism.idFor(new UUID(index, ~index));
			assertTrue(actual > 0);
			assertEquals(actual, Determinism.idFor(new UUID(index, ~index)));
		}
	}

	/** NPC UUIDs preserve the fixed seed-and-identifier namespace without shared state. */
	@Test
	void npcUuidsDependOnlyOnSeedAndIdentifier() {
		UUID expected = UUID.fromString("f2540d75-c9ca-8641-0b78-c7cb95125495");
		assertEquals(expected, Determinism.npcUuid(1234L, "baker"));
		assertEquals(UUID.fromString("063e9503-e835-22a0-8651-4b428284786b"), Determinism.npcUuid(1235L, "baker"));
		assertEquals(UUID.fromString("cc29f9c2-cda9-e5f5-0697-74855b195d75"), Determinism.npcUuid(1234L, "farmer"));
		assertEquals(expected, Determinism.npcUuid(1234L, "baker"));
	}

	/** NPC offsets repeat exactly and remain within every positive exclusive bound. */
	@Test
	void npcOffsetsAreStableAndWithinBounds() {
		assertEquals(10, Determinism.npcOffset(1234L, "baker", 7L, 40));
		for (long seed : new long[]{Long.MIN_VALUE, -1L, 0L, 1234L, Long.MAX_VALUE}) {
			for (long day : new long[]{-1L, 0L, 7L, Long.MAX_VALUE}) {
				for (int bound : new int[]{1, 2, 40, Integer.MAX_VALUE}) {
					int actual = Determinism.npcOffset(seed, "baker", day, bound);
					assertTrue(actual >= 0 && actual < bound);
					assertEquals(actual, Determinism.npcOffset(seed, "baker", day, bound));
				}
			}
		}
	}

	/** Relay draws repeat exactly and incorporate every encounter-key input. */
	@Test
	void relayDrawsDependOnlyOnAllEncounterInputs() {
		assertEquals(0.3638845897971864, Determinism.relayDraw(1234L, "npc:baker", "alice", "news:1", 7L), 0.0);
		assertEquals(0.7105225943982696, Determinism.relayDraw(1235L, "npc:baker", "alice", "news:1", 7L), 0.0);
		assertEquals(0.06858207072673284, Determinism.relayDraw(1234L, "npc:farmer", "alice", "news:1", 7L), 0.0);
		assertEquals(0.6804296144702733, Determinism.relayDraw(1234L, "npc:baker", "bob", "news:1", 7L), 0.0);
		assertEquals(0.881780007537939, Determinism.relayDraw(1234L, "npc:baker", "alice", "news:2", 7L), 0.0);
		assertEquals(0.26625206241996147, Determinism.relayDraw(1234L, "npc:baker", "alice", "news:1", 8L), 0.0);
		assertEquals(0.3638845897971864, Determinism.relayDraw(1234L, "npc:baker", "alice", "news:1", 7L), 0.0);
	}

	private static long[] draws(long seed) {
		RandomSource random = RandomSource.create(seed);
		long[] values = new long[8];
		for (int index = 0; index < values.length; index++) {
			values[index] = random.nextLong();
		}
		return values;
	}
}
