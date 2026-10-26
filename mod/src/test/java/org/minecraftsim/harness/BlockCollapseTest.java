package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Pins the pure one-shot decision without constructing a Minecraft server or a mock world. */
final class BlockCollapseTest {
	/** The last real support loss triggers only when no solid support remains. */
	@Test
	void lastSupportLossTriggersCollapse() {
		assertTrue(BlockCollapse.shouldCollapse(false, true, false));
		assertFalse(BlockCollapse.shouldCollapse(false, true, true));
	}

	/** Initially absent supports and repeat ticks cannot invent or repeat an incident. */
	@Test
	void missingTransitionAndPersistedCollapsedFlagPreventCollapse() {
		assertFalse(BlockCollapse.shouldCollapse(false, false, false));
		assertFalse(BlockCollapse.shouldCollapse(true, true, false));
		assertFalse(BlockCollapse.shouldCollapse(true, false, false));
		assertFalse(BlockCollapse.shouldCollapse(true, true, true));
	}
}
