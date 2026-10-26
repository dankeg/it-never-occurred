package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Pins automatic buoyancy, explicit controls and the effective input's snapshot encoding. */
final class BotInputTest {
	/** Submerged eyes add jump without changing movement, posture or view. */
	@Test
	void submergedEyesHoldJumpAndPreserveOtherControls() {
		BotInput requested = new BotInput(0.5f, -0.25f, false, true, true, 120, -30);
		assertEquals(new BotInput(0.5f, -0.25f, true, true, true, 120, -30), requested.afloat(true, false));
		assertTrue(BotInput.idle(0, 0).afloat(true, false).jump());
	}

	/** The safety rule is a no-op for ordinary dry controls, including explicit jump. */
	@Test
	void dryInputsAndExplicitJumpStayUnchanged() {
		BotInput idle = BotInput.idle(80, 20);
		BotInput jumping = new BotInput(1, 0, true, false, false, 80, 20);
		assertSame(idle, idle.afloat(false, false));
		assertSame(jumping, jumping.afloat(false, false));
		assertSame(jumping, jumping.afloat(true, false));
	}

	/** Automatic jump continues while submerged but cannot latch on after leaving water. */
	@Test
	void surfacingRemovesOnlyAutomaticJump() {
		BotInput idle = BotInput.idle(45, -10);
		BotInput floating = idle.afloat(true, false);
		assertSame(floating, floating.afloat(true, true));
		assertEquals(idle, floating.afloat(false, true));
		assertFalse(floating.afloat(false, true).jump());
		assertTrue(floating.afloat(false, false).jump());
	}

	/** Saving the effective jump and restoring its automatic origin preserves subsequent controls. */
	@Test
	void savedFloatingInputContinuesAndClearsDeterministically() {
		BotInput floating = new BotInput(1, 0, false, false, false, -90, 0).afloat(true, false);
		BotInput restored = BotInput.load(floating.save());
		assertEquals(floating, restored);
		assertEquals(floating.afloat(true, true), restored.afloat(true, true));
		assertEquals(floating.afloat(false, true), restored.afloat(false, true));
	}
}
