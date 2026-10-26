package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import net.minecraft.core.BlockPos;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins local emission, exact distance/time boundaries and transient deterministic caching. */
final class NightSightTest {
	/** Initializes vanilla item/block defaults without a server. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** A dark target beyond eight blocks disappears; nearby and daylight targets do not. */
	@Test
	void darknessOnlyLimitsDistantNightTargets() {
		TestView view = new TestView();
		NightSight sight = new NightSight(view);
		assertFalse(sight.visible(Vec3.ZERO, new Vec3(12, 0, 0), false));
		assertTrue(sight.visible(Vec3.ZERO, new Vec3(8, 0, 0), false));
		assertFalse(sight.visible(Vec3.ZERO, new Vec3(8.001, 0, 0), false));
		view.dayTime = 1_000;
		assertTrue(sight.visible(Vec3.ZERO, new Vec3(12, 0, 0), false));
	}

	/** Distance to the emitting block's center is Euclidean, inclusive at five blocks. */
	@Test
	void nearbyEmittingBlockLightsTheTargetNotTheObserver() {
		TestView view = new TestView();
		view.lights.add(new BlockPos(17, 0, 0));
		NightSight sight = new NightSight(view);
		assertTrue(sight.visible(Vec3.ZERO, new Vec3(12.5, 0.5, 0.5), false));
		assertFalse(sight.visible(Vec3.ZERO, new Vec3(12.49, 0.5, 0.5), false));
		assertFalse(sight.visible(Vec3.ZERO, new Vec3(12.5, 4, 0.5), false));
		view.lights.clear();
		view.lights.add(BlockPos.ZERO);
		sight.invalidate();
		assertFalse(sight.visible(Vec3.ZERO, new Vec3(12.5, 0.5, 0.5), false));
	}

	/** Held lanterns and torches light their holder, but ordinary items and unlit ovens do not. */
	@Test
	void mainHandLightUsesTheItemsBlockEmission() {
		NightSight sight = new NightSight(new TestView());
		for (var item : List.of(Items.LANTERN, Items.TORCH, Items.GLOWSTONE)) {
			assertTrue(NightSight.heldLight(new ItemStack(item)));
			assertTrue(sight.visible(Vec3.ZERO, new Vec3(12, 0, 0), NightSight.heldLight(new ItemStack(item))));
		}
		assertFalse(NightSight.heldLight(new ItemStack(Items.SMOKER)));
		assertFalse(NightSight.heldLight(new ItemStack(Items.BREAD)));
		assertFalse(NightSight.heldLight(ItemStack.EMPTY));
	}

	/** Day-time phases, not elapsed ticks, define the half-open 19:00–05:00 interval. */
	@Test
	void nightBoundariesRepeatEveryDay() {
		assertFalse(NightSight.isNight(12_999));
		assertTrue(NightSight.isNight(13_000));
		assertTrue(NightSight.isNight(22_999));
		assertFalse(NightSight.isNight(23_000));
		assertTrue(NightSight.isNight(37_000));
		assertFalse(NightSight.isNight(47_000));
	}

	/** Per-section results are shared for a tick and discarded on a new tick or a block edit. */
	@Test
	void transientCacheRefreshesForTicksAndSameTickEdits() {
		TestView view = new TestView();
		NightSight sight = new NightSight(view);
		Vec3 target = new Vec3(20.5, 8.5, 8.5);
		assertFalse(sight.visible(Vec3.ZERO, target, false));
		int calls = view.calls;
		assertFalse(sight.visible(Vec3.ZERO, target, false));
		assertEquals(calls, view.calls);
		view.lights.add(new BlockPos(20, 8, 8));
		sight.invalidate();
		assertTrue(sight.visible(Vec3.ZERO, target, false));
		view.lights.clear();
		view.tick++;
		assertFalse(sight.visible(Vec3.ZERO, target, false));
	}

	/** A small loaded-section stand-in avoids servers and the asynchronous light engine. */
	private static final class TestView implements NightSight.View {
		long tick;
		long dayTime = 13_000;
		int calls;
		final List<BlockPos> lights = new ArrayList<>();

		/** Returns the test-controlled elapsed tick. */
		public long tick() { return tick; }
		/** Returns the test-controlled day/night clock. */
		public long dayTime() { return dayTime; }
		/** Counts section reads and supplies only matching emissive block centers. */
		public List<BlockPos> lights(int chunkX, int sectionY, int chunkZ) {
			calls++;
			return lights.stream().filter(position -> Math.floorDiv(position.getX(), 16) == chunkX
				&& Math.floorDiv(position.getY(), 16) == sectionY && Math.floorDiv(position.getZ(), 16) == chunkZ).toList();
		}
	}
}
