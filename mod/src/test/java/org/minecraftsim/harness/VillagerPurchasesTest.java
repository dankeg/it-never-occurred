package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins deterministic fresh-stock selection, exact-count atomicity and per-unit stall takings without a server. */
final class VillagerPurchasesTest {
	/** Initializes real item registries and stack components once per test JVM. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Makes stamped bread with the same freshness API as baking and container observations. */
	private static ItemStack bread(int count, long day) {
		ItemStack stack = new ItemStack(Items.BREAD, count);
		Freshness.stamp(stack, day);
		return stack;
	}

	/** A newer later slot sells before an older first slot, spanning stacks when necessary. */
	@Test
	void freshestFirstAcrossStacksWithExactPerUnitPayment() {
		SimpleContainer shop = new SimpleContainer(bread(5, 1), bread(3, 3), bread(4, 2), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 5, 2, 3, 2));
		assertEquals(5, shop.getItem(0).getCount());
		assertEquals(10, shop.getItem(1).getCount());
		assertTrue(shop.getItem(1).is(Items.EMERALD));
		assertEquals(2, shop.getItem(2).getCount());
		assertEquals(2, Freshness.bakedDay(shop.getItem(2)));
		assertEquals(7, SkillInventory.count(shop, Items.BREAD));
	}

	/** Equal freshness falls back to ascending slots, never a hash iteration or random choice. */
	@Test
	void equalDaysUseAscendingSlots() {
		SimpleContainer shop = new SimpleContainer(bread(4, 2), bread(4, 2));
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 2, 0, 2, 2));
		assertEquals(2, shop.getItem(0).getCount());
		assertEquals(4, shop.getItem(1).getCount());
	}

	/** Bread exactly at the freshness boundary sells; a negative-day mouldy loaf stays untouched. */
	@Test
	void mouldyBreadIsSkippedButTheBoundaryIsFresh() {
		SimpleContainer shop = new SimpleContainer(bread(10, -1), bread(2, 0), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 2, 1, 2, 2));
		assertEquals(10, shop.getItem(0).getCount());
		assertEquals(-1, Freshness.bakedDay(shop.getItem(0)));
		assertEquals(2, SkillInventory.count(shop, Items.EMERALD));
	}

	/** Bread baked before day zero remains fresh until its ordinary strict age boundary. */
	@Test
	void negativeBakedDaysCanStillBeFresh() {
		SimpleContainer shop = new SimpleContainer(bread(2, -2), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 2, 1, 0, 2));
		assertEquals(0, SkillInventory.count(shop, Items.BREAD));
		assertEquals(2, SkillInventory.count(shop, Items.EMERALD));
	}

	/** An unstamped loaf has day zero, becoming unsaleable only after the boundary. */
	@Test
	void unstampedBreadUsesDayZero() {
		for (long today : new long[]{2, 3}) {
			SimpleContainer shop = new SimpleContainer(new ItemStack(Items.BREAD), ItemStack.EMPTY);
			assertEquals(today == 2 ? VillagerPurchases.Outcome.PURCHASED : VillagerPurchases.Outcome.SOLD_OUT,
				VillagerPurchases.purchase(shop, Items.BREAD, 1, 1, today, 2));
			assertEquals(today == 2 ? 1 : 0, SkillInventory.count(shop, Items.EMERALD));
		}
	}

	/** Missing fresh units cannot cause a partial sale, payment, or replacement of existing stacks. */
	@Test
	void insufficientFreshStockIsAtomic() {
		ItemStack fresh = bread(2, 3);
		ItemStack mouldy = bread(20, 0);
		ItemStack emeralds = new ItemStack(Items.EMERALD, 7);
		SimpleContainer shop = new SimpleContainer(fresh, mouldy, emeralds);
		assertEquals(VillagerPurchases.Outcome.SOLD_OUT,
			VillagerPurchases.purchase(shop, Items.BREAD, 3, 2, 3, 2));
		assertSame(fresh, shop.getItem(0));
		assertSame(mouldy, shop.getItem(1));
		assertSame(emeralds, shop.getItem(2));
		assertEquals(2, fresh.getCount());
		assertEquals(20, mouldy.getCount());
		assertEquals(7, emeralds.getCount());
	}

	/** Empty and entirely mouldy stalls report sold out without creating emeralds. */
	@Test
	void unavailableStockNeverCreatesPayment() {
		for (SimpleContainer shop : List.of(new SimpleContainer(2),
				new SimpleContainer(bread(64, -5), ItemStack.EMPTY))) {
			assertEquals(VillagerPurchases.Outcome.SOLD_OUT,
				VillagerPurchases.purchase(shop, Items.BREAD, 1, 3, 3, 2));
			assertEquals(0, SkillInventory.count(shop, Items.EMERALD));
		}
	}

	/** A partial staged emerald merge cannot escape a failed payment-capacity check. */
	@Test
	void insufficientPaymentRoomRollsBackGoodsAndEmeralds() {
		ItemStack stock = bread(2, 0);
		ItemStack takings = new ItemStack(Items.EMERALD, 63);
		SimpleContainer shop = new SimpleContainer(stock, takings);
		assertEquals(VillagerPurchases.Outcome.NO_PAYMENT_ROOM,
			VillagerPurchases.purchase(shop, Items.BREAD, 1, 2, 0, 2));
		assertSame(stock, shop.getItem(0));
		assertSame(takings, shop.getItem(1));
		assertEquals(2, stock.getCount());
		assertEquals(63, takings.getCount());
	}

	/** Removing the last goods may free the only slot needed for payment. */
	@Test
	void paymentCanUseASlotFreedByTheSale() {
		SimpleContainer shop = new SimpleContainer(bread(4, 0));
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 4, 3, 0, 2));
		assertEquals(0, SkillInventory.count(shop, Items.BREAD));
		assertEquals(12, SkillInventory.count(shop, Items.EMERALD));
	}

	/** Large payments split over legal stack sizes and merge with existing takings. */
	@Test
	void paymentMergesAndSplitsAcrossSlots() {
		SimpleContainer shop = new SimpleContainer(bread(4, 0), new ItemStack(Items.EMERALD, 60), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 4, 20, 0, 2));
		assertEquals(140, SkillInventory.count(shop, Items.EMERALD));
		for (int slot = 0; slot < shop.getContainerSize(); slot++) {
			assertTrue(shop.getItem(slot).getCount() <= 64);
		}
	}

	/** Container placement policy still applies to stall takings. */
	@Test
	void paymentRespectsContainerPlacementRules() {
		SimpleContainer shop = new SimpleContainer(bread(2, 0), ItemStack.EMPTY) {
			@Override
			public boolean canPlaceItem(int slot, ItemStack stack) {
				return !stack.is(Items.EMERALD);
			}
		};
		assertEquals(VillagerPurchases.Outcome.NO_PAYMENT_ROOM,
			VillagerPurchases.purchase(shop, Items.BREAD, 1, 1, 0, 2));
		assertEquals(2, SkillInventory.count(shop, Items.BREAD));
		assertEquals(0, SkillInventory.count(shop, Items.EMERALD));
	}

	/** Disabling freshness permits old bread even while market purchases remain enabled. */
	@Test
	void nullFreshnessLimitAllowsOldStock() {
		SimpleContainer shop = new SimpleContainer(bread(2, -50), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.BREAD, 2, 1, 100, null));
		assertEquals(2, SkillInventory.count(shop, Items.EMERALD));
	}

	/** The feature can sell configured nonbread items without accidentally applying bread spoilage. */
	@Test
	void nonbreadGoodsDoNotSpoil() {
		SimpleContainer shop = new SimpleContainer(new ItemStack(Items.WHEAT, 4), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(shop, Items.WHEAT, 3, 2, 100, 0));
		assertEquals(1, SkillInventory.count(shop, Items.WHEAT));
		assertEquals(6, SkillInventory.count(shop, Items.EMERALD));
	}

	/** A free sale needs no emerald slot; a price multiplication cannot wrap into a free sale. */
	@Test
	void zeroAndOverflowingPricesCannotProduceIncorrectPayment() {
		SimpleContainer free = new SimpleContainer(bread(4, 0));
		assertEquals(VillagerPurchases.Outcome.PURCHASED,
			VillagerPurchases.purchase(free, Items.BREAD, 1, 0, 0, 2));
		assertEquals(3, SkillInventory.count(free, Items.BREAD));
		SimpleContainer costly = new SimpleContainer(bread(4, 0), ItemStack.EMPTY);
		assertEquals(VillagerPurchases.Outcome.NO_PAYMENT_ROOM,
			VillagerPurchases.purchase(costly, Items.BREAD, 2, Integer.MAX_VALUE, 0, 2));
		assertEquals(4, SkillInventory.count(costly, Items.BREAD));
		assertEquals(0, SkillInventory.count(costly, Items.EMERALD));
	}

	/** Invalid direct helper calls fail before touching the container. */
	@Test
	void invalidQuantitiesAndPricesAreRejectedWithoutMutation() {
		SimpleContainer shop = new SimpleContainer(bread(4, 0));
		assertThrows(IllegalArgumentException.class, () -> VillagerPurchases.purchase(shop, Items.BREAD, 0, 1, 0, 2));
		assertThrows(IllegalArgumentException.class, () -> VillagerPurchases.purchase(shop, Items.BREAD, 1, -1, 0, 2));
		assertEquals(4, SkillInventory.count(shop, Items.BREAD));
	}

	/** Earlier arrivals precede lexical IDs; same-tick contention resolves by the raw villager ID. */
	@Test
	void arrivalsAreSortedByTickThenNpcId() {
		List<VillagerPurchases.Arrival> arrivals = new ArrayList<>(List.of(
			new VillagerPurchases.Arrival(11, "alice"), new VillagerPurchases.Arrival(10, "zoe"),
			new VillagerPurchases.Arrival(10, "bob")));
		arrivals.sort(null);
		assertEquals(List.of(new VillagerPurchases.Arrival(10, "bob"), new VillagerPurchases.Arrival(10, "zoe"),
			new VillagerPurchases.Arrival(11, "alice")), arrivals);
		SimpleContainer shop = new SimpleContainer(bread(1, 0), ItemStack.EMPTY);
		List<String> buyers = new ArrayList<>();
		for (VillagerPurchases.Arrival arrival : arrivals) {
			if (VillagerPurchases.purchase(shop, Items.BREAD, 1, 1, 0, 2) == VillagerPurchases.Outcome.PURCHASED) {
				buyers.add(arrival.npc());
			}
		}
		assertEquals(List.of("bob"), buyers);
		assertEquals(1, SkillInventory.count(shop, Items.EMERALD));
	}
}
