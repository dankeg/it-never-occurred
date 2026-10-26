package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import net.minecraft.core.component.DataComponents;
import net.minecraft.network.chat.Component;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins inventory counting, staged insertion limits and atomic simulated or committed transfers. */
final class SkillInventoryTest {
	/** Initializes registries once for real item stacks and their component limits. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Counting sums matching items across all slots while ignoring other items and empty slots. */
	@Test
	void countSumsMatchingItemsAcrossAllSlots() {
		SimpleContainer container = new SimpleContainer(new ItemStack(Items.STONE, 3), ItemStack.EMPTY,
			new ItemStack(Items.DIRT, 7), new ItemStack(Items.STONE, 4));
		assertEquals(7, SkillInventory.count(container, Items.STONE));
		assertEquals(7, SkillInventory.count(container, Items.DIRT));
		assertEquals(0, SkillInventory.count(container, Items.STICK));
	}

	/** Insertion fills compatible slots in ascending order without changing the container or offered stack. */
	@Test
	void insertStagesStacksInSlotOrderWithoutMutatingItsInputs() {
		ItemStack original = new ItemStack(Items.STONE, 63);
		SimpleContainer container = new SimpleContainer(original, ItemStack.EMPTY, new ItemStack(Items.DIRT, 64));
		List<ItemStack> slots = SkillInventory.copy(container);
		ItemStack offered = new ItemStack(Items.STONE, 3);
		assertTrue(SkillInventory.insert(container, slots, offered));
		assertTrue(slots.get(0).is(Items.STONE));
		assertEquals(64, slots.get(0).getCount());
		assertTrue(slots.get(1).is(Items.STONE));
		assertEquals(2, slots.get(1).getCount());
		assertTrue(slots.get(2).is(Items.DIRT));
		assertEquals(64, slots.get(2).getCount());
		assertSame(original, container.getItem(0));
		assertEquals(63, original.getCount());
		assertTrue(container.getItem(1).isEmpty());
		assertEquals(3, offered.getCount());
	}

	/** Item stack limits split sixteen-stack items and leave a failed insertion confined to staging. */
	@Test
	void insertRespectsItemStackLimitsAndReportsInsufficientCapacity() {
		SimpleContainer container = new SimpleContainer(2);
		List<ItemStack> slots = SkillInventory.copy(container);
		assertTrue(SkillInventory.insert(container, slots, new ItemStack(Items.SNOWBALL, 17)));
		assertTrue(slots.get(0).is(Items.SNOWBALL));
		assertEquals(16, slots.get(0).getCount());
		assertTrue(slots.get(1).is(Items.SNOWBALL));
		assertEquals(1, slots.get(1).getCount());
		SimpleContainer full = new SimpleContainer(new ItemStack(Items.SNOWBALL, 15));
		List<ItemStack> staged = SkillInventory.copy(full);
		assertFalse(SkillInventory.insert(full, staged, new ItemStack(Items.SNOWBALL, 2)));
		assertEquals(16, staged.getFirst().getCount());
		assertEquals(15, full.getItem(0).getCount());
		assertTrue(container.isEmpty());
	}

	/** Container-specific limits cap insertion even when the item itself can stack higher. */
	@Test
	void insertRespectsContainerStackLimits() {
		SimpleContainer container = new SimpleContainer(2) {
			@Override
			public int getMaxStackSize() {
				return 3;
			}
		};
		List<ItemStack> slots = SkillInventory.copy(container);
		assertTrue(SkillInventory.insert(container, slots, new ItemStack(Items.STONE, 6)));
		assertEquals(3, slots.get(0).getCount());
		assertEquals(3, slots.get(1).getCount());
		assertFalse(SkillInventory.insert(container, slots, new ItemStack(Items.STONE, 1)));
		assertEquals(3, slots.get(0).getCount());
		assertEquals(3, slots.get(1).getCount());
		assertTrue(container.isEmpty());
	}

	/** Equal item types with different components remain separate during insertion. */
	@Test
	void insertDoesNotMergeStacksWithDifferentComponents() {
		ItemStack original = new ItemStack(Items.STONE, 63);
		original.set(DataComponents.CUSTOM_NAME, Component.literal("first"));
		ItemStack offered = new ItemStack(Items.STONE, 1);
		offered.set(DataComponents.CUSTOM_NAME, Component.literal("second"));
		SimpleContainer container = new SimpleContainer(original, ItemStack.EMPTY);
		List<ItemStack> slots = SkillInventory.copy(container);
		assertTrue(SkillInventory.insert(container, slots, offered));
		assertEquals(63, slots.get(0).getCount());
		assertEquals(Component.literal("first"), slots.get(0).get(DataComponents.CUSTOM_NAME));
		assertEquals(1, slots.get(1).getCount());
		assertEquals(Component.literal("second"), slots.get(1).get(DataComponents.CUSTOM_NAME));
	}

	/** A successful noncommitting transfer changes neither container nor any existing stack. */
	@Test
	void transferWithCommitFalseChangesNothing() {
		ItemStack first = new ItemStack(Items.STONE, 4);
		ItemStack second = new ItemStack(Items.STONE, 5);
		ItemStack destination = new ItemStack(Items.STONE, 63);
		SimpleContainer from = new SimpleContainer(first, second);
		SimpleContainer to = new SimpleContainer(destination, ItemStack.EMPTY);
		assertNull(SkillInventory.transfer(from, to, Items.STONE, 7, false));
		assertSame(first, from.getItem(0));
		assertSame(second, from.getItem(1));
		assertSame(destination, to.getItem(0));
		assertEquals(4, first.getCount());
		assertEquals(5, second.getCount());
		assertEquals(63, destination.getCount());
		assertTrue(to.getItem(1).isEmpty());
	}

	/** A committed transfer consumes source slots and fills destination slots in ascending order. */
	@Test
	void committedTransferConsumesAndFillsSlotsInOrder() {
		SimpleContainer from = new SimpleContainer(new ItemStack(Items.STONE, 4), new ItemStack(Items.STONE, 5));
		SimpleContainer to = new SimpleContainer(new ItemStack(Items.STONE, 63), ItemStack.EMPTY);
		assertNull(SkillInventory.transfer(from, to, Items.STONE, 7, true));
		assertTrue(from.getItem(0).isEmpty());
		assertTrue(from.getItem(1).is(Items.STONE));
		assertEquals(2, from.getItem(1).getCount());
		assertTrue(to.getItem(0).is(Items.STONE));
		assertEquals(64, to.getItem(0).getCount());
		assertTrue(to.getItem(1).is(Items.STONE));
		assertEquals(6, to.getItem(1).getCount());
	}

	/** Exactly enough source items transfer successfully without requiring an extra item. */
	@Test
	void transferAcceptsExactlyEnoughItems() {
		SimpleContainer from = new SimpleContainer(new ItemStack(Items.STONE, 5));
		SimpleContainer to = new SimpleContainer(1);
		assertNull(SkillInventory.transfer(from, to, Items.STONE, 5, true));
		assertTrue(from.isEmpty());
		assertTrue(to.getItem(0).is(Items.STONE));
		assertEquals(5, to.getItem(0).getCount());
	}

	/** A source one item short returns missing_items and leaves both inventories unchanged in either mode. */
	@Test
	void transferRejectsOneItemShortWithoutPartialMutation() {
		for (boolean commit : new boolean[]{false, true}) {
			SimpleContainer from = new SimpleContainer(new ItemStack(Items.STONE, 5));
			SimpleContainer to = new SimpleContainer(new ItemStack(Items.STONE, 2));
			assertEquals("missing_items", SkillInventory.transfer(from, to, Items.STONE, 6, commit));
			assertEquals(5, from.getItem(0).getCount());
			assertEquals(2, to.getItem(0).getCount());
		}
	}

	/** Insufficient destination capacity returns inventory_full without partially committing staged items. */
	@Test
	void transferRejectsFullInventoryWithoutPartialMutation() {
		SimpleContainer from = new SimpleContainer(new ItemStack(Items.STONE, 2));
		SimpleContainer to = new SimpleContainer(new ItemStack(Items.STONE, 63));
		assertEquals("inventory_full", SkillInventory.transfer(from, to, Items.STONE, 2, true));
		assertEquals(2, from.getItem(0).getCount());
		assertEquals(63, to.getItem(0).getCount());
	}
}
