package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.core.component.DataComponents;
import net.minecraft.network.chat.Component;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Tests single-entity drop planning and atomic inventory staging without constructing a server or world. */
final class SkillDropTest {
	/** Initializes real item identities and their default components once. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Counts nonadjacent inventory slots and removes only the requested amount in ascending slot order. */
	@Test
	void stagesExactCountAcrossSlotsWithoutChangingInventory() {
		ItemStack first = new ItemStack(Items.STONE, 2);
		ItemStack dirt = new ItemStack(Items.DIRT, 7);
		ItemStack second = new ItemStack(Items.STONE, 5);
		SimpleContainer inventory = new SimpleContainer(dirt, first, ItemStack.EMPTY, second);
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, 4);
		assertNull(drop.reason());
		assertTrue(drop.stack().is(Items.STONE));
		assertEquals(4, drop.stack().getCount());
		assertEquals(4, drop.slots().size());
		assertEquals(7, drop.slots().getFirst().getCount());
		assertTrue(drop.slots().get(1).isEmpty());
		assertEquals(3, drop.slots().get(3).getCount());
		assertSame(first, inventory.getItem(1));
		assertEquals(2, first.getCount());
		assertEquals(5, second.getCount());
		assertNotSame(dirt, drop.slots().getFirst());
		assertNotSame(first, drop.stack());
		SkillInventory.commit(inventory, drop.slots());
		assertEquals(3, SkillInventory.count(inventory, Items.STONE));
		assertEquals(7, SkillInventory.count(inventory, Items.DIRT));
		assertEquals(4, drop.stack().getCount());
	}

	/** Insufficient stock returns the required reason, including very large counts, without partial removal. */
	@Test
	void missingItemsLeaveEverySlotUnchanged() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.STONE, 2), new ItemStack(Items.STONE, 3));
		for (int count : new int[]{6, Integer.MAX_VALUE}) {
			SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, count);
			assertEquals("missing_items", drop.reason());
			assertTrue(drop.stack().isEmpty());
			assertTrue(drop.slots().isEmpty());
			assertEquals(2, inventory.getItem(0).getCount());
			assertEquals(3, inventory.getItem(1).getCount());
		}
		assertEquals("missing_items", SkillDrop.plan(inventory, Items.STICK, 1).reason());
	}

	/** Damage and names survive copying, and mutating a staged drop cannot mutate its original tool. */
	@Test
	void preservesAllComponentsWithoutSharingTheSourceStack() {
		ItemStack pickaxe = named(Items.IRON_PICKAXE, 1, "Incident pickaxe");
		pickaxe.set(DataComponents.DAMAGE, 17);
		SimpleContainer inventory = new SimpleContainer(ItemStack.EMPTY, pickaxe);
		SkillDrop drop = SkillDrop.plan(inventory, Items.IRON_PICKAXE, 1);
		assertNull(drop.reason());
		assertTrue(ItemStack.isSameItemSameComponents(pickaxe, drop.stack()));
		assertEquals(pickaxe.getComponents(), drop.stack().getComponents());
		assertEquals(17, drop.stack().getDamageValue());
		drop.stack().set(DataComponents.DAMAGE, 18);
		assertEquals(17, pickaxe.getDamageValue());
		assertSame(pickaxe, inventory.getItem(1));
		assertTrue(drop.slots().get(1).isEmpty());
	}

	/** A small incompatible group does not hide a later group able to supply the complete request. */
	@Test
	void skipsIncompatibleGroupsAndCombinesOnlyIdenticalComponents() {
		ItemStack named = named(Items.STONE, 2, "Keep me");
		SimpleContainer inventory = new SimpleContainer(named, new ItemStack(Items.STONE, 2), new ItemStack(Items.STONE, 3));
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, 4);
		assertNull(drop.reason());
		assertNull(drop.stack().get(DataComponents.CUSTOM_NAME));
		assertEquals(2, drop.slots().getFirst().getCount());
		assertEquals(named.getComponents(), drop.slots().getFirst().getComponents());
		assertTrue(drop.slots().get(1).isEmpty());
		assertEquals(1, drop.slots().get(2).getCount());
	}

	/** The first sufficient component group wins deterministically, not the largest or currently held stack. */
	@Test
	void choosesFirstSufficientGroupInSlotOrder() {
		SimpleContainer inventory = new SimpleContainer(named(Items.STONE, 2, "First"), named(Items.STONE, 5, "Second"),
			named(Items.STONE, 3, "First"));
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, 4);
		assertNull(drop.reason());
		assertEquals(Component.literal("First"), drop.stack().get(DataComponents.CUSTOM_NAME));
		assertTrue(drop.slots().getFirst().isEmpty());
		assertEquals(5, drop.slots().get(1).getCount());
		assertEquals(1, drop.slots().get(2).getCount());
	}

	/** One entity cannot represent different names, damage or book contents; rejection must not erase components. */
	@Test
	void rejectsRequestsThatWouldMixDifferentComponents() {
		SimpleContainer inventory = new SimpleContainer(named(Items.STONE, 2, "First"), named(Items.STONE, 3, "Second"));
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, 4);
		assertEquals("incompatible_items", drop.reason());
		assertTrue(drop.stack().isEmpty());
		assertTrue(drop.slots().isEmpty());
		assertEquals(2, inventory.getItem(0).getCount());
		assertEquals(3, inventory.getItem(1).getCount());
	}

	/** Requests can span full stacks up to vanilla's absolute persistent item-stack count limit. */
	@Test
	void supportsCountsAboveTheOrdinaryStackLimit() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.STONE, 64), new ItemStack(Items.STONE, 64));
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, Item.ABSOLUTE_MAX_STACK_SIZE);
		assertNull(drop.reason());
		assertEquals(99, drop.stack().getCount());
		assertTrue(drop.slots().getFirst().isEmpty());
		assertEquals(29, drop.slots().get(1).getCount());
		assertEquals(128, SkillInventory.count(inventory, Items.STONE));
	}

	/** An unpersistable oversized entity is rejected rather than losing items on snapshot reload. */
	@Test
	void rejectsCountsAboveThePersistentStackLimit() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.STONE, 64), new ItemStack(Items.STONE, 64));
		SkillDrop drop = SkillDrop.plan(inventory, Items.STONE, Item.ABSOLUTE_MAX_STACK_SIZE + 1);
		assertEquals("stack_too_large", drop.reason());
		assertTrue(drop.stack().isEmpty());
		assertTrue(drop.slots().isEmpty());
		assertEquals(128, SkillInventory.count(inventory, Items.STONE));
	}

	/** Nonpositive counts are invalid, never empty successful drops. */
	@Test
	void rejectsNonpositiveCounts() {
		SimpleContainer inventory = new SimpleContainer(new ItemStack(Items.STONE, 2));
		assertThrows(IllegalArgumentException.class, () -> SkillDrop.plan(inventory, Items.STONE, 0));
		assertThrows(IllegalArgumentException.class, () -> SkillDrop.plan(inventory, Items.STONE, -1));
		assertEquals(2, inventory.getItem(0).getCount());
	}

	/** Builds real component-bearing stacks without requiring a Minecraft world. */
	private static ItemStack named(Item item, int count, String name) {
		ItemStack stack = new ItemStack(item, count);
		stack.set(DataComponents.CUSTOM_NAME, Component.literal(name));
		return stack;
	}
}
