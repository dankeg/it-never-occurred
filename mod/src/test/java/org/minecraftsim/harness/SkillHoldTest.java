package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.core.component.DataComponents;
import net.minecraft.network.chat.Component;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins deterministic hotbar selection and component-preserving inventory swaps without a server. */
final class SkillHoldTest {
	/** Initializes item registries and their stack components. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** An already held match wins over earlier hotbar and inventory matches. */
	@Test
	void keepsTheSelectedMatchingStack() {
		SimpleContainer inventory = new SimpleContainer(41);
		ItemStack held = new ItemStack(Items.IRON_PICKAXE);
		inventory.setItem(1, new ItemStack(Items.IRON_PICKAXE));
		inventory.setItem(7, held);
		assertEquals(7, SkillInventory.hold(inventory, Items.IRON_PICKAXE, 7, true));
		assertSame(held, inventory.getItem(7));
	}

	/** The first matching hotbar slot is selected without moving or merging any stacks. */
	@Test
	void selectsTheFirstHotbarMatch() {
		SimpleContainer inventory = new SimpleContainer(41);
		ItemStack first = new ItemStack(Items.STONE, 2);
		inventory.setItem(2, first);
		inventory.setItem(6, new ItemStack(Items.STONE, 3));
		inventory.setItem(10, new ItemStack(Items.STONE, 4));
		assertEquals(2, SkillInventory.hold(inventory, Items.STONE, 0, false));
		assertEquals(2, SkillInventory.hold(inventory, Items.STONE, 0, true));
		assertSame(first, inventory.getItem(2));
		assertTrue(inventory.getItem(0).isEmpty());
	}

	/** A full hotbar swaps with the first matching inventory slot, preserving both complete stacks. */
	@Test
	void swapsTheFirstInventoryMatchIntoTheSelectedSlot() {
		SimpleContainer inventory = new SimpleContainer(41);
		for (int slot = 0; slot < 9; slot++) {
			inventory.setItem(slot, new ItemStack(Items.DIRT, 64));
		}
		ItemStack previous = inventory.getItem(4);
		ItemStack wanted = new ItemStack(Items.IRON_PICKAXE);
		wanted.set(DataComponents.CUSTOM_NAME, Component.literal("Builder's pick"));
		wanted.set(DataComponents.DAMAGE, 17);
		inventory.setItem(12, wanted);
		inventory.setItem(20, new ItemStack(Items.IRON_PICKAXE));
		assertEquals(4, SkillInventory.hold(inventory, Items.IRON_PICKAXE, 4, false));
		assertSame(previous, inventory.getItem(4));
		assertSame(wanted, inventory.getItem(12));
		assertEquals(4, SkillInventory.hold(inventory, Items.IRON_PICKAXE, 4, true));
		assertSame(wanted, inventory.getItem(4));
		assertSame(previous, inventory.getItem(12));
		assertEquals(17, inventory.getItem(4).getDamageValue());
		assertEquals("Builder's pick", inventory.getItem(4).get(DataComponents.CUSTOM_NAME).getString());
		assertTrue(inventory.getItem(20).is(Items.IRON_PICKAXE));
	}

	/** Absent items leave every slot unchanged in both dry-run and commit modes. */
	@Test
	void missingItemsDoNotChangeInventory() {
		SimpleContainer inventory = new SimpleContainer(41);
		ItemStack held = new ItemStack(Items.STONE, 8);
		inventory.setItem(0, held);
		assertEquals(-1, SkillInventory.hold(inventory, Items.IRON_PICKAXE, 0, false));
		assertEquals(-1, SkillInventory.hold(inventory, Items.IRON_PICKAXE, 0, true));
		assertSame(held, inventory.getItem(0));
		assertEquals(8, held.getCount());
	}
}
