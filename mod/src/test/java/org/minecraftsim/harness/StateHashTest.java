package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.ListTag;
import net.minecraft.nbt.StringTag;
import net.minecraft.nbt.Tag;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Pins the canonical bytes consumed by state hashing without calculating checksums.
 * Revert check (2026-10-18): the three order tests fail when the unordered-list branch is removed (D1, D6).
 */
final class StateHashTest {
	/** Initializes registries once so block-state IDs are available. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Attribute insertion order does not change the canonical hash input. */
	@Test
	void attributesInDifferentInsertionOrderEncodeIdentically() throws IOException {
		CompoundTag health = new CompoundTag();
		health.putString("id", "minecraft:max_health");
		health.putDouble("base", 20.0);
		CompoundTag speed = new CompoundTag();
		speed.putString("id", "minecraft:movement_speed");
		speed.putDouble("base", 0.1);

		assertArrayEquals(canonical(withList("attributes", health, speed)),
			canonical(withList("attributes", speed, health)));
	}

	/** Modifier insertion order remains irrelevant inside an attribute compound. */
	@Test
	void nestedModifiersInDifferentInsertionOrderEncodeIdentically() throws IOException {
		CompoundTag first = new CompoundTag();
		first.putString("id", "minecraft:first");
		first.putDouble("amount", 1.5);
		CompoundTag second = new CompoundTag();
		second.putString("id", "minecraft:second");
		second.putDouble("amount", -2.0);
		CompoundTag forward = withList("modifiers", first, second);
		forward.putString("id", "minecraft:movement_speed");
		CompoundTag reverse = withList("modifiers", second, first);
		reverse.putString("id", "minecraft:movement_speed");

		assertArrayEquals(canonical(withList("attributes", forward)), canonical(withList("attributes", reverse)));
	}

	/** Both recipe-book lists ignore insertion order while retaining duplicate entries. */
	@Test
	void recipeBookListsInDifferentInsertionOrderEncodeIdentically() throws IOException {
		for (String key : new String[]{"recipes", "toBeDisplayed"}) {
			CompoundTag forward = new CompoundTag();
			forward.put("recipeBook", withList(key, StringTag.valueOf("minecraft:bread"),
				StringTag.valueOf("minecraft:stick"), StringTag.valueOf("minecraft:bread")));
			CompoundTag reverse = new CompoundTag();
			reverse.put("recipeBook", withList(key, StringTag.valueOf("minecraft:stick"),
				StringTag.valueOf("minecraft:bread"), StringTag.valueOf("minecraft:bread")));
			CompoundTag fewer = new CompoundTag();
			fewer.put("recipeBook", withList(key, StringTag.valueOf("minecraft:bread"), StringTag.valueOf("minecraft:stick")));

			assertArrayEquals(canonical(forward), canonical(reverse), key);
			assertFalse(Arrays.equals(canonical(forward), canonical(fewer)), key);
		}
	}

	/** An ordinary list retains its order and its exact type-tagged string encoding. */
	@Test
	void ordinaryListsDistinguishElementOrder() throws IOException {
		ListTag forward = new ListTag();
		forward.add(StringTag.valueOf("a"));
		forward.add(StringTag.valueOf("b"));
		ListTag reverse = new ListTag();
		reverse.add(StringTag.valueOf("b"));
		reverse.add(StringTag.valueOf("a"));

		assertArrayEquals(new byte[]{9, 0, 0, 0, 2, 8, 0, 0, 0, 1, 97, 8, 0, 0, 0, 1, 98}, canonical(forward));
		assertArrayEquals(new byte[]{9, 0, 0, 0, 2, 8, 0, 0, 0, 1, 98, 8, 0, 0, 0, 1, 97}, canonical(reverse));
		assertFalse(Arrays.equals(canonical(withList("Items", forward)), canonical(withList("Items", reverse))));
	}

	/** Block states contribute exactly four big-endian registry-ID bytes rather than names. */
	@Test
	void blockStatesEncodeAsIntegerRegistryIds() throws IOException {
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		StateHash.writeBlockState(Blocks.AIR.defaultBlockState(), new DataOutputStream(bytes));
		assertArrayEquals(new byte[]{0, 0, 0, 0}, bytes.toByteArray());
		for (BlockState state : Blocks.OAK_LOG.getStateDefinition().getPossibleStates()) {
			bytes.reset();
			StateHash.writeBlockState(state, new DataOutputStream(bytes));
			assertArrayEquals(ByteBuffer.allocate(Integer.BYTES).putInt(Block.getId(state)).array(), bytes.toByteArray());
		}
	}

	private static CompoundTag withList(String key, Tag... elements) {
		ListTag list = new ListTag();
		for (Tag element : elements) {
			list.add(element);
		}
		CompoundTag compound = new CompoundTag();
		compound.put(key, list);
		return compound;
	}

	private static byte[] canonical(Tag tag) throws IOException {
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		StateHash.writeCanonicalNbt(tag, new DataOutputStream(bytes));
		return bytes.toByteArray();
	}
}
