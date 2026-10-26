package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.EmptyBlockGetter;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.BellBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BellAttachType;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.CollisionContext;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins block-use scope, deterministic bell strikes and the visual occlusion used by Sight. */
final class SkillUseTest {
	/** Initializes the vanilla registry before constructing block states. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Physical controls are usable; unrelated blocks, containers and beds remain outside this skill. */
	@Test
	void useAcceptsOnlyTheNamedBlockFamilies() {
		for (Block block : new Block[]{Blocks.OAK_DOOR, Blocks.OAK_TRAPDOOR, Blocks.OAK_FENCE_GATE,
				Blocks.BELL, Blocks.LEVER, Blocks.STONE_BUTTON, Blocks.OAK_BUTTON}) {
			assertTrue(Skills.usable(block.defaultBlockState()), block.toString());
		}
		for (Block block : new Block[]{Blocks.AIR, Blocks.STONE, Blocks.CHEST, Blocks.BARREL, Blocks.IRON_DOOR, Blocks.IRON_TRAPDOOR,
				BuiltInRegistries.BLOCK.getValue(Identifier.parse("minecraft:red_bed")), Blocks.CRAFTING_TABLE, Blocks.OAK_FENCE}) {
			assertFalse(Skills.usable(block.defaultBlockState()), block.toString());
		}
	}

	/** Bell hit faces obey vanilla's floor-versus-wall axis rule at a valid body height for every mount. */
	@Test
	void bellStrikesHaveADeterministicVanillaUsableFace() {
		BlockPos position = new BlockPos(1, 2, 3);
		BlockHitResult sight = new BlockHitResult(Vec3.atCenterOf(position), Direction.UP, position, false);
		for (BellAttachType attachment : BellAttachType.values()) {
			BlockState state = Blocks.BELL.defaultBlockState().setValue(BellBlock.ATTACHMENT, attachment)
				.setValue(BellBlock.FACING, Direction.NORTH);
			BlockHitResult hit = Skills.useHit(state, position, sight);
			assertEquals(position, hit.getBlockPos());
			assertEquals(Vec3.atCenterOf(position), hit.getLocation());
			assertEquals(attachment == BellAttachType.SINGLE_WALL || attachment == BellAttachType.DOUBLE_WALL
				? Direction.EAST : Direction.NORTH, hit.getDirection());
		}
		assertSame(sight, Skills.useHit(Blocks.LEVER.defaultBlockState(), position, sight));
	}

	/** Sight's vanilla VISUAL shape blocks rays through a closed doorway and clears its middle when open. */
	@Test
	void closedDoorsAlreadyBlockVisualLineOfSight() {
		BlockState closed = Blocks.OAK_DOOR.defaultBlockState().setValue(DoorBlock.FACING, Direction.NORTH)
			.setValue(DoorBlock.OPEN, false);
		Vec3 from = new Vec3(0.5, 0.5, -1);
		Vec3 to = new Vec3(0.5, 0.5, 2);
		assertNotNull(closed.getVisualShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO, CollisionContext.empty())
			.clip(from, to, BlockPos.ZERO));
		assertNull(closed.setValue(DoorBlock.OPEN, true)
			.getVisualShape(EmptyBlockGetter.INSTANCE, BlockPos.ZERO, CollisionContext.empty()).clip(from, to, BlockPos.ZERO));
	}
}
