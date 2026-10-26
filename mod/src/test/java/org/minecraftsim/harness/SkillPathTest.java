package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.EmptyBlockGetter;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.block.state.properties.DoorHingeSide;
import net.minecraft.world.level.block.state.properties.DoubleBlockHalf;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.level.material.Fluids;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.shapes.BooleanOp;
import net.minecraft.world.phys.shapes.Shapes;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins open-state planning, unchanged collision/fluid rules and the village's physical doorway clearance. */
final class SkillPathTest {
	/** Initializes vanilla states before constructing world-independent planning views. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Doors, trapdoors and gates require OPEN regardless of their collision shape or manual usability. */
	@Test
	void controlsArePassableOnlyWhenOpen() {
		for (Block block : new Block[]{Blocks.OAK_DOOR, Blocks.IRON_DOOR, Blocks.OAK_TRAPDOOR,
				Blocks.IRON_TRAPDOOR, Blocks.OAK_FENCE_GATE}) {
			for (boolean open : new boolean[]{false, true}) {
				BlockState state = block.defaultBlockState().setValue(BlockStateProperties.OPEN, open);
				SkillPath.View view = new SkillPath.View(new EmptyWorld(), position -> true,
					Map.of(BlockPos.ZERO, state));
				assertEquals(open, SkillPath.passable(view, BlockPos.ZERO), state.toString());
			}
		}
	}

	/** The new block-family exception does not admit fluids, unloaded cells or unrelated solid blocks. */
	@Test
	void fluidLoadingAndOtherCollisionPoliciesStayUnchanged() {
		for (Block block : new Block[]{Blocks.AIR, Blocks.STONE, Blocks.GLASS, Blocks.OAK_FENCE,
				Blocks.WATER, Blocks.LAVA}) {
			SkillPath.View view = new SkillPath.View(new EmptyWorld(), position -> true,
				Map.of(BlockPos.ZERO, block.defaultBlockState()));
			assertEquals(block == Blocks.AIR, SkillPath.passable(view, BlockPos.ZERO), block.toString());
		}
		BlockState wet = Blocks.OAK_TRAPDOOR.defaultBlockState().setValue(BlockStateProperties.OPEN, true)
			.setValue(BlockStateProperties.WATERLOGGED, true);
		assertFalse(SkillPath.passable(new SkillPath.View(new EmptyWorld(), position -> true,
			Map.of(BlockPos.ZERO, wet)), BlockPos.ZERO));
		assertFalse(SkillPath.passable(new SkillPath.View(new EmptyWorld(), position -> false,
			Map.of(BlockPos.ZERO, Blocks.OAK_DOOR.defaultBlockState().setValue(DoorBlock.OPEN, true))), BlockPos.ZERO));
	}

	/** A bounded corridor routes through each open control and rejects its closed state without a detour. */
	@Test
	void routesThroughOpenControlsAreDeterministicAndClosedControlsHaveNoPath() {
		BlockPos doorway = new BlockPos(0, 1, 0);
		BlockPos start = new BlockPos(0, 1, -2);
		BlockPos goal = new BlockPos(0, 1, 2);
		for (Block block : new Block[]{Blocks.OAK_DOOR, Blocks.OAK_TRAPDOOR, Blocks.OAK_FENCE_GATE}) {
			Map<BlockPos, BlockState> blocks = new HashMap<>();
			for (int depth = -2; depth <= 2; depth++) {
				blocks.put(new BlockPos(0, 0, depth), Blocks.STONE.defaultBlockState());
			}
			SkillPath.View view = new SkillPath.View(new EmptyWorld(),
				position -> position.getX() == 0 && Math.abs(position.getZ()) <= 2
					&& position.getY() >= 0 && position.getY() <= 3, blocks);
			BlockState open = block.defaultBlockState().setValue(BlockStateProperties.OPEN, true);
			blocks.put(doorway, open);
			if (block instanceof DoorBlock) {
				blocks.put(doorway.above(), open.setValue(DoorBlock.HALF, DoubleBlockHalf.UPPER));
			}
			SkillPath.Route route = SkillPath.find(view, start, goal);
			assertNotNull(route, block.toString());
			assertTrue(route.positions().contains(doorway));
			assertEquals(4.0, route.length());
			assertEquals(route, SkillPath.find(view, start, goal));
			blocks.put(doorway, open.setValue(BlockStateProperties.OPEN, false));
			assertNull(SkillPath.find(view, start, goal), block.toString());
			if (block instanceof DoorBlock) {
				blocks.put(doorway, open);
				blocks.put(doorway.above(), open.setValue(DoorBlock.HALF, DoubleBlockHalf.UPPER)
					.setValue(DoorBlock.OPEN, false));
				assertNull(SkillPath.find(view, start, goal));
			}
		}
	}

	/** The builder's south-facing hinge pair leaves both walking lanes clear for a 0.6-wide, 1.8-high bot. */
	@Test
	void villageDoorPanelsFoldAgainstOuterJambsAndClearRealBotCollision() {
		AABB passage = new AABB(0.2, 0, -1, 1.8, 1.8, 2);
		for (int column = 0; column < 2; column++) {
			for (DoubleBlockHalf half : DoubleBlockHalf.values()) {
				BlockPos position = new BlockPos(column, half == DoubleBlockHalf.LOWER ? 0 : 1, 0);
				BlockState open = Blocks.OAK_DOOR.defaultBlockState().setValue(DoorBlock.FACING, Direction.SOUTH)
					.setValue(DoorBlock.HINGE, column == 0 ? DoorHingeSide.RIGHT : DoorHingeSide.LEFT)
					.setValue(DoorBlock.HALF, half).setValue(DoorBlock.OPEN, true);
				assertFalse(Shapes.joinIsNotEmpty(open.getCollisionShape(EmptyBlockGetter.INSTANCE, position)
					.move(position.getX(), position.getY(), position.getZ()), Shapes.create(passage), BooleanOp.AND));
				assertTrue(Shapes.joinIsNotEmpty(open.setValue(DoorBlock.OPEN, false)
					.getCollisionShape(EmptyBlockGetter.INSTANCE, position)
					.move(position.getX(), position.getY(), position.getZ()), Shapes.create(passage), BooleanOp.AND));
			}
		}
	}

	/** A swim leg ends at the closest dry standing cell and ordinary walking continues from there. */
	@Test
	void waterStartPrependsBankBeforeWalkingToTheGoal() {
		SkillPath.View view = river(0, 20, 4, 4);
		BlockPos start = new BlockPos(0, 1, 0);
		BlockPos bank = new BlockPos(4, 1, 0);
		SkillPath.Route swim = SkillPath.swimLeg(view, start);
		assertNotNull(swim);
		assertEquals(bank, swim.positions().getLast());
		assertEquals(4, swim.length());
		SkillPath.Route route = SkillPath.find(view, start, new BlockPos(20, 1, 0));
		assertNotNull(route);
		assertEquals(swim.positions(), route.positions().subList(0, route.swimWaypoints()));
		assertEquals(20, route.length());
		assertEquals(new BlockPos(20, 1, 0), route.positions().getLast());
		assertEquals(route, SkillPath.find(view, start, new BlockPos(20, 1, 0)));
		assertEquals(0, SkillPath.find(view, bank, new BlockPos(20, 1, 0)).swimWaypoints());
	}

	/** Distance outranks coordinate ordering; equal-distance banks use the existing x/y/z tie-break. */
	@Test
	void bankSearchUsesNearestDistanceThenCoordinateTieBreak() {
		BlockPos start = new BlockPos(0, 1, 0);
		SkillPath.View tied = river(-2, 2, 2, 2);
		SkillPath.Route first = SkillPath.swimLeg(tied, start);
		assertNotNull(first);
		assertEquals(new BlockPos(-2, 1, 0), first.positions().getLast());
		assertEquals(first, SkillPath.swimLeg(tied, start));
		SkillPath.Route nearer = SkillPath.swimLeg(river(-3, 2, 3, 2), start);
		assertNotNull(nearer);
		assertEquals(new BlockPos(2, 1, 0), nearer.positions().getLast());
	}

	/** Sixteen travel blocks are allowed, but a bank one block farther cannot be used. */
	@Test
	void bankSearchEnforcesInclusiveSixteenBlockRange() {
		BlockPos start = new BlockPos(0, 1, 0);
		SkillPath.Route boundary = SkillPath.swimLeg(river(0, 16, 16, 16), start);
		assertNotNull(boundary);
		assertEquals(16, boundary.length());
		assertNull(SkillPath.swimLeg(river(0, 17, 17, 17), start));
		assertNull(SkillPath.find(river(0, 17, 17, 17), start, new BlockPos(17, 1, 0)));
	}

	/** Rising through a deep water column and climbing a one-block bank remain part of the swim leg. */
	@Test
	void bankSearchCanRiseThroughWaterAndClimbOut() {
		Map<BlockPos, BlockState> blocks = new HashMap<>();
		for (int depth = 0; depth <= 3; depth++) {
			blocks.put(new BlockPos(0, depth, 0), Blocks.WATER.defaultBlockState());
			blocks.put(new BlockPos(1, depth, 0), Blocks.STONE.defaultBlockState());
		}
		SkillPath.View view = new SkillPath.View(new EmptyWorld(),
			position -> position.getX() >= 0 && position.getX() <= 1 && position.getZ() == 0
				&& position.getY() >= 0 && position.getY() <= 6, blocks);
		SkillPath.Route route = SkillPath.swimLeg(view, BlockPos.ZERO);
		assertNotNull(route);
		assertEquals(new BlockPos(1, 4, 0), route.positions().getLast());
		assertTrue(route.positions().contains(new BlockPos(0, 3, 0)));
		assertEquals(route.positions().size(), route.swimWaypoints());
	}

	/** The search never crosses solid or lava cells and cannot read an unloaded bank. */
	@Test
	void bankSearchRejectsBlockedLavaAndUnloadedRoutes() {
		for (Block barrier : new Block[]{Blocks.STONE, Blocks.LAVA}) {
			Map<BlockPos, BlockState> blocks = new HashMap<>();
			blocks.put(new BlockPos(0, 1, 0), Blocks.WATER.defaultBlockState());
			blocks.put(new BlockPos(2, 0, 0), Blocks.STONE.defaultBlockState());
			for (int height = 0; height <= 5; height++) {
				blocks.put(new BlockPos(1, height, 0), barrier.defaultBlockState());
			}
			SkillPath.View view = new SkillPath.View(new EmptyWorld(),
				position -> position.getX() >= 0 && position.getX() <= 2 && position.getZ() == 0
					&& position.getY() >= 0 && position.getY() <= 5, blocks);
			assertNull(SkillPath.swimLeg(view, new BlockPos(0, 1, 0)), barrier.toString());
		}
		SkillPath.View loaded = river(0, 2, 2, 2);
		SkillPath.View unloadedBank = new SkillPath.View(loaded,
			position -> position.getX() < 2, Map.of());
		assertNull(SkillPath.swimLeg(unloadedBank, new BlockPos(0, 1, 0)));
		assertNull(SkillPath.swimLeg(loaded, new BlockPos(2, 1, 0)));
	}

	/** A bot perched above air beside a desk steps down to the floor first, then walks; standing starts are unchanged. */
	@Test
	void unsupportedStartStepsToAFootholdFirst() {
		Map<BlockPos, BlockState> blocks = new HashMap<>();
		for (int column = 0; column <= 4; column++) {
			blocks.put(new BlockPos(column, 0, 0), Blocks.STONE.defaultBlockState());
		}
		blocks.put(new BlockPos(1, 1, 0), Blocks.OAK_PLANKS.defaultBlockState());  // the desk
		SkillPath.View view = new SkillPath.View(new EmptyWorld(),
			position -> position.getX() >= 0 && position.getX() <= 4 && position.getZ() == 0
				&& position.getY() >= 0 && position.getY() <= 5, blocks);
		BlockPos perched = new BlockPos(2, 2, 0);
		assertFalse(SkillPath.standable(view, perched));
		assertEquals(new BlockPos(2, 1, 0), SkillPath.foothold(view, perched));
		SkillPath.Route route = SkillPath.find(view, perched, new BlockPos(4, 1, 0));
		assertNotNull(route);
		assertEquals(new BlockPos(2, 1, 0), route.positions().getFirst());
		assertEquals(new BlockPos(4, 1, 0), route.positions().getLast());
		assertEquals(SkillPath.find(view, new BlockPos(2, 1, 0), new BlockPos(4, 1, 0)).positions(),
			route.positions().subList(1, route.positions().size()));
	}

	/** Creates a one-cell-wide river with dry end banks and no accidental side exits. */
	private static SkillPath.View river(int minimumX, int maximumX, int leftBank, int rightBank) {
		Map<BlockPos, BlockState> blocks = new HashMap<>();
		for (int column = minimumX; column <= maximumX; column++) {
			blocks.put(new BlockPos(column, 0, 0), Blocks.STONE.defaultBlockState());
			if (column > -leftBank && column < rightBank) {
				blocks.put(new BlockPos(column, 1, 0), Blocks.WATER.defaultBlockState());
			}
		}
		return new SkillPath.View(new EmptyWorld(),
			position -> position.getX() >= minimumX && position.getX() <= maximumX && position.getZ() == 0
				&& position.getY() >= 0 && position.getY() <= 5, blocks);
	}

	/** Supplies air inside real vertical bounds; vanilla EmptyBlockGetter has height zero. */
	private static final class EmptyWorld implements BlockGetter {
		/** Empty test worlds have no block entities. */
		@Override
		public BlockEntity getBlockEntity(BlockPos position) {
			return null;
		}

		/** Planned overlays provide every non-air block. */
		@Override
		public BlockState getBlockState(BlockPos position) {
			return Blocks.AIR.defaultBlockState();
		}

		/** Planned overlays provide every fluid block. */
		@Override
		public FluidState getFluidState(BlockPos position) {
			return Fluids.EMPTY.defaultFluidState();
		}

		/** Includes the test corridor and the village's ground height. */
		@Override
		public int getMinY() {
			return -64;
		}

		/** Keeps test positions inside a nonzero vertical extent. */
		@Override
		public int getHeight() {
			return 256;
		}
	}
}
