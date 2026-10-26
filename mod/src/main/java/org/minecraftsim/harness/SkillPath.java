package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.function.Predicate;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.level.material.Fluids;

/**
 * Performs bounded deterministic bot path searches over loaded blocks and planned overlays.
 * Coordinate tie breaks, neighbor order and search limits are fixed; planning draws no RNG.
 */
public final class SkillPath {
	/** Lexicographic x/y/z ordering used for deterministic candidate tie breaks. */
	public static final Comparator<BlockPos> POSITIONS = Comparator.comparingInt((BlockPos position) -> position.getX())
		.thenComparingInt(position -> position.getY()).thenComparingInt(position -> position.getZ());
	private static final int[][] DIRECTIONS = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
	/** Search limits sized for a 256-block town, including detours (e.g. a river ford) several times the direct distance. */
	private static final int MAX_EXPANSIONS = 200_000;
	private static final int MAX_DEPTH = 1024;
	private static final int SWIM_RANGE = 16;

	private SkillPath() {
	}

	/** Foothold candidates around an unsupported start, in fixed order: straight down first, then the same level. */
	private static final int[][] FOOTHOLDS = {{0, -1, 0}, {1, -1, 0}, {-1, -1, 0}, {0, -1, 1}, {0, -1, -1},
		{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {1, -1, 1}, {1, -1, -1}, {-1, -1, 1}, {-1, -1, -1}};

	/**
	 * Prepends a bounded swim to the nearest bank when starting in water, or one step to a foothold when the start
	 * cell itself cannot be stood in (a bot perched on the edge of a desk or chest), then searches standable positions.
	 * @param view world/overlay collision view
	 * @param start bot foot block at search start
	 * @param goal desired foot block
	 * @return route excluding start and including goal, or null if invalid, unreachable or search-limited
	 */
	public static Route find(View view, BlockPos start, BlockPos goal) {
		if (inWater(view, start)) {
			Route swim = swimLeg(view, start);
			if (swim == null) {
				return null;
			}
			Route walk = find(view, swim.positions().getLast(), goal);
			if (walk == null) {
				return null;
			}
			List<BlockPos> positions = new ArrayList<>(swim.positions());
			positions.addAll(walk.positions());
			return new Route(List.copyOf(positions), swim.length() + walk.length(), swim.positions().size());
		}
		if (!standable(view, start) && standable(view, goal)) {
			BlockPos foothold = foothold(view, start);
			Route walk = foothold == null ? null : find(view, foothold, goal);
			if (walk == null) {
				return null;
			}
			List<BlockPos> positions = new ArrayList<>(List.of(foothold));
			positions.addAll(walk.positions());
			return new Route(List.copyOf(positions), Math.sqrt(start.distSqr(foothold)) + walk.length());
		}
		if (!standable(view, start) || !standable(view, goal)) {
			return null;
		}
		PriorityQueue<Node> open = new PriorityQueue<>(Comparator.comparingDouble(Node::score)
			.thenComparingDouble(Node::cost).thenComparing(Node::position, POSITIONS));
		Map<BlockPos, Double> best = new HashMap<>();
		Node first = new Node(start.immutable(), 0, heuristic(start, goal), 0, null);
		open.add(first);
		best.put(start, 0.0);
		int expansions = 0;
		while (!open.isEmpty() && expansions < MAX_EXPANSIONS) {
			Node current = open.remove();
			if (Double.compare(current.cost(), best.get(current.position())) != 0) {
				continue;
			}
			expansions++;
			if (current.position().equals(goal)) {
				List<BlockPos> path = new ArrayList<>();
				for (Node node = current; node.parent() != null; node = node.parent()) {
					path.add(node.position());
				}
				Collections.reverse(path);
				double length = 0;
				BlockPos previous = start;
				for (BlockPos position : path) {
					length += Math.sqrt(position.distSqr(previous));
					previous = position;
				}
				return new Route(List.copyOf(path), length);
			}
			if (current.depth() == MAX_DEPTH) {
				continue;
			}
			for (int[] direction : DIRECTIONS) {
				BlockPos next = neighbour(view, current.position(), direction[0], direction[1]);
				if (next == null) {
					continue;
				}
				if (direction[0] != 0 && direction[1] != 0
						&& (!crossing(view, current.position(), current.position().offset(direction[0], next.getY() - current.position().getY(), 0))
						|| !crossing(view, current.position(), current.position().offset(0, next.getY() - current.position().getY(), direction[1])))) {
					continue;
				}
				int elevation = next.getY() - current.position().getY();
				double cost = current.cost() + (direction[0] != 0 && direction[1] != 0 ? Math.sqrt(2) : 1)
					+ (elevation > 0 ? 0.5 : -elevation * 0.2);
				if (cost < best.getOrDefault(next, Double.POSITIVE_INFINITY)) {
					best.put(next, cost);
					open.add(new Node(next, cost, cost + heuristic(next, goal), current.depth() + 1, current));
				}
			}
		}
		return null;
	}

	/**
	 * Finds the nearest dry standing cell through clear water or its surface, within sixteen travel blocks.
	 * Equal-length banks and search nodes use x/y/z order; face neighbors never cut a solid corner.
	 * @param view loaded-world collision and fluid view
	 * @param start submerged bot foot block
	 * @return immutable swim waypoints including the bank, or null when no bank is reachable in range
	 */
	public static Route swimLeg(View view, BlockPos start) {
		if (!inWater(view, start) || !swimmable(view, start)) {
			return null;
		}
		PriorityQueue<Node> open = new PriorityQueue<>(Comparator.comparingDouble(Node::cost)
			.thenComparing(Node::position, POSITIONS));
		Map<BlockPos, Double> best = new HashMap<>();
		open.add(new Node(start.immutable(), 0, 0, 0, null));
		best.put(start, 0.0);
		while (!open.isEmpty()) {
			Node current = open.remove();
			if (Double.compare(current.cost(), best.get(current.position())) != 0) {
				continue;
			}
			if (standable(view, current.position())) {
				List<BlockPos> positions = new ArrayList<>();
				for (Node node = current; node.parent() != null; node = node.parent()) {
					positions.add(node.position());
				}
				Collections.reverse(positions);
				return new Route(List.copyOf(positions), current.cost(), positions.size());
			}
			List<BlockPos> neighbours = new ArrayList<>();
			for (Direction direction : Direction.values()) {
				neighbours.add(current.position().relative(direction));
				if (direction.getAxis().isHorizontal()) {
					BlockPos bank = current.position().relative(direction).above();
					if (standable(view, bank) && swimClear(view, current.position().above(2))) {
						neighbours.add(bank);
					}
				}
			}
			neighbours.sort(POSITIONS);
			for (BlockPos next : neighbours) {
				double cost = current.cost() + Math.sqrt(current.position().distSqr(next));
				if (cost <= SWIM_RANGE && (standable(view, next) || swimmable(view, next))
						&& cost < best.getOrDefault(next, Double.POSITIVE_INFINITY)) {
					best.put(next, cost);
					open.add(new Node(next, cost, cost, current.depth() + 1, current));
				}
			}
		}
		return null;
	}

	/**
	 * The first standable cell next to an unsupported start (down first, then level, then diagonally down), so a bot
	 * whose body rests on an obstacle's edge steps off it instead of failing every plan; null if none.
	 * @param view loaded-world collision view
	 * @param start bot foot block that is not standable
	 * @return the foothold, or null
	 */
	public static BlockPos foothold(View view, BlockPos start) {
		for (int[] offset : FOOTHOLDS) {
			BlockPos candidate = start.offset(offset[0], offset[1], offset[2]);
			if (standable(view, candidate)) {
				return candidate.immutable();
			}
		}
		return null;
	}

	/** Tests the foot/head cells for water without loading chunks or admitting lava. */
	public static boolean inWater(View view, BlockPos feet) {
		return water(view, feet) || water(view, feet.above());
	}

	private static boolean water(View view, BlockPos position) {
		FluidState fluid = view.getFluidState(position);
		if (fluid.isEmpty() || fluid.getType() == Fluids.LAVA || fluid.getType() == Fluids.FLOWING_LAVA) {
			return false;
		}
		return fluid.getType() == Fluids.WATER || fluid.getType() == Fluids.FLOWING_WATER || fluid.is(FluidTags.WATER);
	}

	private static boolean swimClear(View view, BlockPos position) {
		return view.loaded(position) && view.getBlockState(position).getCollisionShape(view, position).isEmpty()
			&& (view.getFluidState(position).isEmpty() || water(view, position));
	}

	private static boolean swimmable(View view, BlockPos position) {
		return (inWater(view, position) || water(view, position.below()))
			&& swimClear(view, position) && swimClear(view, position.above());
	}

	private static double heuristic(BlockPos from, BlockPos to) {
		int horizontalX = Math.abs(from.getX() - to.getX());
		int horizontalZ = Math.abs(from.getZ() - to.getZ());
		return Math.max(horizontalX, horizontalZ) + (Math.sqrt(2) - 1) * Math.min(horizontalX, horizontalZ);
	}

	private static BlockPos neighbour(View view, BlockPos from, int deltaX, int deltaZ) {
		for (int elevation : new int[]{0, 1, -1, -2, -3}) {
			BlockPos candidate = from.offset(deltaX, elevation, deltaZ);
			if (standable(view, candidate) && crossing(view, from, candidate)) {
				return candidate;
			}
		}
		return null;
	}

	private static boolean crossing(View view, BlockPos from, BlockPos to) {
		if (!passable(view, to) || !passable(view, to.above())) {
			return false;
		}
		if (to.getY() > from.getY() && !passable(view, from.above(2))) {
			return false;
		}
		for (int height = to.getY() + 2; height <= from.getY() + 1; height++) {
			if (!passable(view, new BlockPos(to.getX(), height, to.getZ()))) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Tests clear feet/head blocks and a loaded sturdy surface below. Dirt paths and farmland (15/16 high) count as
	 * surfaces: they are walkable in vanilla even though their top face is not a full sturdy face.
	 * @param view world/overlay collision view
	 * @param position proposed foot block
	 * @return whether the bot may stand at the position
	 */
	public static boolean standable(View view, BlockPos position) {
		BlockPos below = position.below();
		if (!passable(view, position) || !passable(view, position.above()) || !view.loaded(below)) {
			return false;
		}
		BlockState support = view.getBlockState(below);
		return support.isFaceSturdy(view, below, Direction.UP) || support.is(Blocks.DIRT_PATH) || support.is(Blocks.FARMLAND);
	}

	/**
	 * Tests loaded blocks without water/lava types or tags. Doors, trapdoors and gates require an open state;
	 * all other blocks require an empty collision shape.
	 * @param view world/overlay collision view
	 * @param position block through which the bot would pass
	 * @return whether the block satisfies the planning passability policy
	 */
	public static boolean passable(View view, BlockPos position) {
		if (!view.loaded(position)) {
			return false;
		}
		BlockState state = view.getBlockState(position);
		FluidState fluid = state.getFluidState();
		// An empty fluid is in no tag; skipping its tag lookups also keeps this callable where tags are unbound
		if (!fluid.isEmpty() && (fluid.getType() == Fluids.WATER || fluid.getType() == Fluids.FLOWING_WATER
				|| fluid.getType() == Fluids.LAVA || fluid.getType() == Fluids.FLOWING_LAVA
				|| fluid.is(FluidTags.WATER) || fluid.is(FluidTags.LAVA))) {
			return false;
		}
		if (state.getBlock() instanceof DoorBlock || state.getBlock() instanceof TrapDoorBlock
				|| state.getBlock() instanceof FenceGateBlock) {
			return state.getValue(BlockStateProperties.OPEN);
		}
		return state.getCollisionShape(view, position).isEmpty();
	}

	/**
	 * Planned movement data; find supplies an immutable step list in travel order.
	 * @param positions route steps excluding the start block
	 * @param length accumulated Euclidean step length
	 * @param swimWaypoints number of initial waypoints steered with jump held, including the bank
	 */
	public record Route(List<BlockPos> positions, double length, int swimWaypoints) {
		/** Creates an ordinary walking route without changing its existing budget or waypoints. */
		public Route(List<BlockPos> positions, double length) {
			this(positions, length, 0);
		}

		/**
		 * Estimates the fixed skill budget using route length and the existing speed margin.
		 * @return ceiling of length/0.2158 plus ten ticks
		 */
		public long ticks() {
			return (long) Math.ceil(length / 0.2158) + 10;
		}
	}

	private record Node(BlockPos position, double cost, double score, int depth, Node parent) {
	}

	/**
	 * Reads a server world with an optional planned-state overlay, never loading chunks.
	 * Out-of-bounds/unloaded positions act as bedrock; overlay blocks hide real block entities.
	 */
	public static final class View implements BlockGetter {
		private final BlockGetter level;
		private final Predicate<BlockPos> loaded;
		private final Map<BlockPos, BlockState> overlay;

		/**
		 * Creates an unmodified loaded-world view.
		 * @param level server level to query on the server thread
		 */
		public View(ServerLevel level) {
			this(level, Map.of());
		}

		/**
		 * Creates a planning view retaining the supplied overlay reference.
		 * @param level server level to query on the server thread
		 * @param overlay planned block states keyed by position
		 */
		public View(ServerLevel level, Map<BlockPos, BlockState> overlay) {
			this(level, level::hasChunkAt, overlay);
		}

		View(BlockGetter level, Predicate<BlockPos> loaded, Map<BlockPos, BlockState> overlay) {
			this.level = level;
			this.loaded = loaded;
			this.overlay = overlay;
		}

		/**
		 * Checks vertical bounds before consulting the configured chunk predicate.
		 * @param position queried coordinate
		 * @return whether the view permits reading that coordinate
		 */
		public boolean loaded(BlockPos position) {
			return position.getY() >= level.getMinY() && position.getY() <= level.getMaxY() && loaded.test(position);
		}

		/**
		 * Reads the overlay first, or bedrock when the position is unavailable.
		 * @param position queried block coordinate
		 * @return planned/world state, or the bedrock sentinel
		 */
		@Override
		public BlockState getBlockState(BlockPos position) {
			if (!loaded(position)) {
				return Blocks.BEDROCK.defaultBlockState();
			}
			return overlay.containsKey(position) ? overlay.get(position) : level.getBlockState(position);
		}

		/**
		 * Reads fluid state through the same overlay and availability policy as blocks.
		 * @param position queried block coordinate
		 * @return fluid state of the view's block state
		 */
		@Override
		public FluidState getFluidState(BlockPos position) {
			return getBlockState(position).getFluidState();
		}

		/**
		 * Exposes real block entities only for loaded positions untouched by the overlay.
		 * @param position queried block coordinate
		 * @return world block entity, or null if unavailable, overlaid or absent
		 */
		@Override
		public BlockEntity getBlockEntity(BlockPos position) {
			return !loaded(position) || overlay.containsKey(position) ? null : level.getBlockEntity(position);
		}

		/**
		 * Preserves the wrapped world's vertical extent.
		 * @return world height in blocks
		 */
		@Override
		public int getHeight() {
			return level.getHeight();
		}

		/**
		 * Preserves the wrapped world's lower vertical bound.
		 * @return minimum world y coordinate
		 */
		@Override
		public int getMinY() {
			return level.getMinY();
		}
	}
}
