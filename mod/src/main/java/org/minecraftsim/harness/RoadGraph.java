package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.phys.Vec3;

/**
 * Maintains passable NPC road surfaces and cached routes on the server thread.
 * Dirty nodes, traversal tie breaks and node ordering are deterministic; no RNG is consumed.
 */
public final class RoadGraph {
	private Region region;
	private Set<String> surfaces;
	private TreeSet<Node> nodes = new TreeSet<>();
	private TreeSet<Node> dirty = new TreeSet<>();
	private long version = 1;
	private transient Map<Route, List<Node>> cache = new HashMap<>();

	/**
	 * Scans a loaded region in x/y/z order for surfaces with two clear blocks above.
	 * @param level server level to inspect
	 * @param region inclusive road bounds
	 * @param surfaces allowed surface block identifiers, copied in sorted order
	 * @throws IllegalArgumentException if a scanned surface chunk is not loaded
	 */
	public RoadGraph(ServerLevel level, Region region, Set<String> surfaces) {
		this.region = region;
		this.surfaces = new TreeSet<>(surfaces);
		for (int x = region.x0(); x <= region.x1(); x++) {
			for (int y = region.y0(); y <= region.y1(); y++) {
				for (int z = region.z0(); z <= region.z1(); z++) {
					Node node = new Node(x, y, z);
					if (!level.hasChunkAt(node.block())) {
						throw new IllegalArgumentException("NPC region must be loaded");
					}
					if (passable(level, node)) {
						nodes.add(node);
					}
				}
			}
		}
	}

	/** Reinitializes the transient route cache after restoring persisted graph state. */
	public void restored() {
		cache = new HashMap<>();
	}

	/**
	 * Tests current graph membership without refreshing dirty surfaces.
	 * @param node road surface coordinate
	 * @return whether the node is currently passable
	 */
	public boolean contains(Node node) {
		return nodes.contains(node);
	}

	/**
	 * Reads the revision used to detect route-invalidating topology changes.
	 * @return current graph version
	 */
	public long version() {
		return version;
	}

	/**
	 * Marks the changed block and two supporting surface candidates dirty within the region.
	 * @param position block whose state changed
	 */
	public void changed(BlockPos position) {
		for (int height = 0; height <= 2; height++) {
			Node node = new Node(position.getX(), position.getY() - height, position.getZ());
			if (region.contains(node.x, node.y, node.z)) {
				dirty.add(node);
			}
		}
	}

	/**
	 * Rechecks dirty nodes in sorted order and invalidates routes only on topology changes.
	 * @param level server level containing this graph
	 * @return whether passable-node membership changed
	 */
	public boolean update(ServerLevel level) {
		boolean changed = false;
		for (Node node : dirty) {
			changed |= passable(level, node) ? nodes.add(node) : nodes.remove(node);
		}
		dirty.clear();
		if (changed) {
			version++;
			cache.clear();
		}
		return changed;
	}

	private boolean passable(ServerLevel level, Node node) {
		BlockPos surface = node.block();
		return level.hasChunkAt(surface.above(2)) && surfaces.contains(HarnessState.blockId(level.getBlockState(surface)))
			&& level.getBlockState(surface.above()).getCollisionShape(level, surface.above()).isEmpty()
			&& level.getBlockState(surface.above(2)).getCollisionShape(level, surface.above(2)).isEmpty();
	}

	/**
	 * Resolves a cached deterministic shortest route over current graph nodes.
	 * @param from starting surface node
	 * @param to destination surface node
	 * @return immutable route including both endpoints, or an empty list if unreachable
	 */
	public List<Node> path(Node from, Node to) {
		return cache.computeIfAbsent(new Route(from, to), route -> find(nodes, route.from, route.to));
	}

	static List<Node> find(Set<Node> nodes, Node from, Node to) {
		if (!nodes.contains(from) || !nodes.contains(to)) {
			return List.of();
		}
		PriorityQueue<Search> open = new PriorityQueue<>(Comparator.comparingInt(Search::score)
			.thenComparingInt(Search::cost).thenComparing(Search::node));
		Map<Node, Integer> costs = new HashMap<>();
		open.add(new Search(from, 0, heuristic(from, to), null));
		costs.put(from, 0);
		while (!open.isEmpty()) {
			Search current = open.remove();
			if (current.cost != costs.get(current.node)) {
				continue;
			}
			if (current.node.equals(to)) {
				List<Node> path = new ArrayList<>();
				for (Search entry = current; entry != null; entry = entry.parent) {
					path.add(entry.node);
				}
				Collections.reverse(path);
				return List.copyOf(path);
			}
			for (int[] direction : new int[][]{{-1, 0}, {0, -1}, {0, 1}, {1, 0}}) {
				for (int height = -1; height <= 1; height++) {
					Node next = new Node(current.node.x + direction[0], current.node.y + height, current.node.z + direction[1]);
					int cost = current.cost + 1;
					if (nodes.contains(next) && cost < costs.getOrDefault(next, Integer.MAX_VALUE)) {
						costs.put(next, cost);
						open.add(new Search(next, cost, cost + heuristic(next, to), current));
					}
				}
			}
		}
		return List.of();
	}

	private static int heuristic(Node from, Node to) {
		return Math.abs(from.x - to.x) + Math.abs(from.z - to.z);
	}

	/**
	 * Road surface identity ordered lexicographically by x, y, then z.
	 * @param x surface block x coordinate
	 * @param y surface block y coordinate
	 * @param z surface block z coordinate
	 */
	public record Node(int x, int y, int z) implements Comparable<Node> {
		/**
		 * Converts this surface node without offsets.
		 * @return corresponding block position
		 */
		public BlockPos block() {
			return new BlockPos(x, y, z);
		}

		/**
		 * Places an NPC at the center of the block above this surface.
		 * @return standing foot position
		 */
		public Vec3 feet() {
			return new Vec3(x + 0.5, y + 1.0, z + 0.5);
		}

		/**
		 * Compares coordinates in the graph's stable traversal order.
		 * @param other node to compare
		 * @return negative, zero or positive according to x/y/z lexicographic order
		 */
		@Override
		public int compareTo(Node other) {
			int order = Integer.compare(x, other.x);
			if (order == 0) {
				order = Integer.compare(y, other.y);
			}
			return order == 0 ? Integer.compare(z, other.z) : order;
		}
	}

	private record Route(Node from, Node to) {
	}

	private record Search(Node node, int cost, int score, Search parent) {
	}
}
