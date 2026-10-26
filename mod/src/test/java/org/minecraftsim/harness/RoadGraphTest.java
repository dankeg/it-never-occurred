package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;
import org.minecraftsim.harness.RoadGraph.Node;

/** Pins deterministic shortest-path search over synthetic passable road nodes. */
final class RoadGraphTest {
	/** A direct route wins over a longer connected detour and includes both endpoints. */
	@Test
	void shortestRouteUsesTheFewestRoadSteps() {
		Set<Node> nodes = roads();
		Set<Node> before = Set.copyOf(nodes);
		assertEquals(List.of(node(0, 0), node(1, 0), node(2, 0)), RoadGraph.find(nodes, node(0, 0), node(2, 0)));
		assertEquals(before, nodes);
	}

	/** Closing the direct road reroutes along the remaining shortest detour. */
	@Test
	void closedRoadReroutesAlongTheRemainingDetour() {
		Set<Node> nodes = roads();
		nodes.remove(node(1, 0));
		assertEquals(List.of(node(0, 0), node(0, 1), node(1, 1), node(2, 1), node(2, 0)),
			RoadGraph.find(nodes, node(0, 0), node(2, 0)));
	}

	/** Closing both connecting roads returns no route rather than crossing a missing node. */
	@Test
	void closedRoadReportsNoRouteWhenTheDetourIsAlsoBlocked() {
		Set<Node> nodes = roads();
		nodes.remove(node(1, 0));
		nodes.remove(node(1, 1));
		assertEquals(List.of(), RoadGraph.find(nodes, node(0, 0), node(2, 0)));
	}

	/** Equal-length routes use the existing lexicographic node tie break. */
	@Test
	void equallyShortRoutesUseTheDeterministicNodeTieBreak() {
		Set<Node> nodes = Set.of(node(1, 0), node(1, 1), node(0, 1), node(0, 0));
		assertEquals(List.of(node(0, 0), node(0, 1), node(1, 1)), RoadGraph.find(nodes, node(0, 0), node(1, 1)));
	}

	/** Missing endpoints are unreachable while a present endpoint routes to itself. */
	@Test
	void endpointsMustBePresentEvenForAZeroLengthRoute() {
		Set<Node> nodes = Set.of(node(0, 0));
		assertEquals(List.of(node(0, 0)), RoadGraph.find(nodes, node(0, 0), node(0, 0)));
		assertEquals(List.of(), RoadGraph.find(nodes, node(0, 0), node(1, 0)));
		assertEquals(List.of(), RoadGraph.find(nodes, node(1, 0), node(0, 0)));
		assertEquals(List.of(), RoadGraph.find(nodes, node(1, 0), node(1, 0)));
	}

	/** Road search allows one-block height changes but rejects two-block jumps. */
	@Test
	void adjacentRoadStepsAllowOnlyOneBlockOfHeightChange() {
		Node start = new Node(0, 0, 0);
		Node step = new Node(1, 1, 0);
		Node end = new Node(2, 0, 0);
		assertEquals(List.of(start, step, end), RoadGraph.find(Set.of(start, step, end), start, end));
		assertEquals(List.of(), RoadGraph.find(Set.of(start, new Node(1, 2, 0), end), start, end));
	}

	private static Set<Node> roads() {
		return new TreeSet<>(Set.of(node(0, 0), node(1, 0), node(2, 0), node(0, 1), node(1, 1), node(2, 1)));
	}

	private static Node node(int blockX, int blockZ) {
		return new Node(blockX, 0, blockZ);
	}
}
