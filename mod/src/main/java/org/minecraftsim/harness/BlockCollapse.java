package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.Comparator;
import java.util.TreeSet;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;

/** One-shot, support-triggered removal of existing schematic sections, backed by optional fork state. */
final class BlockCollapse {
	private static final String KEY = "collapses";
	private static final Comparator<BlockPos> POSITION_ORDER = Comparator.<BlockPos>comparingInt(BlockPos::getX)
		.thenComparingInt(BlockPos::getY).thenComparingInt(BlockPos::getZ);
	private final HarnessState harness;
	private Long removingCause;

	/** Binds the controller without allocating canonical state until a rule is configured. */
	BlockCollapse(HarnessState harness) {
		this.harness = harness;
	}

	/** Whether the mixin must track support changes even when ordinary recording and NPCs are disabled. */
	boolean configured() {
		return harness.map(KEY) != null;
	}

	/** The triggering support event, only during synchronous recorded deck removal. */
	Long removingCause() {
		return removingCause;
	}

	/**
	 * Registers one named rule using absolute blocks already registered by {@code structures_set} from schematics.
	 * Validation completes before state is allocated; duplicate names cannot reset a persisted collapsed flag.
	 */
	JsonObject setup(JsonObject args) {
		String name = args.get("name").getAsString();
		JsonObject rules = harness.map(KEY);
		if (name.isBlank() || rules != null && rules.has(name)) {
			throw new IllegalArgumentException("Collapse needs a unique nonblank name");
		}
		TreeSet<BlockPos> supports = new TreeSet<>(POSITION_ORDER);
		for (JsonElement value : args.getAsJsonArray("supports")) {
			JsonObject wrapped = new JsonObject();
			wrapped.add("pos", value);
			BlockPos position = ControlCommands.position(wrapped);
			if (!supports.add(position) || !harness.server().overworld().hasChunkAt(position)
					|| !harness.server().overworld().getBlockState(position).isSolid()) {
				throw new IllegalArgumentException("Collapse supports must be distinct, loaded and solid");
			}
		}
		TreeSet<String> sections = new TreeSet<>();
		TreeSet<BlockPos> blocks = new TreeSet<>(POSITION_ORDER);
		JsonObject structures = harness.map("structures");
		for (JsonElement value : args.getAsJsonArray("sections")) {
			String section = value.getAsString();
			if (!sections.add(section) || structures == null || !structures.has(section)) {
				throw new IllegalArgumentException("Collapse needs distinct registered sections: " + section);
			}
			for (JsonElement entry : structures.getAsJsonObject(section).getAsJsonArray("blocks")) {
				BlockPos position = ControlCommands.position(entry.getAsJsonObject());
				if (!harness.server().overworld().hasChunkAt(position)) {
					throw new IllegalArgumentException("Unloaded collapse section: " + section);
				}
				blocks.add(position);
			}
		}
		if (supports.isEmpty() || sections.isEmpty() || blocks.isEmpty()) {
			throw new IllegalArgumentException("Collapse needs supports and nonempty sections");
		}
		JsonObject rule = new JsonObject();
		JsonArray supportPositions = new JsonArray();
		supports.forEach(position -> supportPositions.add(HarnessState.position(position)));
		rule.add("supports", supportPositions);
		JsonArray sectionNames = new JsonArray();
		sections.forEach(sectionNames::add);
		rule.add("sections", sectionNames);
		JsonArray positions = new JsonArray();
		blocks.forEach(position -> positions.add(HarnessState.position(position)));
		rule.add("blocks", positions);
		rule.addProperty("collapsed", false);
		if (rules == null) {
			harness.putMap(KEY);
		}
		harness.map(KEY).add(name, rule);
		return rule.deepCopy();
	}

	/** Remembers a real solid-to-nonsolid support transition; the last such change supplies causality. */
	void changed(BlockPos position, BlockState from, BlockState to, long eventId) {
		if (!configured() || !from.isSolid() || to.isSolid()) {
			return;
		}
		for (String name : new TreeSet<>(harness.map(KEY).keySet())) {
			JsonObject rule = harness.map(KEY).getAsJsonObject(name);
			if (!rule.get("collapsed").getAsBoolean()
					&& rule.getAsJsonArray("supports").contains(HarnessState.position(position))) {
				rule.addProperty("pendingCauseEventId", eventId);
			}
		}
	}

	/** Pure end-of-tick decision: a real support loss, no remaining solid support, and no earlier collapse. */
	static boolean shouldCollapse(boolean collapsed, boolean supportLost, boolean anySolid) {
		return !collapsed && supportLost && !anySolid;
	}

	/** Removes configured positions once, in x/y/z order, through vanilla setBlock and the recording mixin. */
	void tick() {
		if (!configured()) {
			return;
		}
		var level = harness.server().overworld();
		for (String name : new TreeSet<>(harness.map(KEY).keySet())) {
			JsonObject rule = harness.map(KEY).getAsJsonObject(name);
			boolean anySolid = false;
			for (JsonElement value : rule.getAsJsonArray("supports")) {
				JsonArray coordinates = value.getAsJsonArray();
				BlockPos position = new BlockPos(coordinates.get(0).getAsInt(), coordinates.get(1).getAsInt(),
					coordinates.get(2).getAsInt());
				anySolid |= !level.hasChunkAt(position) || level.getBlockState(position).isSolid();
			}
			if (!shouldCollapse(rule.get("collapsed").getAsBoolean(), rule.has("pendingCauseEventId"), anySolid)) {
				rule.remove("pendingCauseEventId");
				continue;
			}
			long cause = rule.remove("pendingCauseEventId").getAsLong();
			rule.addProperty("collapsed", true);
			JsonArray changes = new JsonArray();
			removingCause = cause;
			try {
				HarnessState.acting("world", () -> {
					for (JsonElement value : rule.getAsJsonArray("blocks")) {
						JsonArray coordinates = value.getAsJsonArray();
						BlockPos position = new BlockPos(coordinates.get(0).getAsInt(), coordinates.get(1).getAsInt(),
							coordinates.get(2).getAsInt());
						if (!level.hasChunkAt(position)) {
							throw new IllegalStateException("Unloaded collapse block: " + HarnessState.key(position));
						}
						if (!level.getBlockState(position).isAir()) {
							long before = harness.counter("eventCounter");
							if (!level.setBlock(position, Blocks.AIR.defaultBlockState(), 3)) {
								throw new IllegalStateException("Collapse removal failed: " + HarnessState.key(position));
							}
							JsonElement recorded = harness.map("latestChanges").get(HarnessState.key(position));
							if (recorded == null || recorded.getAsLong() <= before) {
								throw new IllegalStateException("Collapse removal was not recorded");
							}
							changes.add(recorded.deepCopy());
						}
					}
					return null;
				});
			} finally {
				removingCause = null;
			}
			JsonObject event = harness.event("collapse");
			event.addProperty("name", name);
			event.addProperty("actor", "world");
			event.addProperty("causeEventId", cause);
			event.add("changes", changes);
			harness.logEvent(event);
		}
	}
}
