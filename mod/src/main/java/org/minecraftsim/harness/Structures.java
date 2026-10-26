package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.phys.Vec3;

/**
 * Named structures (e.g. bridge spans, a dock) that agents perceive as a whole.
 *
 * <p>A structure is a set of block positions and the block each should hold. Once any of its positions is within view
 * and in line of sight, an observation reports how many positions hold their block. A person looking at a bridge sees
 * whether it spans the river; a list of nearby notable blocks does not make that plain. The whole structure's state is
 * reported from a partial view, which is fair for the small structures used here.</p>
 *
 * <p>The structure set is canonical harness state (an optional map, absent until first set, so older state hashes are
 * unchanged). Observing reads block states and casts rays only: no RNG, no world change.</p>
 */
final class Structures {
	private static final String KEY = "structures";
	private final HarnessState state;

	/**
	 * A structure in sight: its observation item, the line-of-sight ray that admitted it, and the block that ray
	 * reached with its distance (what the exposure audit checks).
	 */
	record Seen(JsonObject item, Sight.Ray ray, BlockPos target, double targetDistance) {
	}

	/**
	 * Binds the controller to canonical harness state.
	 * @param state owning harness state
	 */
	Structures(HarnessState state) {
		this.state = state;
	}

	/**
	 * Replaces the known structures.
	 * @param args {@code {"structures": [{"name", "anchor": [x, y, z], "blocks": [{"pos": [x, y, z], "block"}]}]}}
	 * @return the number of structures stored
	 * @throws IllegalArgumentException for duplicate or blank names, empty block lists or malformed positions
	 */
	JsonObject setup(JsonObject args) {
		TreeMap<String, JsonObject> structures = new TreeMap<>();
		for (JsonElement value : args.getAsJsonArray("structures")) {
			JsonObject structure = value.getAsJsonObject();
			String name = structure.get("name").getAsString();
			JsonArray blocks = structure.getAsJsonArray("blocks");
			if (name.isBlank() || structures.containsKey(name) || blocks.isEmpty()) {
				throw new IllegalArgumentException("Structures need unique nonblank names and at least one block");
			}
			JsonArray normalized = new JsonArray();
			for (JsonElement element : blocks) {
				JsonObject entry = element.getAsJsonObject();
				JsonObject block = new JsonObject();
				block.add("pos", HarnessState.position(ControlCommands.position(entry)));
				block.addProperty("block", entry.get("block").getAsString());
				normalized.add(block);
			}
			JsonObject stored = new JsonObject();
			stored.add("anchor", HarnessState.position(ControlCommands.position(
				structure.has("anchor") ? wrap(structure.get("anchor")) : normalized.get(0).getAsJsonObject())));
			stored.add("blocks", normalized);
			structures.put(name, stored);
		}
		if (state.map(KEY) == null) {
			state.putMap(KEY);
		}
		JsonObject target = state.map(KEY);
		new ArrayList<>(target.keySet()).forEach(target::remove);
		structures.forEach(target::add);
		JsonObject result = new JsonObject();
		result.addProperty("structures", structures.size());
		return result;
	}

	/**
	 * Finds the structures an observer can see, with how many of their positions hold the right block.
	 * @param observer bot whose eye and level are used
	 * @param radius view radius in blocks
	 * @return seen structures in name order, each with its admitting ray
	 */
	List<Seen> seen(SimBot observer, double radius) {
		List<Seen> result = new ArrayList<>();
		JsonObject structures = state.map(KEY);
		if (structures == null) {
			return result;
		}
		ServerLevel level = observer.level();
		Vec3 eye = observer.getEyePosition();
		for (String name : new TreeSet<>(structures.keySet())) {
			JsonObject structure = structures.getAsJsonObject(name);
			JsonArray blocks = structure.getAsJsonArray("blocks");
			Sight.Ray ray = null;
			BlockPos target = null;
			double targetDistance = 0;
			double nearest = Double.MAX_VALUE;
			int built = 0;
			for (JsonElement element : blocks) {
				JsonObject entry = element.getAsJsonObject();
				BlockPos position = ControlCommands.position(entry);
				if (!level.hasChunkAt(position)) {
					continue;
				}
				if (HarnessState.blockId(level.getBlockState(position)).equals(entry.get("block").getAsString())) {
					built++;
				}
				double distance = eye.distanceTo(Vec3.atCenterOf(position));
				if (distance <= radius) {
					nearest = Math.min(nearest, distance);
					if (ray == null) {
						ray = Sight.block(observer, position);
						target = position;
						targetDistance = distance;
					}
				}
			}
			if (ray == null) {
				continue;
			}
			JsonObject item = new JsonObject();
			item.addProperty("prov", "structure:" + name);
			item.addProperty("name", name);
			item.add("pos", structure.get("anchor").deepCopy());
			item.addProperty("built", built);
			item.addProperty("total", blocks.size());
			item.addProperty("distance", nearest);
			result.add(new Seen(item, ray, target, targetDistance));
		}
		return result;
	}

	private static JsonObject wrap(JsonElement position) {
		JsonObject holder = new JsonObject();
		holder.add("pos", position.deepCopy());
		return holder;
	}
}
