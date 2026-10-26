package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.UUID;
import net.minecraft.world.phys.Vec3;

/** One item the owning bot saw pop out of its mine, remembered for a short game-time window. */
record MinedDrop(UUID uuid, String item, int count, Vec3 position, long expiresAt) {

	/** Returns whether this knowledge still permits collection at the supplied game time. */
	boolean available(long now) {
		return now < expiresAt;
	}

	/** Returns the detached public progress record, without any hidden-world data. */
	JsonObject progress() {
		JsonObject result = new JsonObject();
		result.addProperty("uuid", uuid.toString());
		result.addProperty("item", item);
		result.addProperty("count", count);
		result.add("pos", HarnessState.vector(position));
		return result;
	}

	/** Saves the knowledge deadline along with the drop for deterministic snapshot restoration. */
	JsonObject serialization() {
		JsonObject result = progress();
		result.addProperty("expiresAt", expiresAt);
		return result;
	}

	/** Restores a record without refreshing its original deadline. */
	static MinedDrop restore(JsonObject saved) {
		var position = saved.getAsJsonArray("pos");
		return new MinedDrop(UUID.fromString(saved.get("uuid").getAsString()), saved.get("item").getAsString(),
			saved.get("count").getAsInt(),
			new Vec3(position.get(0).getAsDouble(), position.get(1).getAsDouble(), position.get(2).getAsDouble()),
			saved.get("expiresAt").getAsLong());
	}
}
