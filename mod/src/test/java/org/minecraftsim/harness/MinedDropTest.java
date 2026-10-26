package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.UUID;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.Test;

/** Tests drop progress and saved knowledge without constructing a Minecraft world or inventing pickup physics. */
final class MinedDropTest {
	/** Creates a reproducible item record at the foot of the village's probe trunk. */
	private MinedDrop drop() {
		return new MinedDrop(UUID.fromString("00000000-0000-0000-0000-000000000001"), "minecraft:oak_log", 2,
			new Vec3(-108.75, -60, 8.25), 1200);
	}

	/** Public progress identifies the exact stack and position, not a presumed inventory gain. */
	@Test
	void progressDescribesUnpickedEntity() {
		JsonObject progress = drop().progress();
		assertEquals(4, progress.size());
		assertEquals(drop().uuid().toString(), progress.get("uuid").getAsString());
		assertEquals("minecraft:oak_log", progress.get("item").getAsString());
		assertEquals(2, progress.get("count").getAsInt());
		assertEquals(JsonParser.parseString("[-108.75,-60.0,8.25]"), progress.get("pos"));
		assertFalse(progress.has("expiresAt"));
	}

	/** Changing a returned JSON record cannot mutate another status read or the saved deadline. */
	@Test
	void progressIsDetachedFromSavedKnowledge() {
		JsonObject progress = drop().progress();
		progress.addProperty("count", 99);
		progress.getAsJsonArray("pos").set(0, JsonParser.parseString("0"));
		assertEquals(2, drop().progress().get("count").getAsInt());
		assertEquals(-108.75, drop().progress().getAsJsonArray("pos").get(0).getAsDouble());
	}

	/** Knowledge ends exactly 200 ticks after a mine at tick 1000 and never refreshes on restore. */
	@Test
	void restoredKnowledgeExpiresAtOriginalDeadline() {
		MinedDrop restored = MinedDrop.restore(drop().serialization());
		assertEquals(drop(), restored);
		assertTrue(restored.available(1000));
		assertTrue(restored.available(1199));
		assertFalse(restored.available(1200));
		assertFalse(restored.available(1201));
	}

	/** Storage-only controller restoration preserves knowledge across other skills and keeps legacy snapshots valid. */
	@Test
	void controllerRoundTripsKnowledgeAndClearsItOnLegacyRestore() {
		Skills skills = new Skills(null);
		String legacy = skills.serialization();
		JsonObject saved = JsonParser.parseString(legacy).getAsJsonObject();
		JsonArray drops = new JsonArray();
		drops.add(drop().serialization());
		saved.add("minedDrops", drops);
		skills.restore(saved.toString());
		assertEquals(saved, JsonParser.parseString(skills.serialization()));
		skills.restore(legacy);
		assertEquals(legacy, skills.serialization());
	}

	/** A stack on a one-block plinth can be picked up from a lower adjacent cell, but not arbitrary heights. */
	@Test
	void pickupEnvelopeIncludesLowerAdjacentStandingCells() {
		Vec3 position = new Vec3(2.5, 65, 0.5);
		AABB body = new AABB(2.2, 65, 0.2, 2.8, 66.8, 0.8);
		AABB item = new AABB(0.375, 64, 0.375, 0.625, 64.25, 0.625);
		assertTrue(Skills.pickupAt(body, position, new BlockPos(1, 63, 0), item));
		assertFalse(Skills.pickupAt(body, position, new BlockPos(1, 65, 0), item));
		assertFalse(Skills.pickupAt(body, position, new BlockPos(3, 63, 0), item));
		assertFalse(Skills.pickupAt(body, position, new BlockPos(1, 60, 0), item));
	}
}
