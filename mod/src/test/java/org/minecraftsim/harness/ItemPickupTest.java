package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.gson.JsonObject;
import java.util.Set;
import java.util.UUID;
import net.minecraft.SharedConstants;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Verifies the pickup event's ground-truth payload without creating a world or starting a server. */
final class ItemPickupTest {
	private static final UUID ENTITY = UUID.fromString("00000000-0000-0000-0000-000000000001");
	private static final UUID THROWER = UUID.fromString("00000000-0000-0000-0000-000000000002");

	/** Initializes item identifiers only; this consumes no simulation RNG or ticks. */
	@BeforeAll
	static void bootstrap() {
		SharedConstants.tryDetectVersion();
		Bootstrap.bootStrap();
	}

	/** A partial insertion records exactly its accepted delta, source UUID, position and original thrower. */
	@Test
	void recordsActualPickupFieldsAndOptionalThrower() {
		ItemPickup pickup = new ItemPickup(Items.IRON_INGOT, 2, ENTITY, new Vec3(1.25, -60.5, 2.75), THROWER);
		JsonObject event = pickup.details("security");
		assertEquals(Set.of("bot", "item", "count", "itemEntity", "pos", "prov", "thrower"), event.keySet());
		assertEquals("security", event.get("bot").getAsString());
		assertEquals("minecraft:iron_ingot", event.get("item").getAsString());
		assertEquals(2, event.get("count").getAsInt());
		assertEquals(ENTITY.toString(), event.get("itemEntity").getAsString());
		assertEquals("entity:" + ENTITY, event.get("prov").getAsString());
		assertEquals(HarnessState.vector(pickup.position()), event.get("pos"));
		assertEquals(THROWER.toString(), event.get("thrower").getAsString());
		event.remove("count");
		assertEquals(2, pickup.details("security").get("count").getAsInt());
	}

	/** Naturally created items omit thrower rather than inventing an owner or requiring a live entity lookup. */
	@Test
	void unthrownItemsHaveNoThrower() {
		JsonObject event = new ItemPickup(Items.IRON_PICKAXE, 1, ENTITY, Vec3.ZERO, null).details("security");
		assertFalse(event.has("thrower"));
		assertEquals(Set.of("bot", "item", "count", "itemEntity", "pos", "prov"), event.keySet());
	}

	/** Empty or failed attempts cannot be represented as completed pickups. */
	@Test
	void rejectsNonpositiveDeltas() {
		for (int count : new int[]{0, -1}) {
			assertThrows(IllegalArgumentException.class,
				() -> new ItemPickup(Items.STONE, count, ENTITY, Vec3.ZERO, null));
		}
	}
}
