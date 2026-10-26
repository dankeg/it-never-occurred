package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.google.gson.JsonObject;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Pins the observed state contract shared by door rendering and grounded movement-failure hints. */
final class ObservationTest {
	/** Initializes vanilla block states without starting a server. */
	@BeforeAll
	static void bootstrapRegistries() {
		TestRegistries.bootstrap();
	}

	/** Every supported control exposes a boolean OPEN value, not an inferred default or a state string. */
	@Test
	void visibleDoorsTrapdoorsAndGatesExposeTheirActualOpenState() {
		for (Block block : new Block[]{Blocks.OAK_DOOR, Blocks.IRON_DOOR, Blocks.OAK_TRAPDOOR,
				Blocks.IRON_TRAPDOOR, Blocks.OAK_FENCE_GATE}) {
			for (boolean open : new boolean[]{false, true}) {
				JsonObject item = new JsonObject();
				item.addProperty("block", "existing block id");
				Observation.describeOpenState(item, block.defaultBlockState().setValue(BlockStateProperties.OPEN, open));
				assertEquals(open, item.getAsJsonObject("properties").get("open").getAsBoolean());
				assertEquals(1, item.getAsJsonObject("properties").size());
				assertEquals("existing block id", item.get("block").getAsString());
			}
		}
	}

	/** Ordinary notable blocks retain their established shape, even a barrel with its own OPEN property. */
	@Test
	void unrelatedBlocksGainNoDoorProperties() {
		for (Block block : new Block[]{Blocks.STONE, Blocks.BELL, Blocks.CHEST, Blocks.BARREL}) {
			JsonObject item = new JsonObject();
			Observation.describeOpenState(item, block.defaultBlockState());
			assertFalse(item.has("properties"));
		}
	}
}
