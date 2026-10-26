package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.UUID;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.item.Item;
import net.minecraft.world.phys.Vec3;

/** Immutable successful vanilla inventory insertion, captured before the item entity is discarded or reset. */
public record ItemPickup(Item item, int count, UUID itemEntity, Vec3 position, UUID thrower) {
	/** Rejects empty attempts; the count is the stack delta, not vanilla's original-stack animation count. */
	public ItemPickup {
		if (count <= 0) {
			throw new IllegalArgumentException("Pickup count must be positive");
		}
	}

	/** Logs this bot's actual pickup and queues its witnesses without initializing absent harness state. */
	public void record(SimBot bot) {
		HarnessState state = HarnessState.existing(bot.level().getServer());
		if (state == null) {
			return;
		}
		JsonObject event = state.event("pickup");
		details(bot.getGameProfile().name()).entrySet().forEach(entry -> event.add(entry.getKey(), entry.getValue()));
		state.logEvent(event);
		state.seenActions().record(bot, event);
	}

	/** Projects the pickup payload, retaining entity provenance and an unresolved thrower's UUID when present. */
	JsonObject details(String bot) {
		JsonObject result = new JsonObject();
		result.addProperty("bot", bot);
		result.addProperty("item", BuiltInRegistries.ITEM.getKey(item).toString());
		result.addProperty("count", count);
		result.addProperty("itemEntity", itemEntity.toString());
		result.add("pos", HarnessState.vector(position));
		result.addProperty("prov", "entity:" + itemEntity);
		if (thrower != null) {
			result.addProperty("thrower", thrower.toString());
		}
		return result;
	}
}
