package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.Comparator;
import java.util.List;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;

/** Captures completed item actions in sight and persists each witness's pending, count-free evidence. */
public final class SeenActions {
	private final HarnessState state;

	/** Binds the controller without creating optional state or consuming simulation time. */
	SeenActions(HarnessState state) {
		this.state = state;
	}

	/**
	 * Queues only successful transfers whose actor and container are visible at completion, not at the next poll.
	 * @param actor role agent that completed the transfer
	 * @param action completed chest_take or chest_put skill
	 * @param position transferred container's block position
	 * @param transfer logged transfer event supplying its identity, item and game time
	 */
	void record(SimBot actor, String action, BlockPos position, JsonObject transfer) {
		queue(actor, null, position, fact(actor.getGameProfile().name(), action, position, transfer));
	}

	/** Queues a completed hand-over only for observers who see both participants at completion. */
	void record(SimBot actor, SimBot recipient, JsonObject transfer) {
		queue(actor, recipient, null, fact(actor.getGameProfile().name(), "give", null, transfer));
	}

	/** Queues a completed drop or vanilla pickup for observers who see its actor at completion. */
	void record(SimBot actor, JsonObject event) {
		queue(actor, null, null, fact(actor.getGameProfile().name(), event.get("type").getAsString(), null, event));
	}

	/** Shares deterministic witness admission while retaining the existing container sight evidence unchanged. */
	private void queue(SimBot actor, SimBot recipient, BlockPos position, JsonObject fact) {
		double radius = state.config().number("view_radius");
		state.server().getPlayerList().getPlayers().stream().filter(player -> player instanceof SimBot)
			.map(player -> (SimBot) player).sorted(Comparator.comparing(bot -> bot.getGameProfile().name()))
			.forEach(observer -> {
				if (observer == actor || !observer.isAlive() || observer.level() != actor.level()) {
					return;
				}
				Vec3 eye = observer.getEyePosition();
				double actorDistance = eye.distanceTo(actor.position());
				if (actorDistance > radius) {
					return;
				}
				Sight.Ray actorRay = Sight.entity(observer, actor);
				if (actorRay == null) {
					return;
				}
				JsonObject audit;
				if (position != null) {
					double distance = eye.distanceTo(Vec3.atCenterOf(position));
					if (distance > radius) {
						return;
					}
					Sight.Ray containerRay = Sight.block(observer, position);
					if (containerRay == null) {
						return;
					}
					audit = Observation.admitted(fact, "seen_action", radius, containerRay, false);
					audit.addProperty("distance", distance);
				} else {
					audit = fact.deepCopy();
					audit.addProperty("category", "seen_action");
					audit.addProperty("radius", radius);
					audit.addProperty("inRadius", true);
				}
				audit.add("observerEye", HarnessState.vector(eye));
				audit.add("actorSight", entitySight(actor, eye, radius, actorRay));
				if (recipient != null) {
					if (!recipient.isAlive() || recipient.level() != actor.level() || eye.distanceTo(recipient.position()) > radius) {
						return;
					}
					Sight.Ray targetRay = Sight.entity(observer, recipient);
					if (targetRay == null) {
						return;
					}
					audit.add("targetSight", entitySight(recipient, eye, radius, targetRay));
				}
				if (state.map("seenActions") == null) {
					state.putMap("seenActions");
				}
				JsonObject queues = state.map("seenActions");
				String name = observer.getGameProfile().name();
				if (!queues.has(name)) {
					queues.add(name, new JsonArray());
				}
				queues.getAsJsonArray(name).add(audit);
			});
	}

	/** Copies an admitted participant's completion-time geometry using the ordinary entity sight schema. */
	private static JsonObject entitySight(SimBot participant, Vec3 eye, double radius, Sight.Ray ray) {
		JsonObject sight = new JsonObject();
		sight.add("pos", HarnessState.vector(participant.position()));
		sight.addProperty("distance", eye.distanceTo(participant.position()));
		sight.add("targetEye", HarnessState.vector(participant.getEyePosition()));
		sight.add("targetCenter", HarnessState.vector(participant.getBoundingBox().getCenter()));
		return Observation.admitted(sight, "entity", radius, ray, false);
	}

	/** Projects a logged completion into visible facts without carrying over counts, notes or other arguments. */
	static JsonObject fact(String actor, String action, BlockPos position, JsonObject transfer) {
		if (!List.of("chest_take", "chest_put", "give", "drop", "pickup").contains(action)) {
			throw new IllegalArgumentException("Not an item action: " + action);
		}
		JsonObject result = new JsonObject();
		String kind = List.of("drop", "pickup").contains(action) ? action : "transfer";
		result.addProperty("prov", kind + ":" + transfer.get("id").getAsLong());
		result.add("gameTime", transfer.get("gameTime").deepCopy());
		result.addProperty("actor", actor);
		result.addProperty("action", action);
		if (action.equals("give")) {
			result.add("target", transfer.get("to").deepCopy());
		} else if (position != null) {
			result.add("pos", HarnessState.position(position));
		} else {
			result.add("pos", transfer.get("pos").deepCopy());
		}
		result.add("item", transfer.get("item").deepCopy());
		return result;
	}

	/** Reads pending evidence without creating state; callers drain it only after exposure logging succeeds. */
	JsonArray queued(String observer) {
		JsonObject queues = state.map("seenActions");
		return queues != null && queues.has(observer) ? queues.getAsJsonArray(observer) : new JsonArray();
	}

	/**
	 * Forgets every pending witnessed action, leaving no seenActions state (staging uses this, so role agents start
	 * with no perceptions of the scripted incident beyond what their scenario memory says).
	 * @return how many observers' queues were dropped
	 */
	JsonObject clearAll() {
		JsonObject queues = state.map("seenActions");
		JsonObject result = new JsonObject();
		result.addProperty("cleared", queues == null ? 0 : queues.size());
		state.removeMap("seenActions");
		return result;
	}

	/** Removes a delivered queue after its response and sight evidence have been logged. */
	void clear(String observer) {
		JsonObject queues = state.map("seenActions");
		if (queues != null) {
			queues.remove(observer);
		}
	}
}
