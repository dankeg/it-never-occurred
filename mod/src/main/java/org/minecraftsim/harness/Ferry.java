package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.UUID;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.vehicle.boat.AbstractBoat;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.phys.Vec3;

/**
 * Owns the single kinematic ferry, its landing rules and exclusive rider reservation.
 * Registration and reservation live in canonical harness state; trip progress lives
 * in the rider's saved skill. No entity references or unsaved timers are retained.
 */
public final class Ferry {
	private static final int BOARD_TICKS = 20;
	private static final int LAND_TICKS = 20;
	private static final Vec3 RIDER_OFFSET = new Vec3(0, 0.6, 0);
	private final HarnessState harness;

	Ferry(HarnessState harness) {
		this.harness = harness;
	}

	/** Validates a one-time registration before storing any state or moving the boat. */
	JsonObject setup(JsonObject args) {
		if (data() != null) {
			throw new IllegalStateException("Ferry already configured");
		}
		UUID uuid = UUID.fromString(args.get("boat").getAsString());
		Entity entity = level().getEntity(uuid);
		if (!(entity instanceof AbstractBoat boat) || !boat.isAlive() || boat.isVehicle()) {
			throw new IllegalArgumentException("boat must identify a loaded, unoccupied boat");
		}
		double speed = args.get("speed").getAsDouble();
		if (!Double.isFinite(speed) || speed <= 0) {
			throw new IllegalArgumentException("speed must be finite and positive");
		}
		JsonObject definitions = args.getAsJsonObject("landings");
		if (definitions.size() < 2) {
			throw new IllegalArgumentException("At least two landings are required");
		}
		JsonObject landings = new JsonObject();
		String nearest = null;
		double nearestDistance = Double.POSITIVE_INFINITY;
		for (String name : definitions.keySet().stream().sorted().toList()) {
			if (name.isBlank()) {
				throw new IllegalArgumentException("Landing names must not be blank");
			}
			JsonObject definition = definitions.getAsJsonObject(name);
			BlockPos board = board(definition);
			Vec3 moor = vector(definition.getAsJsonArray("moor"));
			JsonObject landing = new JsonObject();
			landing.add("board", HarnessState.position(board));
			landing.add("moor", HarnessState.vector(moor));
			landings.add(name, landing);
			double distance = boat.position().distanceTo(moor);
			if (distance <= 2 && distance < nearestDistance) {
				nearest = name;
				nearestDistance = distance;
			}
		}
		if (nearest == null) {
			throw new IllegalArgumentException("Boat must be within two blocks of a landing moor");
		}
		JsonObject configured = new JsonObject();
		configured.addProperty("boat", uuid.toString());
		configured.addProperty("speed", speed);
		configured.add("landings", landings);
		configured.addProperty("landing", nearest);
		harness.putMap("ferry");
		configured.entrySet().forEach(entry -> data().add(entry.getKey(), entry.getValue()));
		positionBoat(boat, moor(nearest));
		return data().deepCopy();
	}

	/** Plans without reserving the boat, moving bodies, allocating IDs or drawing RNG. */
	JsonObject check(SimBot bot, JsonObject args) {
		JsonObject plan = new JsonObject();
		plan.addProperty("ticks", 0);
		if (data() == null || !args.has("to") || !landings().has(args.get("to").getAsString())) {
			return failed(plan, "target_not_found");
		}
		if (data().has("rider")) {
			return failed(plan, "busy");
		}
		String from = data().get("landing").getAsString();
		String to = args.get("to").getAsString();
		AbstractBoat boat = boat();
		if (boat == null || from.equals(to) || boat.position().distanceTo(moor(from)) > 2) {
			return failed(plan, "no_boat");
		}
		if (bot.position().distanceTo(feet(from)) > 2) {
			return failed(plan, "out_of_reach");
		}
		SkillPath.View view = new SkillPath.View(level());
		if (!SkillPath.standable(view, board(landing(from)).above())
				|| !SkillPath.standable(view, board(landing(to)).above())) {
			return failed(plan, "no_dock");
		}
		Vec3 origin = moor(from);
		Vec3 destination = moor(to);
		boolean water = water(view, BlockPos.containing(origin)) && water(view, BlockPos.containing(destination))
			&& BlockGetter.traverseBlocks(origin, destination, view,
			(world, position) -> water(world, position) ? null : Boolean.FALSE,
			world -> Boolean.TRUE);
		if (!water) {
			return failed(plan, "no_path");
		}
		double crossing = Math.ceil(origin.distanceTo(destination) / data().get("speed").getAsDouble());
		if (!Double.isFinite(crossing) || crossing >= Long.MAX_VALUE - BOARD_TICKS - LAND_TICKS) {
			throw new IllegalArgumentException("Ferry duration exceeds the skill tick range");
		}
		long travelTicks = (long) crossing;
		plan.addProperty("from", from);
		plan.addProperty("to", to);
		plan.addProperty("travelTicks", travelTicks);
		plan.addProperty("ticks", Math.addExact(travelTicks, BOARD_TICKS + LAND_TICKS));
		return plan;
	}

	/** Reserves the checked boat immediately, including the full boarding interval. */
	void start(SimBot bot, JsonObject plan) {
		data().addProperty("rider", bot.getGameProfile().name());
		positionBoat(boat(), moor(plan.get("from").getAsString()));
		positionRider(bot, feet(plan.get("from").getAsString()), true);
	}

	/** Advances the reserved skill once at tick end, after vanilla player ticks. */
	void tick() {
		if (data() != null && data().has("rider")) {
			String name = data().get("rider").getAsString();
			if (!(harness.server().getPlayerList().getPlayerByName(name) instanceof SimBot bot)) {
				throw new IllegalStateException("Ferry rider is not loaded: " + name + "; restore_bots before stepping");
			}
			bot.skills().tickFerry();
		}
	}

	/** Moves from saved elapsed ticks, avoiding accumulated floating-point drift. */
	String advance(SimBot bot, JsonObject progress, long startedAt) {
		AbstractBoat boat = boat();
		if (boat == null) {
			return "no_boat";
		}
		long elapsed = progress.get("elapsed").getAsLong();
		long travelTicks = progress.get("travelTicks").getAsLong();
		String from = progress.get("from").getAsString();
		String to = progress.get("to").getAsString();
		Vec3 origin = moor(from);
		Vec3 destination = moor(to);
		if (elapsed <= BOARD_TICKS) {
			positionBoat(boat, origin);
			positionRider(bot, feet(from), true);
		} else {
			long traveled = elapsed - BOARD_TICKS;
			Vec3 position = traveled >= travelTicks ? destination : origin.add(destination.subtract(origin)
				.scale(traveled * data().get("speed").getAsDouble() / origin.distanceTo(destination)));
			positionBoat(boat, position);
			positionRider(bot, position.add(RIDER_OFFSET), false);
		}
		if (elapsed < progress.get("estimate").getAsLong()) {
			return null;
		}
		boolean dock = SkillPath.standable(new SkillPath.View(level()), board(landing(to)).above());
		if (dock) {
			land(bot, to);
		} else {
			data().addProperty("landing", to);
		}
		JsonObject event = harness.event("ferry_trip");
		event.addProperty("rider", bot.getGameProfile().name());
		event.addProperty("from", from);
		event.addProperty("to", to);
		event.addProperty("startTick", startedAt);
		event.addProperty("endTick", level().getGameTime());
		event.addProperty("outcome", dock ? "succeeded" : "failed");
		if (!dock) {
			event.addProperty("reason", "no_dock");
		}
		harness.logEvent(event);
		return dock ? "done" : "no_dock";
	}

	/** Allows cancellation before departure, never shortening a crossing or landing. */
	boolean cancel(SimBot bot, JsonObject progress) {
		long elapsed = progress.get("elapsed").getAsLong();
		if (elapsed > BOARD_TICKS) {
			return false;
		}
		String landing = progress.get("from").getAsString();
		if (boat() == null || !SkillPath.standable(new SkillPath.View(level()), board(landing(landing)).above())) {
			return false;
		}
		land(bot, landing);
		return true;
	}

	/** Releases only this bot's reservation when its ferry skill ends. */
	void release(SimBot bot) {
		if (data() != null && data().has("rider") && data().get("rider").getAsString().equals(bot.getGameProfile().name())) {
			data().remove("rider");
		}
	}

	/** Returns the public phase derived from the same saved tick counters as movement. */
	String phase(JsonObject progress) {
		long elapsed = progress.get("elapsed").getAsLong();
		return elapsed <= BOARD_TICKS ? "boarding"
			: elapsed <= BOARD_TICKS + progress.get("travelTicks").getAsLong() ? "crossing" : "landing";
	}

	/**
	 * Identifies the registered vessel without initializing state or consuming RNG.
	 * @param entity the entity considered by vanilla boat physics and collision hooks
	 * @return whether this overworld entity is the configured ferry boat
	 */
	public static boolean isBoat(Entity entity) {
		if (!(entity.level() instanceof ServerLevel level) || level != level.getServer().overworld()) {
			return false;
		}
		HarnessState harness = HarnessState.existing(level.getServer());
		JsonObject ferry = harness == null ? null : harness.map("ferry");
		return ferry != null && ferry.get("boat").getAsString().equals(entity.getUUID().toString());
	}

	private void land(SimBot bot, String name) {
		positionBoat(boat(), moor(name));
		positionRider(bot, feet(name), true);
		data().addProperty("landing", name);
	}

	private static void positionBoat(AbstractBoat boat, Vec3 position) {
		boat.setPos(position);
		boat.setDeltaMovement(Vec3.ZERO);
		boat.fallDistance = 0;
	}

	private static void positionRider(SimBot bot, Vec3 position, boolean grounded) {
		bot.setInput(BotInput.idle(bot.getYRot(), bot.getXRot()));
		bot.setPos(position);
		bot.setDeltaMovement(Vec3.ZERO);
		bot.setOnGround(grounded);
		bot.fallDistance = 0;
	}

	private AbstractBoat boat() {
		Entity entity = level().getEntity(UUID.fromString(data().get("boat").getAsString()));
		return entity instanceof AbstractBoat boat && boat.isAlive() ? boat : null;
	}

	private JsonObject data() {
		return harness.map("ferry");
	}

	private ServerLevel level() {
		return harness.server().overworld();
	}

	private JsonObject landings() {
		return data().getAsJsonObject("landings");
	}

	private JsonObject landing(String name) {
		return landings().getAsJsonObject(name);
	}

	private Vec3 moor(String name) {
		return vector(landing(name).getAsJsonArray("moor"));
	}

	private Vec3 feet(String name) {
		return Vec3.atBottomCenterOf(board(landing(name)).above());
	}

	private static BlockPos board(JsonObject landing) {
		JsonObject args = new JsonObject();
		args.add("pos", landing.get("board"));
		return ControlCommands.position(args);
	}

	private static Vec3 vector(JsonArray array) {
		if (array.size() != 3) {
			throw new IllegalArgumentException("moor must contain three coordinates");
		}
		for (JsonElement coordinate : array) {
			if (!coordinate.isJsonPrimitive() || !coordinate.getAsJsonPrimitive().isNumber()
					|| !Double.isFinite(coordinate.getAsDouble()) || Math.abs(coordinate.getAsDouble()) > 30_000_000) {
				throw new IllegalArgumentException("Invalid moor coordinate");
			}
		}
		return new Vec3(array.get(0).getAsDouble(), array.get(1).getAsDouble(), array.get(2).getAsDouble());
	}

	private static JsonObject failed(JsonObject plan, String reason) {
		plan.addProperty("reason", reason);
		return plan;
	}

	private static boolean water(SkillPath.View view, BlockPos position) {
		return view.loaded(position) && view.getFluidState(position).is(FluidTags.WATER)
			&& view.getBlockState(position).getCollisionShape(view, position).isEmpty();
	}
}
