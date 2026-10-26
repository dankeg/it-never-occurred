package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.Container;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.RandomizableContainer;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.BellBlock;
import net.minecraft.world.level.block.ButtonBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.LeverBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BellAttachType;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.level.storage.LevelResource;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.CollisionContext;

/**
 * Owns one bot's active skill, feasibility checks, tick execution, and saved progress.
 * Dry runs and real starts share {@link #feasibility}; all world access runs on the
 * server thread. Progress and IDs use the persisted sequence and overworld game time,
 * never wall time or the server tick count. Skill-specific checks and tick methods
 * remain together so they share the same bot, configuration, and saved JSON layout.
 */
public final class Skills {
	private static final Direction[] SUPPORTS = {Direction.DOWN, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Direction.UP};
	private static final long MINED_DROP_TICKS = 200;
	private final SimBot bot;
	private final Map<UUID, MinedDrop> minedDrops = new HashMap<>();
	private long sequence;
	private JsonObject active;
	private JsonObject last;

	/**
	 * Creates the skill controller for a bot without starting a skill.
	 * @param bot the bot whose inventory, input, and world the skills use
	 */
	public Skills(SimBot bot) {
		this.bot = bot;
	}

	/**
	 * Returns canonical saved state, including internal progress and the ID sequence.
	 * @return the JSON stored in bot persistence and included in harness hashing
	 */
	public String serialization() {
		JsonObject saved = new JsonObject();
		saved.addProperty("sequence", sequence);
		saved.add("active", active == null ? JsonNull.INSTANCE : active.deepCopy());
		saved.add("last", last == null ? JsonNull.INSTANCE : last.deepCopy());
		if (!minedDrops.isEmpty()) {
			JsonArray drops = new JsonArray();
			minedDrops.values().stream().sorted(Comparator.comparing(drop -> drop.uuid().toString()))
				.forEach(drop -> drops.add(drop.serialization()));
			saved.add("minedDrops", drops);
		}
		return HarnessState.canonical(saved);
	}

	/**
	 * Restores saved progress without advancing the skill or consuming RNG.
	 * @param serialized the saved skill JSON; absent fields retain the legacy defaults
	 * @throws RuntimeException if the JSON or its field types are invalid
	 */
	public void restore(String serialized) {
		JsonObject saved = JsonParser.parseString(serialized).getAsJsonObject();
		sequence = saved.has("sequence") ? saved.get("sequence").getAsLong() : 0;
		active = saved.has("active") && saved.get("active").isJsonObject() ? saved.getAsJsonObject("active").deepCopy() : null;
		last = saved.has("last") && saved.get("last").isJsonObject() ? saved.getAsJsonObject("last").deepCopy() : null;
		minedDrops.clear();
		if (saved.has("minedDrops")) {
			for (JsonElement element : saved.getAsJsonArray("minedDrops")) {
				MinedDrop drop = MinedDrop.restore(element.getAsJsonObject());
				minedDrops.put(drop.uuid(), drop);
			}
		}
	}

	/**
	 * Returns public active, last-completed, or idle status without internal progress.
	 * @return a detached status object; reading it does not advance the skill
	 */
	public JsonObject status() {
		if (active != null) {
			JsonObject result = active.deepCopy();
			result.remove("internal");
			return dropStatus(result);
		}
		if (last != null) {
			return dropStatus(last.deepCopy());
		}
		JsonObject idle = new JsonObject();
		idle.add("skillId", JsonNull.INSTANCE);
		idle.add("skill", JsonNull.INSTANCE);
		idle.add("args", new JsonObject());
		idle.addProperty("state", "idle");
		idle.add("progress", new JsonObject());
		idle.add("startedAt", JsonNull.INSTANCE);
		return idle;
	}

	/** Whether this bot is running `collect`, the only way it takes back an item it dropped itself. */
	public boolean collecting() {
		return active != null && "collect".equals(active.get("skill").getAsString());
	}

	/**
	 * Describes only the running skill and its physical block target, never private arguments or progress.
	 * Observers must separately check sight of the bot and of any returned target. Speech is not an activity:
	 * who perceives it is decided by Communication's talk, chatter and shout radii.
	 * @return detached public activity, or null when no skill or only speech is running
	 */
	public JsonObject activity() {
		if (active == null) {
			return null;
		}
		String skill = active.get("skill").getAsString();
		if (List.of("speak", "ask").contains(skill)) {
			return null;
		}
		JsonObject result = new JsonObject();
		result.addProperty("skill", skill.equals("place_section") ? "build_section" : skill);
		if (List.of("mine", "place", "use", "chest_put", "chest_take", "buy").contains(skill)) {
			result.add("pos", HarnessState.position(ControlCommands.position(active.getAsJsonObject("args"))));
		}
		if (skill.equals("bake")) {
			result.add("pos", internal().get("pos").deepCopy());
		}
		return result;
	}

	/** Refreshes only this mine's surviving drops, omitting items already picked up or no longer remembered. */
	private JsonObject dropStatus(JsonObject result) {
		JsonObject progress = result.getAsJsonObject("progress");
		if (progress != null && result.get("skill").getAsString().equals("mine") && progress.has("drops")) {
			JsonArray remaining = new JsonArray();
			for (JsonElement element : progress.getAsJsonArray("drops")) {
				UUID uuid = UUID.fromString(element.getAsJsonObject().get("uuid").getAsString());
				MinedDrop known = minedDrops.get(uuid);
				if (known != null && known.available(time()) && bot.level().getEntity(uuid) instanceof ItemEntity entity
						&& entity.isAlive() && !entity.getItem().isEmpty()) {
					remaining.add(minedDrop(entity, known.expiresAt()).progress());
				}
			}
			progress.remove("drops");
			if (!remaining.isEmpty()) {
				progress.add("drops", remaining);
			}
		}
		return result;
	}

	/** Describes a real item entity without changing it or consuming world randomness. */
	private static MinedDrop minedDrop(ItemEntity entity, long expiresAt) {
		return new MinedDrop(entity.getUUID(), BuiltInRegistries.ITEM.getKey(entity.getItem().getItem()).toString(),
			entity.getItem().getCount(), entity.position(), expiresAt);
	}

	/**
	 * Checks feasibility and optionally starts a skill using that same check result.
	 * A dry run neither interrupts the active skill nor allocates a skill ID.
	 * @param skill the protocol skill name
	 * @param args the skill arguments, copied when a real start is accepted
	 * @param dryRun whether to return only feasibility and an estimate
	 * @return the existing feasibility or acceptance response, including a failure reason
	 * @throws IllegalArgumentException for invalid arguments, skills, or dimensions
	 * @throws IllegalStateException if the bot has infinite materials
	 */
	public JsonObject start(String skill, JsonObject args, boolean dryRun) {
		if (bot.level() != bot.level().getServer().overworld()) {
			throw new IllegalArgumentException("Skills are limited to the overworld");
		}
		if (args == null) {
			throw new IllegalArgumentException("args must be an object");
		}
		if (bot.hasInfiniteMaterials()) {
			throw new IllegalStateException("Skills require a bot without infinite materials");
		}
		Check checked = ferryRunning() || (!skill.equals("speak") && state().communication().busy(bot.getGameProfile().name()))
			? failed("busy") : feasibility(skill, args);
		JsonObject result = new JsonObject();
		result.addProperty(dryRun ? "feasible" : "accepted", checked.reason() == null);
		if (checked.reason() != null) {
			result.addProperty("reason", checked.reason());
			if (checked.reason().equals("too_heavy")) {
				checked.data().entrySet().forEach(entry -> result.add(entry.getKey(), entry.getValue().deepCopy()));
			}
		}
		if (dryRun) {
			JsonObject estimate = new JsonObject();
			estimate.addProperty("ticks", checked.ticks());
			estimate.add("materials", checked.materials());
			result.add("estimate", estimate);
		} else if (checked.reason() == null) {
			if (active != null) {
				finish("cancelled", "interrupted");
			}
			active = new JsonObject();
			sequence = Math.incrementExact(sequence);
			String id = bot.getGameProfile().name() + ":" + sequence;
			active.addProperty("skillId", id);
			active.addProperty("skill", skill);
			active.add("args", args.deepCopy());
			active.addProperty("state", "running");
			active.addProperty("startedAt", time());
			active.add("progress", new JsonObject());
			JsonObject internal = checked.data().deepCopy();
			internal.addProperty("elapsed", 0);
			internal.addProperty("estimate", checked.ticks());
			active.add("internal", internal);
			if (skill.equals("bake")) {
				ovenLit(position(internal.getAsJsonArray("pos")), true);
			}
			if (skill.equals("place_section")) {
				progress().addProperty("placed", 0);
				progress().addProperty("total", internal.getAsJsonArray("blocks").size());
			}
			result.addProperty("skillId", id);
			if (skill.equals("ferry")) {
				state().ferry().start(bot, internal);
				progress().addProperty("ticks", 0);
				progress().addProperty("phase", "boarding");
			}
			if (skill.equals("speak")) {
				JsonObject utterance = state().communication().startBot(bot, checked.data());
				progress().add("uttId", utterance.get("uttId"));
				result.add("uttId", utterance.get("uttId"));
				clearInput();
				if (checked.ticks() == 0) {
					state().communication().tick();
				}
			}
			if (skill.equals("hold")) {
				tickHold(args);
			}
		}
		return result;
	}

	/**
	 * Cancels the current skill as interrupted, except a ferry after departure.
	 * @return public status, with accepted=false and reason=busy when disembarking is unsafe
	 */
	public JsonObject cancel() {
		if (ferryRunning() && !state().ferry().cancel(bot, internal())) {
			JsonObject result = status();
			result.addProperty("accepted", false);
			result.addProperty("reason", "busy");
			return result;
		}
		if (active != null) {
			finish("cancelled", "interrupted");
		}
		return status();
	}

	/** Tests the persisted active skill without creating state or advancing progress. */
	boolean ferryRunning() {
		return active != null && active.get("skill").getAsString().equals("ferry");
	}

	private Check feasibility(String skill, JsonObject args) {
		if (List.of("move_to", "collect", "place_section").contains(skill) && bot.isInWater()
				&& SkillPath.swimLeg(new SkillPath.View(bot.level()), feet()) == null) {
			return failed("in_water_no_bank");
		}
		return switch (skill) {
			case "speak" -> checkSpeech(args);
			case "ferry" -> checkFerry(args);
			case "move_to" -> checkMove(args);
			case "mine" -> checkMine(args);
			case "use" -> checkUse(args);
			case "drop" -> checkDrop(args);
			case "hold" -> SkillInventory.hold(bot.getInventory(), item(args), bot.getInventory().getSelectedSlot(), false) < 0
				? failed("missing_items") : ok(0, new JsonObject());
			case "eat" -> checkEat(args);
			case "bake" -> checkBake(args);
			case "collect" -> checkCollect(args);
			case "craft" -> checkCraft(args);
			case "place" -> checkPlace(args, new SkillPath.View(bot.level()), bot.getEyePosition(), true);
			case "place_section" -> checkSection(args);
			case "chest_put", "chest_take", "give" -> checkTransfer(skill, args);
			case "buy" -> checkBuy(args);
			case "wait" -> ok(integer(args, "ticks", 0, Integer.MAX_VALUE), new JsonObject());
			default -> throw new IllegalArgumentException("Unknown skill: " + skill);
		};
	}

	/**
	 * Checks a component-preserving, single-entity drop without changing inventory or allocating an entity.
	 * @param args requested item identifier and positive count
	 * @return the configured duration, or the inventory staging failure
	 * @throws IllegalArgumentException if the item or count is invalid
	 */
	private Check checkDrop(JsonObject args) {
		SkillDrop drop = SkillDrop.plan(bot.getInventory(), item(args), integer(args, "count", 1, Integer.MAX_VALUE));
		return drop.reason() == null ? ok(config().integer("drop_ticks"), new JsonObject()) : failed(drop.reason());
	}

	/** Checks one held food item without consuming it, including its configured eating duration. */
	private Check checkEat(JsonObject args) {
		SkillEat checked = SkillEat.plan(bot.getInventory(), item(args), config().freshDays(), Math.floorDiv(time(), 24_000L));
		return checked.reason() == null ? ok(config().integer("eat_ticks"), new JsonObject()) : failed(checked.reason());
	}

	private Check checkSpeech(JsonObject args) {
		JsonObject plan = state().communication().check(bot, args);
		return new Check(plan.has("reason") ? plan.get("reason").getAsString() : null,
			plan.get("ticks").getAsLong(), new JsonObject(), plan);
	}

	private Check checkFerry(JsonObject args) {
		JsonObject plan = state().ferry().check(bot, args);
		return new Check(plan.has("reason") ? plan.get("reason").getAsString() : null,
			plan.get("ticks").getAsLong(), new JsonObject(), plan);
	}

	private Check checkCraft(JsonObject args) {
		JsonObject plan = SkillCrafting.plan(bot, args, config().integer("craft_ticks"), tableAvailable());
		return new Check(plan.has("reason") ? plan.get("reason").getAsString() : null,
			plan.has("ticks") ? plan.get("ticks").getAsLong() : 0,
			plan.has("materials") ? plan.getAsJsonObject("materials") : new JsonObject(), plan);
	}

	/** Selects a reachable visible oven in stable coordinate order and checks the entire batch before lighting it. */
	private Check checkBake(JsonObject args) {
		int count = integer(args, "count", 1, 32);
		if (!config().bakingEnabled()) {
			return failed("no_oven");
		}
		BlockPos oven = null;
		int radius = (int) Math.ceil(config().number("reach"));
		BlockPos center = BlockPos.containing(bot.getEyePosition());
		for (int deltaX = -radius; deltaX <= radius && oven == null; deltaX++) {
			for (int deltaY = -radius; deltaY <= radius && oven == null; deltaY++) {
				for (int deltaZ = -radius; deltaZ <= radius; deltaZ++) {
					BlockPos candidate = center.offset(deltaX, deltaY, deltaZ);
					boolean ownOven = active != null && active.get("skill").getAsString().equals("bake")
						&& candidate.equals(position(internal().getAsJsonArray("pos")));
					if (bot.level().hasChunkAt(candidate) && bot.level().getBlockState(candidate).is(Blocks.SMOKER)
							&& (!bot.level().getBlockState(candidate).getValue(BlockStateProperties.LIT) || ownOven) && spatial(candidate) == null
							&& Documents.permitted(state(), bot, candidate)) {
						oven = candidate;
						break;
					}
				}
			}
		}
		if (oven == null) {
			return failed("no_oven");
		}
		SkillBake plan = SkillBake.plan(bot.getInventory(), count, config(), Math.floorDiv(time(), 24_000L));
		if (plan.reason() != null) {
			return new Check(plan.reason(), 0, new JsonObject(), plan.details());
		}
		JsonObject data = new JsonObject();
		data.add("pos", HarnessState.position(oven));
		JsonObject materials = new JsonObject();
		materials.addProperty("minecraft:wheat", count * 3);
		return new Check(null, plan.ticks(), materials, data);
	}

	/** Keeps the visible oven lit during its saved timer and commits only the fully rechecked batch. */
	private void tickBake(JsonObject args) {
		BlockPos position = position(internal().getAsJsonArray("pos"));
		if (!config().bakingEnabled() || !bot.level().hasChunkAt(position)
				|| !bot.level().getBlockState(position).is(Blocks.SMOKER)) {
			finish("failed", "no_oven");
			return;
		}
		String spatial = spatial(position);
		if (spatial != null || !Documents.permitted(state(), bot, position)) {
			finish("failed", spatial != null ? spatial : "acl_denied");
			return;
		}
		ovenLit(position, true);
		look(Vec3.atCenterOf(position), 0, false, false);
		progress().add("ticks", internal().get("elapsed"));
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		SkillBake plan = SkillBake.plan(bot.getInventory(), args.get("count").getAsInt(), config(), Math.floorDiv(time(), 24_000L));
		if (plan.reason() != null) {
			plan.details().entrySet().forEach(entry -> progress().add(entry.getKey(), entry.getValue()));
			finish("failed", plan.reason());
			return;
		}
		plan.commit(bot.getInventory());
		progress().add("count", args.get("count"));
		finish("succeeded", null);
	}

	/** Changes only a surviving smoker's light property, including cancellation and failed completion. */
	private void ovenLit(BlockPos position, boolean lit) {
		if (bot.level().hasChunkAt(position)) {
			BlockState current = bot.level().getBlockState(position);
			if (current.is(Blocks.SMOKER) && current.getValue(BlockStateProperties.LIT) != lit) {
				bot.level().setBlock(position, SkillBake.light(current, lit), Block.UPDATE_ALL);
			}
		}
	}

	private Check checkMove(JsonObject args) {
		Vec3 target = target(args);
		number(args, "tolerance", 0.5, 0, 256);
		SkillPath.View view = new SkillPath.View(bot.level());
		BlockPos goal = approach(view, BlockPos.containing(target));
		if (goal == null) {
			return failed("no_path");
		}
		if (!goal.equals(BlockPos.containing(target))) {
			target = Vec3.atBottomCenterOf(goal);
		}
		SkillPath.Route route = SkillPath.find(view, feet(), goal);
		return route == null ? failed("no_path") : ok(route.ticks(), movement(route, target, args.has("tolerance") ? args.get("tolerance").getAsDouble() : 0.5));
	}

	/**
	 * Returns the target itself if a bot can stand there. If the target is occupied by a solid object (a chest, a
	 * barrel, a tree), returns the nearest standable block within two blocks horizontally and one vertically, so "walk
	 * to the chest" walks up to it. Water, lava and unsupported air are not approached: walking there is infeasible.
	 * Ties break by height difference, then by position, so the choice is deterministic. Returns null if none applies.
	 */
	private static BlockPos approach(SkillPath.View view, BlockPos target) {
		if (SkillPath.standable(view, target)) {
			return target;
		}
		if (!view.loaded(target) || !view.getFluidState(target).isEmpty()
				|| view.getBlockState(target).getCollisionShape(view, target).isEmpty()) {
			return null;
		}
		List<BlockPos> candidates = new ArrayList<>();
		for (int dx = -2; dx <= 2; dx++) {
			for (int dz = -2; dz <= 2; dz++) {
				for (int dy = -1; dy <= 1; dy++) {
					BlockPos candidate = target.offset(dx, dy, dz);
					if ((dx != 0 || dz != 0 || dy != 0) && SkillPath.standable(view, candidate)) {
						candidates.add(candidate);
					}
				}
			}
		}
		candidates.sort(Comparator.comparingDouble((BlockPos c) -> Math.hypot(c.getX() - target.getX(), c.getZ() - target.getZ()))
			.thenComparingInt(c -> Math.abs(c.getY() - target.getY())).thenComparing(SkillPath.POSITIONS));
		return candidates.isEmpty() ? null : candidates.get(0);
	}

	/** The block the bot stands in, robust to surfaces slightly lower than a full block (dirt paths, farmland). */
	private BlockPos feet() {
		return BlockPos.containing(bot.getX(), bot.getY() + 0.5, bot.getZ());
	}

	private Check checkMine(JsonObject args) {
		BlockPos position = ControlCommands.position(args);
		if (!bot.level().hasChunkAt(position)) {
			return failed("target_not_found");
		}
		BlockState state = bot.level().getBlockState(position);
		if (state.isAir()) {
			return failed("target_not_found");
		}
		if (state.getDestroySpeed(bot.level(), position) < 0) {
			return failed("unbreakable");
		}
		String spatial = spatial(position);
		if (spatial != null) {
			return failed(spatial);
		}
		if (!Documents.permitted(state(), bot, position)) {
			return failed("acl_denied");
		}
		float speed = state.getDestroyProgress(bot, bot.level(), position);
		if (!(speed > 0)) {
			return failed("unbreakable");
		}
		JsonObject data = new JsonObject();
		data.addProperty("blockState", Block.getId(state));
		data.addProperty("destroyProgress", 0.0F);
		return ok((long) Math.ceil(1.0 / speed), data);
	}

	/** Checks only supported, visible, reachable block interactions, without invoking vanilla or changing state. */
	private Check checkUse(JsonObject args) {
		BlockPos position = ControlCommands.position(args);
		String reason = spatial(position);
		if (reason != null) {
			return failed(reason);
		}
		BlockState block = bot.level().getBlockState(position);
		if (!usable(block)) {
			return failed("not_usable");
		}
		if (!Documents.permitted(state(), bot, position)) {
			return failed("acl_denied");
		}
		JsonObject data = new JsonObject();
		data.addProperty("blockState", Block.getId(block));
		return ok(config().integer("use_ticks"), data);
	}

	/** Limits generic use to the sprint's physical controls; containers, beds and workstations remain separate. */
	static boolean usable(BlockState state) {
		Block block = state.getBlock();
		if (block instanceof DoorBlock door) {
			return door.type().canOpenByHand();
		}
		return block instanceof TrapDoorBlock && !state.is(Blocks.IRON_TRAPDOOR) || block instanceof FenceGateBlock
			|| block instanceof BellBlock || block instanceof LeverBlock || block instanceof ButtonBlock;
	}

	/** Selects a deterministic valid strike face for each bell mount, leaving other vanilla hit results intact. */
	static BlockHitResult useHit(BlockState state, BlockPos position, BlockHitResult sight) {
		if (!(state.getBlock() instanceof BellBlock)) {
			return sight;
		}
		Direction face = state.getValue(BellBlock.FACING);
		BellAttachType attachment = state.getValue(BellBlock.ATTACHMENT);
		if (attachment == BellAttachType.SINGLE_WALL || attachment == BellAttachType.DOUBLE_WALL) {
			face = face.getClockWise();
		}
		return new BlockHitResult(Vec3.atCenterOf(position), face, position, false);
	}

	private Check checkCollect(JsonObject args) {
		double radius = number(args, "radius", 6.0, 0, 256);
		List<ItemEntity> items = items(radius);
		if (items.isEmpty()) {
			return failed("target_not_found");
		}
		ItemEntity nearest = items.getFirst();
		JsonObject heavy = CarryLimits.check(bot.getInventory(), nearest.getItem().getItem(), nearest.getItem().getCount(), config());
		if (heavy != null) {
			return new Check("too_heavy", 0, new JsonObject(), heavy);
		}
		if (!SkillInventory.insert(bot.getInventory(), SkillInventory.copy(bot.getInventory()), nearest.getItem())) {
			return failed("inventory_full");
		}
		SkillPath.Route route = routeToItem(nearest);
		return route == null ? failed("no_path") : ok(Math.min(600, route.ticks()), new JsonObject());
	}

	private Check checkPlace(JsonObject args, SkillPath.View view, Vec3 eye, boolean inventory) {
		BlockPos position = ControlCommands.position(args);
		Item item = item(args);
		if (!(item instanceof BlockItem blockItem)) {
			return failed("target_not_found");
		}
		if (!view.loaded(position)) {
			return failed("target_not_found");
		}
		BlockState current = view.getBlockState(position);
		if (!current.isAir() && !current.canBeReplaced()) {
			return failed("target_occupied");
		}
		if (eye.distanceTo(Vec3.atCenterOf(position)) > config().number("reach")) {
			return failed("out_of_reach");
		}
		if (inventory && SkillInventory.count(bot.getInventory(), item) == 0) {
			return failed("missing_items");
		}
		if (!Documents.permitted(state(), bot, position)) {
			return failed("acl_denied");
		}
		boolean supported = false;
		for (Direction offset : SUPPORTS) {
			BlockPos support = position.relative(offset);
			Direction face = offset.getOpposite();
			if (!view.loaded(support) || view.getBlockState(support).getCollisionShape(view, support).isEmpty()
					|| !view.getBlockState(support).isFaceSturdy(view, support, face)) {
				continue;
			}
			supported = true;
			BlockPos feet = inventory ? feet() : BlockPos.containing(eye.add(0, -bot.getEyeHeight(), 0));
			if (!Sight.face(view, eye, feet, support, face)) {
				continue;
			}
			if (inventory) {
				Vec3 hit = Vec3.atCenterOf(support).add(face.getStepX() * 0.5, face.getStepY() * 0.5, face.getStepZ() * 0.5);
				ItemStack stack = ItemStack.EMPTY;
				for (int slot = 0; slot < bot.getInventory().getContainerSize(); slot++) {
					if (bot.getInventory().getItem(slot).is(item)) {
						stack = bot.getInventory().getItem(slot).copy();
						break;
					}
				}
				BlockPlaceContext context = new BlockPlaceContext(bot.level(), bot, InteractionHand.MAIN_HAND, stack,
					new BlockHitResult(hit, face, support, false));
				context = blockItem.updatePlacementContext(context);
				if (context == null || !context.getClickedPos().equals(position) || !context.canPlace()) {
					return failed("target_occupied");
				}
				BlockState placement = blockItem.getBlock().getStateForPlacement(context);
				if (placement == null || !placement.canSurvive(bot.level(), position)) {
					return failed("no_support");
				}
				if (!bot.level().isUnobstructed(placement, position, CollisionContext.placementContext(bot))) {
					return failed("target_occupied");
				}
			}
			JsonObject data = new JsonObject();
			data.add("support", HarnessState.position(support));
			data.addProperty("face", face.name());
			data.addProperty("blockState", Block.getId(current));
			data.addProperty("expected", BuiltInRegistries.BLOCK.getKey(blockItem.getBlock()).toString());
			JsonObject bill = new JsonObject();
			bill.addProperty(BuiltInRegistries.ITEM.getKey(item).toString(), 1);
			return new Check(null, config().integer("place_ticks"), bill, data);
		}
		return failed(supported ? "not_visible" : "no_support");
	}

	private Check checkTransfer(String skill, JsonObject args) {
		Item item = item(args);
		int count = integer(args, "count", 1, Integer.MAX_VALUE);
		Container from = bot.getInventory();
		Container to;
		if (skill.equals("give")) {
			SimBot other = other(args.get("target").getAsString());
			if (other == null || other == bot || other.level() != bot.level()) {
				return failed("target_not_found");
			}
			if (bot.getEyePosition().distanceTo(other.getEyePosition()) > 3.0) {
				return failed("out_of_reach");
			}
			if (Sight.entity(bot, other) == null) {
				return failed("not_visible");
			}
			to = other.getInventory();
		} else {
			BlockPos position = ControlCommands.position(args);
			String reason = spatial(position);
			if (reason != null) {
				return failed(reason);
			}
			if (!Documents.permitted(state(), bot, position)) {
				return failed("acl_denied");
			}
			if (shopPrices(position) != null && !ownsStall(position)) {
				return failed("shop_use_buy");
			}
			if (!(bot.level().getBlockEntity(position) instanceof Container container)) {
				return failed("target_not_found");
			}
			if (container instanceof RandomizableContainer loot && loot.getLootTable() != null) {
				return failed("interrupted");
			}
			to = container;
			if (skill.equals("chest_take")) {
				from = container;
				to = bot.getInventory();
			}
		}
		if (SkillInventory.count(from, item) < count) {
			return failed("missing_items");
		}
		if (skill.equals("give") || skill.equals("chest_take")) {
			JsonObject heavy = CarryLimits.check(to, item, count, config());
			if (heavy != null) {
				if (skill.equals("give")) {
					String recipient = args.get("target").getAsString();
					heavy.addProperty("recipient", recipient);
					heavy.addProperty("hint", recipient + " can carry " + heavy.get("limit").getAsInt() + " "
						+ BuiltInRegistries.ITEM.getKey(item).toString().replace("minecraft:", "") + " at most; "
						+ recipient + " has " + heavy.get("have").getAsInt() + "; " + recipient + " can take "
						+ heavy.get("remaining").getAsInt() + " more");
				}
				return new Check("too_heavy", 0, new JsonObject(), heavy);
			}
		}
		String reason = SkillInventory.transfer(from, to, item, count, skill.equals("chest_put") ? null : config(), false);
		return reason == null ? ok(config().integer(skill.equals("give") ? "give_ticks" : "container_ticks"), new JsonObject()) : failed(reason);
	}

	private Check checkSection(JsonObject args) {
		JsonArray blocks = schematic(args);
		if (blocks == null) {
			return failed("target_not_found");
		}
		Map<BlockPos, BlockState> overlay = new HashMap<>();
		SkillPath.View view = new SkillPath.View(bot.level(), overlay);
		JsonObject bill = new JsonObject();
		for (JsonElement element : blocks) {
			JsonObject block = element.getAsJsonObject();
			BlockPos position = ControlCommands.position(block);
			BlockState expected = block(block.get("block").getAsString()).defaultBlockState();
			if (!view.loaded(position)) {
				return failed("target_not_found");
			}
			BlockState current = view.getBlockState(position);
			if (current.is(expected.getBlock())) {
				continue;
			}
			if (!current.isAir() && !current.canBeReplaced()) {
				return failed("target_occupied");
			}
			String material = block.get("item").getAsString();
			bill.addProperty(material, bill.has(material) ? Math.incrementExact(bill.get(material).getAsInt()) : 1);
			overlay.put(position, expected);
		}
		for (var entry : bill.entrySet()) {
			Item material = BuiltInRegistries.ITEM.getValue(Identifier.parse(entry.getKey()));
			if (SkillInventory.count(bot.getInventory(), material) < entry.getValue().getAsInt()) {
				return new Check("missing_items", 0, bill, new JsonObject());
			}
		}
		overlay.clear();
		Vec3 eye = bot.getEyePosition();
		BlockPos feet = feet();
		long ticks = 0;
		for (JsonElement element : blocks) {
			JsonObject block = element.getAsJsonObject();
			BlockPos position = ControlCommands.position(block);
			BlockState expected = block(block.get("block").getAsString()).defaultBlockState();
			if (view.getBlockState(position).is(expected.getBlock())) {
				continue;
			}
			Check placement = checkPlace(block, view, eye, false);
			if (placement.reason() != null || !SkillPath.standable(view, feet) || occupies(position, feet)) {
				Spot spot = standingSpot(view, feet, position, block);
				if (spot == null) {
					String reason = !hasSupport(view, position) ? "no_support"
						: placement.reason() != null && !placement.reason().equals("out_of_reach") ? placement.reason() : "no_path";
					return new Check(reason, ticks, bill, new JsonObject());
				}
				feet = spot.feet();
				eye = Vec3.atBottomCenterOf(feet).add(0, bot.getEyeHeight(), 0);
				ticks = Math.addExact(ticks, spot.route().ticks());
				placement = checkPlace(block, view, eye, false);
			}
			if (placement.reason() != null) {
				return new Check(placement.reason(), ticks, bill, new JsonObject());
			}
			if (occupies(position, feet)) {
				return new Check("target_occupied", ticks, bill, new JsonObject());
			}
			ticks = Math.addExact(ticks, placement.ticks());
			overlay.put(position, expected);
		}
		JsonObject data = new JsonObject();
		data.add("blocks", blocks);
		data.addProperty("index", 0);
		return new Check(null, ticks, bill, data);
	}

	private JsonArray schematic(JsonObject args) {
		String name = args.get("schematic").getAsString();
		if (!name.matches("[A-Za-z0-9_-][A-Za-z0-9_.-]*")) {
			throw new IllegalArgumentException("Invalid schematic name");
		}
		BlockPos origin = position(args.getAsJsonArray("origin"));
		Path directory = bot.level().getServer().getWorldPath(LevelResource.ROOT).resolve("simharness/schematics");
		Path path = directory.resolve(name + ".json");
		JsonObject source;
		try {
			if (!path.toRealPath().getParent().equals(directory.toRealPath())) {
				throw new IllegalArgumentException("Schematic must be inside simharness/schematics");
			}
			source = JsonParser.parseString(Files.readString(path, StandardCharsets.UTF_8)).getAsJsonObject();
		} catch (java.nio.file.NoSuchFileException failure) {
			return null;
		} catch (IOException failure) {
			throw new UncheckedIOException(failure);
		}
		List<JsonObject> ordered = new ArrayList<>();
		for (JsonElement element : source.getAsJsonArray("blocks")) {
			JsonObject entry = element.getAsJsonObject();
			Block block = block(entry.get("block").getAsString());
			Item material = block.asItem();
			if (!(material instanceof BlockItem)) {
				throw new IllegalArgumentException("Schematic block has no placeable item: " + entry.get("block"));
			}
			JsonObject placement = new JsonObject();
			placement.addProperty("x", Math.addExact(origin.getX(), integer(entry, "dx", -30_000_000, 30_000_000)));
			placement.addProperty("y", Math.addExact(origin.getY(), integer(entry, "dy", -30_000_000, 30_000_000)));
			placement.addProperty("z", Math.addExact(origin.getZ(), integer(entry, "dz", -30_000_000, 30_000_000)));
			placement.addProperty("block", BuiltInRegistries.BLOCK.getKey(block).toString());
			placement.addProperty("item", BuiltInRegistries.ITEM.getKey(material).toString());
			ordered.add(placement);
		}
		ordered.sort(Comparator.comparingInt(entry -> entry.get("y").getAsInt()));
		JsonArray result = new JsonArray();
		ordered.forEach(result::add);
		return result;
	}

	private Spot standingSpot(SkillPath.View view, BlockPos from, BlockPos target, JsonObject placement) {
		int radius = (int) Math.ceil(config().number("reach"));
		Spot best = null;
		for (int deltaX = -radius; deltaX <= radius; deltaX++) {
			for (int deltaY = -radius; deltaY <= radius; deltaY++) {
				for (int deltaZ = -radius; deltaZ <= radius; deltaZ++) {
					BlockPos feet = target.offset(deltaX, deltaY, deltaZ);
					if (!SkillPath.standable(view, feet) || occupies(target, feet)) {
						continue;
					}
					Vec3 eye = Vec3.atBottomCenterOf(feet).add(0, bot.getEyeHeight(), 0);
					if (eye.distanceTo(Vec3.atCenterOf(target)) > config().number("reach")
							|| checkPlace(placement, view, eye, false).reason() != null) {
						continue;
					}
					SkillPath.Route route = SkillPath.find(view, from, feet);
					if (route != null && (best == null || route.length() < best.route().length())) {
						best = new Spot(feet, route);
					}
				}
			}
		}
		return best;
	}

	private static boolean occupies(BlockPos target, BlockPos feet) {
		return target.getX() == feet.getX() && target.getZ() == feet.getZ()
			&& (target.getY() == feet.getY() || target.getY() == feet.getY() + 1);
	}

	private static boolean hasSupport(SkillPath.View view, BlockPos target) {
		for (Direction direction : SUPPORTS) {
			BlockPos support = target.relative(direction);
			if (view.loaded(support) && !view.getBlockState(support).getCollisionShape(view, support).isEmpty()
					&& view.getBlockState(support).isFaceSturdy(view, support, direction.getOpposite())) {
				return true;
			}
		}
		return false;
	}

	private void tickSection() {
		JsonObject data = internal();
		JsonArray blocks = data.getAsJsonArray("blocks");
		int index = data.get("index").getAsInt();
		while (index < blocks.size()) {
			JsonObject block = blocks.get(index).getAsJsonObject();
			BlockPos target = ControlCommands.position(block);
			if (!bot.level().hasChunkAt(target)) {
				finish("failed", "target_not_found");
				return;
			}
			if (HarnessState.blockId(bot.level().getBlockState(target)).equals(block.get("block").getAsString())) {
				index++;
				data.addProperty("index", index);
				data.remove("movement");
				data.remove("placement");
				continue;
			}
			if (data.has("movement")) {
				String result = move(data.getAsJsonObject("movement"));
				if (result == null) {
					return;
				}
				data.remove("movement");
				if (!result.equals("done")) {
					finish("failed", result);
					return;
				}
			}
			Check current = checkPlace(block, new SkillPath.View(bot.level()), bot.getEyePosition(), true);
			if (!data.has("placement") && (bot.isInWater() || current.reason() != null && (current.reason().equals("out_of_reach")
					|| current.reason().equals("not_visible") || current.reason().equals("target_occupied") && occupies(target, feet())))) {
				if (noBank()) {
					finish("failed", "in_water_no_bank");
					return;
				}
				Spot spot = standingSpot(new SkillPath.View(bot.level()), feet(), target, block);
				if (spot == null) {
					finish("failed", !hasSupport(new SkillPath.View(bot.level()), target) ? "no_support"
						: "not_visible".equals(current.reason()) ? "not_visible" : "no_path");
					return;
				}
				data.add("movement", movement(spot.route(), Vec3.atBottomCenterOf(spot.feet()), 0.25));
				return;
			}
			if (!data.has("placement")) {
				Check checked = current;
				if (checked.reason() != null) {
					finish("failed", checked.reason());
					return;
				}
				data.add("placement", checked.data());
				data.addProperty("timer", 0);
			}
			int timer = data.get("timer").getAsInt() + 1;
			data.addProperty("timer", timer);
			String result = place(block, data.getAsJsonObject("placement"), timer);
			if (result == null) {
				return;
			}
			if (!result.equals("done")) {
				finish("failed", result);
				return;
			}
			progress().addProperty("placed", progress().get("placed").getAsInt() + 1);
			data.addProperty("index", index + 1);
			data.remove("placement");
			if (index + 1 == blocks.size()) {
				finish("succeeded", null);
			}
			return;
		}
		finish("succeeded", null);
	}

	private record Spot(BlockPos feet, SkillPath.Route route) {
	}

	/**
	 * Advances the active skill during the bot's vanilla input application on the server thread.
	 * Elapsed progress advances exactly once; completion records use overworld game time.
	 */
	public void tick() {
		minedDrops.values().removeIf(drop -> !drop.available(time())
			|| !(bot.level().getEntity(drop.uuid()) instanceof ItemEntity entity) || !entity.isAlive());
		if (last != null && last.get("skill").getAsString().equals("mine") && last.getAsJsonObject("progress").has("drops")) {
			last = dropStatus(last.deepCopy());
		}
		if (active == null) {
			return;
		}
		if (bot.hasInfiniteMaterials() || !bot.isAlive() || bot.level() != bot.level().getServer().overworld()) {
			finish("failed", "interrupted");
			return;
		}
		if (ferryRunning()) {
			clearInput();
			return;
		}
		JsonObject data = internal();
		data.addProperty("elapsed", data.get("elapsed").getAsLong() + 1);
		clearInput();
		JsonObject args = active.getAsJsonObject("args");
		switch (active.get("skill").getAsString()) {
			case "speak" -> tickSpeech(data);
			case "move_to" -> tickMove(data);
			case "mine" -> tickMine(args);
			case "use" -> tickUse(args);
			case "drop" -> tickDrop(args);
			case "hold" -> tickHold(args);
			case "eat" -> tickEat(args);
			case "bake" -> tickBake(args);
			case "collect" -> tickCollect(args);
			case "craft" -> tickCraft(args);
			case "place" -> tickPlace(args);
			case "place_section" -> tickSection();
			case "give", "chest_put", "chest_take" -> tickTransfer(args);
			case "buy" -> tickBuy(args);
			case "wait" -> tickWait(data);
			default -> throw new IllegalStateException("Invalid saved skill");
		}
	}

	/** Selects or swaps a carried stack and completes immediately, without advancing game time. */
	private void tickHold(JsonObject args) {
		int selected = SkillInventory.hold(bot.getInventory(), item(args), bot.getInventory().getSelectedSlot(), true);
		if (selected < 0) {
			finish("failed", "missing_items");
			return;
		}
		bot.getInventory().setSelectedSlot(selected);
		progress().addProperty("item", BuiltInRegistries.ITEM.getKey(bot.getMainHandItem().getItem()).toString());
		finish("succeeded", null);
	}

	/**
	 * Rechecks inventory after the configured duration and spawns one component-preserving stack at the bot's feet.
	 * The explicit-velocity constructor avoids vanilla's random throw motion; existing entity-creation policy
	 * supplies the seeded entity RNG. Vanilla's 40-tick player-drop pickup delay lets the bot walk away at its next
	 * decision poll; ordinary item physics, damage and the 6,000-tick lifetime remain intact.
	 * Inventory is committed only after the entity is admitted, and owned-drop knowledge retains its short deadline.
	 * With {@code unlimited_lifetime} true (scenario staging only; the agent tool never sends it) the entity never
	 * despawns, so staged evidence such as the incident's dropped tool stays in the world.
	 * The drop event and public progress identify the same entity; the ordinary skill-end record follows.
	 * @param args requested item identifier and positive count, rechecked before committing the staged inventory
	 */
	private void tickDrop(JsonObject args) {
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		SkillDrop drop = SkillDrop.plan(bot.getInventory(), item(args), integer(args, "count", 1, Integer.MAX_VALUE));
		if (drop.reason() != null) {
			finish("failed", drop.reason());
			return;
		}
		ItemEntity dropped = new ItemEntity(bot.level(), bot.getX(), bot.getY(), bot.getZ(), drop.stack(), 0, 0, 0);
		dropped.setThrower(bot);
		dropped.setPickUpDelay(40);
		if (args.has("unlimited_lifetime") && args.get("unlimited_lifetime").getAsBoolean()) {
			dropped.setUnlimitedLifetime();
		}
		if (!bot.level().addFreshEntity(dropped)) {
			finish("failed", "drop_failed");
			return;
		}
		SkillInventory.commit(bot.getInventory(), drop.slots());
		MinedDrop remembered = minedDrop(dropped, time() + MINED_DROP_TICKS);
		minedDrops.put(remembered.uuid(), remembered);
		JsonObject event = state().event("drop");
		event.addProperty("bot", bot.getGameProfile().name());
		event.addProperty("item", BuiltInRegistries.ITEM.getKey(dropped.getItem().getItem()).toString());
		event.addProperty("count", dropped.getItem().getCount());
		event.addProperty("itemEntity", dropped.getUUID().toString());
		event.addProperty("thrower", bot.getUUID().toString());
		event.addProperty("prov", "entity:" + dropped.getUUID());
		event.add("pos", HarnessState.vector(dropped.position()));
		state().logEvent(event);
		progress().add("eventId", event.get("id"));
		state().seenActions().record(bot, event);
		for (String key : new String[]{"item", "count", "itemEntity", "thrower", "prov", "pos"}) {
			progress().add(key, event.get(key).deepCopy());
		}
		finish("succeeded", null);
	}

	/** Completes a timed meal after rechecking inventory, then logs the consumed item and restored food. */
	private void tickEat(JsonObject args) {
		progress().add("ticks", internal().get("elapsed"));
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		int before = bot.getFoodData().getFoodLevel();
		SkillEat eaten = SkillEat.consume(bot.getInventory(), item(args), bot.getFoodData(), config().freshDays(),
			Math.floorDiv(time(), 24_000L));
		if (eaten.reason() != null) {
			finish("failed", eaten.reason());
			return;
		}
		JsonObject event = state().event("eat");
		event.addProperty("bot", bot.getGameProfile().name());
		event.addProperty("item", BuiltInRegistries.ITEM.getKey(item(args)).toString());
		event.addProperty("count", 1);
		event.addProperty("foodBefore", before);
		event.addProperty("foodAfter", bot.getFoodData().getFoodLevel());
		state().logEvent(event);
		progress().add("eventId", event.get("id"));
		for (String key : new String[]{"item", "count", "foodBefore", "foodAfter"}) {
			progress().add(key, event.get(key));
		}
		state().hunger().changed(bot, before, "eat");
		finish("succeeded", null);
	}

	/** Advances the ferry once at tick end so completion cannot precede vanilla physics. */
	void tickFerry() {
		if (!ferryRunning()) {
			throw new IllegalStateException("Reserved ferry rider has no running ferry skill");
		}
		JsonObject data = internal();
		data.addProperty("elapsed", Math.incrementExact(data.get("elapsed").getAsLong()));
		progress().add("ticks", data.get("elapsed"));
		progress().addProperty("phase", state().ferry().phase(data));
		String result = state().ferry().advance(bot, data, active.get("startedAt").getAsLong());
		if (result != null) {
			if (result.equals("done")) {
				progress().addProperty("phase", "landed");
			}
			finish(result.equals("done") ? "succeeded" : "failed", result.equals("done") ? null : result);
		}
	}

	private void tickSpeech(JsonObject data) {
		progress().add("ticks", data.get("elapsed"));
	}

	/**
	 * Completes the speak skill only after endpoint delivery or a failed question/answer, recording who heard the
	 * words (a speaker can see whether anyone was close enough to hear).
	 */
	void speechFinished(String reason, JsonArray listeners) {
		if (active != null && active.get("skill").getAsString().equals("speak")) {
			JsonArray heardBy = new JsonArray();
			listeners.forEach(listener -> heardBy.add(state().communication().name(listener.getAsString())));
			progress().add("heardBy", heardBy);
			finish(reason == null ? "succeeded" : "failed", reason);
		}
	}

	private void tickMove(JsonObject data) {
		String result = move(data);
		if (result != null) {
			finish(result.equals("done") ? "succeeded" : "failed", result.equals("done") ? null : result);
		}
	}

	private void tickWait(JsonObject data) {
		progress().add("ticks", data.get("elapsed"));
		if (data.get("elapsed").getAsLong() >= data.get("estimate").getAsLong()) {
			finish("succeeded", null);
		}
	}

	/** Breaks through vanilla and remembers only the new item entities spawned synchronously by this mine. */
	private void tickMine(JsonObject args) {
		BlockPos position = ControlCommands.position(args);
		if (!bot.level().hasChunkAt(position) || Block.getId(bot.level().getBlockState(position)) != internal().get("blockState").getAsInt()) {
			finish("failed", "interrupted");
			return;
		}
		Check checked = checkMine(args);
		if (checked.reason() != null) {
			finish("failed", checked.reason());
			return;
		}
		look(Vec3.atCenterOf(position), 0, false, false);
		float accumulated = Math.min(1, internal().get("destroyProgress").getAsFloat()
			+ bot.level().getBlockState(position).getDestroyProgress(bot, bot.level(), position));
		internal().addProperty("destroyProgress", accumulated);
		progress().addProperty("destroyProgress", accumulated);
		if (accumulated >= 1) {
			AABB area = new AABB(position).inflate(1);
			List<UUID> before = bot.level().getEntitiesOfClass(ItemEntity.class, area).stream().map(ItemEntity::getUUID).toList();
			boolean destroyed = bot.gameMode.destroyBlock(position);
			boolean changed = destroyed && Block.getId(bot.level().getBlockState(position)) != internal().get("blockState").getAsInt();
			if (changed) {
				JsonArray drops = new JsonArray();
				bot.level().getEntitiesOfClass(ItemEntity.class, area).stream()
					.filter(entity -> entity.isAlive() && !before.contains(entity.getUUID()))
					.sorted(Comparator.comparing(entity -> entity.getUUID().toString())).forEach(entity -> {
						MinedDrop drop = minedDrop(entity, time() + MINED_DROP_TICKS);
						minedDrops.put(drop.uuid(), drop);
						drops.add(drop.progress());
					});
				if (!drops.isEmpty()) {
					progress().add("drops", drops);
				}
			}
			finish(changed ? "succeeded" : "failed", changed ? null : "interrupted");
		}
	}

	private static JsonObject movement(SkillPath.Route route, Vec3 target, double tolerance) {
		JsonObject data = new JsonObject();
		JsonArray path = new JsonArray();
		route.positions().forEach(position -> path.add(HarnessState.position(position)));
		data.add("path", path);
		if (route.swimWaypoints() > 0) {
			data.addProperty("swimWaypoints", route.swimWaypoints());
		}
		data.add("target", HarnessState.vector(target));
		data.addProperty("tolerance", tolerance);
		data.addProperty("waypoint", 0);
		data.addProperty("replans", 0);
		data.addProperty("moveElapsed", 0);
		data.addProperty("moveEstimate", route.ticks());
		data.addProperty("lastImprovement", 0);
		return data;
	}

	private String move(JsonObject data) {
		long elapsed = data.get("moveElapsed").getAsLong() + 1;
		data.addProperty("moveElapsed", elapsed);
		if (elapsed > 2 * data.get("moveEstimate").getAsLong() + 100) {
			return "timeout";
		}
		JsonArray path = data.getAsJsonArray("path");
		int waypoint = data.get("waypoint").getAsInt();
		int swimWaypoints = data.has("swimWaypoints") ? data.get("swimWaypoints").getAsInt() : 0;
		Vec3 target = vector(data.getAsJsonArray("target"));
		while (waypoint < path.size()) {
			BlockPos next = position(path.get(waypoint).getAsJsonArray());
			Vec3 center = Vec3.atBottomCenterOf(next);
			boolean swimming = waypoint < swimWaypoints;
			boolean bank = swimming && waypoint == swimWaypoints - 1;
			boolean wrongHeight = swimming && !bank ? bot.getY() < center.y - 1.1 : Math.abs(bot.getY() - center.y) >= 0.6;
			if (horizontal(bot.position(), center) >= 0.3 || wrongHeight || bank && (!bot.onGround() || bot.isInWater())) {
				break;
			}
			waypoint++;
			data.remove("bestDistance");
			data.addProperty("lastImprovement", elapsed);
		}
		data.addProperty("waypoint", waypoint);
		progress().addProperty("waypoint", waypoint);
		progress().addProperty("waypoints", path.size());
		progress().add("replans", data.get("replans"));
		if (waypoint == path.size() && bot.onGround() && horizontal(bot.position(), target) <= data.get("tolerance").getAsDouble()) {
			return "done";
		}
		Vec3 destination = waypoint < path.size() ? Vec3.atBottomCenterOf(position(path.get(waypoint).getAsJsonArray())) : target;
		double distance = bot.position().distanceTo(destination);
		if (!data.has("bestDistance") || data.get("bestDistance").getAsDouble() - distance > 0.05) {
			data.addProperty("bestDistance", distance);
			data.addProperty("lastImprovement", elapsed);
		}
		if (elapsed - data.get("lastImprovement").getAsLong() >= 40) {
			int replans = data.get("replans").getAsInt();
			if (replans >= 2) {
				return "stuck";
			}
			SkillPath.Route replanned = SkillPath.find(new SkillPath.View(bot.level()), feet(), BlockPos.containing(target));
			if (replanned == null) {
				return noBank() ? "in_water_no_bank" : "stuck";
			}
			JsonObject replacement = movement(replanned, target, data.get("tolerance").getAsDouble());
			data.add("path", replacement.get("path"));
			data.remove("swimWaypoints");
			if (replacement.has("swimWaypoints")) {
				data.add("swimWaypoints", replacement.get("swimWaypoints"));
			}
			data.addProperty("waypoint", 0);
			data.addProperty("replans", replans + 1);
			data.addProperty("lastImprovement", elapsed);
			data.remove("bestDistance");
			return null;
		}
		boolean higher = destination.y > bot.getY() + 0.2;
		boolean swimming = waypoint < swimWaypoints;
		float forward = swimming && horizontal(bot.position(), destination) < 0.2 ? 0 : 1;
		look(destination.add(0, bot.getEyeHeight(), 0), forward, swimming || higher, false);
		return null;
	}

	/** Distinguishes a missing nearby bank from ordinary walking failures, including runtime replans. */
	private boolean noBank() {
		return bot.isInWater() && SkillPath.swimLeg(new SkillPath.View(bot.level()), feet()) == null;
	}

	/** Lists visible items and the owning bot's recent mined or deliberately dropped items; checks enforce pickup paths. */
	private List<ItemEntity> items(double radius) {
		List<ItemEntity> candidates = bot.level().getEntitiesOfClass(ItemEntity.class, bot.getBoundingBox().inflate(radius),
			entity -> entity.isAlive() && entity.position().distanceTo(bot.position()) <= radius
				&& (Sight.entity(bot, entity) != null || knowsDrop(entity)));
		candidates.sort(Comparator.comparingDouble((ItemEntity entity) -> entity.position().distanceToSqr(bot.position()))
			.thenComparing(entity -> entity.getUUID().toString()));
		return candidates;
	}

	/** Tests this controller's own knowledge only; other bots' drops never grant an exception to sight. */
	private boolean knowsDrop(ItemEntity entity) {
		MinedDrop drop = minedDrops.get(entity.getUUID());
		return drop != null && drop.available(time());
	}

	/** Routes to the item, or beside a remembered drop if vanilla pickup can reach it under a low ceiling. */
	private SkillPath.Route routeToItem(ItemEntity entity) {
		SkillPath.View view = new SkillPath.View(bot.level());
		BlockPos target = entity.blockPosition();
		for (int elevation : new int[]{0, 1, -1, -2, -3}) {
			BlockPos candidate = target.above(elevation);
			if (knowsDrop(entity) && !pickupAt(bot.getBoundingBox(), bot.position(), candidate, entity.getBoundingBox())) {
				continue;
			}
			SkillPath.Route route = SkillPath.find(view, feet(), candidate);
			if (route != null) {
				return route;
			}
		}
		if (knowsDrop(entity)) {
			SkillPath.Route best = null;
			for (int deltaX = -1; deltaX <= 1; deltaX++) {
				for (int deltaZ = -1; deltaZ <= 1; deltaZ++) {
					for (int elevation : new int[]{0, 1, -1, -2, -3}) {
						BlockPos candidate = target.offset(deltaX, elevation, deltaZ);
						if (!pickupAt(bot.getBoundingBox(), bot.position(), candidate, entity.getBoundingBox())) {
							continue;
						}
						SkillPath.Route route = SkillPath.find(view, feet(), candidate);
						if (route != null && (best == null || route.length() < best.length())) {
							best = route;
						}
					}
				}
			}
			return best;
		}
		return null;
	}

	/** Tests vanilla's pickup envelope at a proposed standing cell, without a world or an item transfer. */
	static boolean pickupAt(AABB body, Vec3 position, BlockPos feet, AABB item) {
		Vec3 offset = Vec3.atBottomCenterOf(feet).subtract(position);
		return body.move(offset).inflate(1, 0.5, 1).intersects(item);
	}

	/** Walks to a legal pickup position; only vanilla, never this skill, transfers the item into inventory. */
	private void tickCollect(JsonObject args) {
		JsonObject data = internal();
		if (data.get("elapsed").getAsLong() > 600) {
			finish("failed", "timeout");
			return;
		}
		double radius = number(args, "radius", 6.0, 0, 256);
		List<ItemEntity> candidates = items(radius);
		if (candidates.isEmpty()) {
			finish("succeeded", null);
			return;
		}
		ItemEntity nearest = candidates.getFirst();
		JsonObject heavy = CarryLimits.check(bot.getInventory(), nearest.getItem().getItem(), nearest.getItem().getCount(), config());
		if (heavy != null) {
			heavy.entrySet().forEach(entry -> progress().add(entry.getKey(), entry.getValue()));
			finish("failed", "too_heavy");
			return;
		}
		if (!SkillInventory.insert(bot.getInventory(), SkillInventory.copy(bot.getInventory()), nearest.getItem().copyWithCount(1))) {
			finish("failed", "inventory_full");
			return;
		}
		if (horizontal(bot.position(), nearest.position()) < 0.6 && Math.abs(bot.getY() - nearest.getY()) < 1) {
			data.remove("movement");
			return;
		}
		String uuid = nearest.getUUID().toString();
		if (!data.has("movement") || !data.has("itemEntity") || !data.get("itemEntity").getAsString().equals(uuid)
				|| !(data.has("itemPos") ? position(data.getAsJsonArray("itemPos"))
					: BlockPos.containing(vector(data.getAsJsonObject("movement").getAsJsonArray("target")))).equals(nearest.blockPosition())) {
			if (knowsDrop(nearest) && !bot.onGround() && data.has("movement") && data.has("itemEntity")
					&& data.get("itemEntity").getAsString().equals(uuid)) {
				String result = move(data.getAsJsonObject("movement"));
				if (result != null && !result.equals("done")) {
					finish("failed", result);
				}
				return;
			}
			SkillPath.Route route = routeToItem(nearest);
			if (route == null) {
				finish("failed", noBank() ? "in_water_no_bank" : "no_path");
				return;
			}
			data.addProperty("itemEntity", uuid);
			BlockPos goal = route.positions().isEmpty() ? feet() : route.positions().getLast();
			boolean beside = knowsDrop(nearest) && !goal.equals(nearest.blockPosition());
			if (beside) {
				data.add("itemPos", HarnessState.position(nearest.blockPosition()));
			} else {
				data.remove("itemPos");
			}
			data.add("movement", movement(route, beside ? Vec3.atBottomCenterOf(goal) : nearest.position(), beside ? 0.1 : 0.4));
		}
		String result = move(data.getAsJsonObject("movement"));
		if (result != null) {
			if (result.equals("done")) {
				data.remove("movement");
			} else {
				finish("failed", result);
			}
		}
	}

	private void tickPlace(JsonObject args) {
		String reason = place(args, internal(), internal().get("elapsed").getAsLong());
		if (reason != null) {
			finish(reason.equals("done") ? "succeeded" : "failed", reason.equals("done") ? null : reason);
		}
	}

	/** Runs the block's empty-hand vanilla action once after its duration, then exposes the resulting event. */
	private void tickUse(JsonObject args) {
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		Check checked = checkUse(args);
		if (checked.reason() != null) {
			finish("failed", checked.reason());
			return;
		}
		BlockPos position = ControlCommands.position(args);
		BlockState before = bot.level().getBlockState(position);
		if (Block.getId(before) != internal().get("blockState").getAsInt()) {
			finish("failed", "interrupted");
			return;
		}
		BlockHitResult hit = useHit(before, position, Sight.block(bot, position).hit());
		look(hit.getLocation(), 0, false, false);
		if (!before.useWithoutItem(bot.level(), bot, hit).consumesAction()) {
			finish("failed", "not_usable");
			return;
		}
		JsonObject event = state().event("block_use");
		event.addProperty("actor", bot.getGameProfile().name());
		event.add("skillId", active.get("skillId"));
		event.add("pos", HarnessState.position(position));
		event.addProperty("block", HarnessState.blockId(before));
		event.addProperty("state", bot.level().getBlockState(position).toString());
		event.addProperty("prov", "event:" + event.get("id").getAsLong());
		boolean bell = before.getBlock() instanceof BellBlock;
		BlockState after = bot.level().getBlockState(position);
		String opened = after.hasProperty(BlockStateProperties.OPEN)
			? "; it is now " + (after.getValue(BlockStateProperties.OPEN) ? "open" : "closed") : "";
		String text = bell ? "bell rang at " + positionText(position)
			: bot.getGameProfile().name() + " used the " + BuiltInRegistries.BLOCK.getKey(before.getBlock()).getPath()
				+ " at " + positionText(position) + opened;
		state().communication().notice(event, text, bell);
		state().logEvent(event);
		for (String key : List.of("pos", "block", "state", "prov")) {
			progress().add(key, event.get(key).deepCopy());
		}
		progress().add("eventId", event.get("id"));
		finish("succeeded", null);
	}

	private String place(JsonObject args, JsonObject data, long elapsed) {
		BlockPos position = ControlCommands.position(args);
		if (!bot.level().hasChunkAt(position) || Block.getId(bot.level().getBlockState(position)) != data.get("blockState").getAsInt()) {
			return "interrupted";
		}
		Check checked = checkPlace(args, new SkillPath.View(bot.level()), bot.getEyePosition(), true);
		if (checked.reason() != null) {
			return checked.reason();
		}
		Item wanted = item(args);
		int selected = bot.getInventory().getSelectedSlot();
		for (int slot = 0; slot < bot.getInventory().getContainerSize(); slot++) {
			if (bot.getInventory().getItem(slot).is(wanted)) {
				if (slot != selected) {
					ItemStack previous = bot.getInventory().getItem(selected);
					bot.getInventory().setItem(selected, bot.getInventory().getItem(slot));
					bot.getInventory().setItem(slot, previous);
					bot.getInventory().setChanged();
				}
				break;
			}
		}
		BlockPos support = position(checked.data().getAsJsonArray("support"));
		Direction face = Direction.valueOf(checked.data().get("face").getAsString());
		Vec3 hit = Vec3.atCenterOf(support).add(face.getStepX() * 0.5, face.getStepY() * 0.5, face.getStepZ() * 0.5);
		look(hit, 0, false, false);
		if (elapsed < config().integer("place_ticks")) {
			return null;
		}
		bot.setShiftKeyDown(true);
		try {
			bot.gameMode.useItemOn(bot, bot.level(), bot.getMainHandItem(), InteractionHand.MAIN_HAND, new BlockHitResult(hit, face, support, false));
		} finally {
			bot.setShiftKeyDown(false);
		}
		return HarnessState.blockId(bot.level().getBlockState(position)).equals(data.get("expected").getAsString()) ? "done" : "interrupted";
	}

	private void tickCraft(JsonObject args) {
		JsonObject data = internal();
		int timer = data.has("timer") ? data.get("timer").getAsInt() + 1 : 1;
		data.addProperty("timer", timer);
		if (timer < config().integer("craft_ticks")) {
			return;
		}
		if (data.has("needsTable") && data.get("needsTable").getAsBoolean() && !tableAvailable()) {
			finish("failed", "needs_crafting_table");
			return;
		}
		JsonObject result = SkillCrafting.craft(bot, data.get("recipe").getAsString(), args.get("item").getAsString());
		if (result.has("reason")) {
			if (result.get("reason").getAsString().equals("too_heavy")) {
				result.entrySet().forEach(entry -> progress().add(entry.getKey(), entry.getValue()));
			}
			finish("failed", result.get("reason").getAsString());
			return;
		}
		int produced = (progress().has("produced") ? progress().get("produced").getAsInt() : 0) + result.get("produced").getAsInt();
		progress().addProperty("produced", produced);
		data.addProperty("timer", 0);
		if (produced >= (args.has("count") ? args.get("count").getAsInt() : 1)) {
			finish("succeeded", null);
		}
	}

	/**
	 * Checks a purchase from a shop chest at its posted price: in reach and visible, priced, in stock, affordable,
	 * and with room for both the goods (in the buyer's inventory) and the payment (in the shop chest).
	 */
	private Check checkBuy(JsonObject args) {
		BlockPos position = ControlCommands.position(args);
		String reason = spatial(position);
		if (reason != null) {
			return failed(reason);
		}
		JsonObject prices = shopPrices(position);
		if (prices == null || !(bot.level().getBlockEntity(position) instanceof Container shop)) {
			return failed("target_not_found");
		}
		Item item = item(args);
		String id = BuiltInRegistries.ITEM.getKey(item).toString();
		if (!prices.has(id)) {
			return failed("not_for_sale");
		}
		int count = integer(args, "count", 1, Integer.MAX_VALUE);
		JsonObject heavy = CarryLimits.check(bot.getInventory(), item, count, config());
		if (heavy != null) {
			return new Check("too_heavy", 0, new JsonObject(), heavy);
		}
		if (SkillInventory.count(shop, item) < count) {
			return failed("out_of_stock");
		}
		int cost = Math.multiplyExact(prices.get(id).getAsInt(), count);
		if (SkillInventory.count(bot.getInventory(), Items.EMERALD) < cost) {
			return failed("cannot_afford");
		}
		String room = SkillInventory.transfer(shop, bot.getInventory(), item, count, config(), false);
		if (room == null && cost > 0) {
			room = SkillInventory.transfer(bot.getInventory(), shop, Items.EMERALD, cost, false);
		}
		if (room != null) {
			return failed(room);
		}
		JsonObject data = new JsonObject();
		data.addProperty("cost", cost);
		return ok(config().integer("container_ticks"), data);
	}

	/** Returns the shop price list at a position, or null; never creates state (dry-runs must not mutate). */
	private JsonObject shopPrices(BlockPos position) {
		JsonObject shops = state().map("shops");
		return shops == null ? null : shops.getAsJsonObject(HarnessState.key(position));
	}

	/** Ownership permits stocking and taking takings; an absent owner retains the buy-only shop policy. */
	private boolean ownsStall(BlockPos position) {
		JsonObject owners = state().map("containerOwners");
		return config().goodsEnabled() && owners != null
			&& Documents.ownsStall(owners.get(HarnessState.key(position)), bot.getGameProfile().name());
	}

	/** Completes a purchase: payment into the shop chest, goods to the buyer, and a logged purchase event. */
	private void tickBuy(JsonObject args) {
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		Check checked = checkBuy(args);
		if (checked.reason() != null) {
			if (checked.reason().equals("too_heavy")) {
				checked.data().entrySet().forEach(entry -> progress().add(entry.getKey(), entry.getValue()));
			}
			finish("failed", checked.reason());
			return;
		}
		BlockPos position = ControlCommands.position(args);
		Container shop = (Container) bot.level().getBlockEntity(position);
		int count = args.get("count").getAsInt();
		int cost = checked.data().get("cost").getAsInt();
		if (cost > 0) {
			SkillInventory.transfer(bot.getInventory(), shop, Items.EMERALD, cost, true);
		}
		SkillInventory.transfer(shop, bot.getInventory(), item(args), count, config(), true);
		JsonObject event = state().event("purchase");
		event.addProperty("buyer", bot.getGameProfile().name());
		event.addProperty("shop", "container:" + HarnessState.key(position));
		event.addProperty("item", BuiltInRegistries.ITEM.getKey(item(args)).toString());
		event.addProperty("count", count);
		event.addProperty("cost", cost);
		state().logEvent(event);
		progress().addProperty("count", count);
		progress().addProperty("cost", cost);
		finish("succeeded", null);
	}

	private void tickTransfer(JsonObject args) {
		String skill = active.get("skill").getAsString();
		if (internal().get("elapsed").getAsLong() < internal().get("estimate").getAsLong()) {
			return;
		}
		Check checked = checkTransfer(skill, args);
		if (checked.reason() != null) {
			if (checked.reason().equals("too_heavy")) {
				checked.data().entrySet().forEach(entry -> progress().add(entry.getKey(), entry.getValue()));
			}
			finish("failed", checked.reason());
			return;
		}
		Container from = bot.getInventory();
		Container to;
		String fromName = bot.getGameProfile().name();
		String toName;
		if (skill.equals("give")) {
			toName = args.get("target").getAsString();
			to = other(toName).getInventory();
		} else {
			BlockPos position = ControlCommands.position(args);
			to = (Container) bot.level().getBlockEntity(position);
			toName = "container:" + HarnessState.key(position);
			if (skill.equals("chest_take")) {
				from = to;
				to = bot.getInventory();
				String temporary = fromName;
				fromName = toName;
				toName = temporary;
			}
		}
		int count = args.get("count").getAsInt();
		String reason = SkillInventory.transfer(from, to, item(args), count, skill.equals("chest_put") ? null : config(), true);
		if (reason != null) {
			finish("failed", reason);
			return;
		}
		JsonObject event = state().event("transfer");
		event.addProperty("from", fromName);
		event.addProperty("to", toName);
		event.addProperty("item", BuiltInRegistries.ITEM.getKey(item(args)).toString());
		event.addProperty("count", count);
		if (skill.equals("chest_take")) {  // bystanders learn of the take through SeenActions, item only
			JsonObject owners = state().map("containerOwners");
			Documents.taken(event, owners == null ? null : owners.get(HarnessState.key(ControlCommands.position(args))),
				bot.getGameProfile().name());
		}
		state().logEvent(event);
		progress().addProperty("count", count);
		if (skill.equals("give")) {
			state().seenActions().record(bot, other(toName), event);
		} else {
			state().seenActions().record(bot, skill, ControlCommands.position(args), event);
		}
		finish("succeeded", null);
	}

	private void finish(String outcome, String reason) {
		if (active.get("skill").getAsString().equals("bake")) {
			BlockPos position = position(internal().getAsJsonArray("pos"));
			ovenLit(position, false);
			JsonObject baked = state().event("bake");
			baked.addProperty("bot", bot.getGameProfile().name());
			baked.add("pos", HarnessState.position(position));
			baked.add("count", active.getAsJsonObject("args").get("count"));
			baked.addProperty("outcome", outcome);
			state().logEvent(baked);
		}
		if (ferryRunning()) {
			state().ferry().release(bot);
		}
		if (active.get("skill").getAsString().equals("speak") && !outcome.equals("succeeded")) {
			state().communication().cancel(bot.getGameProfile().name());
		}
		active.addProperty("state", outcome);
		if (reason != null) {
			active.addProperty("reason", reason);
		}
		active.addProperty("endedAt", time());
		last = status();
		JsonObject event = state().event("skill_end");
		event.addProperty("bot", bot.getGameProfile().name());
		event.add("skillId", active.get("skillId"));
		event.add("skill", active.get("skill"));
		event.addProperty("outcome", outcome);
		event.add("reason", reason == null ? JsonNull.INSTANCE : new com.google.gson.JsonPrimitive(reason));
		state().logEvent(event);
		active = null;
		clearInput();
	}

	private void clearInput() {
		bot.setInput(BotInput.idle(bot.getYRot(), bot.getXRot()));
	}

	private void look(Vec3 target, float forward, boolean jump, boolean sneak) {
		Vec3 delta = target.subtract(bot.getEyePosition());
		float yaw = (float) Math.toDegrees(Math.atan2(-delta.x, delta.z));
		float pitch = (float) -Math.toDegrees(Math.atan2(delta.y, Math.hypot(delta.x, delta.z)));
		bot.setInput(new BotInput(forward, 0, jump, sneak, false, yaw, forward == 0 ? pitch : 0));
		bot.setYRot(yaw);
		bot.setXRot(forward == 0 ? pitch : 0);
		bot.setYHeadRot(yaw);
	}

	private String spatial(BlockPos position) {
		if (bot.getEyePosition().distanceTo(Vec3.atCenterOf(position)) > config().number("reach")) {
			return "out_of_reach";
		}
		if (!bot.level().hasChunkAt(position)) {
			return "target_not_found";
		}
		return Sight.block(bot, position) == null ? "not_visible" : null;
	}

	private boolean tableAvailable() {
		double reach = config().number("reach");
		BlockPos center = BlockPos.containing(bot.getEyePosition());
		int radius = (int) Math.ceil(reach);
		for (int deltaX = -radius; deltaX <= radius; deltaX++) {
			for (int deltaY = -radius; deltaY <= radius; deltaY++) {
				for (int deltaZ = -radius; deltaZ <= radius; deltaZ++) {
					BlockPos position = center.offset(deltaX, deltaY, deltaZ);
					if (bot.level().hasChunkAt(position) && bot.getEyePosition().distanceTo(Vec3.atCenterOf(position)) <= reach
							&& bot.level().getBlockState(position).is(Blocks.CRAFTING_TABLE) && Sight.block(bot, position) != null) {
						return true;
					}
				}
			}
		}
		return false;
	}

	private SimBot other(String name) {
		var player = bot.level().getServer().getPlayerList().getPlayerByName(name);
		return player instanceof SimBot managed ? managed : null;
	}

	private static Check failed(String reason) {
		return new Check(reason, 0, new JsonObject(), new JsonObject());
	}

	private static Check ok(long ticks, JsonObject data) {
		return new Check(null, ticks, new JsonObject(), data);
	}

	private HarnessState state() {
		return HarnessState.get(bot.level().getServer());
	}

	private SimConfig config() {
		return state().config();
	}

	private long time() {
		return bot.level().getGameTime();
	}

	private JsonObject internal() {
		return active.getAsJsonObject("internal");
	}

	private JsonObject progress() {
		return active.getAsJsonObject("progress");
	}

	private static Vec3 target(JsonObject args) {
		return new Vec3(number(args, "x", null, -30_000_000, 30_000_000), number(args, "y", null, -30_000_000, 30_000_000),
			number(args, "z", null, -30_000_000, 30_000_000));
	}

	private static double number(JsonObject args, String key, Double fallback, double minimum, double maximum) {
		if (!args.has(key) && fallback != null) {
			return fallback;
		}
		JsonElement value = args.get(key);
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()
				|| !Double.isFinite(value.getAsDouble()) || value.getAsDouble() < minimum || value.getAsDouble() > maximum) {
			throw new IllegalArgumentException("Invalid numeric skill argument: " + key);
		}
		return value.getAsDouble();
	}

	private static int integer(JsonObject args, String key, int minimum, int maximum) {
		number(args, key, null, minimum, maximum);
		try {
			return args.get(key).getAsBigDecimal().intValueExact();
		} catch (ArithmeticException failure) {
			throw new IllegalArgumentException(key + " must be an integer", failure);
		}
	}

	private static Item item(JsonObject args) {
		Item item = BuiltInRegistries.ITEM.getOptional(Identifier.parse(args.get("item").getAsString()))
			.orElseThrow(() -> new IllegalArgumentException("Unknown item"));
		if (item == Items.AIR) {
			throw new IllegalArgumentException("air is not an item");
		}
		return item;
	}

	private static Block block(String id) {
		return BuiltInRegistries.BLOCK.getOptional(Identifier.parse(id)).orElseThrow(() -> new IllegalArgumentException("Unknown block: " + id));
	}

	private static BlockPos position(JsonArray array) {
		JsonObject args = new JsonObject();
		args.add("pos", array);
		return ControlCommands.position(args);
	}

	private static Vec3 vector(JsonArray array) {
		return new Vec3(array.get(0).getAsDouble(), array.get(1).getAsDouble(), array.get(2).getAsDouble());
	}

	private static double horizontal(Vec3 from, Vec3 to) {
		return Math.hypot(from.x - to.x, from.z - to.z);
	}

	/** Formats a block coordinate identically in visible-action and bell-hearing notices. */
	private static String positionText(BlockPos position) {
		return "(" + position.getX() + ", " + position.getY() + ", " + position.getZ() + ")";
	}

	private record Check(String reason, long ticks, JsonObject materials, JsonObject data) {
	}
}
