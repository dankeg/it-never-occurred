package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.Container;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.RandomizableContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.component.WrittenBookContent;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;

/**
 * Dispatches control-socket commands to private domain handlers on the server main thread.
 * Lockstep reseeding precedes command argument evaluation; handlers retain the original
 * operation order and do not schedule additional work or wrap synchronous failures.
 */
public final class ControlCommands {
	private final MinecraftServer server;
	private final BotManager bots;
	private final Documents documents;
	private final Observation observation;
	private final Map<String, CommandHandler> handlers;

	/**
	 * Creates the shared document/observation services and registers the command handlers.
	 * Registration performs no world operations or RNG draws.
	 *
	 * @param server server whose main thread must execute commands
	 * @param bots bot manager shared by every command domain
	 */
	public ControlCommands(MinecraftServer server, BotManager bots) {
		this.server = server;
		this.bots = bots;
		documents = new Documents(server, bots);
		observation = new Observation(server, bots, documents);
		LockstepTimeHandlers timeHandlers = new LockstepTimeHandlers();
		BotsSkillsHandlers botHandlers = new BotsSkillsHandlers();
		ObservationDocumentHandlers documentHandlers = new ObservationDocumentHandlers();
		NpcHandlers npcHandlers = new NpcHandlers();
		PersistenceHandlers persistenceHandlers = new PersistenceHandlers();
		handlers = Map.ofEntries(
			Map.entry("ping", args -> CompletableFuture.completedFuture(timeHandlers.ping(args))),
			Map.entry("lockstep", args -> CompletableFuture.completedFuture(timeHandlers.lockstep(args))),
			Map.entry("step", timeHandlers::step),
			Map.entry("hash", args -> CompletableFuture.completedFuture(timeHandlers.hash(args))),
			Map.entry("await_entities", args -> CompletableFuture.completedFuture(timeHandlers.awaitEntities(args))),
			Map.entry("dump_entities", args -> CompletableFuture.completedFuture(timeHandlers.dumpEntities(args))),
			Map.entry("blocks", args -> CompletableFuture.completedFuture(timeHandlers.blocks(args))),
			Map.entry("support_failure", args -> CompletableFuture.completedFuture(timeHandlers.supportFailure(args))),
			Map.entry("collapse_set", args -> CompletableFuture.completedFuture(HarnessState.get(server).collapse().setup(args))),
			Map.entry("container_items", args -> CompletableFuture.completedFuture(timeHandlers.containerItems(args))),
			Map.entry("reseed", args -> CompletableFuture.completedFuture(timeHandlers.reseed(args))),
			Map.entry("command", args -> CompletableFuture.completedFuture(timeHandlers.command(args))),
			Map.entry("spawn_bot", args -> CompletableFuture.completedFuture(botHandlers.spawn(args))),
			Map.entry("bot_state", args -> CompletableFuture.completedFuture(botHandlers.state(args))),
			Map.entry("skill_start", args -> CompletableFuture.completedFuture(botHandlers.startSkill(args))),
			Map.entry("skill_status", args -> CompletableFuture.completedFuture(botHandlers.skillStatus(args))),
			Map.entry("skill_cancel", args -> CompletableFuture.completedFuture(botHandlers.cancelSkill(args))),
			Map.entry("bot_input", args -> CompletableFuture.completedFuture(botHandlers.input(args))),
			Map.entry("bot_break", args -> CompletableFuture.completedFuture(botHandlers.breakBlock(args))),
			Map.entry("bot_place", args -> CompletableFuture.completedFuture(botHandlers.placeBlock(args))),
			Map.entry("bot_select", args -> CompletableFuture.completedFuture(botHandlers.select(args))),
			Map.entry("give", args -> CompletableFuture.completedFuture(botHandlers.give(args))),
			Map.entry("bot_flag", args -> CompletableFuture.completedFuture(botHandlers.flag(args))),
			Map.entry("observe", args -> CompletableFuture.completedFuture(documentHandlers.observe(args))),
			Map.entry("say", args -> CompletableFuture.completedFuture(documentHandlers.say(args))),
			Map.entry("speak", args -> CompletableFuture.completedFuture(documentHandlers.speak(args))),
			Map.entry("ask", args -> CompletableFuture.completedFuture(documentHandlers.ask(args))),
			Map.entry("write_sign", args -> CompletableFuture.completedFuture(documentHandlers.writeSign(args))),
			Map.entry("write_book", args -> CompletableFuture.completedFuture(documentHandlers.writeBook(args))),
			Map.entry("read_item_doc", args -> CompletableFuture.completedFuture(documentHandlers.readItem(args))),
			Map.entry("open_container", args -> CompletableFuture.completedFuture(documentHandlers.openContainer(args))),
			Map.entry("acl_set", args -> CompletableFuture.completedFuture(documentHandlers.setAcl(args))),
			Map.entry("acl_clear", args -> CompletableFuture.completedFuture(documentHandlers.clearAcl(args))),
			Map.entry("shop_set", args -> CompletableFuture.completedFuture(documentHandlers.setShop(args))),
			Map.entry("ferry_setup", args -> CompletableFuture.completedFuture(HarnessState.get(server).ferry().setup(args))),
			Map.entry("structures_set", args -> CompletableFuture.completedFuture(HarnessState.get(server).structures().setup(args))),
			Map.entry("npc_setup", args -> CompletableFuture.completedFuture(npcHandlers.setup(args))),
			Map.entry("npc_metrics", args -> CompletableFuture.completedFuture(npcHandlers.metrics(args))),
			Map.entry("npc_state", args -> CompletableFuture.completedFuture(npcHandlers.state(args))),
			Map.entry("npc_testimony", args -> CompletableFuture.completedFuture(npcHandlers.testimony(args))),
			Map.entry("config_get", args -> CompletableFuture.completedFuture(persistenceHandlers.config(args))),
			Map.entry("config_set", args -> CompletableFuture.completedFuture(persistenceHandlers.setConfig(args))),
			Map.entry("recording", args -> CompletableFuture.completedFuture(persistenceHandlers.recording(args))),
			Map.entry("seen_actions_clear", args -> CompletableFuture.completedFuture(HarnessState.get(server).seenActions().clearAll())),
			Map.entry("save_snapshot", args -> CompletableFuture.completedFuture(persistenceHandlers.saveSnapshot(args))),
			Map.entry("stop", args -> CompletableFuture.completedFuture(persistenceHandlers.saveSnapshot(args))),
			Map.entry("restore_bots", args -> CompletableFuture.completedFuture(persistenceHandlers.restoreBots(args))));
	}

	private static final java.util.Set<String> WORLD_MUTATING = java.util.Set.of(
		"spawn_bot", "bot_break", "bot_place", "give", "command", "support_failure");

	/**
	 * Runs a registered command after the thread check and any lockstep command reseed.
	 * Synchronous handler failures propagate directly; only step completion is deferred.
	 *
	 * @param command exact, case-sensitive protocol command name
	 * @param args command arguments, interpreted by the handler without pre-validation
	 * @return completed JSON result, or the original lockstep future for {@code step}
	 * @throws IOException if bot persistence or NPC testimony I/O fails
	 * @throws IllegalStateException if called off the server main thread or a handler rejects state
	 * @throws IllegalArgumentException if the command is unknown or a handler rejects arguments
	 */
	public CompletableFuture<JsonObject> execute(String command, JsonObject args) throws IOException {
		if (!server.isSameThread()) {
			throw new IllegalStateException("Commands must run on the server main thread");
		}
		if (WORLD_MUTATING.contains(command) && Lockstep.INSTANCE.isEnabled()) {
			// Each world-mutating command's randomness is a pure function of (seed, game time, command index), so it
			// cannot depend on how much shared RNG earlier code consumed. Agent actions arrive this way between ticks.
			Determinism.reseedForCommand(server);
		}
		CommandHandler handler = handlers.get(command);
		if (handler == null) {
			throw new IllegalArgumentException("Unknown command: " + command);
		}
		return handler.execute(args);
	}

	@FunctionalInterface
	private interface CommandHandler {
		/**
		 * Executes a command inline without changing its failure or completion semantics.
		 *
		 * @param args original command arguments
		 * @return completed result or the pending lockstep step future
		 * @throws IOException if the command's persistence or testimony I/O fails
		 */
		CompletableFuture<JsonObject> execute(JsonObject args) throws IOException;
	}

	/** Server time, lockstep, deterministic inspection and vanilla commands; runs inline. */
	private final class LockstepTimeHandlers {
		private JsonObject ping(JsonObject args) {
			return time();
		}

		private JsonObject lockstep(JsonObject args) {
			Lockstep.INSTANCE.setEnabled(args.get("enabled").getAsBoolean());
			JsonObject result = time();
			result.addProperty("enabled", Lockstep.INSTANCE.isEnabled());
			return result;
		}

		private CompletableFuture<JsonObject> step(JsonObject args) {
			long interval = args.has("hash_every") ? integer(args, "hash_every") : 0;
			if (args.has("hash_every") && interval <= 0) {
				throw new IllegalArgumentException("hash_every must be positive");
			}
			Region region = args.has("region") ? Region.fromJson(args.getAsJsonObject("region")) : null;
			return Lockstep.INSTANCE.step(server, integer(args, "n"), interval, region);
		}

		private JsonObject hash(JsonObject args) {
			return StateHash.compute(server, Region.fromJson(args.getAsJsonObject("region")));
		}

		private JsonObject awaitEntities(JsonObject args) {
			return Readiness.awaitEntities(server, args);
		}

		private JsonObject dumpEntities(JsonObject args) {
			JsonObject result = new JsonObject();
			result.add("entities", StateHash.dumpEntities(server, Region.fromJson(args.getAsJsonObject("region"))));
			return result;
		}

		/**
		 * Reads block ids at the given positions without loading chunks, consuming randomness or changing state.
		 * Scenario success predicates use this, so evaluating them never perturbs a run.
		 */
		private JsonObject blocks(JsonObject args) {
			com.google.gson.JsonArray ids = new com.google.gson.JsonArray();
			for (var element : args.getAsJsonArray("positions")) {
				com.google.gson.JsonArray position = element.getAsJsonArray();
				BlockPos pos = new BlockPos(position.get(0).getAsInt(), position.get(1).getAsInt(), position.get(2).getAsInt());
				if (!server.overworld().hasChunkAt(pos)) {
					throw new IllegalArgumentException("Block query outside loaded chunks: " + HarnessState.key(pos));
				}
				ids.add(HarnessState.blockId(server.overworld().getBlockState(pos)));
			}
			JsonObject result = time();
			result.add("blocks", ids);
			return result;
		}

		/**
		 * Removes one to 64 distinct loaded supports in request order as an actorless world failure.
		 * All positions and bot distances are checked before any removal; villagers are not excluded.
		 * Vanilla removal retains water in waterlogged supports and creates no block-item drops.
		 * The existing block-change mixin alone records changes and notifies NPCs; returned IDs come
		 * from its latest-change index, not from synthetic events or an additional logging call.
		 * @param args contains {@code positions}, an array of integer {@code [x,y,z]} arrays
		 * @return game time and {@code changes}, each containing {@code pos} and its real {@code eventId}
		 * @throws IllegalArgumentException if positions are invalid or a bot is within configured view radius
		 * @throws IllegalStateException if recording is disabled or a removal or its recording fails
		 */
		private JsonObject supportFailure(JsonObject args) {
			HarnessState state = HarnessState.get(server);
			if (!state.recording()) {
				throw new IllegalStateException("support_failure requires recording");
			}
			if (!args.has("positions") || !args.get("positions").isJsonArray()) {
				throw new IllegalArgumentException("support_failure requires positions");
			}
			JsonArray requested = args.getAsJsonArray("positions");
			if (requested.isEmpty() || requested.size() > 64) {
				throw new IllegalArgumentException("support_failure requires 1 to 64 positions");
			}
			var level = server.overworld();
			var positions = new ArrayList<BlockPos>();
			var unique = new HashSet<BlockPos>();
			double radius = state.config().number("view_radius");
			for (var element : requested) {
				if (!element.isJsonArray()) {
					throw new IllegalArgumentException("Each support position must be [x,y,z]");
				}
				JsonObject coordinates = new JsonObject();
				coordinates.add("pos", element);
				BlockPos position = position(coordinates);
				if (!unique.add(position) || !level.hasChunkAt(position)) {
					throw new IllegalArgumentException("Duplicate or unloaded support: " + HarnessState.key(position));
				}
				var block = level.getBlockState(position);
				if (block.isAir() || block.equals(block.getFluidState().createLegacyBlock())) {
					throw new IllegalArgumentException("No support at " + HarnessState.key(position));
				}
				for (SimBot bot : bots.all()) {
					if (bot.level() == level && !bot.isRemoved() && bot.position().distanceTo(Vec3.atCenterOf(position)) <= radius) {
						throw new IllegalArgumentException("Bot within view_radius of support: " + bot.getGameProfile().name());
					}
				}
				positions.add(position);
			}
			return HarnessState.acting(null, () -> {
				JsonArray changes = new JsonArray();
				for (BlockPos position : positions) {
					String key = HarnessState.key(position);
					var previous = state.map("latestChanges").get(key);
					if (!level.removeBlock(position, false)) {
						throw new IllegalStateException("Support removal failed at " + key);
					}
					var eventId = state.map("latestChanges").get(key);
					if (eventId == null || previous != null && eventId.getAsLong() <= previous.getAsLong()) {
						throw new IllegalStateException("Support removal was not recorded at " + key);
					}
					JsonObject change = new JsonObject();
					change.add("pos", HarnessState.position(position));
					change.add("eventId", eventId.deepCopy());
					changes.add(change);
				}
				JsonObject result = time();
				result.add("changes", changes);
				return result;
			});
		}

		/**
		 * Lists a container's items, with written books' titles, authors and pages, without exposure logging or any
		 * state change. For run evaluation (e.g. reading filed reports), never for agents: agents read containers
		 * through {@code open_container}, which enforces reach, sight and access.
		 */
		private JsonObject containerItems(JsonObject args) {
			HarnessState harness = HarnessState.existing(server);
			BlockPos pos = position(args);
			if (!server.overworld().hasChunkAt(pos) || !(server.overworld().getBlockEntity(pos) instanceof Container container)
					|| container instanceof RandomizableContainer loot && loot.getLootTable() != null) {
				throw new IllegalArgumentException("No readable container at " + HarnessState.key(pos));
			}
			com.google.gson.JsonArray items = new com.google.gson.JsonArray();
			for (int slot = 0; slot < container.getContainerSize(); slot++) {
				ItemStack stack = container.getItem(slot);
				if (stack.isEmpty()) {
					continue;
				}
				JsonObject item = new JsonObject();
				item.addProperty("slot", slot);
				item.addProperty("item", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
				item.addProperty("count", stack.getCount());
				if (harness != null) {
					Freshness.describe(stack, harness.config(), Math.floorDiv(server.overworld().getGameTime(), 24_000L), item);
				}
				WrittenBookContent book = stack.get(DataComponents.WRITTEN_BOOK_CONTENT);
				if (book != null) {
					item.addProperty("title", book.title().raw());
					item.addProperty("author", book.author());
					com.google.gson.JsonArray pages = new com.google.gson.JsonArray();
					book.pages().forEach(page -> pages.add(page.raw().getString()));
					item.add("pages", pages);
				}
				items.add(item);
			}
			JsonObject result = time();
			result.add("items", items);
			return result;
		}

		private JsonObject reseed(JsonObject args) {
			Determinism.reseed(server, integer(args, "base_seed"));
			return time();
		}

		private JsonObject command(JsonObject args) {
			int[] commandResult = {0};
			boolean[] successful = {false};
			HarnessState.acting("system", () -> {
				server.getCommands().performPrefixedCommand(server.createCommandSourceStack().withSuppressedOutput()
					.withCallback((success, value) -> {
						successful[0] = success;
						commandResult[0] = value;
					}), args.get("command").getAsString());
				return null;
			});
			if (!successful[0]) {
				throw new IllegalArgumentException("Vanilla command failed: " + args.get("command").getAsString());
			}
			JsonObject result = time();
			result.addProperty("value", commandResult[0]);
			return result;
		}
	}

	/** Bot inventory, input, actions and skills; preserves validation and acting order. */
	private final class BotsSkillsHandlers {
		private JsonObject spawn(JsonObject args) {
			return bots.spawn(args);
		}

		private JsonObject state(JsonObject args) {
			return bots.state(bot(args));
		}

		private JsonObject startSkill(JsonObject args) {
			boolean dryRun = args.has("dry_run") && booleanValue(args, "dry_run");
			return bot(args).skills().start(args.get("skill").getAsString(), args.getAsJsonObject("args"), dryRun);
		}

		private JsonObject skillStatus(JsonObject args) {
			return bot(args).skills().status();
		}

		private JsonObject cancelSkill(JsonObject args) {
			return bot(args).skills().cancel();
		}

		private JsonObject input(JsonObject args) {
			SimBot bot = bot(args);
			bot.setInput(BotInput.fromJson(args));
			return bots.state(bot);
		}

		private JsonObject breakBlock(JsonObject args) {
			SimBot bot = bot(args);
			JsonObject result = time();
			result.addProperty("destroyed", HarnessState.acting(bot.getGameProfile().name(), () -> bot.gameMode.destroyBlock(position(args))));
			return result;
		}

		private JsonObject placeBlock(JsonObject args) {
			SimBot bot = bot(args);
			BlockPos position = position(args);
			Direction face = Direction.valueOf(args.get("face").getAsString().toUpperCase(Locale.ROOT));
			Vec3 hit = Vec3.atCenterOf(position).add(face.getStepX() * 0.5, face.getStepY() * 0.5, face.getStepZ() * 0.5);
			var interaction = HarnessState.acting(bot.getGameProfile().name(), () ->
				bot.gameMode.useItemOn(bot, bot.level(), bot.getMainHandItem(), InteractionHand.MAIN_HAND,
					new BlockHitResult(hit, face, position, false)));
			JsonObject result = time();
			result.addProperty("consumed", interaction.consumesAction());
			return result;
		}

		private JsonObject select(JsonObject args) {
			int slot = Math.toIntExact(integer(args, "slot"));
			if (slot < 0 || slot > 8) {
				throw new IllegalArgumentException("Hotbar slot must be between 0 and 8");
			}
			SimBot bot = bot(args);
			bot.getInventory().setSelectedSlot(slot);
			return bots.state(bot);
		}

		private JsonObject give(JsonObject args) {
			int count = Math.toIntExact(integer(args, "count"));
			if (count <= 0 || count > 2304) {
				throw new IllegalArgumentException("Item count must be between 1 and 2304");
			}
			var item = BuiltInRegistries.ITEM.getOptional(Identifier.parse(args.get("item").getAsString()))
					.orElseThrow(() -> new IllegalArgumentException("Unknown item"));
			SimBot bot = bot(args);
			ItemStack stack = new ItemStack(item, count);
			bot.getInventory().add(stack);
			if (!stack.isEmpty()) {
				throw new IllegalStateException("Inventory full; " + stack.getCount() + " items did not fit");
			}
			return bots.state(bot);
		}

		private JsonObject flag(JsonObject args) {
			SimBot bot = bot(args);
			JsonObject flags = new JsonObject();
			flags.addProperty("master_key", booleanValue(args, "master_key"));
			HarnessState.get(server).map("botFlags").add(bot.getGameProfile().name(), flags);
			return flags.deepCopy();
		}
	}

	/** Observation, document and ACL commands; keeps JSON construction and acting scopes unchanged. */
	private final class ObservationDocumentHandlers {
		private JsonObject observe(JsonObject args) {
			return observation.observe(bot(args), !args.has("include_terrain") || booleanValue(args, "include_terrain"));
		}

		private JsonObject say(JsonObject args) {
			return observation.say(bot(args), args.get("text").getAsString());
		}

		private JsonObject speak(JsonObject args) {
			return bot(args).skills().start("speak", args,
				args.has("dry_run") && booleanValue(args, "dry_run"));
		}

		private JsonObject ask(JsonObject args) {
			JsonObject exchange = new JsonObject();
			exchange.add("npc", args.get("npc"));
			if (HarnessState.get(server).config().askAbout() && args.has("about")) {
				exchange.add("about", args.get("about").deepCopy());
			}
			return bot(args).skills().start("speak", exchange,
				args.has("dry_run") && booleanValue(args, "dry_run"));
		}

		private JsonObject writeSign(JsonObject args) {
			SimBot bot = bot(args);
			return HarnessState.acting(bot.getGameProfile().name(), () -> documents.writeSign(bot, args));
		}

		private JsonObject writeBook(JsonObject args) {
			SimBot bot = bot(args);
			return HarnessState.acting(bot.getGameProfile().name(), () -> documents.writeBook(bot, args));
		}

		private JsonObject readItem(JsonObject args) {
			return documents.readItem(bot(args), args);
		}

		private JsonObject openContainer(JsonObject args) {
			return documents.openContainer(bot(args), args);
		}

		/** Configures ownership separately from a nullable lock, validating both before changing canonical state. */
		private JsonObject setAcl(JsonObject args) {
			var allowed = Documents.allowed(args.get("allowed"));
			String owner = Documents.owner(args.get("owner"));
			String key = HarnessState.key(position(args));
			HarnessState state = HarnessState.get(server);
			if (allowed.isJsonNull()) {
				state.map("acls").remove(key);
			} else {
				state.map("acls").add(key, allowed);
			}
			if (args.has("owner")) {
				if (owner != null) {
					if (state.map("containerOwners") == null) {
						state.putMap("containerOwners");
					}
					state.map("containerOwners").addProperty(key, owner);
				} else if (state.map("containerOwners") != null) {
					state.map("containerOwners").remove(key);
				}
			}
			return time();
		}

		/**
		 * Registers (or, with empty prices, removes) a shop chest. A shop sells only through the {@code buy} skill at
		 * these emerald prices; only its explicit owner may transfer stock directly when goods rules are enabled.
		 */
		private JsonObject setShop(JsonObject args) {
			JsonObject prices = args.getAsJsonObject("prices");
			JsonObject validated = new JsonObject();
			for (var entry : prices.entrySet()) {
				BuiltInRegistries.ITEM.getOptional(Identifier.parse(entry.getKey()))
					.orElseThrow(() -> new IllegalArgumentException("Unknown item: " + entry.getKey()));
				int price = entry.getValue().getAsBigDecimal().intValueExact();
				if (price < 0) {
					throw new IllegalArgumentException("Prices must be nonnegative");
				}
				validated.addProperty(entry.getKey(), price);
			}
			String key = HarnessState.key(position(args));
			HarnessState state = HarnessState.get(server);
			if (state.map("shops") == null) {
				state.putMap("shops");
			}
			if (validated.isEmpty()) {
				state.map("shops").remove(key);
			} else {
				state.map("shops").add(key, validated);
			}
			return time();
		}

		private JsonObject clearAcl(JsonObject args) {
			HarnessState.get(server).map("acls").remove(HarnessState.key(position(args)));
			return time();
		}
	}

	/** NPC setup and queries; resolves current villagers per call, including setup-before-query failures. */
	private final class NpcHandlers {
		private JsonObject setup(JsonObject args) {
			return HarnessState.get(server).setupVillagers(args);
		}

		private JsonObject metrics(JsonObject args) {
			return villagers().metrics(args);
		}

		private JsonObject state(JsonObject args) {
			return villagers().state(args.get("id").getAsString());
		}

		private JsonObject testimony(JsonObject args) throws IOException {
			return villagers().testimony(args);
		}

		private Villagers villagers() {
			Villagers villagers = HarnessState.get(server).villagers();
			if (villagers == null) {
				throw new IllegalStateException("npc_setup is required");
			}
			return villagers;
		}
	}

	/** Configuration, recording and snapshots; world saving still precedes bot saving and the time read. */
	private final class PersistenceHandlers {
		private JsonObject config(JsonObject args) {
			return HarnessState.get(server).config().json();
		}

		private JsonObject setConfig(JsonObject args) {
			HarnessState.get(server).config().update(args);
			return HarnessState.get(server).config().json();
		}

		private JsonObject recording(JsonObject args) {
			HarnessState.get(server).recording(booleanValue(args, "enabled"));
			JsonObject result = time();
			result.addProperty("enabled", HarnessState.get(server).recording());
			return result;
		}

		private JsonObject saveSnapshot(JsonObject args) throws IOException {
			server.saveEverything(false, true, true);
			bots.save();
			return time();
		}

		private JsonObject restoreBots(JsonObject args) throws IOException {
			return bots.restore();
		}
	}

	private SimBot bot(JsonObject args) {
		return bots.get(args.get("name").getAsString());
	}

	/**
	 * Reads a block position from x/y/z fields or a nested object/three-element pos array.
	 * Coordinate conversion keeps x, then y, then z evaluation and consumes no RNG.
	 *
	 * @param args coordinates or an object containing {@code pos}
	 * @return block position using exact integer and int-range conversion
	 * @throws IllegalArgumentException if the array size or coordinate values are invalid
	 * @throws ArithmeticException if a coordinate is fractional or outside the supported range
	 * @throws NullPointerException if required coordinate data is missing
	 */
	public static BlockPos position(JsonObject args) {
		if (args.has("pos")) {
			var value = args.get("pos");
			if (value.isJsonObject()) {
				return position(value.getAsJsonObject());
			}
			var array = value.getAsJsonArray();
			if (array.size() != 3) {
				throw new IllegalArgumentException("pos requires three coordinates");
			}
			JsonObject coordinates = new JsonObject();
			coordinates.add("x", array.get(0));
			coordinates.add("y", array.get(1));
			coordinates.add("z", array.get(2));
			return position(coordinates);
		}
		return new BlockPos(Math.toIntExact(integer(args, "x")), Math.toIntExact(integer(args, "y")), Math.toIntExact(integer(args, "z")));
	}

	private static boolean booleanValue(JsonObject args, String key) {
		var value = args.get(key);
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean()) {
			throw new IllegalArgumentException(key + " must be a boolean");
		}
		return value.getAsBoolean();
	}

	private static long integer(JsonObject args, String key) {
		var value = args.get(key);
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException(key + " must be an integer");
		}
		return value.getAsBigDecimal().longValueExact();
	}

	private JsonObject time() {
		JsonObject result = new JsonObject();
		result.addProperty("gameTime", server.overworld().getGameTime());
		return result;
	}
}
