package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.mojang.serialization.JsonOps;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import net.minecraft.core.BlockPos;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.network.Filterable;
import net.minecraft.world.Container;
import net.minecraft.world.RandomizableContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.component.CustomData;
import net.minecraft.world.item.component.WritableBookContent;
import net.minecraft.world.item.component.WrittenBookContent;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.entity.LecternBlockEntity;
import net.minecraft.world.level.block.entity.SignBlockEntity;
import net.minecraft.world.level.block.entity.SignText;
import net.minecraft.world.level.block.entity.SignTextSlot;
import net.minecraft.world.phys.Vec3;

/**
 * Handles managed-overworld bot documents, container reads and their provenance/audit records.
 * Access is server-main-thread-only; reads never resolve loot or load chunks, and
 * document versions, content hashes and slot order retain their canonical encodings.
 */
public final class Documents {
	private static final String DOCUMENT_TAG = "simharness_doc";
	private final MinecraftServer server;
	private final BotManager bots;

	/**
	 * Binds document operations to one server and its managed bots.
	 * @param server owning server
	 * @param bots manager used to validate bot identity
	 * @throws NullPointerException if either dependency is null
	 */
	public Documents(MinecraftServer server, BotManager bots) {
		this.server = Objects.requireNonNull(server);
		this.bots = Objects.requireNonNull(bots);
	}

	/**
	 * Writes reachable visible sign text and records the next document version.
	 * @param bot managed overworld author, accessed on the server thread
	 * @param args position, four front lines and optional four back lines
	 * @return new sign-version provenance object
	 * @throws IllegalArgumentException if bot, text, target or reach/visibility is invalid
	 * @throws IllegalStateException if called off the server main thread
	 * @throws ArithmeticException if the version counter overflows
	 */
	public JsonObject writeSign(SimBot bot, JsonObject args) {
		HarnessState state = state(bot);
		BlockPos position = ControlCommands.position(args);
		List<String> front = strings(args, "front", SignText.LINES);
		List<String> back = args.has("back") ? strings(args, "back", SignText.LINES) : null;
		requireReach(bot, position, state);
		if (!(bot.level().getBlockEntity(position) instanceof SignBlockEntity sign)) {
			throw new IllegalArgumentException("No sign at " + HarnessState.key(position));
		}
		SignText frontText = signText(sign.getText(SignTextSlot.FRONT), front);
		SignText backText = back == null ? sign.getText(SignTextSlot.BACK) : signText(sign.getText(SignTextSlot.BACK), back);
		JsonObject text = signContents(frontText, backText);
		String docId = "sign:" + HarnessState.key(position);
		long version = Math.incrementExact(version(state.map("docVersions"), docId));
		String digest = HarnessState.sha256(HarnessState.canonical(text));
		JsonObject event = state.event("doc_write");
		event.addProperty("author", name(bot));
		event.addProperty("kind", "sign");
		event.add("pos", HarnessState.position(position));
		event.addProperty("contentSha256", digest);
		sign.setText(frontText, SignTextSlot.FRONT);
		if (back != null) {
			sign.setText(backText, SignTextSlot.BACK);
		}
		return recordWrite(state, event, docId, version);
	}

	/**
	 * Replaces an owned writable book with signed content and records its initial version.
	 * @param bot managed overworld author, accessed on the server thread
	 * @param args inventory slot, title and page strings
	 * @return written book's provenance object
	 * @throws IllegalArgumentException if the bot, slot, book or encoded content is invalid
	 * @throws IllegalStateException if called off the server main thread
	 */
	public JsonObject writeBook(SimBot bot, JsonObject args) {
		HarnessState state = state(bot);
		int slot = inventorySlot(bot, args);
		String title = string(args.get("title"), "title");
		if (title.length() > WrittenBookContent.TITLE_MAX_LENGTH) {
			throw new IllegalArgumentException("title must be at most " + WrittenBookContent.TITLE_MAX_LENGTH + " characters");
		}
		List<String> pages = strings(args, "pages", -1);
		if (pages.size() > WritableBookContent.MAX_PAGES) {
			throw new IllegalArgumentException("pages must contain at most " + WritableBookContent.MAX_PAGES + " entries");
		}
		List<Filterable<Component>> writtenPages = new ArrayList<>(pages.size());
		JsonArray pageJson = new JsonArray();
		for (String page : pages) {
			if (page.length() > WrittenBookContent.PAGE_LENGTH) {
				throw new IllegalArgumentException("Book page exceeds " + WrittenBookContent.PAGE_LENGTH + " characters");
			}
			writtenPages.add(Filterable.<Component>passThrough(Component.literal(page)));
			pageJson.add(page);
		}
		ItemStack current = bot.getInventory().getItem(slot);
		if (current.isEmpty() || !current.is(Items.WRITABLE_BOOK)) {
			String held = current.is(Items.WRITTEN_BOOK) ? " (it holds a signed written_book, which cannot be edited)" : "";
			String none = bot.getInventory().contains(stack -> stack.is(Items.WRITABLE_BOOK)) ? "" : "; you carry no writable_book";
			throw new IllegalArgumentException("Inventory slot " + slot + " must contain a writable_book" + held + none);
		}
		WrittenBookContent content = new WrittenBookContent(Filterable.passThrough(title), name(bot), 0, List.copyOf(writtenPages), true);
		WrittenBookContent.CODEC.encodeStart(server.registryAccess().createSerializationContext(JsonOps.INSTANCE), content)
			.getOrThrow(message -> new IllegalArgumentException("Invalid book content: " + message));
		ItemStack replacement = current.transmuteCopy(Items.WRITTEN_BOOK);
		replacement.remove(DataComponents.WRITABLE_BOOK_CONTENT);
		replacement.set(DataComponents.WRITTEN_BOOK_CONTENT, content);
		encodedStack(replacement);
		String digest = HarnessState.sha256(HarnessState.canonical(pageJson));
		JsonObject event = state.event("doc_write");
		String docId = "book:" + event.get("id").getAsLong();
		CustomData.update(DataComponents.CUSTOM_DATA, replacement, tag -> tag.putString(DOCUMENT_TAG, docId));
		event.addProperty("author", name(bot));
		event.addProperty("kind", "book");
		event.addProperty("item", "minecraft:written_book");
		event.addProperty("slot", slot);
		event.addProperty("title", title);
		event.addProperty("contentSha256", digest);
		bot.getInventory().setItem(slot, replacement);
		bot.getInventory().setChanged();
		return recordWrite(state, event, docId, 1L);
	}

	/**
	 * Reads an owned written book and appends its inventory exposure audit.
	 * @param bot managed overworld reader, accessed on the server thread
	 * @param args inventory slot to read
	 * @return book title, author, pages and provenance
	 * @throws IllegalArgumentException if the bot, slot or written-book content is invalid
	 * @throws IllegalStateException if called off the server main thread
	 */
	public JsonObject readItem(SimBot bot, JsonObject args) {
		HarnessState state = state(bot);
		int slot = inventorySlot(bot, args);
		ItemStack stack = bot.getInventory().getItem(slot);
		if (stack.isEmpty() || !stack.is(Items.WRITTEN_BOOK) || !stack.has(DataComponents.WRITTEN_BOOK_CONTENT)) {
			throw new IllegalArgumentException("Inventory slot " + slot + " must contain a written_book");
		}
		JsonObject result = bookDocument(state, stack);
		JsonArray exposures = new JsonArray();
		exposures.add(inventoryExposure(bot, slot, result, state.config().number("reach")));
		state.exposure(name(bot), bot.getEyePosition(), "read_item_doc", exposures);
		return result;
	}

	/**
	 * Reads a visible reachable ACL-permitted container without generating unresolved loot.
	 * Content changes advance the tracked version before item/document exposure is recorded.
	 * @param bot managed overworld reader, accessed on the server thread
	 * @param args container position
	 * @return container provenance and nonempty items in slot order
	 * @throws IllegalArgumentException if access, target, reach, loot state or item encoding is invalid
	 * @throws IllegalStateException if called off the server main thread
	 * @throws ArithmeticException if the container version overflows
	 */
	public JsonObject openContainer(SimBot bot, JsonObject args) {
		HarnessState state = state(bot);
		BlockPos position = ControlCommands.position(args);
		Sight.Ray ray = requireReach(bot, position, state);
		if (!permitted(state, bot, position)) {
			throw new IllegalArgumentException("Container ACL denies access for " + name(bot) + " at " + HarnessState.key(position));
		}
		BlockEntity entity = bot.level().getBlockEntity(position);
		if (!(entity instanceof Container container)) {
			throw new IllegalArgumentException("No container at " + HarnessState.key(position));
		}
		if (container instanceof RandomizableContainer randomizable && randomizable.getLootTable() != null) {
			throw new IllegalArgumentException("Container has unresolved loot; opening it would generate loot and consume world RNG");
		}
		int size = container.getContainerSize();
		List<ItemStack> stacks = new ArrayList<>(size);
		JsonArray contents = new JsonArray();
		for (int slot = 0; slot < size; slot++) {
			ItemStack stack = container.getItem(slot);
			stacks.add(stack);
			JsonObject item = new JsonObject();
			item.addProperty("slot", slot);
			item.add("stack", stack.isEmpty() ? JsonNull.INSTANCE : encodedStack(stack));
			contents.add(item);
		}
		String hash = HarnessState.sha256(HarnessState.canonical(contents));
		String key = HarnessState.key(position);
		JsonObject previous = state.map("containers").getAsJsonObject(key);
		long version = previous == null ? 1L : previous.get("version").getAsLong();
		if (previous != null && !hash.equals(previous.get("hash").getAsString())) {
			version = Math.incrementExact(version);
		}
		JsonObject tracked = new JsonObject();
		tracked.addProperty("hash", hash);
		tracked.addProperty("version", version);
		state.map("containers").add(key, tracked);
		String prov = "container:" + key + "@" + version;
		JsonObject origin = new JsonObject();
		origin.add("pos", HarnessState.position(position));
		origin.addProperty("version", version);
		origin.addProperty("contentSha256", hash);
		state.origin(prov, "container", origin);
		JsonArray items = new JsonArray();
		for (int slot = 0; slot < size; slot++) {
			ItemStack stack = stacks.get(slot);
			if (!stack.isEmpty()) {
				items.add(inventoryItem(bot, slot, stack));
			}
		}
		JsonObject result = new JsonObject();
		result.addProperty("prov", prov);
		result.add("items", items);
		JsonObject exposure = result.deepCopy();
		spatialExposure(exposure, "container", bot, position, state.config().number("reach"), ray);
		JsonArray exposures = new JsonArray();
		exposures.add(exposure);
		for (JsonElement element : exposure.getAsJsonArray("items")) {
			JsonObject item = element.getAsJsonObject();
			if (item.has("doc")) {
				JsonObject document = item.getAsJsonObject("doc").deepCopy();
				spatialExposure(document, "document", bot, position, state.config().number("reach"), ray);
				document.add("slot", item.get("slot"));
				document.addProperty("containerProv", prov);
				exposures.add(document);
			}
		}
		state.exposure(name(bot), bot.getEyePosition(), "open_container", exposures);
		return result;
	}

	/**
	 * Describes an inventory stack, adding written-book metadata without page contents.
	 * @param bot managed overworld owner, accessed on the server thread
	 * @param slot slot index to include in the response
	 * @param stack stack to describe
	 * @return item/count/slot JSON and optional document metadata
	 * @throws IllegalArgumentException if the bot is not a managed overworld bot
	 * @throws IllegalStateException if called off the server main thread
	 */
	public JsonObject inventoryItem(SimBot bot, int slot, ItemStack stack) {
		HarnessState state = state(bot);
		JsonObject result = new JsonObject();
		result.addProperty("slot", slot);
		result.addProperty("item", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
		result.addProperty("count", stack.getCount());
		Freshness.describe(stack, state.config(), Math.floorDiv(bot.level().getGameTime(), 24_000L), result);
		if (!stack.isEmpty() && stack.is(Items.WRITTEN_BOOK) && stack.has(DataComponents.WRITTEN_BOOK_CONTENT)) {
			JsonObject document = bookDocument(state, stack);
			document.remove("pages");
			result.add("doc", document);
		}
		return result;
	}

	/**
	 * Resolves loaded sign/lectern content and its provenance; the caller performs sight/reach admission.
	 * @param bot managed overworld reader, accessed on the server thread
	 * @param position document block position
	 * @return document content and position, or null if unloaded or not a supported document
	 * @throws IllegalArgumentException if bot identity or lectern book content is invalid
	 * @throws IllegalStateException if called off the server main thread
	 */
	public JsonObject document(SimBot bot, BlockPos position) {
		HarnessState state = state(bot);
		if (!bot.level().hasChunkAt(position)) {
			return null;
		}
		BlockEntity entity = bot.level().getBlockEntity(position);
		JsonObject result;
		if (entity instanceof SignBlockEntity sign) {
			JsonObject text = signContents(sign.getText(SignTextSlot.FRONT), sign.getText(SignTextSlot.BACK));
			String docId = "sign:" + HarnessState.key(position);
			long version = version(state.map("docVersions"), docId);
			String prov = provenance(docId, version);
			JsonObject details = new JsonObject();
			details.add("pos", HarnessState.position(position));
			details.addProperty("docId", docId);
			details.addProperty("version", version);
			details.addProperty("contentSha256", HarnessState.sha256(HarnessState.canonical(text)));
			state.map("docVersions").addProperty(docId, version);
			state.origin(prov, "document", details);
			result = new JsonObject();
			result.addProperty("prov", prov);
			result.addProperty("kind", "sign");
			result.add("text", text);
		} else if (entity instanceof LecternBlockEntity lectern && lectern.hasBook()) {
			result = bookDocument(state, lectern.getBook());
			result.addProperty("kind", "lectern");
		} else {
			return null;
		}
		result.add("pos", HarnessState.position(position));
		return result;
	}

	/**
	 * Appends owned-document audits for document-bearing inventory response entries.
	 * @param bot managed overworld owner, accessed on the server thread
	 * @param inventory ordered inventory entries already carrying document metadata
	 * @param exposures destination audit array, mutated in inventory order
	 * @throws IllegalArgumentException if the bot is not a managed overworld bot
	 * @throws IllegalStateException if called off the server main thread
	 */
	public void appendInventoryExposures(SimBot bot, JsonArray inventory, JsonArray exposures) {
		HarnessState state = state(bot);
		for (JsonElement element : inventory) {
			JsonObject item = element.getAsJsonObject();
			if (item.has("doc")) {
				exposures.add(inventoryExposure(bot, item.get("slot").getAsInt(), item.getAsJsonObject("doc"), state.config().number("reach")));
			}
		}
	}

	private HarnessState state(SimBot bot) {
		if (!server.isSameThread()) {
			throw new IllegalStateException("Documents must be accessed on the server main thread");
		}
		if (bots.get(name(bot)) != bot) {
			throw new IllegalArgumentException("Bot is not managed by this harness");
		}
		if (bot.level() != server.overworld()) {
			throw new IllegalArgumentException("Documents and containers are limited to the overworld");
		}
		return HarnessState.get(server);
	}

	private Sight.Ray requireReach(SimBot bot, BlockPos position, HarnessState state) {
		double distance = bot.getEyePosition().distanceTo(Vec3.atCenterOf(position));
		if (!Double.isFinite(distance) || distance > state.config().number("reach")) {
			throw new IllegalArgumentException("Target is outside document/container reach");
		}
		if (!bot.level().hasChunkAt(position)) {
			throw new IllegalArgumentException("Target chunk is not loaded");
		}
		Sight.Ray ray = Sight.block(bot, position);
		if (ray == null) {
			throw new IllegalArgumentException("Target is not visible");
		}
		return ray;
	}

	/**
	 * Checks master-key and position ACL policy without mutating it.
	 * @param state current harness state
	 * @param bot bot requesting access
	 * @param position ACL-protected block position
	 * @return true for a master key, absent ACL, wildcard ACL or explicit name match
	 */
	public static boolean permitted(HarnessState state, SimBot bot, BlockPos position) {
		JsonObject flags = state.map("botFlags").getAsJsonObject(name(bot));
		if (flags != null && flags.has("master_key") && flags.get("master_key").getAsBoolean()) {
			return true;
		}
		return permitted(state.map("acls").get(HarnessState.key(position)), name(bot));
	}

	/** Checks only a position's lock; ownership never grants or denies access. */
	static boolean permitted(JsonElement allowed, String name) {
		if (allowed == null || allowed.isJsonNull()) {
			return true;
		}
		if (allowed.isJsonPrimitive() && allowed.getAsJsonPrimitive().isString()) {
			return allowed.getAsString().equals("*");
		}
		if (allowed.isJsonArray()) {
			for (JsonElement entry : allowed.getAsJsonArray()) {
				if (entry.isJsonPrimitive() && entry.getAsJsonPrimitive().isString() && entry.getAsString().equals(name)) {
					return true;
				}
			}
		}
		return false;
	}

	/** Validates and sorts a lock without mutating its source; JSON null explicitly leaves the container unlocked. */
	static JsonElement allowed(JsonElement value) {
		if (value == null || !(value.isJsonNull() || value.isJsonArray() || value.isJsonPrimitive()
				&& value.getAsJsonPrimitive().isString() && value.getAsString().equals("*"))) {
			throw new IllegalArgumentException("allowed must be null, '*' or an array of bot names");
		}
		if (!value.isJsonArray()) {
			return value.deepCopy();
		}
		java.util.Set<String> names = new java.util.TreeSet<>();
		for (JsonElement entry : value.getAsJsonArray()) {
			if (!entry.isJsonPrimitive() || !entry.getAsJsonPrimitive().isString()
					|| !entry.getAsString().matches("[A-Za-z0-9_]{1,16}")) {
				throw new IllegalArgumentException("ACL entries must be bot names");
			}
			names.add(entry.getAsString());
		}
		JsonArray sorted = new JsonArray();
		names.forEach(sorted::add);
		return sorted;
	}

	/** Validates an optional container owner, with JSON null clearing ownership independently of its lock. */
	static String owner(JsonElement value) {
		if (value == null || value.isJsonNull()) {
			return null;
		}
		if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()
				|| !value.getAsString().matches("[A-Za-z0-9_]{1,16}")) {
			throw new IllegalArgumentException("Owner must be a bot name or null");
		}
		return value.getAsString();
	}

	/** Annotates a successful take without turning an owner's claim into a lock or changing unowned transfers. */
	static void taken(JsonObject event, JsonElement owner, String actor) {
		if (owner != null && !owner.isJsonNull()) {
			event.add("owner", owner.deepCopy());
			event.addProperty("taken_by_non_owner", !owner.getAsString().equals(actor));
		}
	}

	/** An explicitly owned shop allows only its named owner to use direct stock transfers. */
	static boolean ownsStall(JsonElement owner, String actor) {
		return owner != null && !owner.isJsonNull() && owner.getAsString().equals(actor);
	}

	private JsonElement encodedStack(ItemStack stack) {
		return ItemStack.CODEC.encodeStart(server.registryAccess().createSerializationContext(JsonOps.INSTANCE), stack)
			.getOrThrow(message -> new IllegalArgumentException("Cannot encode item stack: " + message));
	}

	private static JsonObject recordWrite(HarnessState state, JsonObject event, String docId, long version) {
		String prov = provenance(docId, version);
		event.addProperty("docId", docId);
		event.addProperty("version", version);
		event.addProperty("prov", prov);
		state.map("docVersions").addProperty(docId, version);
		state.map("origins").add(prov, event.get("id").deepCopy());
		state.logEvent(event);
		JsonObject result = new JsonObject();
		result.addProperty("prov", prov);
		return result;
	}

	private static JsonObject bookDocument(HarnessState state, ItemStack stack) {
		JsonObject result = bookContents(stack);
		JsonArray pages = result.getAsJsonArray("pages");
		String digest = HarnessState.sha256(HarnessState.canonical(result));
		String storedId = stack.getOrDefault(DataComponents.CUSTOM_DATA, CustomData.EMPTY).copyTag().getString(DOCUMENT_TAG).orElse(null);
		String docId;
		JsonObject baselineOrigin = null;
		if (storedId != null && storedId.matches("book:[1-9][0-9]*") && state.map("docVersions").has(storedId)) {
			docId = storedId;
		} else {
			JsonElement externalId = state.map("externalDocs").get(digest);
			if (externalId == null) {
				baselineOrigin = state.event("origin");
				docId = "book:" + baselineOrigin.get("id").getAsLong();
				state.map("externalDocs").addProperty(digest, docId);
			} else {
				docId = externalId.getAsString();
			}
		}
		long version = version(state.map("docVersions"), docId);
		String prov = provenance(docId, version);
		JsonObject details = new JsonObject();
		details.addProperty("docId", docId);
		details.addProperty("version", version);
		details.addProperty("contentSha256", HarnessState.sha256(HarnessState.canonical(pages)));
		state.map("docVersions").addProperty(docId, version);
		if (baselineOrigin == null) {
			state.origin(prov, "document", details);
		} else {
			baselineOrigin.addProperty("kind", "document");
			baselineOrigin.addProperty("prov", prov);
			for (var detail : details.entrySet()) {
				baselineOrigin.add(detail.getKey(), detail.getValue().deepCopy());
			}
			state.map("origins").add(prov, baselineOrigin.get("id").deepCopy());
			state.logEvent(baselineOrigin);
		}
		result.addProperty("prov", prov);
		return result;
	}

	/** Reads book content without changing its JSON fields, page order or raw text. */
	static JsonObject bookContents(ItemStack stack) {
		JsonObject result = new JsonObject();
		JsonArray pages = new JsonArray();
		WrittenBookContent written = stack.get(DataComponents.WRITTEN_BOOK_CONTENT);
		if (written != null) {
			result.addProperty("title", written.title().raw());
			result.addProperty("author", written.author());
			written.pages().forEach(page -> pages.add(page.raw().getString()));
		} else {
			WritableBookContent writable = stack.get(DataComponents.WRITABLE_BOOK_CONTENT);
			if (writable == null) {
				throw new IllegalArgumentException("Item has no book content");
			}
			writable.pages().forEach(page -> pages.add(page.raw()));
		}
		result.add("pages", pages);
		return result;
	}

	static long version(JsonObject versions, String docId) {
		JsonElement version = versions.get(docId);
		return version == null ? 0L : version.getAsLong();
	}

	static String provenance(String docId, long version) {
		return "doc:" + docId + "@v" + version;
	}

	private static JsonObject signContents(SignText front, SignText back) {
		JsonObject result = new JsonObject();
		JsonArray frontLines = new JsonArray();
		JsonArray backLines = new JsonArray();
		front.getMessages(false).forEach(line -> frontLines.add(line.getString()));
		back.getMessages(false).forEach(line -> backLines.add(line.getString()));
		result.add("front", frontLines);
		result.add("back", backLines);
		return result;
	}

	private static SignText signText(SignText current, List<String> lines) {
		SignText.Mutable text = current.asMutable();
		for (int line = 0; line < SignText.LINES; line++) {
			text.setLine(line, Component.literal(lines.get(line)));
		}
		return text.asImmutable();
	}

	private static JsonObject inventoryExposure(SimBot bot, int slot, JsonObject document, double reach) {
		JsonObject result = document.deepCopy();
		result.addProperty("category", "inventory_document");
		result.addProperty("ownedBy", name(bot));
		result.addProperty("slot", slot);
		result.addProperty("distance", 0.0);
		result.addProperty("inRadius", true);
		result.addProperty("reachOk", true);
		result.add("losRay", JsonNull.INSTANCE);
		result.addProperty("radius", reach);
		return result;
	}

	private static void spatialExposure(JsonObject item, String category, SimBot bot, BlockPos position, double reach, Sight.Ray ray) {
		item.addProperty("category", category);
		item.add("pos", HarnessState.position(position));
		item.addProperty("distance", bot.getEyePosition().distanceTo(Vec3.atCenterOf(position)));
		item.addProperty("inRadius", true);
		item.addProperty("reachOk", true);
		item.addProperty("radius", reach);
		ray.audit(item);
	}

	private static int inventorySlot(SimBot bot, JsonObject args) {
		JsonElement value = args.get("slot");
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
			throw new IllegalArgumentException("slot must be an integer inventory slot");
		}
		int slot;
		try {
			slot = value.getAsBigDecimal().intValueExact();
		} catch (ArithmeticException | NumberFormatException failure) {
			throw new IllegalArgumentException("slot must be an integer inventory slot", failure);
		}
		if (slot < 0 || slot >= bot.getInventory().getContainerSize()) {
			throw new IllegalArgumentException("slot must be between 0 and " + (bot.getInventory().getContainerSize() - 1));
		}
		return slot;
	}

	static List<String> strings(JsonObject args, String key, int requiredSize) {
		JsonElement value = args.get(key);
		if (value == null || !value.isJsonArray()) {
			throw new IllegalArgumentException(key + " must be an array of strings");
		}
		JsonArray array = value.getAsJsonArray();
		if (requiredSize >= 0 && array.size() != requiredSize) {
			throw new IllegalArgumentException(key + " must contain exactly " + requiredSize + " strings");
		}
		List<String> result = new ArrayList<>(array.size());
		for (JsonElement entry : array) {
			result.add(string(entry, key + " entry"));
		}
		return result;
	}

	private static String string(JsonElement value, String label) {
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
			throw new IllegalArgumentException(label + " must be a string");
		}
		return value.getAsString();
	}

	private static String name(SimBot bot) {
		return bot.getGameProfile().name();
	}
}
