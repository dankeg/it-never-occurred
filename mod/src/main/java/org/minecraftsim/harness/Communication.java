package org.minecraftsim.harness;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import net.minecraft.core.BlockPos;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;

/** Runs timed utterances, endpoint hearing, attention arbitration and occupied exchanges. */
final class Communication {
	/**
	 * Longest utterance, about 100 words. Length already costs time (about 3 words per second) and listeners must stay
	 * in range for all of it, so the cap only stops monologues; 256 rejected ordinary 45-word sentences.
	 */
	static final int MAX_SPEECH_CHARS = 600;
	static final double WHISPER_RADIUS = 3.0;
	static final double WHISPER_CHATTER_RADIUS = 8.0;

	private final HarnessState harness;

	/** Binds the controller without accessing the not-yet-initialized canonical state. */
	Communication(HarnessState harness) {
		this.harness = harness;
	}

	/** Checks speech without allocating IDs, queues, cooldowns or exchange reservations. */
	JsonObject check(SimBot bot, JsonObject args) {
		String key = bot.getGameProfile().name();
		JsonObject plan = new JsonObject();
		String about = about(args, config().askAbout());
		String text = args.has("npc") ? about == null ? "What news do you have?"
			: "What have you seen or heard about " + about + "?" : string(args, "text");
		String volume = args.has("volume") ? string(args, "volume") : "talk";
		if (!List.of("whisper", "talk", "shout").contains(volume) || text.isBlank() || text.length() > MAX_SPEECH_CHARS) {
			throw new IllegalArgumentException("Speech needs nonempty text of at most " + MAX_SPEECH_CHARS
				+ " characters and whisper/talk/shout volume");
		}
		plan.addProperty("text", text);
		plan.addProperty("volume", volume);
		if (args.has("to")) {
			plan.addProperty("to", string(args, "to"));
		}
		long ticks = duration(text, volume);
		if (volume.equals("shout") && root().has("shouts") && root().getAsJsonObject("shouts").has(key)
				&& shoutCoolingDown(time(), root().getAsJsonObject("shouts").get(key).getAsLong(), config().integer("shout_cooldown_ticks"))) {
			plan.addProperty("reason", "rate_limited");
		} else if (busy(key)) {
			plan.addProperty("reason", "busy");
		}
		if (args.has("npc")) {
			if (!volume.equals("talk")) {
				throw new IllegalArgumentException("An ask exchange uses talk volume");
			}
			String npc = string(args, "npc");
			npc = npcKey(npc);
			Entity target = participants().get(npc);
			if (target == null) {
				plan.addProperty("reason", "target_not_found");
			} else if (outOfTalkRange(target.getEyePosition(), bot.getEyePosition(), config().number("talk_radius"))) {
				plan.addProperty("reason", "out_of_reach");
			} else if (busy(npc)) {
				plan.addProperty("reason", "busy");
			} else {
				JsonArray news = about == null ? harness.news().items(npc, 3) : harness.news().items(npc, 3, about);
				String answer = about == null ? answer(news, config().truthfulNews())
					: answerAbout(news, about, time(), server().overworld().getOverworldClockTime());
				plan.addProperty("npc", npc);
				plan.add("news", news);
				if (about != null) {
					recordAbout(plan, about, news);
				}
				plan.addProperty("answer", answer);
				plan.addProperty("answerTicks", duration(answer, "talk"));
				plan.addProperty("to", name(npc));
				ticks = Math.addExact(ticks, duration(answer, "talk"));
			}
		}
		plan.addProperty("ticks", ticks);
		return plan;
	}

	/** Starts an already checked bot skill, keeping ask answers private until delivered. */
	JsonObject startBot(SimBot bot, JsonObject plan) {
		String key = bot.getGameProfile().name();
		long questionTicks = plan.get("ticks").getAsLong() - (plan.has("answerTicks") ? plan.get("answerTicks").getAsLong() : 0);
		JsonObject utterance = start(key, plan.get("text").getAsString(), plan.get("volume").getAsString(),
			plan.has("to") ? plan.get("to").getAsString() : null, new JsonArray(), questionTicks);
		if (plan.has("npc")) {
			JsonObject exchange = new JsonObject();
			exchange.addProperty("requester", key);
			exchange.add("npc", plan.get("npc"));
			exchange.add("text", plan.get("answer"));
			exchange.add("news", plan.get("news").deepCopy());
			exchange.add("answerTicks", plan.get("answerTicks"));
			if (plan.has("about")) {
				recordAbout(exchange, plan.get("about").getAsString(), exchange.getAsJsonArray("news"));
			}
			utterance.add("exchange", exchange);
		}
		arbitrate();
		return utterance;
	}

	/** Starts one NPC relay without taking ownership of, or delaying, its itinerary. */
	JsonObject startNpc(String npcKey, String text, JsonArray news, String to) {
		if (busy(npcKey) || !participants().containsKey(npcKey)) {
			return null;
		}
		JsonObject utterance = start(npcKey, text, "talk", to, news);
		arbitrate();
		if (utterance.get("endsAt").getAsLong() == time()) {
			tick();
		}
		return utterance;
	}

	/** Returns whether speech or either phase of an ask owns this participant's channel. */
	boolean busy(String key) {
		return busy(root(), key);
	}

	/** Checks the same active utterance and exchange reservations without accessing a server. */
	static boolean busy(JsonObject root, String key) {
		if (!root.has("active")) {
			return false;
		}
		for (JsonElement value : root.getAsJsonObject("active").asMap().values()) {
			JsonObject utterance = value.getAsJsonObject();
			if (utterance.get("speakerKey").getAsString().equals(key)) {
				return true;
			}
			if (utterance.has("exchange")) {
				JsonObject exchange = utterance.getAsJsonObject("exchange");
				if (exchange.get("requester").getAsString().equals(key) || exchange.get("npc").getAsString().equals(key)) {
					return true;
				}
			}
		}
		return false;
	}

	/** Stops an interrupted skill without exposing its unfinished content. */
	void cancel(String key) {
		for (JsonObject utterance : active()) {
			if (utterance.get("speakerKey").getAsString().equals(key)
					|| utterance.has("exchange") && utterance.getAsJsonObject("exchange").get("requester").getAsString().equals(key)) {
				complete(utterance, true);
			}
		}
	}

	/** Completes due phases after bot and NPC motion, in increasing utterance-ID order. */
	void tick() {
		boolean completed;
		do {
			completed = false;
			for (JsonObject utterance : active()) {
				if (time() >= utterance.get("endsAt").getAsLong()) {
					complete(utterance, false);
					completed = true;
				}
			}
		} while (completed && active().stream().anyMatch(item -> item.get("endsAt").getAsLong() <= time()));
	}

	/** Lists living managed speakers/listeners in stable identity order. */
	Map<String, Entity> participants() {
		Map<String, Entity> result = new TreeMap<>();
		server().getPlayerList().getPlayers().forEach(player -> {
			if (player instanceof SimBot bot && bot.level() == server().overworld() && bot.isAlive() && !bot.isRemoved()) {
				result.put(bot.getGameProfile().name(), bot);
			}
		});
		if (harness.villagers() != null) {
			result.putAll(harness.villagers().bodies());
		}
		return result;
	}

	/** Resolves the displayed name while preserving separate stable keys for state and RNG. */
	String name(String key) {
		Entity entity = participants().get(key);
		return entity instanceof SimBot bot ? bot.getGameProfile().name() : entity == null ? key : entity.getName().getString();
	}

	/** Treats stationary idle/waiting bots and itinerary-dwelling NPCs as encounter partners. */
	boolean dwelling(String key) {
		if (key.startsWith("npc:")) {
			return harness.villagers() != null && harness.villagers().dwelling(key);
		}
		Entity entity = server().getPlayerList().getPlayerByName(key);
		if (entity instanceof SimBot bot) {
			JsonObject status = bot.skills().status();
			String skill = status.get("skill").isJsonNull() ? "" : status.get("skill").getAsString();
			return (!status.get("state").getAsString().equals("running") || List.of("wait", "speak").contains(skill))
				&& bot.input().forward() == 0 && bot.input().strafe() == 0 && !bot.input().jump()
				&& Math.hypot(bot.getDeltaMovement().x, bot.getDeltaMovement().z) < 1.0e-4
				&& (bot.onGround() || Math.abs(bot.getDeltaMovement().y) < 1.0e-4);
		}
		return false;
	}

	/** Returns a live pending section; callers clear it only after exposure logging succeeds. */
	JsonArray queued(String category, String listener) {
		if (category.equals("heard")) {
			return harness.heard(listener);
		}
		JsonObject queues = object(root(), category);
		if (!queues.has(listener)) {
			queues.add(listener, new JsonArray());
		}
		return queues.getAsJsonArray(listener);
	}

	/** Drains a section after its response and full audit have been recorded. */
	void clear(String category, String listener) {
		(category.equals("heard") ? harness.map("heard") : object(root(), category)).remove(listener);
	}

	/** Queues grounded action notices at occurrence time, retaining the exact perceptions on their source event. */
	void notice(JsonObject event, String text, boolean bell) {
		BlockPos position = ControlCommands.position(event);
		Vec3 source = Vec3.atCenterOf(position);
		double radius = bell ? 48.0 : config().number("view_radius");
		JsonObject perceptions = new JsonObject();
		event.add("perceptions", perceptions);
		Map<String, Entity> current = participants();
		Entity actor = current.get(event.get("actor").getAsString());
		for (var participant : current.entrySet()) {
			if (!(participant.getValue() instanceof SimBot observer)) {
				continue;
			}
			Vec3 eye = observer.getEyePosition();
			double distance = eye.distanceTo(source);
			if (distance > radius) {
				continue;
			}
			Sight.Ray ray = bell ? null : Sight.block(observer, position);
			if (!bell && ray == null) {
				continue;
			}
			if (!bell && (actor == null || actor == observer || Sight.entity(observer, actor) == null)) {
				continue;  // a sight notice needs another person who sees both the block and the actor
			}
			JsonObject item = new JsonObject();
			for (String key : List.of("prov", "gameTime", "pos")) {
				item.add(key, event.get(key).deepCopy());
			}
			item.add("kind", event.get("type"));
			item.addProperty("sense", bell ? "hearing" : "sight");
			item.addProperty("speaker", bell ? "bell" : event.get("actor").getAsString());
			item.addProperty("text", text);
			item.addProperty("radius", radius);
			item.addProperty("distance", distance);
			item.addProperty("inStart", true);
			item.addProperty("inEnd", true);
			item.add("listenerEye", HarnessState.vector(eye));
			if (ray == null) {
				item.add("losRay", JsonNull.INSTANCE);
			} else {
				ray.audit(item);
			}
			perceptions.add(participant.getKey(), item.deepCopy());
			queued("heard", participant.getKey()).add(item);
		}
	}

	private JsonObject start(String key, String text, String volume, String to, JsonArray news) {
		return start(key, text, volume, to, news, duration(text, volume));
	}

	private JsonObject start(String key, String text, String volume, String to, JsonArray news, long ticks) {
		Entity speaker = participants().get(key);
		if (speaker == null) {
			throw new IllegalArgumentException("Unknown speaker: " + key);
		}
		JsonObject utterance = new JsonObject();
		long id = harness.next("utteranceCounter");
		utterance.addProperty("uttId", id);
		utterance.addProperty("speakerKey", key);
		utterance.addProperty("speaker", name(key));
		utterance.addProperty("text", text);
		utterance.addProperty("volume", volume);
		utterance.addProperty("radius", volume.equals("whisper") ? WHISPER_RADIUS
			: config().number(volume.equals("shout") ? "shout_radius" : "talk_radius"));
		if (volume.equals("whisper")) {
			utterance.addProperty("chatterRadius", WHISPER_CHATTER_RADIUS);
		}
		utterance.addProperty("startedAt", time());
		utterance.addProperty("endsAt", Math.addExact(time(), ticks));
		utterance.addProperty("attentionSlots", config().integer("attention_slots"));
		utterance.add("speakerStartEye", HarnessState.vector(speaker.getEyePosition()));
		utterance.add("speakerPos", HarnessState.vector(speaker.position()));
		JsonObject eyes = new JsonObject();
		participants().forEach((listener, entity) -> eyes.add(listener, HarnessState.vector(entity.getEyePosition())));
		utterance.add("listenerStartEyes", eyes);
		utterance.add("attentionLost", new JsonObject());
		utterance.add("news", news.deepCopy());
		if (to != null) {
			utterance.addProperty("to", to);
		}
		object(root(), "active").add(Long.toString(id), utterance);
		if (volume.equals("shout")) {
			object(root(), "shouts").addProperty(key, time());
		}
		return utterance;
	}

	private void arbitrate() {
		int slots = config().integer("attention_slots");
		if (slots == 0) {
			return;
		}
		for (String listener : participants().keySet()) {
			List<JsonObject> candidates = active().stream().filter(item -> item.get("endsAt").getAsLong() > time()
				&& !item.get("speakerKey").getAsString().equals(listener)
				&& !item.getAsJsonObject("attentionLost").has(listener) && startDistance(item, listener) <= item.get("radius").getAsDouble())
				.sorted(Comparator.comparingDouble((JsonObject item) -> startDistance(item, listener))
					.thenComparing(item -> item.get("speaker").getAsString()).thenComparingLong(item -> item.get("uttId").getAsLong())).toList();
			if (candidates.size() <= slots) {
				continue;
			}
			JsonArray evidence = new JsonArray();
			for (JsonObject candidate : candidates) {
				JsonObject competitor = new JsonObject();
				competitor.add("uttId", candidate.get("uttId"));
				competitor.add("speaker", candidate.get("speaker"));
				competitor.add("startedAt", candidate.get("startedAt"));
				competitor.add("endsAt", candidate.get("endsAt"));
				competitor.addProperty("distance", startDistance(candidate, listener));
				evidence.add(competitor);
			}
			for (int index = slots; index < candidates.size(); index++) {
				candidates.get(index).getAsJsonObject("attentionLost").add(listener, evidence.deepCopy());
			}
		}
	}

	private void complete(JsonObject utterance, boolean interrupted) {
		String key = utterance.get("speakerKey").getAsString();
		Entity speaker = participants().get(key);
		Vec3 start = vector(utterance.getAsJsonArray("speakerStartEye"));
		Vec3 end = speaker == null ? start : speaker.getEyePosition();
		JsonObject event = harness.event("utterance");
		utterance.entrySet().forEach(entry -> event.add(entry.getKey(), entry.getValue().deepCopy()));
		event.addProperty("interrupted", interrupted);
		event.add("pos", HarnessState.vector(end));
		JsonObject listenerEyes = new JsonObject();
		JsonObject whisperSeen = new JsonObject();
		JsonArray listeners = new JsonArray();
		double radius = utterance.get("radius").getAsDouble();
		for (var participant : participants().entrySet()) {
			String listener = participant.getKey();
			if (listener.equals(key)) {
				continue;
			}
			Vec3 eye = participant.getValue().getEyePosition();
			listenerEyes.add(listener, HarnessState.vector(eye));
			double startDistance = startDistance(utterance, listener);
			double endDistance = end.distanceTo(eye);
			boolean inStart = startDistance <= radius;
			boolean inEnd = speaker != null && endDistance <= radius;
			if (inStart || inEnd) {
				boolean attention = utterance.getAsJsonObject("attentionLost").has(listener);
				boolean heard = inStart && inEnd && !attention && !interrupted;
				JsonObject item = evidence(utterance, listener, eye, end, radius);
				if (heard) {
					item.add("text", utterance.get("text"));
					if (utterance.has("to")) {
						item.add("to", utterance.get("to"));
					}
					relayed(utterance, item, time());
					listeners.add(listener);
				} else {
					item.addProperty("reason", interrupted ? "interrupted" : !inStart || !inEnd ? "partial_presence" : "attention");
					if (attention) {
						item.add("attention", utterance.getAsJsonObject("attentionLost").get(listener).deepCopy());
					}
				}
				deliver(listener, heard ? "heard" : "chatter", item);
			} else if (speaker != null && utterance.get("volume").getAsString().equals("whisper")) {
				double chatterRadius = utterance.get("chatterRadius").getAsDouble();
				if (endDistance <= chatterRadius) {
					Sight.Ray ray = Sight.entity(participant.getValue(), speaker);
					if (whisperChatter(endDistance, ray != null)) {
						JsonObject item = evidence(utterance, listener, eye, end, chatterRadius);
						item.addProperty("reason", "out_of_earshot");
						ray.audit(item);
						JsonObject sight = new JsonObject();
						ray.audit(sight);
						whisperSeen.add(listener, sight);
						deliver(listener, "chatter", item);
					}
				}
			}
			if (utterance.get("volume").getAsString().equals("shout") && (startDistance <= 2 * radius || endDistance <= 2 * radius)) {
				deliver(listener, "shout_seen", evidence(utterance, listener, eye, end, 2 * radius));
			}
		}
		event.add("listeners", listeners);
		event.add("listenerEyes", listenerEyes);
		if (!whisperSeen.isEmpty()) {
			event.add("whisperSeen", whisperSeen);
		}
		harness.logEvent(event);
		object(root(), "active").remove(utterance.get("uttId").getAsString());
		if (!interrupted && utterance.has("exchange")) {
			JsonObject exchange = utterance.getAsJsonObject("exchange");
			String npc = exchange.get("npc").getAsString();
			String requester = exchange.get("requester").getAsString();
			if (!exchange.has("answering") && participants().containsKey(npc)
					&& listeners.contains(new com.google.gson.JsonPrimitive(npc))) {
				JsonObject reply = start(npc, exchange.get("text").getAsString(), "talk", name(requester),
					exchange.getAsJsonArray("news"), exchange.get("answerTicks").getAsLong());
				exchange.addProperty("answering", true);
				reply.add("exchange", exchange);
				arbitrate();
			} else {
				finishBot(requester, exchange.has("answering") && listeners.contains(new com.google.gson.JsonPrimitive(requester))
					? null : "not_heard", listeners);
			}
		} else if (!interrupted && !key.startsWith("npc:")) {
			finishBot(key, null, listeners);
		}
	}

	private void finishBot(String key, String reason, JsonArray listeners) {
		var player = server().getPlayerList().getPlayerByName(key);
		if (player instanceof SimBot bot) {
			bot.skills().speechFinished(reason, listeners);
		}
	}

	private JsonObject evidence(JsonObject utterance, String listener, Vec3 eye, Vec3 end, double radius) {
		JsonObject item = new JsonObject();
		for (String key : List.of("uttId", "speaker", "volume", "startedAt", "attentionSlots", "speakerStartEye", "speakerPos")) {
			item.add(key, utterance.get(key).deepCopy());
		}
		item.addProperty("prov", "utt:" + utterance.get("uttId").getAsLong());
		item.addProperty("gameTime", time());
		item.addProperty("radius", radius);
		item.add("speakerEye", HarnessState.vector(end));
		item.add("listenerEye", HarnessState.vector(eye));
		JsonElement startEye = utterance.getAsJsonObject("listenerStartEyes").get(listener);
		item.add("listenerStartEye", startEye == null ? JsonNull.INSTANCE : startEye.deepCopy());
		double distance = startDistance(utterance, listener);
		item.add("startDistance", Double.isFinite(distance) ? new com.google.gson.JsonPrimitive(distance) : JsonNull.INSTANCE);
		item.addProperty("distance", eye.distanceTo(end));
		item.addProperty("inStart", distance <= radius);
		item.addProperty("inEnd", eye.distanceTo(end) <= radius);
		return item;
	}

	/** Delivers detached relay metadata without changing its presence list. */
	static void relayed(JsonObject utterance, JsonObject item, long learnedAt) {
		JsonArray news = new JsonArray();
		for (JsonElement value : utterance.getAsJsonArray("news")) {
			news.add(VillagerNews.relay(value.getAsJsonObject(), utterance.get("speaker").getAsString(), learnedAt));
		}
		if (!news.isEmpty()) {
			item.add("news", news);
			JsonObject report = news.get(0).getAsJsonObject();
			for (String key : List.of("newsId", "kind", "subject", "firsthand", "chain", "originEventId")) {
				item.add(key, report.get(key).deepCopy());
			}
			if (report.has("present")) {
				item.add("present", report.get("present").deepCopy());
			}
		}
	}

	private void deliver(String listener, String category, JsonObject item) {
		if (listener.startsWith("npc:")) {
			if (category.equals("heard")) {
				harness.news().heard(listener, item);
			}
			JsonArray exposures = new JsonArray();
			JsonObject audit = item.deepCopy();
			audit.addProperty("category", category);
			exposures.add(audit);
			harness.exposure(listener, vector(item.getAsJsonArray("listenerEye")), "communication", exposures);
		} else {
			queued(category, listener).add(item);
		}
	}

	private long duration(String text, String volume) {
		return duration(text, volume, config().number("speech_rate"));
	}

	/** Computes speech ticks using the unchanged word count, rate and shout rounding. */
	static long duration(String text, String volume, double rate) {
		if (rate == 0) {
			return 0;
		}
		double ticks = Math.ceil(text.strip().split("\\s+").length / rate);
		ticks = Math.ceil(ticks * (volume.equals("shout") ? 1.5 : 1));
		if (ticks > Integer.MAX_VALUE) {
			throw new IllegalArgumentException("Speech duration is too large");
		}
		return (long) ticks;
	}

	/** Admits wordless whisper chatter only beyond earshot, within eight blocks, and with sight of the speaker. */
	static boolean whisperChatter(double distance, boolean visible) {
		return distance > WHISPER_RADIUS && distance <= WHISPER_CHATTER_RADIUS && visible;
	}

	/** Keeps a shout rate-limited until the configured number of ticks has elapsed. */
	static boolean shoutCoolingDown(long now, long lastShout, int cooldownTicks) {
		return now - lastShout < cooldownTicks;
	}

	/** Resolves ask identifiers only in the NPC participant namespace. */
	static String npcKey(String npc) {
		return npc.startsWith("npc:") ? npc : "npc:" + npc;
	}

	/** Retains the inclusive eye-to-eye talk-radius boundary for ask exchanges. */
	static boolean outOfTalkRange(Vec3 targetEye, Vec3 speakerEye, double talkRadius) {
		return targetEye.distanceTo(speakerEye) > talkRadius;
	}

	/** Ignores the optional argument entirely unless targeted ask exchanges are enabled. */
	static String about(JsonObject args, boolean enabled) {
		if (!enabled || !args.has("npc") || !args.has("about")) {
			return null;
		}
		String about = string(args, "about");
		if (about.isBlank()) {
			throw new IllegalArgumentException("Expected nonempty about name");
		}
		return about;
	}

	/** Carries the exact query and ordered source IDs into both phases and their immutable utterance event logs. */
	static void recordAbout(JsonObject target, String about, JsonArray news) {
		target.addProperty("about", about);
		JsonArray ids = new JsonArray();
		for (JsonElement value : news) {
			ids.add(value.getAsJsonObject().get("newsId").deepCopy());
		}
		target.add("newsIds", ids);
	}

	/** Keeps legacy answers byte-identical when opted out, while rendering every opted-in kind as ordinary speech. */
	static String answer(JsonArray news, boolean truthfulNews) {
		if (!truthfulNews || news.isEmpty()) {
			return answer(news);
		}
		List<String> reports = new ArrayList<>();
		for (JsonElement value : news) {
			JsonObject item = value.getAsJsonObject();
			String report = presence(description(item), item);
			if (townNews(item.get("kind").getAsString()) && reports.size() + 1 < news.size() && report.endsWith(".")) {
				report = report.substring(0, report.length() - 1);
			}
			reports.add(report);
		}
		return String.join("; ", reports);
	}

	/** Adds occurrence-time language and honest second-hand attribution without changing the reports' provenance. */
	static String answerAbout(JsonArray news, String about, long now, long dayTime) {
		if (news.isEmpty()) {
			return "I haven't seen or heard anything about " + about + ".";
		}
		List<String> reports = new ArrayList<>();
		for (JsonElement value : news) {
			JsonObject item = value.getAsJsonObject();
			String report = description(item);
			if (report.endsWith(".")) {
				report = report.substring(0, report.length() - 1);
			}
			report += " " + when(VillagerNews.originatedAt(item), now, dayTime) + ".";
			if (item.has("firsthand") && !item.get("firsthand").getAsBoolean()) {
				if (report.startsWith("The ") || report.startsWith("Someone ")
						|| report.startsWith("Something ") || report.startsWith("There ")) {
					report = Character.toLowerCase(report.charAt(0)) + report.substring(1);
				}
				report = "I heard that " + report;
			}
			reports.add(presence(report, item));
		}
		return String.join("; ", reports);
	}

	/** Converts simulation ticks to clock periods (tick zero is 06:00), including saved worlds with a clock offset. */
	static String when(long originatedAt, long now, long dayTime) {
		long occurred = dayTime - (now - originatedAt);
		long morning = Math.floorDiv(dayTime, 24_000L) * 24_000L;
		if (occurred < morning - 8_000L) {
			return "yesterday";
		}
		long phase = Math.floorMod(occurred, 24_000L);
		return phase < 6_000L ? "this morning" : phase < 12_000L ? "this afternoon"
			: phase < 16_000L ? "this evening" : "last night";
	}

	/** Supplies natural sentences for change news and old memories without exposing engine kind identifiers. */
	private static String description(JsonObject item) {
		String kind = item.get("kind").getAsString();
		JsonElement subject = item.get("subject");
		String name = subject.isJsonPrimitive() ? subject.getAsString() : subject.toString();
		String place = subject.isJsonPrimitive() && !name.regionMatches(true, 0, "the ", 0, 4) ? "the " + name : name;
		if (kind.equals("change")) {
			String action = item.has("action") ? item.get("action").getAsString() : "changed";
			String block = item.has("block") ? item.get("block").getAsString().replace('_', ' ') : "";
			return switch (action) {
				case "placed" -> block.isBlank() || block.equals("unknown") ? "There was work on " + place + "."
					: "Someone put up " + block + "s at " + place + ".";
				case "removed" -> "Something was taken down at " + place + ".";
				case "lit", "unlit" -> "The oven at " + place + " was " + action + ".";
				default -> "There was work on " + place + ".";
			};
		}
		if (townNews(kind)) {
			JsonObject report = item.deepCopy();
			report.remove("present");
			JsonArray single = new JsonArray();
			single.add(report);
			return answer(single);
		}
		return switch (kind) {
			case "bridge_or_port_change" -> "There was work on " + place + ".";
			case "road_blocked" -> "The road was blocked at " + name + ".";
			case "construction" -> "There was work at " + name + ".";
			case "shout" -> item.has("text") ? "Someone shouted: " + item.get("text").getAsString() : "Someone shouted at " + name + ".";
			default -> "There was news about " + name + ".";
		};
	}

	private static boolean townNews(String kind) {
		return List.of("load", "night", "sold_out", "decoration").contains(kind);
	}

	/** Appends only the names recorded in the original admitted scene, preserving the legacy singular/plural clause. */
	private static String presence(String report, JsonObject item) {
		JsonArray present = item.getAsJsonArray("present");
		if (present == null || present.isEmpty()) {
			return report;
		}
		List<String> names = new ArrayList<>();
		present.forEach(name -> names.add(name.getAsString()));
		String people = names.size() == 1 ? names.get(0)
			: String.join(", ", names.subList(0, names.size() - 1)) + " and " + names.get(names.size() - 1);
		return report + (report.endsWith(".") ? " " : ". ") + people
			+ (names.size() == 1 ? " was there." : " were there.");
	}

	/** Describes each item, ending witnessed presence claims with the appropriate singular or plural wording. */
	static String answer(JsonArray news) {
		if (news.isEmpty()) {
			return "I have no news.";
		}
		List<String> reports = new ArrayList<>();
		for (JsonElement value : news) {
			JsonObject item = value.getAsJsonObject();
			String report = item.get("kind").getAsString() + " at " + item.get("subject")
				+ (item.has("text") ? ": " + item.get("text").getAsString() : "");
			String kind = item.get("kind").getAsString();
			if (kind.equals("change")) {
				report = description(item);
			}
			boolean townNews = townNews(kind);
			if (townNews) {
				String subject = item.get("subject").getAsString();
				String place = item.has("place") ? " at " + item.get("place").getAsString() : "";
				String goods = item.has("item") ? item.get("item").getAsString().replaceFirst("^[^:]+:", "")
					.replace('_', ' ') : "goods";
				report = item.has("text") ? item.get("text").getAsString() : switch (kind) {
					case "load" -> subject + " was carrying a sack of " + goods + place + ".";
					case "night" -> subject + " was outside at night" + place + ".";
					case "sold_out" -> "The stall at " + subject + " had sold out of " + goods + ".";
					default -> subject + " was being decorated with lights.";
				};
			}
			report = presence(report, item);
			if (townNews && reports.size() + 1 < news.size() && report.endsWith(".")) {
				report = report.substring(0, report.length() - 1);
			}
			reports.add(report);
		}
		return String.join("; ", reports);
	}

	private List<JsonObject> active() {
		if (!root().has("active")) {
			return List.of();
		}
		return root().getAsJsonObject("active").asMap().values().stream().map(JsonElement::getAsJsonObject)
			.sorted(Comparator.comparingLong(item -> item.get("uttId").getAsLong())).toList();
	}

	private double startDistance(JsonObject utterance, String listener) {
		JsonElement eye = utterance.getAsJsonObject("listenerStartEyes").get(listener);
		return eye == null ? Double.POSITIVE_INFINITY : vector(eye.getAsJsonArray()).distanceTo(vector(utterance.getAsJsonArray("speakerStartEye")));
	}

	private JsonObject root() {
		return harness.map("communication");
	}

	private SimConfig config() {
		return harness.config();
	}

	private MinecraftServer server() {
		return harness.server();
	}

	private long time() {
		return server().overworld().getGameTime();
	}

	private static JsonObject object(JsonObject parent, String key) {
		if (!parent.has(key)) {
			parent.add(key, new JsonObject());
		}
		return parent.getAsJsonObject(key);
	}

	private static Vec3 vector(JsonArray value) {
		return new Vec3(value.get(0).getAsDouble(), value.get(1).getAsDouble(), value.get(2).getAsDouble());
	}

	private static String string(JsonObject args, String key) {
		JsonElement value = args.get(key);
		if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
			throw new IllegalArgumentException("Expected string: " + key);
		}
		return value.getAsString();
	}
}
