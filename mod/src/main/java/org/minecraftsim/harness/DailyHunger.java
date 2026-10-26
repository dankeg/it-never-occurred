package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.util.Comparator;

/** Applies opt-in, game-time food costs without storing a wall-clock or a second schedule cursor. */
final class DailyHunger {
	private final HarnessState state;

	/** Shares canonical configuration and event logging without creating optional saved state. */
	DailyHunger(HarnessState state) {
		this.state = state;
	}

	/** Applies each due cost to living survival bots in stable name order, at the end of a game tick. */
	void tick() {
		SimConfig config = state.config();
		long time = state.server().overworld().getGameTime();
		if (!config.dailyHunger() || time <= 0 || time % config.integer("hunger_interval_ticks") != 0) {
			return;
		}
		state.server().getPlayerList().getPlayers().stream().filter(player -> player instanceof SimBot)
			.map(player -> (SimBot) player).filter(bot -> bot.isAlive() && !bot.hasInfiniteMaterials()
				&& bot.level() == state.server().overworld())
			.sorted(Comparator.comparing(bot -> bot.getGameProfile().name())).forEach(bot -> {
				int before = bot.getFoodData().getFoodLevel();
				int after = foodAfter(time, before, config);
				bot.getFoodData().setFoodLevel(after);
				bot.getFoodData().setSaturation(Math.min(bot.getFoodData().getSaturationLevel(), after));
				changed(bot, before, "daily_hunger");
			});
	}

	/** Returns the food level after this exact game-time boundary; time zero is never a meal cost. */
	static int foodAfter(long time, int food, SimConfig config) {
		return config.dailyHunger() && time > 0 && time % config.integer("hunger_interval_ticks") == 0
			? Math.max(0, food - config.integer("hunger_points")) : food;
	}

	/** Logs a real food-level change, including restoration by eating, without emitting zero-delta events. */
	void changed(SimBot bot, int before, String cause) {
		int after = bot.getFoodData().getFoodLevel();
		if (before != after) {
			JsonObject event = state.event("hunger");
			event.addProperty("bot", bot.getGameProfile().name());
			event.addProperty("from", before);
			event.addProperty("to", after);
			event.addProperty("cause", cause);
			state.logEvent(event);
		}
	}
}
