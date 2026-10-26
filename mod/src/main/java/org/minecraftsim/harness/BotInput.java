package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import net.minecraft.nbt.CompoundTag;

/**
 * Immutable per-tick bot controls; JSON and NBT field names are protocol-stable.
 * @param forward forward movement in [-1, 1]
 * @param strafe sideways movement in [-1, 1]
 * @param jump whether to jump
 * @param sneak whether to sneak
 * @param sprint whether to sprint
 * @param yaw finite horizontal view angle in degrees
 * @param pitch vertical view angle in [-90, 90] degrees
 */
public record BotInput(float forward, float strafe, boolean jump, boolean sneak, boolean sprint, float yaw, float pitch) {
	/**
	 * Creates stationary input while preserving the requested view angles.
	 * @param yaw horizontal view angle in degrees
	 * @param pitch vertical view angle in degrees
	 * @return idle controls with the supplied orientation
	 * @throws IllegalArgumentException if an angle is nonfinite or pitch is out of range
	 */
	public static BotInput idle(float yaw, float pitch) {
		return new BotInput(0, 0, false, false, false, yaw, pitch);
	}

	/**
	 * Holds jump underwater, removing only a previously automatic jump after surfacing.
	 * @param eyesInWater whether the bot's eyes are submerged in water
	 * @param automaticJump whether this input's jump was added by the afloat rule
	 * @return effective controls, preserving every other field and explicit jump input
	 */
	public BotInput afloat(boolean eyesInWater, boolean automaticJump) {
		boolean effectiveJump = eyesInWater || jump && !automaticJump;
		return effectiveJump == jump ? this : new BotInput(forward, strafe, effectiveJump, sneak, sprint, yaw, pitch);
	}

	/**
	 * Validates controls without normalizing or changing their values.
	 * @param forward forward movement in [-1, 1]
	 * @param strafe sideways movement in [-1, 1]
	 * @param jump jump flag
	 * @param sneak sneak flag
	 * @param sprint sprint flag
	 * @param yaw finite horizontal view angle
	 * @param pitch finite vertical view angle in [-90, 90]
	 * @throws IllegalArgumentException if movement or angles violate these bounds
	 */
	public BotInput {
		if (!Float.isFinite(forward) || !Float.isFinite(strafe) || !Float.isFinite(yaw) || !Float.isFinite(pitch)
				|| Math.abs(forward) > 1 || Math.abs(strafe) > 1 || Math.abs(pitch) > 90) {
			throw new IllegalArgumentException("Invalid movement input");
		}
	}

	/**
	 * Reads all seven required input fields using the existing Gson conversions.
	 * @param args protocol input object
	 * @return validated controls
	 * @throws IllegalArgumentException if converted controls are out of range
	 * @throws RuntimeException if fields are absent or cannot be converted
	 */
	public static BotInput fromJson(JsonObject args) {
		return new BotInput(args.get("forward").getAsFloat(), args.get("strafe").getAsFloat(),
				args.get("jump").getAsBoolean(), args.get("sneak").getAsBoolean(), args.get("sprint").getAsBoolean(),
				args.get("yaw").getAsFloat(), args.get("pitch").getAsFloat());
	}

	/**
	 * Serializes controls using the fixed bots.dat input keys and NBT types.
	 * @return newly allocated input compound
	 */
	public CompoundTag save() {
		CompoundTag tag = new CompoundTag();
		tag.putFloat("forward", forward);
		tag.putFloat("strafe", strafe);
		tag.putBoolean("jump", jump);
		tag.putBoolean("sneak", sneak);
		tag.putBoolean("sprint", sprint);
		tag.putFloat("yaw", yaw);
		tag.putFloat("pitch", pitch);
		return tag;
	}

	/**
	 * Restores controls, defaulting absent or mismatched NBT values to idle values.
	 * @param tag saved input compound
	 * @return validated restored controls
	 * @throws IllegalArgumentException if stored numeric values violate input bounds
	 */
	public static BotInput load(CompoundTag tag) {
		return new BotInput(tag.getFloatOr("forward", 0), tag.getFloatOr("strafe", 0), tag.getBooleanOr("jump", false),
				tag.getBooleanOr("sneak", false), tag.getBooleanOr("sprint", false), tag.getFloatOr("yaw", 0), tag.getFloatOr("pitch", 0));
	}
}
