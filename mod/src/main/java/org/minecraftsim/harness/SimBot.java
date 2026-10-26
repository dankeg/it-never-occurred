package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import com.mojang.authlib.GameProfile;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.phys.Vec3;
import org.minecraftsim.harness.mixin.GameListenerAccessor;
import org.minecraftsim.harness.mixin.LivingEntityAccessor;

// Adapted from Carpet's EntityPlayerMPFake (MIT), copyright gnembon.
/**
 * Server-authoritative fake player with persistent input, skills and extra physics state.
 * Skills run before input application; connection resynchronization uses game time.
 */
public final class SimBot extends ServerPlayer {
	private BotInput input = BotInput.idle(0, 0);
	private boolean afloatJump;
	private final Skills skills = new Skills(this);

	/**
	 * Exposes this bot's single skill controller.
	 * @return controller ticked before applying movement input
	 */
	public Skills skills() {
		return skills;
	}

	/**
	 * Creates a fake player with an identity-derived nonzero enchantment seed.
	 * @param server owning server
	 * @param level initial server level
	 * @param profile stable offline player identity
	 */
	public SimBot(MinecraftServer server, ServerLevel level, GameProfile profile) {
		super(server, level, profile, ClientInformation.createDefault());
		this.enchantmentSeed = profile.id().hashCode();
		if (this.enchantmentSeed == 0) {
			this.enchantmentSeed = 1;
		}
		this.setSpeed((float) this.getAttributeValue(Attributes.MOVEMENT_SPEED));
	}

	/**
	 * Reads the immutable controls used until replaced.
	 * @return current input
	 */
	public BotInput input() {
		return input;
	}

	/**
	 * Replaces controls on the server thread without applying movement immediately.
	 * @param input controls for subsequent input application
	 */
	public void setInput(BotInput input) {
		this.input = input;
		this.afloatJump = false;
	}

	/** Reports whether the saved jump input was supplied by the water safety rule. */
	boolean afloatJump() {
		return afloatJump;
	}

	/** Advances skills in this bot's actor scope, then applies input through vanilla movement. */
	@Override
	protected void applyInput() {
		HarnessState.acting(getGameProfile().name(), () -> {
			skills.tick();
			return null;
		});
		boolean requestedJump = input.jump() && !afloatJump;
		input = input.afloat(isEyeInFluid(FluidTags.WATER), afloatJump);
		afloatJump = input.jump() && !requestedJump;
		this.zza = input.forward();
		this.xxa = input.strafe();
		this.setJumping(input.jump());
		this.setShiftKeyDown(input.sneak());
		this.setSprinting(input.sprint());
		this.setYRot(input.yaw());
		this.setXRot(input.pitch());
		this.setYHeadRot(input.yaw());
		this.setSpeed((float) this.getAttributeValue(Attributes.MOVEMENT_SPEED));
		super.applyInput();
	}

	/**
	 * Keeps fake-player movement under server authority.
	 * @return false for every simulation bot
	 */
	@Override
	public boolean isClientAuthoritative() {
		return false;
	}

	/**
	 * Leaves ferry movement to the tick-end controller rather than vanilla travel physics.
	 * @param movement the vanilla movement input for ordinary on-foot skills
	 */
	@Override
	public void travel(Vec3 movement) {
		if (skills.ferryRunning()) {
			setDeltaMovement(Vec3.ZERO);
		} else {
			super.travel(movement);
		}
	}

	/**
	 * Prevents collision impulses from moving a reserved ferry rider.
	 * @return vanilla pushability when no ferry skill is running
	 */
	@Override
	public boolean isPushable() {
		return !skills.ferryRunning() && super.isPushable();
	}

	/** Runs vanilla player ticks and resynchronizes connection/chunk position every ten game ticks. */
	@Override
	public void tick() {
		super.tick();
		this.doTick();
		if (this.level().getGameTime() % 10 == 0) {
			this.connection.resetPosition();
			this.level().getChunkSource().move(this);
		}
	}

	/**
	 * Records vanilla death without respawning or initializing an absent harness.
	 * @param source vanilla damage source responsible for this death
	 */
	@Override
	public void die(DamageSource source) {
		HarnessState state = HarnessState.existing(level().getServer());
		if (state != null) {
			JsonObject event = state.event("death");
			event.addProperty("bot", getGameProfile().name());
			event.addProperty("cause", source.getMsgId());
			event.add("pos", HarnessState.vector(position()));
			state.logEvent(event);
		}
		super.die(source);
	}

	/**
	 * Captures extra physics and connection fields absent from vanilla player saves.
	 * @return newly allocated compound using the fixed snapshot keys
	 */
	public CompoundTag savePhysics() {
		CompoundTag tag = new CompoundTag();
		if (afloatJump) {
			tag.putBoolean("afloatJump", true);
		}
		tag.putInt("tickCount", this.tickCount);
		tag.putBoolean("firstTick", this.firstTick);
		tag.putInt("noJumpDelay", ((LivingEntityAccessor) (Object) this).simharness$getNoJumpDelay());
		tag.putInt("jumpTriggerTime", this.jumpTriggerTime);
		tag.putFloat("speed", this.getSpeed());
		tag.putBoolean("sprinting", this.isSprinting());
		tag.putBoolean("sneaking", this.isShiftKeyDown());
		GameListenerAccessor listener = (GameListenerAccessor) this.connection;
		tag.putInt("clientLoadedTimeout", listener.simharness$getClientLoadedTimeoutTimer());
		tag.putBoolean("waitingForRespawn", listener.simharness$isWaitingForRespawn());
		return tag;
	}

	/**
	 * Restores extra physics and connection fields with the existing per-field defaults.
	 * @param tag saved physics compound
	 */
	public void loadPhysics(CompoundTag tag) {
		this.afloatJump = tag.getBooleanOr("afloatJump", false);
		this.tickCount = tag.getIntOr("tickCount", 0);
		this.firstTick = tag.getBooleanOr("firstTick", false);
		((LivingEntityAccessor) (Object) this).simharness$setNoJumpDelay(tag.getIntOr("noJumpDelay", 0));
		this.jumpTriggerTime = tag.getIntOr("jumpTriggerTime", 0);
		this.setSprinting(tag.getBooleanOr("sprinting", false));
		this.setShiftKeyDown(tag.getBooleanOr("sneaking", false));
		this.setSpeed(tag.getFloatOr("speed", (float) this.getAttributeValue(Attributes.MOVEMENT_SPEED)));
		GameListenerAccessor listener = (GameListenerAccessor) this.connection;
		listener.simharness$setClientLoadedTimeoutTimer(tag.getIntOr("clientLoadedTimeout", 0));
		listener.simharness$setWaitingForRespawn(tag.getBooleanOr("waitingForRespawn", false));
	}
}
