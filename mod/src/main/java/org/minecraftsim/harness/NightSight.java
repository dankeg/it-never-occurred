package org.minecraftsim.harness;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.core.BlockPos;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.Vec3;

/** Shared deterministic night admission; emission comes from loaded block states, never the light engine. */
final class NightSight {
	/** Supplies loaded emissive blocks by section without loading or advancing the world. */
	interface View {
		/** Returns elapsed simulation ticks for cache invalidation. */
		long tick();
		/** Returns the independent overworld day/night clock. */
		long dayTime();
		/** Lists emitting positions from this loaded section, or an empty list if unloaded. */
		List<BlockPos> lights(int chunkX, int sectionY, int chunkZ);
	}

	private record Section(int chunkX, int sectionY, int chunkZ) {
	}

	private final View view;
	private final Map<Section, List<BlockPos>> sections = new HashMap<>();
	private long cachedTick = Long.MIN_VALUE;

	/** Binds a server lazily, scanning each loaded, potentially emissive section at most once per tick. */
	NightSight(MinecraftServer server) {
		this(new View() {
			/** Reads the simulation tick for transient cache expiry. */
			public long tick() { return server.overworld().getGameTime(); }
			/** Reads the overworld's day/night clock, not elapsed simulation time. */
			public long dayTime() { return server.overworld().getOverworldClockTime(); }
			/** Collects only emitting states in an already loaded section. */
			public List<BlockPos> lights(int chunkX, int sectionY, int chunkZ) {
				var level = server.overworld();
				var chunk = level.getChunkSource().getChunkNow(chunkX, chunkZ);
				int sectionIndex = level.getSectionIndex(sectionY * 16);
				if (chunk == null || sectionIndex < 0 || sectionIndex >= chunk.getSections().length) {
					return List.of();
				}
				var section = chunk.getSections()[sectionIndex];
				if (!section.maybeHas(state -> state.getLightEmission() > 0)) {
					return List.of();
				}
				List<BlockPos> result = new ArrayList<>();
				for (int localX = 0; localX < 16; localX++) {
					for (int localY = 0; localY < 16; localY++) {
						for (int localZ = 0; localZ < 16; localZ++) {
							if (section.getBlockState(localX, localY, localZ).getLightEmission() > 0) {
								result.add(new BlockPos(chunkX * 16 + localX, sectionY * 16 + localY, chunkZ * 16 + localZ));
							}
						}
					}
				}
				return result;
			}
		});
	}

	/** Uses an injectable loaded-state view for rule and cache tests without a server. */
	NightSight(View view) {
		this.view = view;
	}

	/** Night includes 19:00 and ends at 05:00, independently of the elapsed game-time clock. */
	static boolean isNight(long dayTime) {
		long phase = Math.floorMod(dayTime, 24_000L);
		return phase >= 13_000 && phase < 23_000;
	}

	/** Only a person's main-hand light-emitting block item lights that person. */
	static boolean heldLight(ItemStack stack) {
		return !stack.isEmpty() && stack.getItem() instanceof BlockItem item
			&& item.getBlock().defaultBlockState().getLightEmission() > 0;
	}

	/** Admits an entity's center by the same rule as blocks; callers still require line of sight. */
	boolean visible(Entity observer, Entity target) {
		boolean held = (target instanceof Player || target instanceof Villager)
			&& heldLight(((LivingEntity) target).getMainHandItem());
		return visible(observer.getEyePosition(), target.getBoundingBox().getCenter(), held);
	}

	/** Admits a target by distance or deterministic local emission, leaving occlusion to Sight. */
	boolean visible(Vec3 eye, Vec3 target, boolean held) {
		if (!isNight(view.dayTime()) || eye.distanceToSqr(target) <= 64 || held) {
			return true;
		}
		if (cachedTick != view.tick()) {
			invalidate();
			cachedTick = view.tick();
		}
		for (int chunkX = Math.floorDiv((int) Math.floor(target.x - 5), 16);
				chunkX <= Math.floorDiv((int) Math.floor(target.x + 5), 16); chunkX++) {
			for (int sectionY = Math.floorDiv((int) Math.floor(target.y - 5), 16);
					sectionY <= Math.floorDiv((int) Math.floor(target.y + 5), 16); sectionY++) {
				for (int chunkZ = Math.floorDiv((int) Math.floor(target.z - 5), 16);
						chunkZ <= Math.floorDiv((int) Math.floor(target.z + 5), 16); chunkZ++) {
					Section section = new Section(chunkX, sectionY, chunkZ);
					List<BlockPos> lights = sections.computeIfAbsent(section,
						key -> view.lights(key.chunkX(), key.sectionY(), key.chunkZ()));
					for (BlockPos light : lights) {
						if (target.distanceToSqr(Vec3.atCenterOf(light)) <= 25) {
							return true;
						}
					}
				}
			}
		}
		return false;
	}

	/** Invalidates transient emission after a block edit, including edits in the same lockstep tick. */
	void invalidate() {
		sections.clear();
		cachedTick = Long.MIN_VALUE;
	}
}
