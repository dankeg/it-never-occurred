package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import java.io.BufferedOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import net.minecraft.core.BlockPos;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.DoubleTag;
import net.minecraft.nbt.FloatTag;
import net.minecraft.nbt.ListTag;
import net.minecraft.nbt.StringTag;
import net.minecraft.nbt.Tag;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.util.ProblemReporter;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.storage.TagValueOutput;
import net.minecraft.world.phys.Vec3;

/**
 * Computes canonical overworld and harness-state digests on the server thread.
 * Hash composition, x/y/z traversal, UUID ordering, raw floating-point bits and
 * NBT encodings are fixed; hashing draws no RNG and excludes no NBT keys by default.
 */
public final class StateHash {
	/** Canonical NBT policy: this intentionally empty exclusion set is internal only. */
	private static final Set<String> VOLATILE_KEYS = new HashSet<>();
	/**
	 * NBT list keys whose element order comes from hash-map iteration (identity hash codes, which vary per JVM),
	 * not from simulation state. Elements are sorted by their canonical encoding. Found by E1 diagnosis
	 * (2026-10-16): the player's "attributes" list order differed between two same-seed runs, with identical values.
	 */
	private static final Set<String> UNORDERED_LIST_KEYS = Set.of("attributes", "modifiers", "recipes", "toBeDisplayed");

	private StateHash() {
	}

	/**
	 * Hashes blocks, block entities, selected entities and canonical harness state.
	 * @param server server to inspect on its main thread
	 * @param region inclusive overworld block bounds; entities are selected by bounding-box center
	 * @return game time, total digest and the four component digests as protocol JSON
	 * @throws IllegalStateException if full entity NBT cannot be saved or SHA-256 is unavailable
	 * @throws UncheckedIOException if canonical output fails
	 */
	public static JsonObject compute(MinecraftServer server, Region region) {
		ServerLevel level = server.overworld();
		long gameTime = level.getGameTime();
		byte[] blocks = digest(output -> forEachBlock(region, position -> {
			// Global block-state registry id: stable within one game version and far cheaper than toString().
			writeBlockState(level.getBlockState(position), output);
		}));
		byte[] blockEntities = digest(output -> forEachBlock(region, position -> {
			BlockEntity blockEntity = level.getBlockEntity(position);
			if (blockEntity != null) {
				output.writeInt(position.getX());
				output.writeInt(position.getY());
				output.writeInt(position.getZ());
				writeCanonicalNbt(blockEntity.saveWithFullMetadata(level.registryAccess()), output);
			}
		}));
		List<Entity> selected = new ArrayList<>();
		for (Entity entity : level.getAllEntities()) {
			Vec3 center = entity.getBoundingBox().getCenter();
			if (region.contains(center.x, center.y, center.z)) {
				selected.add(entity);
			}
		}
		selected.sort(Comparator.comparing(Entity::getUUID));
		byte[] entities = digest(output -> {
			for (Entity entity : selected) {
				writeString(entity.typeHolder().getRegisteredName(), output);
				output.writeLong(Double.doubleToRawLongBits(entity.getX()));
				output.writeLong(Double.doubleToRawLongBits(entity.getY()));
				output.writeLong(Double.doubleToRawLongBits(entity.getZ()));
				Vec3 movement = entity.getDeltaMovement();
				output.writeLong(Double.doubleToRawLongBits(movement.x));
				output.writeLong(Double.doubleToRawLongBits(movement.y));
				output.writeLong(Double.doubleToRawLongBits(movement.z));
				output.writeInt(Float.floatToRawIntBits(entity.getYRot()));
				output.writeInt(Float.floatToRawIntBits(entity.getXRot()));
				ProblemReporter.Collector reporter = new ProblemReporter.Collector();
				TagValueOutput saved = TagValueOutput.createWithContext(reporter, level.registryAccess());
				entity.saveWithoutId(saved);
				if (!reporter.isEmpty()) {
					throw new IllegalStateException("Cannot hash full NBT for " + entity.getUUID() + ": " + reporter.getReport());
				}
				writeCanonicalNbt(saved.buildResult(), output);
			}
		});
		byte[] harness = HexFormat.of().parseHex(HarnessState.sha256(HarnessState.get(server).serialization()));
		byte[] total = digest(output -> {
			output.writeLong(gameTime);
			output.write(blocks);
			output.write(blockEntities);
			output.write(entities);
			output.write(harness);
		});
		JsonObject result = new JsonObject();
		result.addProperty("gameTime", gameTime);
		result.addProperty("total", HexFormat.of().formatHex(total));
		result.addProperty("blocks", HexFormat.of().formatHex(blocks));
		result.addProperty("blockEntities", HexFormat.of().formatHex(blockEntities));
		result.addProperty("entities", HexFormat.of().formatHex(entities));
		result.addProperty("harness", HexFormat.of().formatHex(harness));
		return result;
	}

	/**
	 * Dumps UUID-ordered entity state with sorted-key SNBT for diagnosing hash differences.
	 * @param server server to inspect on its main thread
	 * @param region bounds selecting entities by bounding-box center
	 * @return newly allocated array of entity identity, position, motion, rotation and NBT records
	 */
	public static com.google.gson.JsonArray dumpEntities(MinecraftServer server, Region region) {
		ServerLevel level = server.overworld();
		List<Entity> selected = new ArrayList<>();
		for (Entity entity : level.getAllEntities()) {
			Vec3 center = entity.getBoundingBox().getCenter();
			if (region.contains(center.x, center.y, center.z)) {
				selected.add(entity);
			}
		}
		selected.sort(Comparator.comparing(Entity::getUUID));
		com.google.gson.JsonArray out = new com.google.gson.JsonArray();
		for (Entity entity : selected) {
			JsonObject record = new JsonObject();
			record.addProperty("uuid", entity.getUUID().toString());
			record.addProperty("type", entity.typeHolder().getRegisteredName());
			Vec3 movement = entity.getDeltaMovement();
			record.addProperty("pos", entity.getX() + "," + entity.getY() + "," + entity.getZ());
			record.addProperty("motion", movement.x + "," + movement.y + "," + movement.z);
			record.addProperty("rot", entity.getYRot() + "," + entity.getXRot());
			ProblemReporter.Collector reporter = new ProblemReporter.Collector();
			TagValueOutput saved = TagValueOutput.createWithContext(reporter, level.registryAccess());
			entity.saveWithoutId(saved);
			record.addProperty("nbt", canonicalSnbt(saved.buildResult()));
			out.add(record);
		}
		return out;
	}

	/** Diagnostic encoding: sorts compound keys and the designated unordered lists. */
	private static String canonicalSnbt(Tag tag) {
		return switch (tag) {
			case CompoundTag compound -> {
				StringBuilder builder = new StringBuilder("{");
				boolean first = true;
				for (String key : compound.keySet().stream().sorted().toList()) {
					if (!first) {
						builder.append(',');
					}
					first = false;
					Tag value = compound.get(key);
					if (UNORDERED_LIST_KEYS.contains(key) && value instanceof ListTag list) {
						List<String> parts = new ArrayList<>();
						for (Tag element : list) {
							parts.add(canonicalSnbt(element));
						}
						parts.sort(null);
						builder.append(key).append(":[").append(String.join(",", parts)).append(']');
					} else {
						builder.append(key).append(':').append(canonicalSnbt(value));
					}
				}
				yield builder.append('}').toString();
			}
			case ListTag list -> {
				StringBuilder builder = new StringBuilder("[");
				for (int index = 0; index < list.size(); index++) {
					if (index > 0) {
						builder.append(',');
					}
					builder.append(canonicalSnbt(list.get(index)));
				}
				yield builder.append(']').toString();
			}
			default -> tag.toString();
		};
	}

	/** Hash traversal: x, then y, then z, with inclusive integer bounds. */
	private static void forEachBlock(Region region, BlockWriter writer) throws IOException {
		BlockPos.MutableBlockPos position = new BlockPos.MutableBlockPos();
		for (long x = region.x0(); x <= region.x1(); x++) {
			for (long y = region.y0(); y <= region.y1(); y++) {
				for (long z = region.z0(); z <= region.z1(); z++) {
					position.set((int) x, (int) y, (int) z);
					writer.write(position);
				}
			}
		}
	}

	/** Hash encoding: retains NBT types and raw numeric bits with fixed key/list policies. */
	static void writeCanonicalNbt(Tag tag, DataOutput output) throws IOException {
		output.writeByte(tag.getId());
		switch (tag) {
			case CompoundTag compound -> {
				List<String> keys = compound.keySet().stream().filter(key -> !VOLATILE_KEYS.contains(key)).sorted().toList();
				output.writeInt(keys.size());
				for (String key : keys) {
					writeString(key, output);
					Tag value = compound.get(key);
					if (UNORDERED_LIST_KEYS.contains(key) && value instanceof ListTag list) {
						writeSortedList(list, output);
					} else {
						writeCanonicalNbt(value, output);
					}
				}
			}
			case ListTag list -> {
				output.writeInt(list.size());
				for (Tag element : list) {
					writeCanonicalNbt(element, output);
				}
			}
			case FloatTag number -> output.writeInt(Float.floatToRawIntBits(number.value()));
			case DoubleTag number -> output.writeLong(Double.doubleToRawLongBits(number.value()));
			case StringTag string -> writeString(string.value(), output);
			default -> tag.write(output);
		}
	}

	private static void writeSortedList(ListTag list, DataOutput output) throws IOException {
		List<byte[]> encoded = new ArrayList<>();
		for (Tag element : list) {
			java.io.ByteArrayOutputStream bytes = new java.io.ByteArrayOutputStream();
			writeCanonicalNbt(element, new java.io.DataOutputStream(bytes));
			encoded.add(bytes.toByteArray());
		}
		encoded.sort(java.util.Arrays::compare);
		output.writeByte(list.getId());
		output.writeInt(encoded.size());
		for (byte[] element : encoded) {
			output.write(element);
		}
	}

	private static void writeString(String value, DataOutput output) throws IOException {
		byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
		output.writeInt(bytes.length);
		output.write(bytes);
	}

	/** Writes the same global integer state ID used by the block digest traversal. */
	static void writeBlockState(BlockState state, DataOutput output) throws IOException {
		output.writeInt(net.minecraft.world.level.block.Block.getId(state));
	}

	private static byte[] digest(HashWriter writer) {
		try {
			MessageDigest digest = MessageDigest.getInstance("SHA-256");
			try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(
				new DigestOutputStream(OutputStream.nullOutputStream(), digest)))) {
				writer.write(output);
			}
			return digest.digest();
		} catch (IOException exception) {
			throw new UncheckedIOException(exception);
		} catch (NoSuchAlgorithmException exception) {
			throw new IllegalStateException("SHA-256 is unavailable", exception);
		}
	}

	@FunctionalInterface
	private interface HashWriter {
		/**
		 * Writes one digest payload in its established field order.
		 * @param output digest-backed data stream
		 * @throws IOException if writing fails
		 */
		void write(DataOutputStream output) throws IOException;
	}

	@FunctionalInterface
	private interface BlockWriter {
		/**
		 * Processes the current position before the traversal reuses its mutable object.
		 * @param position current block coordinate; must not be retained
		 * @throws IOException if writing the block payload fails
		 */
		void write(BlockPos position) throws IOException;
	}
}
