package org.minecraftsim.harness;

import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.Shapes;
import net.minecraft.world.phys.shapes.VoxelShape;
import net.minecraft.world.phys.shapes.CollisionContext;

/**
 * Performs visual-shape line-of-sight checks without loading chunks or drawing RNG.
 * Probe order and observer feet/head exclusions are shared by exposure and skill checks.
 */
public final class Sight {
	/** Block probes: center followed by six fixed axis offsets, in protocol audit order. */
	private static final double[][] OFFSETS = {{0, 0, 0}, {-0.49, 0, 0}, {0.49, 0, 0},
		{0, -0.49, 0}, {0, 0.49, 0}, {0, 0, -0.49}, {0, 0, 0.49}};

	private Sight() {
	}

	/**
	 * Finds the first visible probe of a loaded block, including visual-shape-empty targets.
	 * @param observer server-level observer
	 * @param position block to inspect
	 * @return first successful audit ray, or null if unloaded or occluded
	 * @throws ClassCastException if the observer is not in a server level
	 */
	public static Ray block(Entity observer, BlockPos position) {
		ServerLevel level = (ServerLevel) observer.level();
		if (!level.hasChunkAt(position)) {
			return null;
		}
		Vec3 eye = observer.getEyePosition();
		Vec3 center = Vec3.atCenterOf(position);
		HarnessState harness = HarnessState.get(level.getServer());
		if (harness.config().nightSight() && !harness.nightSight().visible(eye, center, false)) {
			return null;
		}
		boolean empty = context(observer, center).getBlockShape(level.getBlockState(position), level, position).isEmpty();
		for (int index = 0; index < OFFSETS.length; index++) {
			double[] offset = OFFSETS[index];
			Vec3 target = center.add(offset[0], offset[1], offset[2]);
			BlockHitResult hit = clip(observer, target);
			if (hit != null && (hit.getType() == HitResult.Type.BLOCK && hit.getBlockPos().equals(position)
					|| empty && reaches(hit, target))) {
				return new Ray(index, eye, target, hit, empty);
			}
		}
		return null;
	}

	/**
	 * Tests the target's eye and then bounding-box center using block occlusion only.
	 * @param observer server-level observer
	 * @param entity target entity
	 * @return first unobstructed loaded ray, or null if neither probe succeeds
	 * @throws ClassCastException if the observer is not in a server level
	 */
	public static Ray entity(Entity observer, Entity entity) {
		HarnessState harness = HarnessState.get(((ServerLevel) observer.level()).getServer());
		if (harness.config().nightSight() && !harness.nightSight().visible(observer, entity)) {
			return null;
		}
		Vec3[] targets = {entity.getEyePosition(), entity.getBoundingBox().getCenter()};
		for (int index = 0; index < targets.length; index++) {
			BlockHitResult hit = clip(observer, targets[index]);
			if (hit != null && reaches(hit, targets[index])) {
				return new Ray(index, observer.getEyePosition(), targets[index], hit, false);
			}
		}
		return null;
	}

	/**
	 * Checks a placement face against the supplied planning view without loading blocks.
	 * @param view world or planned-block view used for clipping
	 * @param eye observer eye position
	 * @param feet observer foot block, excluded along with its head block
	 * @param support target support block
	 * @param face support face that must be hit from its outward side
	 * @return whether a loaded visual ray hits the requested block and face
	 */
	public static boolean face(SkillPath.View view, Vec3 eye, BlockPos feet, BlockPos support, Direction face) {
		Vec3 target = Vec3.atCenterOf(support).add(face.getStepX() * 0.5, face.getStepY() * 0.5, face.getStepZ() * 0.5);
		if ((eye.x - target.x) * face.getStepX() + (eye.y - target.y) * face.getStepY()
				+ (eye.z - target.z) * face.getStepZ() <= 0) {
			return false;
		}
		target = target.add(-face.getStepX() * 0.001, -face.getStepY() * 0.001, -face.getStepZ() * 0.001);
		boolean loaded = BlockGetter.traverseBlocks(eye, target, view,
			(world, position) -> world.loaded(position) ? null : Boolean.FALSE, world -> Boolean.TRUE);
		if (!loaded) {
			return false;
		}
		BlockPos head = BlockPos.containing(eye);
		BlockHitResult hit = view.clip(new ClipContext(eye, target, ClipContext.Block.VISUAL, ClipContext.Fluid.NONE, CollisionContext.empty()) {
			/**
			 * Removes observer-occupied blocks from placement clipping.
			 * @param state state whose visual shape is requested
			 * @param world clipping world
			 * @param position queried coordinate
			 * @return empty shape for feet/head, otherwise the original visual shape
			 */
			@Override
			public VoxelShape getBlockShape(BlockState state, BlockGetter world, BlockPos position) {
				return position.equals(feet) || position.equals(head) ? Shapes.empty() : super.getBlockShape(state, world, position);
			}
		});
		return hit.getType() == HitResult.Type.BLOCK && hit.getBlockPos().equals(support) && hit.getDirection() == face;
	}

	private static boolean reaches(BlockHitResult hit, Vec3 target) {
		return hit.getType() == HitResult.Type.MISS || hit.getLocation().distanceTo(target) <= 0.05;
	}

	private static ClipContext context(Entity observer, Vec3 target) {
		BlockPos feet = observer.blockPosition();
		BlockPos head = BlockPos.containing(observer.getEyePosition());
		return new ClipContext(observer.getEyePosition(), target, ClipContext.Block.VISUAL, ClipContext.Fluid.NONE, observer) {
			/**
			 * Removes observer-occupied blocks from exposure clipping.
			 * @param state state whose visual shape is requested
			 * @param level clipping level
			 * @param position queried coordinate
			 * @return empty shape for feet/head, otherwise the original visual shape
			 */
			@Override
			public VoxelShape getBlockShape(BlockState state, BlockGetter level, BlockPos position) {
				return position.equals(feet) || position.equals(head) ? Shapes.empty() : super.getBlockShape(state, level, position);
			}
		};
	}

	private static BlockHitResult clip(Entity observer, Vec3 target) {
		Vec3 eye = observer.getEyePosition();
		ServerLevel world = (ServerLevel) observer.level();
		boolean loaded = BlockGetter.traverseBlocks(eye, target, world,
			(level, position) -> level.hasChunkAt(position) ? null : Boolean.FALSE, level -> Boolean.TRUE);
		if (!loaded) {
			return null;
		}
		return world.clip(context(observer, target));
	}

	/**
	 * Immutable evidence for one successful fixed-order sight probe.
	 * @param index probe index in the corresponding target sequence
	 * @param from observer eye position
	 * @param to requested probe endpoint
	 * @param hit clipping result
	 * @param emptyTarget whether the target block has no visual shape
	 */
	public record Ray(int index, Vec3 from, Vec3 to, BlockHitResult hit, boolean emptyTarget) {
		/**
		 * Appends the established ray-audit JSON fields without changing their order.
		 * @param item result object to annotate
		 */
		public void audit(JsonObject item) {
			item.addProperty("losRay", index);
			item.add("rayFrom", HarnessState.vector(from));
			item.add("rayTo", HarnessState.vector(to));
			item.addProperty("hitType", hit.getType().name());
			item.add("hitPos", HarnessState.position(hit.getBlockPos()));
			item.add("hitLocation", HarnessState.vector(hit.getLocation()));
			item.addProperty("emptyTarget", emptyTarget);
		}
	}
}
