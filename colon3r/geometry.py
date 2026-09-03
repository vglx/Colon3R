from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .supervision import MultitaskLoss


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape[-2:] == (4, 4):
        return value.copy()
    if value.shape[-2:] != (3, 4):
        raise ValueError(f"expected matrices ending in 3x4 or 4x4, got {value.shape}")
    result = np.zeros(value.shape[:-2] + (4, 4), dtype=np.float64)
    result[..., :3, :] = value
    result[..., 3, 3] = 1.0
    return result


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_homogeneous(c2w))


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_homogeneous(w2c))


def _validate_transform_inputs(points: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value, pose = np.asarray(points, dtype=np.float64), _homogeneous(w2c)
    if value.ndim < 4 or value.shape[-1] != 3:
        raise ValueError("points must end in XYZ")
    if value.shape[:-3] != pose.shape[:-2]:
        raise ValueError(f"point and pose frame dimensions differ: {value.shape}, {pose.shape}")
    return value, pose


def world_to_camera_points(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    points, poses = _validate_transform_inputs(points_world, w2c)
    return np.einsum("...ij,...hwj->...hwi", poses[..., :3, :3], points) + poses[..., :3, 3][..., None, None, :]


def camera_to_world_points(points_camera: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    points, poses = _validate_transform_inputs(points_camera, w2c)
    rotation, translation = poses[..., :3, :3], poses[..., :3, 3]
    return np.einsum("...ji,...hwj->...hwi", rotation, points - translation[..., None, None, :])


def transfer_camera_points(points_camera: np.ndarray, source_w2c: np.ndarray, target_w2c: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("camera points must end in XYZ")
    source, target = _homogeneous(source_w2c), _homogeneous(target_w2c)
    if source.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("cross-view transfer requires one source and target W2C matrix")
    world = np.einsum("ji,...j->...i", source[:3, :3], points - source[:3, 3])
    return np.einsum("ij,...j->...i", target[:3, :3], world) + target[:3, 3]


def normalize_first_camera(w2c: np.ndarray, world_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    poses = _homogeneous(w2c)
    if poses.ndim != 3 or poses.shape[0] != 8:
        raise ValueError("first-camera normalization requires eight W2C poses")
    normalized = poses @ np.linalg.inv(poses[0])[None]
    transformed = None
    if world_points is not None:
        world = np.asarray(world_points, dtype=np.float64)
        if world.shape[0] != 8 or world.shape[-1] != 3:
            raise ValueError("world points must have eight frames ending in XYZ")
        transformed = np.einsum("ij,shwj->shwi", poses[0, :3, :3], world) + poses[0, :3, 3][None, None, None, :]
    return normalized[:, :3], transformed


def dense_pixel_grid(width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("pixel grid dimensions must be positive")
    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64), indexing="xy")
    return np.stack((x, y), axis=-1)


def unproject_z_depth(camera: Any, depth: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(depth, dtype=np.float64)
    poses = _homogeneous(w2c)
    if values.ndim != 3 or values.shape[0] != 8 or poses.shape[0] != 8:
        raise ValueError("depth unprojection requires [8,H,W] depths and eight W2C poses")
    pixels = dense_pixel_grid(values.shape[2], values.shape[1])
    camera_points = camera.camera_point_from_z_depth(pixels, values)
    return camera_points, camera_to_world_points(camera_points, poses)


def project_world_points(camera: Any, world_points: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = world_to_camera_points(world_points, w2c)
    pixels, valid = camera.camera_point_to_pixel(camera_points)
    return pixels, camera_points[..., 2], valid


def directed_pairs(view_count: int = 8) -> tuple[tuple[int, int], ...]:
    if int(view_count) != 8:
        raise ValueError("Colon3R geometry requires exactly eight frames")
    return tuple((source, target) for source in range(8) for target in range(8) if source != target)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = torch.unbind(quaternion, -1)
    scale = 2 / quaternion.square().sum(-1)
    values = torch.stack(
        (
            1 - scale * (y.square() + z.square()),
            scale * (x * y - z * w),
            scale * (x * z + y * w),
            scale * (x * y + z * w),
            1 - scale * (x.square() + z.square()),
            scale * (y * z - x * w),
            scale * (x * z - y * w),
            scale * (y * z + x * w),
            1 - scale * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return values.reshape(quaternion.shape[:-1] + (3, 3))


def pose_encoding_to_extri_fov(
    pose_encoding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pose_encoding.shape[-1] != 9:
        raise ValueError("pose encoding must end in nine values")
    translation = pose_encoding[..., :3]
    rotation = _quaternion_to_matrix(pose_encoding[..., 3:7])
    return (
        torch.cat((rotation, translation[..., None]), dim=-1),
        pose_encoding[..., 7],
        pose_encoding[..., 8],
    )


def pose_fov_to_intri(
    fov_h: torch.Tensor,
    fov_w: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size_hw
    fy = (height / 2) / torch.tan(fov_h / 2)
    fx = (width / 2) / torch.tan(fov_w / 2)
    intrinsics = torch.zeros(
        fov_h.shape + (3, 3), dtype=fov_h.dtype, device=fov_h.device
    )
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 2, 2] = 1
    return intrinsics


def _inverse_se3(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("SE3 matrices must end in 3x4 or 4x4")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3:]
    inverse = torch.zeros(
        transform.shape[:-2] + (4, 4), dtype=transform.dtype, device=transform.device
    )
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3:] = -rotation.transpose(-1, -2) @ translation
    inverse[..., 3, 3] = 1
    return inverse


def unproject_depth(
    depth_map: torch.Tensor | np.ndarray,
    extrinsics: torch.Tensor | np.ndarray,
    intrinsics: torch.Tensor | np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = torch.as_tensor(depth_map).float()
    poses = torch.as_tensor(extrinsics, device=depth.device).float()
    calibration = torch.as_tensor(intrinsics, device=depth.device).float()
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B,V,H,W]")
    batch, views, height, width = depth.shape
    if poses.shape != (batch, views, 3, 4):
        raise ValueError("extrinsics must have shape [B,V,3,4]")
    if calibration.shape != (batch, views, 3, 3):
        raise ValueError("intrinsics must have shape [B,V,3,3]")
    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    fx = calibration[..., 0, 0, None, None]
    fy = calibration[..., 1, 1, None, None]
    cx = calibration[..., 0, 2, None, None]
    cy = calibration[..., 1, 2, None, None]
    camera = torch.stack(
        ((x - cx) * depth / fx, (y - cy) * depth / fy, depth), dim=-1
    )
    camera_to_world = _inverse_se3(poses.reshape(batch * views, 3, 4))
    rotation = camera_to_world[:, :3, :3].reshape(batch, views, 3, 3)
    translation = camera_to_world[:, :3, 3].reshape(batch, views, 3)
    world = camera.reshape(batch, views, height * width, 3) @ rotation.transpose(-1, -2)
    world = (world + translation.unsqueeze(2)).reshape(batch, views, height, width, 3)
    return world, camera, depth > epsilon


def normalize_labels(
    extrinsics: torch.Tensor,
    cam_points: torch.Tensor | None = None,
    world_points: torch.Tensor | None = None,
    depths: torch.Tensor | None = None,
    scale_by_points: bool = True,
    point_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError("extrinsics must have shape [B,V,3,4]")
    if not torch.isfinite(extrinsics).all():
        raise ValueError("extrinsics must be finite")
    batch, views = extrinsics.shape[:2]
    homogeneous = torch.zeros(
        (batch, views, 4, 4), dtype=extrinsics.dtype, device=extrinsics.device
    )
    homogeneous[..., :3, :] = extrinsics
    homogeneous[..., 3, 3] = 1
    normalized_extrinsics = homogeneous @ _inverse_se3(homogeneous[:, 0]).unsqueeze(1)
    normalized_world = None
    if world_points is not None:
        rotation = extrinsics[:, 0, :3, :3]
        translation = extrinsics[:, 0, :3, 3]
        normalized_world = world_points @ rotation.transpose(-1, -2)[:, None, None]
        normalized_world = normalized_world + translation[:, None, None, None]
    if not scale_by_points:
        return normalized_extrinsics[..., :3, :], cam_points, normalized_world, depths
    if normalized_world is None or point_masks is None:
        raise ValueError("scale normalization requires world points and point masks")
    distance = normalized_world.norm(dim=-1)
    scale = (distance * point_masks).sum(dim=(1, 2, 3))
    scale = (scale / (point_masks.sum(dim=(1, 2, 3)) + 1e-3)).clamp(1e-6, 1e6)
    normalized_world = normalized_world / scale[:, None, None, None, None]
    normalized_extrinsics[..., :3, 3] /= scale[:, None, None]
    normalized_depths = None if depths is None else depths / scale[:, None, None, None]
    normalized_camera = None if cam_points is None else cam_points / scale[:, None, None, None, None]
    values = (normalized_extrinsics[..., :3, :], normalized_camera, normalized_world, normalized_depths)
    if any(value is not None and not torch.isfinite(value).all() for value in values):
        raise FloatingPointError("normalized geometry contains non-finite values")
    return values


def _project_points(
    world_points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    homogeneous = torch.cat((world_points, torch.ones_like(world_points[..., :1])), dim=-1)
    camera = torch.bmm(extrinsics, homogeneous.t().unsqueeze(0).expand(len(extrinsics), -1, -1))
    pixel_h = torch.bmm(intrinsics, camera)
    pixels = (pixel_h[:, :2] / pixel_h[:, 2:3]).transpose(1, 2)
    return pixels, camera


def _static_frame_mask(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    world_points: torch.Tensor,
    depths: torch.Tensor,
    candidate: torch.Tensor,
    query_index: int,
    minimum_views: int,
    relative_threshold: float,
    boundary: int,
) -> torch.Tensor:
    views, height, width = world_points.shape[:3]
    result = torch.zeros((height, width), dtype=torch.bool, device=world_points.device)
    if not candidate.any():
        return result
    query = world_points[query_index][candidate]
    rows, columns = torch.nonzero(candidate, as_tuple=True)
    pixels, camera = _project_points(query, extrinsics, intrinsics)
    projected_depth = camera[:, 2]
    floor_pixels = pixels.floor().long()
    inside = (
        (floor_pixels[..., 0] >= boundary)
        & (floor_pixels[..., 0] < width - boundary)
        & (floor_pixels[..., 1] >= boundary)
        & (floor_pixels[..., 1] < height - boundary)
        & torch.isfinite(projected_depth)
        & (projected_depth > 0)
    )
    safe = floor_pixels.clone()
    safe[~inside] = 0
    frame_indices = torch.arange(views, device=depths.device)[:, None].expand(-1, safe.shape[1])
    consistent = torch.zeros_like(inside)
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sample = safe + torch.tensor((offset_x, offset_y), device=safe.device)
        sampled_depth = depths[frame_indices, sample[..., 1], sample[..., 0]]
        difference = (projected_depth - sampled_depth).abs()
        consistent |= (
            torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & (difference < projected_depth * relative_threshold)
            & (difference < sampled_depth * relative_threshold)
        )
    selected = (inside & consistent).sum(0) >= minimum_views
    result[rows[selected], columns[selected]] = True
    return result


@torch.no_grad()
def build_static_masks(
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    candidate_masks: torch.Tensor | None = None,
    visibility_tolerance: int = 2,
    relative_depth_threshold: float = 0.05,
    boundary: int = 4,
) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 4 or depth.shape[1] < 2:
        raise ValueError("static support requires at least two depth frames")
    if visibility_tolerance < 0:
        raise ValueError("visibility tolerance must be nonnegative")
    world, _, valid_depth = unproject_depth(depth, extrinsics, intrinsics)
    candidates = valid_depth if candidate_masks is None else candidate_masks.bool() & valid_depth
    if candidates.shape != depth.shape:
        raise ValueError("candidate masks must match depth")
    minimum = min(depth.shape[1], max(2, depth.shape[1] - visibility_tolerance))
    output = torch.zeros_like(depth, dtype=torch.bool)
    for batch_index in range(depth.shape[0]):
        for query_index in range(depth.shape[1]):
            output[batch_index, query_index] = _static_frame_mask(
                extrinsics[batch_index],
                intrinsics[batch_index],
                world[batch_index],
                depth[batch_index],
                candidates[batch_index, query_index],
                query_index,
                minimum,
                relative_depth_threshold,
                boundary,
            )
    return output


@dataclass(frozen=True)
class GeometryBindings:
    MultitaskLoss: type[Any]
    build_static_masks: Callable[..., Any]
    normalize_labels: Callable[..., Any]
    unproject_depth: Callable[..., Any]
    pose_encoding_to_extri_fov: Callable[..., Any]
    pose_fov_to_intri: Callable[..., Any]


def load_geometry_bindings() -> GeometryBindings:
    return GeometryBindings(
        MultitaskLoss,
        build_static_masks,
        normalize_labels,
        unproject_depth,
        pose_encoding_to_extri_fov,
        pose_fov_to_intri,
    )


def build_static_support(bindings: GeometryBindings, *, depth: Any, extrinsics: Any, intrinsics: Any) -> Any:
    return bindings.build_static_masks(depth=depth, extrinsics=extrinsics, intrinsics=intrinsics).bool()


def _depth_map(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 5 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 5 and value.shape[2] == 1:
        return value[:, :, 0]
    if value.ndim == 4:
        return value
    raise ValueError("depth tensor must have shape [B,V,H,W] with an optional singleton channel")


def _sample_views(values: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    batch, views, height, width = values.shape
    sampled = F.grid_sample(
        values.reshape(batch * views, 1, height, width).float(),
        grid.reshape(batch * views, height, width, 2).float(),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0].reshape(batch, views, height, width)


def predicted_pinhole_cross_view_geometry(
    *,
    teacher_depth: torch.Tensor,
    student_depth: torch.Tensor,
    student_confidence: torch.Tensor,
    teacher_confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    static_mask: torch.Tensor,
    fov_mask: torch.Tensor,
    rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    teacher = _depth_map(teacher_depth).detach().float()
    student = _depth_map(student_depth).float()
    student_confidence = _depth_map(student_confidence).float()
    teacher_confidence = _depth_map(teacher_confidence).detach().float()
    if teacher.shape != student.shape or student_confidence.shape != student.shape or teacher_confidence.shape != student.shape:
        raise ValueError("target depth and confidence tensors must share shape")
    batch, views, height, width = teacher.shape
    if views != 8:
        raise ValueError("Colon3R target geometry requires eight views")
    if intrinsics.shape != (batch, views, 3, 3) or extrinsics.shape != (batch, views, 3, 4):
        raise ValueError("target cameras have invalid shape")
    if static_mask.shape != teacher.shape or fov_mask.shape != teacher.shape:
        raise ValueError("target masks must match depth shape")
    if rgb.shape != (batch, views, 3, height, width):
        raise ValueError("target RGB must have shape [B,V,3,H,W]")
    device = student.device
    teacher = teacher.to(device)
    teacher_confidence = teacher_confidence.to(device)
    intrinsics = intrinsics.detach().float().to(device)
    extrinsics = extrinsics.detach().float().to(device)
    static_mask = static_mask.detach().bool().to(device)
    fov_mask = fov_mask.detach().bool().to(device)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    reference_pixels = torch.stack((xs, ys), dim=-1)[None, None]
    fields: dict[str, list[torch.Tensor]] = {
        "projection_valid": [],
        "projected_depth": [],
        "sampled_depth": [],
        "sampled_confidence": [],
        "projected_fov_valid": [],
        "in_bounds_projection": [],
        "z_buffer_visible": [],
        "forward_backward_pixel_closure": [],
        "forward_backward_pair_closure": [],
    }
    for reference in range(views):
        source_depth = teacher[:, reference]
        source_valid = torch.isfinite(source_depth) & (source_depth > 0)
        safe_depth = torch.where(source_valid, source_depth, torch.ones_like(source_depth))
        source_k = intrinsics[:, reference]
        source_camera = torch.stack(
            (
                (xs[None] - source_k[:, 0, 2, None, None]) * safe_depth / source_k[:, 0, 0, None, None].clamp_min(1e-8),
                (ys[None] - source_k[:, 1, 2, None, None]) * safe_depth / source_k[:, 1, 1, None, None].clamp_min(1e-8),
                safe_depth,
            ),
            dim=-1,
        )
        source_rotation = extrinsics[:, reference, :3, :3]
        source_translation = extrinsics[:, reference, :3, 3]
        source_world = torch.einsum(
            "bij,bhwj->bhwi",
            source_rotation.transpose(-1, -2),
            source_camera - source_translation[:, None, None],
        )
        target_camera = torch.einsum(
            "bvij,bhwj->bvhwi", extrinsics[..., :3], source_world
        ) + extrinsics[..., 3][:, :, None, None]
        projected_depth = target_camera[..., 2]
        safe_projected = projected_depth.clamp_min(1e-8)
        projected_x = intrinsics[..., 0, 0, None, None] * target_camera[..., 0] / safe_projected + intrinsics[..., 0, 2, None, None]
        projected_y = intrinsics[..., 1, 1, None, None] * target_camera[..., 1] / safe_projected + intrinsics[..., 1, 2, None, None]
        in_bounds = (projected_x >= 0) & (projected_x <= width - 1) & (projected_y >= 0) & (projected_y <= height - 1)
        camera_valid = torch.isfinite(target_camera).all(dim=-1) & torch.isfinite(projected_x) & torch.isfinite(projected_y) & (projected_depth > 0)
        grid = torch.stack(
            (
                2 * projected_x / max(width - 1, 1) - 1,
                2 * projected_y / max(height - 1, 1) - 1,
            ),
            dim=-1,
        )
        sampled_teacher = _sample_views(teacher, grid, "bilinear")
        sampled_student = _sample_views(student, grid, "bilinear")
        sampled_student_conf = _sample_views(student_confidence, grid, "bilinear")
        sampled_static = _sample_views(static_mask.float(), grid, "nearest").bool()
        sampled_fov = _sample_views(fov_mask.float(), grid, "nearest").bool()
        target_valid = torch.isfinite(sampled_teacher) & (sampled_teacher > 0)
        safe_sampled = torch.where(target_valid, sampled_teacher, torch.ones_like(sampled_teacher))
        cycle_x = (projected_x - intrinsics[..., 0, 2, None, None]) * safe_sampled / intrinsics[..., 0, 0, None, None].clamp_min(1e-8)
        cycle_y = (projected_y - intrinsics[..., 1, 2, None, None]) * safe_sampled / intrinsics[..., 1, 1, None, None].clamp_min(1e-8)
        cycle_camera = torch.stack((cycle_x, cycle_y, safe_sampled), dim=-1)
        cycle_world = torch.einsum(
            "bvij,bvhwj->bvhwi",
            extrinsics[..., :3].transpose(-1, -2),
            cycle_camera - extrinsics[..., 3][:, :, None, None],
        )
        cycle_reference = torch.einsum(
            "bij,bvhwj->bvhwi", source_rotation, cycle_world
        ) + source_translation[:, None, None, None]
        cycle_z = cycle_reference[..., 2].clamp_min(1e-8)
        cycle_pixels = torch.stack(
            (
                source_k[:, None, 0, 0, None, None] * cycle_reference[..., 0] / cycle_z + source_k[:, None, 0, 2, None, None],
                source_k[:, None, 1, 1, None, None] * cycle_reference[..., 1] / cycle_z + source_k[:, None, 1, 2, None, None],
            ),
            dim=-1,
        )
        cycle_error = torch.linalg.vector_norm(cycle_pixels - reference_pixels, dim=-1)
        off_diagonal = torch.arange(views, device=device)[None, :, None, None] != reference
        source_support = static_mask[:, reference, None] & fov_mask[:, reference, None]
        projected_support = sampled_static & sampled_fov
        valid = source_support & projected_support & source_valid[:, None] & target_valid & camera_valid & in_bounds & off_diagonal
        closure = torch.where(valid & torch.isfinite(cycle_error), torch.exp(-0.5 * cycle_error.square()), torch.zeros_like(cycle_error))
        count = valid.sum(dim=(-2, -1))
        pair_closure = torch.where(count > 0, closure.sum(dim=(-2, -1)) / count.clamp_min(1), torch.zeros_like(count, dtype=torch.float32))
        visible = target_valid & torch.isfinite(projected_depth) & (projected_depth <= sampled_teacher + 0.05 * sampled_teacher.abs().clamp_min(1e-8))
        fields["projection_valid"].append(valid.detach())
        fields["projected_depth"].append(torch.where(source_valid[:, None], projected_depth, torch.full_like(projected_depth, float("nan"))).detach())
        fields["sampled_depth"].append(sampled_student)
        fields["sampled_confidence"].append(sampled_student_conf)
        fields["projected_fov_valid"].append(camera_valid.detach())
        fields["in_bounds_projection"].append(in_bounds.detach())
        fields["z_buffer_visible"].append(visible.detach())
        fields["forward_backward_pixel_closure"].append(closure.detach())
        fields["forward_backward_pair_closure"].append(pair_closure.detach())
    result = {name: torch.stack(values, dim=1) for name, values in fields.items()}
    result.update(
        {
            "extrinsics": extrinsics,
            "teacher_depth": teacher_depth,
            "teacher_confidence": teacher_confidence,
            "fov_mask": fov_mask,
            "rgb": rgb,
        }
    )
    return result
