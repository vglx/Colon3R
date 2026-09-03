from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


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


@dataclass(frozen=True)
class Wat3RBindings:
    root: Path
    MultitaskLoss: type[Any]
    build_static_masks: Callable[..., Any]
    project_teacher_depth_to_student_views: Callable[..., Any]
    normalize_labels: Callable[..., Any]
    unproject_depth: Callable[..., Any]
    pose_encoding_to_extri_fov: Callable[..., Any]
    pose_fov_to_intri: Callable[..., Any]


def _configured_module(root: Path, name: str) -> Any:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        raise ImportError(f"failed to import Wat3R module {name} from configured root {root}") from error
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if root not in module_path.parents:
        raise ImportError(f"Wat3R module {name} resolved outside configured root {root}: {module_path}")
    return module


def load_wat3r_bindings(wat3r_root: str | Path) -> Wat3RBindings:
    root = Path(wat3r_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Wat3R root does not exist: {root}; set config.paths.wat3r_root")
    required = (root / "training" / "loss.py", root / "training" / "losses" / "static_geo_loss.py", root / "training" / "train_utils" / "normalization.py", root / "wat3r" / "utils" / "static_mask.py", root / "wat3r" / "utils" / "geometry.py", root / "wat3r" / "utils" / "pose_enc.py")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Wat3R root is missing required geometry modules: {root}; set config.paths.wat3r_root")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    loss = _configured_module(root, "training.loss")
    static_loss = _configured_module(root, "training.losses.static_geo_loss")
    normalization = _configured_module(root, "training.train_utils.normalization")
    static_mask = _configured_module(root, "wat3r.utils.static_mask")
    geometry = _configured_module(root, "wat3r.utils.geometry")
    pose_enc = _configured_module(root, "wat3r.utils.pose_enc")
    required_symbols = {
        "MultitaskLoss": getattr(loss, "MultitaskLoss", None),
        "project_teacher_depth_to_student_views": getattr(static_loss, "_project_teacher_depth_to_student_views", None),
        "normalize_labels": getattr(normalization, "normalize_camera_extrinsics_and_points_batch", None),
        "build_static_masks": getattr(static_mask, "build_static_masks", None),
        "unproject_depth": getattr(geometry, "unproject_depth_map_to_point_map_bs", None),
        "pose_encoding_to_extri_fov": getattr(pose_enc, "pose_encoding_to_extri_fov", None),
        "pose_fov_to_intri": getattr(pose_enc, "pose_fov_to_intri", None),
    }
    missing = [name for name, value in required_symbols.items() if not callable(value)]
    if missing:
        raise ImportError(f"configured Wat3R root lacks required geometry symbols: {missing}")
    return Wat3RBindings(root, required_symbols["MultitaskLoss"], required_symbols["build_static_masks"], required_symbols["project_teacher_depth_to_student_views"], required_symbols["normalize_labels"], required_symbols["unproject_depth"], required_symbols["pose_encoding_to_extri_fov"], required_symbols["pose_fov_to_intri"])


def build_static_support(bindings: Wat3RBindings, *, depth: Any, extrinsics: Any, intrinsics: Any) -> Any:
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
