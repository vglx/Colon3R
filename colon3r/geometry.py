from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


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
    build_static_masks: Callable[..., Any]
    project_teacher_depth_to_student_views: Callable[..., Any]
    normalize_labels: Callable[..., Any]
    unproject_depth: Callable[..., Any]


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
    required = (root / "training" / "losses" / "static_geo_loss.py", root / "training" / "train_utils" / "normalization.py", root / "wat3r" / "utils" / "static_mask.py", root / "wat3r" / "utils" / "geometry.py")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Wat3R root is missing required geometry modules: {root}; set config.paths.wat3r_root")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    static_loss = _configured_module(root, "training.losses.static_geo_loss")
    normalization = _configured_module(root, "training.train_utils.normalization")
    static_mask = _configured_module(root, "wat3r.utils.static_mask")
    geometry = _configured_module(root, "wat3r.utils.geometry")
    required_symbols = {
        "project_teacher_depth_to_student_views": getattr(static_loss, "_project_teacher_depth_to_student_views", None),
        "normalize_labels": getattr(normalization, "normalize_camera_extrinsics_and_points_batch", None),
        "build_static_masks": getattr(static_mask, "build_static_masks", None),
        "unproject_depth": getattr(geometry, "unproject_depth_map_to_point_map_bs", None),
    }
    missing = [name for name, value in required_symbols.items() if not callable(value)]
    if missing:
        raise ImportError(f"configured Wat3R root lacks required geometry symbols: {missing}")
    return Wat3RBindings(root, required_symbols["build_static_masks"], required_symbols["project_teacher_depth_to_student_views"], required_symbols["normalize_labels"], required_symbols["unproject_depth"])


def build_static_support(bindings: Wat3RBindings, *, depth: Any, extrinsics: Any, intrinsics: Any) -> Any:
    return bindings.build_static_masks(depth=depth, extrinsics=extrinsics, intrinsics=intrinsics).bool()
