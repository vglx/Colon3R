from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .camera import PinholeCalibration, RayProvider
from .geometry import camera_to_world_points


TARGET_WIDTH = 518
PATCH_SIZE = 14


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"manifest does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL manifest row {line_number}: {source}") from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {source}")
    return rows


def patch_compatible_resize(width: int, height: int, target_width: int = TARGET_WIDTH, patch_size: int = PATCH_SIZE) -> dict[str, float | int]:
    if min(width, height, target_width, patch_size) <= 0 or target_width % patch_size:
        raise ValueError("image and target width must be positive; target width must be patch-compatible")
    target_height = max(patch_size, int(round((height * target_width / width) / patch_size)) * patch_size)
    return {"source_width": int(width), "source_height": int(height), "target_width": int(target_width), "target_height": int(target_height), "scale_x": float(target_width / width), "scale_y": float(target_height / height)}


def resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    target_height, target_width = int(shape[0]), int(shape[1])
    if source.ndim != 2 or target_height <= 0 or target_width <= 0:
        raise ValueError("mask must be [height,width] and resize target must be positive")
    source_height, source_width = source.shape
    rows = np.clip(np.round((np.arange(target_height) + 0.5) * source_height / target_height - 0.5).astype(int), 0, source_height - 1)
    cols = np.clip(np.round((np.arange(target_width) + 0.5) * source_width / target_width - 0.5).astype(int), 0, source_width - 1)
    return source[rows][:, cols]


def build_boundary_connected_fov_mask(frames: np.ndarray, *, illumination_threshold: int = 6) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("frames must have shape [time,height,width,3]")
    dark = ~np.any(np.max(array, axis=-1) > int(illumination_threshold), axis=0)
    invalid = np.zeros_like(dark, dtype=bool)
    height, width = dark.shape
    queue: deque[tuple[int, int]] = deque()
    for row in range(height):
        for col in (0, width - 1):
            if dark[row, col] and not invalid[row, col]:
                invalid[row, col] = True
                queue.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if dark[row, col] and not invalid[row, col]:
                invalid[row, col] = True
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + delta_row, col + delta_col
            if 0 <= next_row < height and 0 <= next_col < width and dark[next_row, next_col] and not invalid[next_row, next_col]:
                invalid[next_row, next_col] = True
                queue.append((next_row, next_col))
    return ~invalid


def directed_pairs(view_count: int = 8) -> tuple[tuple[int, int], ...]:
    if int(view_count) != 8:
        raise ValueError("Colon3R geometry requires exactly eight frames")
    return tuple((source, target) for source in range(8) for target in range(8) if source != target)


def _validate_eight_frame_row(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = dict(row)
    frame_ids, paths = value.get("frame_ids"), value.get("rgb_paths")
    if not isinstance(frame_ids, list) or not isinstance(paths, list) or len(frame_ids) != 8 or len(paths) != 8 or int(value.get("clip_length", -1)) != 8:
        raise ValueError(f"{label} rows must contain exactly eight frames")
    ids = [int(item) for item in frame_ids]
    if ids != sorted(ids) or len(set(ids)) != 8:
        raise ValueError(f"{label} frame IDs must be unique and chronological")
    if not all(isinstance(path, str) and path for path in paths):
        raise ValueError(f"{label} RGB paths must be nonempty strings")
    value["frame_ids"] = ids
    value["rgb_paths"] = list(paths)
    return value


def validate_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_eight_frame_row(row, "source")
    source, model = str(value.get("source", "")), str(value.get("camera_model", ""))
    if bool(value.get("fallback_used", False)):
        raise ValueError("source camera fallback is prohibited")
    if source == "c3vd":
        if model != "official_scaramuzza_omnidirectional":
            raise ValueError("C3VD requires official Scaramuzza camera model")
        if not isinstance(value.get("depth_paths"), list) or len(value["depth_paths"]) != 8 or not value.get("pose_path"):
            raise ValueError("C3VD source rows require eight depth paths and a pose path")
    elif source in {"sim", "exp_sim"}:
        if model not in {"pinhole", "explicit_pinhole"}:
            raise ValueError("sim requires explicit pinhole camera model")
        if not value.get("depth_cache") or not value.get("intrinsics_path") or not value.get("pose_path"):
            raise ValueError("sim source rows require depth cache, intrinsics, and poses")
    else:
        raise ValueError(f"unsupported source dataset {source!r}")
    return value


def validate_target_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_eight_frame_row(row, "target")
    if str(value.get("source", "target")) not in {"target", "invivo", "unlabeled"}:
        raise ValueError("target manifest rows must identify an unlabeled target source")
    forbidden = [key for key in value if any(token in key.lower() for token in ("depth", "pose", "trajectory", "mesh", "pointmap", "intrinsic", "calibration"))]
    if forbidden or value.get("supervised") is not False:
        raise ValueError("target rows are RGB-only and cannot contain labels or geometry")
    return value


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preprocess_eight_frames(frames: Sequence[np.ndarray], *, target_width: int = TARGET_WIDTH) -> dict[str, Any]:
    array = np.stack([np.asarray(frame) for frame in frames])
    if array.shape[0] != 8 or array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("preprocessing requires exactly eight uint8 RGB frames")
    if len({tuple(frame.shape) for frame in array}) != 1:
        raise ValueError("all eight RGB frames must have identical dimensions")
    source_height, source_width = array.shape[1:3]
    resize = patch_compatible_resize(source_width, source_height, target_width)
    width, height = int(resize["target_width"]), int(resize["target_height"])
    resized = np.stack([np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8) for frame in array])
    fov = resize_mask_nearest(build_boundary_connected_fov_mask(array), (height, width))
    return {"images": torch.from_numpy(np.ascontiguousarray(resized.transpose(0, 3, 1, 2))).float().div_(255.0), "fov_mask": torch.from_numpy(np.ascontiguousarray(fov)).bool(), "resize": resize}


class TargetManifestDataset:
    def __init__(self, manifest: Path, *, target_width: int = TARGET_WIDTH, image_loader: Callable[[Path], np.ndarray] = _load_rgb) -> None:
        self.rows = tuple(validate_target_row(row) for row in read_jsonl(manifest))
        self.target_width = int(target_width)
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index % len(self.rows)]
        output = preprocess_eight_frames([self.image_loader(Path(path)) for path in row["rgb_paths"]], target_width=self.target_width)
        output.update({"frame_ids": torch.as_tensor(row["frame_ids"], dtype=torch.long), "row": dict(row)})
        return output


def decode_c3vd_depth(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(encoded, dtype=np.uint16)
    valid = (values > 0) & (values < np.iinfo(np.uint16).max)
    depth = values.astype(np.float32) * (100.0 / 65535.0)
    depth[~valid] = 0.0
    return depth, valid


def _load_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.size % 16:
        raise ValueError(f"pose file does not contain 4x4 matrices: {path}")
    return values.reshape(-1, 4, 4)


class SourceManifestDataset:
    def __init__(self, manifest: Path, *, c3vd_calibration: Path, c3vd_fov_mask: Path | None = None, target_width: int = TARGET_WIDTH, image_loader: Callable[[Path], np.ndarray] = _load_rgb) -> None:
        self.rows = tuple(validate_source_row(row) for row in read_jsonl(manifest))
        self.provider = RayProvider(c3vd_calibration)
        self.c3vd_fov_mask = None if c3vd_fov_mask is None else Path(c3vd_fov_mask).expanduser().resolve()
        self.target_width, self.image_loader = int(target_width), image_loader

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, index: int) -> dict[str, Any]:
        row = self.rows[index % len(self.rows)]
        frames = [self.image_loader(Path(path)) for path in row["rgb_paths"]]
        prep = preprocess_eight_frames(frames, target_width=self.target_width)
        height, width = prep["images"].shape[-2:]
        poses = _load_poses(Path(row["pose_path"]))
        ids = row["frame_ids"]
        w2c = np.linalg.inv(poses[ids])
        pixels_y, pixels_x = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
        pixels = np.stack((pixels_x, pixels_y), axis=-1)
        if row["source"] == "c3vd":
            if self.c3vd_fov_mask is None or not self.c3vd_fov_mask.is_file():
                raise FileNotFoundError("C3VD source loading requires a configured FOV mask")
            fov = resize_mask_nearest(np.asarray(Image.open(self.c3vd_fov_mask).convert("L"), dtype=np.uint8) > 0, (height, width))
            depths, valid = zip(*(decode_c3vd_depth(np.asarray(Image.open(path), dtype=np.uint16)) for path in row["depth_paths"]))
            depths = np.stack([np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.NEAREST), dtype=np.float32) for depth in depths])
            valid = np.stack([resize_mask_nearest(mask, (height, width)) for mask in valid])
            model, _ = self.provider.for_manifest_row(row, output_size=(width, height))
        else:
            cache = np.load(Path(row["depth_cache"]), allow_pickle=False)
            depths, valid = np.asarray(cache["depth"])[ids], np.asarray(cache["valid"], dtype=bool)[ids]
            depths = np.stack([np.asarray(Image.fromarray(depth.astype(np.float32)).resize((width, height), Image.Resampling.NEAREST), dtype=np.float32) for depth in depths])
            valid = np.stack([resize_mask_nearest(mask, (height, width)) for mask in valid])
            parameters = json.loads(Path(row["intrinsics_path"]).read_text(encoding="utf-8"))
            calibration = PinholeCalibration(frames[0].shape[1], frames[0].shape[0], float(parameters["fx"]), float(parameters["fy"]), float(parameters["cx"]), float(parameters["cy"]))
            model, _ = self.provider.for_manifest_row(row, pinhole_calibration=calibration, output_size=(width, height))
            fov = np.ones((height, width), dtype=bool)
        rays = model.pixel_to_unit_ray(pixels)
        camera = rays[None] * (depths / rays[..., 2])[..., None]
        world = camera_to_world_points(camera, w2c)
        intrinsics = model.camera_supervision_matrix().astype(np.float32)
        return {"images": prep["images"].numpy(), "depths": depths.astype(np.float32), "point_masks": valid & fov, "cam_points": camera.astype(np.float32), "world_points": world.astype(np.float32), "extrinsics": w2c[:, :3].astype(np.float32), "intrinsics": np.repeat(intrinsics[None], 8, axis=0), "row": dict(row), "resize": prep["resize"]}
