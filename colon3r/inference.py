from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from .checkpoint import COLON3R_FINAL_SHA256, load_colon3r_checkpoint
from .data import preprocess_eight_frames
from .geometry import load_wat3r_bindings
from .model import build_student
from .training import load_config


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_input_frames(values: Sequence[str | Path]) -> tuple[Path, ...]:
    items = [Path(value).expanduser().resolve() for value in values]
    if len(items) == 1 and items[0].is_dir():
        items = sorted(
            path.resolve()
            for path in items[0].iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    elif len(items) == 1 and items[0].is_file() and items[0].suffix.lower() == ".txt":
        parent = items[0].parent
        items = [
            (parent / line.strip()).resolve()
            for line in items[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if len(items) != 8:
        raise ValueError(f"inference requires exactly eight RGB frames, found {len(items)}")
    if not all(path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES for path in items):
        raise FileNotFoundError("one or more input RGB frames do not exist")
    if len(set(items)) != 8:
        raise ValueError("inference frame paths must be unique")
    return tuple(items)


def _atomic_bytes(path: Path, content: bytes) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_prediction_bundle(
    output_directory: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    values = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise FloatingPointError(f"prediction array is non-finite or non-numeric: {name}")
        values[name] = np.ascontiguousarray(array)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output, prefix=".predictions.", suffix=".npz"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    prediction_path = output / "predictions.npz"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, prediction_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata_path = output / "metadata.json"
    _atomic_bytes(
        metadata_path,
        (json.dumps(dict(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return {"predictions": str(prediction_path), "metadata": str(metadata_path)}


def write_rgb_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    destination = Path(path).expanduser().resolve()
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("PLY points and colors must have shape [N,3]")
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("PLY requires finite nonempty points")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    payload = np.empty(
        len(xyz),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    payload["x"], payload["y"], payload["z"] = xyz.T
    payload["red"], payload["green"], payload["blue"] = rgb.T
    _atomic_bytes(destination, header + payload.tobytes(order="C"))


def prepare_native_cloud(
    pointmap: np.ndarray,
    confidence: np.ndarray,
    images: np.ndarray,
    fov_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    points = np.asarray(pointmap, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    rgb = np.asarray(images, dtype=np.float32)
    if points.ndim != 4 or points.shape[0] != 8 or points.shape[-1] != 3:
        raise ValueError("direct pointmap must have shape [8,H,W,3]")
    if scores.shape != points.shape[:-1] or rgb.shape != (8, 3, points.shape[1], points.shape[2]):
        raise ValueError("pointmap, confidence, and RGB shapes differ")
    fov = np.asarray(fov_mask, dtype=bool)
    if fov.shape == points.shape[1:3]:
        fov = np.broadcast_to(fov, scores.shape)
    if fov.shape != scores.shape:
        raise ValueError("FOV mask shape differs from pointmap")
    colors = np.clip(np.rint(rgb.transpose(0, 2, 3, 1) * 255), 0, 255).astype(np.uint8)
    mask = fov & np.isfinite(points).all(-1) & np.isfinite(scores) & (np.linalg.norm(points, axis=-1) > 0)
    frames = tuple((np.ascontiguousarray(points[index][mask[index]]), np.ascontiguousarray(colors[index][mask[index]])) for index in range(8))
    if any(len(frame_points) == 0 for frame_points, _ in frames):
        raise ValueError("one or more pointmap frames contain no valid points")
    return (
        np.ascontiguousarray(np.concatenate([value[0] for value in frames])),
        np.ascontiguousarray(np.concatenate([value[1] for value in frames])),
        frames,
    )


def _head_output(
    head: nn.Module,
    tokens: Any,
    images: torch.Tensor,
    patch_start: int,
) -> Any:
    if "images" in inspect.signature(head.forward).parameters:
        return head(
            tokens,
            images=images,
            patch_start_idx=patch_start,
            frames_chunk_size=4,
        )
    return head(tokens)


@torch.no_grad()
def one_forward(model: nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    module = model.module if hasattr(model, "module") else model
    aggregated = module.aggregator(images)
    if not isinstance(aggregated, tuple) or len(aggregated) < 2 or not isinstance(aggregated[1], int):
        raise RuntimeError("VGGT aggregator returned an unsupported value")
    tokens, patch_start = aggregated[:2]
    poses = module.camera_head(tokens)
    pose = poses[-1] if isinstance(poses, (list, tuple)) else poses
    depth = _head_output(module.depth_head, tokens, images, patch_start)
    pointmap = _head_output(module.point_head, tokens, images, patch_start)
    if isinstance(depth, tuple) and len(depth) == 2:
        depth_value, depth_confidence = depth
    else:
        raise TypeError("VGGT depth head must return depth and confidence")
    if isinstance(pointmap, tuple) and len(pointmap) == 2:
        point_value, point_confidence = pointmap
    else:
        raise TypeError("VGGT point head must return pointmap and confidence")
    return {
        "pose_enc": pose,
        "depth": depth_value,
        "depth_conf": depth_confidence,
        "world_points": point_value,
        "world_points_conf": point_confidence,
    }


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def decode_prediction(
    prediction: Mapping[str, torch.Tensor],
    image_size: tuple[int, int],
    bindings: Any,
) -> dict[str, np.ndarray]:
    extrinsics, fov_h, fov_w = bindings.pose_encoding_to_extri_fov(prediction["pose_enc"])
    intrinsics = bindings.pose_fov_to_intri(
        fov_h=fov_h, fov_w=fov_w, image_size_hw=image_size
    )
    w2c = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).view(1, 1, 4, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1)
    w2c[..., :3, :] = extrinsics
    result = {
        "extrinsics_w2c": _as_numpy(w2c[0]),
        "intrinsics": _as_numpy(intrinsics[0]),
        "depth_z": _as_numpy(prediction["depth"][0, ..., 0]),
        "depth_confidence": _as_numpy(prediction["depth_conf"][0]),
        "direct_pointmap": _as_numpy(prediction["world_points"][0]),
        "pointmap_confidence": _as_numpy(prediction["world_points_conf"][0]),
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite model prediction: {name}")
    return result


def run_inference(
    *,
    input_values: Sequence[str | Path],
    checkpoint: str | Path,
    output_directory: str | Path,
    config_path: str | Path = "configs/colon3r.yaml",
    checkpoint_sha256: str | None = None,
    vggt_path: str | Path | None = None,
    export_ply: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Colon3R inference requires a CUDA GPU")
    frames = resolve_input_frames(input_values)
    raw = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in frames]
    preprocessed = preprocess_eight_frames(raw)
    images = preprocessed["images"].unsqueeze(0).cuda()
    config = load_config(config_path)
    model = build_student(config, vggt_path=vggt_path, device="cuda")
    expected = checkpoint_sha256
    if expected is None and Path(checkpoint).name == "colon3r_10k.pt":
        expected = COLON3R_FINAL_SHA256
    loaded = load_colon3r_checkpoint(
        checkpoint, model, expected_sha256=expected
    )
    model.eval().requires_grad_(False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = one_forward(model, images)
    bindings = load_wat3r_bindings(config["paths"]["wat3r_root"])
    arrays = decode_prediction(
        prediction,
        tuple(int(value) for value in images.shape[-2:]),
        bindings,
    )
    metadata = {
        "frame_paths": [str(path) for path in frames],
        "frame_count": 8,
        "model_forward_count": 1,
        "model_input_shape": list(images.shape),
        "resize": preprocessed["resize"],
        "checkpoint": loaded,
        "pose_convention": "world_to_camera",
        "depth_convention": "OpenCV_camera_axis_Z",
        "coordinate_convention": "native_VGGT_first_frame_normalized_world",
        "alignment": "none",
        "gt_inputs": [],
    }
    saved = save_prediction_bundle(output_directory, arrays, metadata)
    if export_ply:
        cloud, colors, per_frame = prepare_native_cloud(
            arrays["direct_pointmap"],
            arrays["pointmap_confidence"],
            preprocessed["images"].numpy(),
            preprocessed["fov_mask"].numpy(),
        )
        output = Path(output_directory).expanduser().resolve()
        fused = output / "direct_pointmap_fused_rgb.ply"
        write_rgb_ply(fused, cloud, colors)
        for index, (frame_points, frame_colors) in enumerate(per_frame):
            write_rgb_ply(
                output / "per_frame" / f"direct_pointmap_{index:02d}_rgb.ply",
                frame_points,
                frame_colors,
            )
        saved["fused_ply"] = str(fused)
    return {"files": saved, "metadata": metadata}


__all__ = [
    "decode_prediction",
    "one_forward",
    "prepare_native_cloud",
    "resolve_input_frames",
    "run_inference",
    "save_prediction_bundle",
    "write_rgb_ply",
]
