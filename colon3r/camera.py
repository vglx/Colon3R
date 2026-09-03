from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PixelTransform:
    base_width: int
    base_height: int
    width: int
    height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    translate_x: float = 0.0
    translate_y: float = 0.0

    @classmethod
    def identity(cls, width: int, height: int) -> "PixelTransform":
        return cls(int(width), int(height), int(width), int(height))

    @classmethod
    def from_crop_resize(
        cls,
        base_width: int,
        base_height: int,
        width: int,
        height: int,
        crop: tuple[float, float, float, float] | None = None,
        *,
        pixel_center: bool = False,
    ) -> "PixelTransform":
        left, top, crop_width, crop_height = crop or (0.0, 0.0, float(base_width), float(base_height))
        if min(base_width, base_height, width, height) <= 0 or crop_width <= 0 or crop_height <= 0:
            raise ValueError("camera image and crop dimensions must be positive")
        if left < 0 or top < 0 or left + crop_width > base_width or top + crop_height > base_height:
            raise ValueError("camera crop lies outside the calibration image")
        scale_x = float(width) / float(crop_width)
        scale_y = float(height) / float(crop_height)
        center_x = 0.5 * scale_x - 0.5 if pixel_center else 0.0
        center_y = 0.5 * scale_y - 0.5 if pixel_center else 0.0
        return cls(
            int(base_width), int(base_height), int(width), int(height), scale_x, scale_y,
            -float(left) * scale_x + center_x, -float(top) * scale_y + center_y,
        )

    def from_base_pixels(self, pixels: np.ndarray) -> np.ndarray:
        value = np.asarray(pixels, dtype=np.float64)
        if value.shape[-1] != 2:
            raise ValueError("pixels must end in two coordinates")
        return value * np.asarray([self.scale_x, self.scale_y]) + np.asarray([self.translate_x, self.translate_y])

    def to_base_pixels(self, pixels: np.ndarray) -> np.ndarray:
        value = np.asarray(pixels, dtype=np.float64)
        if value.shape[-1] != 2:
            raise ValueError("pixels must end in two coordinates")
        return (value - np.asarray([self.translate_x, self.translate_y])) / np.asarray([self.scale_x, self.scale_y])


def points_from_unit_rays_and_z_depth(rays: np.ndarray, z_depth: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    unit_rays = np.asarray(rays, dtype=np.float64)
    depth = np.asarray(z_depth, dtype=np.float64)
    if unit_rays.shape[-1] != 3:
        raise ValueError("rays must end in XYZ")
    if np.any(np.abs(unit_rays[..., 2]) <= epsilon):
        raise ValueError("camera ray has near-zero Z and cannot represent Z-depth")
    return unit_rays * (depth / unit_rays[..., 2])[..., None]


@dataclass(frozen=True)
class PinholeCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pixel_transform: PixelTransform | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("invalid pinhole calibration")
        if self.pixel_transform is None:
            object.__setattr__(self, "pixel_transform", PixelTransform.identity(self.width, self.height))

    def transformed(self, width: int, height: int, crop: tuple[float, float, float, float] | None = None) -> "PinholeCalibration":
        return replace(self, pixel_transform=PixelTransform.from_crop_resize(self.width, self.height, width, height, crop))


class PinholeCameraModel:
    name = "pinhole"

    def __init__(self, calibration: PinholeCalibration, *, name: str | None = None) -> None:
        self.calibration = calibration
        if name is not None:
            self.name = str(name)

    def pixel_to_unit_ray(self, pixels: np.ndarray) -> np.ndarray:
        base = self.calibration.pixel_transform.to_base_pixels(pixels)
        rays = np.stack(
            ((base[..., 0] - self.calibration.cx) / self.calibration.fx, (base[..., 1] - self.calibration.cy) / self.calibration.fy, np.ones(base.shape[:-1], dtype=np.float64)),
            axis=-1,
        )
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def camera_point_from_z_depth(self, pixels: np.ndarray, z_depth: np.ndarray) -> np.ndarray:
        return points_from_unit_rays_and_z_depth(self.pixel_to_unit_ray(pixels), z_depth)

    def camera_point_to_pixel(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_camera, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("camera points must end in XYZ")
        z = points[..., 2]
        valid = np.isfinite(points).all(axis=-1) & (z > 1e-10)
        safe_z = np.where(valid, z, 1.0)
        base = np.stack((self.calibration.fx * points[..., 0] / safe_z + self.calibration.cx, self.calibration.fy * points[..., 1] / safe_z + self.calibration.cy), axis=-1)
        pixels = self.calibration.pixel_transform.from_base_pixels(base)
        transform = self.calibration.pixel_transform
        inside = (pixels[..., 0] >= 0.0) & (pixels[..., 0] <= transform.width - 1) & (pixels[..., 1] >= 0.0) & (pixels[..., 1] <= transform.height - 1)
        return pixels, valid & inside

    def camera_supervision_matrix(self) -> np.ndarray:
        calibration = self.calibration
        transform = calibration.pixel_transform
        return np.asarray(
            [
                [transform.scale_x * calibration.fx, 0.0, transform.scale_x * calibration.cx],
                [0.0, transform.scale_y * calibration.fy, transform.scale_y * calibration.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


_SCARAMUZZA_KEYS = ("width", "height", "cx", "cy", "a0", "a2", "a3", "a4", "c", "d", "e")


@dataclass(frozen=True)
class ScaramuzzaCalibration:
    width: int
    height: int
    cx: float
    cy: float
    a0: float
    a2: float
    a3: float
    a4: float
    c: float
    d: float
    e: float
    source_path: str
    source_sha256: str
    archive_sha256: str | None = None
    pixel_transform: PixelTransform | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or abs(self.a0) <= 1e-12:
            raise ValueError("invalid Scaramuzza calibration")
        if abs(float(np.linalg.det(np.asarray([[self.c, self.d], [self.e, 1.0]], dtype=np.float64)))) <= 1e-12:
            raise ValueError("singular Scaramuzza stretch matrix")
        if self.pixel_transform is None:
            object.__setattr__(self, "pixel_transform", PixelTransform.identity(self.width, self.height))

    def transformed(self, width: int, height: int, crop: tuple[float, float, float, float] | None = None) -> "ScaramuzzaCalibration":
        return replace(self, pixel_transform=PixelTransform.from_crop_resize(self.width, self.height, width, height, crop, pixel_center=True))


def load_scaramuzza_calibration(path: Path) -> ScaramuzzaCalibration:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"C3VD calibration does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid C3VD calibration: {source}") from error
    missing = [key for key in _SCARAMUZZA_KEYS if key not in value]
    if missing:
        raise ValueError(f"missing C3VD Scaramuzza calibration values: {missing}")
    return ScaramuzzaCalibration(
        **{key: value[key] for key in _SCARAMUZZA_KEYS}, source_path=str(source), source_sha256=_sha256(source), archive_sha256=value.get("archive_sha256")
    )


class ScaramuzzaCameraModel:
    name = "official_scaramuzza_omnidirectional"

    def __init__(self, calibration: ScaramuzzaCalibration) -> None:
        self.calibration = calibration
        self._stretch = np.asarray([[calibration.c, calibration.d], [calibration.e, 1.0]], dtype=np.float64)
        self._inverse_stretch = np.linalg.inv(self._stretch)
        self._rho_max = self._maximum_calibration_radius()
        rho = np.linspace(0.0, self._rho_max * 1.05, 65537, dtype=np.float64)
        theta = np.arctan2(rho, self._poly(rho))
        order = np.argsort(theta)
        theta_sorted, rho_sorted = theta[order], rho[order]
        unique = np.concatenate(([True], np.diff(theta_sorted) > 1e-12))
        self._theta_lut, self._rho_lut = theta_sorted[unique], rho_sorted[unique]

    def _poly(self, rho: np.ndarray) -> np.ndarray:
        c = self.calibration
        return c.a0 + c.a2 * rho**2 + c.a3 * rho**3 + c.a4 * rho**4

    def _poly_derivative(self, rho: np.ndarray) -> np.ndarray:
        c = self.calibration
        return 2 * c.a2 * rho + 3 * c.a3 * rho**2 + 4 * c.a4 * rho**3

    def _corrected_plane(self, base_pixels: np.ndarray) -> np.ndarray:
        centered = base_pixels - np.asarray([self.calibration.cx, self.calibration.cy])
        return np.einsum("ij,...j->...i", self._inverse_stretch, centered)

    def _maximum_calibration_radius(self) -> float:
        corners = np.asarray([[0.0, 0.0], [self.calibration.width - 1.0, 0.0], [0.0, self.calibration.height - 1.0], [self.calibration.width - 1.0, self.calibration.height - 1.0]])
        return float(np.linalg.norm(self._corrected_plane(corners), axis=-1).max())

    def pixel_to_unit_ray(self, pixels: np.ndarray) -> np.ndarray:
        current = np.asarray(pixels, dtype=np.float64)
        if current.shape[-1] != 2:
            raise ValueError("pixels must end in two coordinates")
        base = self.calibration.pixel_transform.to_base_pixels(current)
        corrected = self._corrected_plane(base)
        rho = np.linalg.norm(corrected, axis=-1)
        rays = np.concatenate((corrected, self._poly(rho)[..., None]), axis=-1)
        norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        if np.any(norm <= 1e-12) or not np.isfinite(rays).all():
            raise ValueError("invalid Scaramuzza ray")
        return rays / norm

    def camera_point_from_z_depth(self, pixels: np.ndarray, z_depth: np.ndarray) -> np.ndarray:
        return points_from_unit_rays_and_z_depth(self.pixel_to_unit_ray(pixels), z_depth)

    def _rho_from_direction(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radial = np.linalg.norm(points[..., :2], axis=-1)
        z = points[..., 2]
        theta = np.arctan2(radial, z)
        valid = np.isfinite(points).all(axis=-1) & (radial > 1e-12)
        rho = np.interp(theta, self._theta_lut, self._rho_lut)
        target = z / np.where(radial > 1e-12, radial, 1.0)
        for _ in range(8):
            derivative = self._poly_derivative(rho) - target
            rho = np.clip(rho - (self._poly(rho) - target * rho) / np.where(np.abs(derivative) > 1e-12, derivative, 1.0), 0.0, self._rho_max * 1.1)
        return rho, valid

    def camera_point_to_pixel(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_camera, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("camera points must end in XYZ")
        radial = np.linalg.norm(points[..., :2], axis=-1)
        rho, valid = self._rho_from_direction(points)
        corrected = points[..., :2] * (rho / np.where(radial > 1e-12, radial, 1.0))[..., None]
        base = np.einsum("ij,...j->...i", self._stretch, corrected) + np.asarray([self.calibration.cx, self.calibration.cy])
        center = radial <= 1e-12
        base = np.where(center[..., None], np.asarray([self.calibration.cx, self.calibration.cy]), base)
        valid = (valid | (center & (points[..., 2] > 0.0))) & (points[..., 2] > 0.0)
        pixels = self.calibration.pixel_transform.from_base_pixels(base)
        transform = self.calibration.pixel_transform
        inside = (pixels[..., 0] >= -1e-6) & (pixels[..., 0] <= transform.width - 1 + 1e-6) & (pixels[..., 1] >= -1e-6) & (pixels[..., 1] <= transform.height - 1 + 1e-6)
        return pixels, valid & inside

    def camera_supervision_matrix(self) -> np.ndarray:
        calibration = self.calibration
        transform = calibration.pixel_transform
        base = np.asarray(
            [
                [calibration.a0 * calibration.c, calibration.a0 * calibration.d, calibration.cx],
                [calibration.a0 * calibration.e, calibration.a0, calibration.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base[0] *= transform.scale_x
        base[1] *= transform.scale_y
        return base


class RayProvider:
    def __init__(self, c3vd_calibration_path: Path) -> None:
        self.c3vd_calibration_path = Path(c3vd_calibration_path).expanduser().resolve()
        self._c3vd_calibration = load_scaramuzza_calibration(self.c3vd_calibration_path)

    def for_manifest_row(
        self,
        row: dict[str, Any],
        pinhole_calibration: PinholeCalibration | None = None,
        output_size: tuple[int, int] | None = None,
        crop: tuple[float, float, float, float] | None = None,
    ) -> tuple[ScaramuzzaCameraModel | PinholeCameraModel, dict[str, Any]]:
        source, requested = str(row.get("source", "")), str(row.get("camera_model", ""))
        if source == "c3vd":
            if requested != "official_scaramuzza_omnidirectional":
                raise ValueError(f"C3VD requires official Scaramuzza camera model, got {requested!r}")
            calibration = self._c3vd_calibration if output_size is None else self._c3vd_calibration.transformed(output_size[0], output_size[1], crop)
            model = ScaramuzzaCameraModel(calibration)
            return model, {"dataset_source": source, "requested_model": requested, "selected_model": model.name, "calibration_path": str(self.c3vd_calibration_path), "calibration_sha256": calibration.source_sha256, "fallback_used": False}
        if source in {"sim", "exp_sim"}:
            if requested not in {"pinhole", "explicit_pinhole"}:
                raise ValueError(f"sim requires pinhole camera model, got {requested!r}")
            if pinhole_calibration is None:
                raise ValueError("sim pinhole routing requires explicit calibration")
            calibration = pinhole_calibration if output_size is None else pinhole_calibration.transformed(output_size[0], output_size[1], crop)
            model = PinholeCameraModel(calibration)
            return model, {"dataset_source": source, "requested_model": requested, "selected_model": model.name, "calibration_path": None, "calibration_sha256": None, "fallback_used": False}
        raise ValueError(f"unsupported dataset source {source!r}; camera model cannot be inferred")

    def for_target_predicted_pinhole(self, intrinsics: np.ndarray, *, image_size: tuple[int, int]) -> tuple[PinholeCameraModel, dict[str, Any]]:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        width, height = (int(image_size[0]), int(image_size[1]))
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("Teacher-predicted pinhole intrinsics must be a finite 3x3 pinhole matrix")
        calibration = PinholeCalibration(width, height, float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2]))
        model = PinholeCameraModel(calibration, name="teacher_predicted_pinhole")
        return model, {"dataset_source": "target", "requested_model": "teacher_predicted_pinhole", "selected_model": model.name, "calibration_path": None, "calibration_sha256": None, "fallback_used": False}

    def z_depth_to_camera_point(self, row: dict[str, Any], pixels: np.ndarray, z_depth: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        model, route = self.for_manifest_row(row, **kwargs)
        return model.camera_point_from_z_depth(pixels, z_depth), route

    def camera_point_to_pixel(self, row: dict[str, Any], points_camera: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        model, route = self.for_manifest_row(row, **kwargs)
        pixels, valid = model.camera_point_to_pixel(points_camera)
        return pixels, valid, route

    @staticmethod
    def combine_valid_mask(depth_valid: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
        depth, fov = np.asarray(depth_valid, dtype=bool), np.asarray(fov_mask, dtype=bool)
        if depth.shape != fov.shape:
            raise ValueError("depth-valid and FOV masks must have identical shape")
        return depth & fov
