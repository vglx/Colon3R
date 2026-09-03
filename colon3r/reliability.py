from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


WINDOW_REASON_FLAGS = (
    "non_finite_geometry",
    "critical_fov",
    "severe_exposure",
    "severe_blur_contact",
    "severe_obstruction",
    "critical_overlap",
    "catastrophic_scale",
)

PAIR_COMPONENTS = (
    "usable_overlap",
    "normalized_baseline",
    "rotation_plausibility",
    "reprojection_depth_agreement",
    "valid_support",
    "forward_backward_closure",
)

PIXEL_COMPONENTS = (
    "teacher_confidence",
    "depth_agreement",
    "pixel_closure",
    "visibility_margin",
    "non_specular_support",
)

PIXEL_HARD_ZERO_FLAGS = (
    "invalid_fov_camera",
    "nonpositive_depth",
    "out_of_bounds_projection",
    "z_buffer_occlusion",
    "missing_support",
    "specular_core",
    "numerical_invalid",
)


@dataclass(frozen=True)
class WindowReliability:
    weight: torch.Tensor
    flags: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PairReliability:
    weight: torch.Tensor
    components: Mapping[str, torch.Tensor]
    hard_valid: torch.Tensor


@dataclass(frozen=True)
class PixelReliability:
    weight: torch.Tensor
    components: Mapping[str, torch.Tensor]
    hard_valid: torch.Tensor
    hard_zero_flags: Mapping[str, torch.Tensor]


def _depth_map(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 5:
        raise ValueError("depth must have five dimensions")
    if depth.shape[2] == 1:
        return depth[:, :, 0]
    if depth.shape[-1] == 1:
        return depth[..., 0]
    raise ValueError("depth must have a singleton channel")


def _geometric_mean(stacked: torch.Tensor) -> torch.Tensor:
    has_zero = (stacked == 0).any(dim=0)
    safe = torch.where(stacked > 0, stacked, torch.ones_like(stacked))
    positive = torch.exp(torch.log(safe).mean(dim=0))
    return torch.where(has_zero, torch.zeros_like(positive), positive)


def window_reliability(inputs: Mapping[str, torch.Tensor]) -> WindowReliability:
    required = (
        "finite_geometry",
        "fov_support",
        "exposure_support",
        "blur_support",
        "contact_fraction",
        "obstruction_fraction",
        "overlap_support",
        "scale_ratio",
    )
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"missing window reliability inputs: {missing}")
    with torch.no_grad():
        values = {name: inputs[name].detach() for name in required}
        shape = values["finite_geometry"].shape
        if len(shape) != 1 or any(value.shape != shape for value in values.values()):
            raise ValueError("window reliability inputs must share shape [B]")
        finite = torch.stack(
            [torch.isfinite(values[name].float()) for name in required[1:]], dim=0
        ).all(dim=0)
        flags = {
            "non_finite_geometry": ~(values["finite_geometry"].bool() & finite),
            "critical_fov": finite & (values["fov_support"] < 0.05),
            "severe_exposure": finite & (values["exposure_support"] < 0.05),
            "severe_blur_contact": finite
            & ((values["blur_support"] < 0.02) | (values["contact_fraction"] > 0.95)),
            "severe_obstruction": finite & (values["obstruction_fraction"] > 0.95),
            "critical_overlap": finite & (values["overlap_support"] < 0.01),
            "catastrophic_scale": finite
            & ((values["scale_ratio"] < 0.10) | (values["scale_ratio"] > 10.0)),
        }
        rejected = torch.stack(tuple(flags.values()), dim=0).any(dim=0)
        weight = (~rejected).float().detach()
    return WindowReliability(weight, {name: value.detach() for name, value in flags.items()})


def pair_reliability(
    geometry: Mapping[str, torch.Tensor], *, epsilon: float = 1e-8
) -> PairReliability:
    required = (
        "extrinsics",
        "teacher_depth",
        "static_mask",
        "projection_valid",
        "projected_depth",
        "sampled_depth",
        "forward_backward_pair_closure",
    )
    missing = [name for name in required if name not in geometry]
    if missing:
        raise ValueError(f"missing pair reliability geometry: {missing}")
    with torch.no_grad():
        extrinsics = geometry["extrinsics"].detach().float()
        teacher_depth = _depth_map(geometry["teacher_depth"].detach()).float()
        static_mask = geometry["static_mask"].detach().bool()
        projection_valid = geometry["projection_valid"].detach().bool()
        projected_depth = geometry["projected_depth"].detach().float()
        sampled_depth = geometry["sampled_depth"].detach().float()
        closure = geometry["forward_backward_pair_closure"].detach().float()
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (3, 4):
            raise ValueError("extrinsics must have shape [B,V,3,4]")
        batch, views = extrinsics.shape[:2]
        shape = (batch, views, views, teacher_depth.shape[-2], teacher_depth.shape[-1])
        if teacher_depth.shape[:2] != (batch, views) or static_mask.shape != teacher_depth.shape:
            raise ValueError("teacher depth, static mask, and extrinsics differ")
        for name, value in (
            ("projection_valid", projection_valid),
            ("projected_depth", projected_depth),
            ("sampled_depth", sampled_depth),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape [B,V,V,H,W]")
        if closure.shape != (batch, views, views):
            raise ValueError("forward_backward_pair_closure must have shape [B,V,V]")
        candidate = ~torch.eye(views, dtype=torch.bool, device=extrinsics.device).unsqueeze(0)
        camera_finite = torch.isfinite(extrinsics).all(dim=(-2, -1))
        depth_finite = torch.isfinite(teacher_depth).all(dim=(-2, -1))
        depth_positive = (teacher_depth > 0).any(dim=(-2, -1))
        view_valid = camera_finite & depth_finite & depth_positive
        finite_projection = (
            (~projection_valid)
            | (
                torch.isfinite(projected_depth)
                & torch.isfinite(sampled_depth)
                & (projected_depth > 0)
                & (sampled_depth > 0)
            )
        ).all(dim=(-2, -1))
        valid_count = projection_valid.sum(dim=(-2, -1))
        hard_valid = (
            candidate
            & view_valid[:, :, None]
            & view_valid[:, None, :]
            & finite_projection
            & torch.isfinite(closure)
            & (valid_count > 0)
        )
        safe_extrinsics = torch.nan_to_num(extrinsics)
        rotation = safe_extrinsics[..., :3]
        translation = safe_extrinsics[..., 3]
        centers = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
        baseline = torch.linalg.vector_norm(centers[:, :, None] - centers[:, None, :], dim=-1)
        positive_depth = torch.where(
            torch.isfinite(teacher_depth) & (teacher_depth > 0),
            teacher_depth,
            torch.full_like(teacher_depth, float("nan")),
        )
        scale = torch.nanmedian(positive_depth.reshape(batch, -1), dim=-1).values
        scale = torch.nan_to_num(scale, nan=1.0).clamp_min(epsilon)
        normalized = baseline / scale[:, None, None]
        baseline_score = (4.0 * normalized / (1.0 + normalized).square()).clamp(0.0, 1.0)
        relative_rotation = rotation[:, :, None] @ rotation[:, None, :].transpose(-1, -2)
        trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=-1)
        angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
        rotation_score = torch.exp(-((angle / 1.2) ** 2))
        source_support = static_mask.sum(dim=(-2, -1)).clamp_min(1)
        overlap = (valid_count.float() / source_support[:, :, None].float()).clamp(0.0, 1.0)
        valid_support = (
            valid_count.float() / float(teacher_depth.shape[-2] * teacher_depth.shape[-1])
        ).clamp(0.0, 1.0)
        relative_error = (projected_depth - sampled_depth).abs() / projected_depth.abs().clamp_min(epsilon)
        agreement_pixel = torch.exp(-relative_error / 0.20)
        agreement = (
            torch.where(projection_valid, agreement_pixel, torch.zeros_like(agreement_pixel)).sum(dim=(-2, -1))
            / valid_count.clamp_min(1).float()
        ).clamp(0.0, 1.0)
        raw = {
            "usable_overlap": overlap,
            "normalized_baseline": baseline_score,
            "rotation_plausibility": rotation_score,
            "reprojection_depth_agreement": agreement,
            "valid_support": valid_support,
            "forward_backward_closure": closure.clamp(0.0, 1.0),
        }
        components = {
            name: torch.where(
                hard_valid,
                torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0),
                torch.zeros_like(value),
            ).detach()
            for name, value in raw.items()
        }
        raw_weight = _geometric_mean(torch.stack(tuple(components.values()), dim=0))
        weight = torch.where(hard_valid, 0.1 + 0.9 * raw_weight, torch.zeros_like(raw_weight))
    return PairReliability(weight.clamp(0.0, 1.0).detach(), components, hard_valid.detach())


def pixel_reliability(
    geometry: Mapping[str, torch.Tensor], *, epsilon: float = 1e-8
) -> PixelReliability:
    required = (
        "extrinsics",
        "projection_valid",
        "projected_depth",
        "sampled_depth",
        "sampled_confidence",
        "teacher_confidence",
        "fov_mask",
        "rgb",
        "projected_fov_valid",
        "in_bounds_projection",
        "z_buffer_visible",
        "forward_backward_pixel_closure",
    )
    missing = [name for name in required if name not in geometry]
    if missing:
        raise ValueError(f"missing pixel reliability geometry: {missing}")
    with torch.no_grad():
        projection_input = geometry["projection_valid"].detach()
        projection_valid = projection_input.bool()
        shape = projection_valid.shape
        if projection_valid.ndim != 5 or shape[1] != shape[2]:
            raise ValueError("projection support must have shape [B,V,V,H,W]")
        batch, views, _, height, width = shape
        projected = geometry["projected_depth"].detach().float()
        sampled = geometry["sampled_depth"].detach().float()
        sampled_conf = geometry["sampled_confidence"].detach().float()
        for name, value in (
            ("projected_depth", projected),
            ("sampled_depth", sampled),
            ("sampled_confidence", sampled_conf),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape [B,V,V,H,W]")
        extrinsics = geometry["extrinsics"].detach().float()
        if extrinsics.shape != (batch, views, 3, 4):
            raise ValueError("extrinsics must have shape [B,V,3,4]")
        camera_valid = torch.isfinite(extrinsics).all(dim=(-2, -1))
        camera_pair = camera_valid[:, :, None, None, None] & camera_valid[:, None, :, None, None]
        fov_input = geometry["fov_mask"].detach()
        fov = fov_input.bool()
        if fov.shape != (batch, views, height, width):
            raise ValueError("fov_mask must have shape [B,V,H,W]")
        source_fov = fov[:, :, None].expand(shape)
        projected_fov_input = geometry["projected_fov_valid"].detach()
        in_bounds_input = geometry["in_bounds_projection"].detach()
        z_visible_input = geometry["z_buffer_visible"].detach()
        closure = geometry["forward_backward_pixel_closure"].detach().float()
        projected_fov = projected_fov_input.bool()
        in_bounds = in_bounds_input.bool()
        z_visible = z_visible_input.bool()
        for name, value in (
            ("projected_fov_valid", projected_fov),
            ("in_bounds_projection", in_bounds),
            ("z_buffer_visible", z_visible),
            ("forward_backward_pixel_closure", closure),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape [B,V,V,H,W]")
        confidence = geometry["teacher_confidence"].detach().float()
        if confidence.shape != (batch, views, height, width):
            raise ValueError("teacher_confidence must have shape [B,V,H,W]")
        confidence = confidence[:, :, None].expand(shape)
        rgb = geometry["rgb"].detach().float()
        if rgb.shape != (batch, views, 3, height, width):
            raise ValueError("rgb must have shape [B,V,3,H,W]")
        intensity = rgb.mean(dim=2)
        chroma = rgb.amax(dim=2) - rgb.amin(dim=2)
        specular = ((intensity >= 0.98) & (chroma <= 0.08))[:, :, None].expand(shape)
        numerical = (
            camera_pair
            & torch.isfinite(projection_input.float())
            & torch.isfinite(projected)
            & torch.isfinite(sampled)
            & torch.isfinite(sampled_conf)
            & torch.isfinite(confidence)
            & torch.isfinite(fov_input.float())[:, :, None].expand(shape)
            & torch.isfinite(rgb).all(dim=2)[:, :, None].expand(shape)
            & torch.isfinite(projected_fov_input.float())
            & torch.isfinite(in_bounds_input.float())
            & torch.isfinite(z_visible_input.float())
            & torch.isfinite(closure)
        )
        finite_depth = torch.isfinite(projected) & torch.isfinite(sampled)
        flags = {
            "invalid_fov_camera": ~(source_fov & projected_fov & camera_pair),
            "nonpositive_depth": finite_depth & ((projected <= 0) | (sampled <= 0)),
            "out_of_bounds_projection": ~in_bounds,
            "z_buffer_occlusion": ~z_visible,
            "missing_support": ~projection_valid,
            "specular_core": specular,
            "numerical_invalid": ~numerical,
        }
        hard_valid = ~torch.stack(tuple(flags.values()), dim=0).any(dim=0)
        relative_error = (projected - sampled).abs() / projected.abs().clamp_min(epsilon)
        non_specular = (1.0 - intensity * (1.0 - chroma.clamp(0.0, 1.0))).clamp(0.05, 1.0)
        raw = {
            "teacher_confidence": (1.0 - torch.exp(-confidence.clamp_min(0.0))).clamp(0.0, 1.0),
            "depth_agreement": torch.exp(-relative_error / 0.20),
            "pixel_closure": closure,
            "visibility_margin": torch.exp(
                -torch.relu(projected - sampled) / (0.10 * projected.abs().clamp_min(epsilon))
            ),
            "non_specular_support": non_specular[:, :, None].expand(shape),
        }
        components = {
            name: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0).detach()
            for name, value in raw.items()
        }
        soft = _geometric_mean(torch.stack(tuple(components.values()), dim=0))
        weight = torch.where(hard_valid, soft, torch.zeros_like(soft)).clamp(0.0, 1.0)
    return PixelReliability(
        weight.detach(),
        components,
        hard_valid.detach(),
        {name: value.detach() for name, value in flags.items()},
    )


def compose_reliability(
    window: WindowReliability, pair: PairReliability, pixel: PixelReliability
) -> torch.Tensor:
    if window.weight.ndim != 1 or pair.weight.ndim != 3 or pixel.weight.ndim != 5:
        raise ValueError("hierarchical reliability dimensions differ")
    if window.weight.shape[0] != pair.weight.shape[0] or pair.weight.shape != pixel.weight.shape[:3]:
        raise ValueError("hierarchical reliability shapes differ")
    with torch.no_grad():
        weight = (
            window.weight[:, None, None, None, None].detach().float()
            * pair.weight[..., None, None].detach().float()
            * pixel.weight.detach().float()
        )
        if not bool(torch.isfinite(weight).all()):
            raise FloatingPointError("hierarchical reliability is non-finite")
    return weight.clamp(0.0, 1.0).detach()


compose_hierarchical_reliability = compose_reliability
