from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AnchorLossResult:
    loss: torch.Tensor
    student_log_depth: torch.Tensor
    per_view_gap: torch.Tensor
    window_gap: float


def quasi_rigid_cross_view_loss(
    student_depth: torch.Tensor,
    teacher_projected_depth: torch.Tensor,
    reliability: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if student_depth.shape != teacher_projected_depth.shape or student_depth.shape != reliability.shape:
        raise ValueError("student depth, projected depth, and reliability must share shape")
    residual = (student_depth.float() - teacher_projected_depth.detach().float()).abs()
    weight = reliability.detach().to(device=residual.device, dtype=residual.dtype)
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("reliability must be finite and nonnegative")
    finite = torch.isfinite(residual)
    if bool((~finite & (weight > 0)).any()):
        raise FloatingPointError("cross-view residual is non-finite on retained support")
    residual = torch.where(finite, residual, torch.zeros_like(residual))
    denominator = weight.sum()
    if float(denominator.detach().cpu()) <= float(epsilon):
        return (residual * 0.0).sum()
    return (residual * weight).sum() / denominator.clamp_min(float(epsilon))


def target_quasi_rigid_loss(
    *,
    student_depth: torch.Tensor,
    teacher_projected_depth: torch.Tensor,
    reliability: torch.Tensor,
    student_camera: torch.Tensor | None = None,
    student_pointmap: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if student_camera is not None:
        student_camera.detach()
    if student_pointmap is not None:
        student_pointmap.detach()
    return quasi_rigid_cross_view_loss(
        student_depth, teacher_projected_depth, reliability, epsilon=epsilon
    )


def _view_first(tensor: torch.Tensor, view_count: int, label: str) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"{label} must include view and spatial dimensions")
    if tensor.ndim >= 5 and tensor.shape[0] == 1 and tensor.shape[1] == view_count:
        return tensor[0]
    if tensor.shape[0] == view_count:
        return tensor
    if tensor.shape[0] == 1 and tensor.shape[1] == view_count:
        return tensor[0]
    raise ValueError(f"{label} view dimension does not match reference")


def depth_scale_anchor_loss(
    student_depth: torch.Tensor,
    fixed_mask: torch.Tensor,
    reference_log_depth: torch.Tensor,
    *,
    epsilon: float = 0.001,
) -> AnchorLossResult:
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    reference = reference_log_depth.detach().to(device=student_depth.device, dtype=torch.float32).reshape(-1)
    if reference.numel() == 0 or not bool(torch.isfinite(reference).all()):
        raise ValueError("anchor loss requires a finite reference per view")
    depth = _view_first(student_depth, reference.numel(), "student_depth")
    mask = _view_first(fixed_mask, reference.numel(), "fixed_mask")
    if depth.shape != mask.shape:
        try:
            mask = torch.broadcast_to(mask, depth.shape)
        except RuntimeError as error:
            raise ValueError("fixed_mask shape is incompatible with student_depth") from error
    if mask.dtype != torch.bool:
        raise ValueError("fixed_mask must be boolean")
    flat_depth = depth.float().reshape(reference.numel(), -1)
    flat_mask = mask.detach().reshape(reference.numel(), -1)
    counts = flat_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("anchor loss received an empty fixed mask")
    selected = torch.isfinite(flat_depth) & (flat_depth > 0.0)
    if not bool((selected | ~flat_mask).all()):
        raise ValueError("anchor loss requires positive finite depth on the fixed mask")
    log_depth = torch.log(flat_depth.clamp_min(torch.finfo(torch.float32).tiny))
    locations = (log_depth * flat_mask.float()).sum(dim=1) / counts.float()
    signed_gap = locations - reference
    per_view_gap = signed_gap.detach().abs()
    loss = (torch.sqrt(signed_gap.square() + float(epsilon) ** 2) - float(epsilon)).mean()
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("anchor loss is non-finite")
    return AnchorLossResult(
        loss,
        locations.detach(),
        per_view_gap,
        float(per_view_gap.mean().cpu()),
    )


def source_multitask_loss(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], criterion: Any
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    terms = criterion(prediction, batch)
    if not isinstance(terms, Mapping) or "objective" not in terms:
        raise ValueError("source multitask criterion must return an objective")
    objective = terms["objective"]
    if not torch.is_tensor(objective) or objective.numel() != 1 or not bool(torch.isfinite(objective.detach())):
        raise ValueError("source multitask objective must be one finite tensor")
    return objective, terms
