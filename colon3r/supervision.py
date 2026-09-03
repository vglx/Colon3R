from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _clean(value: torch.Tensor | None, hard_max: float | None = 100) -> torch.Tensor | None:
    if value is None:
        return None
    value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    if hard_max is not None:
        value = value.clamp(min=-hard_max, max=hard_max)
    return value


def _sqrt_positive_part(value: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(value)
    positive = value > 0
    if torch.is_grad_enabled():
        result[positive] = torch.sqrt(value[positive])
        return result
    return torch.where(positive, torch.sqrt(value), result)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in 3x3, got {matrix.shape}")
    batch = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch + (9,)), dim=-1
    )
    magnitudes = _sqrt_positive_part(
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((magnitudes[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, magnitudes[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, magnitudes[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, magnitudes[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    floor_value = torch.tensor(0.1, dtype=magnitudes.dtype, device=magnitudes.device)
    candidates = candidates / (2 * magnitudes[..., None].maximum(floor_value))
    quaternion = candidates[
        F.one_hot(magnitudes.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch + (4,))
    quaternion = quaternion[..., (1, 2, 3, 0)]
    return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def extrinsics_intrinsics_to_pose_encoding(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size_hw
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    quaternion = _matrix_to_quaternion(rotation)
    fov_h = 2 * torch.atan((height / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((width / 2) / intrinsics[..., 0, 0])
    return torch.cat(
        (translation, quaternion, fov_h[..., None], fov_w[..., None]), dim=-1
    ).float()


def _quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    if not 0 <= float(q) <= 1:
        raise ValueError("quantile must be in [0,1]")
    flattened = value.reshape((-1,) + (1,) * (value.ndim - 1))
    index = round(float(q) * (flattened.shape[0] - 1)) + 1
    return torch.kthvalue(flattened, index, 0, keepdim=True)[0].squeeze()


def _filter_quantile(
    value: torch.Tensor,
    valid_range: float,
    min_elements: int = 1000,
    hard_max: float = 100,
) -> torch.Tensor:
    if value.numel() <= min_elements:
        return value
    if value.numel() > 100_000_000:
        indices = torch.randperm(value.numel(), device=value.device)[:1_000_000]
        value = value.reshape(-1)[indices]
    value = value.clamp(max=hard_max)
    threshold = min(_quantile(value.detach(), valid_range), hard_max)
    selected = value < threshold
    return value[selected] if selected.sum() > min_elements else value


def _point_map_to_normal(
    pointmap: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=pointmap.device.type, enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        points = F.pad(
            pointmap.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0
        ).permute(0, 2, 3, 1)
        center = points[:, 1:-1, 1:-1]
        up = points[:, :-2, 1:-1] - center
        left = points[:, 1:-1, :-2] - center
        down = points[:, 2:, 1:-1] - center
        right = points[:, 1:-1, 2:] - center
        normals = torch.stack(
            (
                torch.cross(up, left, dim=-1),
                torch.cross(left, down, dim=-1),
                torch.cross(down, right, dim=-1),
                torch.cross(right, up, dim=-1),
            )
        )
        center_valid = padded_mask[:, 1:-1, 1:-1]
        valid = torch.stack(
            (
                padded_mask[:, :-2, 1:-1] & center_valid & padded_mask[:, 1:-1, :-2],
                padded_mask[:, 1:-1, :-2] & center_valid & padded_mask[:, 2:, 1:-1],
                padded_mask[:, 2:, 1:-1] & center_valid & padded_mask[:, 1:-1, 2:],
                padded_mask[:, 1:-1, 2:] & center_valid & padded_mask[:, :-2, 1:-1],
            )
        )
        return F.normalize(normals, p=2, dim=-1, eps=epsilon), valid


def _normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted, predicted_valid = _point_map_to_normal(prediction, mask, 1e-8)
    expected, expected_valid = _point_map_to_normal(target, mask, 1e-8)
    valid = predicted_valid & expected_valid
    if int(valid.sum()) < 10:
        return prediction.sum() * 0
    loss = 1 - (predicted[valid] * expected[valid]).sum(-1).clamp(-1 + 1e-8, 1 - 1e-8)
    loss = _clean(loss)
    if confidence is not None:
        expanded = confidence[None].expand(4, -1, -1, -1)[valid]
        loss = loss * expanded - 0.2 * torch.log(expanded)
    return loss.mean()


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    expanded_mask = mask[..., None].expand_as(prediction)
    difference = expanded_mask * (prediction - target)
    gradient_x = (difference[:, :, 1:] - difference[:, :, :-1]).abs()
    gradient_y = (difference[:, 1:] - difference[:, :-1]).abs()
    mask_x = expanded_mask[:, :, 1:] * expanded_mask[:, :, :-1]
    mask_y = expanded_mask[:, 1:] * expanded_mask[:, :-1]
    gradient_x = (gradient_x * mask_x).clamp(max=100)
    gradient_y = (gradient_y * mask_y).clamp(max=100)
    if confidence is not None:
        expanded = confidence[..., None].expand_as(prediction)
        confidence_x = expanded[:, :, 1:]
        confidence_y = expanded[:, 1:]
        gradient_x = gradient_x * confidence_x - 0.2 * torch.log(confidence_x)
        gradient_y = gradient_y * confidence_y - 0.2 * torch.log(confidence_y)
    divisor = expanded_mask.sum()
    if int(divisor) == 0:
        return prediction.sum() * 0
    return (gradient_x.sum() + gradient_y.sum()) / divisor


def _multi_scale_gradient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    confidence: torch.Tensor | None,
) -> torch.Tensor:
    scales = 3 if "normal" in kind else 4
    function = _normal_loss if "normal" in kind else _gradient_loss
    values = []
    for scale in range(scales):
        step = 2**scale
        values.append(
            function(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                None if confidence is None else confidence[:, ::step, ::step],
            )
        )
    return sum(values) / scales


def _regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor,
    gradient_kind: str,
    valid_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, views, height, width, channels = prediction.shape
    regular = _clean(torch.linalg.vector_norm(target[mask] - prediction[mask], dim=-1))
    confidence_loss = _clean(regular * confidence[mask] - 0.2 * torch.log(confidence[mask]))
    gradient_confidence = (
        confidence.reshape(batch * views, height, width)
        if "conf" in gradient_kind
        else None
    )
    gradient = _multi_scale_gradient(
        prediction.reshape(batch * views, height, width, channels),
        target.reshape(batch * views, height, width, channels),
        mask.reshape(batch * views, height, width),
        gradient_kind,
        gradient_confidence,
    )
    if confidence_loss.numel():
        if valid_range > 0:
            confidence_loss = _filter_quantile(confidence_loss, valid_range)
        confidence_loss = _clean(confidence_loss).mean()
    else:
        confidence_loss = prediction.sum() * 0
    if regular.numel():
        if valid_range > 0:
            regular = _filter_quantile(regular, valid_range)
        regular = _clean(regular).mean()
    else:
        regular = prediction.sum() * 0
    return confidence_loss, gradient, regular


def _camera_loss(
    prediction: Mapping[str, Any], batch: Mapping[str, torch.Tensor], loss_type: str
) -> dict[str, torch.Tensor]:
    stages = prediction["pose_enc_list"]
    valid_frames = batch["point_masks"][:, 0].sum(dim=(-1, -2)) > 100
    target = extrinsics_intrinsics_to_pose_encoding(
        batch["extrinsics"], batch["intrinsics"], batch["point_masks"].shape[-2:]
    )
    translation = rotation = focal = 0
    count = len(stages)
    for index, stage in enumerate(stages):
        weight = 0.6 ** (count - index - 1)
        if valid_frames.sum() == 0:
            terms = ((stage * 0).mean(),) * 3
        elif loss_type == "l1":
            selected = stage[valid_frames].clone()
            expected = target[valid_frames].clone()
            terms = (
                _clean((selected[..., :3] - expected[..., :3]).abs()).clamp(max=100).mean(),
                _clean((selected[..., 3:7] - expected[..., 3:7]).abs()).mean(),
                _clean((selected[..., 7:] - expected[..., 7:]).abs()).mean(),
            )
        else:
            raise ValueError(f"unsupported camera loss: {loss_type}")
        translation += terms[0] * weight
        rotation += terms[1] * weight
        focal += terms[2] * weight
    translation, rotation, focal = translation / count, rotation / count, focal / count
    return {
        "loss_camera": translation + rotation + 0.5 * focal,
        "loss_T": translation,
        "loss_R": rotation,
        "loss_FL": focal,
    }


def _dense_loss(
    prediction: torch.Tensor,
    confidence: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gradient_kind: str,
    valid_range: float,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if mask.sum() < 100:
        zero = prediction.mean() * 0
        return {
            f"loss_conf_{prefix}": zero,
            f"loss_reg_{prefix}": zero,
            f"loss_grad_{prefix}": zero,
        }
    confidence_loss, gradient, regular = _regression_loss(
        prediction, target, mask, confidence, gradient_kind, valid_range
    )
    return {
        f"loss_conf_{prefix}": confidence_loss,
        f"loss_reg_{prefix}": regular,
        f"loss_grad_{prefix}": gradient,
    }


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    camera: Mapping[str, Any] | None = None
    depth: Mapping[str, Any] | None = None
    point: Mapping[str, Any] | None = None
    track: Mapping[str, Any] | None = None
    ema: Mapping[str, Any] | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        super().__init__()

    def forward(
        self, prediction: Mapping[str, Any], batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        objective: torch.Tensor | int = 0
        if "pose_enc_list" in prediction and self.camera is not None:
            camera = _camera_loss(prediction, batch, str(self.camera.get("loss_type", "l1")))
            terms.update(camera)
            objective = objective + camera["loss_camera"] * float(self.camera["weight"])
        if "depth" in prediction and self.depth is not None:
            depth = _dense_loss(
                prediction["depth"],
                prediction["depth_conf"],
                _clean(batch["depths"])[..., None],
                batch["point_masks"].clone(),
                str(self.depth.get("gradient_loss_fn", "grad")),
                float(self.depth.get("valid_range", -1)),
                "depth",
            )
            terms.update(depth)
            objective = objective + sum(depth.values()) * float(self.depth["weight"])
        if "world_points" in prediction and self.point is not None:
            point = _dense_loss(
                prediction["world_points"],
                prediction["world_points_conf"],
                _clean(batch["world_points"]),
                batch["point_masks"],
                str(self.point.get("gradient_loss_fn", "normal")),
                float(self.point.get("valid_range", -1)),
                "point",
            )
            terms.update(point)
            objective = objective + sum(point.values()) * float(self.point["weight"])
        if not torch.is_tensor(objective):
            raise ValueError("no supervised prediction heads were provided")
        terms["objective"] = objective
        terms["loss_objective"] = objective
        return terms


__all__ = ["MultitaskLoss", "extrinsics_intrinsics_to_pose_encoding"]
