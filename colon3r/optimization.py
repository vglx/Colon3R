from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn

from .lora import lora_named_parameters


GROUP_ORDER = ("lora", "depth", "point", "camera")
GROUP_PEAK_RATES = {"lora": 2e-5, "depth": 2e-7, "point": 1e-7, "camera": 5e-8}


@dataclass(frozen=True)
class ParameterPolicyAudit:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_group_names: Mapping[str, tuple[str, ...]]


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    module = _unwrap(model)
    lora_names = {name for name, _ in lora_named_parameters(module)}
    if not lora_names:
        raise ValueError("model has no injected Q/V/O LoRA parameters")
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "lora": [],
        "depth": [],
        "point": [],
        "camera": [],
        "track": [],
        "base": [],
    }
    for name, parameter in module.named_parameters():
        if name in lora_names:
            group = "lora"
        elif name.startswith("depth_head."):
            group = "depth"
        elif name.startswith("point_head."):
            group = "point"
        elif name.startswith("camera_head."):
            group = "camera"
        elif name.startswith("track_head."):
            group = "track"
        else:
            group = "base"
        groups[group].append((name, parameter))
    missing = [name for name in GROUP_ORDER if not groups[name]]
    if missing:
        raise ValueError(f"model is missing required parameter groups: {missing}")
    return groups


def apply_parameter_policy(model: nn.Module) -> ParameterPolicyAudit:
    groups = parameter_groups(model)
    allowed = {name for group in GROUP_ORDER for name, _ in groups[group]}
    for name, parameter in _unwrap(model).named_parameters():
        parameter.requires_grad_(name in allowed)
    trainable = tuple(name for name, parameter in _unwrap(model).named_parameters() if parameter.requires_grad)
    frozen = tuple(name for name, parameter in _unwrap(model).named_parameters() if not parameter.requires_grad)
    if set(trainable) != allowed:
        raise RuntimeError("trainable parameter inventory differs from the Colon3R policy")
    return ParameterPolicyAudit(
        trainable,
        frozen,
        {group: tuple(name for name, _ in groups[group]) for group in GROUP_ORDER},
    )


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    apply_parameter_policy(model)
    groups = parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "name": name,
                "params": [parameter for _, parameter in groups[name]],
                "lr": GROUP_PEAK_RATES[name],
                "peak_lr": GROUP_PEAK_RATES[name],
            }
            for name in GROUP_ORDER
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
        foreach=False,
    )
    expected = {id(parameter) for name in GROUP_ORDER for _, parameter in groups[name]}
    observed = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if observed != expected:
        raise RuntimeError("optimizer membership differs from the Colon3R policy")
    return optimizer


def target_weight(update: int, maximum: float = 0.5) -> float:
    value = int(update)
    if value != update or value < 0:
        raise ValueError("update must be a nonnegative integer")
    if value <= 100:
        fraction = 0.05 + (0.25 - 0.05) * value / 100
    elif value <= 500:
        fraction = 0.25 + (1.0 - 0.25) * (value - 100) / 400
    else:
        fraction = 1.0
    return float(maximum) * fraction


def learning_rate_scale(update: int, horizon: int = 6000, warmup: int = 50) -> float:
    value = int(update)
    if value != update or value < 0:
        raise ValueError("update must be a nonnegative integer")
    if value == 0:
        return 0.0
    if value <= warmup:
        return value / warmup
    if value >= horizon:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * (value - warmup) / (horizon - warmup)))


def set_optimizer_rates(optimizer: torch.optim.Optimizer, update: int) -> dict[str, float]:
    if tuple(group.get("name") for group in optimizer.param_groups) != GROUP_ORDER:
        raise ValueError("optimizer groups must be lora, depth, point, camera")
    scale = learning_rate_scale(update)
    result = {}
    for group in optimizer.param_groups:
        name = str(group["name"])
        rate = float(group.get("peak_lr", GROUP_PEAK_RATES[name])) * scale
        group["lr"] = rate
        result[name] = rate
    return result


def _direct_statistics(
    groups: Mapping[str, list[tuple[str, nn.Parameter]]],
    gradients: Mapping[str, torch.Tensor | None],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for group, named in groups.items():
        present = [gradients.get(name) for name, _ in named if gradients.get(name) is not None]
        if any(not bool(torch.isfinite(value).all()) for value in present):
            raise FloatingPointError(f"non-finite target gradient in {group}")
        squared = sum(float(value.detach().float().square().sum().cpu()) for value in present)
        result[group] = {"gradient_count": len(present), "l2_norm": squared**0.5}
    return result


def accumulate_target_gradients(model: nn.Module, objective: torch.Tensor) -> dict[str, dict[str, float | int]]:
    if not torch.is_tensor(objective) or objective.numel() != 1 or not bool(torch.isfinite(objective.detach())):
        raise ValueError("target objective must be one finite tensor")
    groups = parameter_groups(model)
    trainable = [(name, parameter) for group in GROUP_ORDER for name, parameter in groups[group]]
    values = torch.autograd.grad(objective, [parameter for _, parameter in trainable], allow_unused=True)
    direct = {name: gradient for (name, _), gradient in zip(trainable, values)}
    statistics = _direct_statistics(groups, direct)
    for forbidden in ("point", "camera"):
        if statistics[forbidden]["gradient_count"]:
            raise RuntimeError(f"target objective produced forbidden {forbidden} gradients")
    for name, parameter in trainable:
        gradient = direct[name]
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.add_(gradient.detach())
    return statistics


def gradient_statistics(model: nn.Module) -> dict[str, dict[str, float | int]]:
    groups = parameter_groups(model)
    result = {}
    for group, named in groups.items():
        gradients = [parameter.grad for _, parameter in named if parameter.grad is not None]
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError(f"non-finite gradient in {group}")
        if group in {"track", "base"} and gradients:
            raise RuntimeError(f"frozen {group} parameters received gradients")
        squared = sum(float(value.detach().float().square().sum().cpu()) for value in gradients)
        result[group] = {"gradient_count": len(gradients), "l2_norm": squared**0.5}
    return result


def all_reduce_gradients(model: nn.Module) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    world_size = dist.get_world_size()
    count = 0
    for name, parameter in _unwrap(model).named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"frozen parameter received a gradient: {name}")
            continue
        present = torch.tensor(int(parameter.grad is not None), device=parameter.device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if int(present.item()) not in {0, world_size}:
            raise RuntimeError(f"rank-local gradient membership differs: {name}")
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite gradient: {name}")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)
            count += 1
    return count
