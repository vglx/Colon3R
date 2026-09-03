from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
from torch import nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _options(base: nn.Linear) -> dict[str, object]:
    return {"device": base.weight.device, "dtype": base.weight.dtype}


class FusedQVLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("FusedQVLoRA requires nn.Linear")
        if base.out_features != 3 * base.in_features:
            raise ValueError("fused QKV must have out_features == 3 * in_features")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(0.0)
        hidden = base.out_features // 3
        options = _options(base)
        self.q_a = nn.Linear(base.in_features, rank, bias=False, **options)
        self.q_b = nn.Linear(rank, hidden, bias=False, **options)
        self.v_a = nn.Linear(base.in_features, rank, bias=False, **options)
        self.v_b = nn.Linear(rank, hidden, bias=False, **options)
        nn.init.kaiming_uniform_(self.q_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_b.weight)
        nn.init.zeros_(self.v_b.weight)
        self.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        query, key, projected_value = self.base(value).chunk(3, dim=-1)
        adapted = self.dropout(value)
        query = query + self.q_b(self.q_a(adapted)) * self.scale
        projected_value = projected_value + self.v_b(self.v_a(adapted)) * self.scale
        return torch.cat((query, key, projected_value), dim=-1)


class OutputLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("OutputLoRALinear requires nn.Linear")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(0.0)
        options = _options(base)
        self.a = nn.Linear(base.in_features, rank, bias=False, **options)
        self.b = nn.Linear(rank, base.out_features, bias=False, **options)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.b(self.a(self.dropout(value))) * self.scale


@dataclass(frozen=True)
class LoRAInjectionAudit:
    block_indices: tuple[int, ...]
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    effective_scale: float
    trainable_parameter_count: int
    trainable_numel: int


def _lora_config(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config.get("lora", config)
    if not isinstance(value, Mapping):
        raise TypeError("config.lora must be a mapping")
    required = {"rank", "alpha", "dropout", "global_blocks", "projections"}
    if set(value) != required:
        raise ValueError("LoRA config schema mismatch")
    if value["rank"] != 16:
        raise ValueError("LoRA rank must be 16")
    if value["alpha"] != 16:
        raise ValueError("LoRA alpha must be 16")
    if float(value["dropout"]) != 0.0:
        raise ValueError("LoRA dropout must be 0")
    if list(value["global_blocks"]) != [12, 24]:
        raise ValueError("LoRA global blocks must be [12, 24]")
    if list(value["projections"]) != ["q", "v", "o"]:
        raise ValueError("LoRA projections must be [q, v, o]")
    return value


def inject_qvo_lora(model: nn.Module, config: Mapping[str, object]) -> LoRAInjectionAudit:
    values = _lora_config(config)
    module = _unwrap(model)
    aggregator = getattr(module, "aggregator", None)
    blocks = getattr(aggregator, "global_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("model must expose aggregator.global_blocks as ModuleList")
    if len(blocks) != 24:
        raise ValueError(f"expected 24 global blocks, found {len(blocks)}")
    indices = tuple(range(12, 24))
    for index in indices:
        attention = getattr(blocks[index], "attn", None)
        if not isinstance(getattr(attention, "qkv", None), nn.Linear):
            raise TypeError(f"global block {index} qkv is not nn.Linear")
        if not isinstance(getattr(attention, "proj", None), nn.Linear):
            raise TypeError(f"global block {index} proj is not nn.Linear")
    module.requires_grad_(False)
    targets = []
    for index in indices:
        attention = blocks[index].attn
        attention.qkv = FusedQVLoRA(
            attention.qkv, int(values["rank"]), int(values["alpha"])
        )
        attention.proj = OutputLoRALinear(
            attention.proj, int(values["rank"]), int(values["alpha"])
        )
        prefix = f"aggregator.global_blocks.{index}.attn"
        targets.extend((f"{prefix}.qkv", f"{prefix}.proj"))
    trainable = tuple(lora_named_parameters(module))
    return LoRAInjectionAudit(
        block_indices=indices,
        target_modules=tuple(targets),
        rank=16,
        alpha=16,
        effective_scale=1.0,
        trainable_parameter_count=len(trainable),
        trainable_numel=sum(parameter.numel() for _, parameter in trainable),
    )


def lora_named_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    modules = dict(_unwrap(model).named_modules())
    for name, parameter in _unwrap(model).named_parameters():
        if any(
            token in name
            for token in (
                ".q_a.weight",
                ".q_b.weight",
                ".v_a.weight",
                ".v_b.weight",
            )
        ):
            yield name, parameter
        elif name.endswith((".a.weight", ".b.weight")):
            parent = name.rsplit(".", 2)[0]
            if isinstance(modules.get(parent), OutputLoRALinear):
                yield name, parameter


__all__ = [
    "FusedQVLoRA",
    "LoRAInjectionAudit",
    "OutputLoRALinear",
    "inject_qvo_lora",
    "lora_named_parameters",
]
