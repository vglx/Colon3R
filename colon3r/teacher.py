from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping

import torch
from torch import nn

from .lora import FusedQVLoRA, OutputLoRALinear


TeacherPhase = Literal["FIXED", "HARD_COPY", "EMA"]
_HEAD_PREFIXES = ("camera_head.", "depth_head.", "point_head.")


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes(order="C")
    except TypeError:
        return tensor.view(torch.uint8).numpy().tobytes(order="C")


def _digest(values: Mapping[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        result.update(name.encode("utf-8"))
        result.update(str(tuple(value.shape)).encode("ascii"))
        result.update(str(value.dtype).encode("ascii"))
        result.update(_bytes(value))
    return result.hexdigest()


def _clone(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in values.items()
    }


def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    result = dict(model.named_parameters())
    for name, value in model.named_buffers():
        if name in result:
            raise ValueError(f"duplicate tensor name: {name}")
        result[name] = value
    return result


def _effective_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for name, module in _unwrap(model).named_modules():
        if isinstance(module, FusedQVLoRA):
            hidden = module.base.out_features // 3
            delta = torch.zeros_like(module.base.weight)
            delta[:hidden] = module.q_b.weight @ module.q_a.weight * module.scale
            delta[2 * hidden :] = module.v_b.weight @ module.v_a.weight * module.scale
            result[name] = module.base.weight.detach() + delta
        elif isinstance(module, OutputLoRALinear):
            delta = module.b.weight @ module.a.weight * module.scale
            result[name] = module.base.weight.detach() + delta
    return result


def _copy_names(
    names: tuple[str, ...],
    source: Mapping[str, torch.Tensor],
    destination: Mapping[str, torch.Tensor],
) -> None:
    for name in names:
        if name not in source or name not in destination:
            raise ValueError(f"tensor inventory mismatch: {name}")
        if source[name].shape != destination[name].shape:
            raise ValueError(f"tensor shape mismatch: {name}")
        destination[name].copy_(
            source[name].detach().to(destination[name].device, destination[name].dtype)
        )


def _phase(update: int, boundary: int) -> TeacherPhase:
    if update < 0:
        raise ValueError("teacher update cannot be negative")
    if update < boundary:
        return "FIXED"
    if update == boundary:
        return "HARD_COPY"
    return "EMA"


class GatedTeacherController:
    def __init__(
        self,
        student: nn.Module,
        plain_teacher: nn.Module,
        *,
        hard_copy_update: int = 500,
        momentum: float = 0.996,
        support_threshold: float = 0.20138970316367508,
        scale_threshold: float = 0.004409444387420081,
    ) -> None:
        if hard_copy_update != 500 or float(momentum) != 0.996:
            raise ValueError("Teacher schedule must use hard-copy 500 and EMA 0.996")
        self.teacher = _unwrap(plain_teacher)
        if any(
            isinstance(module, (FusedQVLoRA, OutputLoRALinear))
            for module in self.teacher.modules()
        ):
            raise ValueError("Teacher must be a plain VGGT model")
        self.hard_copy_update = hard_copy_update
        self.momentum = float(momentum)
        self.support_threshold = float(support_threshold)
        self.scale_threshold = float(scale_threshold)
        student_module = _unwrap(student)
        student_modules = dict(student_module.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        effective = _effective_weights(student_module)
        if not effective:
            raise ValueError("Student has no Q/V/O LoRA projections")
        self.projection_names = tuple(sorted(effective))
        with torch.no_grad():
            for name in self.projection_names:
                source = student_modules[name]
                destination = teacher_modules.get(name)
                if not isinstance(source, (FusedQVLoRA, OutputLoRALinear)):
                    raise TypeError(f"invalid Student projection: {name}")
                if not isinstance(destination, nn.Linear):
                    raise TypeError(f"invalid Teacher projection: {name}")
                destination.weight.copy_(
                    effective[name].to(destination.weight.device, destination.weight.dtype)
                )
                if destination.bias is not None and source.base.bias is not None:
                    destination.bias.copy_(
                        source.base.bias.detach().to(
                            destination.bias.device, destination.bias.dtype
                        )
                    )
        student_parameters = dict(student_module.named_parameters())
        student_buffers = dict(student_module.named_buffers())
        teacher_parameters = dict(self.teacher.named_parameters())
        teacher_buffers = dict(self.teacher.named_buffers())
        self.head_parameter_names = tuple(
            name
            for name, parameter in teacher_parameters.items()
            if name.startswith(_HEAD_PREFIXES) and parameter.is_floating_point()
        )
        self.head_buffer_names = tuple(
            name for name in teacher_buffers if name.startswith(_HEAD_PREFIXES)
        )
        if not self.head_parameter_names:
            raise ValueError("Teacher has no camera, depth, or point head parameters")
        with torch.no_grad():
            _copy_names(
                self.head_parameter_names, student_parameters, teacher_parameters
            )
            _copy_names(self.head_buffer_names, student_buffers, teacher_buffers)
        mutable = {
            *(f"{name}.weight" for name in self.projection_names),
            *self.head_parameter_names,
            *self.head_buffer_names,
        }
        self.frozen_values = _clone(
            {
                name: value
                for name, value in _named_tensors(self.teacher).items()
                if name not in mutable
            }
        )
        self._frozen_digest = _digest(self.frozen_values)
        self.last_successful_update = 0
        self.hard_copy_count = 0
        self.phase: TeacherPhase = "FIXED"
        self.accepted_ema_count = 0
        self.skipped_ema_count = 0
        self.teacher.eval().requires_grad_(False)

    @property
    def resident_device(self) -> torch.device:
        return next(self.teacher.parameters()).device

    @property
    def resident_dtype(self) -> torch.dtype:
        return next(
            value.dtype
            for value in self.teacher.parameters()
            if value.is_floating_point()
        )

    @torch.no_grad()
    def move_to(
        self, device: torch.device | str, dtype: torch.dtype
    ) -> GatedTeacherController:
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("Teacher dtype must be float32 or bfloat16")
        destination = torch.device(device)
        self.teacher.to(destination, dtype=dtype)
        self.frozen_values = {
            name: value.to(
                destination,
                dtype=dtype if value.is_floating_point() else value.dtype,
            ).contiguous()
            for name, value in self.frozen_values.items()
        }
        self._frozen_digest = _digest(self.frozen_values)
        self.teacher.eval().requires_grad_(False)
        return self

    def _student_effective(self, student: nn.Module) -> dict[str, torch.Tensor]:
        values = _effective_weights(student)
        if tuple(sorted(values)) != self.projection_names:
            raise ValueError("Student LoRA projection inventory changed")
        return values

    @torch.no_grad()
    def _synchronize(self, student: nn.Module, ema: bool) -> None:
        source_model = _unwrap(student)
        source_parameters = dict(source_model.named_parameters())
        source_buffers = dict(source_model.named_buffers())
        destination_parameters = dict(self.teacher.named_parameters())
        destination_buffers = dict(self.teacher.named_buffers())
        modules = dict(self.teacher.named_modules())
        for name, value in self._student_effective(student).items():
            destination = modules[name].weight
            source = value.to(destination.device, destination.dtype)
            if ema:
                destination.mul_(self.momentum).add_(source, alpha=1 - self.momentum)
            else:
                destination.copy_(source)
        for name in self.head_parameter_names:
            destination = destination_parameters[name]
            source = source_parameters[name].detach().to(
                destination.device, destination.dtype
            )
            if ema:
                destination.mul_(self.momentum).add_(source, alpha=1 - self.momentum)
            else:
                destination.copy_(source)
        for name in self.head_buffer_names:
            destination = destination_buffers[name]
            source = source_buffers[name].detach().to(
                destination.device, destination.dtype
            )
            if ema and destination.is_floating_point():
                destination.mul_(self.momentum).add_(source, alpha=1 - self.momentum)
            else:
                destination.copy_(source)
        _copy_names(
            tuple(self.frozen_values), self.frozen_values, _named_tensors(self.teacher)
        )
        self.teacher.eval().requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        student: nn.Module,
        successful_update: int,
        *,
        support: float,
        scale_gap: float,
    ) -> dict[str, Any]:
        update = int(successful_update)
        if update != successful_update or update <= self.last_successful_update:
            raise ValueError("Teacher update must increase monotonically")
        if not torch.isfinite(torch.tensor([support, scale_gap])).all():
            raise ValueError("Teacher gate statistics must be finite")
        support_pass = float(support) >= self.support_threshold
        scale_pass = abs(float(scale_gap)) <= self.scale_threshold
        if update < self.hard_copy_update:
            action = "FIXED"
        elif update == self.hard_copy_update:
            self._synchronize(student, ema=False)
            self.hard_copy_count += 1
            action = "HARD_COPY"
        elif support_pass and scale_pass:
            self._synchronize(student, ema=True)
            self.accepted_ema_count += 1
            action = "EMA"
        else:
            self.skipped_ema_count += 1
            action = "EMA_SKIPPED"
        self.last_successful_update = update
        self.phase = _phase(update, self.hard_copy_update)
        return {
            "action": action,
            "update": update,
            "support": float(support),
            "scale_gap": float(scale_gap),
            "support_pass": support_pass,
            "scale_pass": scale_pass,
        }

    def state_dict(self) -> dict[str, Any]:
        state = _clone(self.teacher.state_dict())
        buffers = _clone(dict(self.teacher.named_buffers()))
        return {
            "artifact_version": 1,
            "hard_copy_update": self.hard_copy_update,
            "momentum": self.momentum,
            "projection_names": self.projection_names,
            "head_parameter_names": self.head_parameter_names,
            "head_buffer_names": self.head_buffer_names,
            "phase": self.phase,
            "last_successful_update": self.last_successful_update,
            "hard_copy_count": self.hard_copy_count,
            "teacher_state": state,
            "teacher_buffers": buffers,
            "digests": {
                "teacher": _digest(_named_tensors(self.teacher)),
                "frozen": self._frozen_digest,
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("artifact_version", -1)) != 1:
            raise ValueError("unsupported Teacher state version")
        if int(state.get("hard_copy_update", -1)) != self.hard_copy_update:
            raise ValueError("Teacher hard-copy update mismatch")
        if float(state.get("momentum", -1)) != self.momentum:
            raise ValueError("Teacher momentum mismatch")
        for key, expected in (
            ("projection_names", self.projection_names),
            ("head_parameter_names", self.head_parameter_names),
            ("head_buffer_names", self.head_buffer_names),
        ):
            if tuple(state.get(key, ())) != expected:
                raise ValueError(f"Teacher {key} mismatch")
        update = int(state.get("last_successful_update", -1))
        hard_copy_count = int(state.get("hard_copy_count", -1))
        if str(state.get("phase", "")) != _phase(update, self.hard_copy_update):
            raise ValueError("Teacher phase mismatch")
        if hard_copy_count != int(update >= self.hard_copy_update):
            raise ValueError("Teacher hard-copy count mismatch")
        teacher_state = state.get("teacher_state")
        if not isinstance(teacher_state, Mapping):
            raise ValueError("Teacher tensor state is missing")
        if set(teacher_state) != set(self.teacher.state_dict()):
            raise ValueError("Teacher tensor inventory mismatch")
        candidate = {name: value.detach().clone() for name, value in self.teacher.state_dict().items()}
        self.teacher.load_state_dict(dict(teacher_state), strict=True)
        digests = state.get("digests")
        try:
            if not isinstance(digests, Mapping):
                raise ValueError("Teacher digests are missing")
            if str(digests.get("frozen", "")) != self._frozen_digest:
                raise ValueError("Teacher frozen digest mismatch")
            if str(digests.get("teacher", "")) != _digest(_named_tensors(self.teacher)):
                raise ValueError("Teacher tensor digest mismatch")
            _copy_names(
                tuple(self.frozen_values), self.frozen_values, _named_tensors(self.teacher)
            )
        except Exception:
            self.teacher.load_state_dict(candidate, strict=True)
            raise
        self.last_successful_update = update
        self.hard_copy_count = hard_copy_count
        self.phase = _phase(update, self.hard_copy_update)
        self.teacher.eval().requires_grad_(False)


__all__ = ["GatedTeacherController"]
