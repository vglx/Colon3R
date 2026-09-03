from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .lora import lora_named_parameters


STAGE1_STEP = 4000
STAGE1_SHA256 = "3a6d61a3132763d1bb566cd24dafe371470b7e8e3b81d9df7cc617df1aa0e739"
COLON3R_FINAL_SHA256 = "7407b0681bacc6e7d7fb0c130206da0c68daa9f4306ad6017ad9a37fa9ac1cc5"
_HEAD_PREFIXES = ("camera_head.", "depth_head.", "point_head.")
_TRAINING_KIND = "colon3r_training_checkpoint"
_FORMAL_KIND = "vggt_hybrid_quasirigid_checkpoint"
_TRAINING_KEYS = {
    "artifact_version",
    "kind",
    "global_step",
    "adaptation_update",
    "student_lora_state",
    "student_head_state",
    "teacher_state",
    "optimizer_state",
    "digests",
}
_FORMAL_KEYS = {
    "artifact_version",
    "kind",
    "identity",
    "identity_digest",
    "global_step",
    "adaptation_update",
    "optimizer_step_calls",
    "student_lora_state",
    "student_head_state",
    "teacher_controller_state",
    "optimizer_state",
    "optimizer_state_on_cpu",
    "sampler_ordinals",
    "reliability_monitor_state",
    "empty_support_monitor_state",
    "rng_state",
    "configuration_snapshot",
    "scheduler_state",
    "gradient_scaler_state",
    "ema_counter",
    "trainable_parameter_manifest",
    "student_digest",
    "teacher_digest",
    "digests",
}


def _tensor_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous()
    try:
        return contiguous.numpy().tobytes(order="C")
    except TypeError:
        return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not torch.is_tensor(value):
            raise TypeError(f"checkpoint state value is not a tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def model_digest(model: nn.Module) -> str:
    return tensor_digest(_unwrap(model).state_dict())


def _parameter_group_digests(
    model: nn.Module,
    overrides: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, str]:
    override_values = {} if overrides is None else dict(overrides)
    lora_names = {name for name, _parameter in lora_named_parameters(model)}
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {
        "lora": [],
        "depth": [],
        "point": [],
        "camera": [],
        "track": [],
        "base": [],
    }
    for name, parameter in _unwrap(model).named_parameters():
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
        groups[group].append((name, override_values.get(name, parameter)))
    result = {}
    for group, values in groups.items():
        digest = hashlib.sha256()
        for name, value in values:
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(_tensor_bytes(value))
        result[group] = digest.hexdigest()
    return result


def parameter_group_digests(model: nn.Module) -> dict[str, str]:
    return _parameter_group_digests(model)


def value_digest(value: Any, json_compatible: bool = False) -> str:
    if json_compatible:
        encoded = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    digest = hashlib.sha256()

    def update(current: Any) -> None:
        if torch.is_tensor(current):
            digest.update(b"tensor")
            digest.update(str(current.dtype).encode("ascii"))
            digest.update(str(tuple(current.shape)).encode("ascii"))
            digest.update(_tensor_bytes(current))
        elif isinstance(current, np.ndarray):
            digest.update(b"ndarray")
            digest.update(str(current.dtype).encode("ascii"))
            digest.update(str(tuple(current.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(current).tobytes(order="C"))
        elif isinstance(current, Mapping):
            digest.update(b"mapping")
            for key in sorted(current, key=repr):
                update(key)
                update(current[key])
        elif isinstance(current, (list, tuple)):
            digest.update(type(current).__name__.encode("ascii"))
            for item in current:
                update(item)
        else:
            digest.update(type(current).__name__.encode("ascii"))
            digest.update(repr(current).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _verify_hash(path: Path, expected_sha256: str | None) -> str:
    observed = file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _head_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in _unwrap(model).state_dict().items()
        if name.startswith(_HEAD_PREFIXES)
    }


def _require_complete_heads(state: Mapping[str, torch.Tensor], label: str) -> None:
    missing = [
        prefix.removesuffix(".")
        for prefix in _HEAD_PREFIXES
        if not any(name.startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(f"{label} is missing required heads: {', '.join(missing)}")


def _validate_tensor_state(
    state: Any, expected: Mapping[str, torch.Tensor], label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError(f"checkpoint {label} inventory mismatch")
    result = dict(state)
    for name, value in result.items():
        destination = expected[name]
        if not torch.is_tensor(value):
            raise ValueError(f"checkpoint {label} value is not a tensor: {name}")
        if value.shape != destination.shape or value.dtype != destination.dtype:
            raise ValueError(f"checkpoint {label} shape or dtype mismatch: {name}")
    return result


@torch.no_grad()
def _apply_tensor_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(_unwrap(model).named_parameters())
    buffers = dict(_unwrap(model).named_buffers())
    for name, value in state.items():
        destination = parameters.get(name)
        if destination is None:
            destination = buffers.get(name)
        if destination is None:
            raise RuntimeError(f"checkpoint destination disappeared: {name}")
        destination.copy_(value.to(device=destination.device, dtype=destination.dtype))


def load_stage1_initialization(
    path: str | Path,
    model: nn.Module,
    expected_sha256: str = STAGE1_SHA256,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 initialization checkpoint not found: {source}")
    observed_hash = _verify_hash(source, expected_sha256)
    payload = _torch_load(source)
    expected_keys = {"step", "head_state_dict", "formal", "optimizer"}
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("Stage-1 checkpoint schema mismatch")
    if type(payload["step"]) is not int or payload["step"] != STAGE1_STEP:
        raise ValueError("Stage-1 checkpoint step must be 4000")
    if payload["formal"] is not True:
        raise ValueError("Stage-1 checkpoint is not formal")
    if not isinstance(payload["optimizer"], Mapping) or set(payload["optimizer"]) != {
        "state",
        "param_groups",
    }:
        raise ValueError("Stage-1 optimizer schema mismatch")
    state = _validate_tensor_state(
        payload["head_state_dict"], _head_state(model), "Stage-1 head state"
    )
    _apply_tensor_state(model, state)
    return {"path": str(source), "sha256": observed_hash, "step": STAGE1_STEP}


def _validate_steps(global_step: Any, adaptation_update: Any) -> tuple[int, int]:
    if type(global_step) is not int or type(adaptation_update) is not int:
        raise ValueError("checkpoint steps must be exact integers")
    if not 0 <= adaptation_update <= 6000:
        raise ValueError("checkpoint adaptation update is out of range")
    if global_step != STAGE1_STEP + adaptation_update:
        raise ValueError("checkpoint global step does not match adaptation update")
    return global_step, adaptation_update


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_checkpoint(
    path: str | Path,
    student: nn.Module,
    teacher: Any,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    adaptation_update: int,
) -> dict[str, Any]:
    step, update = _validate_steps(global_step, adaptation_update)
    lora_state = _clone_cpu(dict(lora_named_parameters(student)))
    head_state = _clone_cpu(_head_state(student))
    if not lora_state:
        raise ValueError("training checkpoint requires non-empty LoRA state")
    _require_complete_heads(head_state, "training checkpoint head state")
    if not callable(getattr(teacher, "state_dict", None)):
        raise TypeError("teacher must expose state_dict()")
    teacher_state = _clone_cpu(teacher.state_dict())
    optimizer_state = _clone_cpu(optimizer.state_dict())
    payload = {
        "artifact_version": 1,
        "kind": _TRAINING_KIND,
        "global_step": step,
        "adaptation_update": update,
        "student_lora_state": lora_state,
        "student_head_state": head_state,
        "teacher_state": teacher_state,
        "optimizer_state": optimizer_state,
        "digests": {
            "lora": tensor_digest(lora_state),
            "heads": tensor_digest(head_state),
            "teacher": value_digest(teacher_state),
            "optimizer": value_digest(optimizer_state),
        },
    }
    destination = Path(path)
    _atomic_save(payload, destination)
    resolved = destination.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "global_step": step,
        "adaptation_update": update,
    }


def _validate_training_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TRAINING_KEYS:
        raise ValueError("training checkpoint schema mismatch")
    result = dict(payload)
    if result["artifact_version"] != 1 or result["kind"] != _TRAINING_KIND:
        raise ValueError("training checkpoint kind or version mismatch")
    _validate_steps(result["global_step"], result["adaptation_update"])
    digests = result["digests"]
    if not isinstance(digests, Mapping) or set(digests) != {
        "lora",
        "heads",
        "teacher",
        "optimizer",
    }:
        raise ValueError("training checkpoint digest schema mismatch")
    if tensor_digest(result["student_lora_state"]) != digests["lora"]:
        raise ValueError("checkpoint LoRA digest mismatch")
    if tensor_digest(result["student_head_state"]) != digests["heads"]:
        raise ValueError("checkpoint head digest mismatch")
    if value_digest(result["teacher_state"]) != digests["teacher"]:
        raise ValueError("checkpoint Teacher digest mismatch")
    if value_digest(result["optimizer_state"]) != digests["optimizer"]:
        raise ValueError("checkpoint optimizer digest mismatch")
    return result


def _validate_formal_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _FORMAL_KEYS:
        raise ValueError("formal Colon3R checkpoint schema mismatch")
    result = dict(payload)
    if result["artifact_version"] != 2 or result["kind"] != _FORMAL_KIND:
        raise ValueError("formal Colon3R checkpoint kind or version mismatch")
    step, update = _validate_steps(result["global_step"], result["adaptation_update"])
    if step != 10000 or update != 6000:
        raise ValueError("formal Colon3R checkpoint must be global step 10000")
    if result["optimizer_step_calls"] != update or result["ema_counter"] != update:
        raise ValueError("formal Colon3R checkpoint counters mismatch")
    identity = result["identity"]
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("formal Colon3R checkpoint identity is missing")
    if value_digest(identity, json_compatible=True) != result["identity_digest"]:
        raise ValueError("formal Colon3R checkpoint identity digest mismatch")
    digests = result["digests"]
    required_digests = {
        "lora",
        "heads",
        "teacher_controller",
        "optimizer",
        "sampler_ordinals",
        "reliability",
        "empty_support",
        "rng",
        "parameter_groups",
        "resume_contract",
    }
    if not isinstance(digests, Mapping) or set(digests) != required_digests:
        raise ValueError("formal Colon3R checkpoint digest schema mismatch")
    checks = {
        "lora": tensor_digest(result["student_lora_state"]),
        "heads": tensor_digest(result["student_head_state"]),
        "teacher_controller": value_digest(result["teacher_controller_state"]),
        "optimizer": value_digest(result["optimizer_state"]),
        "sampler_ordinals": value_digest(result["sampler_ordinals"]),
        "reliability": value_digest(result["reliability_monitor_state"]),
        "empty_support": value_digest(result["empty_support_monitor_state"]),
        "rng": value_digest(result["rng_state"]),
    }
    resume = {
        "configuration_snapshot": result["configuration_snapshot"],
        "scheduler_state": result["scheduler_state"],
        "gradient_scaler_state": result["gradient_scaler_state"],
        "ema_counter": result["ema_counter"],
        "trainable_parameter_manifest": result["trainable_parameter_manifest"],
    }
    checks["resume_contract"] = value_digest(resume)
    for name, observed in checks.items():
        if digests[name] != observed:
            raise ValueError(f"formal Colon3R checkpoint {name} digest mismatch")
    controller = result["teacher_controller_state"]
    if not isinstance(controller, Mapping) or "teacher_state" not in controller:
        raise ValueError("formal Colon3R Teacher state is missing")
    return result


def load_colon3r_checkpoint(
    path: str | Path,
    student: nn.Module,
    *,
    teacher: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Colon3R checkpoint not found: {source}")
    observed_hash = _verify_hash(
        source,
        COLON3R_FINAL_SHA256 if expected_sha256 is None else expected_sha256,
    )
    payload = _torch_load(source)
    kind = payload.get("kind") if isinstance(payload, Mapping) else None
    if kind == _TRAINING_KIND:
        result = _validate_training_payload(payload)
        teacher_state = result["teacher_state"]
    elif kind == _FORMAL_KIND:
        result = _validate_formal_payload(payload)
        teacher_state = result["teacher_controller_state"]["teacher_state"]
    else:
        raise ValueError("unsupported Colon3R checkpoint schema")
    lora_state = _validate_tensor_state(
        result["student_lora_state"], dict(lora_named_parameters(student)), "LoRA state"
    )
    head_state = _validate_tensor_state(
        result["student_head_state"], _head_state(student), "head state"
    )
    candidate_student_state = dict(_unwrap(student).state_dict())
    candidate_student_state.update(lora_state)
    candidate_student_state.update(head_state)
    overrides = {**lora_state, **head_state}
    if kind == _FORMAL_KIND:
        if tensor_digest(candidate_student_state) != result["student_digest"]:
            raise ValueError("formal Colon3R checkpoint student digest mismatch")
        if tensor_digest(teacher_state) != result["teacher_digest"]:
            raise ValueError("formal Colon3R checkpoint Teacher digest mismatch")
        if _parameter_group_digests(student, overrides) != result["digests"][
            "parameter_groups"
        ]:
            raise ValueError(
                "formal Colon3R checkpoint parameter-group digest mismatch"
            )
    teacher_is_controller = teacher is not None and hasattr(teacher, "teacher")
    if teacher is not None:
        if teacher_is_controller:
            candidate = copy.deepcopy(teacher)
            candidate.load_state_dict(
                result["teacher_controller_state"]
                if kind == _FORMAL_KIND
                else teacher_state
            )
        else:
            _validate_tensor_state(teacher_state, teacher.state_dict(), "Teacher state")
    if optimizer is not None:
        if not isinstance(result["optimizer_state"], Mapping):
            raise ValueError("checkpoint optimizer state is missing")
        candidate_optimizer = copy.deepcopy(optimizer)
        try:
            candidate_optimizer.load_state_dict(result["optimizer_state"])
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "checkpoint optimizer parameter groups are incompatible"
            ) from error
    _apply_tensor_state(student, lora_state)
    _apply_tensor_state(student, head_state)
    if teacher is not None:
        if teacher_is_controller:
            teacher.load_state_dict(
                result["teacher_controller_state"]
                if kind == _FORMAL_KIND
                else teacher_state
            )
        else:
            teacher.load_state_dict(teacher_state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(result["optimizer_state"])
    return {
        "path": str(source),
        "sha256": observed_hash,
        "kind": kind,
        "global_step": result["global_step"],
        "adaptation_update": result["adaptation_update"],
    }


__all__ = [
    "COLON3R_FINAL_SHA256",
    "STAGE1_SHA256",
    "file_sha256",
    "load_colon3r_checkpoint",
    "load_stage1_initialization",
    "model_digest",
    "parameter_group_digests",
    "save_training_checkpoint",
    "tensor_digest",
    "value_digest",
]
