from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .checkpoint import STAGE1_SHA256, load_stage1_initialization
from .lora import inject_qvo_lora


def _paths(config: Mapping[str, Any]) -> Mapping[str, Any]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise TypeError("config.paths must be a mapping")
    return paths


def _vggt_root(config: Mapping[str, Any], vggt_path: str | Path | None) -> Path:
    configured = (
        vggt_path
        or os.environ.get("VGGT_ROOT")
        or _paths(config).get("vggt_root")
    )
    if not configured:
        raise ValueError("set --vggt-path or VGGT_ROOT to the official VGGT checkout")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"official VGGT root does not exist: {root}; set --vggt-path or VGGT_ROOT"
        )
    expected = root / "vggt" / "models" / "vggt.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"official VGGT module not found at {expected}; set --vggt-path or VGGT_ROOT"
        )
    return root


def _vggt_class(config: Mapping[str, Any], vggt_path: str | Path | None) -> type[nn.Module]:
    root = _vggt_root(config, vggt_path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("vggt.models.vggt")
    except Exception as error:
        raise ImportError(
            f"failed to import official VGGT from {root}; verify --vggt-path or VGGT_ROOT"
        ) from error
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if root not in module_file.parents:
        raise ImportError(
            f"vggt.models.vggt resolved to {module_file}, not configured root "
            f"{root}; restart with --vggt-path or VGGT_ROOT"
        )
    model_class = getattr(module, "VGGT", None)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ImportError(f"official VGGT class is unavailable under {root}")
    return model_class


def _base_model(
    config: Mapping[str, Any],
    vggt_path: str | Path | None,
    device: str | torch.device,
) -> nn.Module:
    model_class = _vggt_class(config, vggt_path)
    loader = getattr(model_class, "from_pretrained", None)
    if not callable(loader):
        raise ImportError("official VGGT must provide VGGT.from_pretrained")
    try:
        model = loader("facebook/VGGT-1B")
    except Exception as error:
        raise RuntimeError(
            "failed to load facebook/VGGT-1B; make the official weights available "
            "to Hugging Face or its local cache"
        ) from error
    if not isinstance(model, nn.Module):
        raise TypeError("VGGT.from_pretrained did not return nn.Module")
    return model.to(device=torch.device(device))


def _initialize_stage1(model: nn.Module, config: Mapping[str, Any]) -> None:
    paths = _paths(config)
    checkpoint = paths.get("stage1_4k_initialization")
    if not checkpoint:
        raise ValueError("config.paths.stage1_4k_initialization is required")
    expected_hash = paths.get("stage1_4k_sha256", STAGE1_SHA256)
    load_stage1_initialization(checkpoint, model, expected_sha256=str(expected_hash))


def build_student(
    config: Mapping[str, Any],
    *,
    vggt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = _base_model(config, vggt_path, device)
    _initialize_stage1(model, config)
    audit = inject_qvo_lora(model, config)
    model.colon3r_lora_audit = audit
    return model


def build_teacher(
    config: Mapping[str, Any],
    *,
    vggt_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = _base_model(config, vggt_path, device)
    _initialize_stage1(model, config)
    model.requires_grad_(False)
    model.eval()
    return model


__all__ = ["build_student", "build_teacher"]
