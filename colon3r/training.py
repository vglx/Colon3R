from __future__ import annotations

import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn

from .checkpoint import (
    STAGE1_STEP,
    load_colon3r_checkpoint,
    parameter_group_digests,
    save_training_checkpoint,
)
from .data import SourceManifestDataset, TargetManifestDataset
from .geometry import load_wat3r_bindings, predicted_pinhole_cross_view_geometry
from .losses import (
    depth_scale_anchor_loss,
    source_multitask_loss,
    target_quasi_rigid_loss,
)
from .model import build_student, build_teacher
from .optimization import (
    accumulate_target_gradients,
    all_reduce_gradients,
    build_optimizer,
    gradient_statistics,
    set_optimizer_rates,
    target_weight,
)
from .reliability import (
    compose_reliability,
    pair_reliability,
    pixel_reliability,
    window_reliability,
)
from .teacher import GatedTeacherController


@dataclass(frozen=True)
class TargetBranchResult:
    objective: torch.Tensor
    support: float
    gate_probe: Callable[[], float]
    metrics: Mapping[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    if set(value) != {"paths", "lora", "training", "teacher"}:
        raise ValueError("configuration schema mismatch")
    training = value["training"]
    teacher = value["teacher"]
    expected_training = {
        "source_microbatches": 3,
        "target_microbatches": 1,
        "teacher_views": 8,
        "student_views": 8,
        "directed_pairs": 56,
        "adaptation_updates": 6000,
        "precision": "bf16",
        "target_maximum": 0.5,
    }
    expected_teacher = {
        "hard_copy_update": 500,
        "ema_momentum": 0.996,
        "anchor_weight": 0.05,
        "support_threshold": 0.20138970316367508,
        "scale_threshold": 0.004409444387420081,
    }
    if training != expected_training or teacher != expected_teacher:
        raise ValueError("scientific training protocol differs from Colon3R")
    root = source.parent.parent
    resolved = dict(value)
    resolved_paths = {}
    for name, item in value["paths"].items():
        candidate = Path(str(item)).expanduser()
        resolved_paths[name] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    resolved["paths"] = resolved_paths
    return resolved


def _initialize_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("launch training with torchrun --nproc_per_node=2")
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2 or dist.get_backend() != "nccl":
        raise RuntimeError("Colon3R training requires exactly two NCCL workers")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _depth_output(model: nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    module = model.module if hasattr(model, "module") else model
    aggregated = module.aggregator(images)
    if isinstance(aggregated, tuple) and len(aggregated) == 2 and isinstance(aggregated[1], int):
        tokens, patch_start = aggregated
    else:
        tokens, patch_start = aggregated, None
    signature = inspect.signature(module.depth_head.forward)
    if "images" in signature.parameters:
        if patch_start is None:
            raise RuntimeError("VGGT aggregator did not return patch_start_idx")
        output = module.depth_head(
            tokens,
            images=images,
            patch_start_idx=patch_start,
            frames_chunk_size=4,
        )
    else:
        output = module.depth_head(tokens)
    if isinstance(output, tuple) and len(output) == 2:
        depth, confidence = output
    elif torch.is_tensor(output):
        depth, confidence = output, torch.ones_like(output)
    else:
        raise TypeError("VGGT depth head returned an invalid value")
    return {"depth": depth, "depth_conf": confidence}


def _prediction_pose(prediction: Mapping[str, Any]) -> torch.Tensor:
    pose = prediction.get("pose_enc")
    if pose is None:
        values = prediction.get("pose_enc_list")
        if isinstance(values, (tuple, list)) and values:
            pose = values[-1]
    if not torch.is_tensor(pose):
        raise ValueError("VGGT prediction does not contain pose encoding")
    return pose


def _depth_map(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 5 and value.shape[-1] == 1:
        return value[..., 0]
    if value.ndim == 5 and value.shape[2] == 1:
        return value[:, :, 0]
    if value.ndim == 4:
        return value
    raise ValueError("invalid depth shape")


@torch.no_grad()
def _teacher_labels(
    prediction: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    fov_mask: torch.Tensor,
    bindings: Any,
) -> dict[str, torch.Tensor]:
    pose = _prediction_pose(prediction)
    extrinsics, fov_h, fov_w = bindings.pose_encoding_to_extri_fov(pose)
    height, width = images.shape[-2:]
    intrinsics = bindings.pose_fov_to_intri(
        fov_h=fov_h, fov_w=fov_w, image_size_hw=(height, width)
    )
    depth = prediction["depth"]
    depth_map = _depth_map(depth)
    confidence = _depth_map(prediction["depth_conf"])
    candidate = fov_mask & torch.isfinite(depth_map) & (depth_map > 0)
    static = bindings.build_static_masks(
        depth=depth_map,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        candidate_masks=candidate,
    )
    canonical_depth = depth_map.unsqueeze(-1)
    world, camera, valid = bindings.unproject_depth(
        canonical_depth, extrinsics, intrinsics
    )
    normalized_extrinsics, _, normalized_world, normalized_depth = bindings.normalize_labels(
        extrinsics=extrinsics,
        cam_points=camera,
        world_points=world,
        depths=depth_map,
        point_masks=candidate,
        scale_by_points=False,
    )
    return {
        "extrinsics": normalized_extrinsics.detach(),
        "intrinsics": intrinsics.detach(),
        "depth": normalized_depth.unsqueeze(-1).detach(),
        "world_points": normalized_world.detach(),
        "static_masks": (static.bool() & valid.bool() & fov_mask).detach(),
        "confidence": confidence.detach(),
    }


def _source_batch(sample: Mapping[str, Any], bindings: Any, device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "images",
        "depths",
        "extrinsics",
        "intrinsics",
        "cam_points",
        "world_points",
        "point_masks",
    )
    batch = {
        name: torch.from_numpy(np.asarray(sample[name])).unsqueeze(0).to(device)
        for name in keys
    }
    batch["point_masks"] = batch["point_masks"].bool()
    extrinsics, cam_points, world_points, depths = bindings.normalize_labels(
        extrinsics=batch["extrinsics"],
        cam_points=batch["cam_points"],
        world_points=batch["world_points"],
        depths=batch["depths"],
        point_masks=batch["point_masks"],
    )
    batch.update(
        extrinsics=extrinsics,
        cam_points=cam_points,
        world_points=world_points,
        depths=depths,
    )
    return batch


def _target_batch(sample: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    images = sample["images"].unsqueeze(0).to(device)
    height, width = images.shape[-2:]
    fov = sample["fov_mask"].to(device).view(1, 1, height, width).expand(1, 8, height, width)
    return {"images": images, "fov_mask": fov}


def _window_inputs(
    labels: Mapping[str, torch.Tensor],
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    teacher_depth = _depth_map(labels["depth"]).float()
    student_depth = _depth_map(prediction["depth"]).detach().float()
    rgb = batch["images"].detach().float()
    fov = batch["fov_mask"].detach().bool()
    size = rgb.shape[0]
    finite_geometry = (
        torch.isfinite(teacher_depth).reshape(size, -1).all(1)
        & torch.isfinite(student_depth).reshape(size, -1).all(1)
        & torch.isfinite(labels["extrinsics"]).reshape(size, -1).all(1)
        & torch.isfinite(labels["intrinsics"]).reshape(size, -1).all(1)
    )
    fov_support = fov.float().reshape(size, -1).mean(1)
    intensity = rgb.amax(2)
    exposure = (intensity > 6 / 255) & (intensity < 250 / 255)
    exposure_support = (exposure & fov).float().reshape(size, -1).sum(1) / fov.float().reshape(size, -1).sum(1).clamp_min(1)
    horizontal = (rgb[..., 1:] - rgb[..., :-1]).abs().mean(2)
    vertical = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(2)
    blur_support = (5 * (horizontal.reshape(size, -1).mean(1) + vertical.reshape(size, -1).mean(1))).clamp(0, 1)
    teacher_scale = torch.nanmedian(torch.where(teacher_depth > 0, teacher_depth, torch.full_like(teacher_depth, float("nan"))).reshape(size, -1), 1).values
    student_scale = torch.nanmedian(torch.where(student_depth > 0, student_depth, torch.full_like(student_depth, float("nan"))).reshape(size, -1), 1).values
    return {
        "finite_geometry": finite_geometry,
        "fov_support": fov_support,
        "exposure_support": exposure_support,
        "blur_support": blur_support,
        "contact_fraction": (intensity > 0.995).float().reshape(size, -1).mean(1),
        "obstruction_fraction": (intensity < 0.01).float().reshape(size, -1).mean(1),
        "overlap_support": labels["static_masks"].float().reshape(size, -1).mean(1),
        "scale_ratio": student_scale / teacher_scale.clamp_min(1e-8),
    }


class HybridUpdateEngine:
    def __init__(
        self,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher: GatedTeacherController,
        *,
        gradient_clip: float = 1.0,
    ) -> None:
        self.student = student
        self.optimizer = optimizer
        self.teacher = teacher
        self.gradient_clip = float(gradient_clip)

    def step(
        self,
        successful_update: int,
        source_branches: tuple[Callable[[], tuple[torch.Tensor, Mapping[str, Any]]], ...],
        target_branch: Callable[[], TargetBranchResult],
    ) -> dict[str, Any]:
        if len(source_branches) != 3:
            raise ValueError("one Colon3R update requires three source microbatches")
        self.optimizer.zero_grad(set_to_none=True)
        source_values = []
        source_metrics = []
        for branch in source_branches:
            objective, metrics = branch()
            if objective.numel() != 1 or not bool(torch.isfinite(objective.detach())):
                raise FloatingPointError("source objective is not finite")
            (objective / 3).backward()
            source_values.append(float(objective.detach().cpu()))
            source_metrics.append(dict(metrics))
        target = target_branch()
        weight = target_weight(successful_update - 1)
        weighted_target = target.objective * weight
        target_gradients = accumulate_target_gradients(self.student, weighted_target)
        reduced = all_reduce_gradients(self.student)
        gradients = gradient_statistics(self.student)
        total_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.student.parameters() if parameter.requires_grad],
            self.gradient_clip,
        )
        if not bool(torch.isfinite(total_norm)):
            raise FloatingPointError("gradient norm is not finite")
        rates = set_optimizer_rates(self.optimizer, successful_update)
        self.optimizer.step()
        scale_gap = float(target.gate_probe())
        support = float(target.support)
        values = torch.tensor([support, scale_gap, 1.0], device=next(self.student.parameters()).device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values)
        support = float(values[0].item() / values[2].item())
        scale_gap = float(values[1].item() / values[2].item())
        teacher = self.teacher.update(
            self.student,
            successful_update,
            support=support,
            scale_gap=scale_gap,
        )
        return {
            "adaptation_update": successful_update,
            "global_step": STAGE1_STEP + successful_update,
            "source_loss": sum(source_values) / len(source_values),
            "target_loss": float(target.objective.detach().cpu()),
            "weighted_target_loss": float(weighted_target.detach().cpu()),
            "target_weight": weight,
            "support": support,
            "scale_gap": scale_gap,
            "teacher": teacher,
            "learning_rates": rates,
            "gradient_norm": float(total_norm.detach().cpu()),
            "gradient_statistics": gradients,
            "target_gradient_statistics": target_gradients,
            "reduced_gradient_count": reduced,
            "source": source_metrics,
            "target": dict(target.metrics),
        }


class Colon3RTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        output_directory: str | Path,
        resume: str | Path | None = None,
        c3vd_fov_mask: str | Path | None = None,
        total_updates: int = 6000,
    ) -> None:
        self.config = dict(config)
        self.rank, self.world_size, self.local_rank, self.device = _initialize_distributed()
        _seed(0, self.rank)
        self.output = Path(output_directory).expanduser().resolve()
        if self.output.exists() and resume is None:
            raise FileExistsError(self.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output / f"metrics_rank{self.rank}.jsonl"
        self.total_updates = int(total_updates)
        if not 1 <= self.total_updates <= 6000:
            raise ValueError("total updates must be in [1,6000]")
        paths = self.config["paths"]
        self.bindings = load_wat3r_bindings(paths["wat3r_root"])
        self.source_data = SourceManifestDataset(
            Path(paths["source_train_manifest"]),
            c3vd_calibration=Path(paths["c3vd_calibration"]),
            c3vd_fov_mask=Path(c3vd_fov_mask) if c3vd_fov_mask else Path(paths["c3vd_calibration"]).with_name("c3vd_fov_mask.png"),
        )
        self.target_data = TargetManifestDataset(Path(paths["target_train_manifest"]))
        plain_teacher = build_teacher(self.config, device=self.device)
        self.anchor_cache = self._build_anchor_cache(plain_teacher)
        self.student = build_student(self.config, device=self.device)
        teacher_values = self.config["teacher"]
        self.teacher = GatedTeacherController(
            self.student,
            plain_teacher,
            hard_copy_update=teacher_values["hard_copy_update"],
            momentum=teacher_values["ema_momentum"],
            support_threshold=teacher_values["support_threshold"],
            scale_threshold=teacher_values["scale_threshold"],
        ).move_to(self.device, torch.float32)
        self.optimizer = build_optimizer(self.student)
        self.start_update = 0
        if resume is not None:
            loaded = load_colon3r_checkpoint(
                resume,
                self.student,
                teacher=self.teacher,
                optimizer=self.optimizer,
            )
            self.start_update = int(loaded["adaptation_update"])
        self.student.train()
        self.source_criterion = self.bindings.MultitaskLoss(
            camera={"weight": 5.0, "loss_type": "l1"},
            depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
            point={"weight": 1.0, "gradient_loss_fn": "normal", "valid_range": 0.98},
            track=None,
            ema={"weight": 0.5},
            device=self.device,
        ).to(self.device)
        self.engine = HybridUpdateEngine(self.student, self.optimizer, self.teacher)
        self.frozen_digests = parameter_group_digests(self.student)

    @torch.no_grad()
    def _build_anchor_cache(self, anchor_teacher: nn.Module) -> dict[int, torch.Tensor]:
        result = {}
        anchor_teacher.eval()
        for index in range(len(self.target_data)):
            batch = _target_batch(self.target_data[index], self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = anchor_teacher(images=batch["images"])
            labels = _teacher_labels(prediction, batch["images"], batch["fov_mask"], self.bindings)
            depth = _depth_map(labels["depth"]).float()
            mask = labels["static_masks"] & batch["fov_mask"] & torch.isfinite(depth) & (depth > 0)
            flat = depth.reshape(8, -1)
            fixed = mask.reshape(8, -1)
            if bool((fixed.sum(1) == 0).any()):
                raise RuntimeError(f"empty fixed depth-anchor support at target row {index}")
            reference = (torch.log(flat.clamp_min(torch.finfo(torch.float32).tiny)) * fixed.float()).sum(1) / fixed.sum(1)
            result[index] = reference.detach().cpu()
        return result

    def _source_branch(self, update: int, microbatch: int) -> tuple[torch.Tensor, Mapping[str, Any]]:
        ordinal = (update - 1) * 3 + microbatch
        index = (ordinal * self.world_size + self.rank) % len(self.source_data)
        sample = self.source_data.load(index)
        batch = _source_batch(sample, self.bindings, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.student(images=batch["images"])
            objective, terms = source_multitask_loss(prediction, batch, self.source_criterion)
        row = sample["row"]
        return objective, {
            "index": index,
            "clip_id": row.get("clip_id"),
            "dataset": row["source"],
            "camera_model": row["camera_model"],
            "terms": {name: float(value.detach().cpu()) for name, value in terms.items() if torch.is_tensor(value) and value.numel() == 1},
        }

    def _target_branch(self, update: int) -> TargetBranchResult:
        ordinal = update - 1
        index = (ordinal * self.world_size + self.rank) % len(self.target_data)
        sample = self.target_data[index]
        batch = _target_batch(sample, self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_prediction = self.teacher.teacher(images=batch["images"])
        labels = _teacher_labels(teacher_prediction, batch["images"], batch["fov_mask"], self.bindings)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_prediction = _depth_output(self.student, batch["images"])
        geometry = predicted_pinhole_cross_view_geometry(
            teacher_depth=labels["depth"],
            student_depth=student_prediction["depth"],
            student_confidence=student_prediction["depth_conf"],
            teacher_confidence=labels["confidence"],
            intrinsics=labels["intrinsics"],
            extrinsics=labels["extrinsics"],
            static_mask=labels["static_masks"],
            fov_mask=batch["fov_mask"],
            rgb=batch["images"],
        )
        geometry["window_inputs"] = _window_inputs(labels, student_prediction, batch)
        window = window_reliability(geometry["window_inputs"])
        pair = pair_reliability(geometry)
        pixel = pixel_reliability(geometry)
        reliability = compose_reliability(window, pair, pixel)
        quasi = target_quasi_rigid_loss(
            student_depth=geometry["sampled_depth"],
            teacher_projected_depth=geometry["projected_depth"],
            reliability=reliability,
        )
        reference = self.anchor_cache[index].to(self.device)
        fixed_mask = labels["static_masks"] & batch["fov_mask"]
        anchor = depth_scale_anchor_loss(
            student_prediction["depth"], fixed_mask, reference
        )
        objective = quasi + self.config["teacher"]["anchor_weight"] * anchor.loss
        geometric_support = geometry["projection_valid"].sum().clamp_min(1)
        support = float((reliability.sum() / geometric_support).detach().cpu())

        def gate_probe() -> float:
            self.student.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                current = _depth_output(self.student, batch["images"])
            self.student.train()
            return depth_scale_anchor_loss(current["depth"], fixed_mask, reference).window_gap

        row = sample["row"]
        return TargetBranchResult(
            objective,
            support,
            gate_probe,
            {
                "index": index,
                "clip_id": row.get("clip_id"),
                "procedure_id": row.get("procedure_id"),
                "quasi_rigid_loss": float(quasi.detach().cpu()),
                "anchor_loss": float(anchor.loss.detach().cpu()),
                "camera_model": "teacher_predicted_pinhole",
                "locked_target_accessed": False,
            },
        )

    def _append_metrics(self, values: Mapping[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def run(self) -> None:
        for update in range(self.start_update + 1, self.total_updates + 1):
            started = time.perf_counter()
            result = self.engine.step(
                update,
                tuple(
                    (lambda microbatch=microbatch: self._source_branch(update, microbatch))
                    for microbatch in range(3)
                ),
                lambda: self._target_branch(update),
            )
            current = parameter_group_digests(self.student)
            if current["base"] != self.frozen_digests["base"] or current["track"] != self.frozen_digests["track"]:
                raise RuntimeError("frozen backbone or track parameters changed")
            result["seconds"] = time.perf_counter() - started
            result["locked_target_accessed"] = False
            self._append_metrics(result)
            checkpoint = update == 20 or update % 1000 == 0 or update == self.total_updates
            if checkpoint and self.rank == 0:
                path = self.output / "checkpoints" / f"colon3r_global_{STAGE1_STEP + update:06d}.pt"
                saved = save_training_checkpoint(
                    path,
                    self.student,
                    self.teacher,
                    self.optimizer,
                    global_step=STAGE1_STEP + update,
                    adaptation_update=update,
                )
                load_colon3r_checkpoint(
                    path,
                    self.student,
                    teacher=self.teacher,
                    optimizer=self.optimizer,
                    expected_sha256=saved["sha256"],
                )
                print(json.dumps({"checkpoint": saved, "metrics": result}, sort_keys=True), flush=True)
            elif update % 50 == 0 and self.rank == 0:
                print(json.dumps(result, sort_keys=True), flush=True)
            if dist.is_initialized():
                dist.barrier()


def run_training(
    *,
    config_path: str | Path,
    output_directory: str | Path,
    resume: str | Path | None = None,
    source_manifest: str | Path | None = None,
    target_manifest: str | Path | None = None,
    c3vd_fov_mask: str | Path | None = None,
    total_updates: int = 6000,
) -> None:
    config = load_config(config_path)
    if source_manifest is not None:
        config["paths"]["source_train_manifest"] = str(Path(source_manifest).expanduser().resolve())
    if target_manifest is not None:
        config["paths"]["target_train_manifest"] = str(Path(target_manifest).expanduser().resolve())
    trainer = Colon3RTrainer(
        config,
        output_directory=output_directory,
        resume=resume,
        c3vd_fov_mask=c3vd_fov_mask,
        total_updates=total_updates,
    )
    trainer.run()


__all__ = [
    "Colon3RTrainer",
    "HybridUpdateEngine",
    "TargetBranchResult",
    "load_config",
    "run_training",
]
