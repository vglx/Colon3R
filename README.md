# Colon3R

Colon3R adapts VGGT to colonoscopy geometry with labeled synthetic/C3VD source clips and unlabeled in-vivo target clips. The released path trains Q/V/O LoRA adapters and the camera, depth, and point heads while retaining a frozen backbone and track head. Target updates use hierarchical quasi-rigid cross-view reliability, a source-prior depth-scale anchor, and a support-and-scale-gated Teacher.

## Repository

```text
Colon3R/
  train.py
  inference.py
  configs/colon3r.yaml
  checkpoints/
    stage1_4k.pt
    colon3r_10k.pt
    SHA256SUMS
  colon3r/
```

VGGT and Wat3R are external dependencies. Their source is not vendored or modified here.

## Environment

The verified environment used Python 3.11, PyTorch 2.2.2, torchvision 0.17.2, CUDA 11.8, NumPy 1.26.4, Pillow 10.3.0, PyYAML 6.0.1, and Kornia 0.7.2. Training requires two CUDA GPUs with bf16 support. Inference requires one CUDA GPU.

```bash
conda create -n colon3r python=3.11 -y
conda activate colon3r
pip install -r requirements.txt

mkdir -p third_party
git clone https://github.com/facebookresearch/vggt.git third_party/VGGT
git clone https://github.com/LSXI7/Wat3R.git third_party/Wat3R
pip install -e third_party/VGGT
pip install -e third_party/Wat3R --no-deps
```

The extracted implementation was validated against Wat3R commit `93147df00e15605afa93f586923fec000b18cefc`. The public config also accepts another compatible official checkout through `paths.wat3r_root`. Set `VGGT_ROOT` or update `paths.vggt_root` when VGGT is not under `third_party/VGGT`.

VGGT base weights are loaded through `VGGT.from_pretrained("facebook/VGGT-1B")`; make that Hugging Face artifact available in the local cache for offline execution.

## Checkpoints

The release includes exact copies of the verified checkpoints:

| File | Step | SHA-256 |
|---|---:|---|
| `checkpoints/stage1_4k.pt` | 4000 | `3a6d61a3132763d1bb566cd24dafe371470b7e8e3b81d9df7cc617df1aa0e739` |
| `checkpoints/colon3r_10k.pt` | 10000 | `7407b0681bacc6e7d7fb0c130206da0c68daa9f4306ad6017ad9a37fa9ac1cc5` |

Verify them before use:

```bash
cd checkpoints
sha256sum -c SHA256SUMS
cd ..
```

The 10k checkpoint contains adaptation update 6000, LoRA and camera/depth/point head tensors, optimizer state, Teacher state, sampler state, and resume metadata.

## Data

All manifests are JSONL. Every row contains exactly eight chronological frames. Paths can be absolute or relative to the repository working directory.

C3VD source rows use official Scaramuzza calibration and camera-Z depth:

```json
{"clip_id":"c3vd-example","source":"c3vd","camera_model":"official_scaramuzza_omnidirectional","fallback_used":false,"clip_length":8,"frame_ids":[0,1,2,3,4,5,6,7],"rgb_paths":["..."],"depth_paths":["..."],"pose_path":".../poses_gt.txt"}
```

Each `poses_gt.txt` row is a row-major 4x4 camera-to-world matrix. The loader inverts it only where a world-to-camera transform is required. C3VD depth images are official 16-bit 0–100 mm camera-Z maps. Put the verified calibration and FOV mask at:

```text
data/source/calibration/cfhq190l_omnidirectional_calibration.json
data/source/calibration/c3vd_fov_mask.png
```

Sim source rows use explicit pinhole cameras:

```json
{"clip_id":"sim-example","source":"sim","camera_model":"explicit_pinhole","fallback_used":false,"clip_length":8,"frame_ids":[0,1,2,3,4,5,6,7],"rgb_paths":["..."],"depth_cache":".../depth.npz","intrinsics_path":".../intrinsics.json","pose_path":".../poses.txt"}
```

The sim depth NPZ contains `depth` and `valid` arrays. The intrinsics JSON contains `fx`, `fy`, `cx`, and `cy`.

Target rows are RGB-only. Any depth, pose, intrinsic, calibration, mesh, trajectory, or pointmap field is rejected:

```json
{"clip_id":"target-example","procedure_id":"procedure-01","source":"target","supervised":false,"clip_length":8,"frame_ids":[0,1,2,3,4,5,6,7],"rgb_paths":["..."]}
```

Update the portable paths in `configs/colon3r.yaml`, or override the source and target manifests on the command line.

## Training

The fixed protocol is:

- two NCCL workers;
- three labeled source microbatches and one unlabeled target microbatch per update;
- eight Teacher views, eight Student views, and 56 directed pairs;
- rank-16, alpha-16, zero-dropout Q/V/O LoRA in global blocks 12–23;
- source updates to LoRA plus camera/depth/point heads;
- target updates to LoRA plus depth head only;
- AdamW with peak rates `2e-5`, `2e-7`, `1e-7`, and `5e-8` for LoRA, depth, point, and camera;
- bf16 model forwards and fp32 geometry/loss reduction;
- fixed Teacher through update 499, hard copy at 500, then gated EMA with momentum 0.996;
- support threshold `0.20138970316367508`, scale threshold `0.004409444387420081`, and depth-anchor weight `0.05`;
- 6000 adaptation updates from global step 4000 to 10000.

Start a new run in a new output directory:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/colon3r.yaml \
  --output outputs/colon3r_run \
  --c3vd-fov-mask data/source/calibration/c3vd_fov_mask.png
```

Resume without changing the protocol:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/colon3r.yaml \
  --output outputs/colon3r_run \
  --resume outputs/colon3r_run/checkpoints/colon3r_global_005000.pt \
  --c3vd-fov-mask data/source/calibration/c3vd_fov_mask.png
```

The trainer writes rank-local JSONL metrics and rank-zero checkpoints at global step 4020, then every 1000 global steps, and at the requested terminal step. Existing checkpoints are never overwritten. Every saved checkpoint is immediately hash-verified and reloaded.

## Inference

Pass one directory containing exactly eight RGB images, one text file containing eight image paths, or eight explicit image paths:

```bash
python inference.py \
  --input examples/frames \
  --checkpoint checkpoints/colon3r_10k.pt \
  --output outputs/example \
  --export-ply
```

Inference performs one frozen eight-frame aggregator pass and saves:

```text
outputs/example/
  predictions.npz
  metadata.json
  direct_pointmap_fused_rgb.ply
  per_frame/direct_pointmap_00_rgb.ply
  ...
```

`predictions.npz` contains `extrinsics_w2c`, `intrinsics`, `depth_z`, `depth_confidence`, `direct_pointmap`, and `pointmap_confidence`. Extrinsics follow OpenCV world-to-camera convention. Depth is camera-axis Z. Direct pointmaps remain in native VGGT first-frame-normalized world coordinates.

PLY export is finite-point filtering and concatenation only. It does not apply confidence quantile filtering, Sim(3), ICP, GT alignment, TSDF, surface fusion, or meshing.

## Scope

C3VD supervision uses the official Scaramuzza omnidirectional ray model with `X_camera = unit_ray * depth_z / unit_ray_z`. Sim supervision uses explicit pinhole intrinsics. Unlabeled in-vivo target geometry uses Teacher-predicted pinhole intrinsics because no calibrated target camera is supplied.

The final model reflects its stated source supervision and target adaptation data. Comparisons with other systems should not be described as architecture-only comparisons when their training data or supervision differ.

## Third-Party Attribution

Colon3R depends on VGGT, introduced in “VGGT: Visual Geometry Grounded Transformer” (CVPR 2025), and on the Wat3R training geometry from “Wat3R: Underwater 3D Geometry Learning without Annotations” (ECCV 2026). Review and comply with the licenses in the respective official repositories before redistribution.
