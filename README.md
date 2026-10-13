<h1 align="center">Colon3R: Cross-Domain 3D Reconstruction from Monocular Colonoscopic Video</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.23961"><strong>Paper</strong></a>
  &nbsp;|&nbsp;
  <a href="https://vglx.github.io/Colon3R/"><strong>Project Page</strong></a>
</p>

<p align="center">
  Zhihao Xing &nbsp;&nbsp; Yingyu Wang &nbsp;&nbsp; Liang Zhao &nbsp;&nbsp; Shoudong Huang
</p>

## In-vivo Reconstruction

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Segment 1</strong><br><br>
      <img src="assets/invivo/segment_1.gif" width="100%" alt="Segment 1: Colon3R reconstruction, rotating without zoom">
    </td>
    <td align="center" width="50%">
      <strong>Segment 2</strong><br><br>
      <img src="assets/invivo/segment_2.gif" width="100%" alt="Segment 2: Colon3R reconstruction, rotating without zoom">
    </td>
  </tr>
</table>

## Overview

Colon3R adapts VGGT from labeled phantom and simulated data to unlabeled in-vivo colonoscopy. It selects cross-view supervision at sequence, directed-pair and pixel levels, while retaining source supervision and constraining target updates to preserve coupled camera, depth and pointmap geometry.

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="Colon3R pipeline: source-supervised geometry, reliability-aware target adaptation, and source geometry preservation">
</p>

## Repository

```text
Colon3R/
  train.py
  inference.py
  configs/colon3r.yaml
  checkpoints/
    colon3r.pt
  colon3r/
```

VGGT is the only external model source dependency. The geometry, supervision, reliability, and adaptation code used by Colon3R is included in the `colon3r` package.

## Environment

The verified environment used Python 3.11, PyTorch 2.2.2, torchvision 0.17.2, CUDA 11.8, NumPy 1.26.4, Pillow 10.3.0, PyYAML 6.0.1, and Kornia 0.7.2. Training and inference require at least one CUDA GPU with bf16 support. Additional GPUs can be selected at launch time.

```bash
conda create -n colon3r python=3.11 -y
conda activate colon3r
pip install -r requirements.txt

mkdir -p third_party
git clone https://github.com/facebookresearch/vggt.git third_party/VGGT
pip install -e third_party/VGGT
```

Set `VGGT_ROOT` or update `paths.vggt_root` when VGGT is not under `third_party/VGGT`.

VGGT base weights are loaded through `VGGT.from_pretrained("facebook/VGGT-1B")`; make that Hugging Face artifact available in the local cache for offline execution.

## Checkpoints

[colon3r.pt](https://studentutsedu-my.sharepoint.com/:u:/g/personal/zhihao_xing_student_uts_edu_au/IQD5OyBnv0gETaZLqk3AusbDAYLPUBSOfLkCfxq3mkW78uc?e=EQSffN)

Place the downloaded checkpoint in `checkpoints/`.

## Data

Colon3R uses labeled C3VD phantom and simulated colonoscopy data for source supervision, and unlabeled in-vivo RGB sequences for target-domain adaptation. No target-domain geometric annotations are required.

Prepare eight-frame clips in JSONL manifests and set the dataset and calibration paths in `configs/colon3r.yaml`.

## Training

Configure data paths and training settings in `configs/colon3r.yaml`, then run:

**Single GPU**

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/colon3r.yaml \
  --output outputs/colon3r_run \
  --c3vd-fov-mask data/source/calibration/c3vd_fov_mask.png
```

**Multiple GPUs (two-GPU example)**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/colon3r.yaml \
  --output outputs/colon3r_run \
  --c3vd-fov-mask data/source/calibration/c3vd_fov_mask.png
```

## Inference

Pass one directory containing exactly eight RGB images, one text file containing eight image paths, or eight explicit image paths:

```bash
python inference.py \
  --input examples/frames \
  --checkpoint checkpoints/colon3r.pt \
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

## Acknowledgements

We thank the authors of [VGGT](https://github.com/facebookresearch/vggt) for their open-source implementation.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{xing2026colon3rcrossdomain3dreconstruction,
      title={Colon3R: Cross-Domain 3D Reconstruction from Monocular Colonoscopic Video},
      author={Zhihao Xing and Yingyu Wang and Liang Zhao and Shoudong Huang},
      year={2026},
      eprint={2609.23961},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.23961},
}
```

