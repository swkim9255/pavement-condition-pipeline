# RTX 5060 Ti Setup

This trimmed OneFormer project targets Mapillary Vistas Swin-L semantic
inference on RTX 50-series / Blackwell GPUs.

## Prerequisites

- NVIDIA driver compatible with CUDA 12.8.
- CUDA Toolkit 12.8 with `nvcc` available at `/usr/local/cuda-12.8/bin/nvcc`.
- Conda available in `PATH`.

If CUDA Toolkit 12.8 is installed elsewhere:

```bash
export CUDA_HOME=/path/to/cuda-12.8
```

## Setup

```bash
bash tools/setup_rtx5060ti_env.sh
```

The script creates or reuses the `oneformer5060` conda environment, installs
PyTorch `2.10.0+cu128`, installs inference dependencies, clones Detectron2 when
needed, builds Detectron2, and rebuilds the OneFormer MSDeformAttn CUDA op for
`sm_120`.

## Verify

```bash
conda activate oneformer5060
python tools/verify_rtx5060ti_env.py
```

Expected:

- `torch` reports a CUDA 12.8 build.
- `torch.cuda.is_available()` is `True`.
- GPU compute capability is `(12, 0)` for RTX 5060 Ti.
- `detectron2`, `oneformer`, `cv2`, and `MultiScaleDeformableAttention` import
  successfully.

## Inference

```bash
python tools/export_mapillary_bike_lane_masks.py \
  --config-file configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml \
  --weights checkpoints/oneformer_mapillary_swin_large.pth \
  --input 'inputs/RGB_frames/*.jpg' \
  --output outputs/bike_lane_masks
```
