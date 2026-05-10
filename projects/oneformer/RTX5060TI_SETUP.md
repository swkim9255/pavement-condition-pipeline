# RTX 5060 Ti OneFormer Setup

The original OneFormer installation targets `torch==1.10.1` and CUDA 11.3.
That stack is not suitable for RTX 50-series Blackwell GPUs. Use the setup
below for the Cityscapes Swin checkpoint in this workspace.

## Prerequisites

- NVIDIA driver compatible with CUDA 12.8.
- CUDA Toolkit 12.8 installed with `nvcc` available at `/usr/local/cuda-12.8/bin/nvcc`.
- Conda available in `PATH`.

If CUDA Toolkit 12.8 is installed elsewhere, export `CUDA_HOME` before running
the setup script.

```bash
export CUDA_HOME=/path/to/cuda-12.8
```

## Setup

```bash
bash tools/setup_rtx5060ti_env.sh
```

The script creates or reuses the `oneformer5060` conda environment, installs
PyTorch `2.10.0+cu128`, installs RTX 50-compatible OneFormer demo dependencies,
clones Detectron2 if it is not already present, builds Detectron2 from the
local source checkout, and rebuilds the
MSDeformAttn CUDA extension for `sm_120`.

Useful overrides:

```bash
ONEFORMER_ENV_NAME=oneformer5060 CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 \
  bash tools/setup_rtx5060ti_env.sh
```

## Verify

```bash
conda activate oneformer5060
python tools/verify_rtx5060ti_env.py
```

Expected:

- `torch` reports a CUDA 12.8 build.
- `torch.cuda.is_available()` is `True`.
- GPU compute capability is `(12, 0)` for RTX 5060 Ti.
- `detectron2`, `oneformer`, `natten`, `cv2`, and `MultiScaleDeformableAttention` import successfully.

## Demo Inference

After extracting video frames into `inputs/RGB_frames`, run:

```bash
python demo/demo.py \
  --config-file configs/cityscapes/swin/oneformer_swin_large_bs16_90k.yaml \
  --input inputs/RGB_frames/*.jpg \
  --output outputs/RGB_video_oneformer \
  --task panoptic \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS checkpoints/oneformer_cityscapes_swin_large.pth
```
