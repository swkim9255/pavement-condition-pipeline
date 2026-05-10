# OneFormer Mapillary Inference

This is a trimmed OneFormer workspace for Mapillary Vistas semantic inference.
It keeps only the code needed to load the Swin-L Mapillary checkpoint and export
binary class masks for the pavement pipeline.

## Included

- `configs/mapillary_vistas`: Mapillary Swin-L inference config.
- `demo`: single-image / frame visualization helpers.
- `oneformer`: model, Swin backbone, pixel decoder, tokenizer, and minimal
  Mapillary metadata registration.
- `tools/export_mapillary_bike_lane_masks.py`: batch semantic inference that
  writes binary masks for class id `7` by default.
- `tools/setup_rtx5060ti_env.sh`: conda + CUDA 12.8 setup for RTX 50-series.

Training, evaluation, ADE20K/COCO/Cityscapes configs, Colab notebooks, and
conversion utilities were removed from this pipeline copy.

## Setup

```bash
bash tools/setup_rtx5060ti_env.sh
```

The script creates `oneformer5060`, installs PyTorch CUDA 12.8 wheels, clones
Detectron2 if needed, builds the local CUDA op, and runs an import check.

## Export Bike-Lane Masks

Put the checkpoint at:

```text
checkpoints/oneformer_mapillary_swin_large.pth
```

Then run:

```bash
python tools/export_mapillary_bike_lane_masks.py \
  --config-file configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml \
  --weights checkpoints/oneformer_mapillary_swin_large.pth \
  --input 'inputs/RGB_frames/*.jpg' \
  --output outputs/bike_lane_masks
```

Large inputs, outputs, checkpoints, and local Detectron2 checkouts are ignored by
Git.
