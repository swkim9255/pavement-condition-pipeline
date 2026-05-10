# Pavement Condition Pipeline

This repository is a monorepo for pavement-condition processing experiments and
pipeline components.

## Layout

- `projects/oneformer`: RTX 5060 Ti / CUDA 12.8 compatible OneFormer setup for
  semantic and panoptic segmentation experiments.

Add future pipeline pieces as separate project directories under `projects/`.
Each project should keep its own setup notes, environment file, requirements,
scripts, and README close to the code it needs.

## OneFormer

```bash
cd projects/oneformer
bash tools/setup_rtx5060ti_env.sh
```

The setup script creates the `oneformer5060` conda environment, installs the
CUDA 12.8 PyTorch stack, clones Detectron2 when needed, and verifies the runtime
imports.

Large local files such as input videos, frames, generated outputs, checkpoints,
and model weights are intentionally excluded from Git.
