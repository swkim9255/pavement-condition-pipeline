#!/usr/bin/env python
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def require_module(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"failed to import {name}: {exc}") from exc


torch = require_module("torch")
print(f"torch: {torch.__version__}")
print(f"torch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch. Check the NVIDIA driver/container GPU access.")

capability = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"compute capability: {capability}")

if capability[0] < 12:
    raise SystemExit(
        f"expected a Blackwell/RTX 50-series GPU with compute capability 12.x, got {capability}"
    )

for module_name in ("detectron2", "oneformer", "cv2"):
    require_module(module_name)

try:
    require_module("MultiScaleDeformableAttention")
except SystemExit as exc:
    raise SystemExit(
        "failed to import MultiScaleDeformableAttention. Rebuild "
        "oneformer/modeling/pixel_decoder/ops with CUDA_HOME pointing to CUDA 12.8."
    ) from exc

print("imports ok")
