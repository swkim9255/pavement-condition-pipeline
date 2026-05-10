#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ONEFORMER_ENV_NAME:-oneformer5060}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
DETECTRON2_REPO="${DETECTRON2_REPO:-https://github.com/facebookresearch/detectron2.git}"
DETECTRON2_REF="${DETECTRON2_REF:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v conda >/dev/null 2>&1 || die "conda was not found in PATH"
command -v git >/dev/null 2>&1 || die "git was not found in PATH"

if [[ -x /usr/local/cuda-12.8/bin/nvcc ]] && ! "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -q "release 12.8"; then
  echo "warning: CUDA_HOME=${CUDA_HOME} is not CUDA 12.8; using /usr/local/cuda-12.8 for RTX 50-series setup." >&2
  CUDA_HOME=/usr/local/cuda-12.8
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  die "CUDA 12.8 nvcc was not found at ${CUDA_HOME}/bin/nvcc. Install CUDA Toolkit 12.8 or set CUDA_HOME."
fi

CUDA_VERSION="$("${CUDA_HOME}/bin/nvcc" --version)"
if ! grep -q "release 12.8" <<<"${CUDA_VERSION}"; then
  die "expected CUDA Toolkit 12.8 at CUDA_HOME=${CUDA_HOME}, but nvcc reported: ${CUDA_VERSION}"
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create --name "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"

if [[ -e detectron2 && ! -d detectron2/.git ]]; then
  die "detectron2 exists but is not a git checkout. Move it aside or clone Detectron2 there."
fi

if [[ ! -d detectron2/.git ]]; then
  git clone --depth 1 --branch "${DETECTRON2_REF}" "${DETECTRON2_REPO}" detectron2
fi

python -m pip install --upgrade pip wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_rtx5060ti_cu128.txt

CUDA_HOME="${CUDA_HOME}" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  python -m pip install --no-build-isolation -e detectron2

pushd oneformer/modeling/pixel_decoder/ops >/dev/null
CUDA_HOME="${CUDA_HOME}" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  python setup.py build install
popd >/dev/null

python tools/verify_rtx5060ti_env.py

cat <<EOF

RTX 5060 Ti OneFormer environment is ready.

Activate it with:
  conda activate ${ENV_NAME}

Run inference with:
  python tools/export_mapillary_bike_lane_masks.py \\
    --config-file configs/mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml \\
    --weights checkpoints/oneformer_mapillary_swin_large.pth \\
    --input inputs/RGB_frames/*.jpg \\
    --output outputs/bike_lane_masks
EOF
