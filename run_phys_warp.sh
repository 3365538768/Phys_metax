#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_phys_warp.sh --model_path model --material_space_config ... --train_config ... --num_gpus 1

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
PHYS_WARP_ENV="${PHYS_WARP_ENV:-$PROJECT_ROOT/envs/warp}"
PYTHON_BIN="$PHYS_WARP_ENV/bin/python"

ensure_ffmpeg_runtime_libs() {
  if [ ! -x /usr/bin/ffmpeg ]; then
    return 0
  fi

  local missing=""
  missing="$(ldd /usr/bin/ffmpeg 2>/dev/null | awk '/libxcb-shape\.so\.0|libxcb-xfixes\.so\.0/ && /not found/ {print $1}')"
  if [ -z "$missing" ]; then
    return 0
  fi

  echo "[run_phys_warp] missing ffmpeg runtime libs: $missing"
  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    echo "[run_phys_warp] WARN: cannot auto-install ffmpeg runtime libs; please install libxcb-shape0 libxcb-xfixes0 manually." >&2
    return 0
  fi

  echo "[run_phys_warp] installing ffmpeg runtime libs via apt-get..."
  apt-get update
  apt-get install -y libxcb-shape0 libxcb-xfixes0
}

if [ ! -d "$PHYS_WARP_ENV" ]; then
  echo "ERROR: conda env not found: $PHYS_WARP_ENV" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python not found in env: $PYTHON_BIN" >&2
  exit 1
fi

# Prefer conda activation when available; otherwise fall back to env python directly.
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  if command -v conda >/dev/null 2>&1; then
    conda activate "$PHYS_WARP_ENV"
    PYTHON_BIN="$(command -v python)"
  fi
fi

export MACA_PATH="${MACA_PATH:-/opt/maca-3.3.0}"
export MACA_CLANG_PATH="${MACA_CLANG_PATH:-$MACA_PATH/mxgpu_llvm/bin}"
export CUDA_PATH="${CUDA_PATH:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$MACA_PATH/tools/cu-bridge}"
export PATH="$CUCC_PATH/tools:$CUCC_PATH/bin:$MACA_CLANG_PATH:$MACA_PATH/bin:$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUCC_PATH/lib:$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:${LD_LIBRARY_PATH:-}"
export MACA_DIRECT_DISPATCH="${MACA_DIRECT_DISPATCH:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ensure_ffmpeg_runtime_libs

exec "$PYTHON_BIN" auto_simulation_runner.py "$@"
