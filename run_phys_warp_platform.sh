#!/usr/bin/env bash
set -euo pipefail

# Platform-oriented launcher for SenseCore-like training jobs.
# - No conda activation required
# - Use absolute python path from env
# - Export MACA/cu-bridge runtime variables
#
# Usage:
#   ./run_phys_warp_platform.sh \
#     --model_path model \
#     --material_space_config configs_auto/material_space_example.json \
#     --train_config configs_auto/train_config_dataset_full.json \
#     --num_gpus 1

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
export PHYS_DIR="${PHYS_DIR:-$PROJECT_ROOT/Phys}"
export PHYS_WARP_ENV="${PHYS_WARP_ENV:-$PROJECT_ROOT/envs/warp}"
export PYTHON_BIN="${PYTHON_BIN:-$PHYS_WARP_ENV/bin/python}"

ensure_ffmpeg_runtime_libs() {
  if [ ! -x /usr/bin/ffmpeg ]; then
    return 0
  fi

  local missing=""
  missing="$(ldd /usr/bin/ffmpeg 2>/dev/null | awk '/libxcb-shape\.so\.0|libxcb-xfixes\.so\.0/ && /not found/ {print $1}')"
  if [ -z "$missing" ]; then
    return 0
  fi

  echo "[platform] missing ffmpeg runtime libs: $missing"
  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    echo "[platform] WARN: cannot auto-install ffmpeg runtime libs; please install libxcb-shape0 libxcb-xfixes0 manually." >&2
    return 0
  fi

  echo "[platform] installing ffmpeg runtime libs via apt-get..."
  apt-get update
  apt-get install -y libxcb-shape0 libxcb-xfixes0
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

# Detect MACA library root and cu-bridge root separately because some
# environments expose them under different prefixes.
_maca_lib_candidates=()
if [ -n "${MACA_PATH:-}" ]; then
  _maca_lib_candidates+=("$MACA_PATH")
fi
_maca_lib_candidates+=("/opt/maca" "/opt/maca-3.3.0")
for _p in /opt/maca-*; do
  if [ -d "$_p" ]; then
    _maca_lib_candidates+=("$_p")
  fi
done

_MACA_LIB_ROOT=""
for _p in "${_maca_lib_candidates[@]}"; do
  if [ -f "$_p/lib/libToolsExt_cu.so" ]; then
    _MACA_LIB_ROOT="$_p"
    break
  fi
done

_bridge_candidates=()
if [ -n "${CUDA_PATH:-}" ]; then
  _bridge_candidates+=("$CUDA_PATH")
fi
if [ -n "${CUCC_PATH:-}" ]; then
  _bridge_candidates+=("$CUCC_PATH")
fi
_bridge_candidates+=(
  "/opt/maca/tools/cu-bridge"
  "/opt/maca-3.3.0/tools/cu-bridge"
)
for _p in "${_maca_lib_candidates[@]}"; do
  _bridge_candidates+=("$_p/tools/cu-bridge")
done

_BRIDGE_ROOT=""
for _p in "${_bridge_candidates[@]}"; do
  if [ -d "$_p" ] && [ -d "$_p/lib" ]; then
    _BRIDGE_ROOT="$_p"
    break
  fi
done

if [ -z "$_MACA_LIB_ROOT" ]; then
  echo "ERROR: no valid MACA lib root found (need lib/libToolsExt_cu.so)." >&2
  exit 1
fi
if [ -z "$_BRIDGE_ROOT" ]; then
  echo "ERROR: no valid cu-bridge root found (need tools/cu-bridge/lib)." >&2
  exit 1
fi

export MACA_PATH="$_MACA_LIB_ROOT"
export CUDA_PATH="$_BRIDGE_ROOT"
export CUCC_PATH="$_BRIDGE_ROOT"

_clang_candidates=(
  "$MACA_PATH/mxgpu_llvm/bin"
  "/opt/maca/mxgpu_llvm/bin"
  "/opt/maca-3.3.0/mxgpu_llvm/bin"
)
_MACA_CLANG_PATH=""
for _p in "${_clang_candidates[@]}"; do
  if [ -d "$_p" ]; then
    _MACA_CLANG_PATH="$_p"
    break
  fi
done
if [ -z "$_MACA_CLANG_PATH" ]; then
  _MACA_CLANG_PATH="$MACA_PATH/mxgpu_llvm/bin"
fi
export MACA_CLANG_PATH="$_MACA_CLANG_PATH"

export MACA_DIRECT_DISPATCH="${MACA_DIRECT_DISPATCH:-1}"
export MACA_COMPAT_PATH="/opt/maca"

# Optional stubs path on some shared platforms.
# Keep it OFF by default for Phys, because stub libToolsExt_cu.so may miss symbols
# required by torch (e.g. wnvtxRangePushA).
export ENABLE_MACA_STUBS="${ENABLE_MACA_STUBS:-0}"
export MACA_STUBS_PATH="${MACA_STUBS_PATH:-/mnt/afs/lixiaoou/intern/linweitao/maca_stubs}"

_ld_parts=()
if [ "$ENABLE_MACA_STUBS" = "1" ] && [ -d "$MACA_STUBS_PATH" ]; then
  _ld_parts+=("$MACA_STUBS_PATH")
fi

# Keep both the detected root and /opt/maca as supplemental search paths
# because some images expose one as a symlink or compatibility path.
for _p in \
  "$CUCC_PATH/lib" \
  "$CUDA_PATH/lib" \
  "$MACA_PATH/lib" \
  "$MACA_PATH/ompi/lib" \
  "$MACA_PATH/ucx/lib" \
  "$MACA_COMPAT_PATH/lib" \
  "$MACA_COMPAT_PATH/ompi/lib" \
  "$MACA_COMPAT_PATH/ucx/lib" \
  "$MACA_COMPAT_PATH/tools/cu-bridge/lib"; do
  if [ -d "$_p" ]; then
    _ld_parts+=("$_p")
  fi
done
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  _ld_parts+=("$LD_LIBRARY_PATH")
fi
export LD_LIBRARY_PATH="$(IFS=:; echo "${_ld_parts[*]}")"

_path_prefix=("$CUCC_PATH/tools" "$CUCC_PATH/bin" "$MACA_CLANG_PATH" "$MACA_PATH/bin" "$CUDA_PATH/bin")
if [ -d "$MACA_COMPAT_PATH/bin" ]; then
  _path_prefix+=("$MACA_COMPAT_PATH/bin")
fi
if [ -d "$MACA_COMPAT_PATH/tools/cu-bridge/bin" ]; then
  _path_prefix+=("$MACA_COMPAT_PATH/tools/cu-bridge/bin")
fi
export PATH="$(IFS=:; echo "${_path_prefix[*]}"):$PATH"

# Torch on some platform images requires this MACA bridge library to be preloaded.
_tools_ext_candidates=(
  "$MACA_PATH/lib/libToolsExt_cu.so"
  "$MACA_COMPAT_PATH/lib/libToolsExt_cu.so"
)
if [ "$ENABLE_MACA_STUBS" = "1" ]; then
  _tools_ext_candidates+=("$MACA_STUBS_PATH/libToolsExt_cu.so")
fi
for _lib in "${_tools_ext_candidates[@]}"; do
  if [ -f "$_lib" ]; then
    if [ -n "${LD_PRELOAD:-}" ]; then
      export LD_PRELOAD="$_lib:$LD_PRELOAD"
    else
      export LD_PRELOAD="$_lib"
    fi
    break
  fi
done

cd "$PHYS_DIR"

echo "[platform] using python: $PYTHON_BIN"
echo "[platform] project dir: $PHYS_DIR"
echo "[platform] maca path:  $MACA_PATH"
echo "[platform] bridge path: $CUDA_PATH"
echo "[platform] ld preload: ${LD_PRELOAD:-<empty>}"

ensure_ffmpeg_runtime_libs

exec "$PYTHON_BIN" auto_simulation_runner.py "$@"
