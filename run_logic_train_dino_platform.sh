#!/usr/bin/env bash
set -euo pipefail

# 最小可用版：仅覆盖单机多卡 DINO 训练
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
export PHYS_DIR="${PHYS_DIR:-$PROJECT_ROOT/Phys}"
export PHYS_TRAIN_ENV="${PHYS_TRAIN_ENV:-$PROJECT_ROOT/envs/train}"
export PYTHON_BIN="${PYTHON_BIN:-$PHYS_TRAIN_ENV/bin/python}"

if [ -n "${CONDA_ENV:-}" ]; then
  _CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
  source "$_CONDA_SH"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="$(command -v python)"
  echo "[logic_train_dino] conda activate: $CONDA_ENV -> PYTHON_BIN=$PYTHON_BIN"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PHYS_DIR"

# 仅保留训练必需环境
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_HOME="${TORCH_HOME:-$PHYS_DIR/.torch}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-4}}"

_DEFAULT_CFG="logic_model/configs/logic_train_dino_dataset_mask_1000.json"
_TRAIN_ARGS=("$@")
if [ "${#_TRAIN_ARGS[@]}" -eq 0 ]; then
  _TRAIN_ARGS=(--config "$_DEFAULT_CFG")
fi

echo "[logic_train_dino] python=$PYTHON_BIN"
echo "[logic_train_dino] cwd=$PHYS_DIR"
echo "[logic_train_dino] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[logic_train_dino] TORCH_HOME=$TORCH_HOME"
echo "[logic_train_dino] nproc_per_node=$NPROC_PER_NODE"
echo "[logic_train_dino] args=${_TRAIN_ARGS[*]}"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m logic_model.train \
  "${_TRAIN_ARGS[@]}"
