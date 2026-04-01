#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
export PHYS_DIR="${PHYS_DIR:-$PROJECT_ROOT/Phys}"
export PHYS_TRAIN_ENV="${PHYS_TRAIN_ENV:-$PROJECT_ROOT/envs/train}"
export PYTHON_BIN="${PYTHON_BIN:-$PHYS_TRAIN_ENV/bin/python}"

if [ -n "${CONDA_ENV:-}" ]; then
  _CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
  source "$_CONDA_SH"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="$(command -v python)"
  echo "[logic_train] conda activate: $CONDA_ENV -> PYTHON_BIN=$PYTHON_BIN"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

# ---- MACA / cu-bridge ----
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_HOME="${MACA_HOME:-$MACA_PATH}"
export CUDA_PATH="${CUDA_PATH:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$CUDA_PATH}"

export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:${LIBRARY_PATH:-}"
export CPATH="$MACA_PATH/include:${CPATH:-}"
export PATH="$MACA_PATH/bin:$MACA_PATH/tools/cu-bridge/bin:${PATH:-}"

if [ -z "${MUSA_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export MUSA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

cd "$PHYS_DIR"

echo "[logic_train] using python: $PYTHON_BIN"
echo "[logic_train] project dir: $PHYS_DIR"
echo "[logic_train] MACA_PATH=$MACA_PATH"
echo "[logic_train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[logic_train] MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-<unset>}"

# ---- distributed env mapping ----
NUM_GPUS="${SENSECORE_ACCELERATE_DEVICE_COUNT:-${NUM_GPUS:-8}}"
NNODES="${SENSECORE_PYTORCH_NNODES:-${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}}"
NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${MLP_ROLE_INDEX:-${MLP_WORKER_RANK:-${RANK:-0}}}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-23456}}"

# ---- align with successful cases ----
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional: only enable on MetaX IB environments after verification
if [ "${ENABLE_MCCL_IB_HCA_HINT:-0}" = "1" ]; then
  export MCCL_IB_HCA="${MCCL_IB_HCA:-mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0}"
fi

if [ "${DIST_DEBUG:-0}" = "1" ]; then
  export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
  export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-/tmp/nccl.%h.%p.log}"
fi

echo "[logic_train] dist env:"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  NNODES=$NNODES"
echo "  NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "  MCCL_IB_HCA=${MCCL_IB_HCA:-<unset>}"

_DIST_LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run)

if [ "$NNODES" -gt 1 ]; then
  exec "${_DIST_LAUNCHER[@]}" \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m logic_model.train \
    "$@"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  exec "${_DIST_LAUNCHER[@]}" \
    --nproc_per_node="$NUM_GPUS" \
    -m logic_model.train \
    "$@"
fi

exec "$PYTHON_BIN" -m logic_model.train "$@"