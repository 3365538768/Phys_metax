#!/bin/bash

CONFIG=${1}
WORK_DIR=${2}

CODE_DIR=${3}

cd ${CODE_DIR}

echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

which python
which pip

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
export COSMOS_PREDICT2_ARGS='--checkpoints /mnt/afs/liyu1/git_models'
export QWORLD_HF_CHECKPOINTS_ROOT=/mnt/afs/liyu1/git_models
export QWORLD_FILE_CLIENT_NAME=tos

export IMAGINAIRE_OUTPUT_ROOT=$WORK_DIR


echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

TORCH_EXTENSIONS_DIR=/mnt/afs/liyu1/rely/tmp/tmp_deepspeed

python -c "import os;print('os.environ:',os.environ)"

export nproc_per_node=8
echo "nproc_per_node: $nproc_per_node"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export IS_METAX=1


export MACA_HOME=/opt/maca
export MACA_PATH=/opt/maca
export MACA_DIR=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH



export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

export TOTAL_PROCESSES=$(($WORLD_SIZE * $nproc_per_node))
export MCCL_IB_HCA=mlx5_0:0,mlx5_1:0,mlx5_4:0,mlx5_5:0

echo "ENV || MASTER_ADDR = $MASTER_ADDR"
echo "ENV || MASTER_PORT = $MASTER_PORT"
echo "ENV || WORLD_SIZE  = $WORLD_SIZE"
echo "ENV || RANK        = $RANK"

export NCCL_P2P_DISABLE=0

# 4. 确保工作目录存在
if [ ! -d "${WORK_DIR}" ]; then
    mkdir -p "${WORK_DIR}"
fi

/opt/conda/bin/accelerate launch \
    --config_file /mnt/afs/liyu1/qworld/configs/ds_yamls/accelerate_config_ht_zero1.yaml \
    --multi_gpu \
    --num_processes $TOTAL_PROCESSES \
    --num_machines $WORLD_SIZE \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    ${CODE_DIR}/tools/train_use_deepspeed.py \
        --config=${CONFIG} \
        --work_dir=${WORK_DIR} 2>&1 | tee ${WORK_DIR}/train_${RANK}.log
