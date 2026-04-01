#!/bin/bash
set -euo pipefail

########################################
# Usage
# 1) With platform-injected multi-node env vars (recommended):
#    bash sdv1.5_d2o_transformer/pg_sdv15_dinov3_transformer_rope_fresh_multinode.sh "$(pwd)/ckpt/dmd2_sdv15" oetkenhasan7 dmd2_sdv15 sdv15_pg_dinov3_transformer_rope_fresh_bs256 8
#
# 2) With manual multi-node args:
#    bash pg_sdv15_dinov3_transformer_rope_fresh_multinode.sh \
#      "$(pwd)/ckpt/dmd2_sdv15" oetkenhasan7 dmd2_sdv15 \
#      sdv15_pg_dinov3_transformer_rope_fresh_bs256 \
#      8 2 0 10.0.0.1 29500
########################################
if [ $# -lt 3 ]; then
  echo "Usage: $0 CHECKPOINT_PATH WANDB_ENTITY WANDB_PROJECT [WANDB_NAME] [NUM_GPUS] [NNODES] [NODE_RANK] [MASTER_ADDR] [MASTER_PORT]"
  exit 1
fi

export CHECKPOINT_PATH="$1"
export WANDB_ENTITY="$2"
export WANDB_PROJECT="$3"
WANDB_NAME="${4:-sdv15_pg_dinov3_transformer_rope_fresh_bs256}"
NUM_GPUS="${5:-8}"

# Platform env vars take priority; otherwise use manual args.
NNODES="${MLP_WORKER_NUM:-${6:-1}}"
NODE_RANK="${MLP_ROLE_INDEX:-${MLP_WORKER_RANK:-${7:-0}}}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-${8:-127.0.0.1}}"
MASTER_PORT="${MLP_WORKER_0_PORT:-${9:-29500}}"

export WANDB_API_KEY=06018bf407288ef05f04cd2a0e13944b220d34c7

CODE_DIR="/mnt/afs/lixiaoou/DMD2-tencent"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

CONDA_BASE="/opt/conda"
DMD_ENV="/mnt/afs/lixiaoou/envs/dmd2_mx4"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${DMD_ENV}"

wandb offline

export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NUM_GPUS}" \
  sdv1.5_d2o_transformer/train_sd.py \
    --train_iters 100000 \
    --batch_size 8 \
    --initialie_generator \
    --num_workers 0 \
    --seed 10 \
    --resolution 512 \
    --pg_interp 224 \
    --latent_resolution 64 \
    --generator_lr 1e-6 \
    --guidance_lr 1e-5 \
    --max_grad_norm 10.0 \
    --log_iters 500 \
    --wandb_iters 50 \
    --log_loss \
    --use_fp16 \
    --gradient_checkpointing \
    --model_id "${CHECKPOINT_PATH}/stable-diffusion-v1-5" \
    --real_image_path "${CHECKPOINT_PATH}/data/sd_vae_latents_laion_500k_lmdb/" \
    --train_prompt_path "${CHECKPOINT_PATH}/data/captions_laion_score6.25.pkl" \
    --output_path "${CHECKPOINT_PATH}/pg_sdv15_dinov3_transformer_bs256" \
    --dfake_gen_update_ratio 1 \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    --projected_gan \
    --pg_backbones dinov3_vit_base \
    --pg_backbone_paths /mnt/afs/lixiaoou/DMD2-tencent/ckpt/dmd2_sdv15/pretrain/dinov3-vitb16-pretrain-lvd1689m \
    --pg_num_discs 4 \
    --pg_proj_type 2 \
    --pg_cout 96 \
    --pg_head_type transformer \
    --pg_head_dim 384 \
    --pg_head_num_heads 6 \
    --pg_pe_type rope_fresh \
    --pg_no_diffaug
