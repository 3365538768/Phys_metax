WORK_POSTFIX=${1}

export CUDA_LAUNCH_BLOCKING=1

CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CURR_FILE_DIR}/task_model_config_mx.py
WORK_DIR=/mnt/afs/liyu1/workdir/qworld/c500_multi_0318_N10/
CODE_DIR=/mnt/afs/liyu1/qworld/
# 环境安装标记文件
# ENV_CHECK_FILE="${WORK_DIR}/.env_setup_done"

seed=42

echo "============================================================"
echo "训练启动"
echo "============================================================"


# --- 自动初始化 Conda 环境 ---
export CONDA_ROOT="/opt/conda"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate base
    echo "[INFO] Successfully activated conda base environment."
else
    # 万一 profile 脚本不存在，直接把 bin 目录加入 PATH 也能解决大部分问题
    export PATH="$CONDA_ROOT/bin:$PATH"
    echo "[WARN] conda.sh not found, added /opt/conda/bin to PATH instead."
fi

# 检查路径是否正确
echo "Python path: $(which python3)"
echo "Pip path: $(which pip)"

# # --- 环境检查与增量安装/补丁逻辑 ---
# if [ ! -f "$ENV_CHECK_FILE" ]; then
#     echo "[INFO] 首次运行，开始配置环境与补丁..."


#     pip install fvcore
#     conda install ffmpeg

#     SDK_PATH="/mnt/afs/liyu1/petrel-oss-sdk-v2.2.2-3-g410c04c-master"
#     if [ -d "$SDK_PATH" ]; then
#         echo "Installing Petrel OSS SDK..."
#         pushd $SDK_PATH > /dev/null
#         pip install .
#         popd > /dev/null
#     else
#         echo "[ERROR] SDK path $SDK_PATH not found!"
#         exit 1
#     fi

#     SRC_FILE="/mnt/afs/liyu1/qworld/qworld/pipelines/diffsynth/_impl/fla_local/layers/gated_deltanet_with_tp.py"
#     DST_DIR="/opt/conda/lib/python3.10/site-packages/fla/layers/"
    
#     if [ -f "$SRC_FILE" ]; then
#         echo "Applying patch: Copying gated_deltanet_with_tp.py to $DST_DIR"
#         mkdir -p "$DST_DIR"
#         cp "$SRC_FILE" "$DST_DIR"
#     else
#         echo "[ERROR] Patch source file not found: $SRC_FILE"
#         exit 1
#     fi

#     # 创建标记文件
#     touch "$ENV_CHECK_FILE"
#     echo "[SUCCESS] 环境与补丁配置完成。"
# else
#     echo "[INFO] 检测到环境已就绪，跳过安装与补丁步骤。"
# fi
# ------------------------------

echo "Config: task_model_config_mx.py"
echo "GPU: 0-7 (8 卡 H20)"
echo "WORK_DIR: ${WORK_DIR}"
echo "============================================================"

mkdir -p ${WORK_DIR}

cp /mnt/afs/liyu1/petreloss.conf /root/

# 执行核心训练脚本
bash ${CURR_FILE_DIR}/train_core_mx_multi.sh ${CONFIG} ${WORK_DIR} ${CODE_DIR}
