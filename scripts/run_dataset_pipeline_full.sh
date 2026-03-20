#!/usr/bin/env bash
# 完整数据集流水线：批量仿真（auto_simulation_runner）→ 扁平数据集（transform_dataset）
# 用法：在 PhysGaussian 根目录执行
#   bash scripts/run_dataset_pipeline_full.sh
# 或：
#   bash scripts/run_dataset_pipeline_full.sh 4    # 第二个参数为 GPU 数量

set -euo pipefail

PG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PG_ROOT"

NUM_GPUS="${1:-${NUM_GPUS:-1}}"

MODEL_PATH="${MODEL_PATH:-model}"
MATERIAL_SPACE="${MATERIAL_SPACE:-configs_auto/material_space_example.json}"
TRAIN_CFG="${TRAIN_CFG:-configs_auto/train_config_dataset_full.json}"

AUTO_OUT="${AUTO_OUT:-$PG_ROOT/auto_output}"
OUT_NAME="${OUT_NAME:-dataset_400}"
LAYOUT="${LAYOUT:-train_test}"
TEST_SUBSTR="${TEST_SUBSTR:-an_empty_aluminum_can}"
SEED="${SEED:-42}"

echo "=== [1/2] 批量仿真 ==="
echo "  PG_ROOT=$PG_ROOT"
echo "  model_path=$MODEL_PATH  train_config=$TRAIN_CFG  num_gpus=$NUM_GPUS"
python auto_simulation_runner.py \
  --model_path "$MODEL_PATH" \
  --material_space_config "$MATERIAL_SPACE" \
  --train_config "$TRAIN_CFG" \
  --num_gpus "$NUM_GPUS"

echo ""
echo "=== [2/2] 生成扁平数据集（含辅助与体积场）==="
python transform_dataset.py \
  --auto_output "$AUTO_OUT" \
  --out_name "$OUT_NAME" \
  --layout "$LAYOUT" \
  --test_substr "$TEST_SUBSTR" \
  --seed "$SEED" \
  --copy_aux \
  --copy_volumetric_fields

echo ""
echo "完成: $AUTO_OUT/$OUT_NAME"
echo "  train/ test/ 下每样本含: images/, gt.json, boundary_conditions.json"
echo "  --copy_aux: run_parameters.json, stress_heatmaps/, tracks_2d/（若仿真已生成）"
echo "  --copy_volumetric_fields: stress_field/, deformation_field/（若仿真已生成）"
