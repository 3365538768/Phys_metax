# my_model（Arch4）训练与数据读取用法

本目录包含最小可用的 **Arch4** 训练链路：从 `auto_output` 的扁平样本目录读取数据（优先读取每个样本下的 `arch4_data.lmdb`），然后训练模型回归材料参数，并对 `stress/flow/force_mask` 场进行辅助监督。

## 1. 快速开始

### 1.1 安装依赖

在 `PhysGaussian/` 根目录执行（确保你环境已安装合适的 PyTorch，GPU 可选）：

```bash
pip install -r requirements.txt
pip install torch lmdb opencv-python-headless
```

> 若你系统已经安装了 torch / lmdb，可跳过对应步骤。

### 1.2 先做训练链路冒烟测试（强烈建议）

这一步会依次检查：LMDB 元数据与键、Dataset 取样张量 shape、前向+反传、最后进行少量优化步。

```bash
cd PhysGaussian
python -m my_model.smoke_train_stages \
  --split_root auto_output/dataset_deformation_stress_500_new/train \
  --train_steps 3
```

如果遇到 LMDB 相关错误，优先回看“常见问题”章节。

### 1.3 用 `configs.json` 管理超参（推荐）

编辑 `my_model/configs.json` 后，直接运行：

```bash
cd PhysGaussian
python -m my_model.train --config my_model/configs.json
```

训练时如果你仍通过命令行额外传入 `--lr/--epochs/...` 等参数，会覆盖配置中的对应字段（但通常没必要）。

## 2. 训练指令（从 LMDB 读取）

最小训练入口在 `my_model/train.py`，使用方式如下：

```bash
cd PhysGaussian
python -m my_model.train \
  --split_root auto_output/dataset_deformation_stress_500_new/train \
  --epochs 10 \
  --batch_size 1 \
  --lr 3e-4 \
  --max_views 3 \
  --num_frames 16 \
  --img_size 224 \
  --num_workers 0 \
  --device cuda
```

训练脚本每个 epoch 会打印 `loss`，并根据 `my_model/configs.json` 中的 `train.checkpoint` 自动保存 checkpoint（支持断点续训）。

### 2.2 多卡并行训练（DDP）

当你用 `torchrun --nproc_per_node=N` 启动时，`my_model/train.py` 会自动进入 DDP 模式：
- 数据通过 `DistributedSampler` 分片到各卡
- 每个 epoch 的 loss 会做 all-reduce 后由 rank0 打印

示例：

```bash
cd PhysGaussian
torchrun --nproc_per_node=2 -m my_model.train --config my_model/configs.json
```

### 2.1 关键参数说明

- `--split_root`：扁平样本目录根，如 `auto_output/<dataset>/train`
- `--auto_output`：当 `--split_root` 为相对路径时使用的父目录名（默认 `auto_output`，通常不需要改）
- `--img_size`：输入/监督张量的空间尺寸（必须与 LMDB `__meta__.img_size` 一致；默认 224）
- `--max_views`：每个样本取用的视角上限（Dataset 会 padding/truncation 保证固定为 `max_views`）
- `--num_frames`：时间维重采样帧数（Dataset 会把 LMDB 的帧序列重采样到该值）
- `--lambda_stress / --lambda_flow / --lambda_force`：辅助场监督权重（默认各 0.15）
- `--no_log_scale`：如开启则不对回归参数做 log1p 变换
- `--num_workers`：DataLoader 读取进程数（排查数据读取问题时可先设为 0）

> 参数选择技能文档在 `my_model/skills/参数选择`，可按你的资源/收敛情况微调 `lr` 与辅助损失权重。

## 2.2 模型架构概览（Arch4 / `Arch4VideoMAEPhysModel`）

该模型是一个“视频时空 token 化 + ViT 编码 + 跨视角融合 + 多任务头”的结构。

### 输入与通道

- Dataset 输出输入 `x` 形状为 `[V, 6, T, H, W]`。
- 其中 `6` 通道 = RGB(3) + force_mask(3)（在模型里作为统一的 `in_channels=6` 输入）。

### 时空编码（tubelet + ViT）

1. `VideoTubeletEmbed (Conv3d)`：把输入 `[B*V, C, T, H, W]` 按 `(tubelet_size, patch_size, patch_size)` 切成 3D tubelet，并映射到 token 序列 `[B*V, N, D]`。
2. `VideoViTEncoder`：`depth` 层 `TransformerEncoder` block（每层带 `MultiheadAttention + MLP`），最后对 token 做 `mean pooling` 得到每个视角的向量特征 `[B, V, D]`。

### 跨视角融合（Multi-View Fusion）

- `MultiViewFusion` 将视角特征做 `TransformerEncoder` 融合，并支持两种池化：
  - `use_attention_pool=true`：用一个可学习 query 做注意力池化；
  - 否则直接对 token 做均值池化。

### 多任务输出头

1. 参数回归头 `ParamRegressionHead`：
   - 输出 `param_pred`，形状 `[B, num_targets]`（默认 `num_targets=4`：`E/nu/density/yield_stress`）
   - 可选 `logvar`（当前训练默认 `use_uncertainty=False`，因此 `logvar` 会被置为全零张量）
2. 场监督头 `FieldHead`（默认开启）：
   - `stress_field_pred` / `flow_field_pred` / `force_pred`
   - 形状均为 `[B, V, 3, dec_h, dec_w]`（训练里 `dec_h=dec_w=56`）

默认关键超参来自模型构造函数（训练只显式传入 `num_views/max_views`、`num_frames`、`img_size`、`dec_h/dec_w`、`use_aux_field_heads=True`）。
如需完整配置表，请见 `my_model/configs.json`。

### 2.3 Checkpoint 自动保存

训练默认会由 rank0 在每个 epoch 结束时保存 checkpoint 到：
- `PhysGaussian/output_checkpoints/arch4_train_<timestamp>/`

文件命名包括：
- `epoch_XXXX.pt`：每 `save_every_epochs`（默认 1）个 epoch 保存一次
- `last.pt`：训练过程中的最新权重（默认开启）
- `best.pt`：按训练集全局平均 loss 最小的权重（默认开启）

以上保存行为由 `my_model/configs.json` 中的 `train.checkpoint` 控制，例如：
- `save_dir`：自定义保存目录（`null` 则自动生成 `arch4_train_<timestamp>/`）
- `save_every_epochs`：每 N 个 epoch 保存一次
- `save_last` / `save_best`：是否保存 `last.pt` / `best.pt`

### 2.4 TensorBoard Loss 记录

训练时（rank0）会把 `loss` 写入 TensorBoard：
- 根目录：`PhysGaussian/my_model/tb_logs/arch4_train_<timestamp>_epochs_<total_epochs>/`

标量包括：
- `train/batch_loss`（总 loss）
- `train/batch_loss_reg`（主损失 reg）
- `train/batch_loss_stress`（辅助损失 stress）
- `train/batch_loss_flow`（辅助损失 flow）
- `train/batch_loss_force`（辅助损失 force）
- `train/avg_loss`（总 loss 平均）
- `train/avg_loss_reg`
- `train/avg_loss_stress`
- `train/avg_loss_flow`
- `train/avg_loss_force`

### 2.5 断点续训（继续训练）

在 `my_model/configs.json` 中设置：

- `train.checkpoint.resume_from`：填入你要恢复的 checkpoint 路径（例如 `output_checkpoints/.../last.pt` 或 `best.pt` 或某个 `epoch_XXXX.pt`）

训练启动后会：
- 加载模型权重
- 加载 optimizer 状态
- 从 checkpoint 里记录的 `epoch` 的下一轮继续训练

## 4. train/test 划分（`train_test_split.py`）

对扁平样本目录（`.../train/<id>/` 这种）按 `id` 做随机划分，输出 `train_ids/test_ids`：

```bash
cd PhysGaussian
python -m my_model.train_test_split \
  --split_root auto_output/dataset_deformation_stress_500_new/train \
  --test_ratio 0.2 \
  --seed 42 \
  --out_json train_test_split.json
```

评估时把该 json 里的 `test_ids` 作为子集即可（见下一节 `eval.py`）。

## 5. 多卡并行评估/推理（`eval.py`）

先准备好模型权重（`--weights`），然后运行评估：

单卡：
```bash
cd PhysGaussian
python -m my_model.eval \
  --config my_model/configs.json \
  --weights /path/to/weights.pt \
  --test_ids_json my_model/train_test_split.json
```

多卡（推荐用 torchrun）：
```bash
cd PhysGaussian
torchrun --nproc_per_node=2 -m my_model.eval \
  --config my_model/configs.json \
  --weights /path/to/weights.pt \
  --test_ids_json my_model/train_test_split.json \
  --num_vis 3 \
  --vis_view 0
```

输出目录下会生成：
- `eval_metrics.json`
- `visual_report.md`
- `vis/<sample_id>/`：包含预测的 `stress/flow/force` 可视化 PNG 和 `params.json`

## 3. 数据集格式（对应 auto_output 的样本目录）

`DatasetArch4` 期望 `--split_root` 下存在很多 **数字子目录**：

```text
auto_output/<dataset>/train/
  000000/
    gt.json
    arch4_data.lmdb/
      data.mdb
      __meta__ ...
  000001/
    ...
```

### 3.1 必需文件

每个样本目录至少需要：

- `gt.json`：真值材料参数与动作信息。训练中使用回归字段 `E / nu / density / yield_stress`
- `arch4_data.lmdb/`：LMDB 环境目录（由 `arch4_lmdb.py` 约定格式写入）

### 3.2 Dataset 输出张量约定

对每个样本，Dataset 输出 5 个对象：

- `x`：输入张量 `[V, 6, T, H, W]`
  - 6 通道 = RGB(3) + force_mask(3)
- `stress_gt`：`[V, 3, T, H, W]`
- `flow_gt`：`[V, 3, T, H, W]`
- `force_gt`：`[V, 3, T, H, W]`（来自 `force_mask` 的监督）
- `params_gt`：回归参数向量 `[4]`（E, nu, density, yield_stress）

其中 `V=max_views`、`T=num_frames`、`H=W=img_size`。

## 4. 常见问题

### 4.1 LMDB img_size 不一致

如果报错类似 `LMDB img_size=... 与请求的 img_size=... 不一致`：

- 说明你使用的 `--img_size` 与 LMDB 写入时的 `__meta__.img_size` 不一致
- 解决：把 `--img_size` 改成 LMDB 元数据里记录的值（默认写入通常是 224）

### 4.2 训练没保存权重

`my_model/train.py` 会自动保存 checkpoint（rank0 写盘，DDP 场景下不会重复保存）。

如果你需要：

- 定期保存 `model.pt`
- 最佳模型选择（按验证集指标）
- 断点续训

告诉我你的期望保存目录/命名规则，我可以直接补齐代码并更新 README。

