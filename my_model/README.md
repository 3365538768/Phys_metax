# PhysGaussian 物理参数预测模型

根据时序形变图像预测物理参数（E, nu, density, yield_stress）和动作类型。

## 模型结构

1. **Image Encoder**: ResNet18 提取每帧特征  
2. **Transformer**: 时序建模  
3. **物理参数头**: 回归 E, nu, density, yield_stress  
   - `--arch 1`：单一 MLP → 4 维  
   - `--arch 2`：4 个独立小 MLP，各输出 1 维  
4. **动作头**: 分类 bend, drop, press, shear, stretch  

**产物目录**：`checkpoints/<arch>/`、`runs/<arch>/`、`visualizations/<arch>/`（详见 `rules/history.txt`）。

**Checkpoint 命名**：中间与最终权重会带配置标签，便于区分 `num_frames` 等实验，例如  
`epoch_100__T20_H224_bs8_lr0.0001_ep30_seed42_ds400.pt`、`final__T20_H224_....pt`。  
同一次训练结束仍会复制一份 **`final.pt`**（内容与带标签的 `final__...pt` 相同），便于默认 `eval`。  
`.pt` 内新增 **`train_hparams`** 字段（含 `num_frames`、`img_size`、`ckpt_config_slug` 等）。

训练时每个 epoch 会打印耗时，并写入 TensorBoard（`train/epoch_seconds`）、  
`visualizations/<arch>/train_epoch_seconds__<标签>.png` 与 `checkpoints/<arch>/epoch_time_log__<标签>.json`。

**混合精度（BF16）**：在支持 BF16 的 NVIDIA GPU（如 Ampere 及以上）上可加 `--amp_bf16`，使用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`；一般无需 GradScaler。示例：`python -m my_model.train --amp_bf16 ...`

训练时默认每个 epoch 显示 **tqdm 进度条**（当前 batch loss）；不需要可加 `--no_progress`。启动时会打印 **每 epoch 的 step 数**（随 `batch_size` 变化）；第 1 个 epoch 结束后打印 **CUDA 峰值显存**。

## 数据集

**推荐（rules/task）**：`auto_output/dataset_400/`（由仓库根目录 `transform_dataset.py` 生成）

- `dataset_400/train`：训练（铝罐样本已排除）
- `dataset_400/test`：测试（仅 `an_empty_aluminum_can` 相关样本）

训练/评估时加 `--data_layout dataset_400`（**默认值**）。扁平集根目录默认 `auto_output/dataset_400/`，可用 **`--dataset_dir`** 改为其它名（相对 `auto_output`）或绝对路径（须含 `train/`、`test/`）。

**旧版**：`auto_output/<action>/...` 树状结构，使用 `--data_layout legacy`，并按 7:3 随机划分。

## 使用

### 安装依赖

```bash
pip install -r my_model/requirements.txt
```

### 训练

```bash
cd PhysGaussian
# dataset_400 + 架构 1（默认）
python -m my_model.train --epochs 30 --batch_size 8 --arch 1
# 架构 2（多物理头）
python -m my_model.train --epochs 30 --batch_size 8 --arch 2
```

### 多卡训练（自动选空闲 GPU）

传入 `--gpus N` 后脚本会：
- 通过 `nvidia-smi` 按 GPU 利用率/显存占用比例挑选最空闲的 N 张卡
- 自动设置 `CUDA_VISIBLE_DEVICES`
- 自动用 `torchrun` 以 DDP 方式启动

示例：

```bash
python -m my_model.train --gpus 4 --epochs 30 --batch_size 8
```

多卡若**几乎不比单卡快**，多半是 **PNG 解码 + 磁盘/NFS** 饱和（4 进程 × 多 worker 同时读图）。脚本在 DDP 下默认 `--num_workers` 较低；可把 `dataset_400` 放到本地 SSD，或尝试 `--num_workers 0` / 增大 `batch_size` 提高 GPU 计算占比。启动时会打印 `world_size`、每 epoch 步数和等效全局 batch，便于确认 DDP 已生效。

### 测试（dataset_400 的 test 划分）

```bash
python -m my_model.eval --arch 1
python -m my_model.eval --arch 2 --checkpoint my_model/checkpoints/2/final.pt
# 或指定带标签的权重（与训练时 num_frames 等一致）：
# python -m my_model.eval --arch 1 --num_frames 20 --checkpoint my_model/checkpoints/1/final__T20_H224_bs8_lr0.0001_ep30_seed42_ds400.pt
```

### 测试 + 可视化（自动生成图 + metrics.json）

```bash
python -m my_model.eval --arch 1
```

每次运行会在 `my_model/visualizations/<arch>/<YYYYMMDD_HHMMSS>/` 下保存图表与 `metrics.json`。

### 自动选择空闲 GPU（评估阶段）

```bash
python -m my_model.eval --arch 1 --auto_gpu
```
