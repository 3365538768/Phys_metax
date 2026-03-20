# PhysGaussian 物理参数预测模型

根据时序形变图像预测物理参数（E, nu, density, yield_stress）和动作类型。

## 模型结构

1. **Image Encoder**: ResNet18 提取每帧特征
2. **Transformer**: 时序建模
3. **物理参数头**: 回归 E, nu, density, yield_stress
4. **动作头**: 分类 bend, drop, press, shear, stretch

## 数据集

- 路径: `PhysGaussian/auto_output`
- 结构: `auto_output/<action>/<OBJECT__PARAMS__action>/<obj>/images/<camera_name>/*.png`
- 按 7:3 分割训练集和测试集

## 使用

### 安装依赖

```bash
pip install -r my_model/requirements.txt
```

### 训练

```bash
cd PhysGaussian
python -m my_model.train --epochs 30 --batch_size 8
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

### 测试

```bash
python -m my_model.eval --checkpoint my_model/checkpoints/final.pt
```

### 测试 + 可视化（自动生成图）

```bash
python -m my_model.eval --checkpoint my_model/checkpoints/final.pt
```

每次运行会在 `my_model/visualizations/<YYYYMMDD_HHMMSS>/` 下保存当次图表（避免覆盖）。

### 自动选择空闲 GPU（评估阶段）

```bash
python -m my_model.eval --checkpoint my_model/checkpoints/final.pt --auto_gpu
```
