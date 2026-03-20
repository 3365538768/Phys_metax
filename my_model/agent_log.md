# my_model agent log

## 2026-03-19

### Issue
- **训练报错**：`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation ...`
- **触发点**：`my_model/model.py` 的 `forward()` 中对 `params_pred[:, i] = ...` 的切片赋值（inplace）可能触发 autograd 版本冲突（尤其是当张量来自 view/stride 操作时）。

### Fix
- **移除 inplace 切片写入**：将
  - `params_pred = params_raw.clone()`
  - `params_pred[:, 0] = ...` 等
  替换为先生成各分量 `e/nu/density/yield_stress`，再用 `torch.stack([...], dim=1)` 组合成 `params_pred`。
- **补充修复**：`my_model/train.py` 中 `_log_params()` 也存在 `out[:, i] = ...` 的 inplace 写入（对带梯度张量做 view 切片赋值）。已改为 `torch.stack` 形式，避免 inplace。
- **诊断开关**：新增 `--detect_anomaly`，可在训练命令中开启 `torch.autograd.set_detect_anomaly(True)` 精确定位残留问题。

### Files changed
- `my_model/model.py`
- `my_model/train.py`
- `my_model/dataset.py`

### Expected outcome
- `loss.backward()` 不再因 inplace 修改触发 autograd 报错，可正常训练。

### Additional issue
- **数据读取报错**：`moov atom not found` / `无法打开视频` 导致 DataLoader worker 抛异常中断训练。

### Additional fix
- 在 `PhysGaussianDataset` 初始化阶段增加 `is_video_readable()` 过滤不可读/损坏视频。
- 在 `__getitem__` 中遇到单个坏样本时自动跳过（最多重试 3 次），避免训练中断。

### Update
- 按需求切换为**不读取视频**：改为从 `images/` 目录直接读取时序帧图片（均匀采样到 `num_frames`）。
- 数据过滤逻辑同步更新为 `find_images_dir()` + `is_image_sequence_readable()`。

