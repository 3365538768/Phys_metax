# 完整参数数据集输出说明

## 一键脚本（推荐）

在 **PhysGaussian 根目录**：

```bash
bash scripts/run_dataset_pipeline_full.sh
# 多卡：bash scripts/run_dataset_pipeline_full.sh 4
```

环境变量可覆盖默认（均在执行前 `export`）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `MODEL_PATH` | `model` | 含 `.ply` 的目录 |
| `MATERIAL_SPACE` | `configs_auto/material_space_example.json` | 材质空间 JSON |
| `TRAIN_CFG` | `configs_auto/train_config_dataset_full.json` | 完整 train_config |
| `AUTO_OUT` | `<仓库>/auto_output` | `transform_dataset --auto_output` |
| `OUT_NAME` | `dataset_400` | 输出子目录名 |
| `LAYOUT` | `train_test` | `train_test` 或 `by_model` |
| `TEST_SUBSTR` | `an_empty_aluminum_can` | 含该子串的物体进 test |
| `SEED` | `42` | 训练集打乱种子 |
| `NUM_GPUS` | `1` | 未传参时与脚本第一个参数一致 |

---

## `train_config` 全字段（`auto_simulation_runner`）

JSON 示例：`configs_auto/train_config_dataset_full.json`。

| 键 | 类型 | 说明 |
|----|------|------|
| `seed` | int | 随机种子（PLY / 动作 / 材质 / 连续参数） |
| `num_simulations` | int | 任务总数 |
| `sim_types` | string[] | 每个 job 随机抽一种 |
| `output_root` | string | 仿真输出根，默认 `auto_output` |
| `output_layout` | string | `by_model` 或 `by_action` |
| `render_img` | bool | 是否输出多视角 RGB |
| `compile_multiview_videos` | bool | **推荐**：统一控制 `images/`、`stress_gaussian/`、`tracks_gaussian/`、`flow_gaussian/` 的 mp4；若开 `output_subsampled_world_tracks`，另合成 `videos/tracks_subsampled_world_ortho_*.mp4`（正交折线，非光栅化）。 |
| `delete_png_sequences_after_compile_video` | bool | 与 compile 同开时：某序列 **ffmpeg 成功** 后删除该 PNG 目录；默认 `false` 保留。 |
| `compile_video` | bool | 未写 `compile_multiview_videos` 时沿用此项（默认 true）。 |
| `num_views` | int | 相机方位角数量 |
| `num_render_views` | int | `-1` 表示同 `num_views` |
| `render_outputs_per_sim_second` | float | `>0`：统一 N，K≈仿真总秒数×N，均匀采样；视频帧率=N。`≤0` 时用 `num_render_timesteps`。 |
| `num_render_timesteps` | int | `0` 表示每仿真帧都输出（未启用上一项时） |
| `white_bg` | bool | 白底渲染 |
| `debug` | bool | 传给 `modified_simulation` |
| `output_deformation` | bool | 写 `deformation_field/`（大） |
| `output_stress` | bool | 写 `stress_field/`（大） |
| `field_output_interval` | int | 体积场输出间隔（仿真帧） |
| `output_view_stress_heatmap` | bool | `stress_heatmaps/`，与 RGB 同视角 |
| `output_view_stress_gaussian` | bool | `stress_gaussian/`，第二次 3DGS 应力渲染，与 RGB 同视角 |
| `stress_gaussian_colormap_steps` | int | JET 阶梯档数（默认 24） |
| `stress_gaussian_vm_pct_low` / `high` | float | 每帧 σ_vm 绝对值色标分位（默认 1 / 99） |
| `output_view_flow_gaussian` | bool | `flow_gaussian/`（PNG 稠密 flow 伪彩）；参数见 `flow_gaussian_*`（下采样、距离/不透明权重、伪彩饱和） |
| `no_volumetric_stress_deformation` | bool | 为 true 时不写体积 npz，仍可渲染 |
| `output_bc_info` | bool | `meta/` 边界条件等 |
| `output_force_info` | bool | `meta/` 外力等 |
| `base_config_by_sim_type` | object | 各 `sim_type` → 模板 config 路径 |

临时合并的仿真 config 写在 `output_root/_tmp_configs/`。

---

## `transform_dataset.py` 全参数

```bash
python transform_dataset.py \
  --auto_output /path/to/PhysGaussian/auto_output \
  --out_name dataset_400 \
  --layout train_test \
  --test_substr an_empty_aluminum_can \
  --seed 42 \
  --copy_aux \
  --copy_volumetric_fields
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--auto_output` | `PhysGaussian/auto_output` | 扫描仿真根目录 |
| `--out_name` | `dataset_400` | 输出为 `auto_output/<out_name>/` |
| `--layout` | `train_test` | `train_test` 或 `by_model` |
| `--test_substr` | 铝罐子串 | `object_slug` 含此串 → test |
| `--seed` | `42` | 训练集 shuffle |
| `--copy_aux` | 关 | 复制 `run_parameters.json`、`stress_heatmaps/`、`flow_gaussian/` 等 |
| `--copy_volumetric_fields` | 关 | 复制 `stress_field/`、`deformation_field/` |

---

## 仅整理已有仿真（不跑 runner）

```bash
cd PhysGaussian
python transform_dataset.py \
  --auto_output auto_output \
  --out_name dataset_400 \
  --layout train_test \
  --copy_aux \
  --copy_volumetric_fields
```

---

## 多视角点云应力图（与 Gaussian 相机一致）

仿真在开启 `--output_deformation` 或 `--output_stress` 时，会在 `meta/stress_pcd_cameras.json` 写入与 `modified_simulation` 中 **synthetic 轨道相机**（`get_camera_position_and_rotation`）一致的 `look_at` / 各视角 `eye` / `world_up` / 默认 60° 竖向 FOV，方位角分布与 `--num_views`（或 `--num_render_views`）相同。

离线渲染（与 `my_utils/visualize_fields.py` 相同的点云 + von Mises 着色，非 Gaussian splat）：

```bash
cd PhysGaussian
python my_utils/visualize_fields.py \
  --simulation_dir <某次 run 目录> \
  --output_path <可视化输出目录> \
  --multiview
```

- 若存在 **`images/`**，默认从首张 `images/<视角>/0000.png`（或 `0001`…）读取 **宽×高**，使应力/轨迹与 3DGS **视场与距离感**一致；若要坚持 `--width`/`--height`，加 **`--no_match_images_size`**。
- 应力：全序列 **log(von Mises)** 后按 **全局 min/max** 映射到 **TURBO/JET** 伪彩色（对比更强）；离群多时用 **`--stress_vmin_pct 1 --stress_vmax_pct 99`** 改成分位数范围。
- 轨迹：**默认不渲染**；需要 `deformation_trajectory.mp4` 时加 **`--trajectory`**。渲染时 **下采样** 至多 `--traj_max_tracks`（默认 4000）条粒子折线；需要每帧稠密点云时加 **`--traj_show_points`**。
- **多视角默认不写 mp4**（只输出 `stress_pcd_multiview/<视角名>/0000.png`…），避免段错误与空视频；需要 mp4 时在**同一行**加 **`--ffmpeg`**（并保留 `--multiview`）。
- **`--no_ffmpeg`**：连单视角应力、轨迹也全部不写 mp4。
- 重新跑仿真后 `meta/stress_pcd_cameras.json` 为 **v3**，含 **`mpm_to_world`**（与 `undo_all_transforms` 一致）、`camera_motion`、`observant_coordinates`。`visualize_fields` 会先把 `deformation_field` 变到与 **3DGS 相同的世界系**，再用与仿真相同的 `look_at` / 轨道相机 / 竖向 FOV，避免「点云在 MPM、相机在世界」导致的偏心、缩放错误。
- `move_camera=True` 时仍按 `deformation_frame_XXXX` 的帧号逐帧对齐 eye（与 RGB 一致）。若仅有旧 meta（无 `mpm_to_world`），多视角会退回 **MPM 系** 相机并打印警告。
- 若存在 `cameras.json` 且 `default_camera_index>-1`，JSON 内 `warnings` 会提示与 RGB 多视角可能不一致。

### 原生 3DGS 应力热力（与 `images/` 光栅化一致）

**仿真内（推荐）**：在跑 `modified_simulation` 时加 **`--output_view_stress_gaussian`**（可与 `--render_img` 同开）。会在 **`stress_gaussian/<视角名>/`** 下写出与 RGB 同相机、同帧编号的 PNG；**不修改** MPM 状态与高斯可学习参数，仅多一次光栅化。着色按 **von Mises 绝对值线性** 映射，默认用 **1%–99% 分位数** 框定色标，**高对比阶梯 JET**；可选 **`--stress_gaussian_colormap_steps`**、**`--stress_gaussian_vm_pct_low` / `--stress_gaussian_vm_pct_high`**、**`--stress_gaussian_opa_*`**、**`--stress_gaussian_sh_blend`**。批量任务在 `train_config` 里设 **`output_view_stress_gaussian: true`**（及可选 `stress_gaussian_colormap_steps`、`stress_gaussian_vm_pct_*`）。

**离线后处理**：与仿真内相同的光栅化路径：**von Mises 绝对值** 线性色标 + 阶梯 JET + SH 混合 + 不透明度调制。全序列色标由 **`stress_field/`** 统计 min/max 或 **`--stress_vmin_pct` / `--stress_vmax_pct`** 分位数。需 **checkpoint**、**`--config`**、含 **`mpm_to_world`** 的 `meta/stress_pcd_cameras.json`。

```bash
cd PhysGaussian
python my_utils/render_stress_gaussian.py \
  --model_path <与训练/仿真相同> \
  --config <scene json> \
  --simulation_dir <某次 run 目录> \
  --output_path <可视化输出目录>
```

输出：`stress_gaussian_multiview/<视角名>/<帧>.png`。可选 `--white_bg`、`--stress_vmin_pct/--stress_vmax_pct`、`--opa_floor/--opa_ceil`。

---

## 磁盘说明

`train_config_dataset_full.json` 中 `output_deformation` / `output_stress` 为 **true** 时体积很大；若只要 RGB + 应力热力图，可改为 `false`，并仍使用 `--copy_aux`。
