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
| `compile_video` | bool | 是否 ffmpeg 合成 mp4 |
| `num_views` | int | 相机方位角数量 |
| `num_render_views` | int | `-1` 表示同 `num_views` |
| `num_render_timesteps` | int | `0` 表示每仿真帧都输出渲染/热力图等 |
| `white_bg` | bool | 白底渲染 |
| `debug` | bool | 传给 `modified_simulation` |
| `output_deformation` | bool | 写 `deformation_field/`（大） |
| `output_stress` | bool | 写 `stress_field/`（大） |
| `field_output_interval` | int | 体积场输出间隔（仿真帧） |
| `output_view_stress_heatmap` | bool | `stress_heatmaps/`，与 RGB 同视角 |
| `output_view_tracks2d` | bool | `tracks_2d/`（若未被代码内临时开关关闭） |
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
| `--copy_aux` | 关 | 复制 `run_parameters.json`、`stress_heatmaps/`、`tracks_2d/` |
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
  --multiview \
  --width 1280 --height 960
```

- **多视角默认不写 mp4**（只输出 `stress_pcd_multiview/<视角名>/0000.png`…），避免段错误与空视频；需要 mp4 时在**同一行**加 **`--ffmpeg`**（并保留 `--multiview`）。
- **`--no_ffmpeg`**：连单视角应力、轨迹也全部不写 mp4。
- 重新跑仿真后 `meta/stress_pcd_cameras.json` 为 **v3**，含 **`mpm_to_world`**（与 `undo_all_transforms` 一致）、`camera_motion`、`observant_coordinates`。`visualize_fields` 会先把 `deformation_field` 变到与 **3DGS 相同的世界系**，再用与仿真相同的 `look_at` / 轨道相机 / 竖向 FOV，避免「点云在 MPM、相机在世界」导致的偏心、缩放错误。
- `move_camera=True` 时仍按 `deformation_frame_XXXX` 的帧号逐帧对齐 eye（与 RGB 一致）。若仅有旧 meta（无 `mpm_to_world`），多视角会退回 **MPM 系** 相机并打印警告。
- 若存在 `cameras.json` 且 `default_camera_index>-1`，JSON 内 `warnings` 会提示与 RGB 多视角可能不一致。

---

## 磁盘说明

`train_config_dataset_full.json` 中 `output_deformation` / `output_stress` 为 **true** 时体积很大；若只要 RGB + 应力热力图，可改为 `false`，并仍使用 `--copy_aux`。
