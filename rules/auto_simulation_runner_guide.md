# auto_simulation_runner 使用说明

本文档描述 `PhysGaussian/auto_simulation_runner.py` 的完整用法：批量调用 `modified_simulation.py`，对多个 PLY 在随机材质与动作组合下跑 MPM + 高斯渲染。

---

## 1. 作用与流程

1. 从 **PLY 目录** 收集所有 `.ply`。
2. 读取 **材质参数空间** JSON，按材质类型（jelly / metal / plasticine / sand 等）定义各物理量的采样范围。
3. 读取 **train_config** JSON，生成 `num_simulations` 个 **Job**：每个 Job 随机选一个 PLY、一个 `sim_type`、一种材质类型、再在该材质的参数空间中逐项采样。
4. 对每个 Job：合并 **该 sim_type 的模板场景 JSON** 与采样到的材质参数 → 写入临时 config → 子进程执行  
   `python modified_simulation.py --ply_path ... --config ... --output_path ...`（及一系列开关）。
5. 输出目录由 `output_layout` 决定（默认 **按物体名分一级目录**，见下文）。

**依赖**：在 **PhysGaussian 仓库根目录** 下运行（或保证 `modified_simulation.py`、config 路径、`gaussian-splatting` 等相对路径正确）；需要 **CUDA**；合成 mp4 需系统安装 **ffmpeg**（否则仅 PNG）。

---

## 2. 命令行

在仓库根目录执行：

```bash
python auto_simulation_runner.py \
  --model_path /path/to/ply_folder \
  --material_space_config configs_auto/material_space_example.json \
  --train_config configs_auto/train_config_example.json \
  [--num_gpus 1]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model_path` | 是 | 含一个或多个 `.ply` 的目录；每次仿真随机抽一个文件。 |
| `--material_space_config` | 是 | 材质参数空间 JSON（见第 4 节）。 |
| `--train_config` | 是 | 批量运行与输出选项 JSON（见第 3 节）。 |
| `--num_gpus` | 否 | 并行 GPU 数。`1`（默认）单卡顺序执行；`0` 为自动检测可见 GPU 数量；`>1` 时每张卡一个队列，任务按 `job_idx % num_gpus` 分配。 |

**多卡说明**：并行时子进程通过 `env CUDA_VISIBLE_DEVICES=<gpu_id>` 绑定卡号 **0,1,…,num_gpus-1**（指你启动前环境里「可见」的 GPU 序号）。若需指定物理卡，请先在 shell 里设置好 `CUDA_VISIBLE_DEVICES` 再运行。

**环境**：主进程会把当前 PyTorch 的 `lib` 目录加入子进程 `LD_LIBRARY_PATH`，避免 `diff_gaussian_rasterization` 等找不到 `libc10.so`。

---

## 3. train_config.json（完整字段）

路径可为相对路径（相对于**当前工作目录**，建议在仓库根执行）。

### 3.1 必填

| 键 | 类型 | 说明 |
|----|------|------|
| `base_config_by_sim_type` | `object` | 键为 `bend` / `drop` / `press` / `shear` / `stretch`，值为**该动作模板**场景 JSON 的路径（如 `config/press_cube_jelly.json`）。Job 里出现的每个 `sim_type` 都必须有对应项。 |

### 3.2 批量与随机

| 键 | 默认 | 说明 |
|----|------|------|
| `num_simulations` | `10` | 生成的仿真任务总数。 |
| `sim_types` | 五种动作列表 | 每个 Job 从中随机选一个。 |
| `seed` | `0` | `random.Random(seed)`，控制 PLY / 动作 / 材质 / 连续参数的随机性。 |

### 3.3 输出根路径与目录布局

| 键 | 默认 | 说明 |
|----|------|------|
| `output_root` | `PhysGaussian/auto_output`（脚本所在目录下的 `auto_output`） | 所有仿真输出的根目录；临时合并 config 写在 `output_root/_tmp_configs/`。 |
| `output_layout` | `"by_model"` | **`by_model`**：`output_root/<PLY主文件名>/<NNNN>/`，`NNNN` 为该物体下第几条仿真（四位数字 `0000` 起）。<br>**`by_action`**：`output_root/<sim_type>/<PLY主文件名>/<NNNN>/`。 |

每个仿真目录内会生成 **`gt_parameters.json`**（`ply_stem`、`sim_type`、`material_params`、`job_index` 等），物理参数以该文件为准，不再编码在长目录名里。

Runner 调用 `modified_simulation` 时传入 **`--ply_flat_output`**，仿真产物（`images/`、`meta/` 等）直接写在上述 `<NNNN>/` 下，不再多嵌一层 ply 文件名。

### 3.4 渲染与视频

| 键 | 默认 | 说明 |
|----|------|------|
| `render_img` | `true` | 是否输出多视角 PNG。 |
| `compile_multiview_videos` | 不填则看 `compile_video` | **统一开关**：`images/`、`stress_gaussian/`、`tracks_gaussian/` 的 PNG 写完后是否合成对应 mp4（`videos/<视角>.mp4`、`*_stress_gaussian.mp4`、`*_tracks_gaussian.mp4`）。需 ffmpeg。 |
| `compile_video` | `true` | 仅当 **未** 在 json 里写 `compile_multiview_videos` 时生效（向后兼容）。 |
| `delete_png_sequences_after_compile_video` | `false` | 与 compile 同开：每个 mp4 **ffmpeg 成功** 后删除对应 PNG 序列目录；默认保留 PNG。 |
| `num_views` | `1` | 方位角均匀分布的相机个数；俯仰角等来自模板 config 的相机段。 |
| `num_render_views` | 不填则等价 `-1` | 传给 `modified_simulation`；`-1` 表示与 `num_views` 相同；否则覆盖渲染视角数。 |
| `render_outputs_per_sim_second` | `0` | **统一 N**：`>0` 时目标张数 `K≈round(仿真总秒数×N)`，在时间上均匀采样；`images` / `stress_gaussian` / `flow_gaussian` / `tracks_gaussian` 等同序；`compile` 时 ffmpeg 帧率亦为 **N**，视频时长≈仿真时长。`≤0` 时看 `num_render_timesteps`。 |
| `num_render_timesteps` | `0` | 仅当 `render_outputs_per_sim_second≤0`：`0` 或 ≥ 仿真总帧则每步输出；否则均匀取 K 帧。 |
| `white_bg` | `false` | 白底渲染。 |
| `debug` | `false` | 传给 `modified_simulation`。 |

### 3.5 场输出与体积数据

| 键 | 默认 | 说明 |
|----|------|------|
| `output_deformation` | `false` | 是否写 `deformation_field/` 大体积 npz（按间隔）。 |
| `output_stress` | `false` | 是否写 `stress_field/` 大体积 npz。 |
| `field_output_interval` | `1` | 每隔多少仿真帧写一次体积场（在 `modified_simulation` 内使用）。 |
| `output_view_stress_heatmap` | `false` | 每视角 2D 应力热力图（不写三维体）。 |
| `output_view_stress_gaussian` | `false` | 每视角额外 3DGS 应力渲染（`stress_gaussian/`，与 `images/` 同相机；不写三维体）。 |
| `output_view_tracks_gaussian` | `false` | 每视角 3DGS 轨迹（黑底）：整物体 MPM 下采样，每输出帧 splat 当前（+可选中点）并叠到轨迹图；与 `images/` 同相机。 |
| `tracks_gaussian_max_tracks` | 2048 | MPM 粒子下采样数量上限。 |
| `tracks_gaussian_sigma_scale` / `point_opacity` | 0.0012 / 1.0 | σ 越小线越细；不透明度通常保持 1.0。 |
| `tracks_gaussian_intensity` / `seed` | 1.0 / 0 | `intensity` 仅在 `accum_no_normalize_save` 时使用；下采样种子。 |
| `tracks_gaussian_accum_mode` | `max` | `max`：逐像素取大（线清晰、后期不糊）；`add`：逐帧相加（易糊）。 |
| `tracks_gaussian_accum_frame_weight` / `accum_decay` | 1.0 / 1.0 | `max` 下为峰值缩放；`add` 下为叠加系数；decay 为 1 则旧迹不淡。 |
| `tracks_gaussian_accum_midpoint` | `false` | 相邻输出时刻位置中点再 splat（约 2× 点数、更连贯）。 |
| `tracks_gaussian_accum_no_normalize_save` | `false` | 为 true 时写 PNG 不按 max 归一化，改用 `tracks_gaussian_intensity` clamp。 |
| `output_subsampled_world_tracks` | `false` | **无光栅化**：随机下采样 MPM 粒子，按输出时刻写 `tracks_subsampled_world/tracks_world.npz`（`xyz_world`: T×N×3）。`compile_multiview_videos` 为 true 时另生成 `videos/tracks_subsampled_world_ortho_<xy|xz|yz>.mp4`（OpenCV 正交折线，非 3DGS）。 |
| `subsampled_tracks_num` / `seed` | 1024 / 0 | 追踪粒子数、随机种子。 |
| `subsampled_tracks_ortho_axes` / `video_size` | `xz` / 512 | 正交预览 mp4 用的世界系两维、正方形边长（像素）。 |
| `stress_gaussian_colormap_steps` | `24` | JET 阶梯档数（传给 `modified_simulation`）。 |
| `stress_gaussian_vm_pct_low` / `high` | `1` / `99` | 每帧 σ_vm **绝对值** 色标分位数；全 min/max 可设 `0` / `100`。（旧键 `stress_gaussian_log_pct_*` 仍兼容） |
| `output_view_flow_gaussian` | `false` | 每视角稠密 flow 伪彩 PNG（`flow_gaussian/<view>/`）；3DGS 同相机，opacity×距离加权 splat，不写 npz。 |
| `flow_gaussian_max_gaussians` | `8192` | 参与 splat 的 MPM 高斯数上限（随机下采样）。 |
| `flow_gaussian_seed` | `0` | 下采样随机种子。 |
| `flow_gaussian_depth_gamma` / `depth_eps` | `1.0` / `0.01` | 距离权重 `(1/(dist+eps))^gamma`。 |
| `flow_gaussian_opacity_power` | `1.0` | 权重中 `opacity^power`。 |
| `flow_gaussian_vis_max_motion` | `0` | 伪彩饱和位移（像素）；`≤0` 为按帧内分位数自动估计。 |
| `no_volumetric_stress_deformation` | `false` | 为 true 时不写 deformation/stress 体积 npz，仍可渲染与视角辅助。 |

### 3.6 Meta 与记录

| 键 | 默认 | 说明 |
|----|------|------|
| `output_bc_info` | `true` | 写边界条件摘要等到 `meta/`。 |
| `output_force_info` | `true` | 写外力等信息到 `meta/`。 |

仿真结束后 `modified_simulation` 会写 `meta/run_parameters.json`（帧数、渲染索引、材质与时间参数、开关等），详见 `rules/dataset_task.txt`。

### 3.7 示例文件

仓库内可参考：`configs_auto/train_config_example.json`（可按需增删字段）。

---

## 4. material_space_config.json

顶层必须包含 **`material_spaces`**：键为 **材质类型字符串**（与 MPM 中 `material` 一致，如 `jelly` / `metal` / `plasticine`），值为「该材质下要采样的参数」对象。

每个参数的取值支持三种写法（由 `_sample_from_space` 解析）：

1. **列表** `[a, b, c]`：等概率随机选一个元素。  
2. **连续均匀**：`{ "type": "uniform", "min": x, "max": y }`  
3. **对数均匀**（要求 min,max > 0）：`{ "type": "log_uniform", "min": x, "max": y }`  
4. **常量**：直接写数字或字符串，不随机。

每个 Job 会先随机选一个 **材质类型**，再对该类型下的每个 key 采样；随后 **`_filter_material_params_for_type`** 会：

- 只保留 solver 认识的键（如 `E`, `nu`, `density`, `material`, `grid_lim`, `n_grid`, `yield_stress`, `hardening`, `softening`, `friction_angle`, …）。
- **非 metal** 会去掉 `yield_stress`, `hardening`。
- **非 sand** 会去掉 `friction_angle`。

示例：`configs_auto/material_space_example.json`。

---

## 5. 模板 config 与参数合并（`_make_run_config`）

- 以 `base_config_by_sim_type[sim_type]` 读入 JSON 为底稿。
- 将 Job 中采样到的 `material_params` **逐项覆盖**到底稿根级字段。
- **特例**：键为 `g`（重力）时，**仅当 `sim_type == "drop"`** 才覆盖；其它动作保留模板里的重力设置。
- `boundary_conditions` 在合并后会被设为列表（可能为空）；**真正边界条件**由 `modified_simulation.py` 内按 `sim_type` 自动构造（press/drop/shear/stretch/bend）。

---

## 6. 下游工具衔接

| 工具 | 说明 |
|------|------|
| `transform_dataset.py` | 从 `output_root` 收集样本，整理为 `dataset_400` 式 `train/test` 或 `by_model` 扁平集；已同时支持 `output_layout` 的 **by_action** 与 **by_model** 原始目录。 |
| `analyze_auto_output_params.py` | 扫描 run 目录：旧布局解析文件夹名；新布局读 `gt_parameters.json`。 |
| `combine_auto_output_videos.py` | 收集 `videos/*.mp4`；新旧目录结构均支持。 |
| `my_model` | `dataset.iter_auto_output_runs` 兼容旧 `__` 命名与数字目录 + `gt_parameters.json`。 |

---

## 7. 常见问题

1. **子进程 import / CUDA 报错**  
   在仓库根目录执行；确认 conda/venv 与 GPU 驱动正常；若仍缺库，检查 `LD_LIBRARY_PATH` 是否被外部环境覆盖。

2. **没有 mp4**  
   安装 `ffmpeg` 并保证在 PATH 中；或关闭 `compile_video` 仅保留 PNG。

3. **路径找不到**  
   `base_config_by_sim_type` 里的路径相对于**运行 runner 时的当前工作目录**，一般用仓库根 + `config/xxx.json`。

4. **多卡 OOM**  
   减小并行度（`--num_gpus 1`）或降低模板里的分辨率/粒子数；或在 `modified_simulation` 侧减小 batch（若有）。

5. **与整理后数据集混放**  
   `output_root` 建议专用原始仿真根目录（如 `auto_output`），不要将 `dataset_400`、`dataset_full_test` 等整理结果与原始 `bend/`、`某物体名/` 混在同一扫描根下，以免下游脚本误扫（runner 本身只写 `_tmp_configs` 与仿真输出）。

---

## 8. 相关文档

- `rules/dataset_task.txt` — 渲染时间采样、视角应力、2D 轨迹、`run_parameters.json` 等约定。  
- `rules/dataset_history.txt` — 变更记录。  
- `modified_simulation.py --help` — 单任务 CLI 细节。
