import sys

sys.path.append("gaussian-splatting")
import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
import subprocess
import glob
import shutil
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov 

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from my_utils.sim_utils import (
    build_press_boundary_conditions,
    build_drop_boundary_conditions,
    build_shear_boundary_conditions,
    build_stretch_boundary_conditions,
    build_bend_boundary_conditions,
    save_boundary_condition_info,
    save_external_force_info,
    setup_field_output_dirs,
    save_fields_for_frame,
)


def _select_best_gpu() -> None:
    """
    自动选择最空闲的 GPU，并通过 CUDA_VISIBLE_DEVICES 屏蔽其它显卡。
    若已通过环境变量指定 CUDA_VISIBLE_DEVICES（如多卡 runner 分配），则不再覆盖。

    策略：
    1. 先打印每块卡的 free_mem 与 util；
    2. 优先在 GPU 利用率 < 10% 的卡中选择 free_mem 最大的；
    3. 如果没有满足利用率条件的卡，则在所有卡中选 free_mem 最大的。
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "":
        # 已由外部指定（如 auto_simulation_runner 多卡分配），不再自动选择
        return
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        lines = result.decode("utf-8").strip().split("\n")

        candidates = []
        print("[modified_simulation] GPU status from nvidia-smi:")
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            gpu_id = int(parts[0])
            mem_free = int(parts[1])  # MiB
            util = int(parts[2])      # %
            candidates.append((gpu_id, mem_free, util))
            print(f"  - GPU {gpu_id}: free_mem={mem_free} MiB, util={util}%")

        if not candidates:
            print("[modified_simulation] nvidia-smi returned no GPU info, using default CUDA device.")
            return

        # 首选利用率较低的 GPU
        low_util = [c for c in candidates if c[2] < 10]
        target_pool = low_util if low_util else candidates

        best_gpu, best_mem, best_util = max(target_pool, key=lambda x: x[1])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(
            "[modified_simulation] Auto-selected GPU id "
            f"{best_gpu} with free_mem={best_mem} MiB, util={best_util}%."
        )
    except Exception as e:
        print(f"[modified_simulation] GPU auto-selection failed ({e}), using default CUDA device.")


# 在 Warp / Taichi 初始化之前自动选择显卡
_select_best_gpu()

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):

    # ----------------------------
    # Find checkpoint
    # ----------------------------
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)

    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # ----------------------------
    # 检测是否存在 f_rest
    # ----------------------------
    with open(checkpt_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("utf-8")
            header += line
            if line.strip() == "end_header":
                break

    if "f_rest_" not in header:
        print("Detected PhysGaussian format (no SH rest terms).")
        print("Switching sh_degree -> 0")
        sh_degree = 0

    # ----------------------------
    # 原始加载流程（不改）
    # ----------------------------
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)

    return gaussians


def load_gaussians_from_ply(ply_path, sh_degree=3):
    """从单个 PLY 文件加载 Gaussian 模型（用于 ply_dir 批量模式）。"""
    with open(ply_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("utf-8")
            header += line
            if line.strip() == "end_header":
                break
    if "f_rest_" not in header:
        sh_degree = 0
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="单场景目录（含 point_cloud/iteration_*/point_cloud.ply）。与 --ply_dir 二选一。",
    )
    parser.add_argument(
        "--ply_dir",
        type=str,
        default=None,
        help="PLY 文件夹：目录内所有 .ply 依次读取，对每个做自动边界压缩实验；输出到 output_path/各文件名/",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default=None,
        help="单个 PLY 文件路径。与 --ply_dir/--model_path 互斥；用于自动化脚本逐个组合调用。",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_h5", action="store_true", help="保存仿真粒子为 h5（不保存 simulation ply）")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    #export deformation
    parser.add_argument("--output_deformation", action="store_true")
    parser.add_argument("--output_stress", action="store_true")
    parser.add_argument("--output_bc_info", action="store_true")
    parser.add_argument("--output_force_info", action="store_true")
    parser.add_argument("--field_output_interval", type=int, default=1)
    parser.add_argument(
        "--sim_type",
        type=str,
        default="press",
        choices=["press", "drop", "shear", "stretch", "bend"],
        help="仿真类型，用于自动设置边界条件。目前支持 press/drop 自动推断。",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="渲染视角数量。仅在方位角上均匀分布（俯仰角用 config 的 init_elevation）。",
    )
    #export deformation
    args = parser.parse_args()

    use_ply_dir = args.ply_dir is not None and os.path.isdir(args.ply_dir)
    use_ply_path = args.ply_path is not None and os.path.isfile(args.ply_path)

    # 默认输出目录：按 sim_type 自动分流
    if args.output_path is None:
        if args.sim_type == "press":
            args.output_path = os.path.join("output", "press_auto_test")
        elif args.sim_type == "drop":
            args.output_path = os.path.join("output", "drop_auto_test")
        else:
            args.output_path = os.path.join("output", f"{args.sim_type}_auto_test")

    if use_ply_path:
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        os.makedirs(args.output_path, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.ply_path))[0]
        tasks = [("ply", args.ply_path, os.path.join(args.output_path, name), os.path.dirname(args.ply_path))]
    elif use_ply_dir:
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        os.makedirs(args.output_path, exist_ok=True)
        ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
        if not ply_files:
            raise FileNotFoundError(f"在 --ply_dir 中未找到任何 .ply 文件: {args.ply_dir}")
        tasks = []
        for ply_path in ply_files:
            name = os.path.splitext(os.path.basename(ply_path))[0]
            tasks.append(("ply", ply_path, os.path.join(args.output_path, name), args.ply_dir))
    else:
        if args.model_path is None or not os.path.exists(args.model_path):
            raise AssertionError("请指定 --model_path 或有效的 --ply_dir/--ply_path。Model path does not exist!")
        if not os.path.exists(args.config):
            raise AssertionError("Scene config does not exist!")
        if args.output_path is not None and not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        tasks = [("checkpoint", args.model_path, args.output_path, args.model_path)]

    # load scene config（所有任务共用）
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    for task_idx, (task_type, path_or_ply, current_output_path, model_path_for_camera) in enumerate(tasks):
        if task_type == "ply":
            ply_path = path_or_ply
            print(f"\n[{task_idx+1}/{len(tasks)}] 加载 PLY: {ply_path}")
            gaussians = load_gaussians_from_ply(ply_path)
            model_path = model_path_for_camera
        else:
            print(f"\n[{task_idx+1}/{len(tasks)}] 加载 checkpoint: {path_or_ply}")
            model_path = path_or_ply
            gaussians = load_checkpoint(model_path)
        args.output_path = current_output_path
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)
        pipeline = PipelineParamsNoparse()
        pipeline.compute_cov3D_python = True
        background = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            if args.white_bg
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        )

        # init the scene
        print("Initializing scene and pre-processing...")
        params = load_params_from_gs(gaussians, pipeline)

        init_pos = params["pos"]
        init_cov = params["cov3D_precomp"]
        init_screen_points = params["screen_points"]
        init_opacity = params["opacity"]
        init_shs = params["shs"]

        # throw away low opacity kernels
        mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        init_pos = init_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_screen_points = init_screen_points[mask, :]
        init_shs = init_shs[mask, :]

        # 对 PLY 输入先绕 x 轴旋转 90 度，再走后续边界判断与仿真
        if task_type == "ply":
            device = init_pos.device
            rot_x90 = generate_rotation_matrices(
                torch.tensor([90.0], device=device, dtype=init_pos.dtype),
                [0],
            )  # 0 = x 轴
            init_pos = apply_rotations(init_pos, rot_x90)
            init_cov = apply_cov_rotations(init_cov, rot_x90)

        # rorate and translate object
        if args.debug:
            if not os.path.exists("./log"):
                os.makedirs("./log")
            particle_position_tensor_to_ply(
                init_pos,
                "./log/init_particles.ply",
            )
        rotation_matrices = generate_rotation_matrices(
            torch.tensor(preprocessing_params["rotation_degree"]),
            preprocessing_params["rotation_axis"],
        )
        rotated_pos = apply_rotations(init_pos, rotation_matrices)

        if args.debug:
            particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

        # select a sim area and save params of unslected particles
        unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
            None,
            None,
            None,
            None,
        )
        if preprocessing_params["sim_area"] is not None:
            boundary = preprocessing_params["sim_area"]
            assert len(boundary) == 6
            mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

            unselected_pos = init_pos[~mask, :]
            unselected_cov = init_cov[~mask, :]
            unselected_opacity = init_opacity[~mask, :]
            unselected_shs = init_shs[~mask, :]

            rotated_pos = rotated_pos[mask, :]
            init_cov = init_cov[mask, :]
            init_opacity = init_opacity[mask, :]
            init_shs = init_shs[mask, :]

        transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
        transformed_pos = shift2center111(transformed_pos)

        # modify covariance matrix accordingly
        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov

        if args.debug:
            particle_position_tensor_to_ply(
                transformed_pos,
                "./log/transformed_particles.ply",
            )

        # fill particles if needed
        gs_num = transformed_pos.shape[0]
        device = "cuda:0"
        filling_params = preprocessing_params["particle_filling"]

        if filling_params is not None:
            print("Filling internal particles...")
            mpm_init_pos = fill_particles(
                pos=transformed_pos,
                opacity=init_opacity,
                cov=init_cov,
                grid_n=filling_params["n_grid"],
                max_samples=filling_params["max_particles_num"],
                grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
                density_thres=filling_params["density_threshold"],
                search_thres=filling_params["search_threshold"],
                max_particles_per_cell=filling_params["max_partciels_per_cell"],
                search_exclude_dir=filling_params["search_exclude_direction"],
                ray_cast_dir=filling_params["ray_cast_direction"],
                boundary=filling_params["boundary"],
                smooth=filling_params["smooth"],
            ).to(device=device)

            if args.debug:
                particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
        else:
            mpm_init_pos = transformed_pos.to(device=device)

        # init the mpm solver
        print("Initializing MPM solver and setting up boundary conditions...")
        mpm_init_vol = get_particle_volume(
            mpm_init_pos,
            material_params["n_grid"],
            material_params["grid_lim"] / material_params["n_grid"],
            unifrom=material_params["material"] == "sand",
        ).to(device=device)

        if filling_params is not None and filling_params["visualize"] == True:
            shs, opacity, mpm_init_cov = init_filled_particles(
                mpm_init_pos[:gs_num],
                init_shs,
                init_cov,
                init_opacity,
                mpm_init_pos[gs_num:],
            )
            gs_num = mpm_init_pos.shape[0]
        else:
            mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
            mpm_init_cov[:gs_num] = init_cov
            shs = init_shs
            opacity = init_opacity

        if args.debug:
            print("check *.ply files to see if it's ready for simulation")

        # set up the mpm solver
        mpm_solver = MPM_Simulator_WARP(10)
        mpm_solver.load_initial_data_from_torch(
            mpm_init_pos,
            mpm_init_vol,
            mpm_init_cov,
            n_grid=material_params["n_grid"],
            grid_lim=material_params["grid_lim"],
        )
        mpm_solver.set_parameters_dict(material_params)

        # 基于仿真类型自动设置边界条件（当前支持 press/drop 自动推断）
        if args.sim_type == "press":
            print("[modified_simulation] 自动根据粒子位置构造 press 边界条件...")
            bc_params = build_press_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "drop":
            print("[modified_simulation] 自动构造 drop 边界条件（参考 drop_cube_jelly.json）...")
            bc_params = build_drop_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "shear":
            print("[modified_simulation] 自动构造 shear 边界条件（上下 1/4 相反方向刚体平面）...")
            bc_params = build_shear_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "stretch":
            print("[modified_simulation] 自动构造 stretch 边界条件（y 两端夹具拉伸）...")
            bc_params = build_stretch_boundary_conditions(mpm_init_pos, material_params)
        elif args.sim_type == "bend":
            print("[modified_simulation] 自动构造 bend 边界条件（沿最长轴方向底面固定+顶部刚性片推压）...")
            bc_params = build_bend_boundary_conditions(mpm_init_pos, material_params)

            # 同时输出一份带自动 BC 的 config 供记录
            try:
                with open(args.config, "r") as f:
                    sim_cfg = json.load(f)
                sim_cfg["boundary_conditions"] = bc_params
                auto_cfg_path = os.path.join(
                    args.output_path, f"auto_config_{args.sim_type}.json"
                )
                with open(auto_cfg_path, "w") as f:
                    json.dump(sim_cfg, f, indent=4)
                print(f"[modified_simulation] 自动生成 config 已保存到: {auto_cfg_path}")
            except Exception as e:
                print(f"[modified_simulation] 写入自动 config 失败: {e}")
        else:
            print(
                f"[modified_simulation] sim_type={args.sim_type}，暂未实现自动 BC，继续使用模板 config 中的 boundary_conditions。"
            )

        # Note: boundary conditions may depend on mass, so the order cannot be changed!
        set_boundary_conditions(mpm_solver, bc_params, time_params)

        mpm_solver.finalize_mu_lam()

        # 输出边界条件和力场信息（如果需要）
        if args.output_path is not None:
            meta_dir = os.path.join(args.output_path, "meta")
            if args.output_bc_info:
                save_boundary_condition_info(bc_params, material_params, meta_dir)
            if args.output_force_info:
                save_external_force_info(bc_params, material_params, meta_dir)
        # camera setting
        mpm_space_viewpoint_center = (
            torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
        )
        mpm_space_vertical_upward_axis = (
            torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
            .reshape((1, 3))
            .cuda()
        )
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )

        # run the simulation
        if args.output_h5:
            directory_to_save = os.path.join(args.output_path, "simulation_ply")
            os.makedirs(directory_to_save, exist_ok=True)
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                0,
                save_to_ply=False,
                save_to_h5=True,
            )

        substep_dt = time_params["substep_dt"]
        frame_dt = time_params["frame_dt"]
        frame_num = time_params["frame_num"]

        # press/shear/bend：总时长固定 4 秒；其中前 3 秒施力（BC end_time=3），最后 1 秒撤出力
        if args.sim_type in ("press", "shear", "bend"):
            total_duration_s = 4.0
            frame_num = int(round(total_duration_s / frame_dt))
            frame_num = max(1, frame_num)
        step_per_frame = int(frame_dt / substep_dt)
        opacity_render = opacity
        shs_render = shs
        height = None
        width = None

        # 准备形变场 / 应力场输出目录
        deformation_dir, stress_dir = setup_field_output_dirs(
            args.output_path,
            args.output_deformation,
            args.output_stress,
        )

        # 输出初始帧场数据
        save_fields_for_frame(
            mpm_solver,
            frame_id=0,
            deformation_dir=deformation_dir,
            stress_dir=stress_dir,
            output_deformation=args.output_deformation,
            output_stress=args.output_stress,
        )
        # 若需要渲染，准备多视角的图像/视频输出目录
        image_root = None
        video_root = None
        if args.output_path is not None:
            if args.render_img:
                image_root = os.path.join(args.output_path, "images")
                os.makedirs(image_root, exist_ok=True)
            if args.render_img and args.compile_video:
                video_root = os.path.join(args.output_path, "videos")
                os.makedirs(video_root, exist_ok=True)

        # 定义多视角：人为指定数量，方位角在 [0, 360) 上均匀分布，俯仰角固定为 config
        views = []
        view_dirs = []
        if args.render_img and image_root is not None:
            num_views = max(1, int(args.num_views))
            az_base = camera_params["init_azimuthm"]
            el_base = camera_params["init_elevation"]
            for k in range(num_views):
                az = az_base + 360.0 * k / num_views
                # 归一化到 [0, 360) 便于命名
                az = az % 360.0
                el = el_base
                views.append((az, el))
                name = f"az{int(round(az))}_el{int(round(el))}"
                d = os.path.join(image_root, name)
                os.makedirs(d, exist_ok=True)
                view_dirs.append(d)

        for frame in tqdm(range(frame_num)):

            for step in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=device)

            # 循环中输出场（按间隔）
            if (frame + 1) % args.field_output_interval == 0:
                save_fields_for_frame(
                    mpm_solver,
                    frame_id=frame + 1,
                    deformation_dir=deformation_dir,
                    stress_dir=stress_dir,
                    output_deformation=args.output_deformation,
                    output_stress=args.output_stress,
                )

            # 循环中输出 ply / h5
            if args.output_h5:
                save_data_at_frame(
                    mpm_solver,
                    directory_to_save,
                    frame + 1,
                    save_to_ply=False,
                    save_to_h5=True,
                )

            if args.render_img and views:
                # 先导出当前粒子信息一次，供所有视角复用
                pos_base = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D_base = mpm_solver.export_particle_cov_to_torch()
                rot_base = mpm_solver.export_particle_R_to_torch()
                cov3D_base = cov3D_base.view(-1, 6)[:gs_num].to(device)
                rot_base = rot_base.view(-1, 3, 3)[:gs_num].to(device)

                # 还原到世界坐标、Undo transform
                pos_world = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos_base), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                cov3D_world = cov3D_base / (scale_origin * scale_origin)
                cov3D_world = apply_inverse_cov_rotations(cov3D_world, rotation_matrices)

                base_opacity = opacity_render
                base_shs = shs_render
                if preprocessing_params["sim_area"] is not None:
                    pos_world = torch.cat([pos_world, unselected_pos], dim=0)
                    cov3D_world = torch.cat([cov3D_world, unselected_cov], dim=0)
                    base_opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    base_shs = torch.cat([shs_render, unselected_shs], dim=0)

                # 对每个视角分别渲染
                for v_idx, (az, el) in enumerate(views):
                    current_camera = get_camera_view(
                        model_path,
                        default_camera_index=camera_params["default_camera_index"],
                        center_view_world_space=viewpoint_center_worldspace,
                        observant_coordinates=observant_coordinates,
                        show_hint=camera_params["show_hint"],
                        init_azimuthm=az,
                        init_elevation=el,
                        init_radius=camera_params["init_radius"],
                        move_camera=camera_params["move_camera"],
                        current_frame=frame,
                        delta_a=camera_params["delta_a"],
                        delta_e=camera_params["delta_e"],
                        delta_r=camera_params["delta_r"],
                    )
                    rasterize = initialize_resterize(
                        current_camera, gaussians, pipeline, background
                    )

                    colors_precomp = convert_SH(
                        base_shs, current_camera, gaussians, pos_world, rot_base
                    )
                    rendering, raddi = rasterize(
                        means3D=pos_world,
                        means2D=init_screen_points,
                        shs=None,
                        colors_precomp=colors_precomp,
                        opacities=base_opacity,
                        scales=None,
                        rotations=None,
                        cov3D_precomp=cov3D_world,
                    )
                    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    if height is None or width is None:
                        height = cv2_img.shape[0] // 2 * 2
                        width = cv2_img.shape[1] // 2 * 2

                    view_dir = view_dirs[v_idx]
                    cv2.imwrite(
                        os.path.join(view_dir, f"{frame:04d}.png"),
                        255 * cv2_img,
                    )

        # 为每个视角分别合成视频，放在 videos 目录下
        if args.render_img and args.compile_video and video_root is not None and views:
            fps = int(1.0 / time_params["frame_dt"])
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin is None:
                print(
                    "[modified_simulation] 未找到 ffmpeg，无法合成视频。"
                    f"请先安装 ffmpeg，或只用 --render_img 输出图片。期望输出目录: {video_root}"
                )
                continue
            for v_idx, (az, el) in enumerate(views):
                view_dir = view_dirs[v_idx]
                name = f"az{int(round(az))}_el{int(round(el))}"
                out_path = os.path.join(video_root, f"{name}.mp4")
                in_pattern = os.path.join(view_dir, "%04d.png")
                cmd = [
                    ffmpeg_bin,
                    "-framerate",
                    str(fps),
                    "-i",
                    in_pattern,
                    "-c:v",
                    "libx264",
                    "-s",
                    f"{width}x{height}",
                    "-y",
                    "-pix_fmt",
                    "yuv420p",
                    out_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"[modified_simulation] ffmpeg 合成失败: {out_path}")
                    print(f"[modified_simulation] cmd: {' '.join(cmd)}")
                    if proc.stderr:
                        print(proc.stderr)
                else:
                    print(f"[modified_simulation] 已输出视频: {out_path}")
