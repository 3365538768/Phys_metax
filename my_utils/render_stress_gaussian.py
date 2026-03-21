#!/usr/bin/env python3
"""
使用与 modified_simulation 相同的 3D Gaussian Splatting 光栅化管线渲染应力场：
相机由 get_camera_view + stress_pcd_cameras.json 与 RGB 完全一致；
按 von Mises（绝对值、线性色标）调制预计算颜色与不透明度（高应力更不透明）。

依赖：CUDA、在 PhysGaussian 根目录运行；需与仿真相同的 --model_path、--config、simulation 输出目录。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 与 modified_simulation 一致
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gaussian-splatting"))
_PHYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PHYS_ROOT not in sys.path:
    sys.path.insert(0, _PHYS_ROOT)

from utils.camera_view_utils import get_camera_view  # noqa: E402
from utils.decode_param import decode_param_json  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402
from utils.render_utils import convert_SH, initialize_resterize  # noqa: E402
from utils.system_utils import searchForMaxIteration  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from my_utils.sim_utils import mpm_positions_to_world_numpy  # noqa: E402
from my_utils.view_auxiliary_output import stress_gaussian_precomp_colors_and_opacity  # noqa: E402


class PipelineParamsNoparse:
    def __init__(self) -> None:
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False


def _von_mises_np(stress: np.ndarray) -> np.ndarray:
    s = 0.5 * (stress + np.swapaxes(stress, 1, 2))
    mean = np.trace(s, axis1=1, axis2=2) / 3.0
    dev = s - mean[:, None, None] * np.eye(3)
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))


def _infer_wh_from_images(sim_dir: str, fallback: tuple[int, int]) -> tuple[int, int]:
    root = os.path.join(sim_dir, "images")
    if not os.path.isdir(root):
        return fallback
    for sd in sorted(os.listdir(root)):
        sd_path = os.path.join(root, sd)
        if not os.path.isdir(sd_path):
            continue
        for name in ("0000.png", "0001.png", "0002.png"):
            p = os.path.join(sd_path, name)
            if not os.path.isfile(p):
                continue
            im = cv2.imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            return max(2, w // 2 * 2), max(2, h // 2 * 2)
    return fallback


def _sorted_field_files(folder: str, prefix: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, f"{prefix}_*.npz")))


def _frame_id_from_path(path: str) -> int:
    m = re.search(r"_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_checkpoint(model_path: str, sh_degree: int = 3) -> GaussianModel:
    checkpt_dir = os.path.join(model_path, "point_cloud")
    iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    with open(checkpt_path, "rb") as f:
        header = ""
        while True:
            line = f.readline().decode("utf-8")
            header += line
            if line.strip() == "end_header":
                break
    if "f_rest_" not in header:
        sh_degree = 0
    g = GaussianModel(sh_degree)
    g.load_ply(checkpt_path)
    return g


def _global_vm_range_linear(
    stress_files: list[str], P: int, robust: tuple[float, float] | None
) -> tuple[float, float]:
    """全序列 von Mises 绝对值的 min/max 或分位数框定（不做 log）。"""
    chunks = []
    for fp in stress_files:
        z = np.load(fp)
        vm = _von_mises_np(z["stress_cauchy"][:P])
        chunks.append(np.maximum(vm.astype(np.float64), 0.0))
    all_v = np.concatenate(chunks)
    all_v = all_v[np.isfinite(all_v)]
    if all_v.size == 0:
        return 0.0, 1.0
    if robust:
        lo, hi = robust
        vmin = float(np.percentile(all_v, lo))
        vmax = float(np.percentile(all_v, hi))
    else:
        vmin = float(np.min(all_v))
        vmax = float(np.max(all_v))
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def main() -> None:
    os.chdir(_PHYS_ROOT)
    parser = argparse.ArgumentParser(
        description="3DGS 原生管线渲染应力热力（相机与 images/ 一致）"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="仿真用 scene json")
    parser.add_argument("--simulation_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--white_bg",
        action="store_true",
        help="白底（默认黑底与训练常见设置一致）",
    )
    parser.add_argument(
        "--stress_vmin_pct",
        type=float,
        default=None,
        help="与 stress_vmax_pct 同时指定时用分位数定全序列 von Mises 绝对值范围",
    )
    parser.add_argument(
        "--stress_vmax_pct",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--opa_floor",
        type=float,
        default=0.12,
        help="低应力侧不透明度比例下限（相对原 opacity 缩放中的 t→0 端）",
    )
    parser.add_argument(
        "--opa_ceil",
        type=float,
        default=1.0,
        help="高应力侧不透明度比例上限",
    )
    parser.add_argument(
        "--stress_sh_blend",
        type=float,
        default=0.35,
        help="SH 底色权重（与仿真内 --stress_gaussian_sh_blend 一致）",
    )
    parser.add_argument(
        "--colormap_steps",
        type=int,
        default=24,
        help="JET 阶梯档数（与仿真内 --stress_gaussian_colormap_steps 一致）",
    )
    args = parser.parse_args()

    robust = None
    if args.stress_vmin_pct is not None and args.stress_vmax_pct is not None:
        robust = (float(args.stress_vmin_pct), float(args.stress_vmax_pct))

    meta_path = os.path.join(args.simulation_dir, "meta", "stress_pcd_cameras.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"需要 {meta_path}（与 visualize_fields 多视角同源）")
    with open(meta_path, "r", encoding="utf-8") as f:
        camera_meta = json.load(f)
    mpm_tw = camera_meta.get("mpm_to_world")
    if not mpm_tw:
        raise RuntimeError("stress_pcd_cameras.json 缺少 mpm_to_world，请用新版仿真重导 meta")

    (
        _material,
        _bc,
        _time,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    device = torch.device("cuda:0")
    # load_ply 已将参数放在 CUDA 上；GaussianModel 非 nn.Module，无 .to()
    gaussians = load_checkpoint(args.model_path)
    P = int(gaussians.get_xyz.shape[0])
    pipeline = PipelineParamsNoparse()
    bg = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    )

    shs_render = gaussians.get_features
    opacity_render = gaussians.get_opacity

    def_path = os.path.join(args.simulation_dir, "deformation_field")
    str_path = os.path.join(args.simulation_dir, "stress_field")
    dfiles = _sorted_field_files(def_path, "deformation_frame")
    sfiles = _sorted_field_files(str_path, "stress_frame")
    n = min(len(dfiles), len(sfiles))
    if n == 0:
        raise RuntimeError("未找到 deformation_field / stress_field npz")

    fb_w = int(camera_meta.get("width_hint", 800))
    fb_h = int(camera_meta.get("height_hint", 800))
    rw, rh = _infer_wh_from_images(args.simulation_dir, (fb_w, fb_h))
    print(f"[render_stress_gaussian] 输出分辨率 {rw}x{rh}（对齐 images/）")

    vm_min, vm_max = _global_vm_range_linear(sfiles, P, robust)
    print(
        f"[render_stress_gaussian] von Mises 绝对值范围: [{vm_min:.6g}, {vm_max:.6g}]"
    )

    viewpoint_center = np.asarray(camera_meta["look_at"], dtype=np.float64)
    observant = np.asarray(camera_meta["observant_coordinates"], dtype=np.float64)

    views = camera_meta["views"]
    motion = camera_meta.get("camera_motion") or {}
    move_cam = bool(motion.get("move_camera", False))
    da = float(motion.get("delta_a", 0.0))
    de = float(motion.get("delta_e", 0.0))
    dr = float(motion.get("delta_r", 0.0))
    init_radius = float(camera_meta.get("init_radius", 2.0))

    out_root = os.path.join(args.output_path, "stress_gaussian_multiview")
    os.makedirs(out_root, exist_ok=True)
    for v in views:
        os.makedirs(os.path.join(out_root, v["name"]), exist_ok=True)

    cov_static = gaussians.get_covariance(1.0)

    for fi in tqdm(range(n), desc="frames"):
        dnp = np.load(dfiles[fi])
        snp = np.load(sfiles[fi])
        pos_np = dnp["position"][:P]
        pos_np = mpm_positions_to_world_numpy(pos_np, mpm_tw).astype(np.float32)
        pos_world = torch.from_numpy(pos_np).to(device=device, dtype=torch.float32)
        stress_np = snp["stress_cauchy"][:P]
        vm_np = _von_mises_np(stress_np)
        vm_t = torch.from_numpy(vm_np.astype(np.float32)).to(device)

        means2d = torch.zeros(
            (P, 2),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        try:
            means2d.retain_grad()
        except Exception:
            pass

        sim_f = _frame_id_from_path(dfiles[fi])

        for vinfo in views:
            az0 = float(vinfo["azimuth"])
            el0 = float(vinfo["elevation"])
            if move_cam:
                az_use = az0 + sim_f * da
                el_use = el0 + sim_f * de
                rad_use = init_radius + sim_f * dr
            else:
                az_use, el_use, rad_use = az0, el0, init_radius

            cam = get_camera_view(
                args.model_path,
                default_camera_index=int(camera_params.get("default_camera_index", -1)),
                center_view_world_space=viewpoint_center,
                observant_coordinates=observant,
                show_hint=False,
                init_azimuthm=az_use,
                init_elevation=el_use,
                init_radius=rad_use,
                move_camera=False,
                current_frame=0,
                delta_a=0.0,
                delta_e=0.0,
                delta_r=0.0,
            )
            # 与 RGB 输出尺寸一致（synthetic 时 GSCamera 内置 800²，此处覆盖为实际图幅）
            cam.image_height = rh
            cam.image_width = rw
            cam.original_image = torch.zeros((3, rh, rw), device=device)
            # 重算投影矩阵（与 FoV 一致）
            cam.world_view_transform = torch.tensor(
                getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
            ).transpose(0, 1).cuda()
            cam.projection_matrix = getProjectionMatrix(
                znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy
            ).transpose(0, 1).cuda()
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(
                    cam.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

            base_c = convert_SH(
                shs_render, cam, gaussians, pos_world, rotation=None
            )
            colors_precomp, stress_op = stress_gaussian_precomp_colors_and_opacity(
                vm_t,
                base_c,
                opacity_render,
                P,
                opa_floor=float(args.opa_floor),
                opa_ceil=float(args.opa_ceil),
                sh_blend=float(args.stress_sh_blend),
                vm_vmin_fixed=vm_min,
                vm_vmax_fixed=vm_max,
                colormap_steps=int(args.colormap_steps),
            )

            rasterize = initialize_resterize(cam, gaussians, pipeline, bg)
            rendering, _r = rasterize(
                means3D=pos_world,
                means2D=means2d,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=stress_op,
                scales=None,
                rotations=None,
                cov3D_precomp=cov_static,
            )
            img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_png = os.path.join(out_root, vinfo["name"], f"{fi:04d}.png")
            cv2.imwrite(out_png, np.clip(255.0 * img, 0, 255).astype(np.uint8))

    print("[render_stress_gaussian] 完成:", out_root)


if __name__ == "__main__":
    main()
