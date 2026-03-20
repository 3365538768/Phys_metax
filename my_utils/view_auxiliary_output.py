"""
渲染视角下的辅助输出：应力热力图（2D）、2D 形变轨迹（投影），避免保存完整三维体数据。

与 modified_simulation 中 Gaussian 光栅化共用同一套相机与可见性（radii > 0）。

应力标量（Cauchy → von Mises）：
  与 mpm_solver.save_stress_field 相同：τ = particle_stress，J = det(F_trial)，σ = τ/J。
  调用方在「整帧所有子步 p2g2p 结束之后」须先执行
  MPM_Simulator_WARP.recompute_particle_stress_from_F_trial(substep_dt)，
  否则 g2p 已更新 F_trial 而 τ 仍停留在子步初，热力图会与物理状态不一致。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def compute_render_frame_indices(
    frame_num: int, num_render_timesteps: int
) -> List[int]:
    """
    在整个仿真时间轴上均匀选取要输出渲染/热力图/轨迹采样的帧（仿真步下标）。

    num_render_timesteps <= 0 或 >= frame_num：每一仿真帧都输出。
    否则在 [0, frame_num-1] 上含端点均匀取 K 个整数帧。
    """
    if frame_num <= 0:
        return []
    if num_render_timesteps <= 0 or num_render_timesteps >= frame_num:
        return list(range(frame_num))
    K = int(num_render_timesteps)
    if K == 1:
        return [0]
    out = sorted(
        {int(round(i * (frame_num - 1) / (K - 1))) for i in range(K)}
    )
    return out


def frame_to_output_index(render_frame_indices: Sequence[int]) -> Dict[int, int]:
    return {int(f): i for i, f in enumerate(render_frame_indices)}


def cauchy_stress_per_particle(mpm_solver) -> torch.Tensor:
    """
    Cauchy 应力 σ，形状 (N, 3, 3)，与 save_stress_field 一致（F_trial 算 J，Kirchhoff τ）。

    须已在当前时刻调用过 recompute_particle_stress_from_F_trial（见模块文档）。
    """
    tau = mpm_solver.export_particle_stress_to_torch()
    F = mpm_solver.export_particle_F_trial_to_torch()
    J = torch.det(F).clamp_min(1e-12)
    sigma = tau / J.view(-1, 1, 1)
    return sigma


def von_mises_cauchy(sigma: torch.Tensor) -> torch.Tensor:
    """
    von Mises 等效应力（基于 Cauchy 张量，单位与 σ 一致）。
    sigma: (N, 3, 3)
    """
    s = sigma
    s11, s12, s13 = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2]
    s22, s23 = s[:, 1, 1], s[:, 1, 2]
    s33 = s[:, 2, 2]
    return torch.sqrt(
        0.5
        * (
            (s11 - s22) ** 2
            + (s22 - s33) ** 2
            + (s33 - s11) ** 2
            + 6.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2)
        )
    )


def stress_scalars_aligned_to_render_chain(
    mpm_solver,
    gs_num: int,
    extra_tail: int,
    device: torch.device,
) -> torch.Tensor:
    """与当前渲染链路的粒子数对齐：前 gs_num 来自 MPM，extra_tail 个补零（如未参与仿真的高斯）。"""
    sigma = cauchy_stress_per_particle(mpm_solver)[:gs_num]
    vm = von_mises_cauchy(sigma)
    if extra_tail > 0:
        z = torch.zeros(extra_tail, device=device, dtype=vm.dtype)
        vm = torch.cat([vm, z], dim=0)
    return vm


def splat_stress_heatmap_bgr(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    stress_values: np.ndarray,
    height: int,
    width: int,
    visible_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将标量应力 splat 到图像平面（取每个像素上的最大应力以便重叠区域可见）。

    means2d: (P, 3) 或 (P, 2)，光栅化后的屏幕坐标
    radii: (P,) 整数半径（像素），<=0 表示不可见
    stress_values: (P,) 与 means2d 行对齐
    """
    P = int(means2d.shape[0])
    acc = np.zeros((height, width), dtype=np.float32)
    m2 = means2d.detach().cpu().numpy()
    rad = radii.detach().cpu().numpy().reshape(-1)
    if visible_mask is None:
        visible_mask = rad > 0
    else:
        visible_mask = np.asarray(visible_mask) & (rad > 0)

    smax = float(np.nanmax(stress_values[visible_mask])) if np.any(visible_mask) else 0.0
    smin = float(np.nanmin(stress_values[visible_mask])) if np.any(visible_mask) else 0.0
    if smax <= smin + 1e-12:
        smax = smin + 1.0

    for i in range(P):
        if not visible_mask[i]:
            continue
        x = float(m2[i, 0])
        y = float(m2[i, 1])
        r = int(max(1, rad[i]))
        val = float(stress_values[i])
        if not np.isfinite(val):
            continue
        xi = int(round(x))
        yi = int(round(y))
        if xi < -r or yi < -r or xi >= width + r or yi >= height + r:
            continue
        layer = np.zeros((height, width), dtype=np.float32)
        cv2.circle(layer, (xi, yi), r, 1.0, thickness=-1)
        acc = np.maximum(acc, layer * val)

    norm = ((acc - smin) / (smax - smin) * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[acc <= 0] = (0, 0, 0)
    return color


def collect_uv_visibility(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (uv, visible)：
    - uv: (P, 2) float32，不可见时为 nan
    - visible: (P,) bool
    """
    P = int(means2d.shape[0])
    m2 = means2d.detach().cpu().numpy()
    rad = radii.detach().cpu().numpy().reshape(-1)
    uv = np.full((P, 2), np.nan, dtype=np.float32)
    vis = np.zeros((P,), dtype=bool)
    for i in range(P):
        if rad[i] <= 0:
            continue
        x = float(m2[i, 0])
        y = float(m2[i, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        uv[i, 0] = x
        uv[i, 1] = y
        vis[i] = True
    return uv, vis


def write_run_parameters_json(
    path: str,
    payload: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class ViewTrackBuffer:
    """每个视角累积 (T_out, P, 2) 的 UV 与 visible。"""

    def __init__(self, num_points: int):
        self.num_points = int(num_points)
        self.uv_list: List[np.ndarray] = []
        self.vis_list: List[np.ndarray] = []
        self.sim_frames: List[int] = []

    def append(self, sim_frame: int, means2d: torch.Tensor, radii: torch.Tensor, h: int, w: int) -> None:
        uv, vis = collect_uv_visibility(means2d, radii, h, w)
        if uv.shape[0] != self.num_points:
            raise ValueError(
                f"ViewTrackBuffer: 期望 P={self.num_points}, 得到 {uv.shape[0]}"
            )
        self.uv_list.append(uv)
        self.vis_list.append(vis)
        self.sim_frames.append(int(sim_frame))

    def save_npz(self, path: str) -> None:
        if not self.uv_list:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        uv = np.stack(self.uv_list, axis=0)
        vis = np.stack(self.vis_list, axis=0)
        sf = np.array(self.sim_frames, dtype=np.int32)
        np.savez_compressed(
            path,
            uv_pixels=uv,
            visible=vis,
            sim_frames=sf,
            description="uv_pixels: (T,P,2) 屏幕像素坐标，不可见为 nan；visible: (T,P)；sim_frames: 对应仿真帧下标",
        )
