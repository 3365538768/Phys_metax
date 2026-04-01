from __future__ import annotations

"""
从 checkpoint 加载模型，在 test 集上推理，导出预测参数 JSON；
将 **test 中全部用于可视化的样本** 画在 **一张图** 里（2×2 子图，横轴 GT、纵轴 Pred，红线 y=x）。
默认对 test **全部样本** 推理；若 ``--num_samples > 0`` 则随机子集（便于快速试跑）。

raw 与 log 训练空间各输出一张总图：``scatter_raw.png``、``scatter_log.png``。

可选 ``--save_field_videos``：在 ``out_dir/field_videos/`` 下为每个样本写出
stress / flow / force_mask 的 **pred vs GT** 并排 **mp4**（默认三通道 → R/G/B 合成；
``--field_video_color_mode jet`` 为通道均值 + JET 伪彩）（需 opencv-python）。

在 Phys 仓库根目录执行::

    python -m logic_model.eval_visual \\
        --config logic_model/configs/logic_train_dataset_mask_1000.json \\
        --checkpoint logic_model/output_dataset_mask_1000/checkpoints/epoch_0002.pt \\
        --out_dir logic_model/eval_visual_run

多卡并行（DDP，分片推理）::

    torchrun --nproc_per_node=8 -m logic_model.eval_visual \\
        --config logic_model/configs/logic_train_dataset_mask_1000.json \\
        --checkpoint logic_model/output_dataset_mask_1000/checkpoints/epoch_0002.pt \\
        --out_dir logic_model/eval_visual_run \\
        --batch_size 2
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch
from logic_model.model import LogicPhysModel


def _pick(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be json object: {path}")
    return cfg


def _resolve_path_relative_to_config(rel: str, config_path: str | None) -> Path:
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    if config_path:
        cand = (Path(config_path).resolve().parent / p).resolve()
        if cand.exists():
            return cand
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return cwd_p


def _load_split_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"split json 顶层必须是 object: {path}")
    return raw


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _resample_time_bvcthw(x: torch.Tensor, target_t: int) -> torch.Tensor:
    t0 = int(x.shape[3])
    if t0 == int(target_t):
        return x
    b, v, c, _, h, w = x.shape
    y = x.permute(0, 1, 2, 4, 5, 3).contiguous().view(b * v * c * h * w, 1, t0)
    y = torch.nn.functional.interpolate(y, size=int(target_t), mode="linear", align_corners=False)
    y = y.view(b, v, c, h, w, int(target_t)).permute(0, 1, 2, 5, 3, 4).contiguous()
    return y


def _axis_lo_hi(xs: np.ndarray, ys: np.ndarray, pad_frac: float, axis_percentiles: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    npt = int(xs.shape[0])
    if axis_percentiles is not None and npt >= 2:
        pl, ph = float(axis_percentiles[0]), float(axis_percentiles[1])
        lo = float(min(np.percentile(xs, pl), np.percentile(ys, pl)))
        hi = float(max(np.percentile(xs, ph), np.percentile(ys, ph)))
    else:
        lo = float(min(xs.min(), ys.min()))
        hi = float(max(xs.max(), ys.max()))
    span = hi - lo
    if span < 1e-12:
        return lo - 0.5, hi + 0.5
    pad = float(max(0.0, pad_frac)) * span
    return lo - pad, hi + pad


def _save_param_yx_scatter_one_figure(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    *,
    space: str,
    pad_frac: float = 0.02,
    axis_percentiles: Optional[Tuple[float, float]] = None,
    names: tuple[str, ...] = ("E", "nu", "density", "yield_stress"),
) -> None:
    """
    单张图 2×2：每个子图横坐标 GT、纵坐标 Pred，全部样本画在一起。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("散点图需要 matplotlib：pip install matplotlib") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, i, name in zip(axes.flat, range(4), names):
        gx = gt[:, i].astype(np.float64)
        py = pred[:, i].astype(np.float64)
        m = np.isfinite(gx) & np.isfinite(py)
        if name == "yield_stress" and space == "raw":
            m = m & (gx >= 0) & (py >= 0)
        if not np.any(m):
            ax.set_visible(False)
            continue
        xs, ys = gx[m], py[m]
        npt = int(xs.shape[0])
        lo, hi = _axis_lo_hi(xs, ys, pad_frac, axis_percentiles)
        ax.scatter(xs, ys, s=12, alpha=0.5, c="tab:blue", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_title(f"{name}  N={npt}")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Param GT vs Pred ({space}) — all test samples in one figure", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _align_gt_field_to_pred(gt_bvcthw: torch.Tensor, pred_bvcthw: torch.Tensor) -> torch.Tensor:
    """GT 与 pred 空间不一致时，将 GT 双线性缩放到 pred 的 [H,W]（与 my_model.eval 思路一致）。"""
    if gt_bvcthw.shape == pred_bvcthw.shape:
        return gt_bvcthw
    b, v, c, t, h, w = gt_bvcthw.shape
    hp, wp = pred_bvcthw.shape[-2], pred_bvcthw.shape[-1]
    x = gt_bvcthw.reshape(b * v * t, c, h, w)
    y = F.interpolate(x, size=(hp, wp), mode="bilinear", align_corners=False)
    return y.view(b, v, c, t, hp, wp).contiguous()


def _field_bvcthw_to_mono(bvcthw: torch.Tensor) -> torch.Tensor:
    """[B,V,3,T,H,W] -> [B,V,1,T,H,W]，三通道取均值便于灰度/热力显示。"""
    return bvcthw.mean(dim=2, keepdim=True)


def _to_color_map(img_2d: torch.Tensor) -> "np.ndarray":
    """单帧 2D 标量 -> JET BGR uint8（与 my_model.eval 一致）。"""
    import cv2

    a = img_2d.detach().cpu().float().numpy()
    amin = float(a.min())
    amax = float(a.max())
    denom = max(1e-8, amax - amin)
    a01 = (a - amin) / denom
    u8 = (a01 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _field_chw_to_bgr_uint8(chw: torch.Tensor) -> "np.ndarray":
    """
    单帧 [3,H,W] 场 → BGR uint8：通道 0/1/2 分别作为 R/G/B，每通道独立 min-max 到 [0,255]。
    与数据集 ``[V,3,T,H,W]`` 三通道语义一致（合成 RGB 显示；OpenCV 写盘用 BGR）。
    """
    import cv2

    x = chw.detach().cpu().float()
    if x.dim() != 3:
        raise ValueError(f"expect [3,H,W], got {tuple(x.shape)}")
    c = int(x.shape[0])
    if c == 1:
        g = x[0]
        lo, hi = g.min(), g.max()
        g01 = (g - lo) / (hi - lo + 1e-8)
        rgb_hwc = (g01.unsqueeze(-1).expand(-1, -1, 3) * 255.0).clamp(0, 255).byte().numpy()
    elif c == 3:
        out = torch.zeros_like(x)
        for ci in range(3):
            ch = x[ci]
            lo, hi = ch.min(), ch.max()
            out[ci] = (ch - lo) / (hi - lo + 1e-8)
        rgb_hwc = (out * 255.0).clamp(0, 255).permute(1, 2, 0).byte().numpy()
    else:
        raise ValueError(f"expect C=1 or 3 for RGB composite, got C={c}")

    return cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)


def _save_field_pred_vs_gt_mp4(
    pred_bvcthw: torch.Tensor,
    tgt_bvcthw: torch.Tensor,
    out_path: Path,
    *,
    view_idx: int = 0,
    fps: int = 8,
    color_mode: str = "rgb",
) -> None:
    """
    pred/tgt: ``color_mode=='rgb'`` 时为 ``[B,V,3,T,H,W]``；``color_mode=='jet'`` 时为 ``[B,V,1,T,H,W]``。
    B=1。左 pred 右 GT；rgb 为三通道合成彩色，jet 为标量 JET。
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    b = int(pred_bvcthw.shape[0])
    if b <= 0:
        return
    v = int(pred_bvcthw.shape[1])
    view_idx = max(0, min(int(view_idx), v - 1))
    t = int(pred_bvcthw.shape[3])
    h = int(pred_bvcthw.shape[4])
    w = int(pred_bvcthw.shape[5])
    if t <= 0:
        return
    cm = str(color_mode).lower().strip()
    if cm not in ("rgb", "jet"):
        raise ValueError(f"color_mode must be 'rgb' or 'jet', got {color_mode!r}")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {out_path}（需 opencv-python）")
    try:
        for ti in range(t):
            if cm == "jet":
                pred_2d = pred_bvcthw[0, view_idx, 0, ti]
                tgt_2d = tgt_bvcthw[0, view_idx, 0, ti]
                pred_color = _to_color_map(pred_2d)
                tgt_color = _to_color_map(tgt_2d)
            else:
                pred_chw = pred_bvcthw[0, view_idx, :, ti, :, :]
                tgt_chw = tgt_bvcthw[0, view_idx, :, ti, :, :]
                pred_color = _field_chw_to_bgr_uint8(pred_chw)
                tgt_color = _field_chw_to_bgr_uint8(tgt_chw)
            frame = cv2.hconcat([pred_color, tgt_color])
            writer.write(frame)
    finally:
        writer.release()


def _safe_sample_id_for_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_")[:200]


def export_field_videos_rank0(
    *,
    model: torch.nn.Module,
    eval_ds: torch.utils.data.Dataset,
    device: torch.device,
    mcore: "LogicPhysModel",
    out_root: Path,
    max_samples: int,
    fps: int,
    view_idx: int,
    color_mode: str = "rgb",
) -> None:
    """
    仅 rank0：对 eval_ds 前 max_samples 个样本各导出 stress / flow / force_mask 的 pred vs GT mp4。
    ``color_mode=rgb``：三通道 → R/G/B 合成（与数据集 ``[V,3,T,H,W]`` 一致）；
    ``color_mode=jet``：通道均值 + JET 伪彩（旧行为）。
    """
    if max_samples <= 0:
        return
    vdir = out_root / "field_videos"
    vdir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_lmdb_gt_batch,
    )
    model.eval()
    n = 0
    pbar = tqdm(
        loader,
        total=min(max_samples, len(eval_ds)),
        desc="field_videos",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in pbar:
            if n >= max_samples:
                break
            rgb = batch["rgb"].to(device)
            stress_gt = batch["stress"].to(device)
            flow_gt = batch["flow"].to(device)
            force_gt = batch["force_mask"].to(device)
            sid = str(batch["sample_id"][0])

            if int(mcore.in_channels) == 1:
                x = rgb[:, :, :1, :, :, :]
            else:
                x = rgb[:, :, : int(mcore.in_channels), :, :, :]
            if int(x.shape[3]) != int(mcore.num_frames):
                x = _resample_time_bvcthw(x, int(mcore.num_frames))
                stress_gt = _resample_time_bvcthw(stress_gt, int(mcore.num_frames))
                flow_gt = _resample_time_bvcthw(flow_gt, int(mcore.num_frames))
                force_gt = _resample_time_bvcthw(force_gt, int(mcore.num_frames))

            out = model(x)
            ps = out["stress_field_pred"]
            pf = out["flow_field_pred"]
            pfm = out["force_pred"]

            stress_gt = _align_gt_field_to_pred(stress_gt, ps)
            flow_gt = _align_gt_field_to_pred(flow_gt, pf)
            force_gt = _align_gt_field_to_pred(force_gt, pfm)

            safe = _safe_sample_id_for_filename(sid)
            cm = str(color_mode).lower().strip()
            if cm == "jet":
                ps_m = _field_bvcthw_to_mono(ps)
                pf_m = _field_bvcthw_to_mono(pf)
                pfm_m = _field_bvcthw_to_mono(pfm)
                sg_m = _field_bvcthw_to_mono(stress_gt)
                fg_m = _field_bvcthw_to_mono(flow_gt)
                fgm_m = _field_bvcthw_to_mono(force_gt)
                _save_field_pred_vs_gt_mp4(
                    ps_m,
                    sg_m,
                    vdir / f"{safe}_stress_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="jet",
                )
                _save_field_pred_vs_gt_mp4(
                    pf_m,
                    fg_m,
                    vdir / f"{safe}_flow_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="jet",
                )
                _save_field_pred_vs_gt_mp4(
                    pfm_m,
                    fgm_m,
                    vdir / f"{safe}_force_mask_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="jet",
                )
            else:
                _save_field_pred_vs_gt_mp4(
                    ps,
                    stress_gt,
                    vdir / f"{safe}_stress_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="rgb",
                )
                _save_field_pred_vs_gt_mp4(
                    pf,
                    flow_gt,
                    vdir / f"{safe}_flow_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="rgb",
                )
                _save_field_pred_vs_gt_mp4(
                    pfm,
                    force_gt,
                    vdir / f"{safe}_force_mask_pred_vs_gt.mp4",
                    view_idx=view_idx,
                    fps=fps,
                    color_mode="rgb",
                )
            n += 1
            pbar.set_postfix(sample_id=sid[:32])

    print(
        f"[eval_visual] field_videos 已写入: {vdir}（mode={color_mode}，每样本 3 个 mp4，共 {n} 个样本）",
        flush=True,
    )


def _init_distributed() -> Tuple[bool, int, int, int]:
    """WORLD_SIZE>1 时初始化进程组，返回 (distributed, rank, local_rank, world_size)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, local_rank, world_size


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Eval checkpoint on test: GT(x) vs Pred(y) scatter, all samples on one figure")
    ap.add_argument("--config", type=str, required=True, help="训练 JSON 配置（与 train 相同）")
    ap.add_argument("--checkpoint", type=str, required=True, help=".pt checkpoint 路径")
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="<=0 表示使用 test 全部样本；>0 时随机抽该数量（用于快速试跑）",
    )
    ap.add_argument("--seed", type=int, default=0, help="仅当 num_samples>0 时用于随机子集抽样")
    ap.add_argument("--batch_size", type=int, default=4, help="推理 batch 大小")
    ap.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    ap.add_argument("--out_dir", type=str, default="logic_model/eval_visual_out", help="JSON 与 PNG 输出目录")
    ap.add_argument("--pad_frac", type=float, default=0.02, help="散点图坐标轴外扩比例")
    ap.add_argument(
        "--axis_percentiles",
        type=float,
        nargs=2,
        metavar=("P_LO", "P_HI"),
        default=None,
        help="可选，如 1 99 用分位数定轴范围",
    )
    ap.add_argument(
        "--ddp",
        action="store_true",
        help="显式要求 DDP（需 torchrun WORLD_SIZE>1；一般可不写，自动检测）",
    )
    ap.add_argument(
        "--save_field_videos",
        action="store_true",
        help="在 out_dir/field_videos 下导出 stress/flow/force_mask 的 pred vs GT mp4（仅 rank0）",
    )
    ap.add_argument(
        "--max_field_video_samples",
        type=int,
        default=50,
        help="与 --save_field_videos 联用：最多导出多少条样本；<=0 表示与当前 eval 子集等长（全量）",
    )
    ap.add_argument("--field_video_fps", type=int, default=8, help="field mp4 帧率")
    ap.add_argument("--field_video_view", type=int, default=0, help="多视角时使用的 view 索引")
    ap.add_argument(
        "--field_video_color_mode",
        type=str,
        choices=("rgb", "jet"),
        default="rgb",
        help="场视频：rgb=三通道 R/G/B 合成；jet=通道均值+JET 伪彩",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = _init_distributed()
    if bool(getattr(args, "ddp", False)) and not distributed:
        raise ValueError("--ddp 已指定但未检测到 WORLD_SIZE>1，请使用 torchrun 启动")
    cfg = _load_config(args.config)
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}

    split_root = str(_pick(cfg_data, "split_root", "") or "").strip() or "auto_output/dataset_deformation_stress_500_new/train"
    split_root = str(Path(split_root).expanduser())
    lmdb_env_subdir = str(_pick(cfg_data, "lmdb_env_subdir", "arch4_data.lmdb") or "arch4_data.lmdb").strip()
    train_ids_json = cfg_data.get("train_ids_json")
    train_id_list: Optional[List[str]] = None
    test_id_list: Optional[List[str]] = None
    if train_ids_json:
        jp = _resolve_path_relative_to_config(str(train_ids_json), args.config)
        if not jp.is_file():
            raise FileNotFoundError(f"train_ids_json 不存在: {jp}")
        split_meta = _load_split_json(jp)
        train_id_list = [str(x) for x in (split_meta.get("train_ids") or [])]
        test_id_list = [str(x) for x in (split_meta.get("test_ids") or [])]
        sr_json = str(split_meta.get("split_root") or "").strip()
        if sr_json and not cfg_data.get("split_root"):
            split_root = sr_json
        lm = split_meta.get("lmdb_env_subdir")
        if lm and not cfg_data.get("lmdb_env_subdir"):
            lmdb_env_subdir = str(lm).strip() or lmdb_env_subdir

    if not test_id_list:
        raise ValueError("split json 中无 test_ids，无法做 test 集可视化；请检查 train_ids_json")

    max_views = int(_pick(cfg_model, "num_views", 4))
    num_frames_cfg = int(_pick(cfg_model, "num_frames", 0))
    img_size = int(_pick(cfg_model, "img_size", 0))
    dec_h = int(_pick(cfg_model, "dec_h", 112))
    dec_w = int(_pick(cfg_model, "dec_w", 112))

    ds_train = LmdbGtDataset(
        split_root=split_root,
        lmdb_env_subdir=lmdb_env_subdir,
        max_views=max_views,
        num_frames=(None if num_frames_cfg <= 0 else num_frames_cfg),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=train_id_list,
    )
    ds_test = LmdbGtDataset(
        split_root=split_root,
        lmdb_env_subdir=lmdb_env_subdir,
        max_views=max_views,
        num_frames=(None if num_frames_cfg <= 0 else num_frames_cfg),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=test_id_list,
        action_to_id=ds_train.action_to_id,
    )

    n_req = int(args.num_samples)
    if n_req <= 0:
        chosen = list(range(len(ds_test)))
        n_take = len(chosen)
    else:
        n_take = min(n_req, len(ds_test))
        rng = random.Random(int(args.seed))
        chosen = rng.sample(range(len(ds_test)), n_take)
    eval_ds = ds_test if n_take == len(ds_test) else Subset(ds_test, chosen)
    eval_sampler: Optional[DistributedSampler] = None
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("多卡 torchrun 推理需要 CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        device_str = str(args.device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

    loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=0,
        collate_fn=collate_lmdb_gt_batch,
    )

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state" not in ckpt:
        raise KeyError(f"checkpoint 缺少 model_state 键: {ckpt_path}")

    mh = ckpt.get("model_hparams") if isinstance(ckpt.get("model_hparams"), dict) else {}
    num_frames_model = int(_pick(cfg_model, "num_frames", mh.get("num_frames", 0)) or 0)
    if num_frames_model <= 0:
        sample0 = ds_test[0]
        num_frames_model = int(sample0["rgb"].shape[2])

    model = LogicPhysModel(
        num_views=int(mh.get("num_views", max_views)),
        in_channels=int(mh.get("in_channels", _pick(cfg_model, "in_channels", 3))),
        num_frames=int(mh.get("num_frames", num_frames_model)),
        img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
        num_targets=4,
        num_actions=ds_train.num_actions,
        dec_h=int(mh.get("dec_h", dec_h)),
        dec_w=int(mh.get("dec_w", dec_w)),
        encoder_embed_dim=int(_pick(cfg_model, "encoder_embed_dim", 384)),
        encoder_depth=int(_pick(cfg_model, "encoder_depth", 6)),
        encoder_num_heads=int(_pick(cfg_model, "encoder_num_heads", 6)),
        tubelet_size=int(_pick(cfg_model, "tubelet_size", 1)),
        patch_size=int(_pick(cfg_model, "patch_size", 32)),
        fusion_dim=int(_pick(cfg_model, "fusion_dim", 512)),
        fusion_heads=int(_pick(cfg_model, "fusion_heads", 8)),
        head_dropout=float(_pick(cfg_model, "head_dropout", 0.1)),
        use_uncertainty=bool(_pick(cfg_model, "use_uncertainty", False)),
        bottleneck_dim=int(_pick(cfg_model, "bottleneck_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    mcore = _unwrap_model(model)
    records: List[Dict[str, Any]] = []
    gt_raw_list: List[np.ndarray] = []
    pred_raw_list: List[np.ndarray] = []

    infer_pbar = tqdm(
        loader,
        total=len(loader),
        desc="eval infer",
        disable=(rank != 0),
        dynamic_ncols=True,
        leave=True,
    )
    with torch.no_grad():
        for batch in infer_pbar:
            rgb = batch["rgb"].to(device)
            params_gt_raw = batch["params"].to(device)
            action_label = batch["action_label"].to(device)

            if int(mcore.in_channels) == 1:
                x = rgb[:, :, :1, :, :, :]
            else:
                x = rgb[:, :, : int(mcore.in_channels), :, :, :]
            if int(x.shape[3]) != int(mcore.num_frames):
                x = _resample_time_bvcthw(x, int(mcore.num_frames))

            out = model(x)
            pred_raw = out["param_pred_raw"].detach().cpu()
            gt_cpu = params_gt_raw.detach().cpu()
            action_pred = out["action_logits"].argmax(dim=1).detach().cpu()
            probs = torch.softmax(out["action_logits"], dim=1).detach().cpu()
            sample_ids = [str(s) for s in batch["sample_id"]]
            gt_names = batch.get("action_name", [""] * len(sample_ids))

            for i in range(len(sample_ids)):
                records.append(
                    {
                        "sample_id": sample_ids[i],
                        "param_gt_raw": gt_cpu[i].tolist(),
                        "param_pred_raw": pred_raw[i].tolist(),
                        "action_gt_label": int(action_label[i].item()),
                        "action_pred_label": int(action_pred[i].item()),
                        "action_prob": probs[i].tolist(),
                        "action_gt_name": str(gt_names[i]) if i < len(gt_names) else "",
                    }
                )
            gt_raw_list.append(gt_cpu.numpy())
            pred_raw_list.append(pred_raw.numpy())
            if rank == 0:
                infer_pbar.set_postfix(n_local=len(records))

    if gt_raw_list:
        local_gt = np.concatenate(gt_raw_list, axis=0)
        local_pred = np.concatenate(pred_raw_list, axis=0)
    else:
        local_gt = np.zeros((0, 4), dtype=np.float64)
        local_pred = np.zeros((0, 4), dtype=np.float64)

    if distributed:
        dist.barrier()
        if rank == 0:
            rec_gather: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
            gt_gather: List[Optional[np.ndarray]] = [None] * world_size
            pred_gather: List[Optional[np.ndarray]] = [None] * world_size
        else:
            rec_gather = None
            gt_gather = None
            pred_gather = None
        dist.gather_object(records, object_gather_list=rec_gather, dst=0)
        dist.gather_object(local_gt, object_gather_list=gt_gather, dst=0)
        dist.gather_object(local_pred, object_gather_list=pred_gather, dst=0)
        if rank == 0:
            records = []
            for sub in rec_gather or []:
                if sub:
                    records.extend(sub)
            records.sort(key=lambda r: str(r.get("sample_id", "")))
            assert gt_gather is not None and pred_gather is not None
            gt_all = np.concatenate([g for g in gt_gather if g is not None], axis=0)
            pred_all = np.concatenate([g for g in pred_gather if g is not None], axis=0)
        else:
            records = []
            gt_all = np.zeros((0, 4), dtype=np.float64)
            pred_all = np.zeros((0, 4), dtype=np.float64)
    else:
        gt_all = local_gt
        pred_all = local_pred

    out_root = Path(args.out_dir).expanduser().resolve()
    pct: Optional[Tuple[float, float]] = None
    if args.axis_percentiles is not None:
        pct = (float(args.axis_percentiles[0]), float(args.axis_percentiles[1]))

    meta = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path),
        "num_samples_requested": int(args.num_samples),
        "num_samples_used": int(n_take),
        "use_full_test": bool(n_take == len(ds_test)),
        "seed": int(args.seed) if int(args.num_samples) > 0 else None,
        "split_root": split_root,
        "test_ids_total": len(test_id_list),
        "distributed": bool(distributed),
        "world_size": int(world_size),
    }
    report = {
        "meta": meta,
        "epoch_in_checkpoint": ckpt.get("epoch"),
        "samples": records,
    }
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "predictions.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        scatter_raw_png = out_root / "scatter_raw.png"
        scatter_log_png = out_root / "scatter_log.png"
        print(
            "[eval_visual] 推理与聚合完成，开始绘制散点图 "
            f"(scatter_raw={scatter_raw_png.name}, scatter_log={scatter_log_png.name}) …",
            flush=True,
        )
        _save_param_yx_scatter_one_figure(
            gt_all,
            pred_all,
            scatter_raw_png,
            space="raw",
            pad_frac=float(args.pad_frac),
            axis_percentiles=pct,
        )
        gt_log = _to_target_params(torch.from_numpy(gt_all).float()).numpy()
        pred_log = _to_target_params(torch.from_numpy(pred_all).float()).numpy()
        _save_param_yx_scatter_one_figure(
            gt_log,
            pred_log,
            scatter_log_png,
            space="log",
            pad_frac=float(args.pad_frac),
            axis_percentiles=pct,
        )

        field_videos_dir: Optional[str] = None
        if bool(getattr(args, "save_field_videos", False)):
            cap = int(args.max_field_video_samples)
            if cap <= 0:
                cap = len(eval_ds)
            print(
                f"[eval_visual] 导出 field 视频（最多 {cap} 条）→ {out_root / 'field_videos'} …",
                flush=True,
            )
            export_field_videos_rank0(
                model=model,
                eval_ds=eval_ds,
                device=device,
                mcore=mcore,
                out_root=out_root,
                max_samples=cap,
                fps=int(args.field_video_fps),
                view_idx=int(args.field_video_view),
                color_mode=str(args.field_video_color_mode),
            )
            field_videos_dir = str(out_root / "field_videos")

        print(
            json.dumps(
                {
                    "out_dir": str(out_root),
                    "num_samples": n_take,
                    "predictions_json": str(out_root / "predictions.json"),
                    "scatter_raw_png": str(scatter_raw_png),
                    "scatter_log_png": str(scatter_log_png),
                    "field_videos_dir": field_videos_dir,
                    "distributed": distributed,
                    "world_size": world_size,
                },
                ensure_ascii=False,
            )
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
