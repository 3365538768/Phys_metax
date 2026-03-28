from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch
from logic_model.losses import (
    Arch4LossConfig,
    Arch4RegressionLoss,
    PhysicsConsistencyConfig,
    action_classification_loss,
    compute_physics_consistency_losses,
)
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


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _resample_time_bvcthw(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    x: [B,V,C,T,H,W] -> [B,V,C,target_t,H,W]
    """
    t0 = int(x.shape[3])
    if t0 == int(target_t):
        return x
    b, v, c, _, h, w = x.shape
    y = x.permute(0, 1, 2, 4, 5, 3).contiguous().view(b * v * c * h * w, 1, t0)
    y = torch.nn.functional.interpolate(y, size=int(target_t), mode="linear", align_corners=False)
    y = y.view(b, v, c, h, w, int(target_t)).permute(0, 1, 2, 5, 3, 4).contiguous()
    return y


def _align_bvcthw_to_ref(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    src/ref: [B,V,C,T,H,W]，按 ref 的 [T,H,W] 对齐 src（时间+空间）。
    """
    if src.shape == ref.shape:
        return src
    if src.dim() != 6 or ref.dim() != 6:
        raise ValueError(f"expect 6D [B,V,C,T,H,W], got src={tuple(src.shape)} ref={tuple(ref.shape)}")
    b, v, c, ts, hs, ws = src.shape
    tr, hr, wr = int(ref.shape[3]), int(ref.shape[4]), int(ref.shape[5])
    x = src.view(b * v, c, ts, hs, ws)
    x = F.interpolate(x, size=(tr, hr, wr), mode="trilinear", align_corners=False)
    return x.view(b, v, c, tr, hr, wr)


def _object_mask_weighted_field_loss(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor,
    *,
    fg_weight: float,
    bg_weight: float,
    bg_black: bool,
) -> torch.Tensor:
    """
    前景加权场监督:
    - 前景(由 object_mask 定义): 拟合 gt
    - 背景: 默认拟合黑色(0)，抑制背景噪声
    """
    pred = pred_bvcthw
    gt = _align_bvcthw_to_ref(gt_bvcthw, pred)
    obj = _align_bvcthw_to_ref(object_mask_bvcthw, pred)
    obj = obj.mean(dim=2, keepdim=True).clamp(0.0, 1.0)
    obj = obj.expand(-1, -1, int(pred.shape[2]), -1, -1, -1)

    err_fg = (pred - gt).pow(2)
    if bg_black:
        err_bg = pred.pow(2)
    else:
        err_bg = (pred - gt).pow(2)

    w_fg = float(max(0.0, fg_weight))
    w_bg = float(max(0.0, bg_weight))
    num = w_fg * (obj * err_fg).sum() + w_bg * ((1.0 - obj) * err_bg).sum()
    den = w_fg * obj.sum() + w_bg * (1.0 - obj).sum()
    return num / (den + 1e-6)


def _tensor_video_to_u8_rgb_frames(
    x_tc_hw: torch.Tensor,
    *,
    colormap: str = "turbo",
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    """
    输入:
    - [T,H,W]（单通道）: 归一化后 colormap 着色为 RGB
    - [T,3,H,W]（RGB）: 按固定 value range 映射为 RGB（便于 pred/gt 同口径对比）
    输出 uint8 [T,H,W,3]。
    """
    x = x_tc_hw.detach().float().cpu().numpy()

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("保存 eval 视频需要 opencv-python（cv2）") from e

    # 单通道: [T,H,W]
    if x.ndim == 3:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if (x_max - x_min) < 1e-8:
            gray = np.zeros_like(x, dtype=np.uint8)
        else:
            x01 = (x - x_min) / (x_max - x_min + 1e-8)
            gray = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)

        cmap_key = str(colormap).strip().lower()
        cmap_map = {
            "turbo": cv2.COLORMAP_TURBO,
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
        }
        cmap = cmap_map.get(cmap_key, cv2.COLORMAP_TURBO)
        frames_rgb = []
        for i in range(int(gray.shape[0])):
            bgr = cv2.applyColorMap(gray[i], cmap)  # [H,W,3] BGR
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
        return np.stack(frames_rgb, axis=0).astype(np.uint8)

    # RGB: [T,3,H,W]
    if x.ndim == 4 and int(x.shape[1]) == 3:
        vmin = float(value_min)
        vmax = float(value_max)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        x01 = (x - vmin) / (vmax - vmin)
        x01 = np.clip(x01, 0.0, 1.0)
        return np.transpose((x01 * 255.0).astype(np.uint8), (0, 2, 3, 1))

    raise ValueError(f"expect [T,H,W] or [T,3,H,W], got {x.shape}")


def _save_rgb_video_mp4(frames_u8_thwc: np.ndarray, out_path: Path, fps: int = 12) -> None:
    """
    保存 RGB 帧 [T,H,W,3] 到 mp4。
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("保存 eval 视频需要 opencv-python（cv2）") from e

    t, h, w, c = map(int, frames_u8_thwc.shape)
    if c != 3:
        raise ValueError(f"expect RGB frames [T,H,W,3], got {frames_u8_thwc.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"无法写入视频: {out_path}")
    try:
        for i in range(t):
            rgb = frames_u8_thwc[i]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
    finally:
        vw.release()


def _auto_pick_least_utilized_gpu() -> int | None:
    """
    使用 nvidia-smi 按 utilization.gpu 选择最空闲 GPU 的“逻辑编号”。
    - 若设置了 CUDA_VISIBLE_DEVICES，会先在可见物理卡内选择，再映射为逻辑序号。
    - 若查询失败，返回 None。
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[1])
            mem = int(parts[2])
        except ValueError:
            continue
        rows.append((util, mem, idx))

    if not rows:
        return None

    # 默认: 物理 index -> (util, mem)
    phys_stats = {int(idx): (int(util), int(mem)) for util, mem, idx in rows}

    # 若设置了 CUDA_VISIBLE_DEVICES，按可见物理卡过滤，再映射到逻辑 index
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env:
        visible_phys: list[int] = []
        for tok in visible_env.split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                visible_phys.append(int(t))
            except ValueError:
                # UUID/MIG 格式时不做物理映射，退回 torch 逻辑索引
                visible_phys = []
                break

        if visible_phys:
            cands = []
            for logical_idx, phys_idx in enumerate(visible_phys):
                st = phys_stats.get(int(phys_idx))
                if st is None:
                    continue
                util, mem = st
                cands.append((util, mem, logical_idx))
            if cands:
                cands.sort()
                return int(cands[0][2])
            return 0

    # 未设置 CUDA_VISIBLE_DEVICES：一般逻辑 index 与物理 index 一致
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        return None
    cands = []
    for phys_idx, (util, mem) in phys_stats.items():
        if 0 <= int(phys_idx) < visible_count:
            cands.append((util, mem, int(phys_idx)))
    if not cands:
        return 0
    cands.sort()
    return int(cands[0][2])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("logic_model minimal trainer")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--split_root", type=str, default="auto_output/dataset_deformation_stress_500_new/train")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=1, help=">0 时仅取前 N 个样本做快速调试")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_views", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=0, help="0=按LMDB实际总帧数")
    ap.add_argument("--img_size", type=int, default=0, help="仅 num_frames>0 时需要")
    ap.add_argument("--dec_h", type=int, default=112)
    ap.add_argument("--dec_w", type=int, default=112)
    ap.add_argument("--device", type=str, default="cuda")
    # loss weights
    ap.add_argument("--lambda_stress", type=float, default=10)
    ap.add_argument("--lambda_flow", type=float, default=10)
    ap.add_argument("--lambda_force", type=float, default=10)
    ap.add_argument("--lambda_action", type=float, default=1.0)
    ap.add_argument("--lambda_phys", type=float, default=0.001)
    ap.add_argument("--use_pred_force_mask_for_phys", action="store_true")
    ap.add_argument("--save_every_epochs", type=int, default=100000)
    ap.add_argument("--eval_every_epochs", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=1, help="每次 eval 评估多少个 batch（最小调试用）")
    ap.add_argument("--eval_video_fps", type=int, default=16)
    ap.add_argument("--eval_max_video_views", type=int, default=2)
    ap.add_argument("--eval_video_value_min", type=float, default=0.0, help="RGB 导出固定映射下界")
    ap.add_argument("--eval_video_value_max", type=float, default=1.0, help="RGB 导出固定映射上界")
    ap.add_argument("--object_mask_fg_weight", type=float, default=10.0, help="object_mask 前景损失权重")
    ap.add_argument("--object_mask_bg_weight", type=float, default=1.0, help="object_mask 背景损失权重")
    ap.add_argument("--object_mask_bg_black", type=int, default=1, help="1=背景监督为黑色(0), 0=背景也拟合gt")
    ap.add_argument(
        "--eval_video_colormap",
        type=str,
        default="turbo",
        help="stress/flow/force 单通道预测转 RGB 时使用的 colormap: turbo/jet/viridis/magma/inferno",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    cfg_train = cfg.get("train") or {}

    split_root = str(args.split_root if args.split_root is not None else _pick(cfg_data, "split_root", ""))
    max_views = int(args.max_views if args.max_views is not None else _pick(cfg_model, "num_views", 4))
    num_frames = int(args.num_frames if args.num_frames is not None else _pick(cfg_model, "num_frames", 0))
    img_size = int(args.img_size if args.img_size is not None else _pick(cfg_model, "img_size", 0))
    dec_h = int(args.dec_h if args.dec_h is not None else _pick(cfg_model, "dec_h", 56))
    dec_w = int(args.dec_w if args.dec_w is not None else _pick(cfg_model, "dec_w", 56))

    ds = LmdbGtDataset(
        split_root=split_root,
        max_views=max_views,
        num_frames=(None if num_frames <= 0 else num_frames),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
    )
    train_source = Subset(ds, list(range(min(int(args.max_samples), len(ds))))) if int(args.max_samples) > 0 else ds
    loader = DataLoader(
        train_source,
        batch_size=int(args.batch_size if args.batch_size is not None else _pick(cfg_train, "batch_size", 1)),
        shuffle=True,
        num_workers=int(args.num_workers if args.num_workers is not None else _pick(cfg_train, "num_workers", 0)),
        collate_fn=collate_lmdb_gt_batch,
    )
    eval_loader = DataLoader(
        train_source,
        batch_size=int(args.batch_size if args.batch_size is not None else _pick(cfg_train, "batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_lmdb_gt_batch,
    )

    device_str = str(args.device if args.device is not None else _pick(cfg_train, "device", "cuda"))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    elif device_str in ("cuda", "auto"):
        picked_gpu = _auto_pick_least_utilized_gpu()
        if picked_gpu is None:
            picked_gpu = 0
        device_str = f"cuda:{picked_gpu}"
    device = torch.device(device_str)

    if num_frames > 0:
        model_num_frames = int(num_frames)
    else:
        # full-frames 模式下，使用第一个样本帧数作为模型固定时间长度
        sample0 = ds[0]
        model_num_frames = int(sample0["rgb"].shape[2])

    model = LogicPhysModel(
        num_views=max_views,
        in_channels=int(_pick(cfg_model, "in_channels", 3)),
        num_frames=int(_pick(cfg_model, "num_frames", model_num_frames)),
        img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
        num_targets=4,
        num_actions=ds.num_actions,
        dec_h=dec_h,
        dec_w=dec_w,
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
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else _pick(cfg_train, "lr", 3e-4)),
    )

    loss_reg = Arch4RegressionLoss(Arch4LossConfig())
    phys_cfg = PhysicsConsistencyConfig(
        # 仅保留 lambda_phys 作为总权重，三个 physics 子项固定等权。
        lambda_stress_flow_consistency=1.0,
        lambda_force_stress_consistency=1.0,
        lambda_force_flow_consistency=1.0,
        use_pred_force_mask=bool(_pick(cfg_train, "use_pred_force_mask_for_phys", args.use_pred_force_mask_for_phys)),
    )

    lambda_stress = float(_pick(cfg_train, "lambda_stress", args.lambda_stress))
    lambda_flow = float(_pick(cfg_train, "lambda_flow", args.lambda_flow))
    lambda_force = float(_pick(cfg_train, "lambda_force", args.lambda_force))
    lambda_action = float(_pick(cfg_train, "lambda_action", args.lambda_action))
    lambda_phys = float(_pick(cfg_train, "lambda_phys", args.lambda_phys))
    object_mask_fg_weight = float(_pick(cfg_train, "object_mask_fg_weight", args.object_mask_fg_weight))
    object_mask_bg_weight = float(_pick(cfg_train, "object_mask_bg_weight", args.object_mask_bg_weight))
    object_mask_bg_black = bool(int(_pick(cfg_train, "object_mask_bg_black", args.object_mask_bg_black)))

    # 统一输出目录：全部保存在 logic_model 下
    output_root = (Path("logic_model") / "output").resolve()
    ckpt_dir = output_root / "checkpoints"
    eval_dir = output_root / "eval"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    save_every_epochs = max(1, int(args.save_every_epochs))
    eval_every_epochs = max(1, int(args.eval_every_epochs))
    eval_batches = max(1, int(args.eval_batches))
    eval_video_fps = max(1, int(args.eval_video_fps))
    eval_max_video_views = max(1, int(args.eval_max_video_views))
    eval_video_value_min = float(args.eval_video_value_min)
    eval_video_value_max = float(args.eval_video_value_max)
    if eval_video_value_max <= eval_video_value_min:
        eval_video_value_max = eval_video_value_min + 1e-6
    eval_video_colormap = str(args.eval_video_colormap)

    epochs = int(args.epochs if args.epochs is not None else _pick(cfg_train, "epochs", 1))
    print(
        f"[logic_train] dataset={len(ds)} num_actions={ds.num_actions} "
        f"lambdas(reg=1, stress={lambda_stress}, flow={lambda_flow}, force={lambda_force}, "
        f"action={lambda_action}, phys={lambda_phys})"
    )
    print(
        "[logic_train] "
        f"object_mask_loss(fg_w={object_mask_fg_weight}, bg_w={object_mask_bg_weight}, "
        f"bg_black={int(object_mask_bg_black)})"
    )
    print(f"[logic_train] action_to_id={json.dumps(ds.action_to_id, ensure_ascii=False)}")
    print(f"[logic_train] model_num_frames={model.num_frames}")
    print(f"[logic_train] device={device}")
    print(f"[logic_train] output_root={output_root}")

    (output_root / "action_to_id.json").write_text(
        json.dumps(ds.action_to_id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "cfg_data": cfg_data,
                "cfg_model": cfg_model,
                "cfg_train": cfg_train,
                "resolved": {
                    "split_root": split_root,
                    "max_views": max_views,
                    "num_frames": num_frames,
                    "img_size": img_size,
                    "dec_h": dec_h,
                    "dec_w": dec_w,
                    "device": str(device),
                    "model_num_frames": int(model.num_frames),
                    "output_root": str(output_root),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            stress_gt = batch["stress"].to(device)
            flow_gt = batch["flow"].to(device)
            force_gt = batch["force_mask"].to(device)
            object_gt = batch["object_mask"].to(device)
            params_gt_raw = batch["params"].to(device)
            action_label = batch["action_label"].to(device)

            # [B,V,3,T,H,W] -> [B,V,C,T,H,W]，兼容 in_channels
            if int(model.in_channels) == 1:
                x = rgb[:, :, :1, :, :, :]
            else:
                x = rgb[:, :, : int(model.in_channels), :, :, :]

            # 兼容 full-frames 数据读取：训练时对齐到模型固定 num_frames
            if int(x.shape[3]) != int(model.num_frames):
                x = _resample_time_bvcthw(x, int(model.num_frames))
                stress_gt = _resample_time_bvcthw(stress_gt, int(model.num_frames))
                flow_gt = _resample_time_bvcthw(flow_gt, int(model.num_frames))
                force_gt = _resample_time_bvcthw(force_gt, int(model.num_frames))
                object_gt = _resample_time_bvcthw(object_gt, int(model.num_frames))

            out = model(x)

            gt_train_space = _to_target_params(params_gt_raw)
            valid_mask = torch.ones_like(gt_train_space)
            valid_mask[:, 3] = (params_gt_raw[:, 3] > 0).to(valid_mask.dtype)

            loss_reg_part = loss_reg(out["param_pred"], gt_train_space, out["logvar"], valid_mask=valid_mask)
            loss_stress_part = _object_mask_weighted_field_loss(
                out["stress_field_pred"],
                stress_gt,
                object_gt,
                fg_weight=object_mask_fg_weight,
                bg_weight=object_mask_bg_weight,
                bg_black=object_mask_bg_black,
            )
            loss_flow_part = _object_mask_weighted_field_loss(
                out["flow_field_pred"],
                flow_gt,
                object_gt,
                fg_weight=object_mask_fg_weight,
                bg_weight=object_mask_bg_weight,
                bg_black=object_mask_bg_black,
            )
            loss_force_part = _object_mask_weighted_field_loss(
                out["force_pred"],
                force_gt,
                object_gt,
                fg_weight=object_mask_fg_weight,
                bg_weight=object_mask_bg_weight,
                bg_black=object_mask_bg_black,
            )

            action_ret = action_classification_loss(out["action_logits"], action_label)
            phys_ret = compute_physics_consistency_losses(
                stress_pred=out["stress_field_pred"],
                flow_pred=out["flow_field_pred"],
                force_pred=out["force_pred"],
                force_gt=force_gt,
                cfg=phys_cfg,
            )

            loss_total = (
                loss_reg_part
                + lambda_stress * loss_stress_part
                + lambda_flow * loss_flow_part
                + lambda_force * loss_force_part
                + lambda_action * action_ret["loss_action"]
                + lambda_phys * phys_ret["loss_phys_total"]
            )

            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = int(rgb.shape[0])
            total += float(loss_total.item()) * bsz
            n += bsz

            if bi % 10 == 0:
                print(
                    "[logic_train] "
                    f"epoch={epoch+1}/{epochs} step={bi} "
                    f"loss={float(loss_total.item()):.6f} "
                    f"reg={float(loss_reg_part.item()):.6f} "
                    f"stress={float(loss_stress_part.item()):.6f} "
                    f"flow={float(loss_flow_part.item()):.6f} "
                    f"force={float(loss_force_part.item()):.6f} "
                    f"action={float(action_ret['loss_action'].item()):.6f} "
                    f"action_acc={float(action_ret['action_acc'].item()):.4f} "
                    f"phys={float(phys_ret['loss_phys_total'].item()):.6f} "
                    f"phys_sf={float(phys_ret['loss_stress_flow_consistency'].item()):.6f} "
                    f"phys_fs={float(phys_ret['loss_force_stress_consistency'].item()):.6f} "
                    f"phys_ff={float(phys_ret['loss_force_flow_consistency'].item()):.6f} "
                    f"objmask_mean={float(object_gt.mean().item()):.4f}"
                )

        print(f"[logic_train] epoch={epoch+1} avg_loss={total / max(n, 1):.6f}")
        avg_loss = float(total / max(n, 1))

        # checkpoint（全部在 logic_model/output/checkpoints）
        if ((epoch + 1) % save_every_epochs) == 0:
            ckpt = {
                "epoch": int(epoch + 1),
                "avg_loss": avg_loss,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "action_to_id": ds.action_to_id,
                "model_hparams": {
                    "num_views": int(max_views),
                    "in_channels": int(model.in_channels),
                    "num_frames": int(model.num_frames),
                    "dec_h": int(dec_h),
                    "dec_w": int(dec_w),
                },
            }
            torch.save(ckpt, str(ckpt_dir / f"epoch_{epoch + 1:04d}.pt"))
            # 明确不维护 best/last checkpoint，避免误用历史 best.pt / last.pt。
            best_ckpt_path = ckpt_dir / "best.pt"
            if best_ckpt_path.exists():
                best_ckpt_path.unlink()
            last_ckpt_path = ckpt_dir / "last.pt"
            if last_ckpt_path.exists():
                last_ckpt_path.unlink()

        # 轻量 eval（全部在 logic_model/output/eval）
        if ((epoch + 1) % eval_every_epochs) == 0:
            model.eval()
            with torch.no_grad():
                e_total = 0.0
                e_n = 0
                e_action_acc = 0.0
                e_steps = 0
                per_sample_preds = []
                epoch_eval_dir = eval_dir / f"epoch_{epoch + 1:04d}"
                videos_dir = epoch_eval_dir / "videos"
                for batch in eval_loader:
                    rgb = batch["rgb"].to(device)
                    stress_gt = batch["stress"].to(device)
                    flow_gt = batch["flow"].to(device)
                    force_gt = batch["force_mask"].to(device)
                    object_gt = batch["object_mask"].to(device)
                    params_gt_raw = batch["params"].to(device)
                    action_label = batch["action_label"].to(device)

                    if int(model.in_channels) == 1:
                        x = rgb[:, :, :1, :, :, :]
                    else:
                        x = rgb[:, :, : int(model.in_channels), :, :, :]
                    if int(x.shape[3]) != int(model.num_frames):
                        x = _resample_time_bvcthw(x, int(model.num_frames))
                        stress_gt = _resample_time_bvcthw(stress_gt, int(model.num_frames))
                        flow_gt = _resample_time_bvcthw(flow_gt, int(model.num_frames))
                        force_gt = _resample_time_bvcthw(force_gt, int(model.num_frames))
                        object_gt = _resample_time_bvcthw(object_gt, int(model.num_frames))

                    out = model(x)
                    gt_train_space = _to_target_params(params_gt_raw)
                    valid_mask = torch.ones_like(gt_train_space)
                    valid_mask[:, 3] = (params_gt_raw[:, 3] > 0).to(valid_mask.dtype)

                    loss_reg_part = loss_reg(out["param_pred"], gt_train_space, out["logvar"], valid_mask=valid_mask)
                    loss_stress_part = _object_mask_weighted_field_loss(
                        out["stress_field_pred"],
                        stress_gt,
                        object_gt,
                        fg_weight=object_mask_fg_weight,
                        bg_weight=object_mask_bg_weight,
                        bg_black=object_mask_bg_black,
                    )
                    loss_flow_part = _object_mask_weighted_field_loss(
                        out["flow_field_pred"],
                        flow_gt,
                        object_gt,
                        fg_weight=object_mask_fg_weight,
                        bg_weight=object_mask_bg_weight,
                        bg_black=object_mask_bg_black,
                    )
                    loss_force_part = _object_mask_weighted_field_loss(
                        out["force_pred"],
                        force_gt,
                        object_gt,
                        fg_weight=object_mask_fg_weight,
                        bg_weight=object_mask_bg_weight,
                        bg_black=object_mask_bg_black,
                    )
                    action_ret = action_classification_loss(out["action_logits"], action_label)
                    phys_ret = compute_physics_consistency_losses(
                        stress_pred=out["stress_field_pred"],
                        flow_pred=out["flow_field_pred"],
                        force_pred=out["force_pred"],
                        force_gt=force_gt,
                        cfg=phys_cfg,
                    )
                    loss_total = (
                        loss_reg_part
                        + lambda_stress * loss_stress_part
                        + lambda_flow * loss_flow_part
                        + lambda_force * loss_force_part
                        + lambda_action * action_ret["loss_action"]
                        + lambda_phys * phys_ret["loss_phys_total"]
                    )
                    bsz = int(x.shape[0])
                    e_total += float(loss_total.item()) * bsz
                    e_n += bsz
                    e_action_acc += float(action_ret["action_acc"].item()) * bsz

                    # 记录每个样本的参数/动作预测，并导出 stress/flow/force 预测视频
                    probs = torch.softmax(out["action_logits"], dim=1).detach().cpu()
                    action_pred = out["action_logits"].argmax(dim=1).detach().cpu()
                    param_raw = out["param_pred_raw"].detach().cpu()
                    stress_pred = out["stress_field_pred"].detach().cpu()  # [B,V,3,T,H,W]
                    flow_pred = out["flow_field_pred"].detach().cpu()
                    force_pred = out["force_pred"].detach().cpu()
                    stress_gt_cpu = stress_gt.detach().cpu()
                    flow_gt_cpu = flow_gt.detach().cpu()
                    force_gt_cpu = force_gt.detach().cpu()
                    object_gt_cpu = object_gt.detach().cpu()
                    sample_ids = [str(s) for s in batch["sample_id"]]
                    gt_actions = batch.get("action_name", [""] * len(sample_ids))
                    action_label_cpu = action_label.detach().cpu()

                    for i, sid in enumerate(sample_ids):
                        item = {
                            "sample_id": sid,
                            "action_gt_name": str(gt_actions[i]) if i < len(gt_actions) else "",
                            "action_gt_label": int(action_label_cpu[i].item()),
                            "action_pred_label": int(action_pred[i].item()),
                            "action_prob": probs[i].tolist(),
                            "param_pred_raw": param_raw[i].tolist(),
                            "object_mask_mean": float(object_gt_cpu[i].mean().item()),
                        }
                        per_sample_preds.append(item)

                        sdir = videos_dir / sid
                        v_lim = min(int(eval_max_video_views), int(stress_pred.shape[1]))
                        for vi in range(v_lim):
                            # 统一导出两套对比视频：pred_raw / gt（同一固定 value range）
                            modal_triplets = (
                                ("stress", stress_pred[i, vi], stress_gt_cpu[i, vi]),
                                ("flow", flow_pred[i, vi], flow_gt_cpu[i, vi]),
                                ("force", force_pred[i, vi], force_gt_cpu[i, vi]),
                            )
                            for modal_name, pred_tchw, gt_tchw in modal_triplets:
                                pred_raw_tc_hw = pred_tchw.permute(1, 0, 2, 3).contiguous()  # [T,3,H,W]
                                gt_tc_hw = gt_tchw.permute(1, 0, 2, 3).contiguous()
                                _save_rgb_video_mp4(
                                    _tensor_video_to_u8_rgb_frames(
                                        pred_raw_tc_hw,
                                        colormap=eval_video_colormap,
                                        value_min=eval_video_value_min,
                                        value_max=eval_video_value_max,
                                    ),
                                    sdir / f"{modal_name}_pred_raw_view{vi:02d}.mp4",
                                    fps=eval_video_fps,
                                )
                                _save_rgb_video_mp4(
                                    _tensor_video_to_u8_rgb_frames(
                                        gt_tc_hw,
                                        colormap=eval_video_colormap,
                                        value_min=eval_video_value_min,
                                        value_max=eval_video_value_max,
                                    ),
                                    sdir / f"{modal_name}_gt_view{vi:02d}.mp4",
                                    fps=eval_video_fps,
                                )

                    e_steps += 1
                    if e_steps >= eval_batches:
                        break
            eval_obj = {
                "epoch": int(epoch + 1),
                "eval_batches": int(min(e_steps, eval_batches)),
                "avg_loss": float(e_total / max(e_n, 1)),
                "avg_action_acc": float(e_action_acc / max(e_n, 1)),
                "num_samples": int(e_n),
                "video_value_range": [float(eval_video_value_min), float(eval_video_value_max)],
                "video_root": str((eval_dir / f"epoch_{epoch + 1:04d}" / "videos").resolve()),
                "per_sample_predictions": per_sample_preds,
            }
            (eval_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                json.dumps(eval_obj, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (eval_dir / "last.json").write_text(
                json.dumps(eval_obj, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            model.train()


if __name__ == "__main__":
    main()

