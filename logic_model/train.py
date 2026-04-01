from __future__ import annotations

"""
单机多卡训练（DDP）示例：在 **Phys 仓库根目录** 执行，配置文件在 ``logic_model/configs/`` 下::

    torchrun --nproc_per_node=8 -m logic_model.train --config logic_model/configs/logic_train_dataset_mask_1000.json

单卡::

    python -m logic_model.train --config logic_model/configs/logic_train_dataset_mask_1000.json

若在 ``Phys/logic_model`` 目录下执行，可用 ``--config configs/logic_train_dataset_mask_1000.json``。

**Eval 与多卡**：eval 仅计算 loss（总 loss 与各分项），不写图或视频。默认仅 **rank0** 跑 eval，其它 rank 在 ``dist.barrier()`` 等待；若设 ``train.eval_use_distributed_sampler: true``，则各卡分片前向后 ``all_reduce`` 聚合指标。

**AMP**：在 **CUDA** 上默认启用 **fp16**（``torch.cuda.amp.autocast`` + ``GradScaler``）。关闭：CLI ``--no_amp`` 或 JSON ``"use_amp": false``。

**DataLoader（CUDA）**：``pin_memory=True``、``.to(..., non_blocking=True)``；若 ``train.num_workers`` > 0 则 ``persistent_workers=True``，``prefetch_factor`` 见 ``train.prefetch_factor``（默认 2，且至少为 2）。
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch
from logic_model.losses import (
    Arch4LossConfig,
    Arch4RegressionLoss,
    PhysicsConsistencyConfig,
    action_classification_loss,
    compute_physics_consistency_losses,
)
from logic_model.model import LogicPhysModel
from logic_model.model2 import LogicPhysModel2


def _pick(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


def _cli_or_pick_cfg(arg_val: Any, cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    """
    配置读取优先级（与仅用 argparse 默认值冲突时以本函数为准）：
    1) 命令行显式传入（非 None）→ 用命令行
    2) 否则若 ``cfg[key]`` 存在且非 None → 用配置
    3) 否则 ``fallback``

    说明：若 CLI 某项 ``default`` 为非 None 的常数，则无法区分「用户未传」与「用户传了默认值」；
    因此对依赖 JSON 配置的项，CLI 默认改为 None，未传时才读 config。
    """
    if arg_val is not None:
        return arg_val
    return _pick(cfg, key, fallback)


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be json object: {path}")
    return cfg


def _resolve_path_relative_to_config(rel: str, config_path: str | None) -> Path:
    """
    相对路径解析顺序：
    1) 相对 **配置文件所在目录**（如 ``logic_model/configs/*.json`` 旁的兄弟路径）；
    2) 相对 **当前工作目录**（便于 ``data.split_root`` 写 ``auto_output/...`` 时在 Phys 根目录运行）。
    """
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


def _default_project_torchhub_dir() -> Path:
    # Phys/logic_model/train.py -> Phys/.torch/hub
    return (Path(__file__).resolve().parent.parent / ".torch" / "hub").resolve()


def _resolve_torchhub_dir(cfg_model: Dict[str, Any]) -> Path:
    cfg_dir = str(_pick(cfg_model, "torchhub_dir", "") or "").strip()
    if cfg_dir:
        p = Path(cfg_dir).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (Path.cwd() / p).resolve()
    env_home = str(os.environ.get("TORCH_HOME", "") or "").strip()
    if env_home:
        return (Path(env_home).expanduser().resolve() / "hub").resolve()
    return _default_project_torchhub_dir()


def _configure_torchhub_dir(hub_dir: Path) -> Path:
    hub_dir = Path(hub_dir).expanduser().resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(hub_dir.parent)
    torch.hub.set_dir(str(hub_dir))
    return hub_dir


def _expected_torchhub_repo_dir(repo: str) -> str:
    # torch.hub: "owner/name:ref" -> "owner_name_ref"
    s = str(repo).replace(":", "_").replace("/", "_")
    return s


def _expected_dino_ckpt_basename(backbone_name: str) -> str:
    name = str(backbone_name).strip().lower()
    table = {
        "dinov2_vits14": "dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
        "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
        "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
    }
    return table.get(name, "")


def _prewarm_torchhub_dino(cfg_model: Dict[str, Any], hub_dir: Path) -> None:
    """
    仅做本地缓存检查，不触发在线下载。
    """
    source = str(_pick(cfg_model, "dino_backbone_source", "torchhub")).lower().strip()
    if source not in ("torchhub", "hub"):
        return
    repo = str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main"))
    repo_dir = hub_dir / _expected_torchhub_repo_dir(repo)
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"torchhub repo 缓存缺失: {repo_dir}。请先手动准备本地 hub 缓存（不走在线下载）。"
        )

    if bool(_pick(cfg_model, "dino_backbone_pretrained", True)):
        ckpt_name = _expected_dino_ckpt_basename(str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")))
        if ckpt_name:
            ckpt_path = hub_dir / "checkpoints" / ckpt_name
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"DINO checkpoint 缓存缺失: {ckpt_path}。请手动下载到 Phys/.torch/hub/checkpoints。"
                )


def _init_distributed() -> Tuple[bool, int, int, int]:
    """仅解析 WORLD_SIZE/RANK/LOCAL_RANK，返回 (distributed, rank, local_rank, world_size)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, local_rank, world_size


def _safe_barrier(local_rank: Optional[int] = None) -> None:
    """
    分布式同步的安全包装：
    - 未初始化/单卡时直接返回
    - CUDA 多卡时显式传 device_ids，避免 NCCL barrier 设备推断异常
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    backend = str(dist.get_backend())
    if backend == "nccl":
        # 某些环境的 NCCL barrier 会触发 CUDA invalid argument；
        # 用 1 元 all_reduce 作为等价同步更稳。
        if not torch.cuda.is_available():
            dist.barrier()
            return
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.zeros(1, device=dev, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return
    dist.barrier()


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


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


def _compute_eval_batch_losses(
    model: torch.nn.Module,
    ev_m: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    loss_reg: Arch4RegressionLoss,
    phys_cfg: PhysicsConsistencyConfig,
    *,
    lambda_stress: float,
    lambda_flow: float,
    lambda_force: float,
    lambda_action: float,
    lambda_phys: float,
    object_mask_fg_weight: float,
    object_mask_bg_weight: float,
    object_mask_bg_black: bool,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """eval 前向一步，返回总 loss 与各分项（与训练时加权方式一致）。"""
    rgb = batch["rgb"].to(device, non_blocking=non_blocking)
    stress_gt = batch["stress"].to(device, non_blocking=non_blocking)
    flow_gt = batch["flow"].to(device, non_blocking=non_blocking)
    force_gt = batch["force_mask"].to(device, non_blocking=non_blocking)
    object_gt = batch["object_mask"].to(device, non_blocking=non_blocking)
    params_gt_raw = batch["params"].to(device, non_blocking=non_blocking)
    action_label = batch["action_label"].to(device, non_blocking=non_blocking)

    if int(ev_m.in_channels) == 1:
        x = rgb[:, :, :1, :, :, :]
    else:
        x = rgb[:, :, : int(ev_m.in_channels), :, :, :]
    if int(x.shape[3]) != int(ev_m.num_frames):
        x = _resample_time_bvcthw(x, int(ev_m.num_frames))
        stress_gt = _resample_time_bvcthw(stress_gt, int(ev_m.num_frames))
        flow_gt = _resample_time_bvcthw(flow_gt, int(ev_m.num_frames))
        force_gt = _resample_time_bvcthw(force_gt, int(ev_m.num_frames))
        object_gt = _resample_time_bvcthw(object_gt, int(ev_m.num_frames))

    _amp = autocast(dtype=torch.float16, enabled=bool(use_amp and device.type == "cuda"))
    with _amp:
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
    parts = {
        "loss_reg": loss_reg_part,
        "loss_stress": loss_stress_part,
        "loss_flow": loss_flow_part,
        "loss_force": loss_force_part,
        "loss_action": action_ret["loss_action"],
        "loss_phys_total": phys_ret["loss_phys_total"],
        "loss_phys_sf": phys_ret["loss_stress_flow_consistency"],
        "loss_phys_fs": phys_ret["loss_force_stress_consistency"],
        "loss_phys_ff": phys_ret["loss_force_flow_consistency"],
        "action_acc": action_ret["action_acc"],
    }
    return loss_total, parts


def _eval_sums_to_record(
    sums: List[float],
    *,
    lambda_stress: float,
    lambda_flow: float,
    lambda_force: float,
    lambda_action: float,
    lambda_phys: float,
) -> Dict[str, Any]:
    """将 eval 累积向量转为 JSON/日志用 dict；sums[0..10] 为加权和，sums[11]=样本数。"""
    n = max(float(sums[11]), 1.0)

    def a(i: int) -> float:
        return float(sums[i]) / n

    return {
        "avg_loss": a(0),
        "avg_loss_reg": a(1),
        "avg_loss_stress": a(2),
        "avg_loss_flow": a(3),
        "avg_loss_force": a(4),
        "avg_loss_action": a(5),
        "avg_loss_phys_total": a(6),
        "avg_loss_phys_sf": a(7),
        "avg_loss_phys_fs": a(8),
        "avg_loss_phys_ff": a(9),
        "avg_action_acc": a(10),
        "weighted_reg": a(1),
        "weighted_stress": float(lambda_stress) * a(2),
        "weighted_flow": float(lambda_flow) * a(3),
        "weighted_force": float(lambda_force) * a(4),
        "weighted_action": float(lambda_action) * a(5),
        "weighted_phys": float(lambda_phys) * a(6),
        "num_samples": int(round(n)),
    }


def _tensorboard_log_eval_metrics(tb: SummaryWriter, m: Dict[str, Any], epoch_1based: int) -> None:
    for key in (
        "avg_loss",
        "avg_loss_reg",
        "avg_loss_stress",
        "avg_loss_flow",
        "avg_loss_force",
        "avg_loss_action",
        "avg_loss_phys_total",
        "avg_loss_phys_sf",
        "avg_loss_phys_fs",
        "avg_loss_phys_ff",
        "avg_action_acc",
        "weighted_reg",
        "weighted_stress",
        "weighted_flow",
        "weighted_force",
        "weighted_action",
        "weighted_phys",
    ):
        if key in m:
            tb.add_scalar(f"eval/{key}", float(m[key]), epoch_1based)


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
    ap.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="数据根目录；缺省时用 config 的 data.split_root",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="未传时读 train.epochs；仍缺省则为 1000",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="未传时读 train.batch_size；仍缺省则为 1",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="未传时读 train.max_samples；>0 仅前 N 个样本；0=全部",
    )
    ap.add_argument("--lr", type=float, default=None, help="未传时读 train.lr；缺省 3e-4")
    ap.add_argument("--num_workers", type=int, default=None, help="未传时读 train.num_workers；缺省 0")
    ap.add_argument("--max_views", type=int, default=None, help="未传时读 model.num_views；缺省 4")
    ap.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="未传时读 model.num_frames；0=按 LMDB；CLI 缺省视为未传",
    )
    ap.add_argument("--img_size", type=int, default=None, help="未传时读 model.img_size")
    ap.add_argument("--dec_h", type=int, default=None, help="未传时读 model.dec_h；缺省 112")
    ap.add_argument("--dec_w", type=int, default=None, help="未传时读 model.dec_w；缺省 112")
    ap.add_argument("--device", type=str, default=None, help="未传时读 train.device；缺省 cuda")
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="覆盖 train.output_root（如多副本并行时按副本区分目录）；未传时读 JSON",
    )
    # loss weights（未传时读 train.*；显式 CLI 覆盖 JSON）
    ap.add_argument("--lambda_stress", type=float, default=None)
    ap.add_argument("--lambda_flow", type=float, default=None)
    ap.add_argument("--lambda_force", type=float, default=None)
    ap.add_argument("--lambda_action", type=float, default=None)
    ap.add_argument("--lambda_phys", type=float, default=None)
    ap.add_argument("--use_pred_force_mask_for_phys", action="store_true")
    ap.add_argument(
        "--save_every_epochs",
        type=int,
        default=None,
        help="覆盖 JSON：仅建议写 train.checkpoint.save_every_epochs；未传 CLI 时只读 checkpoint.save_every_epochs，再缺省 100000",
    )
    ap.add_argument(
        "--eval_every_epochs",
        type=int,
        default=None,
        help="未传时读 train.eval_every_epochs；缺省 100；旧配置仅设 quick_eval.every_epochs 时仍兼容并告警",
    )
    ap.add_argument(
        "--eval_batches",
        type=int,
        default=None,
        help="未传时读 train.eval_max_batches；0=跑满 eval_loader；CLI 缺省同未传",
    )
    ap.add_argument("--object_mask_fg_weight", type=float, default=None, help="未传时读 train.object_mask_fg_weight")
    ap.add_argument("--object_mask_bg_weight", type=float, default=None, help="未传时读 train.object_mask_bg_weight")
    ap.add_argument("--object_mask_bg_black", type=int, default=None, help="未传时读 train.object_mask_bg_black")
    ap.add_argument(
        "--ddp",
        action="store_true",
        help="显式要求 DDP（需 torchrun WORLD_SIZE>1；一般可不写，自动检测）",
    )
    ap.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭 CUDA AMP（默认训练在 GPU 上使用 fp16 autocast + GradScaler）",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    cfg_train = cfg.get("train") or {}

    split_root = (args.split_root or "").strip() or str(_pick(cfg_data, "split_root", "") or "").strip()
    if not split_root:
        split_root = "auto_output/dataset_deformation_stress_500_new/train"
    split_root = str(Path(split_root).expanduser())

    max_views = int(_cli_or_pick_cfg(args.max_views, cfg_model, "num_views", 4))
    num_frames = int(_cli_or_pick_cfg(args.num_frames, cfg_model, "num_frames", 0))
    img_size = int(_cli_or_pick_cfg(args.img_size, cfg_model, "img_size", 0))
    dec_h = int(_cli_or_pick_cfg(args.dec_h, cfg_model, "dec_h", 112))
    dec_w = int(_cli_or_pick_cfg(args.dec_w, cfg_model, "dec_w", 112))

    lmdb_env_subdir = str(
        _pick(cfg_data, "lmdb_env_subdir", "arch4_data.lmdb") or "arch4_data.lmdb"
    ).strip() or "arch4_data.lmdb"

    train_ids_json = cfg_data.get("train_ids_json") or cfg_train.get("train_ids_json")
    train_id_list: Optional[List[str]] = None
    test_id_list: Optional[List[str]] = None
    split_meta: Dict[str, Any] = {}
    if train_ids_json:
        jp = _resolve_path_relative_to_config(str(train_ids_json), args.config)
        if not jp.is_file():
            raise FileNotFoundError(f"train_ids_json 不存在: {jp}")
        split_meta = _load_split_json(jp)
        train_id_list = [str(x) for x in (split_meta.get("train_ids") or [])]
        test_id_list = [str(x) for x in (split_meta.get("test_ids") or [])]
        sr_json = str(split_meta.get("split_root") or "").strip()
        if sr_json and not (args.split_root or "").strip() and not cfg_data.get("split_root"):
            split_root = sr_json
        lm = split_meta.get("lmdb_env_subdir")
        if lm and not cfg_data.get("lmdb_env_subdir"):
            lmdb_env_subdir = str(lm).strip() or lmdb_env_subdir

    if args.max_samples is not None:
        max_samples = int(args.max_samples)
    elif cfg_train.get("max_samples") is not None:
        max_samples = int(cfg_train["max_samples"])
    elif bool(cfg_train.get("debug_overfit")) and cfg_train.get("overfit_num_samples") is not None:
        max_samples = int(cfg_train["overfit_num_samples"])
    else:
        max_samples = 0

    ds = LmdbGtDataset(
        split_root=split_root,
        lmdb_env_subdir=lmdb_env_subdir,
        max_views=max_views,
        num_frames=(None if num_frames <= 0 else num_frames),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=train_id_list,
    )
    train_source = (
        Subset(ds, list(range(min(int(max_samples), len(ds)))))
        if int(max_samples) > 0
        else ds
    )
    shuffle_train = not bool(cfg_train.get("overfit_no_shuffle", False))
    bs = int(_cli_or_pick_cfg(args.batch_size, cfg_train, "batch_size", 1))
    nw = int(_cli_or_pick_cfg(args.num_workers, cfg_train, "num_workers", 0))

    distributed, rank, local_rank, world_size = _init_distributed()
    if bool(getattr(args, "ddp", False)) and not distributed:
        raise ValueError("--ddp 已指定但未检测到 WORLD_SIZE>1，请使用 torchrun 启动")

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_source,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
        )

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 训练需要 CUDA")
        # 先绑定当前进程的本地 GPU，再初始化进程组，避免后续 collective 使用错误设备。
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_available():
            raise RuntimeError("torch.distributed 不可用")
        if not dist.is_initialized():
            backend = str(os.environ.get("DIST_BACKEND", "nccl")).strip() or "nccl"
            init_kwargs: Dict[str, Any] = {"backend": backend, "init_method": "env://"}
            # torch 新版本支持 device_id；支持时可减少 "No device id ..." 警告。
            try:
                dist.init_process_group(**init_kwargs, device_id=device)
            except TypeError:
                dist.init_process_group(**init_kwargs)
    else:
        device_str = str(_cli_or_pick_cfg(args.device, cfg_train, "device", "cuda"))
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        elif device_str in ("cuda", "auto"):
            picked_gpu = _auto_pick_least_utilized_gpu()
            if picked_gpu is None:
                picked_gpu = 0
            device_str = f"cuda:{picked_gpu}"
        device = torch.device(device_str)

    pin_memory = device.type == "cuda"
    nb = pin_memory
    prefetch_factor = max(2, int(_pick(cfg_train, "prefetch_factor", 2)))
    _dl_extras: Dict[str, Any] = {}
    if nw > 0:
        _dl_extras["persistent_workers"] = True
        _dl_extras["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        train_source,
        batch_size=bs,
        shuffle=(shuffle_train and not distributed),
        sampler=train_sampler,
        num_workers=nw,
        collate_fn=collate_lmdb_gt_batch,
        drop_last=False,
        pin_memory=pin_memory,
        **_dl_extras,
    )

    eval_split = str(_pick(cfg_train, "eval_split", "train")).strip().lower()
    if eval_split == "test":
        if not test_id_list:
            if rank == 0:
                print(
                    "[logic_train] WARN eval_split=test 但 split json 无 test_ids，"
                    "回退为 eval_split=train"
                )
            eval_split = "train"
    eval_use_distributed_sampler = bool(_pick(cfg_train, "eval_use_distributed_sampler", False))
    eval_sharded = eval_use_distributed_sampler and distributed
    eval_ds = train_source
    if eval_split == "test" and test_id_list:
        ds_eval = LmdbGtDataset(
            split_root=split_root,
            lmdb_env_subdir=lmdb_env_subdir,
            max_views=max_views,
            num_frames=(None if num_frames <= 0 else num_frames),
            img_size=(None if img_size <= 0 else img_size),
            return_action_name=True,
            sample_ids=test_id_list,
            action_to_id=ds.action_to_id,
        )
        eval_ds = ds_eval
    eval_sampler_eval: Optional[DistributedSampler] = None
    if eval_sharded:
        eval_sampler_eval = DistributedSampler(
            eval_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=bs,
        shuffle=False,
        sampler=eval_sampler_eval,
        num_workers=nw,
        collate_fn=collate_lmdb_gt_batch,
        pin_memory=pin_memory,
        **_dl_extras,
    )

    if bool(getattr(args, "no_amp", False)):
        use_amp = False
    elif device.type != "cuda":
        use_amp = False
    else:
        use_amp = bool(_pick(cfg_train, "use_amp", True))

    if num_frames > 0:
        model_num_frames = int(num_frames)
    else:
        # full-frames 模式下，使用第一个样本帧数作为模型固定时间长度
        sample0 = ds[0]
        model_num_frames = int(sample0["rgb"].shape[2])

    model_arch = str(_pick(cfg_model, "arch", "logic_v1")).strip().lower()
    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        hub_dir = _configure_torchhub_dir(_resolve_torchhub_dir(cfg_model))
        if rank == 0 and bool(_pick(cfg_train, "torchhub_log_dir", True)):
            print(f"[logic_train] torchhub_dir={torch.hub.get_dir()}", flush=True)
            print(f"[logic_train] TORCH_HOME={os.environ.get('TORCH_HOME', '')}", flush=True)
        do_prewarm = bool(_pick(cfg_train, "torchhub_prewarm", True))
        require_cache = bool(_pick(cfg_train, "torchhub_require_cache", True))
        if distributed and do_prewarm:
            if rank == 0:
                if require_cache:
                    _prewarm_torchhub_dino(cfg_model, hub_dir)
            _safe_barrier(local_rank if distributed else None)
        elif (not distributed) and do_prewarm:
            if require_cache:
                _prewarm_torchhub_dino(cfg_model, hub_dir)
    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        model = LogicPhysModel2(
            num_views=max_views,
            in_channels=int(_pick(cfg_model, "in_channels", 3)),
            num_frames=int(_pick(cfg_model, "num_frames", model_num_frames)),
            img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
            num_targets=4,
            num_actions=ds.num_actions,
            dec_h=dec_h,
            dec_w=dec_w,
            fusion_dim=int(_pick(cfg_model, "fusion_dim", 512)),
            fusion_heads=int(_pick(cfg_model, "fusion_heads", 8)),
            head_dropout=float(_pick(cfg_model, "head_dropout", 0.1)),
            use_uncertainty=bool(_pick(cfg_model, "use_uncertainty", False)),
            bottleneck_dim=int(_pick(cfg_model, "bottleneck_dim", 128)),
            dino_backbone_name=str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")),
            dino_backbone_pretrained=bool(_pick(cfg_model, "dino_backbone_pretrained", True)),
            dino_backbone_source=str(_pick(cfg_model, "dino_backbone_source", "torchhub")),
            dino_out_dim=int(_pick(cfg_model, "dino_out_dim", 384)),
            temporal_adapter_type=str(_pick(cfg_model, "temporal_adapter_type", "transformer")),
            temporal_adapter_layers=int(_pick(cfg_model, "temporal_adapter_layers", 2)),
            temporal_adapter_heads=int(_pick(cfg_model, "temporal_adapter_heads", 6)),
            temporal_adapter_dropout=float(_pick(cfg_model, "temporal_adapter_dropout", 0.1)),
            frame_pool=str(_pick(cfg_model, "frame_pool", "mean")),
            freeze_backbone=bool(_pick(cfg_model, "freeze_backbone", True)),
            torchhub_dir=str(torch.hub.get_dir()),
            dino_torchhub_repo=str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main")),
            dino_force_reload=bool(_pick(cfg_model, "dino_force_reload", False)),
            dino_trust_repo=bool(_pick(cfg_model, "dino_trust_repo", True)),
            dino_skip_validation=bool(_pick(cfg_model, "dino_skip_validation", True)),
            dino_hub_verbose=bool(_pick(cfg_model, "dino_hub_verbose", False)),
            dino_log_torchhub_dir=bool(_pick(cfg_model, "dino_log_torchhub_dir", False)),
        ).to(device)
    else:
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
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    base_lr = float(_cli_or_pick_cfg(args.lr, cfg_train, "lr", 3e-4))
    use_param_groups = bool(_pick(cfg_train, "use_param_groups", False))
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    wd_backbone: Optional[float] = None
    wd_head: Optional[float] = None
    if use_param_groups and model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        head_lr = float(_pick(cfg_train, "head_lr", base_lr))
        backbone_lr = float(_pick(cfg_train, "backbone_lr", head_lr * 0.1))
        wd_head = float(_pick(cfg_train, "weight_decay_head", 0.01))
        wd_backbone = float(_pick(cfg_train, "weight_decay_backbone", wd_head))

        mopt = _unwrap_model(model)
        backbone_params: List[torch.nn.Parameter] = []
        head_params: List[torch.nn.Parameter] = []
        for n, p in mopt.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("frame_encoder."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": wd_backbone}
            )
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": head_lr, "weight_decay": wd_head}
            )
        opt = torch.optim.AdamW(param_groups, lr=head_lr)
    else:
        if use_param_groups and rank == 0 and model_arch not in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
            print(
                f"[logic_train] WARN use_param_groups=true 但 model_arch={model_arch} 非 dino，回退为单学习率 AdamW。",
                flush=True,
            )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
        )
    scaler = GradScaler(enabled=use_amp)

    loss_reg = Arch4RegressionLoss(Arch4LossConfig())
    phys_cfg = PhysicsConsistencyConfig(
        # 仅保留 lambda_phys 作为总权重，三个 physics 子项固定等权。
        lambda_stress_flow_consistency=1.0,
        lambda_force_stress_consistency=1.0,
        lambda_force_flow_consistency=1.0,
        use_pred_force_mask=bool(_pick(cfg_train, "use_pred_force_mask_for_phys", args.use_pred_force_mask_for_phys)),
    )

    lambda_stress = float(_cli_or_pick_cfg(args.lambda_stress, cfg_train, "lambda_stress", 10.0))
    lambda_flow = float(_cli_or_pick_cfg(args.lambda_flow, cfg_train, "lambda_flow", 10.0))
    lambda_force = float(_cli_or_pick_cfg(args.lambda_force, cfg_train, "lambda_force", 10.0))
    lambda_action = float(_cli_or_pick_cfg(args.lambda_action, cfg_train, "lambda_action", 1.0))
    lambda_phys = float(_cli_or_pick_cfg(args.lambda_phys, cfg_train, "lambda_phys", 0.001))
    object_mask_fg_weight = float(_cli_or_pick_cfg(args.object_mask_fg_weight, cfg_train, "object_mask_fg_weight", 10.0))
    object_mask_bg_weight = float(_cli_or_pick_cfg(args.object_mask_bg_weight, cfg_train, "object_mask_bg_weight", 1.0))
    object_mask_bg_black = bool(
        int(_cli_or_pick_cfg(args.object_mask_bg_black, cfg_train, "object_mask_bg_black", 1))
    )

    train_out_cli = str(args.output_root or "").strip()
    if train_out_cli:
        train_out = train_out_cli
    else:
        train_out = str(_pick(cfg_train, "output_root", "") or "").strip()
    if train_out:
        output_root = Path(train_out).expanduser().resolve()
    else:
        output_root = (Path("logic_model") / "output").resolve()
    ckpt_dir = output_root / "checkpoints"
    eval_dir = output_root / "eval"
    tensorboard_dir = output_root / "tensorboard"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tb_writer: Optional[SummaryWriter] = None
    if rank == 0:
        tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    ckpt_cfg = cfg_train.get("checkpoint")
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}
    if args.save_every_epochs is not None:
        save_every_epochs = max(1, int(args.save_every_epochs))
    else:
        sv_e = ckpt_cfg.get("save_every_epochs")
        if sv_e is not None:
            save_every_epochs = max(1, int(sv_e))
        else:
            save_every_epochs = max(1, int(100000))

    ev_e = cfg_train.get("eval_every_epochs")
    _eval_from_quick_eval = False
    if ev_e is None:
        qe = cfg_train.get("quick_eval")
        if isinstance(qe, dict) and qe.get("every_epochs") is not None:
            ev_e = qe["every_epochs"]
            _eval_from_quick_eval = True
    if args.eval_every_epochs is not None:
        eval_every_epochs = max(1, int(args.eval_every_epochs))
    else:
        eval_every_epochs = max(1, int(ev_e if ev_e is not None else 100))
    if rank == 0 and _eval_from_quick_eval:
        print(
            "[logic_train] WARN train.quick_eval.every_epochs 已废弃，请改用 train.eval_every_epochs",
            flush=True,
        )
    if args.eval_batches is not None:
        _eb = int(args.eval_batches)
        eval_max_batches = len(eval_loader) if _eb <= 0 else max(1, _eb)
    elif cfg_train.get("eval_max_batches") is not None:
        _emi = int(cfg_train["eval_max_batches"])
        eval_max_batches = len(eval_loader) if _emi <= 0 else max(1, _emi)
    else:
        eval_max_batches = len(eval_loader)
    epochs = int(_cli_or_pick_cfg(args.epochs, cfg_train, "epochs", 1000))
    if rank == 0:
        print(
            f"[logic_train] dataset_train={len(ds)} loader_samples={len(train_source)} "
            f"num_actions={ds.num_actions} eval_split={eval_split} "
            f"ddp={distributed} world_size={world_size} "
            f"lambdas(reg=1, stress={lambda_stress}, flow={lambda_flow}, force={lambda_force}, "
            f"action={lambda_action}, phys={lambda_phys})"
        )
        print(
            "[logic_train] "
            f"object_mask_loss(fg_w={object_mask_fg_weight}, bg_w={object_mask_bg_weight}, "
            f"bg_black={int(object_mask_bg_black)})"
        )
        print(f"[logic_train] action_to_id={json.dumps(ds.action_to_id, ensure_ascii=False)}")
        print(f"[logic_train] model_num_frames={_unwrap_model(model).num_frames}")
        print(f"[logic_train] device={device}")
        print(f"[logic_train] model_arch={model_arch}")
        print(f"[logic_train] amp_fp16={use_amp}")
        print(
            f"[logic_train] dataloader: num_workers={nw} pin_memory={pin_memory} "
            f"persistent_workers={nw > 0} prefetch_factor={prefetch_factor if nw > 0 else 'n/a'}"
        )
        print(f"[logic_train] output_root={output_root}")
        print(f"[logic_train] tensorboard_dir={tensorboard_dir}")
        print(
            f"[logic_train] eval: every {eval_every_epochs} epoch(s), "
            f"max_batches={eval_max_batches}, eval_sharded={eval_sharded}"
        )
        if use_param_groups:
            print(
                f"[logic_train] optimizer param_groups=True backbone_lr={backbone_lr} head_lr={head_lr} "
                f"wd_backbone={wd_backbone} wd_head={wd_head}",
                flush=True,
            )
        else:
            print(f"[logic_train] optimizer param_groups=False lr={base_lr}", flush=True)

    if rank == 0:
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
                        "distributed": distributed,
                        "world_size": world_size,
                        "rank": rank,
                        "split_root": split_root,
                        "lmdb_env_subdir": lmdb_env_subdir,
                        "train_ids_json": str(train_ids_json or ""),
                        "n_train_ids": len(train_id_list or []),
                        "n_test_ids": len(test_id_list or []),
                        "max_samples": int(max_samples),
                        "eval_split": eval_split,
                        "max_views": max_views,
                        "num_frames": num_frames,
                        "img_size": img_size,
                        "dec_h": dec_h,
                        "dec_w": dec_w,
                        "device": str(device),
                        "use_amp_fp16": bool(use_amp),
                        "dataloader_num_workers": int(nw),
                        "dataloader_pin_memory": bool(pin_memory),
                        "dataloader_persistent_workers": bool(nw > 0),
                        "dataloader_prefetch_factor": int(prefetch_factor) if nw > 0 else None,
                        "model_num_frames": int(_unwrap_model(model).num_frames),
                        "model_arch": str(model_arch),
                        "optimizer_use_param_groups": bool(use_param_groups),
                        "optimizer_base_lr": float(base_lr),
                        "optimizer_backbone_lr": (None if backbone_lr is None else float(backbone_lr)),
                        "optimizer_head_lr": (None if head_lr is None else float(head_lr)),
                        "optimizer_weight_decay_backbone": (None if wd_backbone is None else float(wd_backbone)),
                        "optimizer_weight_decay_head": (None if wd_head is None else float(wd_head)),
                        "output_root": str(output_root),
                        "tensorboard_dir": str(tensorboard_dir),
                        "epochs": int(epochs),
                        "save_every_epochs": int(save_every_epochs),
                        "eval_every_epochs": int(eval_every_epochs),
                        "eval_max_batches": int(eval_max_batches),
                        "eval_sharded": bool(eval_sharded),
                        "lambda_stress": float(lambda_stress),
                        "lambda_flow": float(lambda_flow),
                        "lambda_force": float(lambda_force),
                        "lambda_action": float(lambda_action),
                        "lambda_phys": float(lambda_phys),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    epoch_pbar = tqdm(
        range(epochs),
        desc="epoch",
        disable=rank != 0,
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        mcore = _unwrap_model(model)
        train_sums = [0.0] * 12
        train_pbar = tqdm(
            loader,
            desc=f"train {epoch + 1}/{epochs}",
            disable=rank != 0,
            total=len(loader),
            leave=False,
            dynamic_ncols=True,
        )
        last_end = time.perf_counter()
        for bi, batch in enumerate(train_pbar):
            t0 = time.perf_counter()
            data_time = t0 - last_end

            rgb = batch["rgb"].to(device, non_blocking=nb)
            stress_gt = batch["stress"].to(device, non_blocking=nb)
            flow_gt = batch["flow"].to(device, non_blocking=nb)
            force_gt = batch["force_mask"].to(device, non_blocking=nb)
            object_gt = batch["object_mask"].to(device, non_blocking=nb)
            params_gt_raw = batch["params"].to(device, non_blocking=nb)
            action_label = batch["action_label"].to(device, non_blocking=nb)

            # [B,V,3,T,H,W] -> [B,V,C,T,H,W]，兼容 in_channels
            if int(mcore.in_channels) == 1:
                x = rgb[:, :, :1, :, :, :]
            else:
                x = rgb[:, :, : int(mcore.in_channels), :, :, :]

            # 兼容 full-frames 数据读取：训练时对齐到模型固定 num_frames
            if int(x.shape[3]) != int(mcore.num_frames):
                x = _resample_time_bvcthw(x, int(mcore.num_frames))
                stress_gt = _resample_time_bvcthw(stress_gt, int(mcore.num_frames))
                flow_gt = _resample_time_bvcthw(flow_gt, int(mcore.num_frames))
                force_gt = _resample_time_bvcthw(force_gt, int(mcore.num_frames))
                object_gt = _resample_time_bvcthw(object_gt, int(mcore.num_frames))

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            with autocast(dtype=torch.float16, enabled=use_amp):
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
            if use_amp:
                scaler.scale(loss_total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            if rank == 0 and bi < 10:
                print(
                    f"[debug] bi={bi} data_time={data_time:.3f}s step_time={t2 - t1:.3f}s",
                    flush=True,
                )

            last_end = t2

            bsz = int(rgb.shape[0])
            train_sums[0] += float(loss_total.item()) * bsz
            train_sums[1] += float(loss_reg_part.item()) * bsz
            train_sums[2] += float(loss_stress_part.item()) * bsz
            train_sums[3] += float(loss_flow_part.item()) * bsz
            train_sums[4] += float(loss_force_part.item()) * bsz
            train_sums[5] += float(action_ret["loss_action"].item()) * bsz
            train_sums[6] += float(phys_ret["loss_phys_total"].item()) * bsz
            train_sums[7] += float(phys_ret["loss_stress_flow_consistency"].item()) * bsz
            train_sums[8] += float(phys_ret["loss_force_stress_consistency"].item()) * bsz
            train_sums[9] += float(phys_ret["loss_force_flow_consistency"].item()) * bsz
            train_sums[10] += float(action_ret["action_acc"].item()) * bsz
            train_sums[11] += float(bsz)

            if rank == 0:
                train_pbar.set_postfix(
                    loss=f"{float(loss_total.item()):.4f}",
                    reg=f"{float(loss_reg_part.item()):.4f}",
                    acc=f"{float(action_ret['action_acc'].item()):.3f}",
                )

        if distributed:
            t_stat = torch.tensor(train_sums, device=device, dtype=torch.float64)
            dist.all_reduce(t_stat, op=dist.ReduceOp.SUM)
            train_sums = [float(t_stat[i].item()) for i in range(12)]
        n = float(train_sums[11])
        denom = max(n, 1.0)
        avg_loss = float(train_sums[0] / denom)
        if rank == 0:
            epoch_pbar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                acc=f"{train_sums[10]/denom:.3f}",
            )
            print(
                f"[logic_train] epoch={epoch+1} avg_loss={avg_loss:.6f} "
                f"avg_reg={train_sums[1]/denom:.6f} avg_stress={train_sums[2]/denom:.6f} "
                f"avg_flow={train_sums[3]/denom:.6f} avg_force={train_sums[4]/denom:.6f} "
                f"avg_action={train_sums[5]/denom:.6f} avg_phys={train_sums[6]/denom:.6f} "
                f"avg_action_acc={train_sums[10]/denom:.4f}"
            )
            if tb_writer is not None:
                ep = int(epoch + 1)
                tb_writer.add_scalar("train/avg_loss", avg_loss, ep)
                tb_writer.add_scalar("train/avg_loss_reg", train_sums[1] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_stress", train_sums[2] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_flow", train_sums[3] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_force", train_sums[4] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_action", train_sums[5] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_total", train_sums[6] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_sf", train_sums[7] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_fs", train_sums[8] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_ff", train_sums[9] / denom, ep)
                tb_writer.add_scalar("train/avg_action_acc", train_sums[10] / denom, ep)
                tb_writer.add_scalar(
                    "train/weighted_stress", lambda_stress * train_sums[2] / denom, ep
                )
                tb_writer.add_scalar("train/weighted_flow", lambda_flow * train_sums[3] / denom, ep)
                tb_writer.add_scalar("train/weighted_force", lambda_force * train_sums[4] / denom, ep)
                tb_writer.add_scalar("train/weighted_action", lambda_action * train_sums[5] / denom, ep)
                tb_writer.add_scalar("train/weighted_phys", lambda_phys * train_sums[6] / denom, ep)

        # checkpoint（全部在 logic_model/output/checkpoints）
        if ((epoch + 1) % save_every_epochs) == 0 and rank == 0:
            model_hparams = {
                "arch": str(model_arch),
                "num_views": int(max_views),
                "in_channels": int(mcore.in_channels),
                "num_frames": int(mcore.num_frames),
                "dec_h": int(dec_h),
                "dec_w": int(dec_w),
            }
            if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
                model_hparams.update(
                    {
                        "dino_backbone_name": str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")),
                        "dino_backbone_pretrained": bool(_pick(cfg_model, "dino_backbone_pretrained", True)),
                        "dino_backbone_source": str(_pick(cfg_model, "dino_backbone_source", "torchhub")),
                        "dino_torchhub_repo": str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main")),
                        "torchhub_dir": str(torch.hub.get_dir()),
                        "dino_force_reload": bool(_pick(cfg_model, "dino_force_reload", False)),
                        "dino_trust_repo": bool(_pick(cfg_model, "dino_trust_repo", True)),
                        "dino_skip_validation": bool(_pick(cfg_model, "dino_skip_validation", True)),
                        "dino_hub_verbose": bool(_pick(cfg_model, "dino_hub_verbose", False)),
                        "dino_log_torchhub_dir": bool(_pick(cfg_model, "dino_log_torchhub_dir", False)),
                        "dino_out_dim": int(_pick(cfg_model, "dino_out_dim", 384)),
                        "temporal_adapter_type": str(_pick(cfg_model, "temporal_adapter_type", "transformer")),
                        "temporal_adapter_layers": int(_pick(cfg_model, "temporal_adapter_layers", 2)),
                        "temporal_adapter_heads": int(_pick(cfg_model, "temporal_adapter_heads", 6)),
                        "temporal_adapter_dropout": float(_pick(cfg_model, "temporal_adapter_dropout", 0.1)),
                        "frame_pool": str(_pick(cfg_model, "frame_pool", "mean")),
                        "freeze_backbone": bool(_pick(cfg_model, "freeze_backbone", True)),
                    }
                )

            ckpt = {
                "epoch": int(epoch + 1),
                "avg_loss": avg_loss,
                "model_state": mcore.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "action_to_id": ds.action_to_id,
                "model_hparams": model_hparams,
            }
            torch.save(ckpt, str(ckpt_dir / f"epoch_{epoch + 1:04d}.pt"))
            # 明确不维护 best/last checkpoint，避免误用历史 best.pt / last.pt。
            best_ckpt_path = ckpt_dir / "best.pt"
            if best_ckpt_path.exists():
                best_ckpt_path.unlink()
            last_ckpt_path = ckpt_dir / "last.pt"
            if last_ckpt_path.exists():
                last_ckpt_path.unlink()

        if distributed:
            _safe_barrier(local_rank if distributed else None)

        # eval：仅计算 loss（总 loss 与各分项），写入 eval/*.json 与 tensorboard
        if ((epoch + 1) % eval_every_epochs) == 0:
            if eval_sampler_eval is not None:
                eval_sampler_eval.set_epoch(epoch)

            if eval_sharded:
                model.eval()
                ev_m = _unwrap_model(model)
                with torch.no_grad():
                    sums_t = torch.zeros(12, device=device, dtype=torch.float64)
                    batch_steps = 0
                    _eval_cap_sh = min(len(eval_loader), eval_max_batches)
                    eval_pbar = tqdm(
                        eval_loader,
                        total=_eval_cap_sh,
                        desc=f"eval(sharded) e{epoch + 1}",
                        disable=rank != 0,
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for batch in eval_pbar:
                        loss_total, parts = _compute_eval_batch_losses(
                            model,
                            ev_m,
                            batch,
                            device,
                            loss_reg,
                            phys_cfg,
                            lambda_stress=lambda_stress,
                            lambda_flow=lambda_flow,
                            lambda_force=lambda_force,
                            lambda_action=lambda_action,
                            lambda_phys=lambda_phys,
                            object_mask_fg_weight=object_mask_fg_weight,
                            object_mask_bg_weight=object_mask_bg_weight,
                            object_mask_bg_black=object_mask_bg_black,
                            use_amp=use_amp,
                            non_blocking=nb,
                        )
                        bsz = int(batch["rgb"].shape[0])
                        sums_t[0] += float(loss_total.item()) * bsz
                        sums_t[1] += float(parts["loss_reg"].item()) * bsz
                        sums_t[2] += float(parts["loss_stress"].item()) * bsz
                        sums_t[3] += float(parts["loss_flow"].item()) * bsz
                        sums_t[4] += float(parts["loss_force"].item()) * bsz
                        sums_t[5] += float(parts["loss_action"].item()) * bsz
                        sums_t[6] += float(parts["loss_phys_total"].item()) * bsz
                        sums_t[7] += float(parts["loss_phys_sf"].item()) * bsz
                        sums_t[8] += float(parts["loss_phys_fs"].item()) * bsz
                        sums_t[9] += float(parts["loss_phys_ff"].item()) * bsz
                        sums_t[10] += float(parts["action_acc"].item()) * bsz
                        sums_t[11] += float(bsz)
                        batch_steps += 1
                        if rank == 0:
                            _sn = float(sums_t[11].item())
                            eval_pbar.set_postfix(
                                loss=f"{float(sums_t[0].item()/max(_sn,1.0)):.4f}",
                                acc=f"{float(sums_t[10].item()/max(_sn,1.0)):.3f}",
                            )
                        if batch_steps >= eval_max_batches:
                            break

                dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
                sums_list = [float(sums_t[i].item()) for i in range(12)]
                if rank == 0:
                    ep = int(epoch + 1)
                    metrics = _eval_sums_to_record(
                        sums_list,
                        lambda_stress=lambda_stress,
                        lambda_flow=lambda_flow,
                        lambda_force=lambda_force,
                        lambda_action=lambda_action,
                        lambda_phys=lambda_phys,
                    )
                    eval_obj: Dict[str, Any] = {
                        "epoch": ep,
                        "eval_mode": "distributed_sharded",
                        "eval_max_batches": int(eval_max_batches),
                        "eval_batches_run": int(min(batch_steps, eval_max_batches)),
                    }
                    eval_obj.update(metrics)
                    (eval_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                        json.dumps(eval_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    (eval_dir / "last.json").write_text(
                        json.dumps(eval_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    if tb_writer is not None:
                        _tensorboard_log_eval_metrics(tb_writer, metrics, ep)
                    print(
                        f"[logic_train] eval done (sharded): avg_loss={metrics['avg_loss']:.6f} "
                        f"avg_action_acc={metrics['avg_action_acc']:.4f} "
                        f"num_samples={metrics['num_samples']}",
                        flush=True,
                    )
                model.train()

            elif rank == 0:
                model.eval()
                ev_m = _unwrap_model(model)
                with torch.no_grad():
                    sums_list = [0.0] * 12
                    batch_steps = 0
                    _eval_cap = min(len(eval_loader), eval_max_batches)
                    eval_pbar = tqdm(
                        eval_loader,
                        total=_eval_cap,
                        desc=f"eval e{epoch + 1}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for batch in eval_pbar:
                        loss_total, parts = _compute_eval_batch_losses(
                            model,
                            ev_m,
                            batch,
                            device,
                            loss_reg,
                            phys_cfg,
                            lambda_stress=lambda_stress,
                            lambda_flow=lambda_flow,
                            lambda_force=lambda_force,
                            lambda_action=lambda_action,
                            lambda_phys=lambda_phys,
                            object_mask_fg_weight=object_mask_fg_weight,
                            object_mask_bg_weight=object_mask_bg_weight,
                            object_mask_bg_black=object_mask_bg_black,
                            use_amp=use_amp,
                            non_blocking=nb,
                        )
                        bsz = int(batch["rgb"].shape[0])
                        sums_list[0] += float(loss_total.item()) * bsz
                        sums_list[1] += float(parts["loss_reg"].item()) * bsz
                        sums_list[2] += float(parts["loss_stress"].item()) * bsz
                        sums_list[3] += float(parts["loss_flow"].item()) * bsz
                        sums_list[4] += float(parts["loss_force"].item()) * bsz
                        sums_list[5] += float(parts["loss_action"].item()) * bsz
                        sums_list[6] += float(parts["loss_phys_total"].item()) * bsz
                        sums_list[7] += float(parts["loss_phys_sf"].item()) * bsz
                        sums_list[8] += float(parts["loss_phys_fs"].item()) * bsz
                        sums_list[9] += float(parts["loss_phys_ff"].item()) * bsz
                        sums_list[10] += float(parts["action_acc"].item()) * bsz
                        sums_list[11] += float(bsz)
                        batch_steps += 1
                        _sn = sums_list[11]
                        eval_pbar.set_postfix(
                            loss=f"{sums_list[0]/max(_sn,1.0):.4f}",
                            acc=f"{sums_list[10]/max(_sn,1.0):.3f}",
                        )
                        if batch_steps >= eval_max_batches:
                            break

                ep = int(epoch + 1)
                metrics = _eval_sums_to_record(
                    sums_list,
                    lambda_stress=lambda_stress,
                    lambda_flow=lambda_flow,
                    lambda_force=lambda_force,
                    lambda_action=lambda_action,
                    lambda_phys=lambda_phys,
                )
                eval_obj = {
                    "epoch": ep,
                    "eval_mode": "rank0_only",
                    "eval_max_batches": int(eval_max_batches),
                    "eval_batches_run": int(min(batch_steps, eval_max_batches)),
                }
                eval_obj.update(metrics)
                (eval_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                    json.dumps(eval_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (eval_dir / "last.json").write_text(
                    json.dumps(eval_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if tb_writer is not None:
                    _tensorboard_log_eval_metrics(tb_writer, metrics, ep)
                print(
                    f"[logic_train] eval done: avg_loss={metrics['avg_loss']:.6f} "
                    f"avg_action_acc={metrics['avg_action_acc']:.4f} num_samples={metrics['num_samples']}",
                    flush=True,
                )
                model.train()

        if distributed:
            _safe_barrier(local_rank if distributed else None)

    if distributed:
        _safe_barrier(local_rank if distributed else None)
        dist.destroy_process_group()
    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
        print("[logic_train] training finished.", flush=True)


if __name__ == "__main__":
    main()

