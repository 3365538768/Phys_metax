"""
训练脚本：7:3 分割，训练物理参数预测模型
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Deque, List, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

try:
    from my_model.dataset import (
        Dataset400,
        PhysGaussianDataset,
        list_dataset400_sample_dirs,
        resolve_flat_dataset_root,
        train_test_split,
    )
    from my_model.model import create_model
except ImportError:
    from .dataset import (
        Dataset400,
        PhysGaussianDataset,
        list_dataset400_sample_dirs,
        resolve_flat_dataset_root,
        train_test_split,
    )
    from .model import create_model


def _log_params(params: torch.Tensor) -> torch.Tensor:
    """对 E, density, yield_stress 做 log1p 变换，便于回归"""
    e = torch.log1p(torch.clamp(params[:, 0], min=0))
    nu = params[:, 1]
    density = torch.log1p(torch.clamp(params[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(params[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _inv_log_params(params: torch.Tensor) -> torch.Tensor:
    """逆变换"""
    e = torch.expm1(params[:, 0])
    nu = params[:, 1]
    density = torch.expm1(params[:, 2])
    yield_stress = torch.expm1(params[:, 3])
    return torch.stack([e, nu, density, yield_stress], dim=1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_log_scale: bool = True,
    sync_loss_across_ranks: bool = False,
    use_amp_bf16: bool = False,
    epoch_index: int = 0,
    num_epochs: int = 1,
    show_progress: bool = True,
    non_blocking_copy: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    amp_enabled = use_amp_bf16 and device.type == "cuda"

    iterator = loader
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                loader,
                desc=f"Epoch {epoch_index + 1}/{num_epochs}",
                total=len(loader),
                unit="batch",
                dynamic_ncols=True,
                leave=True,
                mininterval=0.5,
            )
        except ImportError:
            pass

    for frames, params_gt, material_gt, action_gt in iterator:
        frames = frames.to(device, non_blocking=non_blocking_copy)
        params_gt = params_gt.to(device, non_blocking=non_blocking_copy)
        action_gt = action_gt.to(device, non_blocking=non_blocking_copy)

        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            params_pred, action_logits = model(frames)

            # 物理参数回归 loss（log 空间）
            if use_log_scale:
                params_gt_log = _log_params(params_gt)
                params_pred_log = _log_params(params_pred)
                physics_loss = F.mse_loss(params_pred_log, params_gt_log)
            else:
                physics_loss = F.mse_loss(params_pred, params_gt)

            # 动作分类 loss
            action_loss = F.cross_entropy(action_logits, action_gt)

            loss = physics_loss + action_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        n_samples += frames.size(0)

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

    if sync_loss_across_ranks and dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, float(n_samples)], device=device, dtype=torch.double)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float((t[0] / t[1]).item()) if t[1].item() > 0 else 0.0
    return total_loss / max(n_samples, 1)


def _is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _log_cuda_memory(device: torch.device, stage: str) -> None:
    """打印当前进程在该 device 上的显存占用（GiB）。"""
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    extra = ""
    try:
        free_b, total_b = torch.cuda.mem_get_info(idx)
        extra = f", 驱动视角 空闲 {free_b / (1024**3):.2f} / 总 {total_b / (1024**3):.2f} GiB"
    except Exception:
        pass
    rk = ""
    if _is_ddp():
        rk = f" rank={_ddp_rank()} local={_ddp_local_rank()}"
    print(
        f"[Train] CUDA{rk} {stage}: PyTorch已分配 {alloc:.2f} GiB, 缓存预留 {reserved:.2f} GiB{extra}",
        flush=True,
    )


def _pick_free_gpus(n: int) -> List[int]:
    """
    从 nvidia-smi 中选择当前最空闲的 n 张 GPU。
    判定“空闲卡”：utilization.gpu < 10 且 memory.used/memory.total < 0.10
    - 若空闲卡数量 >= n：只在空闲卡中按空闲度选前 n
    - 若空闲卡数量 < n：退化为在全部 GPU 中按空闲度选前 n（即使不满足阈值）

    空闲度排序规则：utilization.gpu 越低越优先，其次 memory.used/memory.total 越低越优先。
    """
    if n <= 0:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        raise RuntimeError("无法执行 nvidia-smi 来选择空闲 GPU") from exc

    rows: List[Tuple[int, float, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx = int(parts[0])
        util = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3]) if float(parts[3]) > 0 else 1.0
        mem_used_frac = mem_used / mem_total
        rows.append((idx, util, mem_used_frac))

    if not rows:
        raise RuntimeError("nvidia-smi 未返回任何 GPU 信息")

    rows.sort(key=lambda x: (x[1], x[2]))  # util asc, mem_used_frac asc

    free_rows = [r for r in rows if (r[1] < 10.0 and r[2] < 0.10)]
    if len(free_rows) >= n:
        picked = [r[0] for r in free_rows[:n]]
    else:
        picked = [r[0] for r in rows[:n]]
    return picked


def _candidate_gpu_sets(n: int, max_sets: int = 6) -> List[List[int]]:
    """
    生成候选 GPU 组合列表，用于 OOM 时自动切换。
    以当前空闲度排序后的 GPU 列表为基础，取滑动窗口的前 max_sets 个组合。
    """
    if n <= 0:
        return []
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: List[Tuple[int, float, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx = int(parts[0])
        util = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3]) if float(parts[3]) > 0 else 1.0
        rows.append((idx, util, mem_used / mem_total))
    if not rows:
        return []
    rows.sort(key=lambda x: (x[1], x[2]))
    gpu_order = [r[0] for r in rows]

    cand: List[List[int]] = []
    if len(gpu_order) <= n:
        return [gpu_order[:n]]
    for start in range(0, min(max_sets, len(gpu_order) - n + 1)):
        cand.append(gpu_order[start : start + n])
    return cand


def _relaunch_with_torchrun(args: argparse.Namespace) -> None:
    """
    当用户指定 --gpus N 且当前非 DDP 环境时，自动挑选 GPU 并用 torchrun 重新拉起。
    """
    if args.gpus is None or args.gpus <= 1:
        return
    if _is_ddp() or args._ddp_launched:
        return

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "my_model.train",
    ]

    # 透传参数（剔除内部参数）
    def build_passthrough(batch_size: int) -> List[str]:
        pt: List[str] = []
        for k, v in vars(args).items():
            if k in ("gpus", "_ddp_launched", "batch_size"):
                continue
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    pt.append(f"--{k}")
            else:
                pt.extend([f"--{k}", str(v)])
        pt.extend(["--batch_size", str(batch_size)])
        pt.append("--_ddp_launched")
        return pt

    # 候选 GPU 组合：优先最空闲组合；若训练阶段被抢占导致 OOM，则切换下一组组合重试
    gpu_sets = _candidate_gpu_sets(args.gpus, max_sets=6)
    if not gpu_sets:
        gpu_sets = [_pick_free_gpus(args.gpus)]

    batch_size = int(args.batch_size)
    tried = 0
    for gpu_set in gpu_sets:
        cuda_visible = ",".join(str(i) for i in gpu_set)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible

        # 同一组 GPU 下允许自动降 batch_size 重试（最多 2 次）
        for _bs_try in range(3):
            tried += 1
            passthrough = build_passthrough(batch_size)
            print(f"[DDP] 尝试 #{tried}: GPU={cuda_visible}, batch_size={batch_size}")
            print("[DDP] torchrun 命令: " + " ".join(cmd + passthrough))

            # 重要：实时转发输出，避免看起来“卡住”
            # 同时保留末尾日志用于判断 OOM
            tail_lines: Deque[str] = deque(maxlen=400)
            p = subprocess.Popen(
                cmd + passthrough,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert p.stdout is not None
            for line in p.stdout:
                print(line, end="")
                tail_lines.append(line)
            rc = p.wait()

            if rc == 0:
                raise SystemExit(0)

            combined = "".join(tail_lines)
            combined_lower = combined.lower()
            if "cuda out of memory" in combined_lower or "out of memory" in combined_lower:
                # 先尝试降 batch size；降到 1 仍 OOM，则换下一组 GPU
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"[DDP] 检测到 OOM，自动将 batch_size 降为 {batch_size} 并重试同一组 GPU")
                    continue
                else:
                    print("[DDP] batch_size 已降到 1 仍 OOM，切换下一组 GPU")
                    break

            # 非 OOM 错误：直接输出最后 2000 字符帮助定位，并退出
            tail = combined[-2000:]
            print("[DDP] torchrun 失败（非 OOM）。输出末尾：\n" + tail)
            raise SystemExit(rc)

    raise SystemExit(1)


def checkpoint_config_slug(args: argparse.Namespace) -> str:
    """
    写入 checkpoint 文件名的配置摘要，便于区分 num_frames / 分辨率 / 批次等实验。
    仅使用文件名安全字符（字母数字与下划线、连字符、点）。
    """
    if args.data_layout == "legacy":
        layout_tag = "legacy"
    else:
        stem = Path(str(args.dataset_dir)).name
        layout_tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)[:48] or "flat"
    lr_s = format(float(args.lr), "g").replace("/", "_")
    return (
        f"T{int(args.num_frames)}_H{int(args.img_size)}_bs{int(args.batch_size)}"
        f"_lr{lr_s}_ep{int(args.epochs)}_seed{int(args.seed)}_{layout_tag}"
    )


def build_train_hparams_dict(
    args: argparse.Namespace, ckpt_slug: str, dataset_layout_root: Optional[Path]
) -> dict:
    """随 checkpoint 一并保存，供 eval / 复现。"""
    out = {
        "num_frames": int(args.num_frames),
        "img_size": int(args.img_size),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "data_layout": args.data_layout,
        "arch": int(args.arch),
        "ckpt_config_slug": ckpt_slug,
        "dataset_dir": str(getattr(args, "dataset_dir", "dataset_400"))
        if args.data_layout == "dataset_400"
        else "",
        "dataset_root_resolved": str(dataset_layout_root) if dataset_layout_root is not None else "",
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="训练物理参数预测模型")
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="PhysGaussian 根目录",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--data_layout",
        type=str,
        choices=("dataset_400", "legacy"),
        default="dataset_400",
        help="dataset_400: 使用扁平集（见 --dataset_dir）；legacy: 原 auto_output/<action>/... 且 7:3 划分",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset_400",
        help="仅 data_layout=dataset_400：扁平集根目录。推荐只写 dataset_400（= auto_output 下子目录）；"
        "若写成 auto_output/dataset_400 也会正确解析；或给绝对路径。默认 dataset_400。",
    )
    parser.add_argument(
        "--arch",
        type=int,
        choices=(1, 2),
        default=1,
        help="1=单一物理头，2=多物理头（每标量独立 MLP）；checkpoint/visualizations 默认按序号分子目录",
    )
    parser.add_argument(
        "--no_dataset_preflight",
        action="store_true",
        help="dataset_400 快速启动：跳过多帧+cv2 预检与首帧 imread，仅检查 gt.json/images 及是否有图片文件",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="默认 my_model/checkpoints/<arch>/",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=0,
        help="每隔多少个 epoch 保存一次 checkpoint；0 表示不保存中间 checkpoint，只在最后一代保存",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="默认 my_model/runs/<arch>/",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="多卡训练使用的 GPU 数量 N（自动选择最空闲的 N 张卡并启用 DDP）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker 数；默认单卡 4、DDP 时 2（减轻多进程同时读盘导致多卡加速不明显）",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="num_workers>0 时每个 worker 预取的 batch 数（默认 4，减轻每 epoch 开头卡在 0%%；更占 CPU/内存）",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="关闭每个 epoch 的 tqdm 进度条（便于重定向日志到文件）",
    )
    parser.add_argument(
        "--amp_bf16",
        action="store_true",
        help="CUDA 下使用 torch.autocast(bfloat16) 混合精度训练（需 GPU 支持 BF16，如 Ampere 及以上）",
    )
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="开启 autograd anomaly detection（用于定位 inplace / nan 等问题）",
    )
    parser.add_argument(
        "--_ddp_launched",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # 若用户指定多卡数量且当前不是 DDP，则自动挑卡并 torchrun 重新拉起
    _relaunch_with_torchrun(args)

    root = Path(args.physgaussian_root).resolve()
    auto_output = root / "auto_output"
    dataset_layout_root: Optional[Path] = None
    if args.data_layout == "dataset_400":
        dataset_layout_root = resolve_flat_dataset_root(args.dataset_dir, auto_output)
    save_dir = root / (args.save_dir or f"my_model/checkpoints/{args.arch}")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_slug = checkpoint_config_slug(args)
    train_hparams = build_train_hparams_dict(args, ckpt_slug, dataset_layout_root)
    if not _is_ddp() or _ddp_rank() == 0:
        print(f"[Train] checkpoint 配置标签: {ckpt_slug}", flush=True)
    log_dir = root / (args.log_dir or f"my_model/runs/{args.arch}")
    if (not _is_ddp() or _ddp_rank() == 0):
        log_dir.mkdir(parents=True, exist_ok=True)

    if args.data_layout == "dataset_400":
        train_path = dataset_layout_root / "train"
        test_path = dataset_layout_root / "test"
        if not train_path.is_dir():
            raise SystemExit(
                f"未找到 {train_path}，请先运行 transform_dataset.py 或检查 --dataset_dir，或改用 --data_layout legacy"
            )
        if not _is_ddp() or _ddp_rank() == 0:
            print(f"[Train] 扁平数据集根目录: {dataset_layout_root}", flush=True)
        if not args.no_dataset_preflight and (not _is_ddp() or _ddp_rank() == 0):
            print(
                f"[Train] 正在对 {train_path} 做完整预检（多帧 PNG + resize），"
                "数据量大时可能数分钟无其它输出，属正常现象。",
                flush=True,
            )
            print(
                "[Train] 已加 --no_dataset_preflight 时将只做目录/文件检查，不再 cv2 读图，启动快得多。",
                flush=True,
            )
        train_dataset = Dataset400(
            train_path,
            num_frames=args.num_frames,
            img_size=args.img_size,
            preflight=not args.no_dataset_preflight,
            verbose_preflight=(not _is_ddp() or _ddp_rank() == 0),
        )
        n_test = len(list_dataset400_sample_dirs(test_path)) if test_path.is_dir() else 0
        print(
            f"数据: {args.dataset_dir} -> {dataset_layout_root} | 训练 {len(train_dataset)} 样本 | "
            f"测试目录约 {n_test} 条（eval 时请使用相同 --dataset_dir）"
        )
    else:
        train_samples, test_samples = train_test_split(
            auto_output, train_ratio=args.train_ratio, seed=args.seed
        )
        print(f"训练集: {len(train_samples)} 样本, 测试集: {len(test_samples)} 样本")
        train_dataset = PhysGaussianDataset(
            auto_output,
            num_frames=args.num_frames,
            img_size=args.img_size,
            sample_ids=train_samples,
        )

    # DDP 初始化
    if _is_ddp():
        dist.init_process_group(backend="nccl")
        local_rank = _ddp_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        device = torch.device(args.device)
        sampler = None
        shuffle = True

    num_workers = args.num_workers if args.num_workers is not None else (2 if _is_ddp() else 4)
    pin_mem = device.type == "cuda"

    loader_kw = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kw["prefetch_factor"] = max(2, int(args.prefetch_factor))

    train_loader = DataLoader(train_dataset, **loader_kw)

    if not _is_ddp() or _ddp_rank() == 0:
        ws = dist.get_world_size() if _is_ddp() else 1
        print(
            f"[Train] DDP world_size={ws}, 每卡 batch_size={args.batch_size}, "
            f"DataLoader num_workers={num_workers}, 等效全局 batch≈{ws * args.batch_size}",
            flush=True,
        )
        if ws > 1:
            print(
                "[Train] 若多卡与单卡每 epoch 耗时接近，多为磁盘/NFS 带宽或 PNG 解码瓶颈；"
                "可尝试 --num_workers 0/1、将数据放到本地 SSD，或增大 batch 提高 GPU 占比。"
            )

    use_amp_bf16 = bool(args.amp_bf16)
    if use_amp_bf16:
        if device.type != "cuda":
            use_amp_bf16 = False
            if not _is_ddp() or _ddp_rank() == 0:
                print("[Train] --amp_bf16 需要 CUDA，已关闭混合精度。")
        else:
            bf16_ok = True
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            if not bf16_ok:
                use_amp_bf16 = False
                if not _is_ddp() or _ddp_rank() == 0:
                    print("[Train] 当前环境不支持 CUDA bfloat16，已关闭 --amp_bf16。")
            elif not _is_ddp() or _ddp_rank() == 0:
                print("[Train] 已启用 bfloat16 autocast（torch.autocast）。")

    if not _is_ddp() or _ddp_rank() == 0:
        print("[Train] 正在构建模型并加载 ResNet18 权重（首次运行可能需下载）…", flush=True)
    model = create_model(num_frames=args.num_frames, arch=args.arch).to(device)
    if _is_ddp():
        # gradient_as_bucket_view=True 时，部分 ResNet 1x1 conv 的 grad 非 contiguous，会触发 stride 警告并略损 DDP 性能
        model = DDP(
            model,
            device_ids=[_ddp_local_rank()],
            output_device=_ddp_local_rank(),
            gradient_as_bucket_view=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 仅在 rank0 记录 loss 曲线、每 epoch 耗时和 TensorBoard
    train_losses = []
    epoch_seconds: List[float] = []
    writer = None
    if not _is_ddp() or _ddp_rank() == 0:
        writer = SummaryWriter(log_dir=str(log_dir))

    _log_cuda_memory(
        device,
        "训练开始前（仅模型权重等；首个 batch 反传后 Adam 状态与激活显存会再升高）",
    )

    if device.type == "cuda" and (not _is_ddp() or _ddp_rank() == 0):
        torch.cuda.reset_peak_memory_stats(device)

    n_batches_per_epoch = len(train_loader)
    if not _is_ddp() or _ddp_rank() == 0:
        print(
            f"[Train] 每 epoch: {n_batches_per_epoch} steps（batch_size/卡={args.batch_size}，"
            f"数据集样本数={len(train_dataset)}）；增大 batch 时该 steps 应下降。",
            flush=True,
        )
        print(
            "[Train] 提示：tqdm 在每个 epoch 会停在 0% 直到「第一个 batch」读完——"
            "每个 batch 要从盘里读 batch_size×num_frames 张 PNG；NFS/共享盘上常要几十秒，后面 batch 往往更快。",
            flush=True,
        )
        if num_workers > 0:
            print(
                f"[Train] DataLoader prefetch_factor={loader_kw.get('prefetch_factor', '—')}（可用 --prefetch_factor 调整）。",
                flush=True,
            )

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t_epoch = time.perf_counter()
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sync_loss_across_ranks=_is_ddp(),
            use_amp_bf16=use_amp_bf16,
            epoch_index=epoch,
            num_epochs=args.epochs,
            show_progress=(not args.no_progress) and (not _is_ddp() or _ddp_rank() == 0),
            non_blocking_copy=pin_mem and device.type == "cuda",
        )
        elapsed = time.perf_counter() - t_epoch
        scheduler.step()
        if not _is_ddp() or _ddp_rank() == 0:
            train_losses.append(loss)
            epoch_seconds.append(elapsed)
            print(
                f"Epoch {epoch + 1}/{args.epochs}  steps={n_batches_per_epoch}  "
                f"loss={loss:.4f}  time={elapsed:.2f}s ({elapsed / 60.0:.2f}min)",
                flush=True,
            )
            if epoch == 0 and device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(f"[Train] 第 1 个 epoch 后 CUDA 峰值显存≈{peak:.2f} GiB", flush=True)
            if writer is not None:
                writer.add_scalar("train/loss", loss, epoch + 1)
                writer.add_scalar("train/epoch_seconds", elapsed, epoch + 1)
                if epoch == 0:
                    writer.add_scalar("train/amp_bf16", 1.0 if use_amp_bf16 else 0.0, 1)

            # 根据 ckpt_interval 控制中间 checkpoint 保存频率
            if args.ckpt_interval and args.ckpt_interval > 0:
                if (epoch + 1) % args.ckpt_interval == 0:
                    ckpt_path = save_dir / f"epoch_{epoch + 1}__{ckpt_slug}.pt"
                    state_dict = (
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "arch": args.arch,
                            "data_layout": args.data_layout,
                            "amp_bf16": use_amp_bf16,
                            "train_hparams": train_hparams,
                            "model_state_dict": state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"  保存 checkpoint: {ckpt_path}")

    if not _is_ddp() or _ddp_rank() == 0:
        if writer is not None:
            writer.close()

        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        final_payload = {
            "epoch": args.epochs,
            "arch": args.arch,
            "data_layout": args.data_layout,
            "amp_bf16": use_amp_bf16,
            "train_hparams": train_hparams,
            "model_state_dict": state_dict,
        }
        final_tagged = save_dir / f"final__{ckpt_slug}.pt"
        torch.save(final_payload, final_tagged)
        shutil.copy2(final_tagged, save_dir / "final.pt")
        print(
            f"训练完成，模型已保存至 {final_tagged}，并已复制为 {save_dir / 'final.pt'}（默认识别）"
        )

        # 保存 loss 曲线图（按架构序号子目录）
        vis_dir = root / "my_model" / "visualizations" / str(args.arch)
        vis_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.title(f"Training loss curve ({ckpt_slug})")
        plt.grid(True, alpha=0.3)
        out_path = vis_dir / f"train_loss__{ckpt_slug}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"训练 loss 曲线已保存至 {out_path}")

        # 每 epoch 耗时曲线
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(epoch_seconds) + 1), epoch_seconds, marker="o", color="tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.title(f"Training time per epoch ({ckpt_slug})")
        plt.grid(True, alpha=0.3)
        time_plot = vis_dir / f"train_epoch_seconds__{ckpt_slug}.png"
        plt.tight_layout()
        plt.savefig(time_plot)
        plt.close()
        print(f"每 epoch 耗时曲线已保存至 {time_plot}")

        # 机器可读日志（含 loss 与秒）
        log_records = [
            {"epoch": i + 1, "loss": float(train_losses[i]), "epoch_seconds": float(epoch_seconds[i])}
            for i in range(len(train_losses))
        ]
        epoch_log_path = save_dir / f"epoch_time_log__{ckpt_slug}.json"
        with open(epoch_log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ckpt_config_slug": ckpt_slug,
                    "train_hparams": train_hparams,
                    "total_epochs": len(log_records),
                    "total_train_seconds": float(sum(epoch_seconds)),
                    "epochs": log_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"每 epoch 耗时明细已保存至 {epoch_log_path}")

    if _is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
