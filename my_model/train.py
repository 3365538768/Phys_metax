"""
训练脚本：7:3 分割，训练物理参数预测模型
"""
import argparse
import os
import subprocess
from pathlib import Path
from typing import Deque, List, Tuple
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
    from my_model.dataset import PhysGaussianDataset, train_test_split
    from my_model.model import create_model
except ImportError:
    from .dataset import PhysGaussianDataset, train_test_split
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
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for frames, params_gt, material_gt, action_gt in loader:
        frames = frames.to(device)
        params_gt = params_gt.to(device)
        action_gt = action_gt.to(device)

        optimizer.zero_grad()
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

    return total_loss / max(n_samples, 1)


def _is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


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
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--save_dir", type=str, default="my_model/checkpoints")
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=0,
        help="每隔多少个 epoch 保存一次 checkpoint；0 表示不保存中间 checkpoint，只在最后一代保存",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="my_model/runs",
        help="TensorBoard 日志目录（默认 my_model/runs）",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="多卡训练使用的 GPU 数量 N（自动选择最空闲的 N 张卡并启用 DDP）",
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
    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / args.log_dir
    if (not _is_ddp() or _ddp_rank() == 0):
        log_dir.mkdir(parents=True, exist_ok=True)

    # 7:3 分割
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    model = create_model(num_frames=args.num_frames).to(device)
    if _is_ddp():
        model = DDP(model, device_ids=[_ddp_local_rank()], output_device=_ddp_local_rank())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 仅在 rank0 记录 loss 曲线和 TensorBoard
    train_losses = []
    writer = None
    if not _is_ddp() or _ddp_rank() == 0:
        writer = SummaryWriter(log_dir=str(log_dir))

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        if not _is_ddp() or _ddp_rank() == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}")
            train_losses.append(loss)
            if writer is not None:
                writer.add_scalar("train/loss", loss, epoch + 1)

            # 根据 ckpt_interval 控制中间 checkpoint 保存频率
            if args.ckpt_interval and args.ckpt_interval > 0:
                if (epoch + 1) % args.ckpt_interval == 0:
                    ckpt_path = save_dir / f"epoch_{epoch + 1}.pt"
                    state_dict = (
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
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
        torch.save(state_dict, save_dir / "final.pt")
        print(f"训练完成，模型已保存至 {save_dir / 'final.pt'}")

        # 保存 loss 曲线图
        vis_dir = root / "my_model" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.title("Training loss curve")
        plt.grid(True, alpha=0.3)
        out_path = vis_dir / "train_loss.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"训练 loss 曲线已保存至 {out_path}")

    if _is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
