"""
评估脚本（合并 test + visualize）：
1) 在测试集上评估回归/分类指标
2) 自动生成可视化（GT vs Pred 散点图、相对误差直方图）
3) 支持自动选择空闲 GPU（按 utilization / memory 占用比例）
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from my_model.dataset import (
        Dataset400,
        PhysGaussianDataset,
        resolve_flat_dataset_root,
        train_test_split,
    )
    from my_model.model import create_model
except ImportError:
    from .dataset import (
        Dataset400,
        PhysGaussianDataset,
        resolve_flat_dataset_root,
        train_test_split,
    )
    from .model import create_model


PARAM_NAMES = ["E", "nu", "density", "yield_stress"]


def _pick_free_gpus(n: int) -> List[int]:
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
    rows.sort(key=lambda x: (x[1], x[2]))
    return [r[0] for r in rows[:n]]


def _auto_select_gpu_if_needed(args: argparse.Namespace) -> torch.device:
    """
    若 device 为 cuda 且用户开启 auto_gpu，则自动选择最空闲的 1 张卡，并设置 CUDA_VISIBLE_DEVICES。
    返回最终使用的 torch.device（cuda:0 / cpu）。
    """
    device_str = str(args.device).lower()
    if device_str.startswith("cuda") and args.auto_gpu and torch.cuda.is_available():
        picked = _pick_free_gpus(1)
        if picked:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(picked[0])
            torch.cuda.set_device(0)
            return torch.device("cuda", 0)
    return torch.device(args.device)


def collect_predictions(model, loader, device):
    model.eval()
    all_pred, all_gt, all_action_pred, all_action_gt = [], [], [], []
    with torch.no_grad():
        for frames, params_gt, _, action_gt in loader:
            frames = frames.to(device)
            params_pred, action_logits = model(frames)
            all_pred.append(params_pred.cpu().numpy())
            all_gt.append(params_gt.numpy())
            all_action_pred.append(action_logits.argmax(1).cpu().numpy())
            all_action_gt.append(action_gt.numpy())
    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_action_pred, axis=0),
        np.concatenate(all_action_gt, axis=0),
    )


def compute_metrics(pred: np.ndarray, gt: np.ndarray, action_pred: np.ndarray, action_gt: np.ndarray) -> dict:
    abs_err = np.abs(pred - gt)
    mae_per_param = np.mean(abs_err, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(gt != 0, abs_err / np.abs(gt), np.nan)
    mape_per_param = np.nanmean(rel_err, axis=0) * 100
    action_acc = float((action_pred == action_gt).mean() * 100)
    return {"mae": mae_per_param, "mape": mape_per_param, "action_acc": action_acc}


def plot_scatter(gt, pred, param_name, out_path, log_scale=False):
    plt.figure(figsize=(5, 5))
    if log_scale:
        mask = (gt > 0) & (pred > 0)
        gt = gt[mask]
        pred = pred[mask]
    plt.scatter(gt, pred, s=8, alpha=0.6)
    if log_scale and gt.size > 0 and pred.size > 0:
        plt.xscale("log")
        plt.yscale("log")
    if gt.size > 0 and pred.size > 0:
        lim_min = float(min(gt.min(), pred.min()))
        lim_max = float(max(gt.max(), pred.max()))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="y=x")
    plt.xlabel(f"{param_name} GT")
    plt.ylabel(f"{param_name} Pred")
    plt.title(f"{param_name}: GT vs Pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rel_err_hist(gt, pred, param_name, out_path):
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(gt != 0, (pred - gt) / np.abs(gt), np.nan)
    rel_err = rel_err[np.isfinite(rel_err)]
    if rel_err.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(rel_err, bins=40, edgecolor="black", alpha=0.7)
    plt.xlabel(f"Relative error of {param_name}")
    plt.ylabel("Count")
    plt.title(f"Relative error distribution of {param_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physgaussian_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="默认 my_model/checkpoints/<arch>/final.pt",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--data_layout",
        type=str,
        choices=("dataset_400", "legacy"),
        default="dataset_400",
        help="dataset_400：使用扁平集 test/（见 --dataset_dir）；legacy：auto_output 上 7:3 划分得到的 test",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset_400",
        help="仅 data_layout=dataset_400：推荐 dataset_400；或 auto_output/dataset_400（相对仓库根）；或绝对路径。须与训练一致。",
    )
    parser.add_argument(
        "--arch",
        type=int,
        choices=(1, 2),
        default=1,
        help="须与训练一致；若 checkpoint 内含 arch 字段则以其为准",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="可视化根目录，默认 my_model/visualizations/<arch>/，其下再建时间戳子目录",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--auto_gpu", action="store_true", help="自动选择最空闲的 GPU（仅 eval 用 1 张卡）")
    parser.add_argument(
        "--no_dataset_preflight",
        action="store_true",
        help="dataset_400 快速加载：跳过 cv2 首帧与多帧预检（仅检查目录与是否有图片）",
    )

    args = parser.parse_args()
    root = Path(args.physgaussian_root).resolve()
    auto_output = root / "auto_output"

    ckpt_path = root / (args.checkpoint or f"my_model/checkpoints/{args.arch}/final.pt")
    arch = args.arch
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        arch = int(state.get("arch", arch))
        hp = state.get("train_hparams") or {}
        if hp:
            nf = hp.get("num_frames")
            ih = hp.get("img_size")
            if nf is not None and int(nf) != int(args.num_frames):
                print(
                    f"[eval] 警告：checkpoint 记录 num_frames={nf}，当前 --num_frames={args.num_frames}，"
                    "推理应与训练一致，请调整 CLI。"
                )
            if ih is not None and int(ih) != int(args.img_size):
                print(
                    f"[eval] 警告：checkpoint 记录 img_size={ih}，当前 --img_size={args.img_size}。"
                )
            slug = hp.get("ckpt_config_slug")
            if slug:
                print(f"[eval] checkpoint 配置标签: {slug}")
            ds_ckpt = hp.get("dataset_dir")
            if ds_ckpt and str(ds_ckpt) != str(args.dataset_dir):
                print(
                    f"[eval] 警告：checkpoint 记录 dataset_dir={ds_ckpt!r}，当前 --dataset_dir={args.dataset_dir!r}，"
                    "评估集应与训练一致。"
                )

    base_vis = root / (args.out_dir or f"my_model/visualizations/{arch}")
    base_vis.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_vis / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _auto_select_gpu_if_needed(args)

    if args.data_layout == "dataset_400":
        layout_root = resolve_flat_dataset_root(args.dataset_dir, auto_output)
        test_path = layout_root / "test"
        if not test_path.is_dir():
            raise SystemExit(f"未找到测试集目录: {test_path}（检查 --dataset_dir）")
        test_dataset = Dataset400(
            test_path,
            num_frames=args.num_frames,
            img_size=args.img_size,
            preflight=not args.no_dataset_preflight,
            verbose_preflight=True,
        )
    else:
        _, test_samples = train_test_split(auto_output, args.train_ratio, args.seed)
        test_dataset = PhysGaussianDataset(
            auto_output,
            num_frames=args.num_frames,
            img_size=args.img_size,
            sample_ids=test_samples,
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = create_model(num_frames=args.num_frames, arch=arch).to(device)
    if isinstance(state, dict):
        sd = state.get("model_state_dict", state)
    else:
        sd = state
    model.load_state_dict(sd, strict=True)

    pred, gt, action_pred, action_gt = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(pred, gt, action_pred, action_gt)

    print("=== 测试集评估结果 ===")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name}: MAE={metrics['mae'][i]:.4g}, MAPE={metrics['mape'][i]:.2f}%")
    print(f"  动作分类准确率: {metrics['action_acc']:.2f}%")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": arch,
                "data_layout": args.data_layout,
                "dataset_dir": args.dataset_dir if args.data_layout == "dataset_400" else None,
                "checkpoint": str(ckpt_path),
                "n_test": len(test_dataset),
                "mae": {PARAM_NAMES[i]: float(metrics["mae"][i]) for i in range(4)},
                "mape_pct": {PARAM_NAMES[i]: float(metrics["mape"][i]) for i in range(4)},
                "action_acc_pct": metrics["action_acc"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"指标已写入 {metrics_path}")

    print("生成可视化...")
    for i, name in enumerate(PARAM_NAMES):
        plot_scatter(
            gt[:, i],
            pred[:, i],
            name,
            out_dir / f"scatter_{name}.png",
            log_scale=(name in ("E", "density", "yield_stress")),
        )
        plot_rel_err_hist(gt[:, i], pred[:, i], name, out_dir / f"hist_relerr_{name}.png")
    print(f"可视化结果已保存至 {out_dir}（位于 {base_vis} 下）")


if __name__ == "__main__":
    main()

