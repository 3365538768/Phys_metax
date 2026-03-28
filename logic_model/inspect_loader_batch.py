from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch, save_sample_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Load one batch from LMDB+gt.json and inspect params/images")
    p.add_argument(
        "--split_root",
        type=str,
        default="auto_output/dataset_deformation_stress_500_new/train",
        help="样本目录根路径",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_views", type=int, default=4)
    p.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="0 表示读取 LMDB 实际总帧数；>0 表示重采样到该帧数",
    )
    p.add_argument("--img_size", type=int, default=0, help="仅在 --num_frames > 0 时需要")
    p.add_argument("--lmdb_env_subdir", type=str, default="arch4_data.lmdb")
    p.add_argument("--preview_dir", type=str, default="logic_model/loader_preview")
    p.add_argument("--save_json", type=str, default="logic_model/loader_batch_inspect.json")
    return p.parse_args()


def _shape_dict(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("rgb", "stress", "flow", "force_mask", "object_mask", "params"):
        v = batch[k]
        if isinstance(v, torch.Tensor):
            out[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "min": float(v.min().item()),
                "max": float(v.max().item()),
                "mean": float(v.mean().item()),
            }
    return out


def main() -> None:
    args = parse_args()
    ds = LmdbGtDataset(
        split_root=args.split_root,
        lmdb_env_subdir=args.lmdb_env_subdir,
        max_views=args.max_views,
        num_frames=(None if int(args.num_frames) <= 0 else int(args.num_frames)),
        img_size=(None if int(args.img_size) <= 0 else int(args.img_size)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_lmdb_gt_batch,
    )

    batch = next(iter(loader))
    summary: Dict[str, Any] = {
        "dataset_size": len(ds),
        "batch_size": int(args.batch_size),
        "sample_id": batch["sample_id"],
        "action_label": batch.get("action_label", torch.empty(0, dtype=torch.long)).detach().cpu().numpy().tolist()
        if "action_label" in batch
        else [],
        "action_name": batch.get("action_name", []),
        "action_to_id": ds.action_to_id,
        "num_frames_per_sample": batch["num_frames_per_sample"],
        "tensor_stats": _shape_dict(batch),
        "params_dict": batch["params_dict"],
        "params_tensor": batch["params"].detach().cpu().numpy().tolist(),
    }

    # 仅导出 batch 第一个样本用于检查
    sample_np: Dict[str, np.ndarray] = {}
    for k in ("rgb", "stress", "flow", "force_mask", "object_mask"):
        # [B,V,3,T,H,W] -> [V,3,T,H,W]
        sample_np[k] = batch[k][0].detach().cpu().numpy()
    preview_files = save_sample_images(sample_np, args.preview_dir, max_views=2, max_frames=2)
    summary["preview_dir"] = str(Path(args.preview_dir).resolve())
    summary["preview_files"] = preview_files

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[logic_model] saved batch inspect -> {out}")


if __name__ == "__main__":
    main()

