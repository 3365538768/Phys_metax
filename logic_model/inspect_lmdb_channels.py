from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from logic_model.dataset import list_lmdb_samples, read_lmdb_arrays


def _channel_stats(vcthw: np.ndarray) -> Dict[str, Any]:
    """
    vcthw: [V,3,T,H,W], float32 in [0,1]
    """
    if vcthw.ndim != 5 or int(vcthw.shape[1]) != 3:
        raise ValueError(f"expect [V,3,T,H,W], got {vcthw.shape}")

    flat = vcthw.reshape(vcthw.shape[0], 3, -1)  # [V,3,N]
    merged = flat.transpose(1, 0, 2).reshape(3, -1)  # [3, V*N]

    ch = []
    for ci in range(3):
        x = merged[ci]
        ch.append(
            {
                "channel": ci,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "nonzero_ratio": float(np.mean(x > 1e-8)),
            }
        )

    # 通道间差异（越接近 0 说明越像复制通道）
    d01 = np.abs(merged[0] - merged[1])
    d02 = np.abs(merged[0] - merged[2])
    d12 = np.abs(merged[1] - merged[2])
    pair_diff = {
        "mean_abs_diff_01": float(np.mean(d01)),
        "mean_abs_diff_02": float(np.mean(d02)),
        "mean_abs_diff_12": float(np.mean(d12)),
        "max_abs_diff_01": float(np.max(d01)),
        "max_abs_diff_02": float(np.max(d02)),
        "max_abs_diff_12": float(np.max(d12)),
    }
    likely_repeated = (
        pair_diff["mean_abs_diff_01"] < 1e-5
        and pair_diff["mean_abs_diff_02"] < 1e-5
        and pair_diff["mean_abs_diff_12"] < 1e-5
    )

    # 每个 view 单独看一眼（只输出前 2 个 view）
    per_view: List[Dict[str, Any]] = []
    v_lim = min(2, int(vcthw.shape[0]))
    for vi in range(v_lim):
        vv = vcthw[vi].reshape(3, -1)
        per_view.append(
            {
                "view_index": vi,
                "channel_means": [float(np.mean(vv[0])), float(np.mean(vv[1])), float(np.mean(vv[2]))],
                "channel_stds": [float(np.std(vv[0])), float(np.std(vv[1])), float(np.std(vv[2]))],
                "mean_abs_diff_01": float(np.mean(np.abs(vv[0] - vv[1]))),
                "mean_abs_diff_02": float(np.mean(np.abs(vv[0] - vv[2]))),
                "mean_abs_diff_12": float(np.mean(np.abs(vv[1] - vv[2]))),
            }
        )

    return {
        "shape": list(vcthw.shape),
        "channels": ch,
        "pairwise_diff": pair_diff,
        "likely_repeated_3ch": bool(likely_repeated),
        "per_view_head": per_view,
    }


def _save_channel_images(arrays: Dict[str, np.ndarray], out_dir: Path) -> List[str]:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("保存通道图片需要 opencv-python（cv2）") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for mod, vcthw in arrays.items():
        if vcthw.ndim != 5 or int(vcthw.shape[1]) != 3:
            continue
        vi = 0
        ti = 0
        for ci in range(3):
            fr = vcthw[vi, ci, ti]  # [H,W]
            fmin = float(np.min(fr))
            fmax = float(np.max(fr))
            if (fmax - fmin) < 1e-8:
                u8 = np.zeros_like(fr, dtype=np.uint8)
            else:
                u8 = np.clip((fr - fmin) / (fmax - fmin + 1e-8) * 255.0, 0, 255).astype(np.uint8)
            p = out_dir / f"{mod}_view{vi:02d}_frame{ti:03d}_ch{ci}.png"
            cv2.imwrite(str(p), u8)
            paths.append(str(p))
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Inspect LMDB 3-channel details for one sample")
    ap.add_argument("--split_root", type=str, default="auto_output/dataset_deformation_stress_500_new/train")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--max_views", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=0, help="0=按LMDB实际总帧数")
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--save_json", type=str, default="logic_model/lmdb_channel_inspect.json")
    ap.add_argument("--preview_dir", type=str, default="logic_model/lmdb_channel_preview")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples = list_lmdb_samples(args.split_root)
    if not samples:
        raise RuntimeError(f"未找到LMDB样本: {args.split_root}")
    si = max(0, min(int(args.sample_index), len(samples) - 1))
    sample_dir = samples[si]

    arrays = read_lmdb_arrays(
        str(sample_dir),
        num_frames=(None if int(args.num_frames) <= 0 else int(args.num_frames)),
        img_size=(None if int(args.img_size) <= 0 else int(args.img_size)),
        max_views=int(args.max_views),
    )

    report: Dict[str, Any] = {
        "sample_dir": str(sample_dir.resolve()),
        "sample_id": sample_dir.name,
        "modalities": {},
    }
    for mod, vcthw in arrays.items():
        report["modalities"][mod] = _channel_stats(vcthw)

    preview_paths = _save_channel_images(arrays, Path(args.preview_dir))
    report["preview_dir"] = str(Path(args.preview_dir).resolve())
    report["preview_files"] = preview_paths

    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved_json": str(out_path), "sample_id": sample_dir.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()

