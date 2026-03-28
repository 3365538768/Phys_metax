from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from logic_model.dataset import inspect_lmdb_format, list_lmdb_samples, read_lmdb_arrays, save_rgb_previews


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inspect LMDB format and optionally dump RGB previews")
    p.add_argument(
        "--split_root",
        type=str,
        default="auto_output/dataset_deformation_stress_500_new/train",
        help="样本目录根路径",
    )
    p.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="待检查样本索引（按目录名排序）",
    )
    p.add_argument(
        "--lmdb_env_subdir",
        type=str,
        default="arch4_data.lmdb",
        help="样本内 LMDB 子目录名",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="0 表示按 LMDB 实际总帧数完整读取；>0 表示重采样到指定帧数",
    )
    p.add_argument("--img_size", type=int, default=0, help="仅在 --num_frames > 0 时需要")
    p.add_argument("--max_views", type=int, default=4)
    p.add_argument(
        "--preview_dir",
        type=str,
        default="logic_model/lmdb_preview",
        help="RGB 预览图输出目录",
    )
    p.add_argument(
        "--save_json",
        type=str,
        default="logic_model/lmdb_inspect_result.json",
        help="检查结果 json 保存路径",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    samples = list_lmdb_samples(args.split_root, args.lmdb_env_subdir)
    if not samples:
        raise RuntimeError(f"未找到 LMDB 样本: split_root={args.split_root}")

    idx = max(0, min(int(args.sample_index), len(samples) - 1))
    sample_dir = samples[idx]
    print(f"[logic_model] lmdb samples = {len(samples)}")
    print(f"[logic_model] inspect sample[{idx}] = {sample_dir}")

    report: Dict[str, Any] = inspect_lmdb_format(str(sample_dir), args.lmdb_env_subdir)

    has_true_rgb = any(
        bool(v.get("is_true_rgb", False)) for v in report.get("storage_format", {}).values() if v.get("exists", False)
    )
    preview_files = []
    if has_true_rgb:
        arrays = read_lmdb_arrays(
            str(sample_dir),
            num_frames=(None if int(args.num_frames) <= 0 else int(args.num_frames)),
            img_size=(None if int(args.img_size) <= 0 else int(args.img_size)),
            max_views=args.max_views,
            lmdb_env_subdir=args.lmdb_env_subdir,
        )
        report["loaded_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
        preview_files = save_rgb_previews(arrays, args.preview_dir, max_views=2, max_frames=2)
    report["rgb_preview_saved"] = bool(preview_files)
    report["rgb_preview_files"] = preview_files

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out_json = Path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[logic_model] saved inspect report -> {out_json}")
    if preview_files:
        print(f"[logic_model] saved RGB preview images -> {args.preview_dir}")
    else:
        print("[logic_model] no true RGB modality detected, skip image dump")


if __name__ == "__main__":
    main()

