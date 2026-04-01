# -*- coding: utf-8 -*-
"""
按 ``gt.json`` 中的 ``object`` 字段划分 train / test：

- **test**：``object`` 属于用户指定的若干类型（字符串与 ``gt.json`` 完全一致）。
- **train**：其余所有样本。

默认只统计同时含 ``arch4_data.lmdb`` 与 ``gt.json`` 的目录（与 ``LmdbGtDataset`` 一致）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load_object(gt_path: Path) -> str:
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return ""
    return str(data.get("object", "")).strip()


def _iter_sample_dirs(
    split_root: Path,
    *,
    require_lmdb: bool,
    lmdb_env_subdir: str,
) -> List[Path]:
    split_root = split_root.resolve()
    if not split_root.is_dir():
        raise ValueError(f"split_root 不是目录: {split_root}")

    if require_lmdb:
        from .dataset import list_lmdb_samples

        out: List[Path] = []
        for d in list_lmdb_samples(str(split_root), lmdb_env_subdir):
            if (d / "gt.json").is_file():
                out.append(Path(d).resolve())
        return sorted(out, key=lambda p: p.name)

    out = []
    for d in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        if (d / "gt.json").is_file():
            out.append(d.resolve())
    return out


def split_by_test_objects(
    split_root: Path,
    test_objects: Set[str],
    *,
    require_lmdb: bool = True,
    lmdb_env_subdir: str = "arch4_data.lmdb",
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns:
        train_ids, test_ids, sample_id -> object string
    """
    if not test_objects:
        raise ValueError("test_objects 不能为空")

    dirs = _iter_sample_dirs(
        split_root, require_lmdb=require_lmdb, lmdb_env_subdir=lmdb_env_subdir
    )
    if not dirs:
        raise ValueError(f"未找到可用样本目录: {split_root}")

    id_to_obj: Dict[str, str] = {}
    for d in dirs:
        obj = _load_object(d / "gt.json")
        id_to_obj[d.name] = obj

    train_ids: List[str] = []
    test_ids: List[str] = []
    for sid, obj in sorted(id_to_obj.items(), key=lambda x: x[0]):
        if obj in test_objects:
            test_ids.append(sid)
        else:
            train_ids.append(sid)

    return train_ids, test_ids, id_to_obj


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按 gt.json 的 object 划分 train/test（test 仅含指定 object 类型）"
    )
    ap.add_argument(
        "--split_root",
        type=str,
        required=True,
        help="数据集 split 根目录，如 auto_output/dataset_mask_1000/train",
    )
    ap.add_argument(
        "--test_objects",
        type=str,
        nargs="+",
        required=True,
        help="用于 test 的 object 字符串（与 gt.json 中 object 字段完全一致），可多个",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="logic_model/train_test_split_by_object.json",
        help="输出 JSON 路径（相对当前工作目录或绝对路径）",
    )
    ap.add_argument(
        "--no_require_lmdb",
        action="store_true",
        help="不要求 arch4_data.lmdb，仅要求子目录下存在 gt.json",
    )
    ap.add_argument(
        "--lmdb_env_subdir",
        type=str,
        default="arch4_data.lmdb",
        help="与 LmdbGtDataset 一致的环境子目录名",
    )
    args = ap.parse_args()

    split_root = Path(args.split_root).expanduser().resolve()
    test_set = {str(x).strip() for x in args.test_objects if str(x).strip()}
    if not test_set:
        raise SystemExit("解析后 test_objects 为空，请检查参数")

    train_ids, test_ids, id_to_obj = split_by_test_objects(
        split_root,
        test_set,
        require_lmdb=not bool(args.no_require_lmdb),
        lmdb_env_subdir=str(args.lmdb_env_subdir).strip() or "arch4_data.lmdb",
    )

    if not test_ids:
        raise SystemExit(
            f"test 集合为空：没有任何样本的 object 属于 {sorted(test_set)}。"
            "请检查拼写是否与 gt.json 一致，或尝试 --no_require_lmdb。"
        )

    obj_counter = Counter(id_to_obj.values())
    test_counter = Counter(id_to_obj[s] for s in test_ids)
    train_counter = Counter(id_to_obj[s] for s in train_ids)

    out = {
        "split_root": str(split_root),
        "mode": "by_object",
        "test_objects": sorted(test_set),
        "require_lmdb": not bool(args.no_require_lmdb),
        "lmdb_env_subdir": str(args.lmdb_env_subdir).strip() or "arch4_data.lmdb",
        "train_ids": train_ids,
        "test_ids": test_ids,
        "stats": {
            "n_samples": len(id_to_obj),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "object_counts_all": dict(sorted(obj_counter.items(), key=lambda x: (-x[1], x[0]))),
            "object_counts_train": dict(sorted(train_counter.items(), key=lambda x: (-x[1], x[0]))),
            "object_counts_test": dict(sorted(test_counter.items(), key=lambda x: (-x[1], x[0]))),
        },
    }

    out_path = Path(args.out_json).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] split_root={split_root}\n"
        f"     test_objects={sorted(test_set)}\n"
        f"     train={len(train_ids)} test={len(test_ids)} (total samples={len(id_to_obj)})\n"
        f"     -> {out_path}"
    )


if __name__ == "__main__":
    main()
