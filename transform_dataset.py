#!/usr/bin/env python3
"""
将 auto_output 下 bend/drop/press/shear/stretch 整理为扁平数据集。

布局 A — train_test（默认，与 dataset_400 一致）：
  auto_output/<out_name>/train/<编号>/
  auto_output/<out_name>/test/<编号>/

布局 B — by_model（便于后续自行划分 train/test/val）：
  auto_output/<out_name>/<model_dir>/<编号>/
  - model_dir 由 object_slug 安全化得到，与 PLY/物体名一致
  - 编号为在该 model 下的顺序 000000, 000001, …（按 action、路径排序）

auto_simulation_runner 原始输出（新）：
  auto_output/<model>/<NNNN>/（NNNN 为四位数字），物理参数见该目录下 gt_parameters.json；
  by_action 时为 auto_output/<action>/<model>/<NNNN>/。
  命名数据集（train_config.dataset_name 非空）：
  auto_output/<dataset_name>/[<split>/]<NNNNNN>/（六位），含 gt_parameters.json 与 gt.json；
  采集时支持 <dataset>/<split>/ 下全为六位数字子目录的结构。

每个样本目录包含：
  - images/          帧图片（展平到同一目录，按序命名 000000.png …）
  - boundary_conditions.json  （来自 meta/）
  - gt.json          从外层文件夹名解析的动作类型与物理参数 GT
  可选（--copy_aux）：run_parameters.json、stress_heatmaps/、stress_gaussian/、flow_gaussian/、tracks_gaussian/、tracks_subsampled_world/
  可选（--copy_volumetric_fields）：stress_field/、deformation_field/（须仿真时开启 --output_stress / --output_deformation）

划分规则（仅 layout=train_test）：
  - 外层目录名形如 <object_slug>__<params>__<action>
  - object_slug 中含 test_substr 的样本 → test
  - 其余全部混合 → train

layout=by_model 时：使用全部合格样本，不按 test_substr 划分。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")


def parse_params_to_dict(params_str: str) -> Dict[str, str]:
    """解析 E=1.62e+04_density=1.28e+03_... 为 {key: value}（与 my_model.dataset 一致）。"""
    if not params_str.strip():
        return {}
    tokens = params_str.split("_")
    result: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" in t:
            k, _, v = t.partition("=")
            result[k] = v
            i += 1
        else:
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if "=" not in nxt:
                    result[t] = nxt
                    i += 2
                else:
                    k2, _, v2 = nxt.partition("=")
                    result[f"{t}_{k2}"] = v2
                    i += 2
            else:
                i += 1
    return result


def safe_float(params: Dict[str, str], key: str) -> float:
    v = params.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _sorted_image_files(image_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def is_image_sequence_readable(image_dir: Path) -> bool:
    """不依赖 cv2：优先 PIL，否则仅检查首帧文件非空。"""
    files = _sorted_image_files(image_dir)
    if not files:
        return False
    first = files[0]
    try:
        from PIL import Image

        with Image.open(first) as im:
            im.load()
        return True
    except Exception:
        try:
            return first.stat().st_size > 0
        except OSError:
            return False
TEST_OBJECT_SUBSTR = "an_empty_aluminum_can"


def parse_outer_folder(outer_name: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    """解析外层文件夹名 -> (object_slug, action, params_dict)。"""
    parts = outer_name.split("__")
    if len(parts) < 3:
        return None
    object_slug = parts[0]
    action = parts[-1]
    params_str = "__".join(parts[1:-1])
    params = parse_params_to_dict(params_str)
    return object_slug, action, params


def action_to_index(action: str) -> int:
    a = str(action).lower()
    if a in ACTION_NAMES:
        return ACTION_NAMES.index(a)
    return -1


def _find_images_dir_for_inner(outer: Path, inner: Path) -> Optional[Path]:
    """在指定 outer 下，找到属于 inner 的帧图片目录。"""
    images_root = inner / "images"
    if not images_root.is_dir():
        return None
    subdirs = [d for d in images_root.iterdir() if d.is_dir()]
    if subdirs:
        for sd in sorted(subdirs, key=lambda p: p.name):
            if _sorted_image_files(sd):
                return sd
    if _sorted_image_files(images_root):
        return images_root
    return None


def _material_params_to_str_dict(mp: Any) -> Dict[str, str]:
    if not isinstance(mp, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in mp.items():
        out[str(k)] = str(v)
    return out


def _collect_samples_mult_inner(auto_output: Path) -> List[Tuple[Path, str, str, Dict[str, str]]]:
    """
    支持原始仿真目录：
    1) auto_output/<action>/<obj__params__action>/<obj>/  （旧）
    2) auto_output/<model>/<obj__params__action>/<obj>/    （runner by_model + 旧命名）
    3) auto_output/<model>/<NNNN>/  （新：数字目录 + gt_parameters.json，images/meta 直接在 NNNN 下）
    4) auto_output/<action>/<model>/<NNNN>/  （新：by_action + 数字目录 + gt_parameters.json）
    """
    rows: List[Tuple[Path, str, str, Dict[str, str]]] = []
    seen: set[str] = set()

    def try_append(inner: Path, action_from_name: str, object_slug: str, params: Dict[str, str]) -> None:
        key = str(inner.resolve())
        if key in seen:
            return
        meta = inner / "meta" / "boundary_conditions.json"
        if not meta.is_file():
            return
        outer = inner.parent
        image_dir = _find_images_dir_for_inner(outer, inner)
        if image_dir is None or not is_image_sequence_readable(image_dir):
            return
        seen.add(key)
        rows.append((inner, action_from_name, object_slug, params))

    for action in ACTION_NAMES:
        action_dir = auto_output / action
        if not action_dir.is_dir():
            continue
        # 旧：auto_output/<action>/<obj__params__action>/<obj>/
        for outer in sorted(action_dir.iterdir(), key=lambda p: p.name):
            if not outer.is_dir():
                continue
            parsed = parse_outer_folder(outer.name)
            if parsed is None:
                continue
            object_slug, action_from_name, params = parsed
            for inner in sorted(outer.iterdir(), key=lambda p: p.name):
                if not inner.is_dir():
                    continue
                try_append(inner, action_from_name, object_slug, params)
        # 新：auto_output/<action>/<model_slug>/<NNNN>/ + gt_parameters.json
        for model_sub in sorted(action_dir.iterdir(), key=lambda p: p.name):
            if not model_sub.is_dir():
                continue
            if model_sub.name in ACTION_NAMES:
                continue
            for inner in sorted(model_sub.iterdir(), key=lambda p: p.name):
                if not inner.is_dir() or not inner.name.isdigit():
                    continue
                gt_path = inner / "gt_parameters.json"
                if not gt_path.is_file():
                    continue
                try:
                    with open(gt_path, "r", encoding="utf-8") as gf:
                        gt = json.load(gf)
                except (OSError, json.JSONDecodeError):
                    continue
                action_from_name = str(gt.get("sim_type", ""))
                if action_from_name != action:
                    continue
                object_slug = str(gt.get("ply_stem", model_sub.name))
                params = _material_params_to_str_dict(gt.get("material_params"))
                try_append(inner, action_from_name, object_slug, params)

    skip_top = set(ACTION_NAMES) | {"_tmp_configs", "stats", "combined"}
    for model_top in sorted(auto_output.iterdir()):
        if not model_top.is_dir() or model_top.name in skip_top:
            continue
        for outer in sorted(model_top.iterdir(), key=lambda p: p.name):
            if not outer.is_dir():
                continue
            # 新：auto_output/<model>/<NNNN>/（数字目录即样本根）
            if outer.name.isdigit() and (outer / "gt_parameters.json").is_file():
                try:
                    with open(outer / "gt_parameters.json", "r", encoding="utf-8") as gf:
                        gt = json.load(gf)
                except (OSError, json.JSONDecodeError):
                    continue
                action_from_name = str(gt.get("sim_type", ""))
                if action_from_name not in ACTION_NAMES:
                    continue
                object_slug = str(gt.get("ply_stem", model_top.name))
                params = _material_params_to_str_dict(gt.get("material_params"))
                try_append(outer, action_from_name, object_slug, params)
                continue
            # auto_simulation_runner 命名数据集：auto_output/<dataset_name>/<split>/<NNNNNN>/
            subdirs = [d for d in outer.iterdir() if d.is_dir()]
            if subdirs and all(
                d.name.isdigit() and len(d.name) == 6 for d in subdirs
            ):
                for inner in sorted(subdirs, key=lambda p: p.name):
                    gt_path = inner / "gt_parameters.json"
                    if not gt_path.is_file():
                        continue
                    try:
                        with open(gt_path, "r", encoding="utf-8") as gf:
                            gt = json.load(gf)
                    except (OSError, json.JSONDecodeError):
                        continue
                    action_from_name = str(gt.get("sim_type", ""))
                    if action_from_name not in ACTION_NAMES:
                        continue
                    object_slug = str(gt.get("ply_stem", model_top.name))
                    params = _material_params_to_str_dict(gt.get("material_params"))
                    try_append(inner, action_from_name, object_slug, params)
                continue
            parsed = parse_outer_folder(outer.name)
            if parsed is None:
                continue
            object_slug, action_from_name, params = parsed
            if action_from_name not in ACTION_NAMES:
                continue
            for inner in sorted(outer.iterdir(), key=lambda p: p.name):
                if not inner.is_dir():
                    continue
                try_append(inner, action_from_name, object_slug, params)

    return rows


def build_gt_json(object_slug: str, action: str, params: Dict[str, str]) -> Dict[str, Any]:
    E = safe_float(params, "E")
    nu = safe_float(params, "nu")
    density = safe_float(params, "density")
    ys_str = params.get("yield_stress", "")
    yield_stress: Optional[float]
    try:
        yield_stress = float(ys_str) if ys_str not in ("", None) else None
    except (ValueError, TypeError):
        yield_stress = None

    return {
        "object": object_slug,
        "action": action,
        "action_index": action_to_index(action),
        "params": dict(params),
        "regression": {
            "E": E,
            "nu": nu,
            "density": density,
            "yield_stress": yield_stress,
        },
    }


def safe_model_dir_name(object_slug: str) -> str:
    """用作路径一级子目录名（与 object_slug 一一对应，去掉危险字符）。"""
    s = object_slug.replace("/", "_").replace("\\", "_").strip()
    if not s or s in (".", ".."):
        return "unknown_object"
    return s


def copy_optional_auxiliary(inner: Path, dest: Path) -> None:
    """
    若存在则复制：meta/run_parameters.json、stress_heatmaps/、stress_gaussian/、
    flow_gaussian/、tracks_gaussian/、tracks_subsampled_world/（与 modified_simulation 扩展输出一致）
    """
    rp = inner / "meta" / "run_parameters.json"
    if rp.is_file():
        shutil.copy2(rp, dest / "run_parameters.json")
    for name in (
        "stress_heatmaps",
        "stress_gaussian",
        "flow_gaussian",
        "tracks_gaussian",
        "tracks_subsampled_world",
    ):
        src = inner / name
        if src.is_dir():
            dst = dest / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def copy_volumetric_field_dirs(inner: Path, dest: Path) -> None:
    """若存在则复制 stress_field/、deformation_field/（大体积 npz，仿真需开 --output_stress 等）。"""
    for name in ("stress_field", "deformation_field"):
        src = inner / name
        if src.is_dir():
            dst = dest / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def copy_sample(
    inner: Path,
    object_slug: str,
    action: str,
    params: Dict[str, str],
    dest: Path,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    outer = inner.parent
    image_dir = _find_images_dir_for_inner(outer, inner)
    if image_dir is None:
        raise FileNotFoundError(f"无图片目录: {inner}")

    images_out = dest / "images"
    if images_out.exists():
        shutil.rmtree(images_out)
    images_out.mkdir(parents=True)

    files = _sorted_image_files(image_dir)
    for i, src in enumerate(files):
        suf = src.suffix.lower() if src.suffix else ".png"
        dst = images_out / f"{i:06d}{suf}"
        shutil.copy2(src, dst)

    meta_src = inner / "meta" / "boundary_conditions.json"
    shutil.copy2(meta_src, dest / "boundary_conditions.json")

    gt = build_gt_json(object_slug, action, params)
    with open(dest / "gt.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)


def emit_train_test_layout(
    out_root: Path,
    train_samples: List[Tuple[Path, str, str, Dict[str, str]]],
    test_samples: List[Tuple[Path, str, str, Dict[str, str]]],
    test_substr: str,
    copy_aux: bool,
    copy_volumetric_fields: bool = False,
) -> Dict[str, Any]:
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    manifest: Dict[str, Any] = {
        "layout": "train_test",
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "test_rule": f"object_slug contains {test_substr!r}",
        "copy_aux": bool(copy_aux),
        "copy_volumetric_fields": bool(copy_volumetric_fields),
        "train": [],
        "test": [],
    }

    for idx, (inner, action, object_slug, params) in enumerate(train_samples):
        dest = train_dir / f"{idx:06d}"
        copy_sample(inner, object_slug, action, params, dest)
        if copy_aux:
            copy_optional_auxiliary(inner, dest)
        if copy_volumetric_fields:
            copy_volumetric_field_dirs(inner, dest)
        manifest["train"].append(
            {"index": idx, "source": str(inner), "object": object_slug, "action": action}
        )

    for idx, (inner, action, object_slug, params) in enumerate(test_samples):
        dest = test_dir / f"{idx:06d}"
        copy_sample(inner, object_slug, action, params, dest)
        if copy_aux:
            copy_optional_auxiliary(inner, dest)
        if copy_volumetric_fields:
            copy_volumetric_field_dirs(inner, dest)
        manifest["test"].append(
            {"index": idx, "source": str(inner), "object": object_slug, "action": action}
        )
    return manifest


def emit_by_model_layout(
    out_root: Path,
    samples: List[Tuple[Path, str, str, Dict[str, str]]],
    copy_aux: bool,
    copy_volumetric_fields: bool = False,
) -> Dict[str, Any]:
    """
    auto_output/<out_name>/<safe_model_slug>/<idx>/
    同一 object_slug 下 idx 从 0 递增。
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    groups: Dict[str, List[Tuple[Path, str, str, Dict[str, str]]]] = defaultdict(list)
    for row in samples:
        groups[row[2]].append(row)
    for slug in groups:
        groups[slug].sort(key=lambda r: (r[1], str(r[0])))

    manifest_samples: List[Dict[str, Any]] = []
    models_summary: Dict[str, Any] = {}

    for object_slug in sorted(groups.keys()):
        safe = safe_model_dir_name(object_slug)
        rows = groups[object_slug]
        model_dir = out_root / safe
        model_dir.mkdir(parents=True, exist_ok=True)
        for idx, (inner, action, os_slug, params) in enumerate(rows):
            dest = model_dir / f"{idx:06d}"
            copy_sample(inner, os_slug, action, params, dest)
            if copy_aux:
                copy_optional_auxiliary(inner, dest)
            if copy_volumetric_fields:
                copy_volumetric_field_dirs(inner, dest)
            manifest_samples.append(
                {
                    "object": os_slug,
                    "model_dir": safe,
                    "index_in_model": idx,
                    "relative_path": f"{safe}/{idx:06d}",
                    "source": str(inner),
                    "action": action,
                }
            )
        models_summary[safe] = {
            "object_slug": object_slug,
            "sample_count": len(rows),
        }

    return {
        "layout": "by_model",
        "model_count": len(models_summary),
        "sample_count": len(manifest_samples),
        "copy_aux": bool(copy_aux),
        "copy_volumetric_fields": bool(copy_volumetric_fields),
        "models": models_summary,
        "samples": manifest_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="整理 auto_output 为 dataset_400 式扁平样本（可切换按 model 分子目录）"
    )
    parser.add_argument(
        "--auto_output",
        type=Path,
        default=Path(__file__).resolve().parent / "auto_output",
        help="auto_output 根目录",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="dataset_400",
        help="输出子目录名（位于 auto_output 下），例如 dataset_full_test",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=("train_test", "by_model"),
        default="train_test",
        help="train_test: train/ 与 test/ 扁平编号；by_model: 按 object_slug 分子目录后再编号",
    )
    parser.add_argument(
        "--test_substr",
        type=str,
        default=TEST_OBJECT_SUBSTR,
        help="归入 test 的 object_slug 子串（仅 layout=train_test 生效）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="训练集打乱随机种子（仅 layout=train_test；编号仍为 0..n-1）",
    )
    parser.add_argument(
        "--copy_aux",
        action="store_true",
        help="额外复制 run_parameters.json 及 stress_heatmaps/、stress_gaussian/、flow_gaussian/、tracks_gaussian/、tracks_subsampled_world/（若存在）",
    )
    parser.add_argument(
        "--copy_volumetric_fields",
        action="store_true",
        help="额外复制 stress_field/、deformation_field/（若存在；仿真需 --output_stress / --output_deformation）",
    )
    args = parser.parse_args()

    auto_output: Path = args.auto_output.resolve()
    if not auto_output.is_dir():
        raise SystemExit(f"不存在: {auto_output}")

    samples = _collect_samples_mult_inner(auto_output)
    out_root = auto_output / args.out_name

    if args.layout == "by_model":
        manifest = emit_by_model_layout(
            out_root,
            samples,
            copy_aux=bool(args.copy_aux),
            copy_volumetric_fields=bool(args.copy_volumetric_fields),
        )
        manifest["out_name"] = args.out_name
        manifest["source_auto_output"] = str(auto_output)
        with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"完成: {out_root}  (layout=by_model)")
        print(f"  模型数: {manifest['model_count']}  样本总数: {manifest['sample_count']}")
        print(f"  manifest: {out_root / 'manifest.json'}")
        return

    import random

    test_substr = args.test_substr
    train_samples: List[Tuple[Path, str, str, Dict[str, str]]] = []
    test_samples: List[Tuple[Path, str, str, Dict[str, str]]] = []
    for row in samples:
        inner, action, object_slug, params = row
        if test_substr in object_slug:
            test_samples.append(row)
        else:
            train_samples.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(train_samples)
    test_samples = sorted(test_samples, key=lambda r: (r[2], r[1], str(r[0])))

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    manifest = emit_train_test_layout(
        out_root,
        train_samples,
        test_samples,
        test_substr,
        copy_aux=bool(args.copy_aux),
        copy_volumetric_fields=bool(args.copy_volumetric_fields),
    )
    manifest["out_name"] = args.out_name
    manifest["source_auto_output"] = str(auto_output)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    train_dir = out_root / "train"
    test_dir = out_root / "test"
    print(f"完成: {out_root}")
    print(f"  train: {len(train_samples)}  -> {train_dir}")
    print(f"  test:  {len(test_samples)}  -> {test_dir}")
    print(f"  manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
