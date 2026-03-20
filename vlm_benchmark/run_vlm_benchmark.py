import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    # 作为包运行: python -m vlm_benchmark.run_vlm_benchmark
    from vlm_benchmark.vlm_model_registry import create_vlm_client, VLM_REGISTRY  # type: ignore
except ModuleNotFoundError:
    # 作为脚本运行: python vlm_benchmark/run_vlm_benchmark.py
    from vlm_model_registry import create_vlm_client, VLM_REGISTRY  # type: ignore


ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")
MATERIAL_CATEGORIES = ("jelly", "metal", "plasticine")


def _parse_params_to_dict(params_str: str) -> Dict[str, str]:
    """
    将 E=1.62e+04_density=1.28e+03_... 解析为 {key: value}，兼容 key 中含下划线（如 yield_stress）。
    """
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
                    full_key = f"{t}_{k2}"
                    result[full_key] = v2
                    i += 2
            else:
                i += 1
    return result


def _iter_auto_output_runs(auto_output_root: Path) -> Iterable[Tuple[str, Path, Dict[str, str]]]:
    """
    遍历 auto_output/<action>/<OBJECT__PARAMS__action>/ 结构。
    返回 (action, run_dir, params_dict)。
    """
    for action in ACTION_NAMES:
        action_dir = auto_output_root / action
        if not action_dir.is_dir():
            continue
        for run_dir in action_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if "__" not in name:
                continue
            segs = name.split("__")
            if len(segs) < 3:
                continue
            params_str = "__".join(segs[1:-1])
            params = _parse_params_to_dict(params_str)
            yield action, run_dir, params


def _collect_all_runs(auto_output_root: Path) -> List[Tuple[str, Path, Dict[str, str]]]:
    runs: List[Tuple[str, Path, Dict[str, str]]] = []
    for item in _iter_auto_output_runs(auto_output_root):
        runs.append(item)
    return runs


def _find_video_in_run_dir(run_dir: Path) -> Optional[Path]:
    """
    在单个仿真目录下查找视频文件。
    """
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    for root, _, files in os.walk(run_dir):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in exts):
                return Path(root) / fname
    return None


def _safe_float(params: Dict[str, str], key: str) -> float:
    v = params.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _normalize_material_label(params: Dict[str, str]) -> str:
    raw = params.get("material", "").lower()
    if raw in MATERIAL_CATEGORIES:
        return raw
    if "jelly" in raw:
        return "jelly"
    if "metal" in raw:
        return "metal"
    if "plastic" in raw:
        return "plasticine"
    return "plasticine"


def _load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_benchmark(
    physgaussian_root: Path,
    output_root: Path,
    max_videos: Optional[int] = None,
    random_sample: bool = False,
    debug_print: bool = True,
    vlm_tag: str = "qwen2.5-vl",
) -> None:
    script_dir = physgaussian_root / "vlm_benchmark"
    auto_output_root = physgaussian_root / "auto_output"

    if not auto_output_root.is_dir():
        raise FileNotFoundError(f"未找到 auto_output 目录: {auto_output_root}")

    system_prompt = _load_prompt(script_dir / "prompts" / "system_prompt.txt")
    user_prompt = _load_prompt(script_dir / "prompts" / "user_prompt_video_regression.txt")

    # 依据标签创建统一封装的 VLM client
    client = create_vlm_client(vlm_tag)

    # 当前模型的输出目录：vlm_benchmark/output/<vlm_tag>/
    output_root.mkdir(parents=True, exist_ok=True)
    gt_csv = output_root / "vlm_benchmark_gt.csv"
    pred_csv = output_root / "vlm_benchmark_pred.csv"
    reasoning_dir = output_root / "reasoning"
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    # random_sample 模式下：将抽到的视频打包保存，便于后续对比/复现
    sampled_videos_dir = output_root / "sampled_videos" if random_sample else None
    sampled_manifest_path = output_root / "sampled_videos_manifest.csv" if random_sample else None
    sampled_manifest_f = None
    if sampled_videos_dir is not None and sampled_manifest_path is not None:
        sampled_videos_dir.mkdir(parents=True, exist_ok=True)
        sampled_manifest_f = open(sampled_manifest_path, "w", encoding="utf-8")
        sampled_manifest_f.write(
            "sample_id,action,src_rel_video_path,dst_rel_video_path\n"
        )

    gt_f = open(gt_csv, "w", encoding="utf-8")
    pred_f = open(pred_csv, "w", encoding="utf-8")

    # sample_id 作为第一列，方便对齐和跨模型比较
    gt_f.write("sample_id,action,material_gt,E_gt,nu_gt,density_gt,yield_stress_gt\n")
    pred_f.write(
        "sample_id,action,material_gt,E_gt,nu_gt,density_gt,yield_stress_gt,"
        "E_pred,nu_pred,density_pred,yield_stress_pred,material_pred,motion_pred\n"
    )

    # 收集所有候选样本
    all_runs = _collect_all_runs(auto_output_root)
    if not all_runs:
        print("未在 auto_output 下找到任何仿真目录。")
        gt_f.close()
        pred_f.close()
        return

    # 随机采样模式：打乱顺序
    if random_sample:
        rng = random.Random(42)
        rng.shuffle(all_runs)

    num_processed = 0
    try:
        for action, run_dir, params in all_runs:
            if max_videos is not None and num_processed >= max_videos:
                break

            video_path = _find_video_in_run_dir(run_dir)
            if video_path is None:
                continue

            rel_video_path = video_path.relative_to(physgaussian_root).as_posix()

            material_gt = _normalize_material_label(params)
            E_gt = _safe_float(params, "E")
            nu_gt = _safe_float(params, "nu")
            density_gt = _safe_float(params, "density")
            yield_stress_gt = _safe_float(params, "yield_stress")

            # 样本 ID：物体名称 + 全局编号
            object_name = run_dir.name.split("__")[0]
            sample_id = f"{object_name}_{num_processed + 1:04d}"

            # random_sample 模式：复制本次采样到的视频到 output/<vlm_tag>/sampled_videos/
            if sampled_videos_dir is not None and sampled_manifest_f is not None:
                ext = video_path.suffix.lower() if video_path.suffix else ".mp4"
                dst_name = f"{sample_id}{ext}"
                dst_path = sampled_videos_dir / dst_name
                try:
                    shutil.copy2(video_path, dst_path)
                except Exception as exc:
                    print(f"[WARN] 复制采样视频失败: {video_path} -> {dst_path} ({exc})")
                dst_rel = dst_path.relative_to(output_root).as_posix()
                sampled_manifest_f.write(
                    f"{sample_id},{action},{rel_video_path},{dst_rel}\n"
                )

            gt_f.write(
                f"{sample_id},{action},{material_gt},"
                f"{E_gt},{nu_gt},{density_gt},{yield_stress_gt}\n"
            )

            # 调用 VLM（网络/上传失败时跳过该样本，避免中断长跑）
            try:
                prediction = client.predict_video(
                    str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    video_id=sample_id,
                )
            except Exception as exc:
                print(f"[WARN] VLM 调用失败，跳过样本 {sample_id}: {exc}")
                # 仍然写一条空的 reasoning 文件，方便对齐
                with open(reasoning_dir / f"{sample_id}.txt", "w", encoding="utf-8") as rf:
                    rf.write(f"[ERROR] {exc}\n")
                continue

            # 将 raw model text 保存到 reasoning/<sample_id>.txt
            raw_text = ""
            try:
                raw_text = str(prediction.get("__raw_text__", "") or "")
            except Exception:
                raw_text = ""
            reasoning_path = reasoning_dir / f"{sample_id}.txt"
            with open(reasoning_path, "w", encoding="utf-8") as rf:
                rf.write(raw_text)

            E_pred = float(prediction.get("E", 0.0))
            nu_pred = float(prediction.get("nu", 0.0))
            density_pred = float(prediction.get("density", 0.0))
            yield_stress_pred = float(prediction.get("yield_stress", 0.0))
            material_pred = str(prediction.get("material_type", "")).lower()
            motion_pred = str(prediction.get("motion_type", "")).lower()

            if debug_print:
                print("\n================ VLM 调用结果 ================")
                print(f"样本编号: {num_processed + 1}")
                print(f"vlm_tag: {vlm_tag}")
                print(f"sample_id: {sample_id}")
                print(f"视频路径: {rel_video_path}")
                print(f"动作类型 (GT): {action}")
                print(f"材质类型 (GT): {material_gt}")
                print(
                    f"GT 数值: E={E_gt:.3g}, nu={nu_gt:.3g}, "
                    f"density={density_gt:.3g}, yield_stress={yield_stress_gt:.3g}"
                )
                print("原始 JSON 输出:")
                try:
                    print(json.dumps(prediction, ensure_ascii=False, indent=2))
                except TypeError:
                    print(prediction)
                raw_text = prediction.get("__raw_text__")
                if raw_text:
                    print("\n[RAW MODEL TEXT]")
                    print(str(raw_text))
                print(
                    "解析后数值: "
                    f"E_pred={E_pred:.3g}, nu_pred={nu_pred:.3g}, "
                    f"density_pred={density_pred:.3g}, "
                    f"yield_stress_pred={yield_stress_pred:.3g}, "
                    f"material_pred={material_pred}, motion_pred={motion_pred}"
                )
                print("=============================================\n")

            pred_f.write(
                f"{sample_id},{action},{material_gt},"
                f"{E_gt},{nu_gt},{density_gt},{yield_stress_gt},"
                f"{E_pred},{nu_pred},{density_pred},{yield_stress_pred},"
                f"{material_pred},{motion_pred}\n"
            )

            num_processed += 1
            print(f"[OK] 已处理视频 {num_processed}: {rel_video_path}")
    finally:
        gt_f.close()
        pred_f.close()
        if sampled_manifest_f is not None:
            sampled_manifest_f.close()

    print(f"完成。共写入 {num_processed} 条样本。")
    print(f"GT 文件:   {gt_csv}")
    print(f"预测文件: {pred_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用 VLM benchmark：对 PhysGaussian/auto_output 中的视频进行评测。"
    )
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="PhysGaussian 根目录（包含 auto_output 和 vlm_benchmark）",
    )
    parser.add_argument(
        "--vlm_tag",
        type=str,
        default="qwen2.5-vl",
        choices=list(VLM_REGISTRY.keys()),
        help="选择要使用的 VLM 标签（在 vlm_model_registry.py 中维护具体配置）",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="最多评测多少个视频（默认全部）",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="是否以随机顺序从 bend/drop/press/shear/stretch 采样",
    )

    args = parser.parse_args()
    phys_root = Path(args.physgaussian_root).resolve()

    # 模型专属输出目录：vlm_benchmark/output/<vlm_tag>/
    output_root = phys_root / "vlm_benchmark" / "output" / args.vlm_tag

    run_benchmark(
        physgaussian_root=phys_root,
        output_root=output_root,
        max_videos=args.max_videos,
        random_sample=args.random_sample,
        debug_print=True,
        vlm_tag=args.vlm_tag,
    )


if __name__ == "__main__":
    main()