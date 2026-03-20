#!/usr/bin/env python3
"""
将 auto_output 内所有视频按物体分组拼接，并在画面上叠加参数信息（分行显示）。
- 物体名：旧布局为文件夹名第一个 __ 之前；新布局（数字目录）读 gt_parameters.json 的 ply_stem
- 参数：旧布局解析 __ 之间片段；新布局用 gt_parameters.json 的 material_params
- 输出：auto_output/combined/{物体名}.mp4
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from collections import Counter, defaultdict


# 动作名（用于解析 run 文件夹名末尾段）；支持 auto_output/<action>/ 或 auto_output/<model>/ 两种顶层
ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")


def _walk_root(auto_output_root: Path):
    """返回用于 os.walk 的根路径。Windows 下使用 \\?\ 长路径前缀，避免 260 字符限制导致漏扫深层目录。"""
    root = auto_output_root.resolve()
    if os.name == "nt":
        root_str = str(root)
        if not root_str.startswith("\\\\?\\"):
            root_str = "\\\\?\\" + root_str
        return root_str, Path(root_str)
    return str(root), root


def _material_params_to_overlay_str(material_params: dict) -> str:
    if not material_params:
        return ""
    return "_".join(
        f"{k}={material_params[k]}"
        for k in sorted(material_params.keys(), key=lambda x: str(x))
    )


def find_all_videos(auto_output_root: Path):
    """扫描 .../<run>/.../videos/*.mp4。支持顶层为动作名或物体名，以及数字目录 + gt_parameters.json。"""
    results = []
    root_str, walk_root = _walk_root(auto_output_root)
    ao = auto_output_root.resolve()
    for dirpath, _dirnames, filenames in os.walk(root_str):
        for f in filenames:
            if not f.lower().endswith(".mp4"):
                continue
            mp4_path = Path(dirpath) / f
            try:
                rel = mp4_path.relative_to(walk_root)
                parts = rel.parts
                if len(parts) < 4:
                    continue
                if parts[-2] != "videos":
                    continue
                normal_path = ao / Path(*parts)

                # 新 by_model：<model>/<NNNN>/videos/*.mp4
                if len(parts) == 4 and parts[1].isdigit():
                    gt_path = ao / parts[0] / parts[1] / "gt_parameters.json"
                    if not gt_path.is_file():
                        continue
                    with open(gt_path, "r", encoding="utf-8") as gf:
                        gt = json.load(gf)
                    action = gt.get("sim_type", "")
                    if action not in ACTION_NAMES:
                        continue
                    object_name = str(gt.get("ply_stem", parts[0]))
                    params_str = _material_params_to_overlay_str(gt.get("material_params") or {})
                    results.append((normal_path, object_name, params_str, action))
                    continue

                # 新 by_action：<action>/<model>/<NNNN>/videos/*.mp4
                if (
                    len(parts) == 5
                    and parts[0] in ACTION_NAMES
                    and parts[2].isdigit()
                ):
                    gt_path = ao / parts[0] / parts[1] / parts[2] / "gt_parameters.json"
                    if not gt_path.is_file():
                        continue
                    with open(gt_path, "r", encoding="utf-8") as gf:
                        gt = json.load(gf)
                    action = gt.get("sim_type", "")
                    if action != parts[0] or action not in ACTION_NAMES:
                        continue
                    object_name = str(gt.get("ply_stem", parts[1]))
                    params_str = _material_params_to_overlay_str(gt.get("material_params") or {})
                    results.append((normal_path, object_name, params_str, action))
                    continue

                # 旧：至少 5 段，run 文件夹名含 __
                if len(parts) < 5:
                    continue
                run_folder_name = parts[1]
                if "__" not in run_folder_name:
                    continue
                segs = run_folder_name.split("__")
                if len(segs) < 3:
                    continue
                object_name = segs[0]
                params_str = "__".join(segs[1:-1])
                action = segs[-1]
                if action not in ACTION_NAMES:
                    continue
                if parts[0] in ACTION_NAMES and parts[0] != action:
                    continue
                results.append((normal_path, object_name, params_str, action))
            except (ValueError, IndexError, OSError, json.JSONDecodeError):
                continue
    return results


# 画面上只显示这些参数（按此顺序）
DISPLAY_PARAM_KEYS = ("motion", "E", "density", "nu", "material", "hardening", "softening", "yield_stress")


def _parse_params_to_dict(params_str: str) -> dict:
    """将 E=1.62e+04_density=1.28e+03_... 解析为 {key: value}，支持 key 中含下划线如 yield_stress。"""
    if not params_str.strip():
        return {}
    tokens = params_str.split("_")
    result = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" in t:
            k, _, v = t.partition("=")
            result[k] = v
            i += 1
        else:
            if i + 1 < len(tokens) and "=" not in tokens[i + 1]:
                result[t] = tokens[i + 1]
                i += 2
            else:
                i += 1
    return result


def params_str_to_display_lines(params_str: str, action: str = "") -> str:
    """只显示 motion、E、density、nu、material、hardening、softening、yield_stress，按固定顺序分行。"""
    param_dict = _parse_params_to_dict(params_str)
    if action:
        param_dict["motion"] = action
    lines = []
    for key in DISPLAY_PARAM_KEYS:
        if key in param_dict:
            lines.append(f"{key}={param_dict[key]}")
    return "\n".join(lines) if lines else "(无参数)"


def add_text_overlay_ffmpeg(
    input_path: Path,
    output_path: Path,
    text_lines: str,
    text_file_dir: Path,
    run_cwd: Path,
    fontsize: int = 20,
    margin_x: int = 20,
    margin_y: int = 40,
) -> bool:
    """对单个视频叠加多行文字，输出到 output_path。使用 textfile 避免命令行转义问题。run_cwd 为运行 ffmpeg 的工作目录，用于生成无冒号的相对路径。"""
    # 使用 textfile 时内容里的 % 要写成 %%
    safe_content = text_lines.replace("%", "%%")
    # 确保临时目录存在（Linux 上若用户手动删除 _temp_overlays 会导致 mkstemp 抛 FileNotFoundError）
    text_file_dir.mkdir(parents=True, exist_ok=True)
    fd, text_file = tempfile.mkstemp(suffix=".txt", dir=text_file_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(safe_content)
        text_path = Path(text_file).resolve()
        # Windows 下 drawtext 的 textfile= 中路径若含 ":"（如 C:）会被当成选项分隔符，故用相对路径
        try:
            text_file_rel = text_path.relative_to(run_cwd)
        except ValueError:
            text_file_rel = text_path
        text_file_for_filter = text_file_rel.as_posix()
        # drawtext: 从文件读文本，支持 \n 换行；放在左上角并加描边。
        # 仅在 Windows 下尝试指定字体文件；Linux/其他平台交给 Fontconfig 选择默认字体。
        font_opt = ""
        if os.name == "nt":
            system_root = os.environ.get("SystemRoot", "C:/Windows")
            arial_path = Path(system_root) / "Fonts" / "arial.ttf"
            if arial_path.exists():
                # 仅转义盘符冒号，供 ffmpeg drawtext 解析
                font_path = arial_path.as_posix().replace(":", "\\:", 1)
                font_opt = f":fontfile='{font_path}'"
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf",
            (
                f"drawtext=textfile='{text_file_for_filter}'"
                f"{font_opt}"
                f":fontsize={fontsize}:fontcolor=white:borderw=2:bordercolor=black"
                f":x={margin_x}:y={margin_y}"
                ":line_spacing=0"
            ),
            "-c:a", "copy",
            str(output_path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=run_cwd)
        if out.returncode != 0:
            # 只打印最后几行错误信息，避免刷屏
            err_lines = (out.stderr or "").strip().split("\n")
            err_tail = "\n".join(err_lines[-4:]) if len(err_lines) >= 4 else out.stderr[:500]
            print(f"[WARN] ffmpeg overlay 失败: {err_tail}")
            return False
        return True
    finally:
        try:
            os.unlink(text_file)
        except OSError:
            pass


def concat_videos_ffmpeg(video_paths: list, output_path: Path) -> bool:
    """将多个视频按顺序拼接为 output_path。"""
    if not video_paths:
        return False
    if len(video_paths) == 1:
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return True
    list_path = output_path.with_suffix(".concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in video_paths:
            # 使用正斜杠路径，避免 Windows 下反斜杠转义问题
            path_str = Path(p).resolve().as_posix()
            path_escaped = path_str.replace("'", "'\\''")
            f.write(f"file '{path_escaped}'\n")
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            print(f"[WARN] ffmpeg concat 失败: {out.stderr[:500]}")
            return False
        return True
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass


def main():
    script_dir = Path(__file__).resolve().parent
    run_cwd = script_dir  # ffmpeg 在此目录运行，以便 textfile 使用相对路径（避免 Windows C: 冒号问题）
    auto_output = script_dir / "auto_output"
    if not auto_output.is_dir():
        print(f"未找到目录: {auto_output}")
        return
    combined_dir = auto_output / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = combined_dir / "_temp_overlays"
    temp_dir.mkdir(parents=True, exist_ok=True)

    videos = find_all_videos(auto_output)
    if not videos:
        print("未在 auto_output 下找到任何 mp4 视频。")
        return

    # 按物体名分组；同一物体内可按 (action, params) 排序以便结果稳定
    by_object = defaultdict(list)
    for path, obj, params, action in videos:
        by_object[obj].append((path, params, action))

    action_counts = Counter(v[3] for v in videos)
    print(f"共找到 {len(videos)} 个视频，{len(by_object)} 个物体。各动作数量: {dict(action_counts)}")

    for obj_name, items in sorted(by_object.items()):
        # 同一物体内按 action 再按 params 排序
        items_sorted = sorted(items, key=lambda x: (x[2], x[1]))
        overlay_paths = []
        for idx, (mp4_path, params_str, action) in enumerate(items_sorted):
            full_text = params_str_to_display_lines(params_str, action)
            out_name = f"{obj_name}_{idx:03d}_{action}.mp4".replace("/", "_").replace("\\", "_")
            overlay_path = temp_dir / out_name
            if not add_text_overlay_ffmpeg(mp4_path, overlay_path, full_text, temp_dir, run_cwd):
                # 失败时直接使用原视频路径（不叠加文字）
                overlay_paths.append(mp4_path)
            else:
                overlay_paths.append(overlay_path)
        out_mp4 = combined_dir / f"{obj_name}.mp4"
        if concat_videos_ffmpeg(overlay_paths, out_mp4):
            print(f"已生成: {out_mp4}")
        else:
            print(f"拼接失败: {obj_name}")

    # 清理临时带 overlay 的片段
    for f in temp_dir.iterdir():
        try:
            f.unlink()
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass
    print("完成。")


if __name__ == "__main__":
    main()
