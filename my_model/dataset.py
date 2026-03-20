"""
PhysGaussian 数据集：从 auto_output 读取时序图像 + 物理参数。

数据结构（以实际目录为准）：
auto_output/<action>/<OBJECT__PARAMS__action>/<obj>/images/<camera_name>/*.png
或（仿真 runner 默认 output_layout=by_model）：
auto_output/<OBJECT>/<OBJECT__PARAMS__action>/<obj>/images/...
或（runner 新布局）：auto_output/<OBJECT>/<NNNN>/images/...，参数见 gt_parameters.json

扁平数据集（transform_dataset 生成）：
auto_output/<dataset_dir>/{train,test}/<编号>/images/*.png + gt.json（默认 dataset_dir=dataset_400）
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")
MATERIAL_CATEGORIES = ("jelly", "metal", "plasticine")
NUM_FEATURES = ["E", "nu", "density", "yield_stress"]


def resolve_flat_dataset_root(dataset_dir: str, auto_output: Path) -> Path:
    """
    扁平数据集根目录（其下应有 train/、test/）。

    - **推荐**：相对 ``auto_output`` 的一级名，如 ``dataset_400`` → ``<phys_root>/auto_output/dataset_400``。
    - **兼容误写**：若写成 ``auto_output/dataset_400``，则按 **PhysGaussian 根** 解析，不再与 ``auto_output`` 拼两次
      （否则会错误得到 ``auto_output/auto_output/dataset_400``）。
    - **绝对路径**：直接使用。
    """
    p = Path(dataset_dir).expanduser()
    if p.is_absolute():
        return p.resolve()
    ap = Path(auto_output).resolve()
    parts = p.parts
    if parts and parts[0] == "auto_output":
        return ap.parent.joinpath(*parts).resolve()
    return (ap / p).resolve()


def parse_params_to_dict(params_str: str) -> Dict[str, str]:
    """解析 E=1.62e+04_density=1.28e+03_... 为 {key: value}"""
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


def _material_params_to_str_dict(mp) -> Dict[str, str]:
    if not isinstance(mp, dict):
        return {}
    return {str(k): str(v) for k, v in mp.items()}


def iter_auto_output_runs(auto_output_root: Path):
    """
    遍历 auto_output，返回 (action, run_dir, params_dict)。
    run_dir 为仿真样本根目录（其下含 images/；旧布局时可能为中间层目录且内含 <obj>/）。
    支持：auto_output/<action>/... 与 auto_output/<model>/...，以及数字目录 + gt_parameters.json。
    """
    seen: set[str] = set()

    def emit(act: str, run_dir: Path, params: Dict[str, str]):
        k = str(run_dir.resolve())
        if k in seen:
            return
        seen.add(k)
        return (act, run_dir, params)

    for action in ACTION_NAMES:
        action_dir = auto_output_root / action
        if not action_dir.is_dir():
            continue
        for run_dir in action_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if "__" not in name or len(name.split("__")) < 3:
                continue
            segs = name.split("__")
            params_str = "__".join(segs[1:-1])
            params = parse_params_to_dict(params_str)
            out = emit(action, run_dir, params)
            if out is not None:
                yield out
        # auto_output/<action>/<model>/<NNNN>/
        for model_sub in sorted(action_dir.iterdir(), key=lambda p: p.name):
            if not model_sub.is_dir():
                continue
            if model_sub.name in ACTION_NAMES:
                continue
            for run_dir in sorted(model_sub.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue
                gt_path = run_dir / "gt_parameters.json"
                if not gt_path.is_file():
                    continue
                try:
                    with open(gt_path, "r", encoding="utf-8") as f:
                        gt = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                action_from = str(gt.get("sim_type", ""))
                if action_from != action:
                    continue
                params = _material_params_to_str_dict(gt.get("material_params"))
                out = emit(action_from, run_dir, params)
                if out is not None:
                    yield out

    skip_top = set(ACTION_NAMES) | {"_tmp_configs", "stats", "combined"}
    for model_top in sorted(auto_output_root.iterdir()):
        if not model_top.is_dir() or model_top.name in skip_top:
            continue
        for run_dir in sorted(model_top.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            # 新：样本根即 <model>/<NNNN>/
            if name.isdigit() and (run_dir / "gt_parameters.json").is_file():
                try:
                    with open(run_dir / "gt_parameters.json", "r", encoding="utf-8") as f:
                        gt = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                action_from = str(gt.get("sim_type", ""))
                if action_from not in ACTION_NAMES:
                    continue
                params = _material_params_to_str_dict(gt.get("material_params"))
                out = emit(action_from, run_dir, params)
                if out is not None:
                    yield out
                continue
            if "__" not in name or len(name.split("__")) < 3:
                continue
            segs = name.split("__")
            action_from = segs[-1]
            if action_from not in ACTION_NAMES:
                continue
            params_str = "__".join(segs[1:-1])
            params = parse_params_to_dict(params_str)
            out = emit(action_from, run_dir, params)
            if out is not None:
                yield out


def find_video_in_run_dir(run_dir: Path) -> Optional[Path]:
    """在仿真目录下查找视频文件"""
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    for root, _, files in os.walk(run_dir):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in exts):
                return Path(root) / fname
    return None


def _sorted_image_files(image_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def find_images_dir(run_dir: Path) -> Optional[Path]:
    """
    在单个仿真目录下查找 images 目录（通常在 <obj>/images 下）。
    返回包含帧图片的目录（可能还有一层相机目录）。
    """
    # 常见结构：run_dir/<obj>/images/<camera>/*.png
    for root, dirs, _files in os.walk(run_dir):
        if "images" in dirs:
            images_root = Path(root) / "images"
            # 若 images_root 下还有一层子目录（相机名），优先选第一个有图的
            subdirs = [d for d in images_root.iterdir() if d.is_dir()]
            if subdirs:
                for sd in sorted(subdirs, key=lambda p: p.name):
                    if _sorted_image_files(sd):
                        return sd
            if _sorted_image_files(images_root):
                return images_root
    return None


def is_image_sequence_readable(image_dir: Path) -> bool:
    files = _sorted_image_files(image_dir)
    if not files:
        return False
    img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    return img is not None


def extract_frames_from_images(image_dir: Path, num_frames: int = 16) -> np.ndarray:
    """
    从 images 目录中读取时序帧，返回 (T, H, W, 3) 的 numpy 数组。
    **长度恒为 num_frames**：在 [0, N-1] 上均匀取 num_frames 个索引（N 少则同一帧会重复出现）。
    这样 DataLoader 才能稳定 stack；若用「不足 num_frames 就全读」，batch 内 T 不一致会触发 collate 报错。
    """
    files = _sorted_image_files(image_dir)
    if not files:
        raise ValueError(f"images 目录无图片: {image_dir}")

    n = len(files)
    # 与「帧数 > num_frames 时 subsample」同一规则；n<=num_frames 时也保持输出 T=num_frames
    idxs = np.linspace(0, n - 1, num_frames).astype(int).tolist()

    frames = []
    for i in idxs:
        img = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    if not frames:
        raise ValueError(f"图片读取失败: {image_dir}")
    return np.stack(frames, axis=0)


def safe_float(params: Dict[str, str], key: str) -> float:
    v = params.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def material_to_idx(material: str) -> int:
    m = str(material).lower()
    if m in MATERIAL_CATEGORIES:
        return MATERIAL_CATEGORIES.index(m)
    if "jelly" in m:
        return 0
    if "metal" in m:
        return 1
    return 2  # plasticine


def action_to_idx(action: str) -> int:
    a = str(action).lower()
    if a in ACTION_NAMES:
        return ACTION_NAMES.index(a)
    return 0


class PhysGaussianDataset(Dataset):
    """
    时序图像 + 物理参数数据集。
    每个样本：frames (T, H, W, 3), params (E, nu, density, yield_stress), material_idx, action_idx
    """

    def __init__(
        self,
        auto_output_root: Path,
        num_frames: int = 16,
        img_size: int = 224,
        sample_ids: Optional[List[Tuple[str, Path, Dict[str, str]]]] = None,
    ):
        self.auto_output_root = Path(auto_output_root)
        self.num_frames = num_frames
        self.img_size = img_size

        if sample_ids is None:
            self.samples = [
                (action, run_dir, params)
                for action, run_dir, params in iter_auto_output_runs(self.auto_output_root)
            ]
        else:
            self.samples = sample_ids

        # 过滤出有可读 images 的样本
        self.valid_samples: List[Tuple[str, Path, Dict[str, str], Path]] = []
        for action, run_dir, params in self.samples:
            image_dir = find_images_dir(run_dir)
            if image_dir is None:
                continue
            if is_image_sequence_readable(image_dir):
                self.valid_samples.append((action, run_dir, params, image_dir))

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 如果遇到极少数运行时不可读图片，尝试跳过到下一个样本
        for _ in range(3):
            action, run_dir, params, image_dir = self.valid_samples[idx]
            try:
                frames = extract_frames_from_images(image_dir, self.num_frames)
                break
            except Exception:
                idx = (idx + 1) % len(self.valid_samples)
        else:
            # 连续失败，抛出异常以便用户发现数据问题
            raise ValueError(f"连续多个样本图片无法读取，最后一个路径: {image_dir}")
        #  resize 并归一化到 [0,1]
        frames_resized = []
        for f in frames:
            f = cv2.resize(f, (self.img_size, self.img_size))
            frames_resized.append(f)
        frames = np.stack(frames_resized, axis=0).astype(np.float32) / 255.0
        # (T, H, W, C) -> (T, C, H, W)；transpose 后为非连续数组，勿用 torch.from_numpy 直接进多进程 DataLoader
        frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))

        E = safe_float(params, "E")
        nu = safe_float(params, "nu")
        density = safe_float(params, "density")
        yield_stress = safe_float(params, "yield_stress")
        params_tensor = torch.tensor([E, nu, density, yield_stress], dtype=torch.float32)

        material_idx = material_to_idx(params.get("material", "plasticine"))
        action_idx = action_to_idx(action)

        # torch.tensor 拷贝一份，避免与 numpy 共享不可 resize 的 storage（DDP/多 worker collate 会报错）
        frames_t = torch.tensor(frames, dtype=torch.float32)
        return (
            frames_t,
            params_tensor,
            torch.tensor(material_idx, dtype=torch.long),
            torch.tensor(action_idx, dtype=torch.long),
        )


def list_dataset400_sample_dirs(split_root: Path) -> List[Path]:
    """dataset_400 下按数字排序的样本目录列表。"""
    split_root = Path(split_root)
    if not split_root.is_dir():
        return []
    dirs = [p for p in split_root.iterdir() if p.is_dir() and p.name.isdigit()]
    dirs.sort(key=lambda p: int(p.name))
    return dirs


def _regression_float(reg: Dict, key: str, default: float = 0.0) -> float:
    """gt.json 中 jelly 等材质 yield_stress 常为 null；reg.get(k, d) 在键存在且值为 null 时仍会得到 None。"""
    v = reg.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_gt_from_dir(sample_dir: Path) -> Tuple[str, Dict[str, str], Dict[str, float]]:
    with open(sample_dir / "gt.json", "r", encoding="utf-8") as f:
        gt = json.load(f)
    action = str(gt.get("action", "bend"))
    params = {str(k): str(v) for k, v in gt.get("params", {}).items()}
    reg = gt.get("regression", {})
    regression = {
        "E": _regression_float(reg, "E", 0.0),
        "nu": _regression_float(reg, "nu", 0.0),
        "density": _regression_float(reg, "density", 0.0),
        "yield_stress": _regression_float(reg, "yield_stress", 0.0),
    }
    return action, params, regression


def _images_dir_has_any_frame(img_dir: Path) -> bool:
    """仅检查 images 下是否存在常见图片扩展名文件，不读像素（用于快速启动）。"""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            return True
    return False


def dataset400_sample_fully_readable(sample_dir: Path, num_frames: int, img_size: int) -> bool:
    """
    与 __getitem__ 一致地尝试加载：gt.json + 均匀采样帧 + 每帧 resize。
    用于构建数据集时剔除坏样本，避免 DataLoader worker 里崩或错误地用「下一个样本」顶替当前 idx。
    """
    try:
        _action, _params, _reg = _load_gt_from_dir(sample_dir)
        image_dir = sample_dir / "images"
        frames = extract_frames_from_images(image_dir, num_frames)
        for f in frames:
            _ = cv2.resize(f, (img_size, img_size))
        return True
    except Exception:
        return False


class Dataset400(Dataset):
    """
    读取 auto_output/dataset_400/{train|test}/<id>/ 下的扁平样本。
    gt.json 提供 action、params（材质等）与 regression（E, nu, density, yield_stress）。
    """

    def __init__(
        self,
        split_root: Path,
        num_frames: int = 16,
        img_size: int = 224,
        preflight: bool = True,
        verbose_preflight: bool = True,
    ):
        self.split_root = Path(split_root)
        self.num_frames = num_frames
        self.img_size = img_size

        self.sample_dirs: List[Path] = []
        all_dirs = list_dataset400_sample_dirs(self.split_root)
        if preflight and verbose_preflight:
            print(
                f"[Dataset400] 预检：共 {len(all_dirs)} 个编号目录，每 {max(1, len(all_dirs) // 20)} 个打印一次进度…",
                flush=True,
            )
        step_prog = max(1, len(all_dirs) // 20) if len(all_dirs) > 20 else 5

        for di, d in enumerate(all_dirs):
            if preflight and verbose_preflight and di > 0 and di % step_prog == 0:
                print(f"[Dataset400] 预检进度 {di}/{len(all_dirs)} …", flush=True)
            gt_path = d / "gt.json"
            img_dir = d / "images"
            if not gt_path.is_file() or not img_dir.is_dir():
                continue
            # preflight=False（--no_dataset_preflight）时不再 cv2 读首帧，否则 NFS 上仍要几百次 imread，启动很慢
            if preflight:
                if not is_image_sequence_readable(img_dir):
                    if verbose_preflight:
                        print(f"[Dataset400] 跳过（首帧不可读）: {d}", flush=True)
                    continue
            else:
                if not _images_dir_has_any_frame(img_dir):
                    if verbose_preflight:
                        print(f"[Dataset400] 跳过（images 下无图片文件）: {d}", flush=True)
                    continue
            if preflight:
                if not dataset400_sample_fully_readable(d, num_frames, img_size):
                    if verbose_preflight:
                        print(
                            f"[Dataset400] 跳过（完整加载失败；常见：PNG 损坏，或 gt.json 数值异常）: {d}",
                            flush=True,
                        )
                    continue
            self.sample_dirs.append(d)

        if not self.sample_dirs:
            raise ValueError(
                f"Dataset400: 在 {self.split_root} 下没有可用样本（请检查 gt.json / images 或关闭 preflight 自行承担风险）"
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image_dir = sample_dir / "images"
        action, params, reg = _load_gt_from_dir(sample_dir)
        frames = extract_frames_from_images(image_dir, self.num_frames)

        frames_resized = []
        for f in frames:
            f = cv2.resize(f, (self.img_size, self.img_size))
            frames_resized.append(f)
        frames = np.stack(frames_resized, axis=0).astype(np.float32) / 255.0
        frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))

        params_tensor = torch.tensor(
            [reg["E"], reg["nu"], reg["density"], reg["yield_stress"]],
            dtype=torch.float32,
        )
        material_idx = material_to_idx(params.get("material", "plasticine"))
        action_idx = action_to_idx(action)

        frames_t = torch.tensor(frames, dtype=torch.float32)
        return (
            frames_t,
            params_tensor,
            torch.tensor(material_idx, dtype=torch.long),
            torch.tensor(action_idx, dtype=torch.long),
        )


def train_test_split(
    auto_output_root: Path, train_ratio: float = 0.7, seed: int = 42
) -> Tuple[List[Tuple[str, Path, Dict]], List[Tuple[str, Path, Dict]]]:
    """按 7:3 分割训练集和测试集"""
    import random

    samples = [
        (action, run_dir, params)
        for action, run_dir, params in iter_auto_output_runs(Path(auto_output_root))
    ]
    img_samples = []
    for action, run_dir, params in samples:
        image_dir = find_images_dir(run_dir)
        if image_dir is not None and is_image_sequence_readable(image_dir):
            img_samples.append((action, run_dir, params))

    rng = random.Random(seed)
    rng.shuffle(img_samples)
    n_train = int(len(img_samples) * train_ratio)
    return img_samples[:n_train], img_samples[n_train:]
