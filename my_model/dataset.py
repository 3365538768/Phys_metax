"""
PhysGaussian 数据集：从 auto_output 读取时序图像 + 物理参数。

数据结构（以实际目录为准）：
auto_output/<action>/<OBJECT__PARAMS__action>/<obj>/images/<camera_name>/*.png
"""
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


def iter_auto_output_runs(auto_output_root: Path):
    """遍历 auto_output，返回 (action, run_dir, params_dict)"""
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
            yield action, run_dir, params


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
    采用均匀采样到 num_frames。
    """
    files = _sorted_image_files(image_dir)
    if not files:
        raise ValueError(f"images 目录无图片: {image_dir}")

    if len(files) <= num_frames:
        idxs = list(range(len(files)))
    else:
        idxs = np.linspace(0, len(files) - 1, num_frames).astype(int).tolist()

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
        # (T, H, W, C) -> (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))

        E = safe_float(params, "E")
        nu = safe_float(params, "nu")
        density = safe_float(params, "density")
        yield_stress = safe_float(params, "yield_stress")
        params_tensor = torch.tensor([E, nu, density, yield_stress], dtype=torch.float32)

        material_idx = material_to_idx(params.get("material", "plasticine"))
        action_idx = action_to_idx(action)

        return (
            torch.from_numpy(frames),
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
