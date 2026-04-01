from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# 与写入侧一致的时间重采样（避免重复实现）
from my_utils.arch4_lmdb import _resample_thwc_float, unpack_uint8_thwc

_REQ_MODALITIES = ("rgb", "stress", "flow", "force_mask")
_OPT_MODALITIES = ("object_mask",)


def list_lmdb_samples(split_root: str, lmdb_env_subdir: str = "arch4_data.lmdb") -> List[Path]:
    """列出 split_root 下所有包含 arch4 LMDB 的样本目录。"""
    root = Path(split_root)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for d in sorted(root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        env = d / lmdb_env_subdir
        if env.is_dir() and (env / "data.mdb").is_file():
            out.append(d)
    return out


def _read_meta(env_path: Path) -> Dict[str, Any]:
    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e

    env = lmdb.open(str(env_path), readonly=True, lock=False, max_readers=8)
    try:
        with env.begin() as txn:
            raw = txn.get(b"__meta__")
            if raw is None:
                raise ValueError(f"缺少 __meta__: {env_path}")
            return json.loads(raw.decode("utf-8"))
    finally:
        env.close()


def _is_true_rgb_u8(thwc_u8: np.ndarray) -> bool:
    """
    输入 [T,3,H,W] uint8。
    若三通道并非完全相同，则视为真 RGB。
    """
    if thwc_u8.ndim != 4 or int(thwc_u8.shape[1]) != 3:
        return False
    c0 = thwc_u8[:, 0, :, :]
    c1 = thwc_u8[:, 1, :, :]
    c2 = thwc_u8[:, 2, :, :]
    return bool(np.any(c0 != c1) or np.any(c0 != c2))


def _get_lmdb_blob_exact(
    txn: Any,
    *,
    view_name: str,
    mod: str,
) -> tuple[bytes | None, str]:
    """
    按给定键精确读取 LMDB。
    返回 (blob, 实际命中键名)。
    """
    key_main = f"{view_name}/{mod}"
    blob = txn.get(key_main.encode("utf-8"))
    return blob, key_main


# --- worker 进程内 LMDB env 复用（LRU，避免多样本长期训练时无限增长）---
_LMDB_ENV_LRU_MAX = 64
_lmdb_env_lru: "OrderedDict[str, Any]" = OrderedDict()


def _get_cached_lmdb_env(env_path: Path) -> Any:
    """只读 LMDB；同一 worker 进程内按路径复用 env，超出容量时关闭最久未用的。"""
    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e

    key = str(env_path.resolve())
    global _lmdb_env_lru
    if key in _lmdb_env_lru:
        _lmdb_env_lru.move_to_end(key)
        return _lmdb_env_lru[key]
    while len(_lmdb_env_lru) >= _LMDB_ENV_LRU_MAX:
        _old_k, old_env = _lmdb_env_lru.popitem(last=False)
        old_env.close()
    env = lmdb.open(
        key,
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=256,
    )
    _lmdb_env_lru[key] = env
    return env


def _read_lmdb_arrays_one_txn(
    txn: Any,
    views: List[str],
    *,
    max_views: int,
    num_frames_opt: Optional[int],
) -> Dict[str, np.ndarray]:
    """
    单次事务内读完同一 LMDB 中全部视角与模态（含 object_mask），
    与旧版「先读主再 full read」数值语义一致。
    """
    views_use = list(views)[: int(max_views)]
    if not views_use:
        raise ValueError("__meta__ 里无可用 views")

    nf = num_frames_opt
    resample = nf is not None and int(nf) > 0

    buf: Dict[str, List[np.ndarray]] = {m: [] for m in (_REQ_MODALITIES + _OPT_MODALITIES)}
    for v in views_use:
        for mod in _REQ_MODALITIES:
            blob, key_used = _get_lmdb_blob_exact(txn, view_name=v, mod=mod)
            if blob is None:
                raise KeyError(f"缺少键: {key_used!r}")
            u8 = unpack_uint8_thwc(bytes(blob))
            thwc = u8.astype(np.float32) / 255.0
            if resample:
                thwc = _resample_thwc_float(thwc, int(nf))
            chw_t = np.transpose(thwc, (1, 0, 2, 3))
            buf[mod].append(np.ascontiguousarray(chw_t, dtype=np.float32))

        blob_obj, _ = _get_lmdb_blob_exact(txn, view_name=v, mod="object_mask")
        if blob_obj is None:
            shape_ref = buf["force_mask"][-1].shape
            obj = np.ones(shape_ref, dtype=np.float32)
        else:
            u8 = unpack_uint8_thwc(bytes(blob_obj))
            thwc = u8.astype(np.float32) / 255.0
            if resample:
                thwc = _resample_thwc_float(thwc, int(nf))
            obj = np.transpose(thwc, (1, 0, 2, 3))
            obj = np.ascontiguousarray(obj, dtype=np.float32)
        buf["object_mask"].append(obj)

    out: Dict[str, np.ndarray] = {}
    for mod, arrs in buf.items():
        if not arrs:
            raise ValueError(f"模态为空: {mod}")
        shape0 = arrs[0].shape
        for i, a in enumerate(arrs):
            if a.shape != shape0:
                raise ValueError(
                    f"模态 {mod} 视角帧形状不一致: view0={shape0}, view{i}={a.shape}"
                )
        out[mod] = np.stack(arrs, axis=0)
    return out


def _read_lmdb_arrays_from_env_cached(
    env_path: Path,
    meta: Dict[str, Any],
    *,
    max_views: int,
    num_frames_opt: Optional[int],
    img_size_req: Optional[int],
) -> Dict[str, np.ndarray]:
    """
    一次打开 env（LRU 缓存）、一次事务内读完所有模态；num_frames>0 时与 read_arch4_lmdb_view_tensors 同重采样语义。
    """
    nf = num_frames_opt
    if nf is not None and int(nf) > 0:
        stored_h = int(meta.get("img_size") or 0)
        req = int(img_size_req) if img_size_req is not None else 0
        if stored_h > 0 and req > 0 and req != stored_h:
            raise ValueError(
                f"LMDB img_size={stored_h} 与请求的 img_size={req} 不一致"
            )

    views = list(meta.get("views") or [])
    env = _get_cached_lmdb_env(env_path)
    with env.begin() as txn:
        return _read_lmdb_arrays_one_txn(
            txn, views, max_views=max_views, num_frames_opt=nf
        )


def inspect_lmdb_format(sample_dir: str, lmdb_env_subdir: str = "arch4_data.lmdb") -> Dict[str, Any]:
    """
    检查 LMDB 元信息与实际数据键格式。
    返回每个模态是否为真 RGB（非三通道重复灰度）。
    """
    sample = Path(sample_dir)
    env_path = sample / lmdb_env_subdir
    if not env_path.is_dir():
        raise FileNotFoundError(f"LMDB 目录不存在: {env_path}")

    meta = _read_meta(env_path)
    views = list(meta.get("views") or [])
    if not views:
        raise ValueError(f"__meta__ 里无 views: {env_path}")

    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e

    first_view = views[0]
    out: Dict[str, Any] = {
        "sample_dir": str(sample.resolve()),
        "lmdb_path": str(env_path.resolve()),
        "meta": meta,
        "force_mask_source_dir": str(meta.get("force_mask_source_dir", "")),
        "first_view": first_view,
        "storage_format": {},
    }

    env = lmdb.open(str(env_path), readonly=True, lock=False, readahead=True, max_readers=32)
    try:
        with env.begin() as txn:
            for mod in (_REQ_MODALITIES + _OPT_MODALITIES):
                blob, key_used = _get_lmdb_blob_exact(
                    txn, view_name=first_view, mod=mod
                )
                if blob is None:
                    out["storage_format"][mod] = {"exists": False}
                    continue
                thwc_u8 = unpack_uint8_thwc(bytes(blob))
                t, c, h, w = map(int, thwc_u8.shape)
                out["storage_format"][mod] = {
                    "exists": True,
                    "key": key_used,
                    "shape_TCHW": [t, c, h, w],
                    "dtype": str(thwc_u8.dtype),
                    "is_true_rgb": _is_true_rgb_u8(thwc_u8),
                }
    finally:
        env.close()
    return out


def _read_lmdb_arrays_full_frames(
    sample_dir: str,
    *,
    max_views: int,
    lmdb_env_subdir: str = "arch4_data.lmdb",
) -> Dict[str, np.ndarray]:
    """
    按 LMDB 实际存储帧数读取（不重采样）：
    返回 dict，键为 rgb/stress/flow/force_mask/object_mask，值形状 [V,3,T,H,W]。
    """
    sample = Path(sample_dir)
    env_path = sample / lmdb_env_subdir
    meta = _read_meta(env_path)
    return _read_lmdb_arrays_from_env_cached(
        env_path,
        meta,
        max_views=int(max_views),
        num_frames_opt=None,
        img_size_req=None,
    )


def read_lmdb_arrays(
    sample_dir: str,
    *,
    num_frames: Optional[int] = None,
    img_size: Optional[int] = None,
    max_views: int,
    lmdb_env_subdir: str = "arch4_data.lmdb",
) -> Dict[str, np.ndarray]:
    """
    读取 LMDB 为 float32 (0~1)：
    返回 dict，键为 rgb/stress/flow/force_mask/object_mask，值形状 [V,3,T,H,W]。
    - num_frames=None 或 <=0：按 LMDB 实际总帧数完整读取（不重采样）
    - num_frames>0：单次事务内读全模态并统一重采样（与旧版数值语义一致，不再二次打开 LMDB）
    """
    env_path = Path(sample_dir) / lmdb_env_subdir
    meta = _read_meta(env_path)

    if num_frames is None or int(num_frames) <= 0:
        return _read_lmdb_arrays_from_env_cached(
            env_path,
            meta,
            max_views=int(max_views),
            num_frames_opt=None,
            img_size_req=None,
        )

    img_sz = int(img_size) if img_size is not None else 0
    if img_sz <= 0:
        img_sz = int(meta.get("img_size") or 0)
    if img_sz <= 0:
        raise ValueError("img_size 无效；请传入 --img_size 或确保 __meta__.img_size 存在")

    return _read_lmdb_arrays_from_env_cached(
        env_path,
        meta,
        max_views=int(max_views),
        num_frames_opt=int(num_frames),
        img_size_req=int(img_sz),
    )


def save_rgb_previews(
    arrays: Dict[str, np.ndarray],
    out_dir: str,
    *,
    max_views: int = 2,
    max_frames: int = 2,
) -> List[str]:
    """
    仅保存“真 RGB”模态的预览图，输出 PNG 路径列表。
    arrays[mod]: [V,3,T,H,W] float32 in [0,1].
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("保存预览图需要 opencv-python（cv2）") from e

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for mod, vcthw in arrays.items():
        if vcthw.ndim != 5 or int(vcthw.shape[1]) != 3:
            continue
        # 全量判断是否真 RGB；若不是则跳过导出。
        u8_all = np.clip(vcthw * 255.0, 0, 255).astype(np.uint8)
        thwc_all = np.transpose(u8_all, (0, 2, 1, 3, 4)).reshape(-1, 3, vcthw.shape[3], vcthw.shape[4])
        if not _is_true_rgb_u8(thwc_all):
            continue

        v_lim = min(int(max_views), int(vcthw.shape[0]))
        t_lim = min(int(max_frames), int(vcthw.shape[2]))
        for vi in range(v_lim):
            for ti in range(t_lim):
                chw = vcthw[vi, :, ti, :, :]
                rgb_u8 = np.clip(chw * 255.0, 0, 255).astype(np.uint8)
                rgb_hwc = np.transpose(rgb_u8, (1, 2, 0))
                bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
                p = root / f"{mod}_view{vi:02d}_frame{ti:03d}.png"
                cv2.imwrite(str(p), bgr)
                written.append(str(p))
    return written


def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _load_gt_json(sample_dir: str) -> Dict[str, Any]:
    p = Path(sample_dir) / "gt.json"
    if not p.is_file():
        raise FileNotFoundError(f"gt.json 不存在: {p}")
    gt = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(gt, dict):
        raise ValueError(f"gt.json 不是 object: {p}")
    return gt


def _gt_dict_from_loaded(gt: Dict[str, Any], sample_name: str) -> Dict[str, Any]:
    """与 load_gt_regression 相同字段，供 __init__ 缓存，避免 __getitem__ 反复读 gt.json。"""
    reg = gt.get("regression") or {}
    params = gt.get("params") or gt.get("parameters") or {}
    return {
        "E": _coerce_float(reg.get("E", params.get("E", 0.0))),
        "nu": _coerce_float(reg.get("nu", params.get("nu", 0.0))),
        "density": _coerce_float(reg.get("density", params.get("density", 0.0))),
        "yield_stress": _coerce_float(reg.get("yield_stress", params.get("yield_stress", 0.0))),
        "action": str(gt.get("action", "")),
        "object": str(gt.get("object", "")),
        "sample_id": sample_name,
    }


def load_gt_regression(sample_dir: str) -> Dict[str, Any]:
    """
    从 gt.json 读取回归参数，优先 regression，缺失时回退 params，最后置 0.0。
    """
    gt = _load_gt_json(sample_dir)
    return _gt_dict_from_loaded(gt, Path(sample_dir).name)


def _np_to_torch_float32_share(x: np.ndarray) -> torch.Tensor:
    """保证 C 连续后 from_numpy，避免多余拷贝。"""
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return torch.from_numpy(x)


class LmdbGtDataset(Dataset):
    """
    仅 LMDB + gt.json 的数据集。

    每个样本返回：
    {
      "sample_id": str,
      "rgb"/"stress"/"flow"/"force_mask"/"object_mask": torch.Tensor [V,3,T,H,W],
      "params": torch.Tensor [4],               # [E, nu, density, yield_stress]
      "params_dict": dict,                      # 便于 inspect
    }
    """

    def __init__(
        self,
        split_root: str,
        *,
        lmdb_env_subdir: str = "arch4_data.lmdb",
        max_views: int = 4,
        num_frames: Optional[int] = None,
        img_size: Optional[int] = None,
        action_to_id: Optional[Dict[str, int]] = None,
        return_action_name: bool = False,
        sample_ids: Optional[Sequence[str]] = None,
    ):
        self.split_root = str(split_root)
        self.lmdb_env_subdir = str(lmdb_env_subdir)
        self.max_views = int(max_views)
        self.num_frames = None if num_frames is None else int(num_frames)
        self.img_size = None if img_size is None else int(img_size)
        self.return_action_name = bool(return_action_name)

        self.samples: List[Path] = []
        self._action_name_by_sample: Dict[str, str] = {}
        # __init__ 一次性缓存：gt 回归字段 + LMDB __meta__（高频路径不再读盘）
        self._gt_by_sample: Dict[str, Dict[str, Any]] = {}
        self._meta_by_sample: Dict[str, Dict[str, Any]] = {}
        root = Path(self.split_root)

        if sample_ids is not None:
            missing: List[str] = []
            for sid in sample_ids:
                s = str(sid).strip()
                if not s:
                    continue
                d = root / s
                env = d / self.lmdb_env_subdir
                if (
                    d.is_dir()
                    and (d / "gt.json").is_file()
                    and env.is_dir()
                    and (env / "data.mdb").is_file()
                ):
                    dr = d.resolve()
                    self.samples.append(dr)
                    gt = _load_gt_json(str(dr))
                    self._gt_by_sample[dr.name] = _gt_dict_from_loaded(gt, dr.name)
                    self._action_name_by_sample[dr.name] = (
                        str(gt.get("action", "unknown")).strip() or "unknown"
                    )
                    self._meta_by_sample[dr.name] = _read_meta(dr / self.lmdb_env_subdir)
                else:
                    missing.append(s)
            if missing:
                raise ValueError(
                    f"sample_ids 中 {len(missing)} 个在 {self.split_root} 下不可用（需 LMDB+gt.json）"
                    f"，示例: {missing[:8]}"
                )
        else:
            for d in list_lmdb_samples(self.split_root, self.lmdb_env_subdir):
                if (d / "gt.json").is_file():
                    self.samples.append(d)
                    gt = _load_gt_json(str(d))
                    self._gt_by_sample[d.name] = _gt_dict_from_loaded(gt, d.name)
                    self._action_name_by_sample[d.name] = (
                        str(gt.get("action", "unknown")).strip() or "unknown"
                    )
                    self._meta_by_sample[d.name] = _read_meta(d / self.lmdb_env_subdir)
        if not self.samples:
            raise ValueError(f"未找到包含 LMDB+gt.json 的样本: {self.split_root}")

        if action_to_id is None:
            action_names = sorted(set(self._action_name_by_sample.values()))
            self.action_to_id: Dict[str, int] = {n: i for i, n in enumerate(action_names)}
        else:
            self.action_to_id = {str(k): int(v) for k, v in action_to_id.items()}
        if not self.action_to_id:
            self.action_to_id = {"unknown": 0}
        self.id_to_action: Dict[int, str] = {v: k for k, v in self.action_to_id.items()}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_actions(self) -> int:
        return int(len(self.action_to_id))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d = self.samples[int(index)]
        gt = self._gt_by_sample[d.name]
        meta = self._meta_by_sample[d.name]
        nf = self.num_frames
        arrays = _read_lmdb_arrays_from_env_cached(
            d / self.lmdb_env_subdir,
            meta,
            max_views=self.max_views,
            num_frames_opt=(None if nf is None or int(nf) <= 0 else int(nf)),
            img_size_req=self.img_size,
        )
        action_name = self._action_name_by_sample.get(d.name, "unknown")
        action_label = int(self.action_to_id.get(action_name, self.action_to_id.get("unknown", 0)))
        params = torch.tensor([gt["E"], gt["nu"], gt["density"], gt["yield_stress"]], dtype=torch.float32)
        out = {
            "sample_id": d.name,
            "rgb": _np_to_torch_float32_share(arrays["rgb"]),
            "stress": _np_to_torch_float32_share(arrays["stress"]),
            "flow": _np_to_torch_float32_share(arrays["flow"]),
            "force_mask": _np_to_torch_float32_share(arrays["force_mask"]),
            "object_mask": _np_to_torch_float32_share(arrays["object_mask"]),
            "params": params,
            "params_dict": gt,
            "action_label": torch.tensor(action_label, dtype=torch.long),
        }
        if self.return_action_name:
            out["action_name"] = action_name
        return out


def collate_lmdb_gt_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将样本列表拼成 batch。
    若不同样本时间长度 T 不一致，会在时间维右侧复制最后一帧补齐到 batch 内最大 T。
    若 batch 内 T 全部相同（定长训练常见），直接 stack，跳过 max_t 扫描与 padding。
    """
    if not batch:
        raise ValueError("empty batch")

    out: Dict[str, Any] = {
        "sample_id": [x["sample_id"] for x in batch],
        "params_dict": [x["params_dict"] for x in batch],
        "params": torch.stack([x["params"] for x in batch], dim=0),  # [B,4]
    }
    if "action_label" in batch[0]:
        out["action_label"] = torch.stack([x["action_label"] for x in batch], dim=0)  # [B]
    if "action_name" in batch[0]:
        out["action_name"] = [str(x["action_name"]) for x in batch]

    t0 = int(batch[0]["rgb"].shape[2])
    uniform_t = all(int(x["rgb"].shape[2]) == t0 for x in batch)

    if uniform_t:
        out["num_frames_per_sample"] = [t0] * len(batch)
        for key in ("rgb", "stress", "flow", "force_mask", "object_mask"):
            out[key] = torch.stack([x[key] for x in batch], dim=0)
        return out

    max_t = 1
    for x in batch:
        max_t = max(max_t, int(x["rgb"].shape[2]))
    out["num_frames_per_sample"] = [int(x["rgb"].shape[2]) for x in batch]

    for key in ("rgb", "stress", "flow", "force_mask", "object_mask"):
        padded = []
        for x in batch:
            t = int(x[key].shape[2])
            if t == max_t:
                padded.append(x[key])
                continue
            pad_t = max_t - t
            last = x[key][:, :, -1:, :, :].repeat(1, 1, pad_t, 1, 1)
            padded.append(torch.cat([x[key], last], dim=2))
        out[key] = torch.stack(padded, dim=0)  # [B,V,3,T,H,W]
    return out


def save_sample_images(
    sample_modalities: Dict[str, np.ndarray],
    out_dir: str,
    *,
    max_views: int = 2,
    max_frames: int = 2,
) -> List[str]:
    """
    保存样本图片用于人工检查（不限制是否真 RGB）。
    输入各模态形状 [V,3,T,H,W]，值域 [0,1]。
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("保存预览图需要 opencv-python（cv2）") from e

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for mod, vcthw in sample_modalities.items():
        if not isinstance(vcthw, np.ndarray) or vcthw.ndim != 5 or int(vcthw.shape[1]) != 3:
            continue
        v_lim = min(int(max_views), int(vcthw.shape[0]))
        t_lim = min(int(max_frames), int(vcthw.shape[2]))
        for vi in range(v_lim):
            for ti in range(t_lim):
                chw = vcthw[vi, :, ti, :, :]
                rgb_u8 = np.clip(chw * 255.0, 0, 255).astype(np.uint8)
                hwc = np.transpose(rgb_u8, (1, 2, 0))
                bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
                p = root / f"{mod}_view{vi:02d}_frame{ti:03d}.png"
                cv2.imwrite(str(p), bgr)
                written.append(str(p))
    return written

