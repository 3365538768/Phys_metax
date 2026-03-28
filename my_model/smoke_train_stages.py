# -*- coding: utf-8 -*-
"""
分阶段验证 Arch4 训练链路（推荐在 ``PhysGaussian`` 目录下执行）::

    python -m my_model.smoke_train_stages \\
        --split_root auto_output/dataset_deformation_stress_500_new/train

阶段说明：
  1) **LMDB**：首个样本 ``__meta__`` 与键完整性（不加载 Dataset）。
  2) **Dataset**：样本数、``__getitem__`` 张量形状（需 cv2 / lmdb / torch）。
  3) **前向+反传**：单 batch，检查 ``dict`` 输出与 loss。
  4) **短训**：少量 step（默认 3）优化步。

若未装依赖：``pip install torch lmdb opencv-python-headless``（或与项目 ``requirements.txt`` 一致）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_first_lmdb(split_root: Path, name: str = "arch4_data.lmdb") -> Optional[Path]:
    numeric_dirs = sorted(
        [p for p in split_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda x: int(x.name),
    )
    for d in numeric_dirs:
        lm = d / name
        if lm.is_dir() and (lm / "data.mdb").is_file():
            return lm
    return None


def stage1_lmdb_meta(lmdb_dir: Path) -> Tuple[Dict[str, Any], int]:
    import lmdb

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, max_readers=32)
    try:
        with env.begin() as txn:
            raw = txn.get(b"__meta__")
            if raw is None:
                raise RuntimeError("缺少 __meta__")
            meta = json.loads(raw.decode("utf-8"))
            views: List[str] = list(meta.get("views") or [])
            n_ok = 0
            for v in views:
                for mod in ("rgb", "stress", "flow", "force_mask"):
                    k = f"{v}/{mod}".encode("utf-8")
                    if txn.get(k) is None:
                        raise RuntimeError(f"缺少键 {k!r}")
                    n_ok += 1
            return meta, n_ok
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Arch4 分阶段训练冒烟测试")
    ap.add_argument(
        "--split_root",
        type=str,
        default="auto_output/dataset_deformation_stress_500_new/train",
        help="split 根目录（相对 PhysGaussian 或绝对路径）",
    )
    ap.add_argument(
        "--auto_output",
        type=str,
        default="auto_output",
        help="当 split_root 为相对路径时的父目录（相对 PhysGaussian）",
    )
    ap.add_argument("--lmdb_name", type=str, default="arch4_data.lmdb")
    ap.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="与模型一致；默认从 LMDB meta 的 len(views) 推断（上限仍受显存约束时可改小）",
    )
    ap.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="默认用 LMDB meta 的 num_frames",
    )
    ap.add_argument("--img_size", type=int, default=None, help="默认用 LMDB meta 的 img_size")
    ap.add_argument("--device", type=str, default=None, help="默认 cuda 若可用否则 cpu")
    ap.add_argument("--train_steps", type=int, default=3, help="阶段 4 优化步数")
    ap.add_argument("--dec_h", type=int, default=56)
    ap.add_argument("--dec_w", type=int, default=56)
    ap.add_argument("--skip_stage4", action="store_true", help="只做前 3 阶段")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    from .dataset import resolve_flat_dataset_root

    split = resolve_flat_dataset_root(args.split_root, here / Path(args.auto_output))
    if not split.is_dir():
        print(f"[FAIL] split_root 不存在: {split}", file=sys.stderr)
        sys.exit(1)

    lmdb_dir = _find_first_lmdb(split, args.lmdb_name)
    if lmdb_dir is None:
        print(f"[FAIL] 在 {split} 下未找到 */{args.lmdb_name}/data.mdb", file=sys.stderr)
        sys.exit(1)

    print("========== Stage 1: LMDB 元数据与键 ==========")
    try:
        meta, n_keys = stage1_lmdb_meta(lmdb_dir)
    except Exception as e:
        print(f"[FAIL] Stage 1: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  path: {lmdb_dir}")
    print(f"  views ({len(meta.get('views', []))}): {meta.get('views')}")
    print(f"  num_frames={meta.get('num_frames')} img_size={meta.get('img_size')}")
    print(f"  modality keys checked: {n_keys}")
    print("[OK] Stage 1")

    n_meta = int(meta.get("num_frames") or 0)
    h_meta = int(meta.get("img_size") or 224)
    num_frames = int(args.num_frames) if args.num_frames is not None else max(1, n_meta)
    img_size = int(args.img_size) if args.img_size is not None else h_meta
    n_views_meta = len(meta.get("views") or [])
    max_views = int(args.max_views) if args.max_views is not None else max(1, n_views_meta)

    if img_size != h_meta:
        print(
            f"[WARN] --img_size={img_size} 与 LMDB img_size={h_meta} 不一致，Dataset 可能报错",
            file=sys.stderr,
        )

    print("\n========== Stage 2: DatasetArch4 ==========")
    try:
        import torch
        from .dataset import DatasetArch4
    except ImportError as e:
        print(f"[FAIL] Stage 2 import: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ds = DatasetArch4(
            split,
            img_size=img_size,
            max_views=max_views,
            num_frames=num_frames,
            source="lmdb",
            lmdb_env_subdir=args.lmdb_name,
            preflight=True,
            verbose=True,
        )
    except Exception as e:
        print(f"[FAIL] Stage 2 Dataset: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  len(dataset)={len(ds)}")
    try:
        x, s, f, fm, p = ds[0]
    except Exception as e:
        print(f"[FAIL] Stage 2 __getitem__: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  x={tuple(x.shape)} stress={tuple(s.shape)} flow={tuple(f.shape)} force={tuple(fm.shape)} params={tuple(p.shape)}")
    print("[OK] Stage 2")

    print("\n========== Stage 3: 前向 + 反传（单 batch）==========")
    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    from .arch4_model import build_arch4_model
    from .losses import Arch4LossConfig, Arch4RegressionLoss, arch4_field_supervision_mse

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    x, stress_gt, flow_gt, force_gt, params_gt = next(iter(loader))
    x = x.to(device)
    stress_gt = stress_gt.to(device)
    flow_gt = flow_gt.to(device)
    force_gt = force_gt.to(device)
    params_gt = params_gt.to(device)

    model = build_arch4_model(
        num_views=max_views,
        in_channels=6,
        num_frames=num_frames,
        img_size=img_size,
        dec_h=args.dec_h,
        dec_w=args.dec_w,
        use_aux_field_heads=True,
        encoder_depth=4,
        encoder_embed_dim=192,
        encoder_num_heads=3,
        fusion_dim=256,
        fusion_heads=4,
        patch_size=32,
        tubelet_size=1,
    )
    model.to(device)
    model.train()

    def _log_params(t: torch.Tensor) -> torch.Tensor:
        e = torch.log1p(torch.clamp(t[:, 0], min=0))
        nu = t[:, 1]
        density = torch.log1p(torch.clamp(t[:, 2], min=0))
        ys = torch.log1p(torch.clamp(t[:, 3], min=0))
        return torch.stack([e, nu, density, ys], dim=1)

    out = model(x)
    assert isinstance(out, dict) and "param_pred" in out
    loss_fn = Arch4RegressionLoss(Arch4LossConfig())
    loss_p = loss_fn(_log_params(out["param_pred"]), _log_params(params_gt), out["logvar"])
    loss = (
        loss_p
        + 0.15 * arch4_field_supervision_mse(out["stress_field_pred"], stress_gt, args.dec_h, args.dec_w)
        + 0.15 * arch4_field_supervision_mse(out["flow_field_pred"], flow_gt, args.dec_h, args.dec_w)
        + 0.15 * arch4_field_supervision_mse(out["force_pred"], force_gt, args.dec_h, args.dec_w)
    )
    loss.backward()
    print(f"  device={device} loss={loss.item():.6f}")
    print("[OK] Stage 3")

    if args.skip_stage4:
        print("\n跳过 Stage 4（--skip_stage4）")
        return

    print(f"\n========== Stage 4: 短训 {args.train_steps} step(s) ==========")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = 0
    for x, stress_gt, flow_gt, force_gt, params_gt in itertools.cycle(loader):
        x = x.to(device)
        stress_gt = stress_gt.to(device)
        flow_gt = flow_gt.to(device)
        force_gt = force_gt.to(device)
        params_gt = params_gt.to(device)
        opt.zero_grad()
        out = model(x)
        loss_p = loss_fn(_log_params(out["param_pred"]), _log_params(params_gt), out["logvar"])
        loss = (
            loss_p
            + 0.15 * arch4_field_supervision_mse(out["stress_field_pred"], stress_gt, args.dec_h, args.dec_w)
            + 0.15 * arch4_field_supervision_mse(out["flow_field_pred"], flow_gt, args.dec_h, args.dec_w)
            + 0.15 * arch4_field_supervision_mse(out["force_pred"], force_gt, args.dec_h, args.dec_w)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        print(f"  step {step} loss={loss.item():.6f}")
        if step >= args.train_steps:
            break
    print("[OK] Stage 4")
    print("\n全部阶段通过。")


if __name__ == "__main__":
    main()
