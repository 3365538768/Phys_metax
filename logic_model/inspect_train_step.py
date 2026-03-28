from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch, save_sample_images
from logic_model.losses import (
    Arch4LossConfig,
    Arch4RegressionLoss,
    PhysicsConsistencyConfig,
    action_classification_loss,
    arch4_field_supervision_mse,
    compute_physics_consistency_losses,
)
from logic_model.model import LogicPhysModel


def _tensor_stats(x: torch.Tensor, keep_values: int = 8) -> Dict[str, Any]:
    y = x.detach().float().cpu()
    flat = y.reshape(-1)
    n = int(min(int(keep_values), int(flat.numel())))
    return {
        "shape": list(y.shape),
        "dtype": str(x.dtype),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "values_head": flat[:n].tolist(),
    }


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _resample_time_bvcthw(x: torch.Tensor, target_t: int) -> torch.Tensor:
    t0 = int(x.shape[3])
    if t0 == int(target_t):
        return x
    b, v, c, _, h, w = x.shape
    y = x.permute(0, 1, 2, 4, 5, 3).contiguous().view(b * v * c * h * w, 1, t0)
    y = torch.nn.functional.interpolate(y, size=int(target_t), mode="linear", align_corners=False)
    y = y.view(b, v, c, h, w, int(target_t)).permute(0, 1, 2, 5, 3, 4).contiguous()
    return y


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is None:
            continue
        has_grad = True
        g = p.grad.detach().float()
        total += float((g * g).sum().item())
    return float(total ** 0.5) if has_grad else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run one train-step and dump detailed JSON inspect")
    ap.add_argument("--split_root", type=str, default="auto_output/dataset_deformation_stress_500_new/train")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_views", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--dec_h", type=int, default=112)
    ap.add_argument("--dec_w", type=int, default=112)
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_stress", type=float, default=0.15)
    ap.add_argument("--lambda_flow", type=float, default=0.15)
    ap.add_argument("--lambda_force", type=float, default=0.15)
    ap.add_argument("--lambda_action", type=float, default=1.0)
    ap.add_argument("--lambda_phys", type=float, default=0.1)
    ap.add_argument("--lambda_stress_flow_consistency", type=float, default=1.0)
    ap.add_argument("--lambda_force_stress_consistency", type=float, default=1.0)
    ap.add_argument("--lambda_force_flow_consistency", type=float, default=1.0)
    ap.add_argument("--use_pred_force_mask_for_phys", action="store_true")
    ap.add_argument("--save_json", type=str, default="logic_model/train_step_inspect.json")
    ap.add_argument("--preview_dir", type=str, default="logic_model/train_step_preview")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ds = LmdbGtDataset(
        split_root=args.split_root,
        max_views=int(args.max_views),
        num_frames=(None if int(args.num_frames) <= 0 else int(args.num_frames)),
        img_size=(None if int(args.img_size) <= 0 else int(args.img_size)),
        return_action_name=True,
    )
    si = max(0, min(int(args.sample_index), len(ds) - 1))
    subset = Subset(ds, [si])
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_lmdb_gt_batch,
    )
    batch = next(iter(loader))

    device_str = str(args.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    rgb0 = batch["rgb"]
    model_num_frames = int(rgb0.shape[3]) if int(args.num_frames) <= 0 else int(args.num_frames)

    model = LogicPhysModel(
        num_views=int(args.max_views),
        in_channels=int(args.in_channels),
        num_frames=int(model_num_frames),
        img_size=(224 if int(args.img_size) <= 0 else int(args.img_size)),
        num_targets=4,
        num_actions=ds.num_actions,
        dec_h=int(args.dec_h),
        dec_w=int(args.dec_w),
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_reg = Arch4RegressionLoss(Arch4LossConfig())
    phys_cfg = PhysicsConsistencyConfig(
        lambda_stress_flow_consistency=float(args.lambda_stress_flow_consistency),
        lambda_force_stress_consistency=float(args.lambda_force_stress_consistency),
        lambda_force_flow_consistency=float(args.lambda_force_flow_consistency),
        use_pred_force_mask=bool(args.use_pred_force_mask_for_phys),
    )

    rgb = batch["rgb"].to(device)
    stress_gt = batch["stress"].to(device)
    flow_gt = batch["flow"].to(device)
    force_gt = batch["force_mask"].to(device)
    object_gt = batch["object_mask"].to(device)
    params_gt_raw = batch["params"].to(device)
    action_label = batch["action_label"].to(device)

    x = rgb[:, :, : int(args.in_channels), :, :, :] if int(args.in_channels) > 1 else rgb[:, :, :1, :, :, :]
    if int(x.shape[3]) != int(model.num_frames):
        x = _resample_time_bvcthw(x, int(model.num_frames))
        stress_gt = _resample_time_bvcthw(stress_gt, int(model.num_frames))
        flow_gt = _resample_time_bvcthw(flow_gt, int(model.num_frames))
        force_gt = _resample_time_bvcthw(force_gt, int(model.num_frames))
        object_gt = _resample_time_bvcthw(object_gt, int(model.num_frames))

    out = model(x)
    gt_train_space = _to_target_params(params_gt_raw)
    valid_mask = torch.ones_like(gt_train_space)
    valid_mask[:, 3] = (params_gt_raw[:, 3] > 0).to(valid_mask.dtype)

    loss_reg_part = loss_reg(out["param_pred"], gt_train_space, out["logvar"], valid_mask=valid_mask)
    loss_stress_part = arch4_field_supervision_mse(out["stress_field_pred"], stress_gt, int(args.dec_h), int(args.dec_w))
    loss_flow_part = arch4_field_supervision_mse(out["flow_field_pred"], flow_gt, int(args.dec_h), int(args.dec_w))
    loss_force_part = arch4_field_supervision_mse(out["force_pred"], force_gt, int(args.dec_h), int(args.dec_w))
    action_ret = action_classification_loss(out["action_logits"], action_label)
    phys_ret = compute_physics_consistency_losses(
        stress_pred=out["stress_field_pred"],
        flow_pred=out["flow_field_pred"],
        force_pred=out["force_pred"],
        force_gt=force_gt,
        cfg=phys_cfg,
    )

    loss_total = (
        loss_reg_part
        + float(args.lambda_stress) * loss_stress_part
        + float(args.lambda_flow) * loss_flow_part
        + float(args.lambda_force) * loss_force_part
        + float(args.lambda_action) * action_ret["loss_action"]
        + float(args.lambda_phys) * phys_ret["loss_phys_total"]
    )

    opt.zero_grad()
    loss_total.backward()
    grad_norm_before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    opt.step()

    # 导出第一个样本预览图，便于同时对照输入/标签
    preview_modalities = {
        "rgb": rgb[0].detach().cpu().numpy(),
        "stress_gt": stress_gt[0].detach().cpu().numpy(),
        "flow_gt": flow_gt[0].detach().cpu().numpy(),
        "force_gt": force_gt[0].detach().cpu().numpy(),
        "object_gt": object_gt[0].detach().cpu().numpy(),
    }
    preview_files = save_sample_images(preview_modalities, args.preview_dir, max_views=2, max_frames=2)

    report: Dict[str, Any] = {
        "meta": {
            "split_root": str(args.split_root),
            "sample_index": int(si),
            "sample_id": batch["sample_id"],
            "action_name": batch.get("action_name", []),
            "action_label": action_label.detach().cpu().tolist(),
            "action_to_id": ds.action_to_id,
            "device": str(device),
            "model_num_frames": int(model.num_frames),
        },
        "input": {
            "x": _tensor_stats(x),
            "stress_gt": _tensor_stats(stress_gt),
            "flow_gt": _tensor_stats(flow_gt),
            "force_gt": _tensor_stats(force_gt),
            "object_gt": _tensor_stats(object_gt),
            "params_gt_raw": _tensor_stats(params_gt_raw),
            "params_gt_train_space": _tensor_stats(gt_train_space),
        },
        "bottleneck": {
            "contact_latent": _tensor_stats(out["contact_latent"]),
            "deformation_latent": _tensor_stats(out["deformation_latent"]),
            "stress_latent": _tensor_stats(out["stress_latent"]),
        },
        "predictions": {
            "param_pred": _tensor_stats(out["param_pred"]),
            "param_pred_raw": _tensor_stats(out["param_pred_raw"]),
            "action_logits": _tensor_stats(out["action_logits"]),
            "action_prob": torch.softmax(out["action_logits"].detach().cpu(), dim=1).tolist(),
            "action_pred": out["action_logits"].argmax(dim=1).detach().cpu().tolist(),
            "stress_field_pred": _tensor_stats(out["stress_field_pred"]),
            "flow_field_pred": _tensor_stats(out["flow_field_pred"]),
            "force_pred": _tensor_stats(out["force_pred"]),
            "logvar": _tensor_stats(out["logvar"]),
        },
        "loss": {
            "loss_total": float(loss_total.item()),
            "loss_reg": float(loss_reg_part.item()),
            "loss_stress": float(loss_stress_part.item()),
            "loss_flow": float(loss_flow_part.item()),
            "loss_force": float(loss_force_part.item()),
            "loss_action": float(action_ret["loss_action"].item()),
            "action_acc": float(action_ret["action_acc"].item()),
            "loss_phys_total": float(phys_ret["loss_phys_total"].item()),
            "loss_stress_flow_consistency": float(phys_ret["loss_stress_flow_consistency"].item()),
            "loss_force_stress_consistency": float(phys_ret["loss_force_stress_consistency"].item()),
            "loss_force_flow_consistency": float(phys_ret["loss_force_flow_consistency"].item()),
        },
        "grad_norm": {
            "global_before_clip": grad_norm_before_clip,
            "encoder": _module_grad_norm(model.encoder),
            "fusion": _module_grad_norm(model.fusion),
            "physics_bottleneck": _module_grad_norm(model.physics_bottleneck),
            "param_head": _module_grad_norm(model.param_head),
            "action_head": _module_grad_norm(model.action_head),
            "stress_head": _module_grad_norm(model.stress_head),
            "flow_head": _module_grad_norm(model.flow_head),
            "force_head": _module_grad_norm(model.force_head),
        },
        "preview": {
            "dir": str(Path(args.preview_dir).resolve()),
            "files": preview_files,
        },
    }

    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved_json": str(out_path), "sample_id": batch["sample_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

