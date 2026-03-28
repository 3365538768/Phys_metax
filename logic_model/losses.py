from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from my_model.losses import Arch4LossConfig, Arch4RegressionLoss, arch4_field_supervision_mse


def _as_bvthw(x: torch.Tensor) -> torch.Tensor:
    """
    支持输入:
    - [B,V,1,T,H,W]
    - [B,V,C,T,H,W]
    统一返回 [B,V,T,H,W]（通道取绝对值均值）。
    """
    if x.dim() != 6:
        raise ValueError(f"expect 6D tensor [B,V,C,T,H,W], got {tuple(x.shape)}")
    return x.abs().mean(dim=2)


def _safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3, 4)
    mu = x.mean(dim=dims, keepdim=True)
    var = (x - mu).pow(2).mean(dim=dims, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def _align_bvthw(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    将 src [B,V,T,H,W] 对齐到 ref 的 [T,H,W]（时间线性、空间双线性）。
    """
    if src.shape == ref.shape:
        return src
    b, v, ts, hs, ws = src.shape
    tr, hr, wr = int(ref.shape[2]), int(ref.shape[3]), int(ref.shape[4])
    x = src.view(b * v, 1, ts, hs, ws)
    # 3D 插值: (D,T)=(T,H,W)
    x = F.interpolate(x, size=(tr, hr, wr), mode="trilinear", align_corners=False)
    return x.view(b, v, tr, hr, wr)


def _soft_weight_from_force(force: torch.Tensor, sharpness: float = 8.0) -> torch.Tensor:
    # force: [B,V,T,H,W], normalize to [0,1] then sigmoid sharpen
    fmin = force.amin(dim=(1, 2, 3, 4), keepdim=True)
    fmax = force.amax(dim=(1, 2, 3, 4), keepdim=True)
    fn = (force - fmin) / (fmax - fmin + 1e-6)
    return torch.sigmoid((fn - 0.5) * float(sharpness))


def _masked_contrast_loss(value: torch.Tensor, weight: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """
    软掩码区域对比：
    区域内均值应高于区域外均值至少 margin。
    """
    w_in = weight.clamp(0.0, 1.0)
    w_out = (1.0 - w_in).clamp(0.0, 1.0)
    in_mean = (w_in * value).sum(dim=(1, 2, 3, 4)) / (w_in.sum(dim=(1, 2, 3, 4)) + 1e-6)
    out_mean = (w_out * value).sum(dim=(1, 2, 3, 4)) / (w_out.sum(dim=(1, 2, 3, 4)) + 1e-6)
    # 希望 in_mean - out_mean >= margin
    return F.softplus(float(margin) - (in_mean - out_mean)).mean()


def loss_stress_flow_consistency(
    stress_bvcthw: torch.Tensor,
    flow_bvcthw: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    高 stress 区域倾向于高 flow：
    使用归一化后正相关约束（softplus-margin on correlation）。
    """
    s = _safe_zscore(_as_bvthw(stress_bvcthw))
    f = _safe_zscore(_as_bvthw(flow_bvcthw))
    f = _align_bvthw(f, s)
    corr = (s * f).mean(dim=(1, 2, 3, 4))
    return F.softplus(float(margin) - corr).mean()


def loss_force_stress_consistency(
    force_bvcthw: torch.Tensor,
    stress_bvcthw: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    力/接触邻域 stress 更高：mask 内外软对比。
    """
    force = _as_bvthw(force_bvcthw)
    stress = _as_bvthw(stress_bvcthw)
    force = _align_bvthw(force, stress)
    w = _soft_weight_from_force(force)
    return _masked_contrast_loss(stress, w, margin=float(margin))


def loss_force_flow_consistency(
    force_bvcthw: torch.Tensor,
    flow_bvcthw: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    力/接触邻域 flow 更高：mask 内外软对比。
    """
    force = _as_bvthw(force_bvcthw)
    flow = _as_bvthw(flow_bvcthw)
    force = _align_bvthw(force, flow)
    w = _soft_weight_from_force(force)
    return _masked_contrast_loss(flow, w, margin=float(margin))


@dataclass
class PhysicsConsistencyConfig:
    lambda_stress_flow_consistency: float = 1.0
    lambda_force_stress_consistency: float = 1.0
    lambda_force_flow_consistency: float = 1.0
    margin_stress_flow: float = 0.1
    margin_force_stress: float = 0.05
    margin_force_flow: float = 0.05
    use_pred_force_mask: bool = False


def compute_physics_consistency_losses(
    *,
    stress_pred: torch.Tensor,
    flow_pred: torch.Tensor,
    force_pred: torch.Tensor,
    force_gt: Optional[torch.Tensor],
    cfg: PhysicsConsistencyConfig,
) -> Dict[str, torch.Tensor]:
    force_ref = force_pred if cfg.use_pred_force_mask or (force_gt is None) else force_gt

    l_sf = loss_stress_flow_consistency(stress_pred, flow_pred, margin=cfg.margin_stress_flow)
    l_fs = loss_force_stress_consistency(force_ref, stress_pred, margin=cfg.margin_force_stress)
    l_ff = loss_force_flow_consistency(force_ref, flow_pred, margin=cfg.margin_force_flow)
    l_total = (
        float(cfg.lambda_stress_flow_consistency) * l_sf
        + float(cfg.lambda_force_stress_consistency) * l_fs
        + float(cfg.lambda_force_flow_consistency) * l_ff
    )
    return {
        "loss_stress_flow_consistency": l_sf,
        "loss_force_stress_consistency": l_fs,
        "loss_force_flow_consistency": l_ff,
        "loss_phys_total": l_total,
    }


def action_classification_loss(action_logits: torch.Tensor, action_label: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss = F.cross_entropy(action_logits, action_label.long())
    pred = action_logits.argmax(dim=1)
    acc = (pred == action_label.long()).float().mean()
    return {"loss_action": loss, "action_acc": acc}


__all__ = [
    "Arch4LossConfig",
    "Arch4RegressionLoss",
    "arch4_field_supervision_mse",
    "PhysicsConsistencyConfig",
    "compute_physics_consistency_losses",
    "action_classification_loss",
]

