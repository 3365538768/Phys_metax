from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from my_model.arch4_model import MultiViewFusion, ParamRegressionHead, VideoViTEncoder


class PhysicsBottleneck(nn.Module):
    """
    在跨视角融合后显式拆分三类物理潜变量：
    - contact_latent
    - deformation_latent
    - stress_latent
    """

    def __init__(self, in_dim: int, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
        )
        self.contact_head = nn.Linear(in_dim, latent_dim)
        self.deformation_head = nn.Linear(in_dim, latent_dim)
        self.stress_head = nn.Linear(in_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(z)
        return {
            "contact_latent": self.contact_head(h),
            "deformation_latent": self.deformation_head(h),
            "stress_latent": self.stress_head(h),
        }


class ActionClassificationHead(nn.Module):
    """从 bottleneck 拼接潜变量预测动作类别 logits。"""

    def __init__(self, in_dim: int, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(128, in_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, in_dim // 2), int(num_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalQueryFieldHead(nn.Module):
    """
    时间查询式场解码头：
    - 输入视角特征 [B,V,D]
    - 为每个 t 注入可学习 time embedding
    - 逐帧解码为 [B,V,C,H,W]，再拼成 [B,V,C,T,H,W]
    """

    def __init__(
        self,
        feat_dim: int,
        num_frames: int,
        dec_h: int,
        dec_w: int,
        out_channels: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.out_channels = int(out_channels)
        self.time_embed = nn.Embedding(self.num_frames, int(feat_dim))
        self.decoder = nn.Sequential(
            nn.Linear(int(feat_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.out_channels * self.dec_h * self.dec_w),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, v, d = feat.shape
        t_ids = torch.arange(self.num_frames, device=feat.device, dtype=torch.long)
        t_emb = self.time_embed(t_ids).view(1, 1, self.num_frames, d)  # [1,1,T,D]
        feat_btvd = feat.unsqueeze(2) + t_emb  # [B,V,T,D]
        y = self.decoder(feat_btvd.reshape(b * v * self.num_frames, d))
        y = y.view(b, v, self.num_frames, self.out_channels, self.dec_h, self.dec_w)
        return y.permute(0, 1, 3, 2, 4, 5).contiguous()


class LogicPhysModel(nn.Module):
    """
    最小可行版本：
    输入 `[B,V,C,T,H,W]`
    输出保持现有键，并新增 bottleneck/action 输出。
    """

    def __init__(
        self,
        *,
        num_views: int = 4,
        in_channels: int = 3,
        num_frames: int = 30,
        img_size: int = 224,
        num_targets: int = 4,
        num_actions: int = 4,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        tubelet_size: int = 1,
        patch_size: int = 32,
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        use_attention_pool: bool = True,
        use_uncertainty: bool = False,
        encoder_dropout: float = 0.05,
        dec_h: int = 56,
        dec_w: int = 56,
        use_aux_field_heads: bool = True,
        head_dropout: float = 0.1,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.in_channels = int(in_channels)
        self.num_targets = int(num_targets)
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.use_aux_field_heads = bool(use_aux_field_heads)
        self.use_uncertainty = bool(use_uncertainty)

        self.encoder = VideoViTEncoder(
            self.in_channels,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            tubelet_size,
            patch_size,
            encoder_dropout,
            img_size,
            self.num_frames,
        )
        self.fusion = MultiViewFusion(
            self.num_views,
            encoder_embed_dim,
            fusion_dim,
            fusion_heads,
            use_attention_pool,
            head_dropout,
        )

        self.physics_bottleneck = PhysicsBottleneck(
            in_dim=encoder_embed_dim,
            latent_dim=int(bottleneck_dim),
            dropout=head_dropout,
        )
        bottleneck_out_dim = int(bottleneck_dim) * 3

        self.param_head = ParamRegressionHead(
            bottleneck_out_dim,
            self.num_targets,
            head_dropout,
            self.use_uncertainty,
        )
        self.action_head = ActionClassificationHead(
            bottleneck_out_dim,
            int(num_actions),
            dropout=head_dropout,
        )

        # 将 bottleneck 全局表征注入每个视角，再走场解码头。
        self.view_condition_proj = nn.Linear(bottleneck_out_dim, encoder_embed_dim)
        # 显式视角身份编码：在场头解码前为每个 view 注入可学习区分信号。
        self.view_id_emb = nn.Embedding(self.num_views, encoder_embed_dim)
        self.view_id_scale = nn.Parameter(torch.tensor(1.0))
        # 场输出改为 RGB 三通道: [B,V,3,T,H,W]。
        # 使用时间查询式解码，避免单向量一次性硬回归整段时间序列。
        self.stress_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )
        self.flow_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )
        self.force_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(rgb.shape)}")
        b, v, c, t, h, w = rgb.shape
        if v != self.num_views:
            raise ValueError(f"V mismatch: expect {self.num_views}, got {v}")
        if c != self.in_channels:
            raise ValueError(f"C mismatch: expect {self.in_channels}, got {c}")

        feat_v = self.encoder(rgb.reshape(b * v, c, t, h, w)).view(b, v, -1)  # [B,V,D]
        z_fused = self.fusion(feat_v)  # [B,D]

        pb = self.physics_bottleneck(z_fused)
        z_phys = torch.cat(
            [pb["contact_latent"], pb["deformation_latent"], pb["stress_latent"]],
            dim=1,
        )  # [B,3*Db]

        param_pred, logvar_raw = self.param_head(z_phys)
        action_logits = self.action_head(z_phys)

        # [B,D] -> [B,V,D]
        view_bias = self.view_condition_proj(z_phys).unsqueeze(1).expand(-1, v, -1)
        view_ids = torch.arange(v, device=feat_v.device, dtype=torch.long)
        view_bias_id = self.view_id_emb(view_ids).unsqueeze(0).expand(b, -1, -1)
        feat_v_cond = feat_v + view_bias + self.view_id_scale * view_bias_id

        if self.use_aux_field_heads:
            stress_field_pred = self.stress_head(feat_v_cond)
            flow_field_pred = self.flow_head(feat_v_cond)
            force_pred = self.force_head(feat_v_cond)
        else:
            z0 = feat_v.new_zeros(b, v, 3, self.num_frames, self.dec_h, self.dec_w)
            stress_field_pred = z0
            flow_field_pred = z0
            force_pred = z0

        param_pred_raw = torch.stack(
            [
                torch.expm1(torch.clamp(param_pred[:, 0], min=0.0)),
                torch.clamp(param_pred[:, 1], min=0.0, max=0.5),
                torch.expm1(torch.clamp(param_pred[:, 2], min=0.0)),
                torch.expm1(torch.clamp(param_pred[:, 3], min=0.0)),
            ],
            dim=1,
        )
        logvar = logvar_raw if (self.use_uncertainty and logvar_raw is not None) else param_pred.new_zeros(b, self.num_targets)

        return {
            # 兼容旧输出
            "param_pred": param_pred,
            "param_pred_raw": param_pred_raw,
            "stress_field_pred": stress_field_pred,
            "flow_field_pred": flow_field_pred,
            "force_pred": force_pred,
            "logvar": logvar,
            # 新增输出
            "contact_latent": pb["contact_latent"],
            "deformation_latent": pb["deformation_latent"],
            "stress_latent": pb["stress_latent"],
            "action_logits": action_logits,
        }

