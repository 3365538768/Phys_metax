from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from my_model.arch4_model import MultiViewFusion, ParamRegressionHead
from logic_model.model import ActionClassificationHead, PhysicsBottleneck, TemporalQueryFieldHead


class DinoFrameEncoder(nn.Module):
    """
    2D 预训练骨干包装：
    - 输入: [N,C,H,W]
    - 输出: [N,D]
    优先使用 torch.hub 的 DINOv2；若失败可回退到 torchvision ViT。
    """

    def __init__(
        self,
        *,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        source: str = "torchhub",
        image_size: int = 224,
        out_dim: int = 384,
        torchhub_dir: str | None = None,
        torchhub_repo: str = "facebookresearch/dinov2:main",
        force_reload: bool = False,
        trust_repo: bool = True,
        skip_validation: bool = True,
        hub_verbose: bool = False,
        log_torchhub_dir: bool = False,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.pretrained = bool(pretrained)
        self.source = str(source).lower().strip()
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.torchhub_dir = (str(torchhub_dir).strip() if torchhub_dir is not None else "")
        self.torchhub_repo = str(torchhub_repo).strip() or "facebookresearch/dinov2:main"
        self.force_reload = bool(force_reload)
        self.trust_repo = bool(trust_repo)
        self.skip_validation = bool(skip_validation)
        self.hub_verbose = bool(hub_verbose)
        self.log_torchhub_dir = bool(log_torchhub_dir)

        self.backbone: nn.Module
        self.backbone_dim: int
        self._build_backbone()
        self.proj = (
            nn.Identity()
            if int(self.backbone_dim) == int(self.out_dim)
            else nn.Linear(int(self.backbone_dim), int(self.out_dim))
        )

        # DINOv2 采用 ImageNet 风格归一化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _build_backbone(self) -> None:
        name = self.backbone_name

        if self.source in ("torchhub", "hub"):
            hub_dir = self._resolve_torchhub_dir()
            hub_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(hub_dir.parent)
            torch.hub.set_dir(str(hub_dir))
            if self.log_torchhub_dir:
                print(f"[LogicPhysModel2] torchhub_dir={torch.hub.get_dir()}", flush=True)
                print(f"[LogicPhysModel2] torchhub_repo={self.torchhub_repo} backbone={name}", flush=True)
            bb = self._load_dinov2_from_hub(name)
            dim = int(getattr(bb, "embed_dim", getattr(bb, "num_features", self.out_dim)))
            self.backbone = bb
            self.backbone_dim = dim
            return

        if self.source in ("torchvision", "tv"):
            from torchvision.models import vit_b_16, vit_l_16

            if name in ("vit_b_16", "vitb16"):
                bb = vit_b_16(weights="DEFAULT" if self.pretrained else None)
            elif name in ("vit_l_16", "vitl16"):
                bb = vit_l_16(weights="DEFAULT" if self.pretrained else None)
            else:
                raise ValueError(f"unsupported torchvision backbone_name: {name}")
            dim = int(bb.hidden_dim)
            self.backbone = bb
            self.backbone_dim = dim
            return

        raise ValueError(f"unsupported backbone_source: {self.source}")

    def _resolve_torchhub_dir(self) -> Path:
        if self.torchhub_dir:
            p = Path(self.torchhub_dir).expanduser()
            if p.is_absolute():
                return p.resolve()
            return (Path.cwd() / p).resolve()
        env_home = str(os.environ.get("TORCH_HOME", "") or "").strip()
        if env_home:
            return (Path(env_home).expanduser().resolve() / "hub").resolve()
        # Phys/logic_model/model2.py -> Phys/.torch/hub
        return (Path(__file__).resolve().parent.parent / ".torch" / "hub").resolve()

    def _load_dinov2_from_hub(self, name: str) -> nn.Module:
        kwargs: Dict[str, object] = {
            "pretrained": bool(self.pretrained),
            "force_reload": bool(self.force_reload),
            "trust_repo": bool(self.trust_repo),
            "skip_validation": bool(self.skip_validation),
            "verbose": bool(self.hub_verbose),
        }
        drop_order = ["trust_repo", "skip_validation", "verbose"]
        for i in range(len(drop_order) + 1):
            try:
                return torch.hub.load(self.torchhub_repo, name, **kwargs)
            except TypeError:
                if i >= len(drop_order):
                    raise
                kwargs.pop(drop_order[i], None)
        raise RuntimeError("unreachable")

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone

        # DINOv2 / timm 常见接口
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    return feats["x_norm_clstoken"]
                if "x_prenorm" in feats and feats["x_prenorm"].dim() == 3:
                    return feats["x_prenorm"][:, 0]
            if torch.is_tensor(feats):
                if feats.dim() == 2:
                    return feats
                if feats.dim() == 3:
                    return feats[:, 0]
            raise RuntimeError("unsupported forward_features output format")

        # torchvision ViT 接口
        if hasattr(bb, "_process_input") and hasattr(bb, "encoder"):
            y = bb._process_input(x)
            n = y.shape[0]
            cls = bb.class_token.expand(n, -1, -1)
            y = torch.cat([cls, y], dim=1)
            y = bb.encoder(y)
            return y[:, 0]

        y = bb(x)
        if y.dim() == 2:
            return y
        if y.dim() == 3:
            return y[:, 0]
        raise RuntimeError("unsupported backbone output shape")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expect [N,C,H,W], got {tuple(x.shape)}")
        if int(x.shape[1]) != 3:
            raise ValueError(f"DINO encoder expects C=3, got C={int(x.shape[1])}")
        x = (x - self.mean.to(x)) / self.std.to(x)
        f = self._backbone_forward(x)
        return self.proj(f)


class TemporalAdapter(nn.Module):
    """将逐帧特征 [B,V,T,D] 聚合到 [B,V,D]。"""

    def __init__(
        self,
        *,
        dim: int,
        num_frames: int,
        adapter_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1,
        frame_pool: str = "mean",
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.adapter_type = str(adapter_type).lower().strip()
        self.frame_pool = str(frame_pool).lower().strip()

        self.frame_pos = nn.Parameter(torch.zeros(1, self.num_frames, self.dim))
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        if self.frame_pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        if self.adapter_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=int(num_heads),
                dim_feedforward=int(4 * self.dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
            self.norm = nn.LayerNorm(self.dim)
        elif self.adapter_type == "mean":
            self.encoder = None
            self.norm = nn.LayerNorm(self.dim)
        else:
            raise ValueError(f"unknown temporal_adapter_type: {self.adapter_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expect [B,V,T,D], got {tuple(x.shape)}")
        b, v, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"D mismatch: expect {self.dim}, got {d}")
        n = b * v
        y = x.view(n, t, d)
        pos = self.frame_pos[:, :t, :]
        y = y + pos

        if self.adapter_type == "transformer":
            if self.frame_pool == "cls":
                if self.cls_token is None:
                    raise RuntimeError("frame_pool='cls' but cls_token is not initialized")
                cls = self.cls_token.expand(n, -1, -1)
                y = torch.cat([cls, y], dim=1)
                y = self.encoder(y)
                z = y[:, 0, :]
            else:
                y = self.encoder(y)
                z = y.mean(dim=1)
        else:
            z = y.mean(dim=1)

        z = self.norm(z)
        return z.view(b, v, d).contiguous()


class LogicPhysModel2(nn.Module):
    """
    DINOv2 版本逻辑模型，保持与 LogicPhysModel 相同核心接口:
    - 输入: [B,V,C,T,H,W]
    - 返回键: param_pred/param_pred_raw/stress_field_pred/flow_field_pred/force_pred/logvar/action_logits
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
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        use_attention_pool: bool = True,
        use_uncertainty: bool = False,
        dec_h: int = 56,
        dec_w: int = 56,
        use_aux_field_heads: bool = True,
        head_dropout: float = 0.1,
        bottleneck_dim: int = 128,
        dino_backbone_name: str = "dinov2_vits14",
        dino_backbone_pretrained: bool = True,
        dino_backbone_source: str = "torchhub",
        dino_out_dim: int = 384,
        temporal_adapter_type: str = "transformer",
        temporal_adapter_layers: int = 2,
        temporal_adapter_heads: int = 6,
        temporal_adapter_dropout: float = 0.1,
        frame_pool: str = "mean",
        freeze_backbone: bool = True,
        torchhub_dir: str | None = None,
        dino_torchhub_repo: str = "facebookresearch/dinov2:main",
        dino_force_reload: bool = False,
        dino_trust_repo: bool = True,
        dino_skip_validation: bool = True,
        dino_hub_verbose: bool = False,
        dino_log_torchhub_dir: bool = False,
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
        self.freeze_backbone = bool(freeze_backbone)

        enc_dim = int(dino_out_dim)
        self.frame_encoder = DinoFrameEncoder(
            backbone_name=dino_backbone_name,
            pretrained=bool(dino_backbone_pretrained),
            source=dino_backbone_source,
            image_size=int(img_size),
            out_dim=enc_dim,
            torchhub_dir=torchhub_dir,
            torchhub_repo=dino_torchhub_repo,
            force_reload=bool(dino_force_reload),
            trust_repo=bool(dino_trust_repo),
            skip_validation=bool(dino_skip_validation),
            hub_verbose=bool(dino_hub_verbose),
            log_torchhub_dir=bool(dino_log_torchhub_dir),
        )
        self.temporal_adapter = TemporalAdapter(
            dim=enc_dim,
            num_frames=self.num_frames,
            adapter_type=temporal_adapter_type,
            num_layers=int(temporal_adapter_layers),
            num_heads=int(temporal_adapter_heads),
            dropout=float(temporal_adapter_dropout),
            frame_pool=frame_pool,
        )

        if self.freeze_backbone:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False
            self.frame_encoder.eval()

        self.fusion = MultiViewFusion(
            self.num_views,
            enc_dim,
            int(fusion_dim),
            int(fusion_heads),
            bool(use_attention_pool),
            float(head_dropout),
        )

        self.physics_bottleneck = PhysicsBottleneck(
            in_dim=enc_dim,
            latent_dim=int(bottleneck_dim),
            dropout=float(head_dropout),
        )
        bottleneck_out_dim = int(bottleneck_dim) * 3

        self.param_head = ParamRegressionHead(
            bottleneck_out_dim,
            self.num_targets,
            float(head_dropout),
            self.use_uncertainty,
        )
        self.action_head = ActionClassificationHead(
            bottleneck_out_dim,
            int(num_actions),
            dropout=float(head_dropout),
        )

        self.view_condition_proj = nn.Linear(bottleneck_out_dim, enc_dim)
        self.view_id_emb = nn.Embedding(self.num_views, enc_dim)
        self.view_id_scale = nn.Parameter(torch.tensor(1.0))

        self.stress_head = TemporalQueryFieldHead(
            enc_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=float(head_dropout),
        )
        self.flow_head = TemporalQueryFieldHead(
            enc_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=float(head_dropout),
        )
        self.force_head = TemporalQueryFieldHead(
            enc_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=float(head_dropout),
        )

    def _to_three_channels(self, x: torch.Tensor) -> torch.Tensor:
        # DINO 系列骨干默认输入 3 通道
        c = int(x.shape[1])
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c > 3:
            return x[:, :3, :, :]
        raise ValueError(f"unsupported input channels for DINO backbone: C={c}")

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(rgb.shape)}")
        b, v, c, t, h, w = rgb.shape
        if v != self.num_views:
            raise ValueError(f"V mismatch: expect {self.num_views}, got {v}")
        if c != self.in_channels:
            raise ValueError(f"C mismatch: expect {self.in_channels}, got {c}")

        # [B,V,C,T,H,W] -> [B*V*T,C,H,W]
        x = rgb.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        x = self._to_three_channels(x)

        if self.freeze_backbone:
            with torch.no_grad():
                frame_feat = self.frame_encoder(x)
        else:
            frame_feat = self.frame_encoder(x)

        frame_feat = frame_feat.view(b, v, t, -1)
        feat_v = self.temporal_adapter(frame_feat)  # [B,V,D]

        z_fused = self.fusion(feat_v)  # [B,D]
        pb = self.physics_bottleneck(z_fused)
        z_phys = torch.cat(
            [pb["contact_latent"], pb["deformation_latent"], pb["stress_latent"]],
            dim=1,
        )

        param_pred, logvar_raw = self.param_head(z_phys)
        action_logits = self.action_head(z_phys)

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
            "param_pred": param_pred,
            "param_pred_raw": param_pred_raw,
            "stress_field_pred": stress_field_pred,
            "flow_field_pred": flow_field_pred,
            "force_pred": force_pred,
            "logvar": logvar,
            "contact_latent": pb["contact_latent"],
            "deformation_latent": pb["deformation_latent"],
            "stress_latent": pb["stress_latent"],
            "action_logits": action_logits,
        }

