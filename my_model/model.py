"""
物理参数预测模型：Image Encoder + Transformer + 物理参数头 + 动作头
"""
import inspect
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .dataset import ACTION_NAMES, MATERIAL_CATEGORIES, NUM_FEATURES


class ImageEncoder(nn.Module):
    """ResNet18 作为图像编码器，提取每帧特征"""

    def __init__(self, pretrained: bool = True, feat_dim: int = 512):
        super().__init__()
        try:
            resnet = models.resnet18(weights="DEFAULT" if pretrained else None)
        except TypeError:
            resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        out: (B, T, feat_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)
        # 保证 contiguous，减轻 DDP 下部分 Conv 梯度 stride 与 bucket 不一致的警告
        feat = feat.contiguous().view(B, T, -1)
        return feat


class PhysPredictor(nn.Module):
    """
    时序图像 -> 物理参数 + 动作分类
    Image Encoder -> Transformer -> Physics Head + Action Head

    physics_head_mode:
      - "single": 一个 MLP 输出 4 维（架构 1）
      - "multi": 四个独立小 MLP 各输出 1 维（架构 2）
    """

    def __init__(
        self,
        num_frames: int = 16,
        feat_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        physics_head_mode: str = "single",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.physics_head_mode = physics_head_mode

        self.image_encoder = ImageEncoder(pretrained=True, feat_dim=feat_dim)
        self.feat_proj = nn.Linear(feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True 时 PyTorch 不会用 nested tensor，仍会报警告；显式关闭可消音
        _te_sig = inspect.signature(nn.TransformerEncoder.__init__)
        _te_kw = {}
        if "enable_nested_tensor" in _te_sig.parameters:
            _te_kw["enable_nested_tensor"] = False
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, **_te_kw
        )

        # 物理参数回归头 (E, nu, density, yield_stress)，E/density/yield_stress 需为正
        if physics_head_mode == "single":
            self.physics_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, len(NUM_FEATURES)),
            )
            self.physics_heads = None
        elif physics_head_mode == "multi":
            self.physics_head = None
            hid = max(64, d_model // 2)
            self.physics_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, hid),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hid, 1),
                    )
                    for _ in range(len(NUM_FEATURES))
                ]
            )
        else:
            raise ValueError(f"未知 physics_head_mode: {physics_head_mode}")

        # 动作分类头
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(ACTION_NAMES)),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        frames: (B, T, C, H, W)
        Returns:
            params_pred: (B, 4) - E, nu, density, yield_stress
            action_logits: (B, 5)
        """
        # Image encoder
        feat = self.image_encoder(frames)  # (B, T, feat_dim)
        feat = self.feat_proj(feat)  # (B, T, d_model)

        # Transformer
        seq = self.transformer(feat)  # (B, T, d_model)

        # 取最后一帧或 mean pooling 作为全局表示
        pooled = seq[:, -1, :]  # (B, d_model)

        if self.physics_head_mode == "single":
            params_raw = self.physics_head(pooled)
        else:
            parts = [h(pooled).squeeze(-1) for h in self.physics_heads]
            params_raw = torch.stack(parts, dim=1)
        # E, nu, density, yield_stress: 确保 E/density/yield_stress 非负，nu 在 [0,1]
        # 注意：避免对 tensor view 做 inplace slice 赋值（会触发 autograd 的 inplace 版本冲突）
        e = F.softplus(params_raw[:, 0])
        nu = torch.sigmoid(params_raw[:, 1])
        density = F.softplus(params_raw[:, 2])
        yield_stress = F.softplus(params_raw[:, 3])
        params_pred = torch.stack([e, nu, density, yield_stress], dim=1)

        action_logits = self.action_head(pooled)

        return params_pred, action_logits


def create_model(
    num_frames: int = 16,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    arch: int = 1,
) -> PhysPredictor:
    """
    arch=1: 单一物理参数头（shared MLP -> 4 维）
    arch=2: 多物理头（每个标量独立 MLP）
    """
    mode = "single" if int(arch) == 1 else "multi"
    return PhysPredictor(
        num_frames=num_frames,
        feat_dim=512,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        physics_head_mode=mode,
    )
