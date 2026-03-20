"""
物理参数预测模型：Image Encoder + Transformer + 物理参数头 + 动作头
"""
import math
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
        feat = feat.view(B, T, -1)
        return feat


class PhysPredictor(nn.Module):
    """
    时序图像 -> 物理参数 + 动作分类
    Image Encoder -> Transformer -> Physics Head + Action Head
    """

    def __init__(
        self,
        num_frames: int = 16,
        feat_dim: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 物理参数回归头 (E, nu, density, yield_stress)，E/density/yield_stress 需为正
        self.physics_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(NUM_FEATURES)),
        )

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

        params_raw = self.physics_head(pooled)
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
) -> PhysPredictor:
    return PhysPredictor(
        num_frames=num_frames,
        feat_dim=512,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
