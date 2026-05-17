import math
from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ShadowImageEncoder(nn.Module):
    """
    输入单帧阴影图: [B, 1, H, W]
    输出图像特征: [B, feat_dim]
    """
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.layer4 = ConvBlock(256, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.stem(x)    # [B, 32, H/2, W/2]
        x = self.layer1(x)  # [B, 64, H/4, W/4]
        x = self.layer2(x)  # [B, 128, H/8, W/8]
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        x = self.layer4(x)  # [B, 256, H/32, W/32]
        x = self.pool(x).flatten(1)   # [B, 256]
        x = self.fc(x)                # [B, feat_dim]
        return x


class LightEncoder(nn.Module):
    """
    输入光照方向 [B, 3]
    输出光照特征 [B, light_feat_dim]
    """
    def __init__(self, light_feat_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, light_feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, light_dir):
        return self.mlp(light_dir)


class PointCloudDecoder(nn.Module):
    """
    输入全局特征 [B, global_dim]
    输出粗点云 P0 [B, num_points, 3]
    """
    def __init__(self, global_dim: int = 256, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_points * 3),
        )

    def forward(self, z):
        x = self.mlp(z)                         # [B, num_points * 3]
        x = torch.tanh(x)                       # 粗点云限制到 [-1, 1]
        x = x.view(z.shape[0], self.num_points, 3)
        return x                                # [B, num_points, 3]


class SimpleTransformerRefiner(nn.Module):
    """
    最简单的 Transformer Refiner：纯几何细化，不注入 global_feat。

    coarse_points 已经编码了 10 帧阴影信息（通过 decoder 从 global_feat 解出来），
    所以 refiner 只需要让每个点感知其他点的位置，预测自己的位移。

    结构:
        coarse_points [B, N, 3]
            → Linear(3 → hidden_dim)
            → TransformerEncoderLayer × num_blocks  （全局自注意力）
            → Linear(hidden_dim → 3)
            → tanh × delta_scale
            → coarse_points + delta
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        delta_scale: float = 0.05,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.delta_scale = delta_scale

        # 坐标嵌入：3D 坐标 → 高维特征
        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 标准 Transformer Encoder（batch_first=True 让输入是 [B, N, C]）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        # 位移头：特征 → 3D 位移
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coarse_points: torch.Tensor) -> torch.Tensor:
        """
        coarse_points: [B, N, 3]
        return:        [B, N, 3]
        """
        feat = self.coord_embed(coarse_points)        # [B, N, hidden_dim]
        feat = self.transformer(feat)                 # [B, N, hidden_dim]
        delta = self.delta_head(feat)                 # [B, N, 3]
        delta = self.delta_scale * torch.tanh(delta)  # 限幅
        return coarse_points + delta                  # 残差

class ShadowPointBaseline(nn.Module):
    def __init__(
            self,
            image_feat_dim: int = 256,
            light_feat_dim: int = 128,
            fused_dim: int = 256,
            num_points: int = 2048,
            num_frames: int = 10,
            use_refiner: bool = True,  # 改名：use_pct_refiner → use_refiner
            refiner_hidden_dim: int = 128,
            refiner_blocks: int = 2,
            refiner_num_heads: int = 4,
            refiner_delta_scale: float = 0.05,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.fused_dim = fused_dim
        self.image_encoder = ShadowImageEncoder(feat_dim=image_feat_dim)
        self.light_encoder = LightEncoder(light_feat_dim=light_feat_dim)
        self.global_dim = fused_dim * num_frames

        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + light_feat_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = PointCloudDecoder(global_dim=self.global_dim, num_points=num_points)

        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = SimpleTransformerRefiner(
                hidden_dim=refiner_hidden_dim,
                num_blocks=refiner_blocks,
                num_heads=refiner_num_heads,
                delta_scale=refiner_delta_scale,
            )
        else:
            self.refiner = None

    def encode_global_feature(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = shadow_seq.shape

        shadow_seq = shadow_seq.view(b * k, c, h, w)              # [B*K, 1, H, W]
        light_dir = light_dir.view(b * k, 3)                      # [B*K, 3]

        img_feat = self.image_encoder(shadow_seq)                 # [B*K, image_feat_dim]
        light_feat = self.light_encoder(light_dir)                # [B*K, light_feat_dim]

        fused = torch.cat([img_feat, light_feat], dim=-1)         # [B*K, image+light]
        fused = self.fusion(fused)                                # [B*K, fused_dim]
        fused = fused.view(b, k, -1)                              # [B, K, fused_dim]

        global_feat = fused.reshape(b, k * fused.shape[-1])  # [B, K * fused_dim]
        return global_feat

    def forward(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        global_feat = self.encode_global_feature(shadow_seq, light_dir)
        coarse_points = self.decoder(global_feat)

        if self.refiner is None:
            return coarse_points

        refined_points = self.refiner(coarse_points)  # 不再传 global_feat
        return refined_points
