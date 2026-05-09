from typing import Tuple

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
        x = self.stem(x)    # [B, 32, ...]
        x = self.layer1(x)  # [B, 64, ...]
        x = self.layer2(x)  # [B, 128, ...]
        x = self.layer3(x)  # [B, 256, ...]
        x = self.layer4(x)  # [B, 256, ...]
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
    输出点云 [B, num_points, 3]
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
        x = self.mlp(z)  # [B, num_points * 3]
        x = torch.tanh(x)  # 限制到 [-1, 1]
        x = x.view(z.shape[0], self.num_points, 3)
        return x


class ShadowPointBaseline(nn.Module):
    """
    第一版 baseline:
        每帧 shadow -> CNN feature
        每帧 light_dir -> MLP feature
        concat 后做融合
        对 K 帧做 mean pooling
        解码为点云

    输入:
        shadow_seq: [B, K, 1, H, W]
        light_dir:  [B, K, 3]
    输出:
        points:     [B, N, 3]
    """
    def __init__(
        self,
        image_feat_dim: int = 256,
        light_feat_dim: int = 128,
        fused_dim: int = 256,
        num_points: int = 2048,
        num_frames: int = 10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.fused_dim = fused_dim

        self.image_encoder = ShadowImageEncoder(feat_dim=image_feat_dim)
        self.light_encoder = LightEncoder(light_feat_dim=light_feat_dim)

        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + light_feat_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = PointCloudDecoder(global_dim=fused_dim * num_frames, num_points=num_points)

    def forward(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        """
        shadow_seq: [B, K, 1, H, W]
        light_dir:  [B, K, 3]
        """
        b, k, c, h, w = shadow_seq.shape

        shadow_seq = shadow_seq.view(b * k, c, h, w)
        light_dir = light_dir.view(b * k, 3)

        img_feat = self.image_encoder(shadow_seq)  # [B*K, image_feat_dim]
        light_feat = self.light_encoder(light_dir) # [B*K, light_feat_dim]

        fused = torch.cat([img_feat, light_feat], dim=-1)
        fused = self.fusion(fused)                 # [B*K, fused_dim]
        fused = fused.view(b, k, -1)              # [B, K, fused_dim]

        global_feat = fused.reshape(b, k * fused.shape[-1])  # [B, K * fused_dim]

        points = self.decoder(global_feat)        # [B, N, 3]
        return points