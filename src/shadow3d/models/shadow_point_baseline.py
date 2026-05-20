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
    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        delta_scale: float = 0.05,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,   # 新增:global_feat 的维度
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.use_cond = cond_dim is not None   # 新增:是否使用条件注入

        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 新增:把 global_feat 投影到 hidden_dim,作为每个 token 的 condition
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        coarse_points: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,   # 新增参数
    ) -> torch.Tensor:
        feat = self.coord_embed(coarse_points)        # [B, N, hidden_dim]

        # 新增:把 global_feat 广播加到每个点的特征上
        if self.use_cond:
            assert global_feat is not None, "use_cond=True 时必须传 global_feat"
            cond = self.cond_proj(global_feat)        # [B, hidden_dim]
            feat = feat + cond.unsqueeze(1)           # [B, N, hidden_dim],广播到 N 个点

        feat = self.transformer(feat)
        delta = self.delta_head(feat)
        delta = self.delta_scale * torch.tanh(delta)
        return coarse_points + delta


class ShadowPointBaseline(nn.Module):
    """
    Baseline + Simple Transformer Refiner:
        shadow_seq + light_dir -> CNN/light encoder -> global_feat
        global_feat -> coarse MLP decoder -> P0
        P0 -> SimpleTransformerRefiner -> refined points
    """
    def __init__(
        self,
        image_feat_dim: int = 256,
        light_feat_dim: int = 128,
        fused_dim: int = 256,
        num_points: int = 2048,
        num_frames: int = 10,
        use_refiner: bool = True,
        refiner_hidden_dim: int = 128,
        refiner_blocks: int = 2,
        refiner_num_heads: int = 4,
        refiner_delta_scale: float = 0.05,
    ):
        super().__init__()
        # num_frames 在这里表示固定的最大帧槽位数，例如统一固定为 10。
        # 真实输入可以是 1 / 3 / 5 / 10 帧，但最后都会补齐到 num_frames 个特征槽
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
                cond_dim=self.global_dim,   # 控制是否注入
            )
        else:
            self.refiner = None

    def encode_global_feature(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        """
        shadow_seq: [B, K, 1, H, W]
        light_dir:  [B, K, 3]

        K 可以是 1 / 3 / 5 / 10。
        但模型内部会把 K 帧特征补齐到 self.num_frames 帧，
        因此输出 global_feat 永远是 [B, self.num_frames * fused_dim]。
        """
        b, k, c, h, w = shadow_seq.shape

        if k > self.num_frames:
            raise ValueError(
                f"Input frames K={k} is larger than model.num_frames={self.num_frames}. "
                f"For fixed-slot ablation, model.num_frames should be the maximum slot number, e.g. 10."
            )

        shadow_seq = shadow_seq.view(b * k, c, h, w)
        light_dir = light_dir.view(b * k, 3)

        img_feat = self.image_encoder(shadow_seq)  # [B*K, image_feat_dim]
        light_feat = self.light_encoder(light_dir)  # [B*K, light_feat_dim]

        fused = torch.cat([img_feat, light_feat], dim=-1)
        fused = self.fusion(fused)  # [B*K, fused_dim]
        fused = fused.view(b, k, -1)  # [B, K, fused_dim]

        # 关键：在特征层补零，而不是在图像层补黑图。
        # 这样补进去的帧不会被 CNN 解释成某种真实阴影。
        if k < self.num_frames:
            pad_frames = self.num_frames - k
            pad_feat = fused.new_zeros(b, pad_frames, fused.shape[-1])
            fused = torch.cat([fused, pad_feat], dim=1)  # [B, self.num_frames, fused_dim]

        global_feat = fused.reshape(b, self.num_frames * fused.shape[-1])
        return global_feat

    def forward(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        global_feat = self.encode_global_feature(shadow_seq, light_dir)
        coarse_points = self.decoder(global_feat)

        if self.refiner is None:
            return coarse_points

        # 修改:把 global_feat 传进去(refiner 内部会根据 use_cond 决定用不用)
        refined_points = self.refiner(coarse_points, global_feat=global_feat)
        return refined_points

