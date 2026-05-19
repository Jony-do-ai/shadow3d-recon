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
        )

    def forward(self, light_dir):
        return self.mlp(light_dir)

class LightFiLMModulator(nn.Module):
    """
    残差式 FiLM:
        out = img_feat * (1 + gamma) + beta

    零初始化最后一层:训练初期 gamma=beta=0,等价于不调制;
    训练中 gamma/beta 自由学习调制强度,不再被 film_scale 永久压制。
    """
    def __init__(
        self,
        light_feat_dim: int = 128,
        image_feat_dim: int = 256,
        film_scale: float = 0.1,
    ):
        super().__init__()
        self.film_scale = film_scale
        self.mlp = nn.Sequential(
            nn.Linear(light_feat_dim, image_feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(image_feat_dim * 2, image_feat_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, img_feat: torch.Tensor, light_feat: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.mlp(light_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        img_feat = img_feat * (1.0 + self.film_scale * gamma) + self.film_scale * beta
        return img_feat


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
        film_scale: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.fused_dim = fused_dim
        self.image_encoder = ShadowImageEncoder(feat_dim=image_feat_dim)
        self.light_encoder = LightEncoder(light_feat_dim=light_feat_dim)
        self.light_film = LightFiLMModulator(
            light_feat_dim=light_feat_dim,
            image_feat_dim=fused_dim,
            film_scale=film_scale,
        )
        self.fusion_linear1 = nn.Linear(image_feat_dim, fused_dim)
        self.fusion_linear2 = nn.Linear(fused_dim, fused_dim)

        self.global_dim = fused_dim * num_frames
        self.decoder = PointCloudDecoder(
            global_dim=self.global_dim,
            num_points=num_points,
        )

        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = SimpleTransformerRefiner(
                hidden_dim=refiner_hidden_dim,
                num_blocks=refiner_blocks,
                num_heads=refiner_num_heads,
                delta_scale=refiner_delta_scale,
                cond_dim=self.global_dim,
            )
        else:
            self.refiner = None

    def encode_global_feature(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = shadow_seq.shape

        shadow_seq = shadow_seq.reshape(b * k, c, h, w)
        light_dir = light_dir.reshape(b * k, 3)

        img_feat = self.image_encoder(shadow_seq)  # [B*K, image_feat_dim]
        light_feat = self.light_encoder(light_dir)  # [B*K, light_feat_dim]

        h1 = self.fusion_linear1(img_feat)  # [B*K, fused_dim]
        h1 = self.light_film(h1, light_feat)  # [B*K, fused_dim]
        h1 = torch.relu(h1)

        h2 = self.fusion_linear2(h1)  # [B*K, fused_dim]
        fused = torch.relu(h2)

        fused = fused.reshape(b, k, -1)
        global_feat = fused.reshape(b, k * fused.shape[-1])
        return global_feat

    def forward(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        global_feat = self.encode_global_feature(shadow_seq, light_dir)
        coarse_points = self.decoder(global_feat)

        if self.refiner is None:
            return coarse_points

        # 修改:把 global_feat 传进去(refiner 内部会根据 use_cond 决定用不用)
        refined_points = self.refiner(coarse_points, global_feat=global_feat)
        return refined_points

