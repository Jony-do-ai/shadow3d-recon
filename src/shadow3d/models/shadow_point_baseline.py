import math
from typing import Optional
from .module import NeighborEmbedding, OA
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


def pairwise_square_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, C]
    y: [B, M, C]
    return: [B, N, M]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)      # [B, N, 1]
    yy = (y ** 2).sum(dim=-1).unsqueeze(1)       # [B, 1, M]
    xy = torch.bmm(x, y.transpose(1, 2))         # [B, N, M]
    return torch.clamp(xx + yy - 2.0 * xy, min=0.0)


def knn_indices(coords: torch.Tensor, k: int) -> torch.Tensor:
    """
    coords: [B, N, 3]
    return: [B, N, k]，每个点最近的 k 个邻居索引，不包含自身。
    """
    b, n, _ = coords.shape
    k_eff = min(k + 1, n)

    with torch.no_grad():
        dist = pairwise_square_distance(coords, coords)      # [B, N, N]
        idx = dist.topk(k=k_eff, dim=-1, largest=False)[1]   # [B, N, k+1]
        if k_eff > 1:
            idx = idx[:, :, 1:]                              # 去掉最近的自己

        if idx.shape[-1] < k:
            pad = idx[:, :, -1:].expand(-1, -1, k - idx.shape[-1])
            idx = torch.cat([idx, pad], dim=-1)

    return idx                                               # [B, N, k]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx:    [B, N, K]
    return: [B, N, K, C]
    """
    b, n, c = points.shape
    k = idx.shape[-1]
    batch_offset = torch.arange(b, device=points.device).view(b, 1, 1) * n
    flat_idx = (idx + batch_offset).reshape(-1)
    flat_points = points.reshape(b * n, c)
    neighbors = flat_points[flat_idx].view(b, n, k, c)
    return neighbors


class LocalNeighborEmbedding(nn.Module):
    """
    不做 FPS 降采样的局部邻域嵌入。
    输入 2048 个点，输出仍然是 2048 个点的特征。
    """
    def __init__(self, channels: int = 128, k: int = 16):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, coords: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        idx = knn_indices(coords, self.k)                         # [B, N, k]
        neighbor_feat = index_points(feat, idx)                   # [B, N, k, C]
        center_feat = feat.unsqueeze(2).expand_as(neighbor_feat)  # [B, N, k, C]
        edge_feat = torch.cat([neighbor_feat - center_feat, center_feat], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)                      # [B, N, k, C]
        local_feat = edge_feat.max(dim=2).values                  # [B, N, C]
        return local_feat


class OffsetAttentionBlock(nn.Module):
    """
    PCT 风格 Offset-Attention。
    输入/输出形状都保持 [B, N, C]。
    """
    def __init__(
        self,
        channels: int = 128,
        global_dim: int = 512,
        qk_dim: Optional[int] = None,
        use_condition: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.qk_dim = qk_dim or max(channels // 4, 1)
        self.use_condition = use_condition

        self.q_proj = nn.Linear(channels, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(channels, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(channels, channels)

        self.trans = nn.Linear(channels, channels)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

        if use_condition:
            self.cond_proj = nn.Linear(global_dim, channels * 2)
        else:
            self.cond_proj = None

    def apply_condition(self, x: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        if self.cond_proj is None:
            return x
        gamma_beta = self.cond_proj(global_feat).unsqueeze(1)     # [B, 1, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                 # [B, 1, C], [B, 1, C]
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = self.apply_condition(x, global_feat)                  # [B, N, C]

        q = self.q_proj(x)                                        # [B, N, C/4]
        k = self.k_proj(x)                                        # [B, N, C/4]
        v = self.v_proj(x)                                        # [B, N, C]

        energy = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.qk_dim)  # [B, N, N]
        attention = self.softmax(energy)                          # [B, N, N]
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-9)

        attended = torch.bmm(attention.transpose(1, 2), v)         # [B, N, C]
        offset = x - attended                                     # [B, N, C]

        out = self.trans(offset)                                  # [B, N, C]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)       # BatchNorm1d 要求 [B, C, N]
        out = self.act(out)                                       # [B, N, C]
        return x + out                                            # [B, N, C]


class ShadowConditionedPCTRefiner(nn.Module):
    """
    输入:
        coarse_points: [B, N, 3]
        global_feat:   [B, global_dim]
    输出:
        refined_points: [B, N, 3]
    """
    def __init__(
        self,
        global_dim: int = 512,
        hidden_dim: int = 128,
        coord_dim: int = 64,
        shadow_dim: int = 128,
        num_blocks: int = 4,
        knn_k: int = 16,
        delta_scale: float = 0.05,
        qk_dim: Optional[int] = None,
        use_condition: bool = True,
    ):
        super().__init__()
        self.delta_scale = delta_scale

        self.coord_embed = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_dim, coord_dim),
            nn.ReLU(inplace=True),
        )

        self.shadow_embed = nn.Sequential(
            nn.Linear(global_dim, shadow_dim),
            nn.ReLU(inplace=True),
            nn.Linear(shadow_dim, shadow_dim),
            nn.ReLU(inplace=True),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(coord_dim + shadow_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.neighbor_embed = LocalNeighborEmbedding(channels=hidden_dim, k=knn_k)

        self.blocks = nn.ModuleList([
            OffsetAttentionBlock(
                channels=hidden_dim,
                global_dim=global_dim,
                qk_dim=qk_dim,
                use_condition=use_condition,
            )
            for _ in range(num_blocks)
        ])

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coarse_points: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        b, n, _ = coarse_points.shape

        coord_feat = self.coord_embed(coarse_points)              # [B, N, coord_dim]
        shadow_feat = self.shadow_embed(global_feat)              # [B, shadow_dim]
        shadow_feat = shadow_feat.unsqueeze(1).expand(-1, n, -1)  # [B, N, shadow_dim]

        feat = torch.cat([coord_feat, shadow_feat], dim=-1)       # [B, N, coord_dim + shadow_dim]
        feat = self.input_proj(feat)                              # [B, N, hidden_dim]

        local_feat = self.neighbor_embed(coarse_points, feat)     # [B, N, hidden_dim]
        feat = feat + local_feat                                  # [B, N, hidden_dim]

        for block in self.blocks:
            feat = block(feat, global_feat)                       # [B, N, hidden_dim]

        delta = self.delta_head(feat)                             # [B, N, 3]
        delta = self.delta_scale * torch.tanh(delta)              # 限制每个点的最大修正幅度
        refined_points = coarse_points + delta                    # [B, N, 3]
        return refined_points


class OriginalPaperPCTRefiner(nn.Module):
    def __init__(self, num_points=2048, delta_scale=0.01):
        super().__init__()

        self.num_points = num_points
        self.delta_scale = delta_scale

        # samples=[2048, 2048] 可以保证输出点数还是 2048
        self.neighbor_embedding = NeighborEmbedding(samples=[num_points, num_points])

        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.delta_head = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        # 关键：让 refiner 初始时接近 identity，不要一开始就把点云拉乱
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, coarse_points, global_feat=None):
        """
        coarse_points: [B, 2048, 3]
        return:        [B, 2048, 3]
        """
        x = coarse_points.transpose(1, 2).contiguous()  # [B, 3, 2048]

        # 注意：NeighborEmbedding 现在返回采样后的坐标和特征
        sampled_points, x = self.neighbor_embedding(x)  # sampled_points: [B, 2048, 3], x: [B, 256, 2048]

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x_concat = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 1024, 2048]

        delta = self.delta_head(x_concat)              # [B, 3, 2048]
        delta = self.delta_scale * torch.tanh(delta)   # [B, 3, 2048]

        refined_points = sampled_points + delta.transpose(1, 2).contiguous()

        return refined_points

class ShadowPointBaseline(nn.Module):
    """
    Baseline + Shadow-conditioned PCT Refiner:
        shadow_seq + light_dir -> CNN/light encoder -> global_feat
        global_feat -> coarse MLP decoder -> P0
        P0 + global_feat -> PCT Refiner -> refined points
    """
    def __init__(
        self,
        image_feat_dim: int = 256,
        light_feat_dim: int = 128,
        fused_dim: int = 256,
        num_points: int = 2048,
        num_frames: int = 10,
        use_pct_refiner: bool = True,
        pct_hidden_dim: int = 128,
        pct_coord_dim: int = 64,
        pct_shadow_dim: int = 128,
        pct_blocks: int = 4,
        pct_knn_k: int = 16,
        pct_delta_scale: float = 0.01,
        pct_qk_dim: Optional[int] = None,
        pct_use_condition: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.fused_dim = fused_dim
        self.image_encoder = ShadowImageEncoder(feat_dim=image_feat_dim)
        self.light_encoder = LightEncoder(light_feat_dim=light_feat_dim)
        self.use_pct_refiner = use_pct_refiner
        self.global_dim = fused_dim * num_frames

        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + light_feat_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = PointCloudDecoder(global_dim=self.global_dim, num_points=num_points)

        if self.use_pct_refiner:
            self.pct_refiner = OriginalPaperPCTRefiner(
                num_points=num_points,
                delta_scale=pct_delta_scale,
            )
        else:
            self.pct_refiner = None

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
        global_feat = self.encode_global_feature(shadow_seq, light_dir)  # [B, global_dim]
        coarse_points = self.decoder(global_feat)                        # [B, N, 3]

        if self.pct_refiner is None:
            return coarse_points

        refined_points = self.pct_refiner(coarse_points, global_feat)     # [B, N, 3]
        return refined_points


