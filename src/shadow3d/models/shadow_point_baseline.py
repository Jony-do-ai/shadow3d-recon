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
    输出空间特征图: [B, feat_dim, H', W']    (H' = H/32, W' = W/32)
    不再做 global pooling,把空间信息留给后续的 cross-attention。
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
        self.layer4 = ConvBlock(256, feat_dim, stride=2)

    def forward(self, x):
        x = self.stem(x)    # [B, 32,  H/2,  W/2]
        x = self.layer1(x)  # [B, 64,  H/4,  W/4]
        x = self.layer2(x)  # [B, 128, H/8,  W/8]
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        x = self.layer4(x)  # [B, D,   H/32, W/32]
        return x            # 注意:保留空间维度


class LightEncoder(nn.Module):
    """
    输入光照方向 [B, 3]
    输出光照特征 [B, light_feat_dim]
    """
    def __init__(self, light_feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, light_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(light_feat_dim, light_feat_dim),
        )

    def forward(self, light_dir):
        return self.mlp(light_dir)


class LightQueryCrossAttention(nn.Module):
    """
    Cross-attention:光照方向作为 query,阴影 feature map 作为 key/value。

    输入:
        light_token:   [B*K, 1, D]      每帧一个 query token
        shadow_tokens: [B*K, HW, D]     每帧 H'*W' 个空间 token
    输出:
        frame_token:   [B*K, D]         每帧融合后的特征
    """
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 一个轻量的 FFN,让 cross-attention 的输出再过一次非线性
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, light_token: torch.Tensor, shadow_tokens: torch.Tensor) -> torch.Tensor:
        # light_token:   [BK, 1,  D]
        # shadow_tokens: [BK, HW, D]
        q = self.norm_q(light_token)
        kv = self.norm_kv(shadow_tokens)

        attended, _ = self.attn(q, kv, kv)             # [BK, 1, D]
        out = light_token + attended                   # 残差: query 自身 + attention 结果

        # FFN
        out = out + self.ffn(self.norm_ffn(out))       # [BK, 1, D]
        return out.squeeze(1)                          # [BK, D]


class FrameAggregator(nn.Module):
    """
    把 K 帧的 frame_token 聚合成一个全局向量。
    支持两种模式:
        - 'mean':  简单平均(对帧数和顺序都不敏感,稳健)
        - 'attn':  learnable query + cross-attention 聚合(参数多一点,但能学到帧的重要性)
    """
    def __init__(self, dim: int = 256, mode: str = "mean", num_heads: int = 4):
        super().__init__()
        assert mode in ("mean", "attn")
        self.mode = mode
        self.dim = dim

        if mode == "attn":
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, batch_first=True,
            )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        # frame_tokens: [B, K, D]
        if self.mode == "mean":
            return frame_tokens.mean(dim=1)                                  # [B, D]

        # attn 模式
        b = frame_tokens.shape[0]
        q = self.query.expand(b, -1, -1)                                     # [B, 1, D]
        q = self.norm_q(q)
        kv = self.norm_kv(frame_tokens)
        out, _ = self.attn(q, kv, kv)                                        # [B, 1, D]
        return out.squeeze(1)                                                # [B, D]


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
        x = self.mlp(z)
        x = torch.tanh(x)
        x = x.view(z.shape[0], self.num_points, 3)
        return x


class SimpleTransformerRefiner(nn.Module):
    """
    点云精修器:在 coarse_points 上做 transformer + delta 预测。
    保持和原版一致,只多了 cond_dim 支持。
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_heads: int = 4,
        delta_scale: float = 0.05,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.use_cond = cond_dim is not None

        self.coord_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self.coord_embed(coarse_points)

        if self.use_cond:
            assert global_feat is not None
            cond = self.cond_proj(global_feat)
            feat = feat + cond.unsqueeze(1)

        feat = self.transformer(feat)
        delta = self.delta_head(feat)
        delta = self.delta_scale * torch.tanh(delta)
        return coarse_points + delta


class ShadowPointBaseline(nn.Module):
    """
    方案 B 版本:
        shadow_seq → CNN(保留空间) → [B*K, D, H', W']
        light_dir  → MLP            → [B*K, D]   (作为 query token)

        每帧:cross-attention(query=light, key/val=shadow_tokens)
             → frame_token [B, K, D]

        K 帧聚合(mean / attn) → global_feat [B, D]
        global_feat → coarse decoder → P0 [B, N, 3]
        P0 → refiner → refined points
    """
    def __init__(
        self,
        feat_dim: int = 256,                # 统一的特征维度 D(image / light / cross-attn)
        num_points: int = 2048,
        num_frames: int = 10,
        cross_attn_heads: int = 4,
        frame_agg_mode: str = "mean",       # 'mean' 或 'attn'
        use_refiner: bool = True,
        refiner_hidden_dim: int = 128,
        refiner_blocks: int = 2,
        refiner_num_heads: int = 4,
        refiner_delta_scale: float = 0.05,
        refiner_use_cond: bool = True,
        # ---- 旧版兼容参数:让旧的 yaml / train.py 无需修改也能跑 ----
        # 旧版有 image_feat_dim / light_feat_dim / fused_dim 三个独立维度,
        # 新版统一成 feat_dim。这里只取 fused_dim(或 image_feat_dim)作为 feat_dim。
        image_feat_dim: Optional[int] = None,
        light_feat_dim: Optional[int] = None,
        fused_dim: Optional[int] = None,
    ):
        super().__init__()

        # 旧参数 -> 新参数 映射
        # 优先使用 fused_dim(它在旧版里就是聚合后的维度,语义最接近 feat_dim)
        if fused_dim is not None:
            feat_dim = int(fused_dim)
        elif image_feat_dim is not None:
            feat_dim = int(image_feat_dim)
        # light_feat_dim 在新版里没有独立意义(被合并到 feat_dim),忽略即可
        _ = light_feat_dim  # 显式忽略,避免 lint 报警

        # feat_dim 必须是 4 的倍数(2D sincos pos embed 要求)
        if feat_dim % 4 != 0:
            raise ValueError(f"feat_dim must be a multiple of 4, got {feat_dim}")

        self.num_frames = num_frames
        self.feat_dim = feat_dim
        self.global_dim = feat_dim          # 聚合后只有一个 D 维向量

        self.image_encoder = ShadowImageEncoder(feat_dim=feat_dim)
        self.light_encoder = LightEncoder(light_feat_dim=feat_dim)

        # 把 CNN 的空间特征加上 2D 位置编码(让 attention 知道空间位置)
        # 这里采用可学习的 2D positional embedding,初始化为小值
        # 注意: H', W' 在 forward 里根据输入 H,W 动态决定,所以我们用一个足够大的 buffer
        # 这里先不显式放 pos_emb,改用更通用的做法:在 forward 里按需创建。
        # 为了简洁稳定,使用 sinusoidal 2D pos encoding(无参数)。

        self.cross_attn = LightQueryCrossAttention(
            dim=feat_dim, num_heads=cross_attn_heads,
        )

        self.frame_agg = FrameAggregator(
            dim=feat_dim, mode=frame_agg_mode, num_heads=cross_attn_heads,
        )

        self.decoder = PointCloudDecoder(global_dim=self.global_dim, num_points=num_points)

        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = SimpleTransformerRefiner(
                hidden_dim=refiner_hidden_dim,
                num_blocks=refiner_blocks,
                num_heads=refiner_num_heads,
                delta_scale=refiner_delta_scale,
                cond_dim=self.global_dim if refiner_use_cond else None,
            )
        else:
            self.refiner = None

    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
        """
        2D sinusoidal positional embedding,无可学习参数。
        返回: [1, H*W, dim]
        """
        assert dim % 4 == 0, "feat_dim 必须是 4 的倍数(2D sincos pos embed 需要)"
        d_each = dim // 4
        # 频率
        omega = torch.arange(d_each, device=device, dtype=dtype) / d_each
        omega = 1.0 / (10000 ** omega)                                       # [d_each]

        y = torch.arange(h, device=device, dtype=dtype)                      # [H]
        x = torch.arange(w, device=device, dtype=dtype)                      # [W]

        out_y = torch.einsum("h,d->hd", y, omega)                            # [H, d_each]
        out_x = torch.einsum("w,d->wd", x, omega)                            # [W, d_each]

        pe_y = torch.cat([out_y.sin(), out_y.cos()], dim=-1)                 # [H, 2*d_each]
        pe_x = torch.cat([out_x.sin(), out_x.cos()], dim=-1)                 # [W, 2*d_each]

        pe_y = pe_y.unsqueeze(1).expand(h, w, -1)                            # [H, W, 2*d_each]
        pe_x = pe_x.unsqueeze(0).expand(h, w, -1)                            # [H, W, 2*d_each]

        pe = torch.cat([pe_y, pe_x], dim=-1)                                 # [H, W, dim]
        pe = pe.reshape(1, h * w, dim)
        return pe

    def encode_global_feature(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        """
        shadow_seq: [B, K, 1, H, W]
        light_dir:  [B, K, 3]
        return:     [B, D]
        """
        b, k, c, h, w = shadow_seq.shape

        # 1) CNN 提取空间特征
        shadow_seq = shadow_seq.view(b * k, c, h, w)                         # [BK, 1, H, W]
        fmap = self.image_encoder(shadow_seq)                                # [BK, D, H', W']
        bk, d, hp, wp = fmap.shape
        shadow_tokens = fmap.flatten(2).transpose(1, 2)                      # [BK, H'*W', D]

        # 2) 加 2D positional embedding(让 cross-attention 知道空间位置)
        pos = self._build_2d_sincos_pos_embed(
            hp, wp, d, device=shadow_tokens.device, dtype=shadow_tokens.dtype,
        )                                                                    # [1, H'W', D]
        shadow_tokens = shadow_tokens + pos                                  # 广播 [BK, H'W', D]

        # 3) 光照方向编码为 query token
        light_dir = light_dir.view(b * k, 3)                                 # [BK, 3]
        light_feat = self.light_encoder(light_dir)                           # [BK, D]
        light_token = light_feat.unsqueeze(1)                                # [BK, 1, D]

        # 4) Cross-attention:每帧独立做
        frame_token = self.cross_attn(light_token, shadow_tokens)            # [BK, D]
        frame_tokens = frame_token.view(b, k, d)                             # [B, K, D]

        # 5) K 帧聚合
        global_feat = self.frame_agg(frame_tokens)                           # [B, D]
        return global_feat

    def forward(self, shadow_seq: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
        global_feat = self.encode_global_feature(shadow_seq, light_dir)
        coarse_points = self.decoder(global_feat)

        if self.refiner is None:
            return coarse_points

        refined_points = self.refiner(coarse_points, global_feat=global_feat)
        return refined_points