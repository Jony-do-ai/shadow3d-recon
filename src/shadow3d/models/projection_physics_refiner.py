import torch
import torch.nn as nn
import torch.nn.functional as F


def make_light_plane_basis(light_dir: torch.Tensor, eps: float = 1e-6):
    """
    light_dir: [B*K, 3]
    return:
        u: [B*K, 3]
        v: [B*K, 3]
    """
    l = F.normalize(light_dir, dim=-1, eps=eps)
    bk = l.shape[0]
    device = l.device
    dtype = l.dtype

    ref_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).repeat(bk, 1)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).repeat(bk, 1)

    parallel = torch.abs((l * ref_z).sum(dim=-1)) > 0.95
    ref = torch.where(parallel[:, None], ref_y, ref_z)

    u = torch.cross(l, ref, dim=-1)
    u = F.normalize(u, dim=-1, eps=eps)

    v = torch.cross(l, u, dim=-1)
    v = F.normalize(v, dim=-1, eps=eps)

    return u, v


class MultiLightProjectionPhysicsRefiner(nn.Module):
    """
    多光线投影物理细化模块。

    输入:
        coarse_points: [B, N, 3]
        light_dir:     [B, K, 3]
        global_feat:   [B, global_dim]，可选但建议使用

    逻辑:
        1. 对每一帧光线单独构造垂直投影平面 u_k, v_k；
        2. 对每个点、每帧光线预测 delta_uv_k；
        3. 转成每帧对应的 delta_3d_k；
        4. 多帧 delta_3d_k mean 融合；
        5. refined = coarse + delta_scale * tanh(delta_3d)。

    注意:
        该模块不是 PCT，不做 KNN/attention；
        它是一个轻量级、作用在粗点云上的投影物理残差修正器。
    """

    def __init__(
        self,
        global_dim: int,
        hidden_dim: int = 128,
        global_context_dim: int = 64,
        light_context_dim: int = 32,
        delta_scale: float = 0.02,
        fuse: str = "mean",
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.fuse = str(fuse).lower()

        if self.fuse != "mean":
            raise ValueError(f"First version only supports fuse='mean', got {fuse}")

        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, global_context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_context_dim, global_context_dim),
            nn.ReLU(inplace=True),
        )

        self.light_proj = nn.Sequential(
            nn.Linear(3, light_context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(light_context_dim, light_context_dim),
            nn.ReLU(inplace=True),
        )

        # 每个点、每帧的输入:
        # xyz(3) + uv(2) + light_feat(light_context_dim) + global_feat(global_context_dim)
        in_dim = 3 + 2 + light_context_dim + global_context_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        # 初始接近 identity：最后一层输出从 0 开始
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, coarse_points: torch.Tensor, light_dir: torch.Tensor, global_feat: torch.Tensor):
        """
        return:
            refined_points: [B, N, 3]
            delta_3d:       [B, N, 3]
        """
        if light_dir.dim() == 2:
            light_dir = light_dir[:, None, :]

        b, n, _ = coarse_points.shape
        k = light_dir.shape[1]
        device = coarse_points.device
        dtype = coarse_points.dtype

        light_bk = light_dir.reshape(b * k, 3)
        u, v = make_light_plane_basis(light_bk)
        u_bk = u.view(b, k, 3)
        v_bk = v.view(b, k, 3)

        # [B, K, N, 3]
        points_bkn = coarse_points[:, None, :, :].expand(b, k, n, 3)

        # 每帧投影 uv: [B, K, N, 2]
        uv_x = (points_bkn * u_bk[:, :, None, :]).sum(dim=-1)
        uv_y = (points_bkn * v_bk[:, :, None, :]).sum(dim=-1)
        uv = torch.stack([uv_x, uv_y], dim=-1)

        global_ctx = self.global_proj(global_feat)  # [B, Cg]
        global_ctx = global_ctx[:, None, None, :].expand(b, k, n, -1)

        light_ctx = self.light_proj(light_dir.reshape(b * k, 3)).view(b, k, -1)
        light_ctx = light_ctx[:, :, None, :].expand(b, k, n, -1)

        feat = torch.cat([points_bkn, uv, light_ctx, global_ctx], dim=-1)  # [B, K, N, C]
        feat = feat.reshape(b * k * n, -1)

        delta_uv = self.mlp(feat).view(b, k, n, 2)  # [B, K, N, 2]

        # 每帧平面内 delta 转成 3D
        delta_3d_all = (
            delta_uv[..., 0:1] * u_bk[:, :, None, :]
            + delta_uv[..., 1:2] * v_bk[:, :, None, :]
        )  # [B, K, N, 3]

        # 多帧平均融合
        delta_3d = delta_3d_all.mean(dim=1)  # [B, N, 3]

        # 限制最大修正幅度
        delta_3d = self.delta_scale * torch.tanh(delta_3d)

        refined_points = coarse_points + delta_3d
        return refined_points, delta_3d
