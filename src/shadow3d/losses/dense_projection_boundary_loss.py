import torch
import torch.nn as nn
import torch.nn.functional as F


def make_light_plane_basis(light_dir: torch.Tensor, eps: float = 1e-6):
    """
    light_dir: [BK, 3]
    return:
        u: [BK, 3]
        v: [BK, 3]
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


def project_points_all_frames(points: torch.Tensor, light_dir: torch.Tensor):
    """
    points:    [B, N, 3]
    light_dir: [B, K, 3]

    return:
        uv:        [B*K, N, 2]
        points_bk: [B*K, N, 3]
    """
    if light_dir.dim() == 2:
        light_dir = light_dir[:, None, :]

    b, n, _ = points.shape
    k = light_dir.shape[1]

    points_bk = points[:, None, :, :].expand(b, k, n, 3).reshape(b * k, n, 3)
    light_bk = light_dir.reshape(b * k, 3)

    u, v = make_light_plane_basis(light_bk)

    x = (points_bk * u[:, None, :]).sum(dim=-1)
    y = (points_bk * v[:, None, :]).sum(dim=-1)
    uv = torch.stack([x, y], dim=-1)

    return uv, points_bk


def masked_chamfer_2d(
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    pred_valid: torch.Tensor,
    gt_valid: torch.Tensor,
    squared: bool = True,
    p2g_weight: float = 0.5,
    g2p_weight: float = 1.0,
):
    """
    pred_uv:    [BK, M, 2]
    gt_uv:      [BK, M, 2]
    pred_valid: [BK, M] bool
    gt_valid:   [BK, M] bool
    """
    device = pred_uv.device
    dtype = pred_uv.dtype

    pred_count = pred_valid.float().sum(dim=1)  # [BK]
    gt_count = gt_valid.float().sum(dim=1)      # [BK]
    sample_valid = (pred_count > 0) & (gt_count > 0)

    if sample_valid.sum() == 0:
        zero = pred_uv.new_tensor(0.0)
        return zero, zero, zero

    dist = torch.cdist(pred_uv, gt_uv, p=2)  # [BK, M, M]
    if squared:
        dist = dist ** 2

    large = torch.tensor(1e6, device=device, dtype=dtype)

    # p2g: 每个 pred 边界点找最近 gt 边界点
    dist_p2g = dist.masked_fill(~gt_valid[:, None, :], large)
    p2g_each = dist_p2g.min(dim=2).values  # [BK, M]
    p2g_each = p2g_each.masked_fill(~pred_valid, 0.0)
    p2g = p2g_each.sum(dim=1) / pred_count.clamp_min(1.0)

    # g2p: 每个 gt 边界点找最近 pred 边界点
    dist_g2p = dist.masked_fill(~pred_valid[:, :, None], large)
    g2p_each = dist_g2p.min(dim=1).values  # [BK, M]
    g2p_each = g2p_each.masked_fill(~gt_valid, 0.0)
    g2p = g2p_each.sum(dim=1) / gt_count.clamp_min(1.0)

    p2g = p2g[sample_valid].mean()
    g2p = g2p[sample_valid].mean()
    loss = p2g_weight * p2g + g2p_weight * g2p
    return loss, p2g, g2p


class DenseProjectionBoundaryLoss(nn.Module):
    """
    Dense projected boundary Chamfer loss.

    主要逻辑:
    1. 对 pred/gt 点云在所有光线方向上投影到 uv 平面；
    2. 在 uv 平面中栅格化成 occupancy；
    3. 用 max_pool / conv 在 CUDA 上提取 boundary band；
    4. 选出最多 max_boundary_points 个边界候选点；
    5. pred 边界候选点和 GT 边界候选点做 2D masked Chamfer。

    注意:
    - grid 只用于筛选边界点；
    - 最终 CD 使用连续 uv 坐标；
    - 不使用 cpu(), tolist(), dict，不逐 cell 循环。
    """

    def __init__(
        self,
        grid_size: int = 64,
        value_range=(-1.5, 1.5),
        splat_radius: int = 1,
        boundary_band: int = 1,
        max_boundary_points: int = 512,
        squared: bool = True,
        p2g_weight: float = 0.5,
        g2p_weight: float = 1.0,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.value_range = tuple(value_range)
        self.splat_radius = int(splat_radius)
        self.boundary_band = int(boundary_band)
        self.max_boundary_points = int(max_boundary_points)
        self.squared = bool(squared)
        self.p2g_weight = float(p2g_weight)
        self.g2p_weight = float(g2p_weight)

    @staticmethod
    def _max_pool_mask(mask: torch.Tensor, radius: int):
        """
        mask: [BK, 1, S, S] float/bool
        """
        if radius <= 0:
            return mask.float()
        k = 2 * radius + 1
        return F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=radius)

    def _select_boundary_points(self, uv: torch.Tensor, points3d: torch.Tensor):
        """
        uv:       [BK, N, 2]
        points3d: [BK, N, 3]

        return:
            edge_uv:    [BK, M, 2]
            edge_xyz:   [BK, M, 3]
            edge_valid: [BK, M] bool
            point_mask: [BK, N] bool, 原始点是否在 boundary band
        """
        bk, n, _ = uv.shape
        s = self.grid_size
        m = min(self.max_boundary_points, n)
        device = uv.device
        dtype = uv.dtype

        min_v, max_v = self.value_range
        xy = (uv - min_v) / (max_v - min_v)
        xy = xy.clamp(0.0, 1.0) * (s - 1)

        ix = xy[..., 0].long()
        iy = xy[..., 1].long()
        linear = iy * s + ix  # [BK, N]

        occ_flat = torch.zeros(bk, s * s, device=device, dtype=dtype)
        src = torch.ones(bk, n, device=device, dtype=dtype)
        occ_flat.scatter_(dim=1, index=linear, src=src)
        occ = occ_flat.view(bk, 1, s, s)

        # 稀疏点云投影先 splat 成局部占用区域
        occ_solid = self._max_pool_mask(occ, self.splat_radius)
        occ_bool = occ_solid > 0.5

        # 3x3 邻域不满则为边界
        kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype)
        neigh_count = F.conv2d(occ_solid, kernel, padding=1)
        boundary = occ_bool & (neigh_count < 9.0)

        # boundary band: 允许边界附近一圈点参与
        band = self._max_pool_mask(boundary.float(), self.boundary_band) > 0.5
        band_flat = band.view(bk, s * s)

        point_mask = torch.gather(band_flat, dim=1, index=linear)  # [BK, N] bool

        # 用 topk 固定输出 M 个点，不用 nonzero + Python 循环
        if self.training:
            score = torch.where(
                point_mask,
                torch.rand(bk, n, device=device, dtype=dtype),
                torch.full((bk, n), -1.0, device=device, dtype=dtype),
            )
        else:
            # eval 时确定性选择：按原始点顺序尽量靠前
            order = torch.linspace(1.0, 0.0, steps=n, device=device, dtype=dtype).view(1, n)
            score = torch.where(
                point_mask,
                order.expand(bk, n),
                torch.full((bk, n), -1.0, device=device, dtype=dtype),
            )

        top_score, top_idx = torch.topk(score, k=m, dim=1, largest=True)
        edge_valid = top_score >= 0.0

        idx_uv = top_idx.unsqueeze(-1).expand(-1, -1, 2)
        idx_xyz = top_idx.unsqueeze(-1).expand(-1, -1, 3)

        edge_uv = torch.gather(uv, dim=1, index=idx_uv)
        edge_xyz = torch.gather(points3d, dim=1, index=idx_xyz)

        return edge_uv, edge_xyz, edge_valid, point_mask

    def moved_to_gt_loss(
        self,
        pred_edge_xyz: torch.Tensor,
        gt_points_bk: torch.Tensor,
        pred_valid: torch.Tensor,
        squared: bool = True,
    ):
        """
        单向 3D 安全约束:
        被选中的 pred 边界候选点，移动后不要离 GT 表面太远。

        pred_edge_xyz: [BK, M, 3]
        gt_points_bk:  [BK, N, 3]
        pred_valid:    [BK, M]
        """
        count = pred_valid.float().sum(dim=1)
        sample_valid = count > 0

        if sample_valid.sum() == 0:
            return pred_edge_xyz.new_tensor(0.0)

        dist = torch.cdist(pred_edge_xyz, gt_points_bk, p=2)
        if squared:
            dist = dist ** 2

        nearest = dist.min(dim=2).values  # [BK, M]
        nearest = nearest.masked_fill(~pred_valid, 0.0)
        loss = nearest.sum(dim=1) / count.clamp_min(1.0)
        return loss[sample_valid].mean()

    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        light_dir: torch.Tensor,
        compute_moved_3d: bool = False,
    ):
        """
        pred_points: [B, N, 3]
        gt_points:   [B, N, 3]
        light_dir:   [B, K, 3]

        return dict:
            loss_boundary
            loss_p2g
            loss_g2p
            loss_moved_3d
            pred_boundary_ratio
            gt_boundary_ratio
        """
        if light_dir.dim() == 2:
            light_dir = light_dir[:, None, :]

        uv_pred, pred_bk = project_points_all_frames(pred_points, light_dir)
        uv_gt, gt_bk = project_points_all_frames(gt_points, light_dir)

        pred_edge_uv, pred_edge_xyz, pred_valid, pred_point_mask = self._select_boundary_points(uv_pred, pred_bk)
        gt_edge_uv, _, gt_valid, gt_point_mask = self._select_boundary_points(uv_gt, gt_bk)

        loss_boundary, loss_p2g, loss_g2p = masked_chamfer_2d(
            pred_edge_uv,
            gt_edge_uv,
            pred_valid,
            gt_valid,
            squared=self.squared,
            p2g_weight=self.p2g_weight,
            g2p_weight=self.g2p_weight,
        )

        if compute_moved_3d:
            loss_moved_3d = self.moved_to_gt_loss(
                pred_edge_xyz=pred_edge_xyz,
                gt_points_bk=gt_bk,
                pred_valid=pred_valid,
                squared=self.squared,
            )
        else:
            loss_moved_3d = pred_points.new_tensor(0.0)

        return {
            "loss_boundary": loss_boundary,
            "loss_p2g": loss_p2g.detach(),
            "loss_g2p": loss_g2p.detach(),
            "loss_moved_3d": loss_moved_3d,
            "pred_boundary_ratio": pred_point_mask.float().mean().detach(),
            "gt_boundary_ratio": gt_point_mask.float().mean().detach(),
        }
