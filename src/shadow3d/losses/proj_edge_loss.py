from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# proj_edge_loss.py 新增两个函数，替换 forward 里的 loss 计算部分
def compute_soft_boundary_map(occ: torch.Tensor, pool_size: int = 3) -> torch.Tensor:
    """
    从 soft occupancy map 提取边界响应。
    occ: [B, 1, G, G]
    return: boundary_map [B, 1, G, G]，边界处值高，内部/背景接近 0
    """
    # max_pool 膨胀一圈，减去原图，边界处差值最大
    dilated = F.max_pool2d(occ, kernel_size=pool_size, stride=1,
                           padding=pool_size // 2)
    boundary = (dilated - occ).clamp(min=0.0)
    # 归一化到 [0, 1]
    b_max = boundary.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    return boundary / b_max


def sample_boundary_weight_per_point(
    uv_norm: torch.Tensor,
    boundary_map: torch.Tensor,
) -> torch.Tensor:
    """
    把边界图采样到每个点的 uv 坐标上，得到 per-point 边界权重。
    uv_norm:      [B, N, 2]，归一化到 [0,1]
    boundary_map: [B, 1, G, G]
    return:       [B, N]，每个点的边界权重
    """
    # grid_sample 要求坐标在 [-1, 1]
    grid = uv_norm * 2.0 - 1.0          # [B, N, 2]
    grid = grid.unsqueeze(1)             # [B, 1, N, 2]

    # bilinear 插值，完全可微
    w = F.grid_sample(
        boundary_map,
        grid,
        mode='bilinear',
        padding_mode='zeros',   # 超出边界的点权重=0，自然忽略
        align_corners=True,
    )                                    # [B, 1, 1, N]
    return w.squeeze(1).squeeze(1)       # [B, N]


def weighted_chamfer_2d(
    uv_pred: torch.Tensor,
    uv_gt: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    在 2D 投影平面上做加权单向 CD（pred → gt 方向）。
    uv_pred:  [B, N, 2]
    uv_gt:    [B, M, 2]
    weights:  [B, N]，每个 pred 点的边界权重
    return:   scalar loss
    """
    # pred 每个点到 gt 最近点的距离
    dist = torch.cdist(uv_pred, uv_gt)  # [B, N, M]

    # pred→gt：加权，聚焦边界
    min_p2g, _ = dist.min(dim=-1)  # [B, N]
    w_sum = weights.sum(dim=-1).clamp(min=eps)
    loss_p2g = (weights * min_p2g ** 2).sum(dim=-1) / w_sum  # [B]

    # gt→pred：不加权，保证覆盖率
    min_g2p, _ = dist.min(dim=-2)  # [B, M]
    loss_g2p = min_g2p.pow(2).mean(dim=-1)  # [B]

    return (loss_p2g + loss_g2p).mean()


def make_light_plane_basis(light_dir: torch.Tensor, eps: float = 1e-6):
    """
    light_dir: [B, 3]

    return:
        u: [B, 3]
        v: [B, 3]
    """
    l = F.normalize(light_dir, dim=-1, eps=eps)
    B = l.shape[0]
    device = l.device
    dtype = l.dtype

    ref_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).repeat(B, 1)

    parallel = torch.abs((l * ref_z).sum(dim=-1)) > 0.95
    ref = torch.where(parallel[:, None], ref_y, ref_z)

    u = torch.cross(l, ref, dim=-1)
    u = F.normalize(u, dim=-1, eps=eps)

    v = torch.cross(l, u, dim=-1)
    v = F.normalize(v, dim=-1, eps=eps)

    return u, v


def project_to_light_plane(points: torch.Tensor, light_dir: torch.Tensor):
    """
    points: [B, N, 3]
    light_dir: [B, 3]

    return:
        uv: [B, N, 2]
    """
    u, v = make_light_plane_basis(light_dir)

    x = (points * u[:, None, :]).sum(dim=-1)
    y = (points * v[:, None, :]).sum(dim=-1)

    return torch.stack([x, y], dim=-1)


def normalize_uv_by_gt_box(
    uv: torch.Tensor,
    uv_gt: torch.Tensor,
    padding: float = 0.05,
    eps: float = 1e-6,
):
    """
    用 GT 投影范围定义 2D grid 坐标系。

    uv:    [B, N, 2]
    uv_gt: [B, M, 2]

    return:
        uv_norm: [B, N, 2]，大致落在 [0, 1]
    """
    gt_min = uv_gt.detach().amin(dim=1)  # [B, 2]
    gt_max = uv_gt.detach().amax(dim=1)  # [B, 2]

    center = 0.5 * (gt_min + gt_max)     # [B, 2]
    extent = gt_max - gt_min             # [B, 2]

    # 用统一尺度保持投影平面中的长宽比例，避免 x/y 被分别拉伸导致形状变形。
    scale = extent.amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = scale * (1.0 + 2.0 * float(padding))

    uv_norm = (uv - center[:, None, :]) / scale[:, None, :] + 0.5
    return uv_norm


def make_grid_centers(grid_size: int, device, dtype):
    """
    return:
        centers: [grid_size * grid_size, 2]，范围 [0, 1]
    """
    if grid_size <= 1:
        raise ValueError(f"grid_size must be > 1, got {grid_size}")

    t = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return centers


def soft_rasterize_uv_to_grid(
    uv_norm: torch.Tensor,
    grid_size: int = 64,
    sigma: float = 1.0,
    chunk_size: int = 1024,
):
    """
    CUDA/Tensor soft rasterization，不做 CPU numpy，不用逐点 Python 画图。

    uv_norm: [B, N, 2]，归一化后的投影坐标。

    return:
        occ: [B, 1, grid_size, grid_size]，soft occupancy map，范围 [0, 1]

    说明：
    - 每个投影点用一个高斯核 splat 到 64x64 grid；
    - occ = 1 - exp(-sum(weight))，多个点落入同一格不会无限增大；
    - pred 分支对 uv_norm 可导，因此 loss 能反传到 pred_points。
    """
    B, N, _ = uv_norm.shape
    device = uv_norm.device
    dtype = uv_norm.dtype

    centers = make_grid_centers(grid_size, device=device, dtype=dtype)  # [P, 2]
    P = centers.shape[0]

    # sigma 用“像素”为单位；sigma=1.0 约等于一个 grid cell 宽度。
    sigma_norm = float(sigma) / float(grid_size - 1)
    denom = 2.0 * sigma_norm * sigma_norm

    weight_sum = uv_norm.new_zeros((B, P))
    chunk_size = int(chunk_size) if chunk_size is not None and chunk_size > 0 else N

    for start in range(0, N, chunk_size):
        cur = uv_norm[:, start:start + chunk_size, :]  # [B, C, 2]
        dist2 = ((cur[:, :, None, :] - centers[None, None, :, :]) ** 2).sum(dim=-1)  # [B, C, P]
        weight_sum = weight_sum + torch.exp(-dist2 / denom).sum(dim=1)               # [B, P]

    occ = 1.0 - torch.exp(-weight_sum)
    return occ.view(B, 1, grid_size, grid_size)


def weighted_grid_loss(
    pred_occ: torch.Tensor,
    gt_occ: torch.Tensor,
    pos_weight: float = 4.0,
    use_smooth_l1: bool = True,
    eps: float = 1e-6,
):
    """
    pred_occ / gt_occ: [B, 1, G, G]

    正样本 grid 通常比背景少，所以对 GT 前景区域加权，避免全背景把 loss 稀释掉。
    """
    gt_occ = gt_occ.detach()

    if use_smooth_l1:
        loss_map = F.smooth_l1_loss(pred_occ, gt_occ, reduction="none")
    else:
        loss_map = (pred_occ - gt_occ) ** 2

    weight = 1.0 + float(pos_weight) * gt_occ
    return (loss_map * weight).sum() / (weight.sum() + eps)


class LightProjectionEdgeLoss(nn.Module):
    """
    Grid Projection Edge/Occupancy Loss.

    改动点：
    旧版本是 support-direction：每个方向只选 1 个最外侧支撑点；
    新版本是 grid：把点云沿光线方向投影到垂直光线的 2D 平面，
    再 soft rasterize 成 64x64 occupancy grid，比较 pred / gt 的投影 grid。

    这样同一条直线/同一段轮廓上的多个边界点可以同时进入 loss，
    不会被“一个方向一个 argmax 点”压缩掉。

    pred_points: [B, N, 3]
    gt_points:   [B, M, 3]
    light_dir:   [B, K, 3] or [B, 3]
    """

    def __init__(
        self,
        num_dirs: int = 64,
        squared: bool = True,
        max_frames: int = 1,
        frame_stride: int = 1,
        random_frames: bool = True,
        random_rotate_dirs: bool = True,
        support_weight: float = 1.0,
        chamfer_weight: float = 0.0,
        use_smooth_l1: bool = True,
        grid_size: Optional[int] = None,
        grid_padding: float = 0.05,
        grid_sigma: float = 1.0,
        grid_chunk_size: int = 1024,
        grid_pos_weight: float = 4.0,
        use_grid_loss: bool = True,
    ):
        super().__init__()
        self.use_grid_loss = bool(use_grid_loss)

        # 兼容旧配置：如果 train.py/yaml 还传 num_dirs，就把 num_dirs 当成 grid_size。
        self.num_dirs = num_dirs
        self.grid_size = int(grid_size if grid_size is not None else num_dirs)

        # squared / random_rotate_dirs / chamfer_weight 保留是为了兼容旧 train.py 和旧 yaml。
        # 新 grid loss 不再使用方向旋转，也不再使用 support point chamfer。
        self.squared = squared
        self.random_rotate_dirs = random_rotate_dirs
        self.chamfer_weight = chamfer_weight

        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.random_frames = random_frames
        self.grid_weight = support_weight
        self.use_smooth_l1 = use_smooth_l1

        self.grid_padding = float(grid_padding)
        self.grid_sigma = float(grid_sigma)
        self.grid_chunk_size = int(grid_chunk_size)
        self.grid_pos_weight = float(grid_pos_weight)

    def select_frame_ids(self, K: int, device):
        """
        训练时随机选光线帧，避免永远只用第 0 帧。
        验证/非训练时用固定 stride。
        """
        if self.training and self.random_frames:
            if self.max_frames is not None and self.max_frames > 0:
                num = min(K, self.max_frames)
                return torch.randperm(K, device=device)[:num]
            return torch.arange(K, device=device)

        frame_ids = torch.arange(0, K, max(1, self.frame_stride), device=device)

        if self.max_frames is not None and self.max_frames > 0:
            frame_ids = frame_ids[:self.max_frames]

        return frame_ids

    def forward(self, pred_points, gt_points, light_dir, return_stats: bool = False):
        if light_dir.dim() == 2:
            light_dir = light_dir[:, None, :]
        _, K, _ = light_dir.shape
        device = pred_points.device

        frame_ids = self.select_frame_ids(K, device=device)
        total_loss = pred_points.new_tensor(0.0)

        # === 诊断统计累加器（不参与反传） ===
        stat_loss_cd_sum = 0.0      # 加权 2D chamfer 总和
        stat_loss_grid_sum = 0.0    # grid loss 总和
        stat_p2g_sum = 0.0          # pred->gt 那一支
        stat_g2p_sum = 0.0          # gt->pred 那一支
        stat_w_mean_sum = 0.0       # 每个 pred 点的边界权重平均
        stat_w_active_sum = 0.0     # 权重 > 0.05 的 pred 点占比
        stat_uv_inside_sum = 0.0    # uv_pred_norm 落在 [0,1] 内的点占比

        for frame_pos in range(frame_ids.numel()):
            k_tensor = frame_ids[frame_pos:frame_pos + 1]
            cur_light = light_dir.index_select(dim=1, index=k_tensor).squeeze(1)

            uv_pred = project_to_light_plane(pred_points, cur_light)
            uv_gt = project_to_light_plane(gt_points, cur_light)

            uv_pred_norm = normalize_uv_by_gt_box(uv_pred, uv_gt, padding=self.grid_padding)
            uv_gt_norm = normalize_uv_by_gt_box(uv_gt, uv_gt, padding=self.grid_padding)

            # GT occupancy & 边界图（no_grad，只做监督）
            with torch.no_grad():
                gt_occ = soft_rasterize_uv_to_grid(
                    uv_gt_norm, self.grid_size, self.grid_sigma, self.grid_chunk_size
                )
                boundary_map = compute_soft_boundary_map(gt_occ)  # [B,1,G,G]

            # per-point 边界权重（grid_sample 可微）
            w = sample_boundary_weight_per_point(uv_pred_norm, boundary_map)  # [B,N]
            w = 0.3 + 0.7 * w  # base=0.3, 边界点权重 ≈ 1.0
            # 加权 CD（边界聚焦，梯度连续）—— 用拆分版同时拿到 p2g/g2p
            loss_cd, loss_p2g_val, loss_g2p_val = self._weighted_chamfer_2d_split(
                uv_pred_norm, uv_gt_norm, w
            )

            # 保留一个小权重的全局 grid loss 做正则（防止 pred 完全坍缩）
            # use_grid_loss=False 时跳过 pred 端 rasterize（最贵的一步，因为有 grad）
            if self.use_grid_loss:
                pred_occ = soft_rasterize_uv_to_grid(
                    uv_pred_norm, self.grid_size, self.grid_sigma, self.grid_chunk_size
                )
                loss_grid = weighted_grid_loss(
                    pred_occ, gt_occ,
                    pos_weight=self.grid_pos_weight,
                    use_smooth_l1=self.use_smooth_l1,
                )
                total_loss = total_loss + self.grid_weight * loss_cd + 0.1 * loss_grid
            else:
                loss_grid = total_loss.new_tensor(0.0)
                total_loss = total_loss + self.grid_weight * loss_cd

            # === 诊断统计（detach 防止参与反传） ===
            with torch.no_grad():
                stat_loss_cd_sum += float(loss_cd.detach())
                stat_loss_grid_sum += float(loss_grid.detach())
                stat_p2g_sum += float(loss_p2g_val.detach())
                stat_g2p_sum += float(loss_g2p_val.detach())
                stat_w_mean_sum += float(w.mean().detach())
                stat_w_active_sum += float((w > 0.05).float().mean().detach())
                inside = ((uv_pred_norm >= 0.0) & (uv_pred_norm <= 1.0)).all(dim=-1).float()
                stat_uv_inside_sum += float(inside.mean().detach())

        n_frames = max(frame_ids.numel(), 1)
        # 直接除帧数做归一化（不乘 run_every_batch）
        out_loss = total_loss / n_frames

        if not return_stats:
            return out_loss

        stats = {
            "proj_loss_raw": float(out_loss.detach()),
            "proj_loss_cd_2d": stat_loss_cd_sum / n_frames,         # 加权 2D chamfer
            "proj_loss_grid": stat_loss_grid_sum / n_frames,        # grid 监督
            "proj_loss_p2g": stat_p2g_sum / n_frames,               # pred->gt
            "proj_loss_g2p": stat_g2p_sum / n_frames,               # gt->pred
            "proj_w_mean": stat_w_mean_sum / n_frames,              # 平均边界权重
            "proj_w_active_ratio": stat_w_active_sum / n_frames,    # 有效梯度点占比
            "proj_uv_inside_ratio": stat_uv_inside_sum / n_frames,  # 投影点落在 grid 内占比
            "proj_n_frames": n_frames,
        }
        return out_loss, stats

    @staticmethod
    def _weighted_chamfer_2d_split(uv_pred, uv_gt, weights, eps: float = 1e-8):
        """
        和 weighted_chamfer_2d 完全等价，但额外返回 p2g 和 g2p 两支用于诊断。
        """
        dist = torch.cdist(uv_pred, uv_gt)  # [B, N, M]

        min_p2g, _ = dist.min(dim=-1)
        w_sum = weights.sum(dim=-1).clamp(min=eps)
        loss_p2g = (weights * min_p2g ** 2).sum(dim=-1) / w_sum

        min_g2p, _ = dist.min(dim=-2)
        loss_g2p = min_g2p.pow(2).mean(dim=-1)

        total = (loss_p2g + loss_g2p).mean()
        return total, loss_p2g.mean(), loss_g2p.mean()