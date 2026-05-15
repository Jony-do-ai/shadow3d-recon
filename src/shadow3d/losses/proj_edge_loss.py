import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def make_support_dirs(
    num_dirs: int,
    device,
    dtype,
    random_rotate: bool = False,
):
    """
    return:
        dirs: [D, 2]
    """
    if random_rotate:
        offset = torch.rand((), device=device, dtype=dtype) * (2.0 * math.pi / num_dirs)
    else:
        offset = torch.zeros((), device=device, dtype=dtype)

    idx = torch.arange(num_dirs, device=device, dtype=dtype)
    theta = idx / num_dirs * (2.0 * math.pi) + offset

    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # [D, 2]
    return dirs


def gather_support_points(uv: torch.Tensor, dirs: torch.Tensor):
    """
    uv:   [B, N, 2]
    dirs: [D, 2]

    return:
        support_points: [B, D, 2]
        support_values: [B, D]
        support_indices: [B, D]
    """
    # 用 detach 选支撑点，避免 argmax 选择过程参与梯度
    scores_detached = torch.matmul(uv.detach(), dirs.t())  # [B, N, D]
    support_indices = torch.argmax(scores_detached, dim=1)  # [B, D]

    gather_idx = support_indices.unsqueeze(-1).expand(-1, -1, 2)  # [B, D, 2]
    support_points = torch.gather(uv, dim=1, index=gather_idx)     # [B, D, 2]

    # 注意这里用没有 detach 的 support_points，保证 pred_points 有梯度
    support_values = (support_points * dirs[None, :, :]).sum(dim=-1)  # [B, D]

    return support_points, support_values, support_indices


def batch_chamfer_2d(a: torch.Tensor, b: torch.Tensor, squared: bool = True):
    """
    a: [B, Na, 2]
    b: [B, Nb, 2]

    return:
        scalar
    """
    dist = torch.cdist(a, b, p=2)  # [B, Na, Nb]

    if squared:
        dist = dist ** 2

    a2b = dist.min(dim=2)[0].mean(dim=1)  # [B]
    b2a = dist.min(dim=1)[0].mean(dim=1)  # [B]

    return (a2b + b2a).mean()


class LightProjectionEdgeLoss(nn.Module):
    """
    Support-Direction Projection Edge Loss.

    核心思想：
    对每个光线方向，把点云投影到垂直光线的 2D 平面；
    在 2D 平面中设置多个外法线方向；
    每个方向只选最外侧支撑点；
    约束 pred 支撑点 / 支撑值 与 gt 一致。

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
        chamfer_weight: float = 0.5,
        use_smooth_l1: bool = True,
    ):
        super().__init__()
        self.num_dirs = num_dirs
        self.squared = squared
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.random_frames = random_frames
        self.random_rotate_dirs = random_rotate_dirs
        self.support_weight = support_weight
        self.chamfer_weight = chamfer_weight
        self.use_smooth_l1 = use_smooth_l1

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

    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        light_dir: torch.Tensor,
    ):
        if light_dir.dim() == 2:
            light_dir = light_dir[:, None, :]

        B, K, _ = light_dir.shape
        device = pred_points.device
        dtype = pred_points.dtype

        frame_ids = self.select_frame_ids(K, device=device)

        dirs = make_support_dirs(
            num_dirs=self.num_dirs,
            device=device,
            dtype=dtype,
            random_rotate=(self.training and self.random_rotate_dirs),
        )  # [D, 2]

        total_loss = pred_points.new_tensor(0.0)
        valid_count = 0

        for k_tensor in frame_ids:
            k = int(k_tensor.item())
            cur_light = light_dir[:, k, :]  # [B, 3]

            uv_pred = project_to_light_plane(pred_points, cur_light)  # [B, N, 2]
            uv_gt = project_to_light_plane(gt_points, cur_light)      # [B, M, 2]

            pred_support_pts, pred_support_values, _ = gather_support_points(uv_pred, dirs)
            gt_support_pts, gt_support_values, _ = gather_support_points(uv_gt, dirs)

            # 1. 支撑值一致：每个方向的最外轮廓范围一致
            if self.use_smooth_l1:
                loss_support = F.smooth_l1_loss(
                    pred_support_values,
                    gt_support_values.detach(),
                    reduction="mean",
                )
            else:
                loss_support = F.mse_loss(
                    pred_support_values,
                    gt_support_values.detach(),
                    reduction="mean",
                )

            # 2. 支撑点位置一致：允许局部轮廓点集匹配
            loss_chamfer = batch_chamfer_2d(
                pred_support_pts,
                gt_support_pts.detach(),
                squared=self.squared,
            )

            loss = (
                self.support_weight * loss_support
                + self.chamfer_weight * loss_chamfer
            )

            total_loss = total_loss + loss
            valid_count += 1

        if valid_count == 0:
            return pred_points.new_tensor(0.0)

        return total_loss / valid_count