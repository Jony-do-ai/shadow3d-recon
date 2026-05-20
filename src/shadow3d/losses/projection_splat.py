from typing import Dict, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def _to_uint8(mask: torch.Tensor) -> np.ndarray:
    """
    mask: [H, W], value range roughly [0, 1]
    return uint8 image array.
    """
    arr = mask.detach().float().cpu().clamp(0.0, 1.0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def build_light_projection_basis(
    light_dir: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal 2D projection basis perpendicular to light_dir.

    Args:
        light_dir: [B, K, 3], light direction vectors.
        eps: numerical epsilon.

    Returns:
        u_axis: [B, K, 3]
        v_axis: [B, K, 3]
        d_axis: [B, K, 3], normalized light direction.
    """
    if light_dir.ndim != 3 or light_dir.shape[-1] != 3:
        raise ValueError(f"Expected light_dir shape [B, K, 3], got {tuple(light_dir.shape)}")

    # If a zero light vector is accidentally passed, replace it with +Z to avoid NaNs.
    # This should not happen in the real multi-light projection experiment.
    norm = torch.linalg.norm(light_dir, dim=-1, keepdim=True)
    fallback = torch.zeros_like(light_dir)
    fallback[..., 2] = 1.0
    light_dir_safe = torch.where(norm > eps, light_dir, fallback)
    d_axis = F.normalize(light_dir_safe, dim=-1, eps=eps)

    ref_z = torch.zeros_like(d_axis)
    ref_z[..., 2] = 1.0
    ref_y = torch.zeros_like(d_axis)
    ref_y[..., 1] = 1.0

    # Avoid cross product degeneracy when d_axis is nearly parallel to +Z/-Z.
    use_ref_y = (torch.abs((d_axis * ref_z).sum(dim=-1, keepdim=True)) > 0.9)
    ref = torch.where(use_ref_y, ref_y, ref_z)

    u_axis = torch.cross(ref, d_axis, dim=-1)
    u_axis = F.normalize(u_axis, dim=-1, eps=eps)
    v_axis = torch.cross(d_axis, u_axis, dim=-1)
    v_axis = F.normalize(v_axis, dim=-1, eps=eps)
    return u_axis, v_axis, d_axis


def project_points_to_light_plane(
    points: torch.Tensor,
    light_dir: torch.Tensor,
    image_size: int = 64,
    projection_range: float = 1.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Orthographically project 3D points to each light-view plane.

    Args:
        points: [B, N, 3], usually normalized to unit sphere.
        light_dir: [B, K, 3].
        image_size: output H=W.
        projection_range: coordinates in [-projection_range, projection_range]
            map to image range [0, image_size-1].

    Returns:
        x_pix: [B, K, N]
        y_pix: [B, K, N]
        diag: scalar diagnostics.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape [B, N, 3], got {tuple(points.shape)}")

    b, n, _ = points.shape
    bk, k, _ = light_dir.shape
    if bk != b:
        raise ValueError(f"Batch mismatch: points B={b}, light_dir B={bk}")

    u_axis, v_axis, _ = build_light_projection_basis(light_dir, eps=eps)

    # [B, 1, N, 3] dot [B, K, 1, 3] -> [B, K, N]
    p = points.unsqueeze(1)
    u_coord = (p * u_axis.unsqueeze(2)).sum(dim=-1)
    v_coord = (p * v_axis.unsqueeze(2)).sum(dim=-1)

    scale = (float(image_size) - 1.0) / (2.0 * float(projection_range))
    x_pix = (u_coord + float(projection_range)) * scale
    y_pix = (v_coord + float(projection_range)) * scale

    valid_point = (
        (x_pix >= 0.0) & (x_pix <= float(image_size - 1)) &
        (y_pix >= 0.0) & (y_pix <= float(image_size - 1))
    )

    diag = {
        "valid_ratio": valid_point.float().mean(),
        "u_min": x_pix.detach().amin(),
        "u_max": x_pix.detach().amax(),
        "v_min": y_pix.detach().amin(),
        "v_max": y_pix.detach().amax(),
    }
    return x_pix, y_pix, diag


def splat_render_points(
    points: torch.Tensor,
    light_dir: torch.Tensor,
    image_size: int = 64,
    sigma: float = 2.0,
    projection_range: float = 1.2,
    kernel_radius: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Differentiably render point clouds to multi-light soft occupancy maps.

    Args:
        points: [B, N, 3].
        light_dir: [B, K, 3].
        image_size: H=W of rendered mask.
        sigma: Gaussian splat sigma in pixels.
        projection_range: projected coordinates in [-range, range] map to image.
        kernel_radius: local splat radius. Default ceil(3*sigma).

    Returns:
        mask: [B, K, H, W], soft occupancy in [0, 1].
        diag: diagnostics.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    b, n, _ = points.shape
    _, k, _ = light_dir.shape
    h = w = int(image_size)
    radius = int(kernel_radius if kernel_radius is not None else max(1, int(np.ceil(3.0 * float(sigma)))))

    x_pix, y_pix, proj_diag = project_points_to_light_plane(
        points=points,
        light_dir=light_dir,
        image_size=h,
        projection_range=projection_range,
        eps=eps,
    )

    device = points.device
    dtype = points.dtype

    center_x = torch.round(x_pix).long()
    center_y = torch.round(y_pix).long()

    offsets = torch.arange(-radius, radius + 1, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(1, 1, 1, -1)
    oy = oy.reshape(1, 1, 1, -1)

    x_idx = center_x.unsqueeze(-1) + ox
    y_idx = center_y.unsqueeze(-1) + oy

    dx = x_pix.unsqueeze(-1) - x_idx.to(dtype)
    dy = y_pix.unsqueeze(-1) - y_idx.to(dtype)
    weights = torch.exp(-(dx * dx + dy * dy) / (2.0 * float(sigma) * float(sigma)))

    valid_pix = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
    weights = weights * valid_pix.to(dtype)

    # Flatten index: [B, K, N, S] -> [B*K*H*W]
    batch_ids = torch.arange(b, device=device).view(b, 1, 1, 1)
    light_ids = torch.arange(k, device=device).view(1, k, 1, 1)
    base = (batch_ids * k + light_ids) * (h * w)

    x_safe = x_idx.clamp(0, w - 1)
    y_safe = y_idx.clamp(0, h - 1)
    flat_idx = base + y_safe * w + x_safe

    acc = points.new_zeros(b * k * h * w)
    acc.scatter_add_(0, flat_idx.reshape(-1), weights.reshape(-1))
    acc = acc.view(b, k, h, w)

    # Soft occupancy without hard clipping. Large accumulated density smoothly saturates to 1.
    mask = 1.0 - torch.exp(-acc)
    mask = mask.clamp(min=0.0, max=1.0)

    diag = {
        **proj_diag,
        "mask_mean": mask.detach().mean(),
        "mask_max": mask.detach().amax(),
    }
    return mask, diag


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_f = target.reshape(target.shape[0], target.shape[1], -1)
    inter = (pred_f * target_f).sum(dim=-1)
    denom = pred_f.sum(dim=-1) + target_f.sum(dim=-1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_f = target.reshape(target.shape[0], target.shape[1], -1)
    inter = (pred_f * target_f).sum(dim=-1)
    union = pred_f.sum(dim=-1) + target_f.sum(dim=-1) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def mask_bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(min=eps, max=1.0 - eps)
    return F.binary_cross_entropy(pred, target)


def multi_light_projection_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    light_dir: torch.Tensor,
    image_size: int = 64,
    sigma: float = 2.0,
    projection_range: float = 1.2,
    loss_type: str = "dice_bce",
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    iou_weight: float = 0.5,
    eps: float = 1e-6,
    detach_gt: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Multi-light differentiable splat projection loss.

    First recommended target:
        SplatRender(pred_points, light_i) vs SplatRender(gt_points, light_i)

    Args:
        pred_points: [B, N, 3]
        gt_points: [B, M, 3]
        light_dir: [B, K, 3]
        loss_type: "bce", "dice", "iou", "dice_bce", "iou_bce".

    Returns:
        dict with loss and diagnostics.
    """
    pred_mask, pred_diag = splat_render_points(
        pred_points,
        light_dir,
        image_size=image_size,
        sigma=sigma,
        projection_range=projection_range,
        eps=eps,
    )
    gt_mask, gt_diag = splat_render_points(
        gt_points,
        light_dir,
        image_size=image_size,
        sigma=sigma,
        projection_range=projection_range,
        eps=eps,
    )
    if detach_gt:
        gt_mask = gt_mask.detach()

    loss_type = str(loss_type).lower()
    loss_bce = mask_bce_loss(pred_mask, gt_mask, eps=eps)
    loss_dice = soft_dice_loss(pred_mask, gt_mask, eps=eps)
    loss_iou = soft_iou_loss(pred_mask, gt_mask, eps=eps)

    if loss_type == "bce":
        loss_total = loss_bce
    elif loss_type == "dice":
        loss_total = loss_dice
    elif loss_type in ("iou", "soft_iou"):
        loss_total = loss_iou
    elif loss_type in ("dice_bce", "bce_dice"):
        loss_total = float(bce_weight) * loss_bce + float(dice_weight) * loss_dice
    elif loss_type in ("iou_bce", "bce_iou"):
        loss_total = float(bce_weight) * loss_bce + float(iou_weight) * loss_iou
    else:
        raise ValueError(f"Unknown projection loss_type: {loss_type}")

    return {
        "loss_proj": loss_total,
        "loss_proj_bce": loss_bce.detach(),
        "loss_proj_dice": loss_dice.detach(),
        "loss_proj_iou": loss_iou.detach(),
        "pred_mask": pred_mask.detach(),
        "gt_mask": gt_mask.detach(),
        "proj_valid_ratio": pred_diag["valid_ratio"].detach(),
        "proj_pred_mask_mean": pred_diag["mask_mean"].detach(),
        "proj_gt_mask_mean": gt_diag["mask_mean"].detach(),
        "proj_pred_mask_max": pred_diag["mask_max"].detach(),
        "proj_gt_mask_max": gt_diag["mask_max"].detach(),
        "proj_u_min": pred_diag["u_min"].detach(),
        "proj_u_max": pred_diag["u_max"].detach(),
        "proj_v_min": pred_diag["v_min"].detach(),
        "proj_v_max": pred_diag["v_max"].detach(),
    }


@torch.no_grad()
def save_projection_debug(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    save_dir: str,
    prefix: str = "debug",
    max_samples: int = 1,
    save_all_lights: bool = True,
) -> None:
    """
    Save pred / gt / diff / overlay masks for renderer diagnosis.

    Args:
        pred_mask: [B, K, H, W]
        gt_mask: [B, K, H, W]
        save_dir: output directory.
        prefix: file prefix, e.g. epoch/global_step.
    """
    os.makedirs(save_dir, exist_ok=True)
    b, k, h, w = pred_mask.shape
    num_samples = min(int(max_samples), b)
    light_indices = range(k) if save_all_lights else range(min(1, k))

    for bi in range(num_samples):
        sample_dir = os.path.join(save_dir, f"sample_{bi:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        for li in light_indices:
            pred = pred_mask[bi, li]
            gt = gt_mask[bi, li]
            diff = (pred - gt).abs().clamp(0.0, 1.0)

            pred_u8 = _to_uint8(pred)
            gt_u8 = _to_uint8(gt)
            diff_u8 = _to_uint8(diff)

            Image.fromarray(pred_u8).save(os.path.join(sample_dir, f"{prefix}_light_{li:02d}_pred.png"))
            Image.fromarray(gt_u8).save(os.path.join(sample_dir, f"{prefix}_light_{li:02d}_gt.png"))
            Image.fromarray(diff_u8).save(os.path.join(sample_dir, f"{prefix}_light_{li:02d}_diff.png"))

            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[..., 0] = pred_u8  # red: pred
            overlay[..., 1] = gt_u8    # green: gt
            overlay[..., 2] = diff_u8  # blue: absolute difference
            Image.fromarray(overlay).save(os.path.join(sample_dir, f"{prefix}_light_{li:02d}_overlay.png"))
