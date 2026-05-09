from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_square_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, 3]
    y: [B, M, 3]
    return: [B, N, M]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)              # [B, N, 1]
    yy = (y ** 2).sum(dim=-1).unsqueeze(1)               # [B, 1, M]
    xy = torch.bmm(x, y.transpose(1, 2))                 # [B, N, M]
    dist = xx + yy - 2.0 * xy
    return torch.clamp(dist, min=0.0)


def chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    fscore_taus=(0.01, 0.02, 0.05),
    hd_percentile: float = 90.0,
    hd_mode: str = "symmetric",
) -> Dict[str, torch.Tensor]:
    """
    pred: [B, N, 3]
    gt:   [B, M, 3]

    return:
        loss_cd:  CD-P2G + CD-G2P
        loss_p2g: pred -> gt
        loss_g2p: gt -> pred
    """
    dist = pairwise_square_distance(pred, gt)            # [B, N, M]

    min_pred_to_gt = dist.min(dim=2)[0]                  # [B, N]
    min_gt_to_pred = dist.min(dim=1)[0]                  # [B, M]

    loss_p2g = min_pred_to_gt.mean(dim=1).mean()
    loss_g2p = min_gt_to_pred.mean(dim=1).mean()
    loss_cd = loss_p2g + loss_g2p

    out = {
        "loss_cd": loss_cd,
        "loss_p2g": loss_p2g,
        "loss_g2p": loss_g2p,
    }
    hd_dict = robust_hausdorff_from_min_dist(
        min_pred_to_gt=min_pred_to_gt,
        min_gt_to_pred=min_gt_to_pred,
        hd_percentile=hd_percentile,
        mode=hd_mode,
    )
    out.update(hd_dict)

    for tau in fscore_taus:
        threshold = tau ** 2

        precision = (min_pred_to_gt < threshold).float().mean(dim=1)  # [B]
        recall = (min_gt_to_pred < threshold).float().mean(dim=1)  # [B]

        fscore = 2.0 * precision * recall / (precision + recall + 1e-8)

        tau_key = str(tau).replace(".", "_")
        out[f"precision_{tau_key}"] = precision.mean()
        out[f"recall_{tau_key}"] = recall.mean()
        out[f"fscore_{tau_key}"] = fscore.mean()

    return out

def robust_hausdorff_from_min_dist(
    min_pred_to_gt: torch.Tensor,
    min_gt_to_pred: torch.Tensor,
    hd_percentile: float = 90.0,
    mode: str = "symmetric",
) -> Dict[str, torch.Tensor]:
    """
    Robust Hausdorff / Percentile Hausdorff.

    min_pred_to_gt: [B, N]，每个预测点到最近 GT 点的平方距离
    min_gt_to_pred: [B, M]，每个 GT 点到最近预测点的平方距离

    hd_percentile:
        90.0 表示关注最差约 10% 的点
        95.0 表示关注最差约 5% 的点

    mode:
        symmetric: pred->gt 和 gt->pred 都用
        g2p:       只用 gt->pred，更关注 GT 有但 Pred 漏掉的区域
        p2g:       只用 pred->gt，更关注 Pred 多出来的错误点
    """
    hd_percentile = float(hd_percentile)
    hd_percentile = max(0.0, min(99.9, hd_percentile))

    # 90 percentile -> top 10%
    # 95 percentile -> top 5%
    top_ratio = (100.0 - hd_percentile) / 100.0
    top_ratio = max(top_ratio, 1e-6)

    k_p = max(1, int(round(min_pred_to_gt.shape[1] * top_ratio)))
    k_g = max(1, int(round(min_gt_to_pred.shape[1] * top_ratio)))

    hd_p2g = torch.topk(
        min_pred_to_gt,
        k=k_p,
        dim=1,
        largest=True,
    ).values.mean()

    hd_g2p = torch.topk(
        min_gt_to_pred,
        k=k_g,
        dim=1,
        largest=True,
    ).values.mean()

    if mode == "symmetric":
        hd_raw = hd_p2g + hd_g2p
    elif mode == "g2p":
        hd_raw = hd_g2p
    elif mode == "p2g":
        hd_raw = hd_p2g
    else:
        raise ValueError(f"Unknown hd mode: {mode}, expected symmetric/g2p/p2g")

    return {
        "loss_hd_raw": hd_raw,
        "loss_hd_p2g": hd_p2g,
        "loss_hd_g2p": hd_g2p,
    }

def center_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_center = pred.mean(dim=1)   # [B, 3]
    gt_center = gt.mean(dim=1)       # [B, 3]
    return F.mse_loss(pred_center, gt_center)


def bbox_regularization(pred: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    约束预测点云不要发散到过大空间
    """
    excess = torch.relu(torch.abs(pred) - radius)
    return excess.mean()


def point_recon_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_cd: float = 1.0,
    lambda_center: float = 0.1,
    lambda_bbox: float = 0.01,
    bbox_radius: float = 1.0,
    lambda_hd: float = 0.0,
    hd_percentile: float = 90.0,
    hd_mode: str = "symmetric",
) -> Dict[str, torch.Tensor]:
    cd_dict = chamfer_distance(
        pred,
        gt,
        hd_percentile=hd_percentile,
        hd_mode=hd_mode,
    )

    loss_cd = cd_dict["loss_cd"]
    loss_p2g = cd_dict["loss_p2g"]
    loss_g2p = cd_dict["loss_g2p"]

    loss_hd_raw = cd_dict["loss_hd_raw"]
    loss_hd_p2g = cd_dict["loss_hd_p2g"]
    loss_hd_g2p = cd_dict["loss_hd_g2p"]
    loss_hd = lambda_hd * loss_hd_raw

    loss_center = center_loss(pred, gt)
    loss_bbox = bbox_regularization(pred, radius=bbox_radius)

    total = (
            lambda_cd * loss_cd
            + lambda_center * loss_center
            + lambda_bbox * loss_bbox
            + loss_hd
    )

    return {
        "loss_total": total,
        "loss_cd": loss_cd,
        "loss_p2g": loss_p2g,
        "loss_g2p": loss_g2p,

        "loss_hd_raw": loss_hd_raw,
        "loss_hd": loss_hd,
        "loss_hd_p2g": loss_hd_p2g,
        "loss_hd_g2p": loss_hd_g2p,

        "loss_center": loss_center,
        "loss_bbox": loss_bbox,

        "precision_0_01": cd_dict["precision_0_01"],
        "recall_0_01": cd_dict["recall_0_01"],
        "fscore_0_01": cd_dict["fscore_0_01"],

        "precision_0_02": cd_dict["precision_0_02"],
        "recall_0_02": cd_dict["recall_0_02"],
        "fscore_0_02": cd_dict["fscore_0_02"],

        "precision_0_05": cd_dict["precision_0_05"],
        "recall_0_05": cd_dict["recall_0_05"],
        "fscore_0_05": cd_dict["fscore_0_05"],
    }