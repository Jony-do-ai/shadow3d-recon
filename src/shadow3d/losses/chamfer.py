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
) -> Dict[str, torch.Tensor]:
    cd_dict = chamfer_distance(pred, gt)
    loss_cd = cd_dict["loss_cd"]
    loss_p2g = cd_dict["loss_p2g"]
    loss_g2p = cd_dict["loss_g2p"]
    loss_center = center_loss(pred, gt)
    loss_bbox = bbox_regularization(pred, radius=bbox_radius)

    total = (
        lambda_cd * loss_cd
        + lambda_center * loss_center
        + lambda_bbox * loss_bbox
    )

    return {
        "loss_total": total,
        "loss_cd": loss_cd,
        "loss_p2g": loss_p2g,
        "loss_g2p": loss_g2p,
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