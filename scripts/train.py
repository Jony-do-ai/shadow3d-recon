import argparse
import os
import random
import sys
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让 scripts/train.py 可以找到 src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shadow3d.datasets.shadow_sequence_dataset import ShadowSequenceDataset
from shadow3d.losses.chamfer import point_recon_loss
from shadow3d.losses.apml_loss import APML
from shadow3d.losses.projection_splat import (
    multi_light_projection_loss,
    save_projection_debug,
)
from shadow3d.models.shadow_point_baseline import ShadowPointBaseline

import open3d as o3d


def save_point_cloud_ply(points: torch.Tensor, ply_path: str) -> None:
    """
    保存点云为PLY文件
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().float().numpy()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {points.shape}")

    if points.shape[0] == 0:
        print(f"⚠️  Empty point cloud, skipping {ply_path}")
        return

    # 确保目录存在
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    # 使用Open3D保存（更可靠）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 可选：添加颜色（根据高度）
    if points.shape[0] > 0:
        z = points[:, 2]
        z_normalized = (z - z.min()) / (z.max() - z.min() + 1e-8)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_normalized  # 红色通道
        colors[:, 2] = 1 - z_normalized  # 蓝色通道
        pcd.colors = o3d.utility.Vector3dVector(colors)

    success = o3d.io.write_point_cloud(ply_path, pcd)
    if success:
        print(f"✅ Saved point cloud to {ply_path}")
    else:
        print(f"❌ Failed to save point cloud to {ply_path}")


def visualize_and_save_pcd(points, save_path, title="Point Cloud"):
    """
    可视化并保存点云
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().float().numpy()

    if points.shape[0] == 0:
        print("Empty point cloud, cannot visualize")
        return

    # 保存
    save_point_cloud_ply(torch.from_numpy(points), save_path)

    # 可选：实时可视化（会阻塞训练）
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd], window_name=title)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, step, out_path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(ckpt, out_path)

def format_seconds(seconds: float) -> str:
    """
    将秒数格式化为 h m s，方便打印训练用时。
    """
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def init_train_log(log_path: str,resume: bool = False):
    """
    初始化 epoch 级训练日志。
    每一行记录一个 epoch 的平均 loss。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    header = [
        "epoch",
        "global_step",
        "lr",
        "loss_total",
        "loss_cd",
        "loss_apml",
        "loss_p2g",
        "loss_g2p",
        "loss_center",
        "loss_bbox",
        "loss_hd",
        "loss_repulsion",

        "loss_proj",
        "loss_proj_bce",
        "loss_proj_dice",
        "loss_proj_iou",
        "proj_weight",
        "proj_valid_ratio",
        "proj_pred_mask_mean",
        "proj_gt_mask_mean",
        "proj_pred_mask_max",
        "proj_gt_mask_max",
        "proj_u_min",
        "proj_u_max",
        "proj_v_min",
        "proj_v_max",

        "precision_0_01",
        "recall_0_01",
        "fscore_0_01",

        "precision_0_02",
        "recall_0_02",
        "fscore_0_02",

        "precision_0_05",
        "recall_0_05",
        "fscore_0_05",

        "epoch_time_sec",
    ]

    if resume and os.path.exists(log_path):
        print(f"[INFO] Resume mode: append to existing log {log_path}")
        return

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()


def append_train_log(
    log_path: str,
    epoch: int,
    global_step: int,
    lr: float,
    stats: dict,
    epoch_time_sec: float,
):
    """
    追加写入一行 epoch 训练结果。
    """
    row = {
        "epoch": epoch,
        "global_step": global_step,
        "lr": lr,
        "loss_total": float(stats["loss_total"]),
        "loss_cd": float(stats["loss_cd"]),
        "loss_apml": float(stats.get("loss_apml", 0.0)),
        "loss_p2g": float(stats["loss_p2g"]),
        "loss_g2p": float(stats["loss_g2p"]),
        "loss_center": float(stats["loss_center"]),
        "loss_bbox": float(stats["loss_bbox"]),
        "loss_hd": float(stats.get("loss_hd", 0.0)),
        "loss_repulsion": float(stats.get("loss_repulsion", 0.0)),

        "loss_proj": float(stats.get("loss_proj", 0.0)),
        "loss_proj_bce": float(stats.get("loss_proj_bce", 0.0)),
        "loss_proj_dice": float(stats.get("loss_proj_dice", 0.0)),
        "loss_proj_iou": float(stats.get("loss_proj_iou", 0.0)),
        "proj_weight": float(stats.get("proj_weight", 0.0)),
        "proj_valid_ratio": float(stats.get("proj_valid_ratio", 0.0)),
        "proj_pred_mask_mean": float(stats.get("proj_pred_mask_mean", 0.0)),
        "proj_gt_mask_mean": float(stats.get("proj_gt_mask_mean", 0.0)),
        "proj_pred_mask_max": float(stats.get("proj_pred_mask_max", 0.0)),
        "proj_gt_mask_max": float(stats.get("proj_gt_mask_max", 0.0)),
        "proj_u_min": float(stats.get("proj_u_min", 0.0)),
        "proj_u_max": float(stats.get("proj_u_max", 0.0)),
        "proj_v_min": float(stats.get("proj_v_min", 0.0)),
        "proj_v_max": float(stats.get("proj_v_max", 0.0)),

        "precision_0_01": float(stats["precision_0_01"]),
        "recall_0_01": float(stats["recall_0_01"]),
        "fscore_0_01": float(stats["fscore_0_01"]),

        "precision_0_02": float(stats["precision_0_02"]),
        "recall_0_02": float(stats["recall_0_02"]),
        "fscore_0_02": float(stats["fscore_0_02"]),

        "precision_0_05": float(stats["precision_0_05"]),
        "recall_0_05": float(stats["recall_0_05"]),
        "fscore_0_05": float(stats["fscore_0_05"]),

        "epoch_time_sec": float(epoch_time_sec),
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

def compute_point_loss_dict(
    pred_points: torch.Tensor,
    points_gt: torch.Tensor,
    loss_cfg: dict,
    apml_criterion=None,
):
    """
    统一计算点云 loss。

    point_loss_type = "cd":
        保持原来的 point_recon_loss，不改变训练逻辑。

    point_loss_type = "apml":
        用 APML 替代 CD 参与训练；
        但仍然调用 point_recon_loss 计算 loss_cd / p2g / g2p / fscore 等指标，
        方便和以前日志对比。
    """
    point_loss_type = str(loss_cfg.get("point_loss_type", "cd")).lower()

    if point_loss_type == "cd":
        loss_dict = point_recon_loss(
            pred=pred_points,
            gt=points_gt,
            lambda_cd=loss_cfg.get("cd", 1.0),
            lambda_center=loss_cfg.get("center", 0.1),
            lambda_bbox=loss_cfg.get("bbox", 0.01),
            bbox_radius=loss_cfg.get("bbox_radius", 1.0),
            lambda_hd=loss_cfg.get("hd", 0.0),
            hd_top_ratio=loss_cfg.get("hd_top_ratio", 0.1),
            lambda_repulsion=loss_cfg.get("repulsion", 0.0),
            repulsion_radius=loss_cfg.get("repulsion_radius", 0.03),
            repulsion_k=loss_cfg.get("repulsion_k", 16),
        )
        loss_dict["loss_apml"] = pred_points.new_tensor(0.0)
        return loss_dict

    if point_loss_type == "apml":
        if apml_criterion is None:
            raise RuntimeError("point_loss_type='apml' but apml_criterion is None.")

        # 1. APML 是真正参与反向传播的主点云 loss
        loss_apml = apml_criterion(pred_points, points_gt)
        apml_weight = float(loss_cfg.get("apml_weight", 1.0))

        # 2. 仍然计算 CD / P2G / G2P / F-score，用来做指标和日志
        #    这里 lambda_cd=0.0，意思是 CD 不参与训练总 loss。
        metric_dict = point_recon_loss(
            pred=pred_points,
            gt=points_gt,
            lambda_cd=0.0,
            lambda_center=loss_cfg.get("center", 0.0),
            lambda_bbox=loss_cfg.get("bbox", 0.0),
            bbox_radius=loss_cfg.get("bbox_radius", 1.0),
            lambda_hd=loss_cfg.get("hd", 0.0),
            hd_top_ratio=loss_cfg.get("hd_top_ratio", 0.1),
            lambda_repulsion=loss_cfg.get("repulsion", 0.0),
            repulsion_radius=loss_cfg.get("repulsion_radius", 0.03),
            repulsion_k=loss_cfg.get("repulsion_k", 16),
        )

        # metric_dict["loss_total"] 现在只包含 center / bbox / hd / repulsion 等非 CD 项
        # 最终训练 loss = APML + 原来的非 CD 正则项
        reg_loss = metric_dict["loss_total"]
        metric_dict["loss_apml"] = loss_apml
        metric_dict["loss_total"] = apml_weight * loss_apml + reg_loss

        return metric_dict

    raise ValueError(f"Unknown point_loss_type: {point_loss_type}")


def get_projection_cfg(loss_cfg: dict) -> dict:
    """
    兼容两种配置写法：
    loss:
      projection: {...}
    或：
      projection_loss: {...}
    """
    if "projection" in loss_cfg and isinstance(loss_cfg["projection"], dict):
        return loss_cfg["projection"]
    if "projection_loss" in loss_cfg and isinstance(loss_cfg["projection_loss"], dict):
        return loss_cfg["projection_loss"]
    return {}


def compute_projection_weight(proj_cfg: dict, epoch_idx: int, stage_epoch_idx: int = None) -> float:
    """
    计算 projection loss 当前 epoch 的权重。

    默认按 stage_epoch_idx 调度：
    - 从 epoch100 checkpoint 开第二阶段时，stage_epoch_idx=1 表示第二阶段第 1 个 epoch。
    - 因此配置 start_epoch: 6, ramp_epochs: 10 表示第二阶段第 6 个 epoch 开始逐步开启。

    如果想用绝对 epoch 编号，例如 epoch106 开启，可以设：
      schedule_by: absolute
      start_epoch: 106
    """
    if not bool(proj_cfg.get("enabled", False)):
        return 0.0

    base_weight = float(proj_cfg.get("weight", 0.0))
    if base_weight <= 0.0:
        return 0.0

    schedule_by = str(proj_cfg.get("schedule_by", "stage")).lower()
    schedule_epoch = epoch_idx if schedule_by == "absolute" else int(stage_epoch_idx or epoch_idx)

    start_epoch = int(proj_cfg.get("start_epoch", 1))
    ramp_epochs = int(proj_cfg.get("ramp_epochs", 0))

    if schedule_epoch < start_epoch:
        return 0.0

    if ramp_epochs <= 0:
        return base_weight

    progress = (schedule_epoch - start_epoch + 1) / float(ramp_epochs)
    progress = max(0.0, min(1.0, progress))
    return base_weight * progress


def zero_projection_stats(device: torch.device) -> dict:
    z = torch.tensor(0.0, device=device)
    return {
        "loss_proj": z,
        "loss_proj_bce": z,
        "loss_proj_dice": z,
        "loss_proj_iou": z,
        "proj_weight": z,
        "proj_valid_ratio": z,
        "proj_pred_mask_mean": z,
        "proj_gt_mask_mean": z,
        "proj_pred_mask_max": z,
        "proj_gt_mask_max": z,
        "proj_u_min": z,
        "proj_u_max": z,
        "proj_v_min": z,
        "proj_v_max": z,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_cfg,
    epoch_idx,
    global_step,
    log_every=10,
    save_ply_every=5,
    out_dir=None,
    use_light=True,
    apml_criterion=None,
    stage_epoch_idx=None,
):
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_cd": 0.0,
        "loss_apml": 0.0,
        "loss_p2g": 0.0,
        "loss_g2p": 0.0,
        "loss_center": 0.0,
        "loss_bbox": 0.0,
        "loss_hd": 0.0,
        "loss_repulsion": 0.0,

        "loss_proj": 0.0,
        "loss_proj_bce": 0.0,
        "loss_proj_dice": 0.0,
        "loss_proj_iou": 0.0,
        "proj_weight": 0.0,
        "proj_valid_ratio": 0.0,
        "proj_pred_mask_mean": 0.0,
        "proj_gt_mask_mean": 0.0,
        "proj_pred_mask_max": 0.0,
        "proj_gt_mask_max": 0.0,
        "proj_u_min": 0.0,
        "proj_u_max": 0.0,
        "proj_v_min": 0.0,
        "proj_v_max": 0.0,

        "precision_0_01": 0.0,
        "recall_0_01": 0.0,
        "fscore_0_01": 0.0,

        "precision_0_02": 0.0,
        "recall_0_02": 0.0,
        "fscore_0_02": 0.0,

        "precision_0_05": 0.0,
        "recall_0_05": 0.0,
        "fscore_0_05": 0.0,
    }

    proj_cfg = get_projection_cfg(loss_cfg)
    proj_debug_cfg = proj_cfg.get("debug", {}) if isinstance(proj_cfg.get("debug", {}), dict) else {}

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        shadow_seq = batch["shadow_seq"].to(device)  # [B, K, 1, H, W]
        raw_light_dir = batch["light_dir"].to(device)  # [B, K, 3]
        points_gt = batch["points_gt"].to(device)    # [B, N, 3]

        # 消融实验：不使用真实光线作为模型输入。
        # 注意：projection loss 默认仍使用 raw_light_dir 做几何投影，
        # 如果在 no-light 消融中不想引入光照监督，应关闭 projection.enabled。
        model_light_dir = raw_light_dir
        if not use_light:
            model_light_dir = torch.zeros_like(raw_light_dir)

        pred_points = model(shadow_seq, model_light_dir)

        loss_dict = compute_point_loss_dict(
            pred_points=pred_points,
            points_gt=points_gt,
            loss_cfg=loss_cfg,
            apml_criterion=apml_criterion,
        )

        proj_stats = zero_projection_stats(device)
        proj_weight = compute_projection_weight(
            proj_cfg=proj_cfg,
            epoch_idx=epoch_idx,
            stage_epoch_idx=stage_epoch_idx,
        )
        proj_stats["proj_weight"] = torch.tensor(float(proj_weight), device=device)

        if bool(proj_cfg.get("enabled", False)) and proj_weight > 0.0:
            # 默认用真实 light_dir 做投影。若显式设置 use_model_light_dir=true，则使用模型输入光照。
            proj_light_dir = model_light_dir if bool(proj_cfg.get("use_model_light_dir", False)) else raw_light_dir

            proj_dict = multi_light_projection_loss(
                pred_points=pred_points,
                gt_points=points_gt,
                light_dir=proj_light_dir,
                image_size=int(proj_cfg.get("render_size", proj_cfg.get("image_size", 64))),
                sigma=float(proj_cfg.get("sigma", 2.0)),
                projection_range=float(proj_cfg.get("projection_range", 1.2)),
                loss_type=str(proj_cfg.get("loss_type", "dice_bce")),
                bce_weight=float(proj_cfg.get("bce_weight", 0.5)),
                dice_weight=float(proj_cfg.get("dice_weight", 0.5)),
                iou_weight=float(proj_cfg.get("iou_weight", 0.5)),
                eps=float(proj_cfg.get("eps", 1e-6)),
                detach_gt=bool(proj_cfg.get("detach_gt", True)),
            )

            loss_dict["loss_total"] = loss_dict["loss_total"] + float(proj_weight) * proj_dict["loss_proj"]
            proj_stats.update({
                "loss_proj": proj_dict["loss_proj"].detach(),
                "loss_proj_bce": proj_dict["loss_proj_bce"],
                "loss_proj_dice": proj_dict["loss_proj_dice"],
                "loss_proj_iou": proj_dict["loss_proj_iou"],
                "proj_valid_ratio": proj_dict["proj_valid_ratio"],
                "proj_pred_mask_mean": proj_dict["proj_pred_mask_mean"],
                "proj_gt_mask_mean": proj_dict["proj_gt_mask_mean"],
                "proj_pred_mask_max": proj_dict["proj_pred_mask_max"],
                "proj_gt_mask_max": proj_dict["proj_gt_mask_max"],
                "proj_u_min": proj_dict["proj_u_min"],
                "proj_u_max": proj_dict["proj_u_max"],
                "proj_v_min": proj_dict["proj_v_min"],
                "proj_v_max": proj_dict["proj_v_max"],
            })

            if bool(proj_debug_cfg.get("enabled", False)) and out_dir is not None:
                save_every_iter = int(proj_debug_cfg.get("save_every_iter", 500))
                if save_every_iter > 0 and (global_step + 1) % save_every_iter == 0:
                    debug_dir = os.path.join(out_dir, "projection_debug", f"epoch_{epoch_idx:04d}_step_{global_step + 1:08d}")
                    save_projection_debug(
                        pred_mask=proj_dict["pred_mask"],
                        gt_mask=proj_dict["gt_mask"],
                        save_dir=debug_dir,
                        prefix=f"epoch_{epoch_idx:04d}_step_{global_step + 1:08d}",
                        max_samples=int(proj_debug_cfg.get("num_samples", 1)),
                        save_all_lights=bool(proj_debug_cfg.get("save_all_lights", True)),
                    )

        loss_dict.update(proj_stats)
        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in running.keys():
            running[k] += float(loss_dict[k].detach().cpu().item())

        global_step += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            avg_total = running["loss_total"] / (batch_idx + 1)
            avg_cd = running["loss_cd"] / (batch_idx + 1)
            avg_apml = running["loss_apml"] / (batch_idx + 1)
            avg_p2g = running["loss_p2g"] / (batch_idx + 1)
            avg_g2p = running["loss_g2p"] / (batch_idx + 1)
            avg_hd = running["loss_hd"] / (batch_idx + 1)
            avg_rep = running["loss_repulsion"] / (batch_idx + 1)
            avg_f002 = running["fscore_0_02"] / (batch_idx + 1)
            avg_proj = running["loss_proj"] / (batch_idx + 1)
            avg_proj_w = running["proj_weight"] / (batch_idx + 1)
            avg_valid = running["proj_valid_ratio"] / (batch_idx + 1)

            pbar.set_postfix(
                total=f"{avg_total:.4f}",
                cd=f"{avg_cd:.4f}",
                apml=f"{avg_apml:.4f}",
                proj=f"{avg_proj:.4f}",
                pw=f"{avg_proj_w:.1e}",
                valid=f"{avg_valid:.2f}",
                p2g=f"{avg_p2g:.4f}",
                g2p=f"{avg_g2p:.4f}",
                hd=f"{avg_hd:.4f}",
                rep=f"{avg_rep:.4f}",
                f002=f"{avg_f002:.4f}",
            )

    num_batches = max(len(loader), 1)
    epoch_stats = {k: v / num_batches for k, v in running.items()}
    return global_step, epoch_stats

def build_first_sample_per_category(dataset, max_categories=None):
    """
    找到每个类别的第一个有效样本，并缓存下来。
    这样后续每 10 个 epoch 都用同一批固定样本做收敛观察。
    """
    fixed_samples = []
    seen_categories = set()

    for idx, sample_meta in enumerate(dataset.samples):
        seq_name = sample_meta["seq_name"]

        # seq_name 格式: 类别ID_实例ID
        cat_id = seq_name.split("_", 1)[0]

        if cat_id in seen_categories:
            continue

        fixed_sample = dataset[idx]
        fixed_samples.append(fixed_sample)
        seen_categories.add(cat_id)

        if max_categories is not None and len(fixed_samples) >= max_categories:
            break

    return fixed_samples

def save_fixed_category_predictions(
    model,
    fixed_samples,
    device,
    epoch_idx,
    out_dir,
    use_light=True,
):
    """
    每隔若干 epoch，对每个类别的固定样本保存预测点云。
    GT 点云只保存一次。
    """
    model.eval()

    with torch.no_grad():
        for fixed_sample in fixed_samples:
            shadow_seq = fixed_sample["shadow_seq"].unsqueeze(0).to(device)  # [1, K, 1, H, W]
            light_dir = fixed_sample["light_dir"].unsqueeze(0).to(device)    # [1, K, 3]
            points_gt = fixed_sample["points_gt"].unsqueeze(0).to(device)    # [1, N, 3]
            seq_name = fixed_sample["seq_name"]

            if not use_light:
                light_dir = torch.zeros_like(light_dir)

            pred_points = model(shadow_seq, light_dir)

            # 沿用原来的 point_clouds 目录
            save_dir = os.path.join(out_dir, "point_clouds", seq_name)
            os.makedirs(save_dir, exist_ok=True)

            # GT 只保存一次
            gt_save_path = os.path.join(save_dir, "gt.ply")
            if not os.path.isfile(gt_save_path):
                save_point_cloud_ply(points_gt[0], gt_save_path)

            # 预测点云每 10 个 epoch 保存一次
            pred_save_path = os.path.join(
                save_dir,
                f"epoch_{epoch_idx:04d}_pred.ply"
            )
            save_point_cloud_ply(pred_points[0], pred_save_path)

    print(f"[VIS] Saved fixed category predictions for epoch {epoch_idx}")

    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_no_light.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .pt checkpoint. Can be normal resume or model-only stage2 initialization.",
    )
    parser.add_argument(
        "--model_only_resume",
        action="store_true",
        help="Load only checkpoint['model']; do not load optimizer state. Recommended for stage2 fine-tuning.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ablation_cfg = cfg.get("ablation", {})
    model_cfg = cfg["model"]
    use_light = bool(ablation_cfg.get("use_light", True))
    print(f"[INFO] use_light = {use_light}")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # -------------------------
    # data
    # -------------------------
    data_cfg = cfg["data"]

    # effective_num_frames: 真实读取几帧，用于帧数消融，1 / 3 / 5 / 10
    effective_num_frames = int(data_cfg.get("frames_per_seq", 10))

    # max_num_frames: 模型固定最大帧槽位数。为了公平消融，统一固定为 10。
    max_num_frames = int(model_cfg.get("num_frames", 10))

    if effective_num_frames > max_num_frames:
        raise ValueError(
            f"data.frames_per_seq ({effective_num_frames}) cannot be larger than "
            f"model.num_frames ({max_num_frames}). "
            f"For 1/3/5/10 ablation, set model.num_frames=10."
        )

    print(
        f"[INFO] effective frames_per_seq = {effective_num_frames}, "
        f"fixed model.num_frames = {max_num_frames}"
    )

    dataset = ShadowSequenceDataset(
        root=data_cfg["root"],
        sequences_dir=data_cfg.get("sequences_dir", "dataset"),
        frames_per_seq=effective_num_frames,
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        image_key=data_cfg.get("image_key", "shadow_mask.png"),
        num_points=int(cfg["model"].get("num_points", 2048)),
    )

    #每10个epoch就对每个类别第一个样本做收敛的可视化观察
    fixed_vis_max_categories = cfg.get("log", {}).get("fixed_vis_max_categories", None)
    if fixed_vis_max_categories is not None:
        fixed_vis_max_categories = int(fixed_vis_max_categories)

    fixed_category_samples = build_first_sample_per_category(
        dataset,
        max_categories=fixed_vis_max_categories,
    )

    print(f"[INFO] Fixed category visualization samples: {len(fixed_category_samples)}")
    for s in fixed_category_samples:
        print(f"  - {s['seq_name']}")


    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"[INFO] Dataset size: {len(dataset)} dataset")

    # -------------------------
    # model
    # -------------------------
    model_cfg = cfg["model"]

    model = ShadowPointBaseline(
        image_feat_dim=int(model_cfg.get("image_feat_dim", 256)),
        light_feat_dim=int(model_cfg.get("light_feat_dim", 128)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        num_points=int(model_cfg.get("num_points", 2048)),

        # 注意：这里传的是固定最大帧槽位数，不是真实读取帧数。
        # 例如 1/3/5/10 帧消融时，这里都固定为 10。
        num_frames=max_num_frames,

        use_refiner=bool(model_cfg.get("use_refiner", True)),
        refiner_hidden_dim=int(model_cfg.get("refiner_hidden_dim", 128)),
        refiner_blocks=int(model_cfg.get("refiner_blocks", 2)),
        refiner_num_heads=int(model_cfg.get("refiner_num_heads", 4)),
        refiner_delta_scale=float(model_cfg.get("refiner_delta_scale", 0.05)),
    ).to(device)

    # -------------------------
    # optim
    # -------------------------
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(optim_cfg.get("epochs", 50))

    # -------------------------
    # resume / stage2 init from checkpoint (if any)
    # -------------------------
    start_epoch = 0  # 已完成的 epoch 数；下一个 epoch = start_epoch + 1
    load_optimizer = bool(optim_cfg.get("load_optimizer", True)) and not args.model_only_resume
    keep_epoch_number = bool(optim_cfg.get("keep_epoch_number", True))

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        print(f"[INFO] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])

        if load_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[INFO] Loaded optimizer state from checkpoint.")
        else:
            print("[INFO] Model-only resume: optimizer is newly initialized from current config.")

        if keep_epoch_number:
            start_epoch = int(ckpt.get("step", 0))
        else:
            start_epoch = 0

        print(f"[INFO] Checkpoint epoch={ckpt.get('step', 'unknown')}, "
              f"start_epoch={start_epoch}, will train from epoch {start_epoch + 1} to {num_epochs}")
        if start_epoch >= num_epochs:
            print(f"[WARN] start_epoch ({start_epoch}) >= num_epochs ({num_epochs}), "
                  f"nothing to train.")


    # -------------------------
    # log / save
    # -------------------------
    log_cfg = cfg["log"]
    out_dir = log_cfg.get("out_dir", "data/train_runs/exp_default")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "plots"))

    # resume 模式下不覆盖原 config_dump.yaml，保留原始训练记录
    config_dump_path = os.path.join(out_dir, "config_dump.yaml")
    if args.resume is None or not os.path.exists(config_dump_path):
        with open(config_dump_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_log_path = os.path.join(out_dir, "train_log.csv")
    init_train_log(
        train_log_path,
        resume=(args.resume is not None and bool(log_cfg.get("append_on_resume", load_optimizer))),
    )
    print(f"[INFO] Train log will be saved to: {train_log_path}")

    # -------------------------
    # train
    # -------------------------
    loss_cfg = cfg["loss"]
    point_loss_type = str(loss_cfg.get("point_loss_type", "cd")).lower()

    apml_criterion = None
    if point_loss_type == "apml":
        apml_criterion = None
        if point_loss_type == "apml":
            apml_criterion = APML(
                min_softmax_value=float(loss_cfg.get("apml_p_min", 0.8)),
            ).to(device)

            print(
                f"[INFO] point_loss_type = APML, "
                f"min_softmax_value={loss_cfg.get('apml_p_min', 0.8)}"
            )
        else:
            print("[INFO] point_loss_type = CD")
        print(
            f"[INFO] point_loss_type = APML, "
            f"p_min={loss_cfg.get('apml_p_min', 0.8)}, "
            f"sinkhorn_iters={loss_cfg.get('apml_sinkhorn_iters', 20)}"
        )
    else:
        print("[INFO] point_loss_type = CD")

    global_step = 0
    best_loss = float("inf")

    # 创建点云保存目录
    pcd_out_dir = os.path.join(out_dir, "point_clouds")
    ensure_dir(pcd_out_dir)

    train_start_time = time.time()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start_time = time.time()

        global_step, stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
            epoch_idx=epoch,
            global_step=global_step,
            log_every=int(log_cfg.get("print_every", 10)),
            save_ply_every=int(log_cfg.get("save_ply_every", 5)),  # 每5个epoch保存一次
            out_dir=out_dir,
            use_light=use_light,
            apml_criterion=apml_criterion,
            stage_epoch_idx=epoch - start_epoch,
        )
        # 每个 epoch 都计算训练耗时
        epoch_time_sec = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed_sec = time.time() - train_start_time
        completed_this_run = max(epoch - start_epoch, 1)
        avg_epoch_sec = elapsed_sec / completed_this_run
        remaining_epochs = num_epochs - epoch
        remaining_sec = avg_epoch_sec * remaining_epochs
        estimated_total_sec = avg_epoch_sec * num_epochs

        eta_time = datetime.now() + timedelta(seconds=remaining_sec)

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"total={stats['loss_total']:.6f}, "
            f"cd={stats['loss_cd']:.6f}, "
            f"apml={stats.get('loss_apml', 0.0):.6f}, "
            f"p2g={stats['loss_p2g']:.6f}, "
            f"g2p={stats['loss_g2p']:.6f}, "
            f"f@0.02={stats['fscore_0_02']:.6f}, "
            f"center={stats['loss_center']:.6f}, "
            f"bbox={stats['loss_bbox']:.6f}, "
            f"hd={stats.get('loss_hd', 0.0):.6f}, "
            f"rep={stats.get('loss_repulsion', 0.0):.6f}, "
            f"proj={stats.get('loss_proj', 0.0):.6f}, "
            f"proj_w={stats.get('proj_weight', 0.0):.2e}, "
            f"proj_valid={stats.get('proj_valid_ratio', 0.0):.3f}, "
            f"epoch_time={format_seconds(epoch_time_sec)}, "
            f"elapsed={format_seconds(elapsed_sec)}, "
            f"eta={format_seconds(remaining_sec)}, "
            f"total_est={format_seconds(estimated_total_sec)}, "
            f"finish≈{eta_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # 每个 epoch 都写入日志
        append_train_log(
            log_path=train_log_path,
            epoch=epoch,
            global_step=global_step,
            lr=current_lr,
            stats=stats,
            epoch_time_sec=epoch_time_sec,
        )

        fixed_vis_every = int(log_cfg.get("fixed_vis_every_epoch", 10))

        if epoch % fixed_vis_every == 0:
            save_fixed_category_predictions(
                model=model,
                fixed_samples=fixed_category_samples,
                device=device,
                epoch_idx=epoch,
                out_dir=out_dir,
                use_light=use_light,
            )

        ckpt_latest = os.path.join(out_dir, "checkpoints", "latest.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_latest)

        if stats["loss_total"] < best_loss:
            best_loss = stats["loss_total"]
            ckpt_best = os.path.join(out_dir, "checkpoints", "best.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_best)

        save_every = int(log_cfg.get("save_every_epoch", 10))
        if epoch % save_every == 0:
            ckpt_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()