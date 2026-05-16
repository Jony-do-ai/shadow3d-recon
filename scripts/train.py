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
from shadow3d.losses.proj_edge_loss import LightProjectionEdgeLoss
from shadow3d.models.shadow_point_baseline import ShadowPointBaseline
from shadow3d.losses.dense_projection_boundary_loss import DenseProjectionBoundaryLoss
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

def load_model_checkpoint(model, ckpt_path: str, device, strict: bool = False):
    if ckpt_path is None or str(ckpt_path).strip() == "":
        return

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    print(f"[INFO] Loaded coarse checkpoint: {ckpt_path}")
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


def freeze_coarse_model_for_stage2(model):
    """
    第二阶段只训练 phys_refiner。
    """
    for name, p in model.named_parameters():
        if name.startswith("phys_refiner"):
            p.requires_grad = True
        else:
            p.requires_grad = False

    print("[INFO] Stage2 freeze: only phys_refiner parameters are trainable.")


def set_stage2_train_mode(model):
    """
    model.train() 会把所有 BatchNorm 重新切回 train。
    第二阶段需要粗模型保持 eval，只让 phys_refiner train。
    """
    model.train()

    model.image_encoder.eval()
    model.light_encoder.eval()
    model.fusion.eval()
    model.decoder.eval()

    if getattr(model, "pct_refiner", None) is not None:
        model.pct_refiner.eval()

    if getattr(model, "phys_refiner", None) is not None:
        model.phys_refiner.train()

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

def init_train_log(log_path: str):
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
        "loss_proj_edge",
        "loss_p2g",
        "loss_g2p",

        "loss_dense_boundary",
        "loss_dense_p2g",
        "loss_dense_g2p",
        "loss_moved_3d",
        "loss_delta_reg",
        "delta_mean",
        "delta_max",

        "loss_center",
        "loss_bbox",
        "loss_hd",
        "loss_repulsion",

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
        "loss_proj_edge": float(stats.get("loss_proj_edge", 0.0)),
        "loss_p2g": float(stats["loss_p2g"]),
        "loss_g2p": float(stats["loss_g2p"]),

        "loss_dense_boundary": float(stats.get("loss_dense_boundary", 0.0)),
        "loss_dense_p2g": float(stats.get("loss_dense_p2g", 0.0)),
        "loss_dense_g2p": float(stats.get("loss_dense_g2p", 0.0)),
        "loss_moved_3d": float(stats.get("loss_moved_3d", 0.0)),
        "loss_delta_reg": float(stats.get("loss_delta_reg", 0.0)),
        "delta_mean": float(stats.get("delta_mean", 0.0)),
        "delta_max": float(stats.get("delta_max", 0.0)),

        "loss_center": float(stats["loss_center"]),
        "loss_bbox": float(stats["loss_bbox"]),
        "loss_hd": float(stats.get("loss_hd", 0.0)),
        "loss_repulsion": float(stats.get("loss_repulsion", 0.0)),

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

def train_one_epoch(model, loader, optimizer, device,
                    loss_cfg, epoch_idx, global_step,
                    log_every=10, save_ply_every=5, out_dir=None,
                    use_light=True,apml_criterion=None,proj_edge_loss_fn=None, proj_edge_weight=0.0,proj_edge_run_every_batch=1,

                    stage2_enabled=False,
                    dense_boundary_loss_fn=None,
                    dense_boundary_weight=0.0,
                    dense_boundary_run_every_batch=1,

                    moved_3d_enabled=False,
                    moved_3d_weight=0.0,
                    delta_reg_enabled=False,
                    delta_reg_weight=0.0,
                    ):

    if stage2_enabled:
        set_stage2_train_mode(model)
    else:
        model.train()

    running = {
        "loss_total": 0.0,
        "loss_cd": 0.0,
        "loss_apml": 0.0,
        "loss_proj_edge": 0.0,
        "loss_p2g": 0.0,
        "loss_g2p": 0.0,

        "loss_dense_boundary": 0.0,
        "loss_dense_p2g": 0.0,
        "loss_dense_g2p": 0.0,
        "loss_moved_3d": 0.0,
        "loss_delta_reg": 0.0,
        "delta_mean": 0.0,
        "delta_max": 0.0,

        "loss_center": 0.0,
        "loss_bbox": 0.0,
        "loss_hd": 0.0,
        "loss_repulsion": 0.0,

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

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        shadow_seq = batch["shadow_seq"].to(device)  # [B, K, 1, H, W]
        light_dir = batch["light_dir"].to(device)    # [B, K, 3]
        points_gt = batch["points_gt"].to(device)    # [B, N, 3]

        # 消融实验：不使用真实光线输入
        if not use_light:
            light_dir = torch.zeros_like(light_dir)

        if stage2_enabled:
            pred_points, coarse_points, delta_3d = model(
                shadow_seq,
                light_dir,
                return_coarse=True,
                return_delta=True,
            )
        else:
            pred_points = model(shadow_seq, light_dir)
            coarse_points = None
            delta_3d = None

        loss_dict = compute_point_loss_dict(
            pred_points=pred_points,
            points_gt=points_gt,
            loss_cfg=loss_cfg,
            apml_criterion=apml_criterion,
        )

        # 默认没有投影边界 loss
        loss_proj_edge = pred_points.new_tensor(0.0)

        use_proj_edge_this_batch = (
                proj_edge_loss_fn is not None
                and proj_edge_weight > 0.0
                and use_light
                and proj_edge_run_every_batch > 0
                and (batch_idx % proj_edge_run_every_batch == 0)
        )

        if use_proj_edge_this_batch:
            loss_proj_edge = proj_edge_loss_fn(
                pred_points=pred_points,
                gt_points=points_gt,
                light_dir=light_dir,
            )

            # 因为不是每个 batch 都算，所以这里乘 run_every_batch，
            # 让平均梯度强度大致接近“每 batch 都算”的情况。
            loss_dict["loss_total"] = (
                    loss_dict["loss_total"]
                    + proj_edge_weight * proj_edge_run_every_batch * loss_proj_edge
            )
        else:
            loss_dict["loss_total"] = loss_dict["loss_total"]

        loss_dict["loss_proj_edge"] = loss_proj_edge

        loss_dense_boundary = pred_points.new_tensor(0.0)
        loss_dense_p2g = pred_points.new_tensor(0.0)
        loss_dense_g2p = pred_points.new_tensor(0.0)
        loss_moved_3d = pred_points.new_tensor(0.0)

        use_dense_this_batch = (
            dense_boundary_loss_fn is not None
            and dense_boundary_weight > 0.0
            and use_light
            and dense_boundary_run_every_batch > 0
            and (batch_idx % dense_boundary_run_every_batch == 0)
        )

        if use_dense_this_batch:
            dense_out = dense_boundary_loss_fn(
                pred_points=pred_points,
                gt_points=points_gt,
                light_dir=light_dir,
                compute_moved_3d=bool(moved_3d_enabled),
            )

            loss_dense_boundary = dense_out["loss_boundary"]
            loss_dense_p2g = dense_out["loss_p2g"]
            loss_dense_g2p = dense_out["loss_g2p"]
            loss_moved_3d = dense_out["loss_moved_3d"]

            loss_dict["loss_total"] = (
                loss_dict["loss_total"]
                + dense_boundary_weight * dense_boundary_run_every_batch * loss_dense_boundary
            )

            if moved_3d_enabled and moved_3d_weight > 0.0:
                loss_dict["loss_total"] = (
                    loss_dict["loss_total"]
                    + moved_3d_weight * dense_boundary_run_every_batch * loss_moved_3d
                )

        loss_delta_reg = pred_points.new_tensor(0.0)
        delta_mean = pred_points.new_tensor(0.0)
        delta_max = pred_points.new_tensor(0.0)

        if stage2_enabled and delta_3d is not None:
            delta_norm = torch.norm(delta_3d, dim=-1)
            delta_mean = delta_norm.mean()
            delta_max = delta_norm.max()

            if delta_reg_enabled and delta_reg_weight > 0.0:
                loss_delta_reg = (delta_3d ** 2).sum(dim=-1).mean()
                loss_dict["loss_total"] = loss_dict["loss_total"] + delta_reg_weight * loss_delta_reg

        loss_dict["loss_dense_boundary"] = loss_dense_boundary
        loss_dict["loss_dense_p2g"] = loss_dense_p2g
        loss_dict["loss_dense_g2p"] = loss_dense_g2p
        loss_dict["loss_moved_3d"] = loss_moved_3d
        loss_dict["loss_delta_reg"] = loss_delta_reg
        loss_dict["delta_mean"] = delta_mean
        loss_dict["delta_max"] = delta_max

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
            avg_proj_edge = running["loss_proj_edge"] / (batch_idx + 1)
            avg_p2g = running["loss_p2g"] / (batch_idx + 1)
            avg_g2p = running["loss_g2p"] / (batch_idx + 1)
            avg_dense = running["loss_dense_boundary"] / (batch_idx + 1)
            avg_delta = running["delta_mean"] / (batch_idx + 1)
            avg_center = running["loss_center"] / (batch_idx + 1)
            avg_bbox = running["loss_bbox"] / (batch_idx + 1)
            avg_hd = running["loss_hd"] / (batch_idx + 1)
            avg_rep = running["loss_repulsion"] / (batch_idx + 1)
            avg_f002 = running["fscore_0_02"] / (batch_idx + 1)

            pbar.set_postfix(
                total=f"{avg_total:.4f}",
                cd=f"{avg_cd:.4f}",
                apml=f"{avg_apml:.4f}",
                pe=f"{avg_proj_edge:.4f}",
                p2g=f"{avg_p2g:.4f}",
                g2p=f"{avg_g2p:.4f}",
                db=f"{avg_dense:.4f}",
                dmean=f"{avg_delta:.4f}",
                hd=f"{avg_hd:.4f}",
                rep=f"{avg_rep:.4f}",
                # center=f"{avg_center:.4f}",
                # bbox=f"{avg_bbox:.4f}",
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
    stage2_enabled=False,
    save_stage2_coarse_every_time=True,
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

            if stage2_enabled:
                pred_points, coarse_points = model(
                    shadow_seq,
                    light_dir,
                    return_coarse=True,
                )
            else:
                pred_points = model(shadow_seq, light_dir)
                coarse_points = None

            # 沿用原来的 point_clouds 目录
            save_dir = os.path.join(out_dir, "point_clouds", seq_name)
            os.makedirs(save_dir, exist_ok=True)

            # GT 只保存一次
            gt_save_path = os.path.join(save_dir, "gt.ply")
            if not os.path.isfile(gt_save_path):
                save_point_cloud_ply(points_gt[0], gt_save_path)

            # 预测点云每 10 个 epoch 保存一次
            if stage2_enabled and coarse_points is not None:
                if save_stage2_coarse_every_time:
                    coarse_save_path = os.path.join(
                        save_dir,
                        f"epoch_{epoch_idx:04d}_coarse.ply"
                    )
                else:
                    coarse_save_path = os.path.join(save_dir, "coarse.ply")

                if save_stage2_coarse_every_time or not os.path.isfile(coarse_save_path):
                    save_point_cloud_ply(coarse_points[0], coarse_save_path)

                pred_save_path = os.path.join(
                    save_dir,
                    f"epoch_{epoch_idx:04d}_refined.ply"
                )
            else:
                pred_save_path = os.path.join(
                    save_dir,
                    f"epoch_{epoch_idx:04d}_pred.ply"
                )

            save_point_cloud_ply(pred_points[0], pred_save_path)

    print(f"[VIS] Saved fixed category predictions for epoch {epoch_idx}")

    if stage2_enabled:
        set_stage2_train_mode(model)
    else:
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_no_light.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ablation_cfg = cfg.get("ablation", {})
    model_cfg = cfg["model"]
    use_light = bool(ablation_cfg.get("use_light", True))
    use_pct_refiner = bool(model_cfg.get("use_pct_refiner", True))
    pct_use_condition = bool(model_cfg.get("pct_use_condition", True))
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
    dataset = ShadowSequenceDataset(
        root=data_cfg["root"],
        sequences_dir=data_cfg.get("sequences_dir", "dataset"),
        frames_per_seq=int(data_cfg.get("frames_per_seq", 10)),
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
    pct_qk_dim = model_cfg.get("pct_qk_dim", None)
    if pct_qk_dim is not None:
        pct_qk_dim = int(pct_qk_dim)

    model = ShadowPointBaseline(
        image_feat_dim=int(model_cfg.get("image_feat_dim", 256)),
        light_feat_dim=int(model_cfg.get("light_feat_dim", 128)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        num_points=int(model_cfg.get("num_points", 2048)),
        use_pct_refiner=bool(model_cfg.get("use_pct_refiner", False)),
        pct_hidden_dim=int(model_cfg.get("pct_hidden_dim", 128)),
        pct_coord_dim=int(model_cfg.get("pct_coord_dim", 64)),
        pct_shadow_dim=int(model_cfg.get("pct_shadow_dim", 128)),
        pct_blocks=int(model_cfg.get("pct_blocks", 4)),
        pct_knn_k=int(model_cfg.get("pct_knn_k", 16)),
        pct_delta_scale=float(model_cfg.get("pct_delta_scale", 0.05)),
        pct_qk_dim=pct_qk_dim,
        pct_use_condition=bool(model_cfg.get("pct_use_condition", True)),
        num_frames=int(model_cfg.get("num_frames", 10)),

        use_phys_refiner=bool(model_cfg.get("use_phys_refiner", False)),
        phys_hidden_dim=int(model_cfg.get("phys_hidden_dim", 128)),
        phys_global_context_dim=int(model_cfg.get("phys_global_context_dim", 64)),
        phys_light_context_dim=int(model_cfg.get("phys_light_context_dim", 32)),
        phys_delta_scale=float(model_cfg.get("phys_delta_scale", 0.02)),
        phys_fuse=str(model_cfg.get("phys_fuse", "mean")),
    ).to(device)

    # -------------------------
    # optim
    # -------------------------
    optim_cfg = cfg["optim"]

    stage2_cfg = cfg.get("stage2", {})
    stage2_enabled = bool(stage2_cfg.get("enabled", False))

    if stage2_enabled:
        coarse_ckpt = stage2_cfg.get("coarse_ckpt", "")
        strict_load = bool(stage2_cfg.get("strict_load", False))

        load_model_checkpoint(
            model=model,
            ckpt_path=coarse_ckpt,
            device=device,
            strict=strict_load,
        )

        if bool(stage2_cfg.get("freeze_coarse", True)):
            freeze_coarse_model_for_stage2(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable parameter tensors: {len(trainable_params)}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = int(optim_cfg.get("epochs", 50))

    # -------------------------
    # log / save
    # -------------------------
    log_cfg = cfg["log"]
    out_dir = log_cfg.get("out_dir", "data/train_runs/exp_default")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    ensure_dir(os.path.join(out_dir, "plots"))

    with open(os.path.join(out_dir, "config_dump.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_log_path = os.path.join(out_dir, "train_log.csv")
    init_train_log(train_log_path)
    print(f"[INFO] Train log will be saved to: {train_log_path}")

    # -------------------------
    # train
    # -------------------------
    loss_cfg = cfg["loss"]
    point_loss_type = str(loss_cfg.get("point_loss_type", "cd")).lower()

    proj_edge_cfg = loss_cfg.get("proj_edge", {})
    proj_edge_loss_fn = None
    proj_edge_weight = 0.0
    proj_edge_run_every_batch = 1

    if bool(proj_edge_cfg.get("enabled", False)):
        if not use_light:
            print("[WARN] proj_edge enabled but use_light=False, disable proj_edge loss.")
        else:
            proj_edge_weight = float(proj_edge_cfg.get("weight", 0.01))
            proj_edge_run_every_batch = int(proj_edge_cfg.get("run_every_batch", 1))

            proj_edge_loss_fn = LightProjectionEdgeLoss(
                num_dirs=int(proj_edge_cfg.get("num_dirs", 64)),
                squared=bool(proj_edge_cfg.get("squared", True)),
                max_frames=int(proj_edge_cfg.get("max_frames", 1)),
                frame_stride=int(proj_edge_cfg.get("frame_stride", 1)),
                random_frames=bool(proj_edge_cfg.get("random_frames", True)),
                random_rotate_dirs=bool(proj_edge_cfg.get("random_rotate_dirs", True)),
                support_weight=float(proj_edge_cfg.get("support_weight", 1.0)),
                chamfer_weight=float(proj_edge_cfg.get("chamfer_weight", 0.5)),
                use_smooth_l1=bool(proj_edge_cfg.get("use_smooth_l1", True)),
            ).to(device)

            print(
                f"[INFO] proj_edge enabled: "
                f"weight={proj_edge_weight}, "
                f"num_dirs={proj_edge_cfg.get('num_dirs', 64)}, "
                f"max_frames={proj_edge_cfg.get('max_frames', 1)}, "
                f"frame_stride={proj_edge_cfg.get('frame_stride', 1)}, "
                f"random_frames={proj_edge_cfg.get('random_frames', True)}, "
                f"random_rotate_dirs={proj_edge_cfg.get('random_rotate_dirs', True)}, "
                f"support_weight={proj_edge_cfg.get('support_weight', 1.0)}, "
                f"chamfer_weight={proj_edge_cfg.get('chamfer_weight', 0.5)}, "
                f"run_every_batch={proj_edge_run_every_batch}"
            )
    else:
        print("[INFO] proj_edge disabled")

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

    dense_cfg = loss_cfg.get("dense_boundary", {})
    dense_boundary_loss_fn = None
    dense_boundary_weight = 0.0
    dense_boundary_run_every_batch = 1

    if bool(dense_cfg.get("enabled", False)):
        if not use_light:
            print("[WARN] dense_boundary enabled but use_light=False, disable dense_boundary.")
        else:
            dense_boundary_weight = float(dense_cfg.get("weight", 0.01))
            dense_boundary_run_every_batch = int(dense_cfg.get("run_every_batch", 1))

            dense_boundary_loss_fn = DenseProjectionBoundaryLoss(
                grid_size=int(dense_cfg.get("grid_size", 64)),
                value_range=tuple(dense_cfg.get("value_range", [-1.5, 1.5])),
                splat_radius=int(dense_cfg.get("splat_radius", 1)),
                boundary_band=int(dense_cfg.get("boundary_band", 1)),
                max_boundary_points=int(dense_cfg.get("max_boundary_points", 512)),
                squared=bool(dense_cfg.get("squared", True)),
                p2g_weight=float(dense_cfg.get("p2g_weight", 0.5)),
                g2p_weight=float(dense_cfg.get("g2p_weight", 1.0)),
            ).to(device)

            print(
                f"[INFO] dense_boundary enabled: "
                f"weight={dense_boundary_weight}, "
                f"grid_size={dense_cfg.get('grid_size', 64)}, "
                f"value_range={dense_cfg.get('value_range', [-1.5, 1.5])}, "
                f"max_boundary_points={dense_cfg.get('max_boundary_points', 512)}, "
                f"run_every_batch={dense_boundary_run_every_batch}"
            )
    else:
        print("[INFO] dense_boundary disabled")

    moved_3d_cfg = loss_cfg.get("moved_3d", {})
    moved_3d_enabled = bool(moved_3d_cfg.get("enabled", False))
    moved_3d_weight = float(moved_3d_cfg.get("weight", 0.0))

    delta_reg_cfg = loss_cfg.get("delta_reg", {})
    delta_reg_enabled = bool(delta_reg_cfg.get("enabled", False))
    delta_reg_weight = float(delta_reg_cfg.get("weight", 0.0))


    global_step = 0
    best_loss = float("inf")

    # 创建点云保存目录
    pcd_out_dir = os.path.join(out_dir, "point_clouds")
    ensure_dir(pcd_out_dir)

    train_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
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
            save_ply_every=int(log_cfg.get("save_ply_every", 5)),
            out_dir=out_dir,
            use_light=use_light,
            apml_criterion=apml_criterion,
            proj_edge_loss_fn=proj_edge_loss_fn,
            proj_edge_weight=proj_edge_weight,
            proj_edge_run_every_batch=proj_edge_run_every_batch,
            stage2_enabled=stage2_enabled,
            dense_boundary_loss_fn=dense_boundary_loss_fn,
            dense_boundary_weight=dense_boundary_weight,
            dense_boundary_run_every_batch=dense_boundary_run_every_batch,
            moved_3d_enabled=moved_3d_enabled,
            moved_3d_weight=moved_3d_weight,
            delta_reg_enabled=delta_reg_enabled,
            delta_reg_weight=delta_reg_weight,
        )
        # 每个 epoch 都计算训练耗时
        epoch_time_sec = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed_sec = time.time() - train_start_time
        avg_epoch_sec = elapsed_sec / epoch
        remaining_epochs = num_epochs - epoch
        remaining_sec = avg_epoch_sec * remaining_epochs
        estimated_total_sec = avg_epoch_sec * num_epochs

        eta_time = datetime.now() + timedelta(seconds=remaining_sec)

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"total={stats['loss_total']:.6f}, "
            f"cd={stats['loss_cd']:.6f}, "
            f"apml={stats.get('loss_apml', 0.0):.6f}, "
            f"proj_edge={stats.get('loss_proj_edge', 0.0):.6f}, "
            f"p2g={stats['loss_p2g']:.6f}, "
            f"g2p={stats['loss_g2p']:.6f}, "
            f"f@0.02={stats['fscore_0_02']:.6f}, "
            f"dense={stats.get('loss_dense_boundary', 0.0):.6f}, "
            f"moved3d={stats.get('loss_moved_3d', 0.0):.6f}, "
            f"dreg={stats.get('loss_delta_reg', 0.0):.6f}, "
            f"dmean={stats.get('delta_mean', 0.0):.6f}, "
            f"center={stats['loss_center']:.6f}, "
            f"bbox={stats['loss_bbox']:.6f}, "
            f"hd={stats.get('loss_hd', 0.0):.6f}, "
            f"rep={stats.get('loss_repulsion', 0.0):.6f}, "
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
                stage2_enabled=stage2_enabled,
                save_stage2_coarse_every_time=bool(
                    log_cfg.get("save_stage2_coarse_every_time", True)
                ),
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