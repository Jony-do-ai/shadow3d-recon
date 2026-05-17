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


def load_checkpoint(model, optimizer, ckpt_path, device, load_optimizer: bool = True):
    """
    从 ckpt 恢复 model 和 optimizer。
    返回 ckpt 里保存的 epoch（即上次训练结束时已完成的 epoch 号）。
    新训练应该从 last_epoch + 1 开始。
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume ckpt not found: {ckpt_path}")

    print(f"[RESUME] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容 strict=False，防止后续模型有小改动时直接挂掉
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[RESUME][WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[RESUME][WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if load_optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[RESUME] Optimizer state restored.")
        except Exception as e:
            print(f"[RESUME][WARN] Failed to load optimizer state: {e}. Using fresh optimizer.")
    else:
        print(f"[RESUME] Skipping optimizer state (load_optimizer={load_optimizer}).")

    last_epoch = int(ckpt.get("step", 0))
    print(f"[RESUME] Last completed epoch in ckpt: {last_epoch}. Training will continue from epoch {last_epoch + 1}.")
    return last_epoch

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
        "loss_center",
        "loss_bbox",
        "loss_hd",
        "loss_repulsion",

        # === proj_edge 诊断 ===
        "proj_loss_raw",
        "proj_loss_cd_2d",
        "proj_loss_grid",
        "proj_loss_p2g",
        "proj_loss_g2p",
        "proj_w_mean",
        "proj_w_active_ratio",
        "proj_uv_inside_ratio",
        "proj_grad_ratio",   # ||grad_proj|| / ||grad_cd||

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
        "loss_center": float(stats["loss_center"]),
        "loss_bbox": float(stats["loss_bbox"]),
        "loss_hd": float(stats.get("loss_hd", 0.0)),
        "loss_repulsion": float(stats.get("loss_repulsion", 0.0)),

        # === proj_edge 诊断 ===
        "proj_loss_raw": float(stats.get("proj_loss_raw", 0.0)),
        "proj_loss_cd_2d": float(stats.get("proj_loss_cd_2d", 0.0)),
        "proj_loss_grid": float(stats.get("proj_loss_grid", 0.0)),
        "proj_loss_p2g": float(stats.get("proj_loss_p2g", 0.0)),
        "proj_loss_g2p": float(stats.get("proj_loss_g2p", 0.0)),
        "proj_w_mean": float(stats.get("proj_w_mean", 0.0)),
        "proj_w_active_ratio": float(stats.get("proj_w_active_ratio", 0.0)),
        "proj_uv_inside_ratio": float(stats.get("proj_uv_inside_ratio", 0.0)),
        "proj_grad_ratio": float(stats.get("proj_grad_ratio", 0.0)),

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
                    proj_edge_start_epoch=1, proj_edge_warmup_epochs=0,):
    model.train()

    # === 计算本 epoch 的 effective_weight（支持 start_epoch + 线性 warmup） ===
    if epoch_idx < proj_edge_start_epoch:
        proj_edge_effective_weight = 0.0
    elif proj_edge_warmup_epochs > 0 and epoch_idx < proj_edge_start_epoch + proj_edge_warmup_epochs:
        # 线性 ramp：start_epoch 时 progress=0（仍为 0），start_epoch+warmup 时 progress=1
        progress = (epoch_idx - proj_edge_start_epoch + 1) / float(proj_edge_warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        proj_edge_effective_weight = proj_edge_weight * progress
    else:
        proj_edge_effective_weight = proj_edge_weight

    if proj_edge_loss_fn is not None and proj_edge_weight > 0.0:
        print(
            f"[INFO] epoch {epoch_idx}: proj_edge effective_weight = "
            f"{proj_edge_effective_weight:.6f} (raw={proj_edge_weight}, "
            f"start={proj_edge_start_epoch}, warmup={proj_edge_warmup_epochs})"
        )

    running = {
        "loss_total": 0.0,
        "loss_cd": 0.0,
        "loss_apml": 0.0,
        "loss_proj_edge": 0.0,
        "loss_p2g": 0.0,
        "loss_g2p": 0.0,
        "loss_center": 0.0,
        "loss_bbox": 0.0,
        "loss_hd": 0.0,
        "loss_repulsion": 0.0,

        # === proj_edge 诊断 ===
        "proj_loss_raw": 0.0,
        "proj_loss_cd_2d": 0.0,
        "proj_loss_grid": 0.0,
        "proj_loss_p2g": 0.0,
        "proj_loss_g2p": 0.0,
        "proj_w_mean": 0.0,
        "proj_w_active_ratio": 0.0,
        "proj_uv_inside_ratio": 0.0,
        "proj_grad_ratio": 0.0,
        "_proj_count": 0.0,  # 实际跑了 proj_edge 的 batch 数，最后做除法

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

        pred_points = model(shadow_seq, light_dir)

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
                and proj_edge_effective_weight > 0.0
                and use_light
                and proj_edge_run_every_batch > 0
                and (batch_idx % proj_edge_run_every_batch == 0)
        )

        if use_proj_edge_this_batch:
            # 拿到 raw loss 和细分 stats
            loss_proj_edge, proj_stats = proj_edge_loss_fn(
                pred_points=pred_points,
                gt_points=points_gt,
                light_dir=light_dir,
                return_stats=True,
            )

            # === 梯度诊断：分别看 cd 主项和 proj_edge 项对 pred_points 的梯度量级 ===
            # 注意：这里调用 autograd.grad 不释放图（retain_graph=True），
            # 也不参与反传（只读梯度做对比）。
            try:
                cd_grad_src = loss_dict.get("loss_cd", None)
                if cd_grad_src is not None and cd_grad_src.requires_grad:
                    g_cd = torch.autograd.grad(
                        cd_grad_src, pred_points,
                        retain_graph=True, create_graph=False, allow_unused=True,
                    )[0]
                    g_pe = torch.autograd.grad(
                        loss_proj_edge, pred_points,
                        retain_graph=True, create_graph=False, allow_unused=True,
                    )[0]
                    if g_cd is not None and g_pe is not None:
                        n_cd = float(g_cd.detach().norm().item())
                        n_pe = float(g_pe.detach().norm().item())
                        # 用 effective_weight 反映实际参与优化的强度
                        n_pe_weighted = n_pe * float(proj_edge_effective_weight)
                        proj_stats["proj_grad_ratio"] = (
                            n_pe_weighted / max(n_cd, 1e-12)
                        )
                    else:
                        proj_stats["proj_grad_ratio"] = 0.0
                else:
                    proj_stats["proj_grad_ratio"] = 0.0
            except Exception as e:
                # 防御性：grad 诊断失败不阻断训练
                proj_stats["proj_grad_ratio"] = 0.0

            # 用 effective_weight 加权进入总 loss
            loss_dict["loss_total"] = (
                    loss_dict["loss_total"]
                    + proj_edge_effective_weight * loss_proj_edge
            )

            # 累加诊断 stats（只在跑了 proj_edge 的 batch 上累加）
            for sk in ["proj_loss_raw", "proj_loss_cd_2d", "proj_loss_grid",
                       "proj_loss_p2g", "proj_loss_g2p", "proj_w_mean",
                       "proj_w_active_ratio", "proj_uv_inside_ratio",
                       "proj_grad_ratio"]:
                running[sk] += float(proj_stats.get(sk, 0.0))
            running["_proj_count"] += 1.0
        else:
            loss_dict["loss_total"] = loss_dict["loss_total"]

        loss_dict["loss_proj_edge"] = loss_proj_edge
        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 注意：proj_* 字段已在 use_proj_edge_this_batch 分支里手动累加，
        # 这里只累加非诊断字段。
        _proj_keys = {"proj_loss_raw", "proj_loss_cd_2d", "proj_loss_grid",
                      "proj_loss_p2g", "proj_loss_g2p", "proj_w_mean",
                      "proj_w_active_ratio", "proj_uv_inside_ratio",
                      "proj_grad_ratio", "_proj_count"}
        for k in running.keys():
            if k in _proj_keys:
                continue
            running[k] += float(loss_dict[k].detach().cpu().item())

        global_step += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            avg_total = running["loss_total"] / (batch_idx + 1)
            avg_cd = running["loss_cd"] / (batch_idx + 1)
            avg_apml = running["loss_apml"] / (batch_idx + 1)
            avg_proj_edge = running["loss_proj_edge"] / (batch_idx + 1)
            avg_p2g = running["loss_p2g"] / (batch_idx + 1)
            avg_g2p = running["loss_g2p"] / (batch_idx + 1)
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
                hd=f"{avg_hd:.4f}",
                rep=f"{avg_rep:.4f}",
                # center=f"{avg_center:.4f}",
                # bbox=f"{avg_bbox:.4f}",
                f002=f"{avg_f002:.4f}",
            )

    num_batches = max(len(loader), 1)
    proj_count = max(running.pop("_proj_count"), 1.0)
    _proj_keys = {"proj_loss_raw", "proj_loss_cd_2d", "proj_loss_grid",
                  "proj_loss_p2g", "proj_loss_g2p", "proj_w_mean",
                  "proj_w_active_ratio", "proj_uv_inside_ratio",
                  "proj_grad_ratio"}
    epoch_stats = {}
    for k, v in running.items():
        if k in _proj_keys:
            epoch_stats[k] = v / proj_count
        else:
            epoch_stats[k] = v / num_batches
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
    # resume
    # -------------------------
    resume_from = optim_cfg.get("resume_from", None)
    resume_load_optimizer = bool(optim_cfg.get("resume_load_optimizer", True))
    start_epoch = 1  # 默认从 epoch 1 开始

    if resume_from is not None and str(resume_from).strip() != "":
        last_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            ckpt_path=str(resume_from),
            device=device,
            load_optimizer=resume_load_optimizer,
        )
        start_epoch = last_epoch + 1

        if start_epoch > num_epochs:
            raise ValueError(
                f"Resume failed: ckpt last_epoch={last_epoch}, but optim.epochs={num_epochs}. "
                f"Nothing to train."
            )
        print(f"[RESUME] Training from epoch {start_epoch} to {num_epochs}.")
    else:
        print(f"[INFO] No resume_from set, training from scratch (epoch 1 to {num_epochs}).")

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
    # resume 时不重新初始化日志，避免覆盖之前的 epoch 记录
    if start_epoch == 1 or not os.path.isfile(train_log_path):
        init_train_log(train_log_path)
        print(f"[INFO] Train log will be saved to: {train_log_path}")
    else:
        print(f"[INFO] Appending to existing train log: {train_log_path}")

    # -------------------------
    # train
    # -------------------------
    loss_cfg = cfg["loss"]
    point_loss_type = str(loss_cfg.get("point_loss_type", "cd")).lower()

    proj_edge_cfg = loss_cfg.get("proj_edge", {})
    proj_edge_loss_fn = None
    proj_edge_weight = 0.0
    proj_edge_run_every_batch = 1
    proj_edge_start_epoch = 1
    proj_edge_warmup_epochs = 0

    if bool(proj_edge_cfg.get("enabled", False)):
        if not use_light:
            print("[WARN] proj_edge enabled but use_light=False, disable proj_edge loss.")
        else:
            proj_edge_weight = float(proj_edge_cfg.get("weight", 0.01))
            proj_edge_run_every_batch = int(proj_edge_cfg.get("run_every_batch", 1))
            proj_edge_start_epoch = int(proj_edge_cfg.get("start_epoch", 1))
            proj_edge_warmup_epochs = int(proj_edge_cfg.get("warmup_epochs", 0))

            proj_edge_loss_fn = LightProjectionEdgeLoss(
                # 兼容旧配置：num_dirs 仍可用；新逻辑里它等价于 grid_size。
                num_dirs=int(proj_edge_cfg.get("num_dirs", 64)),
                grid_size=int(proj_edge_cfg.get("grid_size", proj_edge_cfg.get("num_dirs", 64))),
                grid_padding=float(proj_edge_cfg.get("grid_padding", 0.05)),
                grid_sigma=float(proj_edge_cfg.get("grid_sigma", 1.0)),
                grid_chunk_size=int(proj_edge_cfg.get("grid_chunk_size", 1024)),
                grid_pos_weight=float(proj_edge_cfg.get("grid_pos_weight", 4.0)),
                use_grid_loss=bool(proj_edge_cfg.get("use_grid_loss", True)),

                squared=bool(proj_edge_cfg.get("squared", True)),
                max_frames=int(proj_edge_cfg.get("max_frames", 1)),
                frame_stride=int(proj_edge_cfg.get("frame_stride", 1)),
                random_frames=bool(proj_edge_cfg.get("random_frames", True)),
                random_rotate_dirs=bool(proj_edge_cfg.get("random_rotate_dirs", True)),
                support_weight=float(proj_edge_cfg.get("support_weight", 1.0)),
                chamfer_weight=float(proj_edge_cfg.get("chamfer_weight", 0.0)),
                use_smooth_l1=bool(proj_edge_cfg.get("use_smooth_l1", True)),
            ).to(device)

            print(
                f"[INFO] proj_edge grid enabled: "
                f"weight={proj_edge_weight}, "
                f"start_epoch={proj_edge_start_epoch}, "
                f"warmup_epochs={proj_edge_warmup_epochs}, "
                f"grid_size={proj_edge_cfg.get('grid_size', proj_edge_cfg.get('num_dirs', 64))}, "
                f"use_grid_loss={proj_edge_cfg.get('use_grid_loss', True)}, "
                f"grid_padding={proj_edge_cfg.get('grid_padding', 0.05)}, "
                f"grid_sigma={proj_edge_cfg.get('grid_sigma', 1.0)}, "
                f"grid_chunk_size={proj_edge_cfg.get('grid_chunk_size', 1024)}, "
                f"grid_pos_weight={proj_edge_cfg.get('grid_pos_weight', 4.0)}, "
                f"max_frames={proj_edge_cfg.get('max_frames', 1)}, "
                f"frame_stride={proj_edge_cfg.get('frame_stride', 1)}, "
                f"random_frames={proj_edge_cfg.get('random_frames', True)}, "
                f"support_weight={proj_edge_cfg.get('support_weight', 1.0)}, "
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

    global_step = 0
    best_loss = float("inf")

    # 创建点云保存目录
    pcd_out_dir = os.path.join(out_dir, "point_clouds")
    ensure_dir(pcd_out_dir)

    train_start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
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
            proj_edge_start_epoch=proj_edge_start_epoch,
            proj_edge_warmup_epochs=proj_edge_warmup_epochs,
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
            f"center={stats['loss_center']:.6f}, "
            f"bbox={stats['loss_bbox']:.6f}, "
            f"hd={stats.get('loss_hd', 0.0):.6f}, "
            f"rep={stats.get('loss_repulsion', 0.0):.6f}, "
            # === proj_edge 诊断 ===
            f"[pe_raw={stats.get('proj_loss_raw', 0.0):.6f} "
            f"pe_cd2d={stats.get('proj_loss_cd_2d', 0.0):.6f} "
            f"pe_grid={stats.get('proj_loss_grid', 0.0):.6f} "
            f"pe_p2g={stats.get('proj_loss_p2g', 0.0):.6f} "
            f"pe_g2p={stats.get('proj_loss_g2p', 0.0):.6f} "
            f"w_mean={stats.get('proj_w_mean', 0.0):.4f} "
            f"w_act={stats.get('proj_w_active_ratio', 0.0):.3f} "
            f"uv_in={stats.get('proj_uv_inside_ratio', 0.0):.3f} "
            f"grad_ratio={stats.get('proj_grad_ratio', 0.0):.4f}], "
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