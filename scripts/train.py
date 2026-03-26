import argparse
import os
import random
import sys
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
from shadow3d.models.shadow_point_baseline import ShadowPointBaseline

import open3d as o3d
import numpy as np


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


def train_one_epoch(model, loader, optimizer, device, loss_cfg, epoch_idx, global_step, log_every=10, save_ply_every=5, out_dir=None):
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_cd": 0.0,
        "loss_center": 0.0,
        "loss_bbox": 0.0,
    }

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        shadow_seq = batch["shadow_seq"].to(device)  # [B, K, 1, H, W]
        light_dir = batch["light_dir"].to(device)    # [B, K, 3]
        points_gt = batch["points_gt"].to(device)    # [B, N, 3]

        pred_points = model(shadow_seq, light_dir)

        loss_dict = point_recon_loss(
            pred=pred_points,
            gt=points_gt,
            lambda_cd=loss_cfg.get("cd", 1.0),
            lambda_center=loss_cfg.get("center", 0.1),
            lambda_bbox=loss_cfg.get("bbox", 0.01),
            bbox_radius=loss_cfg.get("bbox_radius", 1.0),
        )

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
            avg_center = running["loss_center"] / (batch_idx + 1)
            avg_bbox = running["loss_bbox"] / (batch_idx + 1)
            pbar.set_postfix(
                total=f"{avg_total:.4f}",
                cd=f"{avg_cd:.4f}",
                center=f"{avg_center:.4f}",
                bbox=f"{avg_bbox:.4f}",
            )

        # 保存第一个batch的点云用于可视化（每save_ply_every个epoch）
        if batch_idx == 0 and out_dir and epoch_idx % save_ply_every == 0:
            with torch.no_grad():
                # 保存预测点云
                pred_save_path = os.path.join(out_dir, "point_clouds", f"epoch_{epoch_idx:04d}_pred.ply")
                save_point_cloud_ply(pred_points[0], pred_save_path)

                # 保存GT点云
                gt_save_path = os.path.join(out_dir, "point_clouds", f"epoch_{epoch_idx:04d}_gt.ply")
                save_point_cloud_ply(points_gt[0], gt_save_path)

                print(f"\n💾 Saved point clouds for epoch {epoch_idx}")

    num_batches = max(len(loader), 1)
    epoch_stats = {k: v / num_batches for k, v in running.items()}
    return global_step, epoch_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

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
        sequences_dir=data_cfg.get("sequences_dir", "sequences"),
        frames_per_seq=int(data_cfg.get("frames_per_seq", 5)),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        image_key=data_cfg.get("image_key", "shadow_mask.png"),
        num_points=int(cfg["model"].get("num_points", 2048)),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"[INFO] Dataset size: {len(dataset)} sequences")

    # -------------------------
    # model
    # -------------------------
    model_cfg = cfg["model"]
    model = ShadowPointBaseline(
        image_feat_dim=int(model_cfg.get("image_feat_dim", 256)),
        light_feat_dim=int(model_cfg.get("light_feat_dim", 128)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        num_points=int(model_cfg.get("num_points", 2048)),
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
    # log / save
    # -------------------------
    log_cfg = cfg["log"]
    out_dir = log_cfg.get("out_dir", "data/train_runs/exp_default")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))

    with open(os.path.join(out_dir, "config_dump.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # -------------------------
    # train
    # -------------------------
    loss_cfg = cfg["loss"]
    global_step = 0
    best_loss = float("inf")

    # 创建点云保存目录
    pcd_out_dir = os.path.join(out_dir, "point_clouds")
    ensure_dir(pcd_out_dir)

    for epoch in range(1, num_epochs + 1):
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
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"total={stats['loss_total']:.6f}, "
            f"cd={stats['loss_cd']:.6f}, "
            f"center={stats['loss_center']:.6f}, "
            f"bbox={stats['loss_bbox']:.6f}"
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