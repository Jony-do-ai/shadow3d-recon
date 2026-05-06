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


def get_rng_state() -> dict:
    """保存随机数状态，尽量让断点续跑更稳定。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """恢复随机数状态。旧 checkpoint 没有 rng_state 时会自动跳过。"""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def torch_load_checkpoint(path: str, device: torch.device) -> dict:
    """兼容不同 PyTorch 版本的 torch.load。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_epoch(ckpt: dict) -> int:
    """兼容新版 epoch 字段和旧版 step 字段。"""
    return int(ckpt.get("epoch", ckpt.get("step", 0)))


def backup_latest_checkpoint(checkpoints_dir: str) -> None:
    """保存 latest.pt 的上一版为 previous.pt，便于最新 checkpoint 出问题时回退。"""
    latest_path = os.path.join(checkpoints_dir, "latest.pt")
    previous_path = os.path.join(checkpoints_dir, "previous.pt")
    if os.path.isfile(latest_path):
        try:
            import shutil
            shutil.copy2(latest_path, previous_path)
        except Exception as e:
            print(f"[WARN] Failed to backup latest.pt to previous.pt: {e}")


def save_checkpoint(model, optimizer, epoch, global_step, best_loss, out_path, cfg=None):
    """保存可续跑 checkpoint。

    使用临时文件 + os.replace，避免直接写 latest.pt 时断电导致文件半损坏。
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "rng_state": get_rng_state(),
        "config": cfg,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, out_path)


def list_loadable_checkpoints(checkpoints_dir: str, device: torch.device):
    """返回所有可读取 checkpoint，自动跳过损坏文件。"""
    if not os.path.isdir(checkpoints_dir):
        return []

    paths = []
    latest_path = os.path.join(checkpoints_dir, "latest.pt")
    if os.path.isfile(latest_path):
        paths.append(latest_path)

    previous_path = os.path.join(checkpoints_dir, "previous.pt")
    if os.path.isfile(previous_path):
        paths.append(previous_path)

    for name in sorted(os.listdir(checkpoints_dir)):
        if name.startswith("epoch_") and name.endswith(".pt"):
            paths.append(os.path.join(checkpoints_dir, name))

    best_path = os.path.join(checkpoints_dir, "best.pt")
    if os.path.isfile(best_path):
        paths.append(best_path)

    # 去重但保留顺序
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    loadable = []
    for path in unique_paths:
        try:
            ckpt = torch_load_checkpoint(path, device)
            ep = checkpoint_epoch(ckpt)
            loadable.append({"path": path, "epoch": ep, "ckpt": ckpt})
        except Exception as e:
            print(f"[RESUME][WARN] Skip broken checkpoint: {path} ({e})")

    loadable.sort(key=lambda x: x["epoch"])
    return loadable


def get_last_logged_epoch(log_path: str):
    """读取 train_log.csv 中最后记录到的 epoch。"""
    if not os.path.isfile(log_path):
        return None

    last_epoch = None
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                last_epoch = int(row["epoch"])
            except Exception:
                continue
    return last_epoch


def truncate_train_log_from_epoch(log_path: str, start_epoch: int) -> None:
    """删除 start_epoch 及之后的不可靠日志。

    例如 70 epoch 断了，要求从 69 重跑：
        start_epoch = 69
    那么日志里 epoch >= 69 的记录都会被删除，后面重新训练时会重新写入。
    """
    if not os.path.isfile(log_path):
        return

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            try:
                ep = int(row["epoch"])
            except Exception:
                continue
            if ep < start_epoch:
                rows.append(row)

    if fieldnames is None:
        return

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[RESUME] Truncated train log: removed epoch >= {start_epoch}")


def get_log_state_before_epoch(log_path: str, start_epoch: int):
    """从保留下来的日志中恢复 global_step 和 best_loss。"""
    global_step = 0
    best_loss = float("inf")

    if not os.path.isfile(log_path):
        return global_step, best_loss

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except Exception:
                continue
            if ep >= start_epoch:
                continue

            try:
                global_step = max(global_step, int(float(row.get("global_step", 0))))
            except Exception:
                pass

            try:
                best_loss = min(best_loss, float(row["loss_total"]))
            except Exception:
                pass

    return global_step, best_loss


def cleanup_checkpoints_from_epoch(checkpoints_dir: str, start_epoch: int, device: torch.device) -> None:
    """删除 start_epoch 及之后的 checkpoint，保证后续重跑会覆盖它们。"""
    if not os.path.isdir(checkpoints_dir):
        return

    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isfile(path) or not name.endswith(".pt"):
            continue

        remove = False

        if name.startswith("epoch_"):
            try:
                ep = int(name.replace("epoch_", "").replace(".pt", ""))
                remove = ep >= start_epoch
            except Exception:
                remove = False
        elif name in {"latest.pt", "previous.pt", "best.pt"}:
            try:
                ckpt = torch_load_checkpoint(path, device)
                ep = checkpoint_epoch(ckpt)
                remove = ep >= start_epoch
            except Exception:
                # 损坏的 latest/best 也删掉，避免下一次继续误读
                remove = True

        if remove:
            try:
                os.remove(path)
                print(f"[RESUME] Removed stale checkpoint: {path}")
            except FileNotFoundError:
                pass


def cleanup_prediction_plys_from_epoch(out_dir: str, start_epoch: int) -> None:
    """删除 start_epoch 及之后的可视化预测点云，避免旧图混在新实验里。"""
    pcd_root = os.path.join(out_dir, "point_clouds")
    if not os.path.isdir(pcd_root):
        return

    for root, _, files in os.walk(pcd_root):
        for name in files:
            if not (name.startswith("epoch_") and name.endswith("_pred.ply")):
                continue
            try:
                ep = int(name.split("_")[1])
            except Exception:
                continue
            if ep >= start_epoch:
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    print(f"[RESUME] Removed stale prediction: {path}")
                except FileNotFoundError:
                    pass


def resolve_resume_state(
    resume_arg: str,
    checkpoints_dir: str,
    train_log_path: str,
    device: torch.device,
    restart_epoch=None,
    rollback_epochs: int = 1,
):
    """决定从哪个 checkpoint 加载，以及从哪个 epoch 重新训练。

    默认保守策略：
        如果最后痕迹显示跑到/断在 70，rollback_epochs=1，
        则 start_epoch = 69，日志和 checkpoint 中 >=69 的内容会被清理，
        后续重新训练会覆盖 69pt 和 70pt。
    """
    if resume_arg is None:
        return None, 1, 0, float("inf")

    loadable = list_loadable_checkpoints(checkpoints_dir, device)
    if len(loadable) == 0:
        raise FileNotFoundError(f"No loadable checkpoints found under: {checkpoints_dir}")

    if resume_arg != "auto":
        ckpt = torch_load_checkpoint(resume_arg, device)
        source = {"path": resume_arg, "epoch": checkpoint_epoch(ckpt), "ckpt": ckpt}
        last_ckpt_epoch = source["epoch"]
    else:
        last_ckpt_epoch = max(x["epoch"] for x in loadable)
        source = max(loadable, key=lambda x: x["epoch"])

    last_log_epoch = get_last_logged_epoch(train_log_path)
    known_last_epoch = max([x for x in [last_ckpt_epoch, last_log_epoch] if x is not None])

    if restart_epoch is not None:
        start_epoch = int(restart_epoch)
    else:
        rollback_epochs = max(int(rollback_epochs), 0)
        start_epoch = max(1, known_last_epoch - rollback_epochs)

    # 不加载 start_epoch 之后的 checkpoint，避免 70pt 虽然能读但内容不可信。
    # 最严谨：如果要重跑 epoch 69，优先加载 epoch 68 的状态；
    # 兼容你的使用习惯：如果没有 epoch 68，但有 epoch 69/latest=69，则加载 69 并重跑 69。
    strict_candidates = [x for x in loadable if x["epoch"] == start_epoch - 1]
    same_epoch_candidates = [x for x in loadable if x["epoch"] == start_epoch]

    if len(strict_candidates) > 0:
        source = max(strict_candidates, key=lambda x: x["epoch"])
        source_mode = "strict_previous_epoch"
    elif len(same_epoch_candidates) > 0:
        source = max(same_epoch_candidates, key=lambda x: x["epoch"])
        source_mode = "same_epoch_fallback"
        print(
            f"[RESUME][WARN] No checkpoint from epoch {start_epoch - 1}. "
            f"Will load epoch {start_epoch} and rerun epoch {start_epoch}."
        )
    else:
        raise RuntimeError(
            f"Cannot restart from epoch {start_epoch}: need checkpoint epoch {start_epoch - 1} "
            f"or epoch {start_epoch}. Available epochs: {[x['epoch'] for x in loadable]}"
        )

    print(f"[RESUME] Known last epoch from checkpoint/log: {known_last_epoch}")
    print(f"[RESUME] Restart training from epoch: {start_epoch}")
    print(f"[RESUME] Load checkpoint: {source['path']} (epoch={source['epoch']}, mode={source_mode})")

    ckpt = source["ckpt"]
    global_step, best_loss = get_log_state_before_epoch(train_log_path, start_epoch)
    if global_step == 0:
        global_step = int(ckpt.get("global_step", 0))
    if best_loss == float("inf"):
        best_loss = float(ckpt.get("best_loss", float("inf")))

    return ckpt, start_epoch, global_step, best_loss

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

def init_train_log(log_path: str, overwrite: bool = True):
    """
    初始化 epoch 级训练日志。
    overwrite=True 表示新实验，重新写表头；
    overwrite=False 表示续跑，已有日志不清空。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if (not overwrite) and os.path.isfile(log_path):
        print(f"[RESUME] Keep existing train log: {log_path}")
        return

    header = [
        "epoch",
        "global_step",
        "lr",
        "loss_total",
        "loss_cd",
        "loss_p2g",
        "loss_g2p",
        "loss_center",
        "loss_bbox",

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
        "loss_p2g": float(stats["loss_p2g"]),
        "loss_g2p": float(stats["loss_g2p"]),
        "loss_center": float(stats["loss_center"]),
        "loss_bbox": float(stats["loss_bbox"]),

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

def train_one_epoch(model, loader, optimizer, device, loss_cfg, epoch_idx, global_step, log_every=10, save_ply_every=5, out_dir=None,use_light=True,):
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_cd": 0.0,
        "loss_p2g": 0.0,
        "loss_g2p": 0.0,
        "loss_center": 0.0,
        "loss_bbox": 0.0,

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
            avg_p2g = running["loss_p2g"] / (batch_idx + 1)
            avg_g2p = running["loss_g2p"] / (batch_idx + 1)
            avg_center = running["loss_center"] / (batch_idx + 1)
            avg_bbox = running["loss_bbox"] / (batch_idx + 1)
            avg_f002 = running["fscore_0_02"] / (batch_idx + 1)

            pbar.set_postfix(
                total=f"{avg_total:.4f}",
                cd=f"{avg_cd:.4f}",
                p2g=f"{avg_p2g:.4f}",
                g2p=f"{avg_g2p:.4f}",
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
        help="Use 'auto' to resume from checkpoints automatically, or pass a checkpoint path.",
    )
    parser.add_argument(
        "--resume-start-epoch",
        type=int,
        default=None,
        help="Force restart from this epoch. Example: 70 interrupted -> set 69.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resume_cfg = cfg.get("resume", {})
    resume_arg = args.resume
    if resume_arg is None and bool(resume_cfg.get("auto_resume", False)):
        resume_arg = "auto"

    restart_epoch = args.resume_start_epoch
    if restart_epoch is None:
        restart_epoch = resume_cfg.get("restart_epoch", None)

    # 默认回退 1 个 epoch：如果最后痕迹显示 70，则从 69 重跑。
    rollback_epochs = int(resume_cfg.get("rollback_epochs", 1))

    ablation_cfg = cfg.get("ablation", {})
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
    dataset = ShadowSequenceDataset(
        root=data_cfg["root"],
        sequences_dir=data_cfg.get("sequences_dir", "dataset"),
        frames_per_seq=int(data_cfg.get("frames_per_seq", 10)),
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        image_key=data_cfg.get("image_key", "shadow_mask.png"),
        num_points=int(cfg["model"].get("num_points", 2048)),
        frame_sample_mode=data_cfg.get("frame_sample_mode", "uniform"),
        frame_order=data_cfg.get("frame_order", "natural"),
        frame_shuffle_seed=int(data_cfg.get("frame_shuffle_seed", 42)),
    )

    #每10个epoch就对每个类别第一个样本做收敛的可视化观察
    fixed_category_samples = build_first_sample_per_category(
        dataset,
        max_categories=None,
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
        temporal_module=model_cfg.get("temporal_module", "mean_max"),
        temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", 3)),
        temporal_dilations=model_cfg.get("temporal_dilations", [1, 2, 4]),
        temporal_dropout=float(model_cfg.get("temporal_dropout", 0.1)),
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
    ensure_dir(os.path.join(out_dir, "plots"))

    with open(os.path.join(out_dir, "config_dump.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    train_log_path = os.path.join(out_dir, "train_log.csv")
    is_resuming = resume_arg is not None
    init_train_log(train_log_path, overwrite=not is_resuming)
    print(f"[INFO] Train log will be saved to: {train_log_path}")

    # -------------------------
    # train / resume
    # -------------------------
    loss_cfg = cfg["loss"]
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if is_resuming:
        ckpt, start_epoch, global_step, best_loss = resolve_resume_state(
            resume_arg=resume_arg,
            checkpoints_dir=os.path.join(out_dir, "checkpoints"),
            train_log_path=train_log_path,
            device=device,
            restart_epoch=restart_epoch,
            rollback_epochs=rollback_epochs,
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt.get("rng_state", None))

        # 清理 start_epoch 及之后的不可靠结果，后续训练会重新覆盖 69pt/70pt 等文件。
        truncate_train_log_from_epoch(train_log_path, start_epoch)
        cleanup_checkpoints_from_epoch(os.path.join(out_dir, "checkpoints"), start_epoch, device)
        cleanup_prediction_plys_from_epoch(out_dir, start_epoch)

    if start_epoch > num_epochs:
        print(f"[INFO] Nothing to train: start_epoch={start_epoch}, num_epochs={num_epochs}")
        return

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
            save_ply_every=int(log_cfg.get("save_ply_every", 5)),  # 每5个epoch保存一次
            out_dir=out_dir,
            use_light=use_light,
        )
        # 每个 epoch 都计算训练耗时
        epoch_time_sec = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed_sec = time.time() - train_start_time
        finished_epochs_this_run = epoch - start_epoch + 1
        avg_epoch_sec = elapsed_sec / max(finished_epochs_this_run, 1)
        remaining_epochs = num_epochs - epoch
        remaining_sec = avg_epoch_sec * remaining_epochs
        estimated_total_sec = avg_epoch_sec * (num_epochs - start_epoch + 1)

        eta_time = datetime.now() + timedelta(seconds=remaining_sec)

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"total={stats['loss_total']:.6f}, "
            f"cd={stats['loss_cd']:.6f}, "
            f"p2g={stats['loss_p2g']:.6f}, "
            f"g2p={stats['loss_g2p']:.6f}, "
            f"f@0.02={stats['fscore_0_02']:.6f}, "
            f"center={stats['loss_center']:.6f}, "
            f"bbox={stats['loss_bbox']:.6f}, "
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

        checkpoints_dir = os.path.join(out_dir, "checkpoints")
        backup_latest_checkpoint(checkpoints_dir)
        ckpt_latest = os.path.join(checkpoints_dir, "latest.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            out_path=ckpt_latest,
            cfg=cfg,
        )

        if stats["loss_total"] < best_loss:
            best_loss = stats["loss_total"]
            ckpt_best = os.path.join(out_dir, "checkpoints", "best.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                out_path=ckpt_best,
                cfg=cfg,
            )

        # 为了支持“70 断了从 69 重跑并覆盖 69pt/70pt”，默认每个 epoch 都保存 epoch_XXXX.pt。
        # 如果想减少磁盘占用，可以在 log 里设置 save_epoch_checkpoint_every_epoch: false。
        save_every = int(log_cfg.get("save_every_epoch", 10))
        save_epoch_checkpoint_every_epoch = bool(log_cfg.get("save_epoch_checkpoint_every_epoch", True))
        if save_epoch_checkpoint_every_epoch or epoch % save_every == 0:
            ckpt_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                out_path=ckpt_path,
                cfg=cfg,
            )

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()