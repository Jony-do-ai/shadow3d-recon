import os
import sys
import json
import argparse
from typing import Any, Dict

import torch
import yaml

# 让 scripts/ 下运行时也能找到 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from shadow3d.datasets.shadow_sequence_dataset import ShadowSequenceDataset
from shadow3d.models.shadow_point_baseline import ShadowPointBaseline


def load_config(config_path: str) -> Dict[str, Any]:
    """
    读取 yaml 配置文件
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_point_cloud_ply(points: torch.Tensor, ply_path: str) -> None:
    """
    points: [N, 3] torch.Tensor or numpy-compatible
    保存为 ASCII PLY，便于直接查看
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().float().numpy()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {points.shape}")

    ensure_dir(os.path.dirname(ply_path) or ".")

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> ShadowPointBaseline:
    """
    按配置构建模型。

    固定 10-slot 帧数消融逻辑：
    - data.frames_per_seq 表示真实读取几帧，例如 1 / 3 / 5 / 10
    - model.num_frames 表示模型固定最大帧槽位数，例如统一固定为 10
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    effective_num_frames = int(data_cfg.get("frames_per_seq", 10))
    max_num_frames = int(model_cfg.get("num_frames", 10))

    if effective_num_frames > max_num_frames:
        raise ValueError(
            f"data.frames_per_seq ({effective_num_frames}) cannot be larger than "
            f"model.num_frames ({max_num_frames}). "
            f"For fixed-slot ablation, set model.num_frames=10."
        )

    print(
        f"[INFO] Inference effective frames_per_seq = {effective_num_frames}, "
        f"fixed model.num_frames = {max_num_frames}"
    )

    model = ShadowPointBaseline(
        image_feat_dim=int(model_cfg.get("image_feat_dim", 256)),
        light_feat_dim=int(model_cfg.get("light_feat_dim", 128)),
        fused_dim=int(model_cfg.get("fused_dim", 256)),
        num_points=int(model_cfg.get("num_points", 2048)),

        # 注意：这里必须是固定最大槽位数，不是真实读取帧数。
        # 1/3/5/10 帧消融时，这里都应该是 10。
        num_frames=max_num_frames,

        use_refiner=bool(model_cfg.get("use_refiner", True)),
        refiner_hidden_dim=int(model_cfg.get("refiner_hidden_dim", 128)),
        refiner_blocks=int(model_cfg.get("refiner_blocks", 2)),
        refiner_num_heads=int(model_cfg.get("refiner_num_heads", 4)),
        refiner_delta_scale=float(model_cfg.get("refiner_delta_scale", 0.05)),
    )
    model.to(device)
    return model


def build_dataset_from_config(cfg: Dict[str, Any]) -> ShadowSequenceDataset:
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})

    # 推理时优先使用 test 路径；没有的话再退回训练集 root
    root = cfg.get("test", data_cfg["root"])

    effective_num_frames = int(data_cfg.get("frames_per_seq", 10))
    max_num_frames = int(model_cfg.get("num_frames", 10))

    if effective_num_frames > max_num_frames:
        raise ValueError(
            f"data.frames_per_seq ({effective_num_frames}) cannot be larger than "
            f"model.num_frames ({max_num_frames})."
        )

    print(
        f"[INFO] Dataset frames_per_seq = {effective_num_frames}, "
        f"model fixed slots = {max_num_frames}"
    )

    dataset = ShadowSequenceDataset(
        root=root,
        sequences_dir=data_cfg.get("sequences_dir", "dataset"),

        # 这里是真实读取几帧，不是固定槽位数。
        frames_per_seq=effective_num_frames,

        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        image_key=data_cfg.get("image_key", "shadow_mask.png"),
        num_points=int(cfg["model"].get("num_points", 2048)),
    )
    return dataset


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    兼容常见 checkpoint 格式：
    - 直接就是 state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # 你的 train.py 里保存的是 {"model": model.state_dict(), ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # 兜底：直接把整个 ckpt 当成 state_dict（兼容你以后改成 torch.save(model.state_dict()) 的情况）
        state_dict = ckpt

    # 兼容 DataParallel 保存出来带 module. 前缀的情况
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    print(f"[INFO] Checkpoint loaded from: {ckpt_path}")
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: ShadowSequenceDataset,
    index: int,
    device: torch.device,
    use_light: bool = True,
):
    """
    返回：
    pred_points: [N, 3]
    gt_points:   [N, 3]
    seq_name:    str
    """
    sample = dataset[index]

    shadow_seq = sample["shadow_seq"].unsqueeze(0).to(device)   # [1, K, 1, H, W]
    light_dir = sample["light_dir"].unsqueeze(0).to(device)     # [1, K, 3]
    points_gt = sample["points_gt"]                             # [N, 3]
    seq_name = sample.get("seq_name", f"sample_{index:04d}")

    if not use_light:
        light_dir = torch.zeros_like(light_dir)

    pred_points = model(shadow_seq, light_dir)                  # [1, N, 3]
    pred_points = pred_points[0].detach().cpu()                 # [N, 3]

    return pred_points, points_gt, seq_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to epoch_xxxx.pt")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--all", action="store_true", help="Infer all samples in dataset")
    parser.add_argument("--base_out_dir", type=str, default="outputs/infer", help="Base directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INFO] Using device: {device}")

    cfg = load_config(args.config)
    # --- 获取消融实验名称并构建输出路径 ---
    log_out_dir = cfg.get("log", {}).get("out_dir", "")
    if log_out_dir:
        # os.path.normpath 处理尾部可能存在的斜杠，basename 获取最后一级目录名（即消融实验名）
        exp_name = os.path.basename(os.path.normpath(log_out_dir))
    else:
        exp_name = "default_exp"

    final_output_dir = os.path.join(args.base_out_dir, exp_name)
    print(f"[INFO] Output directory set to: {final_output_dir}")

    dataset = build_dataset_from_config(cfg)
    print(f"[INFO] Dataset size: {len(dataset)}")

    model = build_model_from_config(cfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if args.all:
        indices = range(len(dataset))
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"index out of range: {args.index}, dataset size = {len(dataset)}")
        indices = [args.index]

    ablation_cfg = cfg.get("ablation", {})
    use_light = bool(ablation_cfg.get("use_light", True))
    print(f"[INFO] Inference use_light = {use_light}")

    for idx in indices:
        pred_points, gt_points, seq_name = run_inference(
            model=model,
            dataset=dataset,
            index=idx,
            device=device,
            use_light=use_light,
        )

        save_dir = os.path.join(final_output_dir, seq_name)
        ensure_dir(save_dir)

        pred_ply_path = os.path.join(save_dir, "pred.ply")
        gt_ply_path = os.path.join(save_dir, "gt.ply")
        meta_json_path = os.path.join(save_dir, "meta.json")

        save_point_cloud_ply(pred_points, pred_ply_path)
        save_point_cloud_ply(gt_points, gt_ply_path)

        meta = {
            "seq_name": seq_name,
            "dataset_index": idx,
            "checkpoint": args.checkpoint,
            "pred_ply": pred_ply_path,
            "gt_ply": gt_ply_path,
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Inference done: index={idx}, seq_name={seq_name}")
        print(f"[INFO] pred ply : {pred_ply_path}")
        print(f"[INFO] gt ply   : {gt_ply_path}")
        print(f"[INFO] meta json: {meta_json_path}")



if __name__ == "__main__":
    main()