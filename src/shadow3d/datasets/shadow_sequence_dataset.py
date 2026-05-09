import os
import re
import struct
from typing import Dict, List, Tuple
import hashlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def stable_seed_from_string(s: str) -> int:
    """
    根据样本名生成稳定随机种子。
    避免 Python 内置 hash() 在不同进程/不同运行中变化。
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def normalize_points_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    中心化 + 单位球归一化
    """
    center = points.mean(axis=0, keepdims=True)
    points = points - center

    scale = np.max(np.linalg.norm(points, axis=1))
    scale = max(float(scale), 1e-6)
    points = points / scale
    return points.astype(np.float32)

def parse_light_info(light_info_path: str) -> np.ndarray:
    """
    解析 light_info.txt
    支持格式示例：
        theta:45.0, phi:120.0
        theta=45.0 phi=120.0
    返回:
        light_dir: np.ndarray shape [3], 单位向量
    """
    with open(light_info_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    theta_match = re.search(r"theta\s*[:=]\s*([-+]?\d*\.?\d+)", text)
    phi_match = re.search(r"phi\s*[:=]\s*([-+]?\d*\.?\d+)", text)

    if theta_match is None or phi_match is None:
        raise ValueError(f"Cannot parse theta/phi from {light_info_path}: {text}")

    theta_deg = float(theta_match.group(1))
    phi_deg = float(phi_match.group(1))

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # 与常见球坐标一致：theta 为极角，phi 为方位角
    lx = np.sin(theta) * np.cos(phi)
    ly = np.sin(theta) * np.sin(phi)
    lz = np.cos(theta)

    light_dir = np.array([lx, ly, lz], dtype=np.float32)
    norm = np.linalg.norm(light_dir) + 1e-8
    return light_dir / norm


def read_image_gray(image_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    """
    读取阴影图，转为灰度并 resize
    输出范围 [0, 1], shape [H, W]
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _parse_ply_header(f) -> Dict:
    """
    解析 PLY header，支持：
    - ascii
    - binary_little_endian
    只读取 vertex 的 x y z
    """
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header.")
        line = line.decode("utf-8").strip()
        header_lines.append(line)
        if line == "end_header":
            break

    fmt = None
    vertex_count = None
    in_vertex_element = False
    vertex_properties = []

    for line in header_lines:
        if line.startswith("format "):
            if "ascii" in line:
                fmt = "ascii"
            elif "binary_little_endian" in line:
                fmt = "binary_little_endian"
            else:
                raise ValueError(f"Unsupported PLY format: {line}")

        elif line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            in_vertex_element = True

        elif line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex_element = False

        elif line.startswith("property") and in_vertex_element:
            # e.g. property float x
            parts = line.split()
            prop_name = parts[-1]
            vertex_properties.append(prop_name)

    if fmt is None or vertex_count is None:
        raise ValueError("Invalid PLY header: missing format or vertex count.")

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_properties": vertex_properties,
    }


def read_ply_xyz(ply_path: str) -> np.ndarray:
    """
    读取 PLY 顶点坐标，返回 [N, 3]
    支持 ascii / binary_little_endian
    仅依赖 numpy + struct
    """
    with open(ply_path, "rb") as f:
        header = _parse_ply_header(f)
        fmt = header["format"]
        vertex_count = header["vertex_count"]
        props = header["vertex_properties"]

        if not all(k in props for k in ("x", "y", "z")):
            raise ValueError(f"PLY missing x/y/z in {ply_path}")

        x_idx = props.index("x")
        y_idx = props.index("y")
        z_idx = props.index("z")
        num_props = len(props)

        if fmt == "ascii":
            pts = []
            for _ in range(vertex_count):
                line = f.readline().decode("utf-8").strip()
                vals = line.split()
                pts.append([float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])])
            return np.asarray(pts, dtype=np.float32)

        if fmt == "binary_little_endian":
            # 这里简化处理：默认 vertex 属性都是 4-byte float
            # 对你自己的 GT 点云，通常够用
            data = np.fromfile(f, dtype=np.float32, count=vertex_count * num_props)
            data = data.reshape(vertex_count, num_props)
            xyz = data[:, [x_idx, y_idx, z_idx]].astype(np.float32)
            return xyz

    raise ValueError(f"Unsupported PLY format in {ply_path}")


def sample_or_pad_points(points: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    """
    将 GT 点云统一到固定点数 num_points。
    使用固定随机种子，保证同一个样本每次采样一致。
    """
    n = points.shape[0]
    rng = np.random.default_rng(seed)

    if n == num_points:
        return points

    if n > num_points:
        idx = rng.choice(n, num_points, replace=False)
        return points[idx]

    extra = rng.choice(n, num_points - n, replace=True)
    idx = np.concatenate([np.arange(n), extra], axis=0)
    return points[idx]


class ShadowSequenceDataset(Dataset):
    """
    每个样本 = 一个 sequence
    输出:
        {
            "shadow_seq": [K, 1, H, W],
            "light_dir":  [K, 3],
            "points_gt":  [N, 3],
            "seq_name":   str,
        }
    """

    def __init__(
        self,
        root: str,
        sequences_dir: str = "dataset",
        frames_per_seq: int = 10,
        image_size: Tuple[int, int] = (256, 256),
        image_key: str = "shadow_mask.png",
        num_points: int = 2048,
    ):
        super().__init__()
        self.root = root
        self.sequences_root = os.path.join(root, sequences_dir)
        self.frames_per_seq = frames_per_seq
        self.image_size = image_size
        self.image_key = image_key
        self.num_points = num_points
        self.fixed_points_gt = None

        if not os.path.isdir(self.sequences_root):
            raise FileNotFoundError(f"Sequences directory not found: {self.sequences_root}")

        self.samples = self._build_index()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid sequences found under {self.sequences_root}")

        self.fixed_points_gt = self._build_fixed_gt_cache()

    def _load_fixed_gt_points(self, sample: Dict) -> np.ndarray:
        """
        读取并固定单个样本的 GT 点云。
        注意：这里的固定包含三件事：
        1. PLY 原始点云读取；
        2. 单位球归一化；
        3. 按 seq_name 生成稳定 seed 后采样/补点到 self.num_points。
        """
        points_gt = read_ply_xyz(sample["gt_path"])
        points_gt = normalize_points_unit_sphere(points_gt)
        seed = stable_seed_from_string(sample["seq_name"])
        points_gt = sample_or_pad_points(points_gt, self.num_points, seed=seed)
        return points_gt.astype(np.float32)

    def _build_fixed_gt_cache(self) -> List[np.ndarray]:
        """
        初始化时为所有样本构建固定 GT 点云缓存。
        后续 __getitem__ 直接取缓存，避免每个 epoch 重新读 PLY / 重新采样。
        """
        return [self._load_fixed_gt_points(sample) for sample in self.samples]


    def _build_index(self) -> List[Dict]:
        samples = []

        # 1. 第一层：获取所有分类目录 (如 02747177)
        category_ids = sorted([
            d for d in os.listdir(self.sequences_root)
            if os.path.isdir(os.path.join(self.sequences_root, d))
        ])

        for cat_id in category_ids:
            cat_dir = os.path.join(self.sequences_root, cat_id)

            # 2. 第二层：获取分类下的所有实例目录 (如 1b7d468a...)
            instance_ids = sorted([
                d for d in os.listdir(cat_dir)
                if os.path.isdir(os.path.join(cat_dir, d))
            ])

            for inst_id in instance_ids:
                seq_dir = os.path.join(cat_dir, inst_id)
                # 为了区分不同类下的同名实例，seq_name 可以结合类名
                seq_name = f"{cat_id}_{inst_id}"

                # --- 以下逻辑基本保持不变，只需确保路径正确 ---
                frame_dirs = sorted([
                    os.path.join(seq_dir, d)
                    for d in os.listdir(seq_dir)
                    if d.startswith("frame_") and os.path.isdir(os.path.join(seq_dir, d))
                ])

                if len(frame_dirs) < self.frames_per_seq:
                    continue

                frame_dirs = frame_dirs[:self.frames_per_seq]

                geom_dir = os.path.join(seq_dir, "object_geometry")
                if not os.path.isdir(geom_dir):
                    continue

                gt_path = os.path.join(geom_dir, "gt.ply")
                if not os.path.isfile(gt_path):
                    continue

                # 验证帧内文件
                ok = True
                for frame_dir in frame_dirs:
                    image_path = os.path.join(frame_dir, self.image_key)
                    light_path = os.path.join(frame_dir, "light_info.txt")
                    if not os.path.isfile(image_path) or not os.path.isfile(light_path):
                        ok = False
                        break

                if ok:
                    samples.append({
                        "seq_name": seq_name,
                        "seq_dir": seq_dir,
                        "frame_dirs": frame_dirs,
                        "gt_path": gt_path,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        shadow_imgs = []
        light_dirs = []

        for frame_dir in sample["frame_dirs"]:
            image_path = os.path.join(frame_dir, self.image_key)
            light_path = os.path.join(frame_dir, "light_info.txt")

            img = read_image_gray(image_path, self.image_size)  # [H, W]
            light_dir = parse_light_info(light_path)            # [3]

            shadow_imgs.append(img[None, ...])   # [1, H, W]
            light_dirs.append(light_dir)

        shadow_seq = np.stack(shadow_imgs, axis=0).astype(np.float32)   # [K, 1, H, W]
        light_dir = np.stack(light_dirs, axis=0).astype(np.float32)     # [K, 3]

        if self.fixed_points_gt is not None:
            points_gt = self.fixed_points_gt[index]
        else:
            # 备用路径：即使不缓存，也仍然用稳定 seed，保证同一样本采样固定。
            points_gt = self._load_fixed_gt_points(sample)

        return {
            "shadow_seq": torch.from_numpy(shadow_seq),   # [K, 1, H, W]
            "light_dir": torch.from_numpy(light_dir),     # [K, 3]
            "points_gt": torch.from_numpy(points_gt.copy()),     # [N, 3]
            "seq_name": sample["seq_name"],
        }