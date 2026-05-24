#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把推理结果的 Chamfer Distance 重新归一化到 DRWR Table 3 的尺度（包围盒对角线=1），
以便与 DRWR 论文报告的逐物体归一化 CD（如 airplane=0.0527）做同尺度对比。

目录结构假设:
    outputs/infer/drwr/<modelID>/pred.ply
    outputs/infer/drwr/<modelID>/gt.ply

脚本会同时报告两套尺度下的平均 CD:
  (A) 单位球尺度 (与你训练时一致, 用于自检是否≈0.0024)
  (B) 对角线=1 尺度 (与 DRWR Table 3 同尺度, 用于对比)

依赖: numpy, scipy
    pip install numpy scipy
"""

import os
import argparse
import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------- PLY 读取 (与你 dataset 里的实现保持一致) ----------
def read_ply_xyz(ply_path):
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF in header")
            s = line.decode("utf-8").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt, vcount, props, in_vtx = None, None, [], False
        for s in header:
            if s.startswith("format "):
                fmt = "ascii" if "ascii" in s else ("ble" if "binary_little_endian" in s else None)
            elif s.startswith("element vertex"):
                vcount = int(s.split()[-1]); in_vtx = True
            elif s.startswith("element "):
                in_vtx = False
            elif s.startswith("property") and in_vtx:
                props.append(s.split()[-1])
        xi, yi, zi, npx = props.index("x"), props.index("y"), props.index("z"), len(props)
        if fmt == "ascii":
            pts = []
            for _ in range(vcount):
                v = f.readline().decode("utf-8").split()
                pts.append([float(v[xi]), float(v[yi]), float(v[zi])])
            return np.asarray(pts, dtype=np.float64)
        elif fmt == "ble":
            data = np.fromfile(f, dtype=np.float32, count=vcount * npx).reshape(vcount, npx)
            return data[:, [xi, yi, zi]].astype(np.float64)
        raise ValueError("unsupported ply format")


# ---------- CD 计算 ----------
def chamfer_distance(pred, gt):
    """
    对称 Chamfer Distance（平方距离均值之和）。
    CD = mean_i min_j ||p_i - g_j||^2 + mean_j min_i ||g_j - p_i||^2
    返回 (cd, p2g, g2p)
    """
    if HAVE_SCIPY:
        tg = cKDTree(gt); tp = cKDTree(pred)
        d_pg, _ = tg.query(pred, k=1)   # 每个 pred 到最近 gt
        d_gp, _ = tp.query(gt, k=1)     # 每个 gt 到最近 pred
        p2g = float(np.mean(d_pg ** 2))
        g2p = float(np.mean(d_gp ** 2))
    else:
        # 无 scipy 时的暴力实现（点数大会慢）
        def nn_sq(a, b):
            d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
            return d.min(1)
        p2g = float(np.mean(nn_sq(pred, gt)))
        g2p = float(np.mean(nn_sq(gt, pred)))
    return p2g + g2p, p2g, g2p


# ---------- 归一化 ----------
def to_unit_sphere(gt, pred):
    """按 gt 的最远点半径归一化（与你训练一致）。pred 跟随同一变换。"""
    c = gt.mean(0, keepdims=True)
    gt2, pred2 = gt - c, pred - c
    scale = max(float(np.max(np.linalg.norm(gt2, axis=1))), 1e-9)
    return gt2 / scale, pred2 / scale


def to_bbox_diag1(gt, pred):
    """按 gt 的包围盒对角线=1 归一化（与 DRWR Table 3 一致）。pred 跟随同一变换。"""
    c = gt.mean(0, keepdims=True)
    gt2, pred2 = gt - c, pred - c
    diag = max(float(np.linalg.norm(gt2.max(0) - gt2.min(0))), 1e-9)
    return gt2 / diag, pred2 / diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("outputs", "infer", "drwr"),
                    help="推理结果根目录，下面每个子文件夹一个样本")
    ap.add_argument("--pred_name", default="pred.ply")
    ap.add_argument("--gt_name", default="gt.ply")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"找不到目录: {args.root}")

    sample_dirs = sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
    )
    if not sample_dirs:
        raise SystemExit(f"{args.root} 下没有样本子目录")

    cd_sphere, cd_diag = [], []
    skipped = 0
    for name in sample_dirs:
        pred_p = os.path.join(args.root, name, args.pred_name)
        gt_p = os.path.join(args.root, name, args.gt_name)
        if not (os.path.isfile(pred_p) and os.path.isfile(gt_p)):
            skipped += 1
            continue
        pred = read_ply_xyz(pred_p)
        gt = read_ply_xyz(gt_p)

        # (A) 单位球尺度
        g_s, p_s = to_unit_sphere(gt, pred)
        cd_s, _, _ = chamfer_distance(p_s, g_s)
        cd_sphere.append(cd_s)

        # (B) 对角线=1 尺度 (对齐 DRWR Table 3)
        g_d, p_d = to_bbox_diag1(gt, pred)
        cd_d, _, _ = chamfer_distance(p_d, g_d)
        cd_diag.append(cd_d)

    n = len(cd_sphere)
    print(f"[信息] 处理样本数: {n}  (跳过缺文件: {skipped})")
    print(f"[A] 单位球尺度  平均 CD = {np.mean(cd_sphere):.6f}   "
          f"(自检: 应≈你训练/推理报告的 0.0024)")
    print(f"[B] 对角线=1 尺度 平均 CD = {np.mean(cd_diag):.6f}   "
          f"(与 DRWR Table 3 同尺度, 飞机参考值 0.0527)")
    print()
    print("说明: [A] 用于验证本脚本 CD 实现与你的一致; [B] 才是可与 DRWR 对比的数字。")
    print("若 [A] 明显偏离 0.0024, 说明 CD 定义/采样点数与训练时不同, [B] 需谨慎使用。")


if __name__ == "__main__":
    main()
