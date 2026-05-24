#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 dataset 的形态学处理副本 (fuckpng)
=======================================

将 dataset 目录整体复制到其同级的 fuckpng 目录,目录结构完全一致。
区别只在于每个 frame 目录(即包含 shadow_mask.png 的目录)里:
  - light_info.txt        原样复制
  - shadow_mask.png       做腐蚀(erode)或膨胀(dilate)处理后写入,文件名不变
  - rgb_with_shadow.png   不复制
其它所有目录与文件均原样复制。

用法
----
# 膨胀,半径 3(默认结构元素 ellipse,迭代 1 次)
python build_fuckpng.py --src data/test_runs/dataset --op dilate --radius 3

# 腐蚀,半径 5,自定义输出目录名
python build_fuckpng.py --src data/test_runs/dataset --op erode --radius 5 --dst-name fuckpng

参数
----
--src         源 dataset 目录
--op          erode(腐蚀) 或 dilate(膨胀),必填
--radius      结构元素半径(像素),必填。半径 r 对应 (2r+1) 的核
--dst         目标目录完整路径(默认: 在 src 同级新建 <dst-name>)
--dst-name    目标目录名,默认 fuckpng(仅当未指定 --dst 时生效)
--kernel      结构元素形状 ellipse/rect/cross,默认 ellipse
--iterations  形态学迭代次数,默认 1
--threshold   二值化阈值 0-255,默认 127
--mask-name   掩码文件名,默认 shadow_mask.png
--skip-name   frame 目录下不复制的文件名,默认 rgb_with_shadow.png
--overwrite   若目标目录已存在则先删除重建

依赖: numpy, opencv-python  (pip install numpy opencv-python)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少依赖 opencv-python,请先运行: pip install opencv-python numpy")


# --------------------------------------------------------------------------- #
# 形态学工具
# --------------------------------------------------------------------------- #
def load_binary_mask(path: Path, threshold: int) -> np.ndarray:
    """读入掩码并二值化为 {0,255} 的 uint8 单通道图。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取图像: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            if alpha.max() > 0 and alpha.min() < alpha.max():
                img = alpha
            else:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img > threshold).astype(np.uint8) * 255


def make_kernel(shape: str, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    shapes = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    if shape not in shapes:
        raise ValueError(f"未知 kernel 形状: {shape}")
    return cv2.getStructuringElement(shapes[shape], (size, size))


def apply_morph(mask: np.ndarray, op: str, kernel: np.ndarray, iterations: int) -> np.ndarray:
    morph_ops = {"erode": cv2.erode, "dilate": cv2.dilate}
    if op not in morph_ops:
        raise ValueError(f"未知操作: {op}")
    return morph_ops[op](mask, kernel, iterations=iterations)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(args):
    src = Path(args.src).resolve()
    if not src.is_dir():
        sys.exit(f"源目录不存在或不是目录: {src}")

    dst = Path(args.dst).resolve() if args.dst else (src.parent / args.dst_name)
    if dst.exists():
        if args.overwrite:
            shutil.rmtree(dst)
        else:
            sys.exit(f"目标目录已存在: {dst}\n(如需覆盖请加 --overwrite)")

    kernel = make_kernel(args.kernel, args.radius)

    stats = {"frames": 0, "copied_files": 0, "processed_masks": 0,
             "skipped_rgb": 0, "dirs": 0, "errors": 0}

    # 遍历源目录全部内容
    for src_path in sorted(src.rglob("*")):
        rel = src_path.relative_to(src)
        dst_path = dst / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            stats["dirs"] += 1
            continue

        # src_path 是文件。判断它所在目录是否为 frame 目录
        # frame 目录的判定标准:同目录下存在掩码文件
        parent = src_path.parent
        is_frame_dir = (parent / args.mask_name).is_file()

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if is_frame_dir:
            name = src_path.name
            if name == args.skip_name:
                stats["skipped_rgb"] += 1
                continue  # rgb_with_shadow.png 不复制
            if name == args.mask_name:
                # 处理后写入同名文件
                try:
                    mask = load_binary_mask(src_path, args.threshold)
                    proc = apply_morph(mask, args.op, kernel, args.iterations)
                    cv2.imwrite(str(dst_path), proc)
                    stats["processed_masks"] += 1
                except Exception as e:
                    print(f"[掩码处理失败] {src_path}: {e}", file=sys.stderr)
                    stats["errors"] += 1
                continue
            # frame 目录下的其它文件(如 light_info.txt)原样复制
            shutil.copy2(src_path, dst_path)
            stats["copied_files"] += 1
        else:
            # 非 frame 目录的文件,原样复制
            shutil.copy2(src_path, dst_path)
            stats["copied_files"] += 1

    # 统计 frame 数 = 处理过的掩码数
    stats["frames"] = stats["processed_masks"]
    return dst, stats


def main():
    parser = argparse.ArgumentParser(
        description="构建 dataset 的腐蚀/膨胀副本 (fuckpng)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True, help="源 dataset 目录")
    parser.add_argument("--op", required=True, choices=["erode", "dilate"],
                        help="形态学操作: erode 腐蚀 / dilate 膨胀")
    parser.add_argument("--radius", required=True, type=int,
                        help="结构元素半径(像素)")
    parser.add_argument("--dst", default=None, help="目标目录完整路径")
    parser.add_argument("--dst-name", default="fuckpng",
                        help="目标目录名(在 src 同级创建,默认 fuckpng)")
    parser.add_argument("--kernel", choices=["ellipse", "rect", "cross"],
                        default="ellipse", help="结构元素形状")
    parser.add_argument("--iterations", type=int, default=1, help="形态学迭代次数")
    parser.add_argument("--threshold", type=int, default=127, help="二值化阈值 0-255")
    parser.add_argument("--mask-name", default="shadow_mask.png", help="掩码文件名")
    parser.add_argument("--skip-name", default="rgb_with_shadow.png",
                        help="frame 目录下不复制的文件名")
    parser.add_argument("--overwrite", action="store_true",
                        help="目标目录已存在则先删除重建")
    args = parser.parse_args()

    if args.radius < 0:
        sys.exit("radius 不能为负")

    print(f"源: {args.src}")
    print(f"操作: {args.op}  半径: {args.radius}  核: {args.kernel}  迭代: {args.iterations}")
    dst, stats = build(args)

    print(f"\n完成。副本目录: {dst}")
    print(f"  处理后掩码 (frame 数): {stats['processed_masks']}")
    print(f"  原样复制文件:         {stats['copied_files']}")
    print(f"  跳过的 rgb 文件:      {stats['skipped_rgb']}")
    print(f"  创建目录:             {stats['dirs']}")
    if stats["errors"]:
        print(f"  处理失败:             {stats['errors']}")


if __name__ == "__main__":
    main()
