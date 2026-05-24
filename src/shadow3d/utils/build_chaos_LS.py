#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 chaos_LS:打乱每个 instance 内各 frame 的 light/shadow 配对
==============================================================

在 dataset 同级新建 chaos_LS 目录,结构与 dataset 一致。
对每个 instance(包含若干 frame_xxx 的目录),将各 frame 下的
  - light_info.txt
  - shadow_mask.png
作为内容单元,按指定方式重新分配到该 instance 的各个 frame 槽位。
  - rgb_with_shadow.png 一律丢弃(不复制)
非 frame 层级的其它文件/目录原样复制。

互换方式 (--mode)
-----------------
  shuffle  随机打乱(默认,呼应 chaos):各 frame 的内容被随机重排到各槽位
  roll     循环移位:frame_000 内容 -> frame_001 -> ... -> 最后绕回 frame_000
  swap     两两对调:(000<->001), (002<->003), ... 奇数个时最后一个保持不动

绑定方式
--------
  默认:light_info.txt 与 shadow_mask.png 作为一对整体一起移动(配对保持不变)
  --independent:两者各自独立打乱(light 与 shadow 的配对会被打乱,更"混乱")

用法
----
# 每个 instance 各自随机打乱(默认)
python build_chaos_LS.py --src data/test_runs/dataset

# 只处理某一个 instance,循环移位
python build_chaos_LS.py --src data/test_runs/dataset/02747177/1b7d468a27208ee3dad910e221d16b18 --mode roll

# 随机打乱且 light/shadow 各自独立打乱,固定随机种子保证可复现
python build_chaos_LS.py --src data/test_runs/dataset --mode shuffle --independent --seed 42

参数
----
--src          源目录:可为 dataset 根(批量处理每个 instance),
               也可为单个 instance 目录
--mode         shuffle / roll / swap,默认 shuffle
--independent  light_info 与 shadow_mask 各自独立打乱(默认配对整体移动)
--seed         随机种子(仅 shuffle 模式),默认随机
--dst          目标目录完整路径
--dst-name     目标目录名,默认 chaos_LS(在 src 的 dataset 同级创建)
--overwrite    目标已存在则删除重建
--light-name   默认 light_info.txt
--mask-name    默认 shadow_mask.png
--skip-name    丢弃的文件名,默认 rgb_with_shadow.png

依赖: 仅标准库
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# 排列生成
# --------------------------------------------------------------------------- #
def make_permutation(n: int, mode: str, rng: random.Random) -> list:
    """
    返回长度 n 的目标索引列表 perm,语义:
      源槽位 i 的内容 -> 放到目标槽位 perm[i]
    """
    if n <= 1:
        return list(range(n))

    if mode == "roll":
        # i -> (i+1) % n
        return [(i + 1) % n for i in range(n)]

    if mode == "swap":
        perm = list(range(n))
        for i in range(0, n - 1, 2):
            perm[i], perm[i + 1] = i + 1, i  # i<->i+1
        return perm

    if mode == "shuffle":
        # 生成一个尽量没有不动点的随机排列(派遣排列),失败则退回普通随机
        idx = list(range(n))
        for _ in range(100):
            perm = idx[:]
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm
        rng.shuffle(perm)
        return perm

    raise ValueError(f"未知 mode: {mode}")


# --------------------------------------------------------------------------- #
# instance 处理
# --------------------------------------------------------------------------- #
def find_frame_dirs(instance_dir: Path, mask_name: str):
    """返回该 instance 下所有 frame 目录(含 mask 文件的直接子目录),按名称排序。"""
    frames = []
    for child in sorted(instance_dir.iterdir()):
        if child.is_dir() and (child / mask_name).is_file():
            frames.append(child)
    return frames


def process_instance(src_inst: Path, dst_inst: Path, args, rng, stats):
    """
    处理单个 instance:
      - 先把非 frame 内容原样复制
      - 再按排列把各 frame 的 light/shadow 重新分配写入目标 frame
    """
    frame_dirs = find_frame_dirs(src_inst, args.mask_name)

    # 复制该 instance 下的非 frame 文件/目录(frame 目录单独处理)
    frame_set = {f.name for f in frame_dirs}
    for child in sorted(src_inst.iterdir()):
        if child.name in frame_set:
            continue
        target = dst_inst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
        stats["copied_other"] += 1

    if not frame_dirs:
        return

    # 先建好所有目标 frame 目录
    for f in frame_dirs:
        (dst_inst / f.name).mkdir(parents=True, exist_ok=True)

    n = len(frame_dirs)

    # 收集每个源 frame 的 light / shadow 路径(可能缺失)
    light_src = [f / args.light_name for f in frame_dirs]
    mask_src = [f / args.mask_name for f in frame_dirs]

    if args.independent:
        # light 与 shadow 各自独立打乱
        perm_light = make_permutation(n, args.mode, rng)
        perm_mask = make_permutation(n, args.mode, rng)
    else:
        # 配对整体移动:同一个排列
        perm_light = make_permutation(n, args.mode, rng)
        perm_mask = perm_light

    # 源槽位 i 的内容 -> 目标槽位 perm[i]
    for i in range(n):
        # light
        if light_src[i].is_file():
            dst_light = dst_inst / frame_dirs[perm_light[i]].name / args.light_name
            shutil.copy2(light_src[i], dst_light)
            stats["moved_light"] += 1
        # shadow
        if mask_src[i].is_file():
            dst_mask = dst_inst / frame_dirs[perm_mask[i]].name / args.mask_name
            shutil.copy2(mask_src[i], dst_mask)
            stats["moved_mask"] += 1

    stats["instances"] += 1
    # 记录映射,便于核对
    mapping = ", ".join(f"{frame_dirs[i].name}->{frame_dirs[perm_light[i]].name}" for i in range(n))
    print(f"  [{src_inst.name}] {n} frames | light: {mapping}"
          + ("" if not args.independent else
             " | shadow: " + ", ".join(f"{frame_dirs[i].name}->{frame_dirs[perm_mask[i]].name}" for i in range(n))))


def is_instance_dir(d: Path, mask_name: str) -> bool:
    """该目录是否为 instance:其直接子目录中存在含 mask 的 frame 目录。"""
    for child in d.iterdir():
        if child.is_dir() and (child / mask_name).is_file():
            return True
    return False


def collect_instances(src: Path, mask_name: str):
    """
    若 src 本身就是 instance,返回 [src];
    否则递归查找其下所有 instance 目录。
    """
    if is_instance_dir(src, mask_name):
        return [src]
    instances = []
    for d in sorted(p for p in src.rglob("*") if p.is_dir()):
        if is_instance_dir(d, mask_name):
            instances.append(d)
    return instances


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(args):
    src = Path(args.src).resolve()
    if not src.is_dir():
        sys.exit(f"源目录不存在或不是目录: {src}")

    instances = collect_instances(src, args.mask_name)
    if not instances:
        sys.exit(f"在 {src} 下未找到任何 instance(含 {args.mask_name} 的 frame 目录)")

    # 决定目标根目录,并确定相对结构的基准
    if args.dst:
        dst_root = Path(args.dst).resolve()
        base = src  # 以 src 为相对基准
    else:
        # 在 src 同级创建 dst_name;相对结构以 src 的父目录为基准
        dst_root = src.parent / args.dst_name
        base = src

    if dst_root.exists():
        if args.overwrite:
            shutil.rmtree(dst_root)
        else:
            sys.exit(f"目标目录已存在: {dst_root}\n(如需覆盖请加 --overwrite)")

    rng = random.Random(args.seed)
    stats = {"instances": 0, "moved_light": 0, "moved_mask": 0,
             "copied_other": 0, "skipped_rgb": 0}

    print(f"源: {src}")
    print(f"模式: {args.mode}  独立打乱: {args.independent}  "
          f"种子: {args.seed}  instance 数: {len(instances)}")
    print("映射详情:")

    for inst in instances:
        rel = inst.relative_to(base)
        dst_inst = dst_root / rel
        dst_inst.mkdir(parents=True, exist_ok=True)
        process_instance(inst, dst_inst, args, rng, stats)

    # rgb 一律不复制:统计被丢弃数量
    stats["skipped_rgb"] = sum(
        1 for inst in instances for f in find_frame_dirs(inst, args.mask_name)
        if (f / args.skip_name).is_file()
    )

    return dst_root, stats


def main():
    parser = argparse.ArgumentParser(
        description="构建 chaos_LS:打乱每个 instance 内 frame 的 light/shadow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True,
                        help="dataset 根目录,或单个 instance 目录")
    parser.add_argument("--mode", choices=["shuffle", "roll", "swap"],
                        default="shuffle", help="互换方式")
    parser.add_argument("--independent", action="store_true",
                        help="light 与 shadow 各自独立打乱(默认配对整体移动)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子(shuffle 模式可复现)")
    parser.add_argument("--dst", default=None, help="目标目录完整路径")
    parser.add_argument("--dst-name", default="chaos_LS",
                        help="目标目录名(在 src 同级创建,默认 chaos_LS)")
    parser.add_argument("--overwrite", action="store_true",
                        help="目标目录已存在则删除重建")
    parser.add_argument("--light-name", default="light_info.txt")
    parser.add_argument("--mask-name", default="shadow_mask.png")
    parser.add_argument("--skip-name", default="rgb_with_shadow.png",
                        help="丢弃的文件名")
    args = parser.parse_args()

    dst_root, stats = build(args)

    print(f"\n完成。副本目录: {dst_root}")
    print(f"  处理 instance 数:   {stats['instances']}")
    print(f"  迁移 light_info:    {stats['moved_light']}")
    print(f"  迁移 shadow_mask:   {stats['moved_mask']}")
    print(f"  丢弃 rgb 文件:      {stats['skipped_rgb']}")
    print(f"  其它原样复制:       {stats['copied_other']}")


if __name__ == "__main__":
    main()
