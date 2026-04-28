import argparse
import random
import shutil
from pathlib import Path

#############################
#将训练集对应数量的数据移动到测试集#
#############################
def split_dataset(
    train_root: Path,
    test_root: Path,
    ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
    min_one: bool = False,
):
    train_root = train_root.resolve()
    test_root = test_root.resolve()

    if not train_root.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_root}")

    test_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    print(f"[INFO] train_root = {train_root}")
    print(f"[INFO] test_root  = {test_root}")
    print(f"[INFO] ratio      = {ratio}")
    print(f"[INFO] seed       = {seed}")
    print(f"[INFO] dry_run    = {dry_run}")
    print("-" * 80)

    total_moved = 0
    skipped_categories = 0

    # 遍历 data/train_runs/dataset 下的一级目录，例如 A
    for category_dir in sorted(train_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        # --- 新增功能：如果目标路径已存在该类别文件夹，则跳过 ---
        target_category_path = test_root / category_name
        if target_category_path.exists():
            print(f"[SKIP CATEGORY] {category_name}: 目标目录已存在，跳过该类别")
            skipped_categories += 1
            continue

        # 当前 A 目录下的所有子文件夹，例如 B
        sample_dirs = [
            p for p in category_dir.iterdir()
            if p.is_dir()
        ]

        num_samples = len(sample_dirs)

        if num_samples == 0:
            print(f"[SKIP] {category_name}: 没有子文件夹")
            continue

        move_count = get_test_count(num_samples)

        # 如果你希望每个非空类别至少抽 1 个，可以加 --min_one
        if min_one and move_count == 0 and num_samples > 0:
            move_count = 1

        # 防止极端情况：测试集不能把该类别全部移走
        move_count = min(move_count, max(num_samples - 1, 0))

        if move_count <= 0:
            print(f"[SKIP] {category_name}: 文件夹数 {num_samples}, 无法抽取测试集")
            continue

        selected_dirs = random.sample(sample_dirs, move_count)

        print(
            f"[CATEGORY] {category_name}: "
            f"总数 {num_samples}, 抽取 {move_count}"
        )

        for src_dir in selected_dirs:
            relative_path = src_dir.relative_to(train_root)
            dst_dir = test_root / relative_path

            print(f"  MOVE: {src_dir} -> {dst_dir}")

            if dry_run:
                continue

            if dst_dir.exists():
                raise FileExistsError(
                    f"目标目录已存在，为避免覆盖，已停止: {dst_dir}"
                )

            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_dir), str(dst_dir))

            total_moved += 1

    print("-" * 80)

    if dry_run:
        print("[DRY-RUN] 预览完成，没有实际移动任何文件夹")
    else:
        print(f"[DONE] 实际移动文件夹数量: {total_moved}")
        print(f"[DONE] 跳过的类别总数: {skipped_categories}")

def get_test_count(num_samples: int) -> int:
    """
    根据当前类别的样本数量，动态决定测试集数量。
    """

    if num_samples < 70:
        return 3
    elif num_samples < 100:
        return 4
    elif num_samples < 120:
        return 5
    elif num_samples < 150:
        return 6
    elif num_samples < 200:
        return 8
    elif num_samples < 300:
        return 10
    elif num_samples < 500:
        return 20
    else:
        return 30


def main():
    parser = argparse.ArgumentParser(
        description="从 train_runs/dataset 中按类别随机抽取部分文件夹移动到 test_runs/dataset"
    )

    parser.add_argument(
        "--train_root",
        type=str,
        default="data/train_runs/dataset",
        help="训练集根目录",
    )

    parser.add_argument(
        "--test_root",
        type=str,
        default="data/test_runs/dataset",
        help="测试集根目录",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="每个类别抽取比例，默认 0.2",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证每次抽取结果可复现",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只预览，不实际移动",
    )

    parser.add_argument(
        "--min_one",
        action="store_true",
        help="如果某个类别数量太少，至少抽取 1 个",
    )

    args = parser.parse_args()

    split_dataset(
        train_root=Path(args.train_root),
        test_root=Path(args.test_root),
        ratio=args.ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        min_one=args.min_one,
    )


if __name__ == "__main__":
    main()