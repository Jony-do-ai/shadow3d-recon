import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_train_log(csv_path):
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "global_step": int(row["global_step"]),
                "lr": float(row["lr"]),
                "loss_total": float(row["loss_total"]),
                "loss_cd": float(row["loss_cd"]),
                "loss_center": float(row["loss_center"]),
                "loss_bbox": float(row["loss_bbox"]),
                "epoch_time_sec": float(row["epoch_time_sec"]),
            })

    if len(rows) == 0:
        raise RuntimeError(f"No rows found in {csv_path}")

    return rows


def plot_single_metric(rows, metric_name, out_path):
    epochs = [r["epoch"] for r in rows]
    values = [r[metric_name] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_losses(rows, out_path):
    epochs = [r["epoch"] for r in rows]

    metrics = [
        "loss_total",
        "loss_cd",
        "loss_p2g",
        "loss_g2p",
        "loss_center",
        "loss_bbox",
        "fscore_0_01",
        "fscore_0_02",
        "fscore_0_05",
    ]

    plt.figure(figsize=(9, 6))

    for metric in metrics:
        values = [r[metric] for r in rows]
        plt.plot(epochs, values, marker="o", linewidth=1.5, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_all_fscores(rows, out_path):
    epochs = [r["epoch"] for r in rows]

    metrics = [
        "fscore_0_01",
        "fscore_0_02",
        "fscore_0_05",
    ]

    plt.figure(figsize=(9, 6))

    for metric in metrics:
        values = [r[metric] for r in rows]
        plt.plot(epochs, values, marker="o", linewidth=1.5, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("F-score")
    plt.title("F-score Curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to train_log.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    csv_path = args.log

    if args.out_dir is None:
        exp_dir = os.path.dirname(csv_path)
        out_dir = os.path.join(exp_dir, "plots")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    rows = read_train_log(csv_path)

    for metric in ["loss_total", "loss_cd", "loss_p2g",
    "loss_g2p","loss_center", "loss_bbox"]:
        out_path = os.path.join(out_dir, f"{metric}.png")
        plot_single_metric(rows, metric, out_path)
        print(f"[OK] Saved {out_path}")

    out_path = os.path.join(out_dir, "losses_all.png")
    plot_all_losses(rows, out_path)
    print(f"[OK] Saved {out_path}")

    out_path = os.path.join(out_dir, "fscores_all.png")
    plot_all_fscores(rows, out_path)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()