"""
Aggregate ablation runs into a results table and comparison figures.

Reads every results/<run_name>/training_log.csv and summary.json, then emits:
  results/ABLATION_TABLE.md   - the table for the paper
  results/ablation_comparison.png   - val mIoU / disease mIoU / loss curves
  results/ablation_bars.png         - final-metric bar chart per configuration

Run after training:
    python scripts/08_compare_runs.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUN_LABELS = {
    "full": "Full model (CBAM + sev + consistency)",
    "no_cbam": "- CBAM",
    "no_cons": "- consistency loss",
    "no_sev": "- severity head (seg only)",
    "no_pretrained": "- pretrained encoder",
}

RUN_ORDER = ["full", "no_cbam", "no_cons", "no_sev", "no_pretrained"]


def load_runs(results_root):
    runs = {}
    for summary_path in sorted(results_root.glob("*/summary.json")):
        name = summary_path.parent.name
        csv_path = summary_path.parent / "training_log.csv"
        if not csv_path.exists():
            continue

        rows = list(csv.DictReader(csv_path.open()))
        if not rows:
            continue

        summary = json.loads(summary_path.read_text())
        # Best epoch, by validation mIoU.
        best_idx = int(np.argmax([float(r["Val_mIoU"]) for r in rows]))
        best = rows[best_idx]

        runs[name] = {
            "summary": summary,
            "rows": rows,
            "best_epoch": int(best["Epoch"]),
            "best_miou": float(best["Val_mIoU"]),
            "best_mdice": float(best["Val_mDice"]),
            "best_acc": float(best["Val_Acc"]),
            "best_disease_iou": float(best["Val_Disease_mIoU"]),
            "best_sev_mae": float(best["Sev_MAE"]),
            "final_seg_loss": float(rows[-1]["Seg_Loss"]),
            "params_M": summary.get("params_M"),
            "per_class_iou": summary.get("per_class_iou", {}),
            "ablation": summary.get("ablation", {}),
        }
    return runs


def write_table(runs, out_path):
    order = [r for r in RUN_ORDER if r in runs] + \
            [r for r in runs if r not in RUN_ORDER]

    lines = [
        "# Ablation results",
        "",
        "All numbers are validation metrics at each run's best epoch (selected by",
        "validation mIoU). Dataset: potato leaf disease, 758 train / 108 valid",
        "images, 5 classes.",
        "",
        "| Configuration | Params | Best epoch | mIoU | Dice | Disease mIoU | Sev MAE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        r = runs[name]
        label = RUN_LABELS.get(name, name)
        lines.append(
            f"| {label} | {r['params_M']:.2f}M | {r['best_epoch']} | "
            f"{r['best_miou']:.4f} | {r['best_mdice']:.4f} | "
            f"**{r['best_disease_iou']:.4f}** | {r['best_sev_mae']:.4f} |"
        )

    lines += ["", "## Per-class IoU (full model)", ""]
    full = runs.get("full")
    if full and full["per_class_iou"]:
        lines.append("| Class | IoU |")
        lines.append("|---|---|")
        for cls, iou in full["per_class_iou"].items():
            lines.append(f"| {cls} | {iou:.4f} |")

    lines += [
        "",
        "## Per-class IoU across runs",
        "",
    ]
    classes = list(full["per_class_iou"].keys()) if full else []
    if classes:
        lines.append("| Configuration | " + " | ".join(classes) + " |")
        lines.append("|---" * (len(classes) + 1) + "|")
        for name in order:
            pc = runs[name]["per_class_iou"]
            if pc:
                cells = " | ".join(f"{pc.get(c, float('nan')):.4f}" for c in classes)
                lines.append(f"| {RUN_LABELS.get(name, name)} | {cells} |")

    out_path.write_text("\n".join(lines) + "\n")
    return order


def plot_curves(runs, order, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for name in order:
        rows = runs[name]["rows"]
        ep = [int(r["Epoch"]) for r in rows]
        label = RUN_LABELS.get(name, name)
        axes[0][0].plot(ep, [float(r["Val_mIoU"]) for r in rows], label=label)
        axes[0][1].plot(ep, [float(r["Val_Disease_mIoU"]) for r in rows],
                        label=label)
        axes[1][0].plot(ep, [float(r["Seg_Loss"]) for r in rows], label=label)
        axes[1][1].plot(ep, [float(r["Sev_MAE"]) for r in rows], label=label)

    titles = [
        ("Validation mIoU", "(all 5 classes)"),
        ("Validation Disease mIoU", "(mean over 4 disease classes)"),
        ("Segmentation Loss", "(train, weighted CE)"),
        ("Severity MAE", "(vs continuous lesion coverage)"),
    ]
    for ax, (title, sub) in zip(axes.flat, titles):
        ax.set_title(f"{title}\n{sub}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0][0].set_ylim(0, 1)
    axes[0][1].set_ylim(0, 1)
    axes[1][1].set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bars(runs, order, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    metrics = [
        ("mIoU", "best_miou", "Validation mIoU"),
        ("Disease mIoU", "best_disease_iou", "Validation Disease mIoU"),
        ("Sev MAE", "best_sev_mae", "Severity MAE (lower better)"),
    ]

    labels = [RUN_LABELS.get(n, n).replace(" (CBAM + sev + consistency)", "")
              for n in order]
    colors = ["#2b6cb0" if n == "full" else "#a0aec0" for n in order]

    for ax, (short, key, title) in zip(axes, metrics):
        vals = [runs[n][key] for n in order]
        bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="#2d3748")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    results_root = Path(args.results_root)
    runs = load_runs(results_root)
    if not runs:
        print("No completed runs found. Train first.")
        return

    out_dir = results_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    order = write_table(runs, out_dir / "ABLATION_TABLE.md")
    plot_curves(runs, order, out_dir / "ablation_curves.png")
    plot_bars(runs, order, out_dir / "ablation_bars.png")

    print(f"Aggregated {len(runs)} runs -> {out_dir}\n")
    header = f"{'run':<16}{'params':>9}{'ep':>5}{'mIoU':>9}{'Dice':>9}{'disIoU':>9}{'sevMAE':>9}"
    print(header)
    print("-" * len(header))
    for name in order:
        r = runs[name]
        print(f"{name:<16}{r['params_M']:>8.2f}M{r['best_epoch']:>5}"
              f"{r['best_miou']:>9.4f}{r['best_mdice']:>9.4f}"
              f"{r['best_disease_iou']:>9.4f}{r['best_sev_mae']:>9.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_root",
                   default=str(PROJECT_ROOT / "results"))
    a = p.parse_args()
    main(a)
