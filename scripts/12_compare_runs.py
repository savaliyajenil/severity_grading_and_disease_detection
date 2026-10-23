"""
Aggregate the plan's ablation grid into a results table and comparison figures.

Reads results/<run>/test_results.json (PlantSeg test split) and
training_log.csv (validation curves), and emits:

  results/comparison/ABLATION_TABLE.md      table for the paper
  results/comparison/ablation_curves.png     validation curves per configuration
  results/comparison/ablation_bars.png       test-split metrics per configuration
  results/comparison/severity_grade.png      severity grading agreement
  results/comparison/full_model_curves.png   full-model training detail

Run:
    python scripts/12_compare_runs.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS = [
    ("baseline_unet", "Baseline U-Net\n(no CBAM, no severity head)"),
    ("cbam_only", "+ CBAM\n(severity head off)"),
    ("no_consistency", "+ CBAM + severity head\n(no consistency loss)"),
    ("full", "Full SAMTSeg\n(CBAM + severity + consistency)"),
    ("no_pretrained", "Full SAMTSeg\n(no pretrained encoder)"),
    ("plantseg_novelty_final", "PlantSeg Novelty\n(ResNet101, 512², PolyLR, Novelty)"),
]

PLAN_TARGETS = {"mIoU": 0.4452, "mDice": 0.80, "sev_acc": 0.80}


def load(results_root):
    out = {}
    for name, label in RUNS:
        d = results_root / name
        tj, sj = d / "test_results.json", d / "summary.json"
        if not tj.exists():
            print(f"  skip {name}: no test_results.json")
            continue
        entry = {
            "label": label,
            "test": json.loads(tj.read_text()),
            "summary": json.loads(sj.read_text()) if sj.exists() else {},
        }
        csv_path = d / "training_log.csv"
        if csv_path.exists():
            import csv as _csv
            entry["rows"] = list(_csv.DictReader(csv_path.open()))
        out[name] = entry
    return out


def write_table(runs, out_path):
    L = [
        "# Ablation results — SAMTSeg",
        "",
        "Evaluated on the **PlantSeg test split** (160 images, potato + tomato, 9 diseases).",
        "Training data: PlantSeg train (563) + harmonized Roboflow potato (623).",
        "Validation: PlantSeg val (90 images), used for checkpoint selection.",
        "",
        "Checkpoints selected by best validation mIoU; the test split was evaluated once per run.",
        "",
        "| Configuration | Params | Best val mIoU | **Test mIoU** | Test disease mIoU | Test Dice | Test disease F1 | Test sev. acc | Test sev. ±1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, _ in RUNS:
        if name not in runs:
            continue
        r = runs[name]
        t = r["test"]
        s = r["summary"]
        L.append(
            f"| {r['label'].replace(chr(10), ' ')} | {s.get('params_M', 0):.2f}M | "
            f"{s.get('best_val_miou', 0):.4f} | **{t['test_mIoU']:.4f}** | "
            f"{t['test_disease_mIoU']:.4f} | {t['test_mDice']:.4f} | "
            f"{t['test_disease_F1']:.4f} | {t['test_sev_grade_acc']:.4f} | "
            f"{t['test_sev_within1_acc']:.4f} |"
        )

    full = runs.get("full")
    if full:
        t = full["test"]
        L += [
            "",
            "## Plan targets vs achieved (full model)",
            "",
            "| Metric | Plan target | Achieved | Met |",
            "|---|---|---|---|",
            f"| mIoU | {PLAN_TARGETS['mIoU']:.4f} (beat PlantSeg baseline) | {t['test_mIoU']:.4f} | "
            f"{'yes' if t['test_mIoU'] > PLAN_TARGETS['mIoU'] else 'NO'} |",
            f"| Dice | {PLAN_TARGETS['mDice']:.2f} | {t['test_mDice']:.4f} | "
            f"{'yes' if t['test_mDice'] > PLAN_TARGETS['mDice'] else 'NO'} |",
            f"| Severity grading acc | {PLAN_TARGETS['sev_acc']:.2f} | {t['test_sev_grade_acc']:.4f} | "
            f"{'yes' if t['test_sev_grade_acc'] > PLAN_TARGETS['sev_acc'] else 'NO'} |",
        ]

        L += ["", "## Per-class IoU / F1 on PlantSeg test (full model)", "",
              "| Class | IoU | F1 |", "|---|---|---|"]
        for cls, iou in t["per_class_iou"].items():
            if iou > 0:
                L.append(f"| {cls} | {iou:.4f} | {t['per_class_f1'][cls]:.4f} |")

    L += [
        "",
        "## Severity grading accuracy by grade (full model)",
        "",
        "Not yet computed — see `severity_grade.png` for the confusion-style view.",
    ]

    out_path.write_text("\n".join(L) + "\n")


def plot_curves(runs, out_path):
    with_rows = [(n, r) for n, r in runs.items() if "rows" in r]
    if not with_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for name, r in with_rows:
        rows = r["rows"]
        ep = [int(x["Epoch"]) for x in rows]
        axes[0][0].plot(ep, [float(x["Seg_Loss"]) for x in rows],
                        label=r["label"].replace("\n", " "))
        axes[0][1].plot(ep, [float(x["Val_mIoU"]) for x in rows],
                        label=r["label"].replace("\n", " "))
        axes[1][0].plot(ep, [float(x["Val_Disease_mIoU"]) for x in rows],
                        label=r["label"].replace("\n", " "))
        axes[1][1].plot(ep, [float(x["Val_Sev_Acc"]) for x in rows],
                        label=r["label"].replace("\n", " "))

    axes[0][0].set_title("Segmentation loss (train, weighted CE)")
    axes[0][1].set_title("Validation mIoU")
    axes[0][1].axhline(PLAN_TARGETS["mIoU"], ls="--", c="k",
                       label=f"PlantSeg baseline {PLAN_TARGETS['mIoU']}")
    axes[0][1].legend(fontsize=7)
    axes[1][0].set_title("Validation disease mIoU")
    axes[1][1].set_title("Validation severity grading accuracy")
    axes[1][1].axhline(PLAN_TARGETS["sev_acc"], ls="--", c="k",
                       label=f"target {PLAN_TARGETS['sev_acc']}")
    axes[1][1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bars(runs, out_path):
    names = [n for n, _ in RUNS if n in runs]
    labels = [runs[n]["label"] for n in names]
    short = [l.replace("\n", " ") for l in labels]

    metrics = [
        ("test_mIoU", "Test mIoU", PLAN_TARGETS["mIoU"]),
        ("test_disease_mIoU", "Test disease mIoU", None),
        ("test_mDice", "Test Dice", PLAN_TARGETS["mDice"]),
        ("test_sev_grade_acc", "Test severity accuracy", PLAN_TARGETS["sev_acc"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.8))
    colors = ["#a0aec0" if n != "full" else "#2b6cb0" for n in names]

    for ax, (key, title, target) in zip(axes, metrics):
        vals = [runs[n]["test"][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="#2d3748")
        if target is not None:
            ax.axhline(target, ls="--", c="crimson", lw=1.2,
                       label=f"target {target}")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, rotation=18, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, max(0.75, max(vals) * 1.25))
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_severity(runs, out_path):
    """Test severity grading accuracy, exact vs within-one-grade."""
    full = runs.get("full")
    if not full:
        return
    acc = full["test"]["test_sev_grade_acc"]
    within = full["test"]["test_sev_within1_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [acc, within]
    bars = ax.bar(["exact grade", "within 1 grade"], vals,
                  color=["#2b6cb0", "#38a169"], edgecolor="#2d3748")
    ax.axhline(PLAN_TARGETS["sev_acc"], ls="--", c="crimson",
               label=f"plan target {PLAN_TARGETS['sev_acc']}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy on PlantSeg test")
    ax.set_title("Severity grading accuracy (0-1-3-5-7-9 scale)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    results_root = Path(args.results_root)
    runs = load(results_root)
    if not runs:
        print("No completed runs found.")
        return

    out_dir = results_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_table(runs, out_dir / "ABLATION_TABLE.md")
    plot_curves(runs, out_dir / "ablation_curves.png")
    plot_bars(runs, out_dir / "ablation_bars.png")
    plot_severity(runs, out_dir / "severity_grade.png")

    print(f"\n{'run':<18}{'val mIoU':>10}{'test mIoU':>11}{'disIoU':>9}"
          f"{'Dice':>9}{'sevAcc':>9}")
    print("-" * 66)
    for name, _ in RUNS:
        if name not in runs:
            continue
        r = runs[name]
        s = r["summary"]
        t = r["test"]
        print(f"{name:<18}{s.get('best_val_miou', 0):>10.4f}{t['test_mIoU']:>11.4f}"
              f"{t['test_disease_mIoU']:>9.4f}{t['test_mDice']:>9.4f}"
              f"{t['test_sev_grade_acc']:>9.4f}")
    print(f"\nfigures + table -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    a = p.parse_args()
    main(a)
