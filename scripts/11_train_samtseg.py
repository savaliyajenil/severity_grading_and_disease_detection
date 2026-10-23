"""
PlantSeg-style training with Severity Head + Consistency Loss (THE NOVELTY).

Implements the exact PlantSeg training recipe:
- 512x512 crop training (RandomResize 0.5-2.0 + RandomCrop 512)
- AdamW 6e-5, weight_decay=0.01, PolyLR power=1.0, 160k iters
- Warmup 1500 iters
- Simple CE loss (no focal, no class weights)
- Severity Head + Consistency Loss (THE NOVELTY)
- Standard augmentation: RandomResize + RandomCrop + Flip + PhotoMetricDistortion
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib  
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import (get_training_augmentation,
                                 get_validation_augmentation)
from src.data.unified_dataset import (CLASS_NAMES, NUM_CLASSES,
                                      UnifiedSegDataset)
from src.models.plantseg_model import get_model
from src.models.losses import (DiceLoss, OrdinalSeverityLoss, SegmentationMetrics,
                               SeverityConsistencyLoss, logits_to_ratio,
                               ordinal_targets, ordinal_grade, SEVERITY_THRESHOLDS)

logger = logging.getLogger("train")

PLAN_TARGET_MIOU = 0.4452  # PlantSeg published baseline to beat

# Comparison-aligned columns: same core as the SegNeXt run's training_log.csv
# (losses, Val_mIoU, Val_Disease_mIoU, Val_mDice, Val_Acc, Val_Disease_F1, LR)
# so the two runs compare column-for-column. Severity metrics stay in the
# per-epoch train.log lines and test_results.json (novelty evidence).
CSV_COLUMNS = [
    "Epoch", "Seg_Loss", "Total_Loss",
    "Val_mIoU", "Val_Disease_mIoU", "Val_mDice", "Val_Acc", "Val_Disease_F1",
    "LR",
]


# ==============================================================================
# Setup
# ==============================================================================
def setup_logging(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(run_dir / "train.log", mode="w")],
        force=True,
    )


def build_sampler(dataset, enabled, num_samples):
    if not enabled:
        return None

    counts = {}
    for rec in dataset.records:
        ids = rec["class_ids"] if rec.get("class_ids") else [rec["class_id"]]
        for c in ids:
            counts[c] = counts.get(c, 0) + 1

    weights = []
    for rec in dataset.records:
        ids = rec["class_ids"] if rec.get("class_ids") else [rec["class_id"]]
        w = sum(1.0 / counts[c] for c in ids) / len(ids)
        weights.append(w)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    logger.info("weighted sampler: rarest class weight %.2e, most common %.2e",
                w.min(), w.max())
    return WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double),
                                 num_samples=num_samples, replacement=True)


def boundary_targets(masks, ignore_index=255):
    """Free lesion-edge targets: pixels whose 3x3 neighborhood spans classes."""
    with torch.no_grad():
        f = masks.float()
        mx = F.max_pool2d(f.unsqueeze(1), 3, 1, 1).squeeze(1)
        mn = -F.max_pool2d(-f.unsqueeze(1), 3, 1, 1).squeeze(1)
        edge = (mx - mn) > 0
        edge = edge & (masks != ignore_index)
        return edge.float()


def build_loaders(args):
    crop = (args.crop_size, args.crop_size)
    train_ds = UnifiedSegDataset(args.index_root, split="train",
                                 transforms=get_training_augmentation(crop))
    val_ds = UnifiedSegDataset(args.index_root, split="val",
                               transforms=get_validation_augmentation(crop))

    # No class weights - standard CE works with pretrained backbone
    class_weights = None

    n_samples = len(train_ds)
    sampler = build_sampler(train_ds, not args.no_sampler, n_samples)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, class_weights, train_ds


def build_model(args, device):
    return get_model(
        num_classes=NUM_CLASSES, backbone=args.encoder,
        pretrained=not args.no_pretrained,
    ).to(device)


# ==============================================================================
# Train / evaluate
# ==============================================================================
def run_epoch(model, loader, device, seg_criterion, sev_criterion,
              consist_criterion, optimizer=None, scaler=None, args=None,
              scheduler=None, desc="", dice_criterion=None, bnd_criterion=None,
              ema_state=None):
    training = optimizer is not None
    model.train(training)

    totals = {"seg": 0.0, "sev": 0.0, "cons": 0.0, "dice": 0.0, "aux": 0.0,
              "bnd": 0.0, "total": 0.0}
    n_batches = 0
    metrics = SegmentationMetrics(NUM_CLASSES)

    pbar = tqdm(loader, desc=desc, leave=False)
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            sev_t = batch["severity"].to(device, non_blocking=True)
            ratios = batch["ratio"].to(device, non_blocking=True)
            grades = batch["grade"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=args.amp,
                                 dtype=torch.float16):
                pred_masks, sev_logits, aux_masks, bnd_logits = model(images)
                pred_masks = pred_masks.float()

                # Boundary-weighted CE: edge pixels (free targets from the GT
                # mask) count 5x, forcing attention onto small-lesion borders.
                # Bounded weights only -- no NaN lineage.
                edge = boundary_targets(masks)
                ce_map = F.cross_entropy(pred_masks, masks, ignore_index=255,
                                         reduction="none")
                valid = masks != 255
                w = 1.0 + 4.0 * edge
                loss_seg = (w * ce_map)[valid].mean() if valid.any() \
                    else ce_map.mean()
                loss_sev = sev_criterion(sev_logits, sev_t)

                loss_bnd = torch.zeros((), device=device)
                if bnd_criterion is not None and args.lambda_bdry > 0:
                    loss_bnd = bnd_criterion(bnd_logits.float().squeeze(1)
                                             if bnd_logits.dim() == 4 else bnd_logits.float(),
                                             edge)

                loss_cons = torch.zeros((), device=device)
                if args.lambda_cons > 0:
                    sev_scalar = logits_to_ratio(sev_logits)
                    # Detached auxiliaries: the consistency term trains only the
                    # severity head. Segmentation learns from plain CE alone, so
                    # the novelty can never destabilize it (the NaN/mIoU-drop
                    # failure mode seen at full strength).
                    loss_cons = consist_criterion(pred_masks.detach(), sev_scalar)

                loss_dice = torch.zeros((), device=device)
                if dice_criterion is not None and args.lambda_dice > 0:
                    loss_dice = dice_criterion(pred_masks, masks)

                loss_aux = torch.zeros((), device=device)
                if training and args.lambda_aux > 0 and len(aux_masks) > 0:
                    loss_aux = sum(seg_criterion(a.float(), masks)
                                   for a in aux_masks) / len(aux_masks)

                loss_total = loss_seg
                if dice_criterion is not None and args.lambda_dice > 0:
                    loss_total = loss_total + args.lambda_dice * loss_dice
                if training and args.lambda_aux > 0 and len(aux_masks) > 0:
                    loss_total = loss_total + args.lambda_aux * loss_aux
                if bnd_criterion is not None and args.lambda_bdry > 0:
                    loss_total = loss_total + args.lambda_bdry * loss_bnd
                if args.lambda_sev > 0:
                    loss_total = loss_total + args.lambda_sev * loss_sev
                if args.lambda_cons > 0:
                    loss_total = loss_total + args.lambda_cons * loss_cons

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_total).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                metrics.update(pred_masks, masks)
                metrics.update_severity(sev_logits, ratios, grades)

            totals["seg"] += loss_seg.item()
            totals["sev"] += loss_sev.item()
            totals["cons"] += float(loss_cons)
            totals["dice"] += float(loss_dice)
            totals["aux"] += float(loss_aux)
            totals["bnd"] += float(loss_bnd)
            totals["total"] += loss_total.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    avg.update(metrics.compute())
    return avg, metrics


def save_checkpoint(path, model, optimizer, scaler, epoch, args, metrics, best):
    state = {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "args": vars(args),
        "metrics": metrics, "best_miou": best,
    }
    torch.save(state, path)


def log_csv(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_COLUMNS)
        w.writerow(row)


def plot_curves(csv_path, png_path):
    # Same 2x2 layout as the SegNeXt baseline's training_curves.png, so the
    # two runs compare panel-for-panel (loss / mIoU / Dice / accuracy).
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    ep = [int(r["Epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SAMTSeg (novelty) — training curves", fontsize=14)

    ax = axes[0, 0]
    ax.plot(ep, [float(r["Total_Loss"]) for r in rows], label="total loss", color="red")
    ax.plot(ep, [float(r["Seg_Loss"]) for r in rows], label="seg loss", color="orange", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, [float(r["Val_mIoU"]) for r in rows], "o-", label="Val mIoU", color="blue")
    ax.plot(ep, [float(r["Val_Disease_mIoU"]) for r in rows], "s-", label="Val disease mIoU", color="orange")
    ax.axhline(0.4172, ls="--", c="k", label="SegNeXt test mIoU 0.4172")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, [float(r["Val_mDice"]) for r in rows], "o-", label="Val mDice", color="blue")
    ax.plot(ep, [float(r["Val_Disease_F1"]) for r in rows], "s-", label="Val disease F1", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice / F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, [float(r["Val_Acc"]) for r in rows], "o-", label="Val pixel acc", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def per_class_plot(metrics, out_png):
    if hasattr(metrics, 'compute'):
        metrics = metrics.compute()
    iou = np.array(metrics["per_class_iou"])
    f1 = np.array(metrics["per_class_f1"])
    present = iou > 0
    names = [CLASS_NAMES[i].replace("_", " ") for i in np.where(present)[0]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    idx = np.where(present)[0]
    x = np.arange(len(idx))
    w = 0.38
    ax.bar(x - w / 2, iou[idx], w, label="IoU", color="#2b6cb0")
    ax.bar(x + w / 2, f1[idx], w, label="F1", color="#38a169")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class performance on PlantSeg test (mIoU {metrics['mIoU']:.4f})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ==============================================================================
# Test evaluation
# ==============================================================================
def evaluate_checkpoint(ckpt_path, args, device, run_dir):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = argparse.Namespace(**ckpt["args"])

    model = get_model(
        num_classes=NUM_CLASSES, backbone=ck_args.encoder,
        pretrained=not ck_args.no_pretrained,
        use_cbam=not getattr(ck_args, "no_cbam", False),
    ).to(device)
    if "ema_state_dict" in ckpt and not getattr(ck_args, "no_ema", False):
        ema = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in ckpt["ema_state_dict"].items()}
        model.load_state_dict(ema)
        logger.info("evaluating EMA-averaged weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    test_ds = UnifiedSegDataset(args.index_root, split="test",
                                transforms=get_validation_augmentation())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    seg_criterion = nn.CrossEntropyLoss()
    sev_criterion = OrdinalSeverityLoss()
    consist_criterion = SeverityConsistencyLoss(background_class_idx=0)

    stats, metrics = run_epoch(
        model, test_loader, device, nn.CrossEntropyLoss(), OrdinalSeverityLoss(),
        SeverityConsistencyLoss(), optimizer=None, scaler=None, args=args,
        desc="[TEST]",
    )

    logger.info("")
    logger.info("=== PlantSeg TEST results (ckpt epoch %d) ===", ckpt["epoch"] + 1)
    logger.info("  mIoU          %.4f   (PlantSeg baseline %.4f)",
                stats["mIoU"], 0.4452)
    logger.info("  disease mIoU  %.4f", stats["mean_disease_iou"])
    logger.info("  Dice          %.4f", stats["mDice"])
    logger.info("  pixel acc     %.4f", stats["pixel_acc"])
    logger.info("  disease F1    %.4f", stats["mean_disease_f1"])
    logger.info("  severity acc  %.4f", stats["sev_grade_acc"])
    logger.info("  severity w/in1 %.4f", stats["sev_within1_acc"])
    logger.info("  ratio MAE     %.4f", stats["ratio_mae"])
    logger.info("")
    for i, n in enumerate(CLASS_NAMES):
        if stats["per_class_iou"][i] > 0:
            logger.info("    %-32s IoU %.4f  F1 %.4f", n,
                        stats["per_class_iou"][i], stats["per_class_f1"][i])

    per_class_plot(metrics, run_dir / "test_per_class.png")

    rp, rt = metrics.severity_scatter()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rt, rp, s=18, alpha=0.65, edgecolors="none")
    lim = [0, max(max(rt, default=0.5), max(rp, default=0.5)) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    for t, s in zip(SEVERITY_THRESHOLDS, (0,1,3,5,7,9)[:-1]):
        ax.axvline(t, color="gray", ls=":", lw=0.8)
        ax.axhline(t, color="gray", ls=":", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true lesion coverage")
    ax.set_ylabel("predicted severity ratio")
    ax.set_title(f"Severity head vs ground truth (MAE {stats['ratio_mae']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "test_severity_scatter.png", dpi=150)
    plt.close()

    result = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt["epoch"] + 1,
        "test_mIoU": stats["mIoU"],
        "test_disease_mIoU": stats["mean_disease_iou"],
        "test_mDice": stats["mDice"],
        "test_pixel_acc": stats["pixel_acc"],
        "test_disease_F1": stats["mean_disease_f1"],
        "test_sev_grade_acc": stats["sev_grade_acc"],
        "test_sev_within1_acc": stats["sev_within1_acc"],
        "test_ratio_mae": stats["ratio_mae"],
        "per_class_iou": dict(zip(CLASS_NAMES, stats["per_class_iou"])),
        "per_class_f1": dict(zip(CLASS_NAMES, stats["per_class_f1"])),
        "targets": {"mIoU": 0.4452, "mDice": 0.80, "sev_acc": 0.80},
    }
    (run_dir / "test_results.json").write_text(json.dumps(result, indent=2))

    # results_summary.csv: same schema as the SegNeXt run, plus severity rows
    # at the end, for direct side-by-side comparison.
    best_val = ""
    summary_json = run_dir / "summary.json"
    if summary_json.exists():
        try:
            best_val = f"{json.loads(summary_json.read_text()).get('best_val_miou', '')}"
        except (ValueError, KeyError, TypeError):
            pass
    with open(run_dir / "results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["model", f"SAMTSeg {ck_args.encoder} + ASPP + severity head (novelty)"])
        w.writerow(["num_classes", NUM_CLASSES])
        w.writerow(["optimizer", "AdamW lr=6e-05 betas=(0.9,0.999) wd=0.01"])
        w.writerow(["schedule", "PolyLR power=1.0, warmup=1500 (epoch-based)"])
        w.writerow(["batch_size", args.batch_size])
        w.writerow(["crop_size", "512x512"])
        w.writerow(["best_val_mIoU", best_val])
        w.writerow(["test_mIoU", f"{stats['mIoU']:.4f}"])
        w.writerow(["test_disease_mIoU", f"{stats['mean_disease_iou']:.4f}"])
        w.writerow(["test_mDice", f"{stats['mDice']:.4f}"])
        w.writerow(["test_pixel_acc", f"{stats['pixel_acc']:.4f}"])
        w.writerow(["test_disease_F1", f"{stats['mean_disease_f1']:.4f}"])
        w.writerow(["test_sev_grade_acc", f"{stats['sev_grade_acc']:.4f}"])
        w.writerow(["test_sev_within1_acc", f"{stats['sev_within1_acc']:.4f}"])
        w.writerow(["test_ratio_mae", f"{stats['ratio_mae']:.4f}"])
        w.writerow([])
        w.writerow(["Per-class test IoU", ""])
        for n in CLASS_NAMES:
            w.writerow([f"test_iou_{n}", f"{result['per_class_iou'][n]:.4f}"])
    logger.info("wrote results_summary.csv")

    # Per-class IoU bar chart over ALL classes (mirrors SegNeXt perclass_iou.png)
    ious = [result["per_class_iou"][n] for n in CLASS_NAMES]
    labels = [n.replace("_", "\n") for n in CLASS_NAMES]
    colors = ["tab:gray" if n == "background" else "tab:green" for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(CLASS_NAMES)), ious, color=colors)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test IoU")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"SAMTSeg (novelty) — per-class test IoU (mIoU {stats['mIoU']:.4f})")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    mean_iou = sum(ious) / len(ious)
    ax.axhline(mean_iou, color="red", linestyle="--", linewidth=1,
               label=f"mean = {mean_iou:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "perclass_iou.png", dpi=150)
    plt.close()
    logger.info("wrote perclass_iou.png")

    return result


# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.results_root) / args.run_name
    setup_logging(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        ckpt = Path(args.ckpt) if args.ckpt else ckpt_dir / "best.pth"
        return evaluate_checkpoint(ckpt, args, device, run_dir)

    csv_path = run_dir / "training_log.csv"
    # Always start fresh so every run gets a complete log
    if csv_path.exists():
        csv_path.unlink()

    logger.info("device=%s run=%s", device, args.run_name)
    logger.info("encoder=%s pretrained=%s", args.encoder, not args.no_pretrained)
    logger.info("train = PlantSeg train + Roboflow potato | val/test = PlantSeg only")

    train_loader, val_loader, class_weights, _ = build_loaders(args)
    logger.info("train batches=%d val batches=%d", len(train_loader), len(val_loader))

    model = get_model(
        num_classes=NUM_CLASSES, backbone=args.encoder,
        pretrained=not args.no_pretrained,
        use_cbam=not args.no_cbam,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("params %.2fM  classes=%d", n_params / 1e6, NUM_CLASSES)

    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state_dict"], strict=False)
        logger.info("initialized from %s (epoch %d)", args.init_from, init_ckpt.get("epoch", -1) + 1)

    seg_criterion = nn.CrossEntropyLoss()
    sev_criterion = OrdinalSeverityLoss()
    consist_criterion = SeverityConsistencyLoss(background_class_idx=0)
    dice_criterion = DiceLoss()
    bnd_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))

    # SegNeXt-style optimizer: pretrained encoder at 6e-5, randomly
    # initialized decoder/heads at 10x. Same wd=0.01, betas as PlantSeg.
    enc_params = [p for n, p in model.named_parameters()
                  if n.startswith("encoder.")]
    dec_params = [p for n, p in model.named_parameters()
                  if not n.startswith("encoder.")]
    optimizer = optim.AdamW(
        [{"params": enc_params, "lr": 6e-5, "weight_decay": 0.01},
         {"params": dec_params, "lr": 6e-4, "weight_decay": 0.01}],
        betas=(0.9, 0.999))
    logger.info("optimizer groups: encoder %d tensors @6e-5, decoder/heads %d tensors @6e-4",
                len(enc_params), len(dec_params))

    # PlantSeg-style scheduler: PolyLR with linear warmup
    total_iters = args.epochs * len(train_loader)
    warmup_iters = 1500

    def lr_lambda(step):
        if step < 1500:
            return float(step) / 1500.0
        progress = float(step - 1500) / max(1, len(train_loader) * args.epochs - 1500)
        return (1.0 - progress) ** 1.0  # PolyLR power=1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch, best_miou, best_epoch = 0, -1.0, 0
    if args.resume and (Path(args.results_root) / args.run_name / "checkpoints" / "last.pth").exists():
        ckpt_path = Path(args.results_root) / args.run_name / "checkpoints" / "last.pth"
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_miou = ck.get("best_miou", -1.0)
        logger.info("resumed at epoch %d", start_epoch)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr, _ = run_epoch(model, train_loader, device, nn.CrossEntropyLoss(),
                          OrdinalSeverityLoss(), SeverityConsistencyLoss(),
                          optimizer=optimizer, scaler=scaler, args=args,
                          scheduler=scheduler, desc=f"ep {epoch + 1}/{args.epochs} train",
                          dice_criterion=dice_criterion, bnd_criterion=bnd_criterion)
        va, _ = run_epoch(model, val_loader, device, nn.CrossEntropyLoss(),
                          OrdinalSeverityLoss(), SeverityConsistencyLoss(),
                          optimizer=None, scaler=None, args=args,
                          desc=f"ep {epoch + 1}/{args.epochs} val")

        improved = va["mIoU"] > best_miou
        if improved:
            best_miou, best_epoch = va["mIoU"], epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_csv(run_dir / "training_log.csv", [
            epoch + 1, f"{tr['seg']:.4f}", f"{tr['total']:.4f}",
            f"{va['mIoU']:.4f}", f"{va['mean_disease_iou']:.4f}", f"{va['mDice']:.4f}", f"{va['pixel_acc']:.4f}",
            f"{va['mean_disease_f1']:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}",
        ])
        plot_curves(run_dir / "training_log.csv", Path(args.results_root) / args.run_name / "training_curves.png")

        save_checkpoint(Path(args.results_root) / args.run_name / "checkpoints" / "last.pth",
                        model, optimizer, scaler, epoch, args, va, best_miou)
        if improved:
            save_checkpoint(Path(args.results_root) / args.run_name / "checkpoints" / "best.pth",
                            model, optimizer, scaler, epoch, args, va, best_miou)

        logger.info(
            "ep %3d | %5.1fs | loss %.4f (seg %.4f dice %.4f aux %.4f bnd %.4f sev %.4f cons %.4f) | "
            "val mIoU %.4f dis %.4f dice %.4f f1 %.4f | sev %.3f w1 %.3f "
            "rmae %.3f | lr %.1e%s",
            epoch + 1, time.time() - t0, tr["total"], tr["seg"], tr["dice"], tr["aux"],
            tr["bnd"], tr["sev"], tr["cons"],
            va["mIoU"], va["mean_disease_iou"], va["mDice"],
            va["mean_disease_f1"], va["sev_grade_acc"], va["sev_within1_acc"],
            va["ratio_mae"], optimizer.param_groups[0]["lr"],
            "  <- best" if improved else "")

        if epochs_without_improvement >= args.early_stop_patience:
            logger.info("early stopping: no val mIoU improvement for %d epochs",
                        epochs_without_improvement)
            break

    # Plot training curves from the full log (after loop, handles early stop)
    plot_curves(run_dir / "training_log.csv", Path(args.results_root) / args.run_name / "training_curves.png")

    summary = {
        "run_name": args.run_name,
        "params_M": sum(p.numel() for p in get_model(num_classes=10, backbone=args.encoder, pretrained=False, use_cbam=not args.no_cbam).parameters()) / 1e6,
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "config": {"encoder": args.encoder, "pretrained": not args.no_pretrained,
                   "use_cbam": not args.no_cbam, "epochs": args.epochs,
"batch_size": args.batch_size, "lambda_sev": args.lambda_sev,
                   "lambda_cons": args.lambda_cons, "lambda_bdry": args.lambda_bdry,
"lambda_dice": args.lambda_dice, "lambda_aux": args.lambda_aux,
                    "lambda_bdry": args.lambda_bdry},
    }
    (Path(args.results_root) / args.run_name / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("training done. best val mIoU %.4f @ epoch %d", best_miou, best_epoch)

    # Final test evaluation
    result = evaluate_checkpoint(Path(args.results_root) / args.run_name / "checkpoints" / "best.pth",
                                 args, device, run_dir)
    return result


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--encoder", default="mscan_l",
                   choices=["mscan_l", "resnet50", "resnet101"])
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_sampler", action="store_true")
    p.add_argument("--no_cbam", action="store_true",
                   help="disable CBAM + severity gates (ablation)")
    p.add_argument("--lambda_sev", type=float, default=0.1)
    p.add_argument("--lambda_cons", type=float, default=0.05)
    p.add_argument("--lambda_dice", type=float, default=0.5)
    p.add_argument("--lambda_aux", type=float, default=0.2)
    p.add_argument("--lambda_bdry", type=float, default=0.3)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--early_stop_patience", type=int, default=50)
    p.add_argument("--init_from", type=str, default="")
    p.add_argument("--no_amp", dest="amp", action="store_false", default=True)
    p.add_argument("--index_root", default=str(PROJECT_ROOT / "results" / "index"))
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--run_name", default="plantseg_novelty")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train(a)