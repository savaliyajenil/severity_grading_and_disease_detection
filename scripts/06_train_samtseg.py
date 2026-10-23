"""
Training / evaluation entry point for SAMTSeg.

Trains the severity-aware multi-task segmentation network on the Roboflow potato
leaf disease dataset, with the two novelty components ablatable:

    --no_cbam              drop CBAM from the bottleneck and decoder blocks
    --lambda_sev 0         drop severity supervision entirely
    --lambda_cons 0        drop the severity-consistency loss
    --encoder scratch      from-scratch baseline instead of pretrained ResNet

Every run writes to results/<run_name>/ with its own CSV, so ablation rows can be
compared directly.

Example:
    python scripts/06_train_samtseg.py --run_name baseline_full --epochs 40
    python scripts/06_train_samtseg.py --run_name ablation_nocbam --no_cbam
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (NUM_CLASSES, SEVERITY_THRESHOLDS,
                              PotatoLeafSegDataset)
from src.data.transforms import (get_training_augmentation,
                                 get_validation_augmentation)
from src.models.samt_seg import (SAMTSeg, OrdinalSeverityLoss,
                                 SegmentationMetrics,
                                 SeverityConsistencyLoss,
                                 logits_to_severity, ordinal_grade)

logger = logging.getLogger("train")

DEFAULT_DATA_ROOT = PROJECT_ROOT / "potato-leaf-disease-segmentation-4g26g"
CLASS_NAMES = ["healthy/background", "Early Blight", "Late Blight",
               "Leaf Miner", "Nematode"]

CSV_COLUMNS = ["Epoch", "Seg_Loss", "Sev_Loss", "Consistency_Loss",
               "Total_Loss", "Val_Seg_Loss", "Val_mIoU", "Val_mDice",
               "Val_Acc", "Val_Disease_mIoU", "Sev_MAE"]


# ==============================================================================
# Setup
# ==============================================================================
def setup_logging(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train.log", mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=handlers,
        force=True,
    )


def build_loaders(args):
    train_ds = PotatoLeafSegDataset(
        args.data_root, split="train", transforms=get_training_augmentation()
    )
    valid_ds = PotatoLeafSegDataset(
        args.data_root, split="valid", transforms=get_validation_augmentation()
    )

    if args.class_weights == "auto":
        counts = train_ds.class_pixel_counts()
        total = counts.sum()
        weights = total / (NUM_CLASSES * np.maximum(counts, 1))
        weights = weights / weights.mean()
        logger.info("class pixel counts: %s", counts.tolist())
        logger.info("class weights:      %s",
                    np.round(weights, 4).tolist())
        class_weights = torch.tensor(weights, dtype=torch.float32)
    elif args.class_weights == "none":
        class_weights = None
    else:
        class_weights = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, valid_loader, class_weights


def build_model(args, device):
    model = SAMTSeg(
        num_classes=args.num_classes,
        encoder=args.encoder,
        pretrained=not args.no_pretrained,
        num_thresholds=len(SEVERITY_THRESHOLDS),
        use_cbam=not args.no_cbam,
    ).to(device)
    return model


# ==============================================================================
# Train / evaluate
# ==============================================================================
def run_epoch(model, loader, device, seg_criterion, sev_criterion,
              consist_criterion, optimizer=None, scaler=None, args=None,
              desc=""):
    """
    One pass over a loader. If `optimizer` is given, this is a training epoch;
    otherwise an evaluation pass under no_grad.
    """
    training = optimizer is not None
    model.train(training)

    totals = {"seg": 0.0, "sev": 0.0, "cons": 0.0, "total": 0.0}
    sev_abs_err = 0.0
    n_batches = 0
    metrics = SegmentationMetrics(args.num_classes)

    pbar = tqdm(loader, desc=desc, leave=False)
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            sev_targets = batch["severity"].to(device, non_blocking=True)
            coverage = batch["coverage"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=args.amp):
                pred_masks, sev_logits = model(images)
                pred_masks = pred_masks.float()

                loss_seg = seg_criterion(pred_masks, masks)
                loss_sev = sev_criterion(sev_logits, sev_targets)

                loss_total = loss_seg
                loss_cons = torch.zeros((), device=device)
                if args.lambda_cons > 0:
                    sev_scalar = logits_to_severity(sev_logits)
                    loss_cons = consist_criterion(pred_masks, sev_scalar)
                    loss_total = loss_total + args.lambda_cons * loss_cons
                if args.lambda_sev > 0:
                    loss_total = loss_total + args.lambda_sev * loss_sev

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_total).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                metrics.update(pred_masks, masks)
                # Severity MAE against continuous coverage, in percent.
                sev_pred = logits_to_severity(sev_logits)
                sev_abs_err += (sev_pred - coverage.view(-1, 1)).abs().mean().item()

            totals["seg"] += loss_seg.item()
            totals["sev"] += loss_sev.item()
            totals["cons"] += float(loss_cons)
            totals["total"] += loss_total.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    result = metrics.compute()
    result["sev_mae"] = sev_abs_err / max(n_batches, 1)
    avg.update(result)
    return avg


def save_checkpoint(path, model, optimizer, scaler, epoch, args, metrics,
                    best_miou):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "best_miou": best_miou,
    }, path)


def log_csv(path, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)


def plot_curves(csv_path, png_path):
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return

    ep = [int(r["Epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0][0].plot(ep, [float(r["Total_Loss"]) for r in rows], color="red")
    axes[0][0].set_title("Total Train Loss")

    axes[0][1].plot(ep, [float(r["Seg_Loss"]) for r in rows], label="seg")
    axes[0][1].plot(ep, [float(r["Sev_Loss"]) for r in rows], label="sev")
    axes[0][1].plot(ep, [float(r["Consistency_Loss"]) for r in rows],
                    label="consistency")
    axes[0][1].set_title("Train Loss Components")
    axes[0][1].legend()

    axes[1][0].plot(ep, [float(r["Val_mIoU"]) for r in rows], label="mIoU",
                    color="blue")
    axes[1][0].plot(ep, [float(r["Val_Disease_mIoU"]) for r in rows],
                    label="disease mIoU", color="orange")
    axes[1][0].plot(ep, [float(r["Val_mDice"]) for r in rows], label="mDice",
                    color="green")
    axes[1][0].set_title("Validation Segmentation Metrics")
    axes[1][0].set_ylim(0, 1)
    axes[1][0].legend()

    axes[1][1].plot(ep, [float(r["Sev_MAE"]) for r in rows], color="purple")
    axes[1][1].set_title("Severity MAE vs coverage (lower better)")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.results_root) / args.run_name
    setup_logging(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "training_log.csv"
    if csv_path.exists() and not args.resume:
        csv_path.unlink()

    logger.info("device: %s", device)
    logger.info("run_name: %s", args.run_name)
    logger.info("ablation: cbam=%s lambda_sev=%.2f lambda_cons=%.2f encoder=%s "
                "pretrained=%s",
                not args.no_cbam, args.lambda_sev, args.lambda_cons,
                args.encoder, not args.no_pretrained)

    train_loader, valid_loader, class_weights = build_loaders(args)
    logger.info("train batches: %d  valid batches: %d",
                len(train_loader), len(valid_loader))

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params: %.2fM", n_params / 1e6)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        ignore_index=255)
    sev_criterion = OrdinalSeverityLoss()
    consist_criterion = SeverityConsistencyLoss(background_class_idx=0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler(args.amp_dtype, enabled=args.amp)

    start_epoch, best_miou = 0, -1.0
    if args.resume:
        ckpt_path = ckpt_dir / "last.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_miou = ckpt.get("best_miou", -1.0)
            logger.info("resumed from epoch %d (best mIoU %.4f)",
                        start_epoch, best_miou)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_stats = run_epoch(
            model, train_loader, device, seg_criterion, sev_criterion,
            consist_criterion, optimizer=optimizer, scaler=scaler, args=args,
            desc=f"epoch {epoch + 1}/{args.epochs} [train]",
        )
        val_stats = run_epoch(
            model, valid_loader, device, seg_criterion, sev_criterion,
            consist_criterion, optimizer=None, scaler=None, args=args,
            desc=f"epoch {epoch + 1}/{args.epochs} [valid]",
        )

        scheduler.step(val_stats["mIoU"])

        improved = val_stats["mIoU"] > best_miou
        if improved:
            best_miou = val_stats["mIoU"]

        log_csv(csv_path, [
            epoch + 1,
            f"{train_stats['seg']:.4f}", f"{train_stats['sev']:.4f}",
            f"{train_stats['cons']:.4f}", f"{train_stats['total']:.4f}",
            f"{val_stats['seg']:.4f}", f"{val_stats['mIoU']:.4f}",
            f"{val_stats['mDice']:.4f}", f"{val_stats['pixel_acc']:.4f}",
            f"{val_stats['mean_disease_iou']:.4f}",
            f"{val_stats['sev_mae']:.4f}",
        ])
        plot_curves(csv_path, run_dir / "training_curves.png")

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scaler, epoch,
                        args, val_stats, best_miou)
        if improved:
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scaler,
                            epoch, args, val_stats, best_miou)

        logger.info(
            "epoch %3d | %.1fs | train loss %.4f (seg %.4f sev %.4f cons %.4f) "
            "| val mIoU %.4f dice %.4f acc %.4f disIoU %.4f sevMAE %.4f%s",
            epoch + 1, time.time() - t0, train_stats["total"],
            train_stats["seg"], train_stats["sev"], train_stats["cons"],
            val_stats["mIoU"], val_stats["mDice"], val_stats["pixel_acc"],
            val_stats["mean_disease_iou"], val_stats["sev_mae"],
            "  <- best" if improved else "",
        )

    summary = {
        "run_name": args.run_name,
        "params_M": n_params / 1e6,
        "best_val_miou": best_miou,
        "per_class_iou": dict(zip(CLASS_NAMES, val_stats["per_class_iou"])),
        "ablation": {
            "cbam": not args.no_cbam,
            "lambda_sev": args.lambda_sev,
            "lambda_cons": args.lambda_cons,
            "encoder": args.encoder,
            "pretrained": not args.no_pretrained,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("best val mIoU: %.4f  ->  %s", best_miou, run_dir)
    return summary


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--encoder", type=str, default="resnet34",
                   choices=["resnet18", "resnet34", "scratch"])
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_cbam", action="store_true")
    p.add_argument("--lambda_sev", type=float, default=1.0)
    p.add_argument("--lambda_cons", type=float, default=0.5)
    p.add_argument("--class_weights", type=str, default="auto",
                   choices=["auto", "none"])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--amp_dtype", type=str, default="cuda")
    p.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--results_root", type=str,
                   default=str(PROJECT_ROOT / "results"))
    p.add_argument("--run_name", type=str, default="default")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
