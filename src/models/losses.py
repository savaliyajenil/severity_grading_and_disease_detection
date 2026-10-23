"""
Loss functions for PlantSeg-style model with novelty (severity + consistency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.unified_dataset import SEVERITY_THRESHOLDS


class OrdinalSeverityLoss(nn.Module):
    """
    CORAL-style ordinal loss for severity regression.
    
    For K ordered thresholds t_1 < ... < t_K and target ratio c,
    each logit is trained with BCE against indicator 1[c > t_k].
    This preserves the ordering that a 3-way softmax would discard.
    """
    def __init__(self):
        super(OrdinalSeverityLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, sev_logits, sev_targets):
        # sev_logits: (B, K), sev_targets: (B, K) with 0/1 per threshold
        return self.bce(sev_logits, sev_targets)


def ordinal_targets(ratio, thresholds=SEVERITY_THRESHOLDS):
    """
    Convert continuous ratio to ordinal targets (CORAL encoding).

    Args:
        ratio: (B,) tensor of lesion coverage ratios
        thresholds: K thresholds for 6 ordinal levels (0,1,3,5,7,9)
    
    Returns:
        (B, K) tensor with 1.0 where ratio > threshold, else 0.0
    """
    if not isinstance(ratio, torch.Tensor):
        ratio = torch.tensor(ratio, dtype=torch.float32)
    targets = torch.stack([ratio > t for t in thresholds], dim=1).float()
    return targets


def ordinal_grade(logits, thresholds=SEVERITY_THRESHOLDS):
    """Decode ordinal logits to grade index in {0, ..., K}."""
    return (logits > 0).sum(dim=1).long()


def logits_to_ratio(sev_logits, thresholds=SEVERITY_THRESHOLDS):
    """
    Decode ordinal logits to continuous severity ratio in [0, 1].
    
    Uses midpoints between thresholds as interval representatives.
    """
    probs = torch.sigmoid(sev_logits)  # (B, K)
    bounds = torch.tensor([0.0] + list(thresholds) + [1.0], device=sev_logits.device)
    mids = (bounds[:-1] + bounds[1:]) / 2.0  # (K+1,)
    # Interval k chosen when prob_k >= 0.5
    idx = (probs >= 0.5).sum(dim=1)
    return mids[idx].unsqueeze(1)  # (B, 1)


class DiceLoss(nn.Module):
    """
    Multiclass Dice loss (stable, no class weights).

    Mean 1 - Dice over classes present in the batch's valid pixels.
    Complements cross-entropy by optimizing overlap directly; bounded in
    [0, 1] so it cannot destabilize training the way reweighting did.
    """
    def __init__(self, ignore_index=255, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits, targets: (B, H, W) indices
        valid = targets != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0.0
        probs = F.softmax(inputs, dim=1)
        with torch.no_grad():
            tgt = torch.zeros_like(probs)
            tgt_valid = targets.clamp(0, probs.size(1) - 1)
            tgt.scatter_(1, tgt_valid.unsqueeze(1), 1.0)
            tgt = tgt * valid.unsqueeze(1).float()
        probs = probs * valid.unsqueeze(1).float()
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        present = denom > 0
        dice = torch.ones_like(denom)
        dice[present] = (2.0 * inter[present] + self.eps) / (denom[present] + self.eps)
        return 1.0 - dice[present].mean()


# ==============================================================================
# Severity-Consistency Loss (THE NOVELTY)
# ==============================================================================
class SeverityConsistencyLoss(nn.Module):
    """
    Consistency Loss coupling segmentation and severity heads (THE NOVELTY).
    
    Forces the segmentation head's implied lesion ratio to match
    the severity head's prediction. This creates a feedback loop:
    
    1. Segmentation predicts lesion mask
    2. Severity head predicts lesion coverage ratio
    3. Consistency loss forces them to agree
    
    This is THE NOVELTY for the paper.
    """
    def __init__(self, background_class_idx=0):
        super(SeverityConsistencyLoss, self).__init__()
        self.bg_idx = background_class_idx
        self.mse = nn.MSELoss()

    def forward(self, predicted_logits, predicted_severity):
        """
        Args:
            predicted_logits: (B, C, H, W) segmentation logits
            predicted_severity: (B, 1) scalar severity ratio [0, 1]
        """
        B, C, H, W = predicted_logits.size()

        # Differentiable estimate: expected disease mass per pixel
        probs = F.softmax(predicted_logits, dim=1)
        disease_probs = 1.0 - probs[:, self.bg_idx, :, :]  # (B, H, W)
        inferred_severity = disease_probs.sum(dim=(1, 2)) / (H * W)
        inferred_severity = inferred_severity.view(-1, 1)

        # MSE between severity head output and segmentation-inferred ratio
        return self.mse(predicted_severity, inferred_severity)


# ==============================================================================
# Metrics
# ==============================================================================
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes,
                                     dtype=torch.int64)
        self.sev_pred = []
        self.sev_true = []
        self.ratio_pred = []
        self.ratio_true = []

    @torch.no_grad()
    def update(self, pred_logits, true_masks):
        preds = torch.argmax(pred_logits, dim=1).cpu()
        targets = true_masks.cpu()

        valid = (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]

        idx = targets * self.num_classes + preds
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion.float()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        union = tp + fp + fn
        valid = union > 0

        iou = torch.zeros(self.num_classes)
        iou[valid] = tp[valid] / union[valid]

        dice_denom = 2 * tp + fp + fn
        dice = torch.zeros(self.num_classes)
        dice[valid] = (2 * tp[valid]) / dice_denom[valid]

        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        f1 = torch.zeros(self.num_classes)
        p_denom = tp + fp
        r_denom = tp + fn
        precision[p_denom > 0] = tp[p_denom > 0] / p_denom[p_denom > 0]
        recall[r_denom > 0] = tp[r_denom > 0] / r_denom[r_denom > 0]
        denom = precision + recall
        f1[denom > 0] = (2 * precision * recall)[denom > 0] / denom[denom > 0]

        total = tp.sum()
        acc = (total / cm.sum()).item() if cm.sum() > 0 else 0.0

        out = {
            "pixel_acc": acc,
            "mIoU": iou[valid].mean().item() if valid.any() else 0.0,
            "mDice": dice[valid].mean().item() if valid.any() else 0.0,
            "per_class_iou": iou.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "mean_disease_iou": iou[1:][union[1:] > 0].mean().item() if (union[1:] > 0).any() else 0.0,
            "mean_disease_f1": f1[1:][union[1:] > 0].mean().item() if (union[1:] > 0).any() else 0.0,
        }

        # Severity metrics
        if self.sev_pred:
            pred_g = torch.cat(self.sev_pred)
            true_g = torch.cat(self.sev_true)
            out["sev_grade_acc"] = (pred_g == true_g).float().mean().item()
            out["sev_within1_acc"] = ((pred_g - true_g).abs() <= 1).float().mean().item()
            out["sev_grade_mae"] = (pred_g - true_g).abs().float().mean().item()
        else:
            out["sev_grade_acc"] = 0.0
            out["sev_within1_acc"] = 0.0
            out["sev_grade_mae"] = 0.0

        if self.ratio_pred:
            p = torch.cat(self.ratio_pred)
            t = torch.cat(self.ratio_true)
            out["ratio_mae"] = (p - t).abs().mean().item()
        else:
            out["ratio_mae"] = 0.0

        return out

    @torch.no_grad()
    def update_severity(self, sev_logits, ratios, grades):
        """Accumulate severity predictions for grading metrics."""
        self.sev_pred.append(ordinal_grade(sev_logits).cpu())
        self.sev_true.append(grades.cpu())
        self.ratio_pred.append(logits_to_ratio(sev_logits).flatten().cpu())
        self.ratio_true.append(ratios.flatten().cpu())

    def severity_scatter(self):
        """Return (pred_ratio, true_ratio) lists for scatter plot."""
        if not self.ratio_pred:
            return [], []
        return torch.cat(self.ratio_pred).tolist(), torch.cat(self.ratio_true).tolist()


# Re-exported from the dataset module (single source of truth).
SEVERITY_SCALE = (0, 1, 3, 5, 7, 9)