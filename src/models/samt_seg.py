import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.unified_dataset import SEVERITY_THRESHOLDS

# ==============================================================================
# CBAM: Convolutional Block Attention Module
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==============================================================================
# UNet Decoder Block (with CBAM attention -- core novelty component)
# ==============================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.use_cbam = use_cbam
        self.cbam = CBAM(out_channels) if use_cbam else None

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle odd input dimensions that don't perfectly match after upsampling
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.cbam is not None:
            x = self.cbam(x)
        return x


# ==============================================================================
# ResNet encoder wrapper (exposes 5 stages for the UNet decoder)
# ==============================================================================
class ResNetEncoder(nn.Module):
    """
    torchvision ResNet used as a pretrained encoder.

    Returns five feature maps at strides 2, 4, 8, 16, 32 with channel counts
    (64, 64, 128, 256, 512) for resnet18/resnet34. Set `pretrained=False` for the
    from-scratch ablation baseline.
    """

    OUT_CHANNELS = {
        "resnet18": (64, 64, 128, 256, 512),
        "resnet34": (64, 64, 128, 256, 512),
    }

    def __init__(self, name="resnet34", pretrained=True):
        super(ResNetEncoder, self).__init__()
        import torchvision

        if name not in self.OUT_CHANNELS:
            raise ValueError(f"Unsupported encoder '{name}'. "
                             f"Choose from {list(self.OUT_CHANNELS)}")

        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2,  64
        self.pool = net.maxpool                                    # /4
        self.layer1 = net.layer1                                   # /4,  64
        self.layer2 = net.layer2                                   # /8,  128
        self.layer3 = net.layer3                                   # /16, 256
        self.layer4 = net.layer4                                   # /32, 512

        self.out_channels = self.OUT_CHANNELS[name]

    def forward(self, x):
        e1 = self.stem(x)                 # /2
        x = self.pool(e1)
        e2 = self.layer1(x)               # /4
        e3 = self.layer2(e2)              # /8
        e4 = self.layer3(e3)              # /16
        b = self.layer4(e4)               # /32
        return e1, e2, e3, e4, b


class EfficientNetEncoder(nn.Module):
    """
    torchvision EfficientNet encoder (plan Section 5: EfficientNet backbone).

    Only the B0/B1 variants are wired up. EfficientNet's feature maps run to
    stride 32 with a final 7x7 spatial average pool that we do not apply, so the
    five returned stages align with the UNet decoder's expectations.
    """

    # torchvision's EfficientNet `features` uses 9 blocks; the useful skip
    # stages are not contiguous. Measured channel/stride per block for B0:
    #   0:(32,/2) 1:(16,/2) 2:(24,/4) 3:(40,/8) 4:(80,/16) 5:(112,/16)
    #   6:(192,/32) 7:(320,/32) 8:(1280,/32)
    # EfficientNet blocks are densely connected via partial-channel concatenation:
    # block k only reads a *slice* of the previous block's output. Blocks cannot
    # be executed selectively, so all blocks 0..7 run in order and we tap the
    # outputs of 1, 2, 3, 5, 7 to get strides 2, 4, 8, 16, 32.
    LAST_BLOCK = 7
    TAP_BLOCKS = (1, 2, 3, 5, 7)

    OUT_CHANNELS = {
        "efficientnet_b0": (16, 24, 40, 112, 320),
        "efficientnet_b1": (16, 24, 40, 112, 320),
    }

    def __init__(self, name="efficientnet_b0", pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        import torchvision

        if name not in self.OUT_CHANNELS:
            raise ValueError(f"Unsupported encoder '{name}'. "
                             f"Choose from {list(self.OUT_CHANNELS)}")

        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        self.blocks = nn.ModuleList(list(net.features)[:self.LAST_BLOCK + 1])
        self.out_channels = self.OUT_CHANNELS[name]

    def forward(self, x):
        taps = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.TAP_BLOCKS:
                taps[i] = x
        return (taps[1], taps[2], taps[3], taps[5], taps[7])


# ==============================================================================
# Scratch encoder (matches the original from-scratch implementation)
# ==============================================================================
class ScratchEncoder(nn.Module):
    """
    From-scratch encoder for the ablation baseline.

    Uses the same stride schedule as ResNetEncoder (2, 4, 8, 16, 32) so that a
    single decoder definition works for both. The original implementation had no
    downsampling stem, which produced segmentation logits at 2x the input
    resolution and broke skip alignment.
    """

    OUT_CHANNELS = (64, 64, 128, 256, 512)

    def __init__(self, **_):
        super(ScratchEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )                                            # /2,  64
        self.pool = nn.MaxPool2d(3, 2, 1)             # /4
        self.layer1 = self._conv_block(64, 64)        # /4,  64
        self.layer2 = self._conv_block(64, 128)       # /8,  128
        self.layer3 = self._conv_block(128, 256)      # /16, 256
        self.layer4 = self._conv_block(256, 512)      # /32, 512
        self.out_channels = self.OUT_CHANNELS

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.stem(x)              # /2
        x = self.pool(e1)
        e2 = self.layer1(x)            # /4
        e3 = self.layer2(e2)           # /8
        e4 = self.layer3(e3)           # /16
        b = self.layer4(e4)            # /32
        return e1, e2, e3, e4, b


# ==============================================================================
# SAMTSeg: Severity-Aware Multi-Task Segmentation Network
# ==============================================================================
class SAMTSeg(nn.Module):
    """
    Novel Architecture: Severity-Aware Multi-Task Segmentation Network.

    Novelty components (both ablatable for the paper):
      1. CBAM attention at the bottleneck and in every decoder block.
      2. Severity-Consistency Loss coupling the two task heads.

    The severity head is *ordinal*: it predicts independent logits for each
    coverage threshold (CORAL-style) rather than a 3-way class softmax, which
    preserves the mild < moderate < severe ordering.

    Encoder defaults to a pretrained ResNet34; `efficientnet_b0` is also
    available per the plan. The contribution lives in the heads and the loss, not
    the backbone.
    """

    def __init__(self, num_classes=10, encoder="resnet34", pretrained=True,
                 num_thresholds=5, use_cbam=True):
        super(SAMTSeg, self).__init__()

        if encoder == "scratch":
            self.encoder = ScratchEncoder()
        elif encoder.startswith("efficientnet"):
            self.encoder = EfficientNetEncoder(encoder, pretrained=pretrained)
        else:
            self.encoder = ResNetEncoder(encoder, pretrained=pretrained)

        c1, c2, c3, c4, c5 = self.encoder.out_channels
        bottleneck_ch = c5

        # --- Task 1: Segmentation decoder (CBAM inside every block) ---
        self.dec4 = DecoderBlock(bottleneck_ch, c4, c4, use_cbam=use_cbam)
        self.dec3 = DecoderBlock(c4, c3, c3, use_cbam=use_cbam)
        self.dec2 = DecoderBlock(c3, c2, c2, use_cbam=use_cbam)
        self.dec1 = DecoderBlock(c2, c1, c1, use_cbam=use_cbam)

        self.bottleneck_cbam = CBAM(bottleneck_ch) if use_cbam else nn.Identity()

        # Upsample the /32 bottleneck to /2 so the final skip resolves at stride 2
        self.final_up = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)

        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # --- Task 2: Ordinal severity head ---
        # Reads global-pooled bottleneck features: context, not fine detail.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sev_head = nn.Sequential(
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_thresholds)   # one logit per threshold
        )

        self.num_thresholds = num_thresholds

    def forward(self, x):
        in_h, in_w = x.shape[-2:]

        e1, e2, e3, e4, b = self.encoder(x)
        b = self.bottleneck_cbam(b)

        # Segmentation
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d1 = self.final_up(d1)   # /1, matches input resolution

        seg_mask = self.seg_head(d1)  # logits (B, num_classes, H, W)

        # Ordinal severity
        sev_feat = self.global_pool(b).flatten(1)
        sev_logits = self.sev_head(sev_feat)   # (B, num_thresholds)

        return seg_mask, sev_logits


# ==============================================================================
# Losses
# ==============================================================================
class OrdinalSeverityLoss(nn.Module):
    """
    CORAL-style ordinal loss.

    For K ordered thresholds t_1 < ... < t_K and a target coverage c, each head
    logit is trained with BCE against the indicator 1[c > t_k]. Independent
    binary decisions retain the ordering that a 3-way softmax would discard.
    """

    def __init__(self):
        super(OrdinalSeverityLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, sev_logits, sev_targets):
        return self.bce(sev_logits, sev_targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for dense prediction (Lin et al., ICCV 2017).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    For segmentation, we apply it per-pixel with class weights alpha.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        # Ignore the ignore_index pixels
        valid = (targets != self.ignore_index)
        if valid.any():
            return focal_loss[valid].mean()
        return focal_loss.mean()


def logits_to_severity(sev_logits):
    """
    Decode ordinal logits to a scalar severity ratio in [0, 1] for the
    consistency loss: the mean probability of exceeding each threshold.

    Monotonic in the true coverage, and differentiable w.r.t. the segmentation
    logits through the consistency term.
    """
    probs = torch.sigmoid(sev_logits)          # (B, K)
    return probs.mean(dim=1, keepdim=True)     # (B, 1) in [0, 1]


# ==============================================================================
# Severity-Consistency Loss (core novelty component -- kept unchanged in spirit)
# ==============================================================================
class SeverityConsistencyLoss(nn.Module):
    def __init__(self, background_class_idx=0):
        super(SeverityConsistencyLoss, self).__init__()
        self.bg_idx = background_class_idx
        self.mse = nn.MSELoss()

    def forward(self, predicted_logits, predicted_severity):
        """
        predicted_logits: (B, C, H, W) segmentation logits
        predicted_severity: (B, 1) scalar severity score [0, 1]

        Couples the coarse ordinal severity estimate to the fine segmentation:
        the segmentation head is pulled toward agreeing with a *discretized*
        severity judgement, so agreement between a coarse and a fine estimate of
        the same quantity carries real information rather than restating the
        segmentation loss.
        """
        B, C, H, W = predicted_logits.size()

        # Differentiable estimate: expected disease mass per pixel. This is the
        # same quantity the plan's formula asks for ((Disease Pixels / Total
        # Pixels) x 100%), computed as a soft expectation so the consistency term
        # actually delivers gradient into the segmentation decoder. A hard argmax
        # count would be non-differentiable and the term would train nothing.
        probs = F.softmax(predicted_logits, dim=1)
        disease_probs = 1.0 - probs[:, self.bg_idx, :, :]   # (B, H, W)
        inferred_severity = disease_probs.sum(dim=(1, 2)) / (H * W)
        inferred_severity = inferred_severity.view(-1, 1)

        return self.mse(predicted_severity, inferred_severity)


# ==============================================================================
# Metrics
# ==============================================================================
class SegmentationMetrics:
    """
    Accumulates confusion-based pixel metrics over an epoch.

    Classes absent from both prediction and ground truth are excluded from the
    mean (union == 0), so the reported mIoU is not diluted by the two classes
    that never appear in this dataset.
    """

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

        # Per-class precision / recall / F1 (plan Section 6).
        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        f1 = torch.zeros(self.num_classes)
        p_denom = tp + fp
        r_denom = tp + fn
        precision[p_denom > 0] = tp[p_denom > 0] / p_denom[p_denom > 0]
        recall[r_denom > 0] = tp[r_denom > 0] / r_denom[r_denom > 0]
        denom = precision + recall
        f1[denom > 0] = (2 * precision * recall)[denom > 0] / denom[denom > 0]

        dice_denom = 2 * tp + fp + fn
        dice = torch.zeros(self.num_classes)
        dice[valid] = (2 * tp[valid]) / dice_denom[valid]

        total = tp.sum()
        acc = (total / cm.sum()).item() if cm.sum() > 0 else 0.0

        out = {
            "pixel_acc": acc,
            "mIoU": iou[valid].mean().item() if valid.any() else 0.0,
            "mDice": dice[valid].mean().item() if valid.any() else 0.0,
            "per_class_iou": iou.tolist(),
            "disease_mIoU": iou[1:][union[1:] > 0].mean().item()
                            if (union[1:] > 0).any() else 0.0,
            # Per-class precision / recall / F1 (plan Section 6 targets > 0.85).
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "mean_disease_f1": f1[1:][union[1:] > 0].mean().item()
                              if (union[1:] > 0).any() else 0.0,
        }

        # Severity grading accuracy on the 0-1-3-5-7-9 scale, plus MAE in ratio
        # units and within-one-grade agreement (plan Section 6: target > 80%).
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
        """Accumulate severity predictions / ground truth for grading metrics."""
        self.sev_pred.append(ordinal_grade(sev_logits).cpu())
        self.sev_true.append(grades.cpu())
        self.ratio_pred.append(logits_to_ratio(sev_logits).flatten().cpu())
        self.ratio_true.append(ratios.flatten().cpu())

    def severity_scatter(self):
        """(pred_ratio, true_ratio) as lists, for the scatter plot."""
        if not self.ratio_pred:
            return [], []
        return (torch.cat(self.ratio_pred).tolist(),
                torch.cat(self.ratio_true).tolist())


def logits_to_ratio(sev_logits, thresholds=SEVERITY_THRESHOLDS):
    """
    Invert the ordinal head to a continuous severity ratio estimate.

    The head predicts K independent probabilities p_k = P(ratio > t_k). Because
    the thresholds partition [0, 1] into K+1 ordered intervals, the midpoints
    between adjacent thresholds give a piecewise-linear estimate of the ratio.
    Clamped to [0, 1]. This keeps the consistency loss interpretable as a ratio
    rather than an uncalibrated probability.
    """
    probs = torch.sigmoid(sev_logits)                       # (B, K)
    bounds = torch.tensor([0.0] + list(thresholds) + [1.0],
                          device=sev_logits.device)
    mids = (bounds[:-1] + bounds[1:]) / 2.0                  # (K+1,)
    # Interval k is chosen when p_k >= 0.5.
    idx = (probs >= 0.5).sum(dim=1)                          # (B,)
    return mids[idx].unsqueeze(1)                            # (B, 1)


def ordinal_grade(logits, thresholds=SEVERITY_THRESHOLDS):
    """
    Decode ordinal logits to a discrete grade index in {0, ..., K}.

    Count how many thresholds the logit clears.
    thresholds: sequence of length K, ascending.
    """
    return (logits > 0).sum(dim=1).long()


if __name__ == "__main__":
    model = SAMTSeg(num_classes=10, encoder="resnet34", pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    seg, sev = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input: {x.shape}")
    print(f"Segmentation output: {seg.shape}")
    print(f"Severity logits:    {sev.shape}")
    print(f"Params: {n_params / 1e6:.2f}M")
