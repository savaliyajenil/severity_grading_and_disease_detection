"""
SAMTSeg v2: severity-aware segmentation with attention decoder.

Encoder options:
- 'mscan_l' (default): MSCAN-L backbone, ImageNet-pretrained, stages
  (64, 128, 320, 512) at strides 4/8/16/32. Same weights file the SegNeXt
  baseline uses.
- 'resnet101' / 'resnet50': torchvision ImageNet weights (legacy path).

Novelty components (each independently ablatable):
  1. CBAM attention after every decoder block + bottleneck.
  2. SeverityGate: coarse disease map from detached bottleneck features
     re-weights each skip connection residually (skip * (1 + gate)).
  3. Ordinal severity head (CORAL-style) + severity-consistency loss.

Decoder v2 upgrades: BatchNorm after every decoder conv, widened skip path,
deep-supervision auxiliary heads on the two coarsest decoder stages.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.data.unified_dataset import SEVERITY_THRESHOLDS

_SEGNEXT_DIR = r'D:\PlantSeg SegNeXt'
MSCAN_CHECKPOINT = os.path.join(_SEGNEXT_DIR, 'mscan_l_20230227-cef260d4.pth')


def _load_mscan(pretrained):
    """MSCAN-L backbone (stages 64/128/320/512 @ /4//8//16//32)."""
    if os.path.isdir(_SEGNEXT_DIR) and _SEGNEXT_DIR not in sys.path:
        sys.path.insert(0, _SEGNEXT_DIR)
    try:
        from segnext_pytorch import MSCAN_L
    except ImportError as e:
        raise ImportError(
            f'Could not import MSCAN backbone from {_SEGNEXT_DIR}. '
            'Clone / copy segnext_pytorch.py there first.'
        ) from e
    if pretrained:
        if not os.path.exists(MSCAN_CHECKPOINT):
            raise FileNotFoundError(
                f'MSCAN-L weights not found at {MSCAN_CHECKPOINT}. '
                'Download mscan_l_20230227-cef260d4.pth there first.'
            )
        return MSCAN_L(pretrained=True, checkpoint_path=MSCAN_CHECKPOINT)
    return MSCAN_L(pretrained=False)


# ==============================================================================
# CBAM attention (novelty component 1)
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
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


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
# SeverityGate: disease-gated skip modulation (novelty component 2)
# ==============================================================================
class SeverityGate(nn.Module):
    """
    Coarse disease map from *detached* bottleneck features re-weights a skip
    connection residually: skip * (1 + gate), gate in [0, 1].

    The detach means gate gradients flow only into this 1x1 projector, never
    back into the segmentation path -- the coupling can amplify disease
    regions but cannot destabilize segmentation training.
    """

    def __init__(self, in_channels):
        super(SeverityGate, self).__init__()
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat, skip):
        gate = torch.sigmoid(self.proj(feat.detach()))
        gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear',
                             align_corners=False)
        return skip * (1.0 + gate)


# ==============================================================================
# ASPP (Atrous Spatial Pyramid Pooling) - from DeepLabV3+
# ==============================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        h, w = x.shape[-2:]
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                y = conv(x)
                y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
                res.append(y)
            else:
                res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# ==============================================================================
# Decoder block v2 (BN + optional CBAM)
# ==============================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_cbam=True):
        super(DecoderBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
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

        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class AuxHead(nn.Module):
    """Deep-supervision head: 1x1 classifier on a coarse decoder stage."""

    def __init__(self, in_channels, num_classes):
        super(AuxHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BoundaryHead(nn.Module):
    """Lesion-boundary head (novelty): 1x1 edge classifier on fine decoder
    features. Supervised by edges derived free from GT masks."""

    def __init__(self, in_channels):
        super(BoundaryHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==============================================================================
# Severity Head (Ordinal Regression - CORAL style, novelty component 3)
# ==============================================================================
class SeverityHead(nn.Module):
    """
    Ordinal severity head predicting lesion coverage ratio.
    Uses CORAL-style ordinal regression: K thresholds -> K binary logits.
    K follows len(SEVERITY_THRESHOLDS) so every scale value stays reachable.
    """
    def __init__(self, in_channels, num_thresholds=None):
        super(SeverityHead, self).__init__()
        if num_thresholds is None:
            num_thresholds = len(SEVERITY_THRESHOLDS)
        self.num_thresholds = num_thresholds
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_thresholds)  # One logit per threshold
        )

    def forward(self, x):
        # Global average pooling of bottleneck features
        x = self.global_pool(x).flatten(1)
        return self.fc(x)  # (B, num_thresholds)


# ==============================================================================
# SAMTSeg v2 model
# ==============================================================================
class PlantSegNoveltyModel(nn.Module):
    """
    Severity-aware segmentation with attention decoder.

    Encoder: MSCAN-L (ImageNet-pretrained, same weights file as the SegNeXt
    baseline) or ResNet{50,101} (torchvision ImageNet weights, legacy path).

    The contribution lives in the decoder and heads, not the backbone:
    CBAM attention, severity-gated skips, deep supervision, ordinal severity
    head + consistency coupling.
    """

    def __init__(self, num_classes=10, backbone='mscan_l', pretrained=True,
                 use_cbam=True):
        super(PlantSegNoveltyModel, self).__init__()
        self.backbone_name = backbone
        self.use_cbam = use_cbam
        self.num_thresholds = len(SEVERITY_THRESHOLDS)

        if backbone == 'mscan_l':
            # MSCAN-L stages: (64, 128, 320, 512) @ strides 4/8/16/32.
            self.encoder = _load_mscan(pretrained)
            c1, c2, c3, c4 = 64, 128, 320, 512
            # ASPP rates matched to the /32 map (~16-20px): rates above the
            # map size only convolve padding, so (6,12,18) is replaced.
            self.aspp = ASPP(c4, 256, atrous_rates=(3, 6, 9))
            # Decoder (/32 -> /4)
            self.dec4 = DecoderBlock(256, c3, c3, use_cbam=use_cbam)
            self.dec3 = DecoderBlock(c3, c2, c2, use_cbam=use_cbam)
            self.dec2 = DecoderBlock(c2, c1, c1, use_cbam=use_cbam)
            # /4 -> /1 in two learned 2x steps
            self.final_ups = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
                    nn.BatchNorm2d(c1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
                    nn.BatchNorm2d(c1),
                    nn.ReLU(inplace=True),
                ),
            ])
            self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)
            # Deep-supervision heads on the two coarsest decoder stages
            self.aux4 = AuxHead(c3, num_classes)
            self.aux3 = AuxHead(c2, num_classes)
            # Boundary head reads the finest decoder stage
            self.bnd_head = BoundaryHead(c1)
            # Severity head + gates read bottleneck features
            self.severity_head = SeverityHead(in_channels=c4)
            self.bottleneck_cbam = CBAM(c4) if use_cbam else nn.Identity()
            self.gate3 = SeverityGate(c4) if use_cbam else None
            self.gate2 = SeverityGate(c4) if use_cbam else None
            self.gate1 = SeverityGate(c4) if use_cbam else None
        elif backbone in ('resnet101', 'resnet50', 'resnet34'):
            if backbone == 'resnet101':
                resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
                low_level_channels = 256
                aspp_channels = 2048
            elif backbone == 'resnet50':
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
                low_level_channels = 256
                aspp_channels = 2048
            else:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
                low_level_channels = 64
                aspp_channels = 512

            # Stem
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool

            # ResNet stages
            self.layer1 = resnet.layer1  # 1/4
            self.layer2 = resnet.layer2  # 1/8
            self.layer3 = resnet.layer3  # 1/16
            self.layer4 = resnet.layer4  # 1/32

            c1 = low_level_channels
            # ASPP
            self.aspp = ASPP(aspp_channels, 256, atrous_rates=(6, 12, 18))
            # Decoder (single-scale skips like the original: bottleneck + c1)
            self.dec4 = None
            self.dec3 = None
            self.dec2 = DecoderBlock(256, c1, c1, use_cbam=use_cbam)
            self.bottleneck_cbam = nn.Identity()
            self.gate3 = self.gate2 = self.gate1 = None
            self.final_ups = nn.ModuleList([
                nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            ])
            self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)
            self.aux4 = self.aux3 = None
            self.bnd_head = BoundaryHead(c1)
            self.severity_head = SeverityHead(in_channels=aspp_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _forward_mscan(self, x):
        h, w = x.shape[-2:]
        s1, s2, s3, s4 = self.encoder(x)  # /4, /8, /16, /32
        b = self.bottleneck_cbam(s4)

        # Severity-gated skips (residual amplification of disease regions)
        if self.gate3 is not None:
            s3 = self.gate3(b, s3)
            s2 = self.gate2(b, s2)
            s1 = self.gate1(b, s1)

        a = self.aspp(b)              # /32
        d4 = self.dec4(a, s3)         # /16
        d3 = self.dec3(d4, s2)        # /8
        d2 = self.dec2(d3, s1)        # /4
        y = d2
        for up in self.final_ups:
            y = up(y)                 # /1
        seg_out = self.seg_head(y)
        seg_out = F.interpolate(seg_out, size=(h, w), mode='bilinear',
                                align_corners=False)

        aux = [F.interpolate(self.aux4(d4), size=(h, w), mode='bilinear',
                             align_corners=False),
               F.interpolate(self.aux3(d3), size=(h, w), mode='bilinear',
                             align_corners=False)]
        bnd = F.interpolate(self.bnd_head(d2), size=(h, w), mode='bilinear',
                            align_corners=False)
        sev_logits = self.severity_head(b)
        return seg_out, sev_logits, aux, bnd

    def _forward_resnet(self, x):
        h, w = x.shape[-2:]
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4

        # ResNet stages
        c1 = self.layer1(x)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        # ASPP
        a = self.aspp(c4)  # 1/32, 256 channels

        # Segmentation decoder
        d = self.dec2(a, c1)
        y = d
        for up in self.final_ups:
            y = up(y)

        # Upsample to input resolution
        seg_out = self.seg_head(y)
        seg_out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=False)

        # Severity prediction (THE NOVELTY)
        sev_logits = self.severity_head(c4)  # Use bottleneck features

        bnd = F.interpolate(self.bnd_head(d), size=(h, w), mode='bilinear',
                            align_corners=False)

        return seg_out, sev_logits, [], bnd

    def forward(self, x):
        if self.backbone_name == 'mscan_l':
            return self._forward_mscan(x)
        return self._forward_resnet(x)


def get_model(num_classes=10, backbone='mscan_l', pretrained=True, use_cbam=True):
    """Factory function for SAMTSeg v2 WITH novelty."""
    return PlantSegNoveltyModel(num_classes=num_classes, backbone=backbone,
                                pretrained=pretrained, use_cbam=use_cbam)


if __name__ == "__main__":
    model = get_model(num_classes=10, backbone='mscan_l', pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        seg, sev, aux, bnd = model(x)
    print(f"Input: {x.shape}")
    print(f"Segmentation: {seg.shape}")
    print(f"Severity: {sev.shape} (K={sev.shape[1]} thresholds)")
    print(f"Aux heads: {[a.shape for a in aux]}")
    print(f"Boundary: {bnd.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
