"""
Unified multi-source segmentation dataset.

Implements Phase 1 of the research plan: a single Dataset that consumes the
harmonized index produced by `scripts/10_build_index.py` and serves both sources
in one unified label space.

Sources
-------
PlantSeg  : pre-rendered binary PNG masks carrying the PlantSeg disease value.
            Real severity ground truth from `Metadata.csv:Mask ratio`.
            Supplies train / val / test. This is the benchmark: val and test are
            PlantSeg-only and are never augmented with external data.

Roboflow  : COCO instance polygons for potato Early/Late Blight, rasterized at
            load time. Leaf Miner and Nematode annotations are dropped.
            Training only, per the plan's contamination rule.

Severity supervision
--------------------
`severity_ratio` is PlantSeg's `Mask ratio` where available, and the rasterized
lesion coverage otherwise. Both are the same quantity: the fraction of image area
covered by disease. This is verified, not assumed -- see the docstring in
scripts/10_build_index.py.

The severity *grade* uses the plan's 0-1-3-5-7-9 national agricultural scale,
encoded ordinally as CORAL-style threshold indicators. Thresholds were placed at
the 20th/40th/60th/80th/90th percentiles of the PlantSeg training severity
distribution, giving a roughly balanced 6-way split so that every scale value
(0, 1, 3, 5, 7, 9) is reachable. NOTE: these cut points are calibrated to this
dataset's distribution, not universal standard bands -- report them as such.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_CLASSES = 10  # 0 = background/healthy, 1..9 = diseases

# National agricultural grading scale from the research plan (Section 6).
SEVERITY_SCALE = (0, 1, 3, 5, 7, 9)

# Cut points between adjacent scale values, at the 20/40/60/80/90th percentile of
# PlantSeg training severity_ratio. K = 5 thresholds for 6 ordinal levels, so
# grade 9 (coverage above the top threshold) is reachable.
SEVERITY_THRESHOLDS = (0.0449, 0.0914, 0.1527, 0.2625, 0.3735)

CLASS_NAMES = [
    "background", "potato_early_blight", "potato_late_blight",
    "tomato_bacterial_leaf_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_mosaic_virus", "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
]


def ordinal_targets(ratio, thresholds=SEVERITY_THRESHOLDS):
    """CORAL encoding: one binary target per threshold, shape (K,)."""
    return torch.tensor([1.0 if ratio > t else 0.0 for t in thresholds],
                        dtype=torch.float32)


def ratio_to_scale(ratio, thresholds=SEVERITY_THRESHOLDS, scale=SEVERITY_SCALE):
    """Map a severity ratio to its index on the 0-1-3-5-7-9 scale."""
    grade = 0
    for t in thresholds:
        if ratio > t:
            grade += 1
    return grade


def scale_label(ratio, thresholds=SEVERITY_THRESHOLDS, scale=SEVERITY_SCALE):
    return scale[ratio_to_scale(ratio, thresholds, scale)]


class UnifiedSegDataset(Dataset):
    """
    Records are loaded from index JSONs and filtered by split.

    splits:
        "train"  -> PlantSeg train + all Roboflow (plan Section 2.2)
        "val"    -> PlantSeg val only
        "test"   -> PlantSeg test only
    """

    def __init__(self, index_root, split="train", transforms=None,
                 include_roboflow=True, require_roboflow_train_only=True):
        self.root = Path(index_root)
        self.split = split
        self.transforms = transforms
        self.records = []

        plantseg = json.loads((self.root / "plantseg_potato_tomato.json").read_text())
        roboflow = json.loads((self.root / "roboflow_potato.json").read_text())

        if split == "train":
            self.records += [r for r in plantseg if r["split"] == "train"]
            if include_roboflow:
                if require_roboflow_train_only:
                    # Plan Section 2.2: Roboflow contributes to training only.
                    self.records += [r for r in roboflow
                                     if r["split"] == "roboflow_train"]
                else:
                    self.records += roboflow
        elif split in ("val", "test"):
            # Benchmark splits stay PlantSeg-only so external data cannot inflate
            # reported metrics.
            self.records += [r for r in plantseg if r["split"] == split]
        else:
            raise ValueError(f"unknown split '{split}'")

        if not self.records:
            raise RuntimeError(f"no records for split '{split}'")

        n_ps = sum(1 for r in self.records if r["source"] == "plantseg")
        logger.info("UnifiedSegDataset[%s]: %d records (%d PlantSeg, %d Roboflow)",
                    split, len(self.records), n_ps, len(self.records) - n_ps)

    def __len__(self):
        return len(self.records)

    def _rasterize_roboflow(self, record, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in record["coco_anns"]:
            cls = int(ann["category_id"])
            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue
                pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], cls)
        return mask

    def __getitem__(self, idx):
        record = self.records[idx]

        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {record['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if record["mask"] is not None:
            # PlantSeg: remap its per-disease mask value onto our class index.
            raw = cv2.imread(record["mask"], cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"failed to decode {record['mask']}")
            if raw.ndim == 3:
                raw = raw[..., 0]
            mask = np.where(raw == record["mask_value"], record["class_id"], 0)
            mask = mask.astype(np.uint8)
        else:
            mask = self._rasterize_roboflow(record, h, w)

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = image.float()
        mask = torch.as_tensor(np.ascontiguousarray(mask)).long()

        # Severity ratio: PlantSeg's annotated value when available, otherwise the
        # coverage actually present in the augmented mask.
        if record["severity_ratio"] is not None:
            ratio = float(record["severity_ratio"])
        else:
            num_px = float(mask.numel())
            ratio = float((mask > 0).sum()) / num_px if num_px else 0.0

        return {
            "image": image,
            "mask": mask,
            "ratio": torch.tensor(ratio, dtype=torch.float32),
            "severity": ordinal_targets(ratio),
            "grade": torch.tensor(ratio_to_scale(ratio), dtype=torch.long),
            "class_id": int(record.get("class_id",
                                       (record.get("class_ids") or [0])[0])),
            "source": record["source"],
        }

    def class_pixel_counts(self):
        """Per-class pixel counts across the split. Used for loss weighting."""
        # class_map.json stores disease -> [class_index, plantseg_mask_value];
        # invert it into the mask_value -> class_index lookup we need here.
        cmap = json.loads((self.root / "class_map.json").read_text())
        disease_to_class = cmap["disease_to_class"]        # disease -> class index
        disease_to_value = cmap["disease_to_mask_value"]  # disease -> mask value
        mask_value_to_class = {
            int(disease_to_value[d]): int(disease_to_class[d])
            for d in disease_to_class
        }

        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for record in self.records:
            if record["mask"] is not None:
                raw = cv2.imread(record["mask"], cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                if raw.ndim == 3:
                    raw = raw[..., 0]
                counts[0] += int((raw == 0).sum())
                for v in np.unique(raw):
                    if v == 0:
                        continue
                    cls = mask_value_to_class.get(int(v))
                    if cls is not None:
                        counts[cls] += int((raw == v).sum())
            else:
                img = cv2.imread(record["image"])
                if img is None:
                    continue
                m = self._rasterize_roboflow(record, img.shape[0], img.shape[1])
                counts += np.bincount(m.reshape(-1), minlength=NUM_CLASSES)
        return counts


def class_weights_from_counts(counts, method="effective_num", beta=0.9999, gamma=2.0):
    """
    Class weights for cross-entropy.

    `inv`           : w_c proportional to 1 / n_c   (aggressive; background ~0.005)
    `sqrt_inv`      : w_c proportional to 1 / sqrt(n_c)  (gentler)
    `effective_num` : Class-Balanced Loss (Cui et al., CVPR 2019).
                      Falls back to sqrt_inv when pixel counts are too large
                      (>1e6) since beta^n saturates. For true effective_num,
                      use image-level counts instead of pixel counts.
    `focal`         : Returns uniform weights; trainer should use FocalLoss
                      with gamma instead of weighted CE.
    """
    counts = np.asarray(counts, dtype=np.float64)
    safe = np.maximum(counts, 1.0)

    if method == "inv":
        w = 1.0 / safe
    elif method == "sqrt_inv":
        w = 1.0 / np.sqrt(safe)
    elif method == "effective_num":
        # Pixel counts are too large for beta^n; fall back to sqrt_inv
        if safe.max() > 1e6:
            w = 1.0 / np.sqrt(safe)
        else:
            effective_num = (1.0 - np.power(beta, safe)) / (1.0 - beta)
            w = 1.0 / effective_num
    elif method == "focal":
        w = np.ones_like(safe)
    else:
        raise ValueError(f"Unknown weight_scheme: {method}")

    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
