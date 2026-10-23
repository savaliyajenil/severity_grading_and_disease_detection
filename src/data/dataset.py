import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Lesion-coverage thresholds defining mild / moderate / severe.
# Chosen from the training split's coverage distribution (see scripts/07_severity_calibration.py):
# splits train 29.9 / 46.3 / 23.7 % and valid 36.1 / 46.3 / 17.6 %.
SEVERITY_THRESHOLDS = (0.01, 0.05)

NUM_CLASSES = 5  # 0 = healthy/background, 1..4 = the annotated diseases


def ordinal_targets(coverage, thresholds=SEVERITY_THRESHOLDS):
    """
    CORAL-style ordinal encoding.

    Instead of a 3-way softmax over {mild, moderate, severe}, which discards the
    ordering, we predict independent binary decisions for each threshold:
    "is coverage above t0?", "is coverage above t1?". This keeps the ordering
    information and gives a loss that is still a simple sum of BCE terms.

    Returns a float32 tensor of shape (len(thresholds),).
    """
    return torch.tensor([1.0 if coverage > t else 0.0 for t in thresholds],
                        dtype=torch.float32)


def coverage_to_grade(coverage, thresholds=SEVERITY_THRESHOLDS):
    """Inverse of ordinal_targets: map coverage to a 0..K-1 severity grade."""
    grade = 0
    for t in thresholds:
        if coverage > t:
            grade += 1
    return grade


class PotatoLeafSegDataset(Dataset):
    """
    Semantic segmentation dataset for the Roboflow-exported potato leaf dataset.

    On-disk layout (this is what Roboflow actually produces -- note there are no
    separate image/annotation subfolders, and the COCO file is named
    `_annotations.coco.json`, not `annotation_<split>.json`):

        <root>/<split>/*.jpg
        <root>/<split>/_annotations.coco.json

    Class mapping uses the COCO `category_id` directly. `category_id` 0 is the
    declared supercategory "objects" and has zero annotations in every split, so
    it doubles as the healthy/background class: any pixel not covered by a
    polygon is label 0. No remapping is needed and no annotation can collide
    with the background index.

    Severity labels are *derived*, not independently annotated: the target is the
    fraction of image area covered by lesions. This must be stated explicitly in
    the paper's dataset section -- it is a weak/derived supervision signal, and
    that is what makes the severity head worth studying rather than a redundant
    readout of the mask.
    """

    def __init__(self, root_dir, split="train", transforms=None,
                 thresholds=SEVERITY_THRESHOLDS, require_annotations=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.thresholds = tuple(thresholds)

        self.img_dir = self.root_dir / split
        coco_file = self.img_dir / "_annotations.coco.json"

        if not coco_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {coco_file}. "
                f"Expected the Roboflow layout <root>/{split}/_annotations.coco.json"
            )

        with open(coco_file, "r") as f:
            self.coco = json.load(f)

        self.cat_names = {c["id"]: c["name"] for c in self.coco["categories"]}

        anns_by_image = {}
        for a in self.coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        # Filter to images that actually exist on disk. The exported archive is
        # incomplete (222/980 train, 32/140 valid, 74/280 test are listed in the
        # JSON but absent from disk), and a missing file would otherwise raise
        # mid-epoch.
        self.samples = []
        skipped_missing, skipped_empty, mismatched = 0, 0, 0
        for img in self.coco["images"]:
            path = self.img_dir / img["file_name"]
            if not path.exists():
                skipped_missing += 1
                continue
            anns = anns_by_image.get(img["id"], [])
            if require_annotations and not anns:
                skipped_empty += 1
                continue
            self.samples.append({
                "path": path,
                "anns": anns,
                "height": img["height"],
                "width": img["width"],
            })

        if skipped_missing or skipped_empty:
            logger.warning(
                "%s: using %d/%d images (skipped %d listed in JSON but missing "
                "on disk, %d with zero annotations)",
                split, len(self.samples), len(self.coco["images"]),
                skipped_missing, skipped_empty,
            )

        if not self.samples:
            raise RuntimeError(f"{split}: no usable images found under {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def _rasterize(self, sample, height, width):
        """Render COCO polygons into a class-index mask at the given size."""
        mask = np.zeros((height, width), dtype=np.uint8)

        h, w = sample["height"], sample["width"]
        sx = width / float(w)
        sy = height / float(h)

        for ann in sample["anns"]:
            cls = int(ann["category_id"])
            segs = ann.get("segmentation") or []
            for seg in segs:
                if len(seg) < 6:  # need at least 3 points
                    continue
                pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
                pts[:, 0] *= sx
                pts[:, 1] *= sy
                cv2.fillPoly(mask, [pts.astype(np.int32)], cls)

        return mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image: {sample['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Rasterize at the source resolution, then let augmentation resample. This
        # rasterizes each polygon's edges exactly once instead of once per epoch.
        mask = self._rasterize(sample, sample["height"], sample["width"])

        if image.shape[:2] != (sample["height"], sample["width"]):
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        # Severity target is recomputed from the *augmented* mask so the label
        # always describes the pixels the model actually receives.
        # After ToTensorV2 the mask is a tensor; before it, a numpy array.
        num_pixels = float(mask.numel() if torch.is_tensor(mask) else mask.size)
        disease_pixels = float((mask > 0).sum())
        coverage = disease_pixels / num_pixels
        severity = ordinal_targets(coverage, self.thresholds)

        image = image.float()
        mask = torch.as_tensor(np.ascontiguousarray(mask)).long()

        return {
            "image": image,
            "mask": mask,
            "coverage": torch.tensor(coverage, dtype=torch.float32),
            "severity": severity,
        }

    def class_pixel_counts(self):
        """Per-class pixel counts over the split. Useful for class weighting."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for sample in self.samples:
            mask = self._rasterize(sample, sample["height"], sample["width"])
            counts += np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        return counts
