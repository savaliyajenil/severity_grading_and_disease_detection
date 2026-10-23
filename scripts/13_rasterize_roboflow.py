"""
Rasterize Roboflow COCO annotations to grayscale PNG masks matching PlantSeg format.

PlantSeg format: 
- PNG file in annotations/<split>/<image_name>.png
- Pixel value = class index (0=background, 1=potato_early_blight, 2=potato_late_blight, etc.)
- Mask dimensions match the source image

Roboflow annotations are in _annotations.coco.json with polygon segmentations.
This script renders those polygons into PNG masks and writes them to the same
directory structure as PlantSeg expects.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rasterize")

# Roboflow class name -> our unified class index (matching PlantSeg)
ROBOFLOW_TO_CLASS = {
    "Early Blight": 1,    # potato_early_blight
    "Late Blight": 2,     # potato_late_blight
    # Leaf Miner, Nematode -> dropped (not in PlantSeg)
}


def rasterize_split(roboflow_root, split):
    """Rasterize one split (train/valid/test) to PNG masks."""
    split_dir = Path(roboflow_root) / split
    coco_file = split_dir / "_annotations.coco.json"
    if not coco_file.exists():
        logger.warning("%s not found, skipping %s", coco_file, split)
        return 0

    coco = json.loads(coco_file.read_text())
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image
    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    out_dir = split_dir
    n_written = 0
    n_dropped = 0

    for img in coco["images"]:
        fname = img["file_name"]
        img_path = split_dir / fname
        if not img_path.exists():
            continue

        anns = by_image.get(img["id"], [])
        if not anns:
            continue

        # Filter to kept classes
        keep = []
        for a in anns:
            name = cat_names.get(int(a["category_id"]), "")
            if name in ROBOFLOW_TO_CLASS:
                keep.append((ROBOFLOW_TO_CLASS[name], a))

        if not keep:
            n_dropped += 1
            continue

        h, w = img["height"], img["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for cls, ann in keep:
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], cls)

        out_name = Path(fname).with_suffix(".png").name
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), mask)
        n_written += 1

    logger.info("%s: wrote %d PNG masks, dropped %d (only Leaf Miner/Nematode)",
                split, n_written, n_dropped)
    return n_written


def main():
    roboflow_root = Path(r"D:\pdeu_kaam\potato-leaf-disease-segmentation-4g26g")
    total = 0
    for split in ("train", "valid", "test"):
        total += rasterize_split(roboflow_root, split)
    logger.info("Total PNG masks written: %d", total)


if __name__ == "__main__":
    main()