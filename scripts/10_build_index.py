"""
Class harmonization and unified dataset index for the SAMTSeg project.

Implements Section 3.3 of the research plan: Roboflow's free-text class names are
mapped onto PlantSeg's integer mask values, and classes absent from PlantSeg
(Leaf Miner, Nematode) are dropped.

CLASS_MAP (10 channels, index 0 = healthy/background):
    0   background / healthy leaf
    1   potato early blight        (PlantSeg mask value 76)
    2   potato late blight         (PlantSeg mask value 77)
    3   tomato bacterial leaf spot (97)
    4   tomato early blight        (98)
    5   tomato late blight         (99)
    6   tomato leaf mold           (100)
    7   tomato mosaic virus        (101)
    8   tomato septoria leaf spot  (102)
    9   tomato yellow leaf curl    (103)

Verified against the extracted dataset: every PlantSeg potato/tomato mask is
binary (background + exactly one disease value), and each disease maps to exactly
one mask value across all 813 images. `Mask ratio` in Metadata.csv equals
(mask > 0).mean() to within 0.00000 over a 40-image sample, which means it is a
usable severity ground truth rather than an independent annotation.

Outputs a single JSON index consumed by `UnifiedSegDataset`:
    results/index/plantseg_potato_tomato.json
    results/index/roboflow_potato.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("harmonize")

NUM_CLASSES = 10

CLASS_NAMES = [
    "background", "potato_early_blight", "potato_late_blight",
    "tomato_bacterial_leaf_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_mosaic_virus", "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
]

# disease string -> (class index, PlantSeg PNG mask value)
DISEASE_TO_CLASS = {
    "potato early blight": (1, 76),
    "potato late blight": (2, 77),
    "tomato bacterial leaf spot": (3, 97),
    "tomato early blight": (4, 98),
    "tomato late blight": (5, 99),
    "tomato leaf mold": (6, 100),
    "tomato mosaic virus": (7, 101),
    "tomato septoria leaf spot": (8, 102),
    "tomato yellow leaf curl virus": (9, 103),
}

# Roboflow free-text name -> class index. Leaf Miner / Nematode intentionally
# absent: they are not in PlantSeg, so they are dropped (plan Section 3.3).
ROBOFLOW_TO_CLASS = {
    "Early Blight": 1,   # potato early blight
    "Late Blight": 2,    # potato late blight
    # "Leaf Miner"  -> dropped
    # "Nematode"    -> dropped
}

SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}


def build_plantseg_index(plantseg_root, plants=("Potato", "Tomato")):
    """Index PlantSeg potato+tomato images with real mask-ratio severity labels."""
    root = Path(plantseg_root)
    md = pd.read_csv(root / "Metadata.csv")
    md["split"] = md["Split"].map(SPLIT_MAP)
    md = md[md.Plant.isin(plants)].copy()

    records, dropped = [], []
    for _, row in md.iterrows():
        disease = row["Disease"].strip()
        if disease not in DISEASE_TO_CLASS:
            dropped.append((row["Name"], disease))
            continue
        cls, mask_val = DISEASE_TO_CLASS[disease]

        img = root / "images" / row["split"] / row["Name"]
        mask = root / "annotations" / row["split"] / row["Label file"]
        if not (img.exists() and mask.exists()):
            dropped.append((row["Name"], "MISSING FILE"))
            continue

        records.append({
            "source": "plantseg",
            "image": str(img),
            "mask": str(mask),
            "split": row["split"],
            "class_id": cls,
            "mask_value": mask_val,
            # Ground-truth severity: fraction of image area covered by the lesion.
            "severity_ratio": float(row["Mask ratio"]),
            "plant": row["Plant"],
            "disease": disease,
            "name": row["Name"],
        })

    if dropped:
        logger.warning("dropped %d PlantSeg records: %s",
                       len(dropped), dropped[:5])
    return records


def build_roboflow_index(roboflow_root, flagged=None):
    """
    Index Roboflow potato images, remapping category IDs to the shared 10-class
    space. Images whose annotations are entirely Leaf Miner / Nematode are
    excluded; images mixing a kept disease with a dropped one keep only the kept
    annotations.

    `flagged` is an optional set of filenames from the pHash audit; those are
    purged per the plan's contamination protocol.
    """
    root = Path(roboflow_root)
    flagged = flagged or set()
    cat_names = {}
    records = []

    kept_anns = dropped_anns = 0
    n_img_dropped_class = 0
    n_img_purged = 0

    for split in ("train", "valid", "test"):
        split_dir = root / split
        coco_file = split_dir / "_annotations.coco.json"
        if not coco_file.exists():
            continue
        coco = json.loads(coco_file.read_text())

        for c in coco["categories"]:
            cat_names[c["id"]] = c["name"]

        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        for img in coco["images"]:
            fname = img["file_name"]
            path = split_dir / fname
            if not path.exists():
                continue
            if fname in flagged:
                n_img_purged += 1
                continue

            anns = by_image.get(img["id"], [])
            keep, drop = [], []
            for a in anns:
                name = cat_names.get(int(a["category_id"]), "")
                if name in ROBOFLOW_TO_CLASS:
                    keep.append((ROBOFLOW_TO_CLASS[name], a))
                else:
                    drop.append(a)

            if not keep:
                # Every annotation is a dropped class -> exclude the whole image.
                n_img_dropped_class += 1
                dropped_anns += len(drop)
                continue

            kept_anns += len(keep)
            dropped_anns += len(drop)

            records.append({
                "source": "roboflow",
                "image": str(path),
                # Polygons rasterized at load time; no PNG mask exists.
                "mask": None,
                "coco_anns": [
                    {"category_id": c, "segmentation": a["segmentation"]}
                    for c, a in keep
                ],
                "split": "roboflow_" + split,
                "class_ids": sorted({c for c, _ in keep}),
                "severity_ratio": None,   # computed from rasterized coverage
                "name": fname,
            })

    logger.info("Roboflow: %d images kept, %d excluded (only dropped classes), "
                "%d purged by pHash", len(records), n_img_dropped_class, n_img_purged)
    logger.info("Roboflow annotations: %d kept, %d dropped (Leaf Miner/Nematode)",
                kept_anns, dropped_anns)
    return records


def summarize(records):
    df = pd.DataFrame([{
        "source": r["source"],
        "split": r["split"],
        "class_id": r.get("class_id", (r.get("class_ids") or [None])[0]),
    } for r in records])

    logger.info("")
    logger.info("=== unified index summary ===")
    logger.info("total records: %d", len(df))
    logger.info("")
    logger.info("by split:")
    for k, v in df["split"].value_counts().sort_index().items():
        logger.info("   %-18s %5d", k, v)
    logger.info("")
    logger.info("by class (plantseg records have 1 class; roboflow may have >1):")
    for k, v in df["class_id"].value_counts().sort_index().items():
        name = CLASS_NAMES[int(k)] if k is not None and k >= 0 else "?"
        logger.info("   %-32s %5d", name, v)
    return df


def main(args):
    plantseg_records = build_plantseg_index(args.plantseg_root)
    logger.info("PlantSeg potato+tomato indexed: %d", len(plantseg_records))

    flagged = set()
    flagged_file = Path(args.results_root) / "dedup" / "roboflow_flagged_images.csv"
    if flagged_file.exists():
        flagged = set(pd.read_csv(flagged_file)["roboflow_file"])
        logger.info("loading %d pHash-flagged filenames for purging", len(flagged))

    roboflow_records = build_roboflow_index(args.roboflow_root, flagged)

    all_records = plantseg_records + roboflow_records

    out_dir = Path(args.results_root) / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "plantseg_potato_tomato.json").write_text(
        json.dumps(plantseg_records, indent=1))
    (out_dir / "roboflow_potato.json").write_text(
        json.dumps(roboflow_records, indent=1))
    (out_dir / "class_map.json").write_text(json.dumps({
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "disease_to_class": {k: v[0] for k, v in DISEASE_TO_CLASS.items()},
        "disease_to_mask_value": {k: v[1] for k, v in DISEASE_TO_CLASS.items()},
        "roboflow_to_class": ROBOFLOW_TO_CLASS,
    }, indent=2))

    summarize(all_records)

    # Severity label distribution on PlantSeg, used to place severity thresholds.
    ratios = np.array([r["severity_ratio"] for r in plantseg_records])
    logger.info("")
    logger.info("=== PlantSeg severity_ratio (real ground truth) ===")
    for q in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        logger.info("   p%-3d = %.4f", q, np.percentile(ratios, q))

    logger.info("")
    logger.info("index -> %s", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plantseg_root", default="D:/pdeu_kaam/plantseg")
    p.add_argument("--roboflow_root",
                   default="D:/pdeu_kaam/potato-leaf-disease-segmentation-4g26g")
    p.add_argument("--results_root", default=str(PROJECT_ROOT / "results"))
    a = p.parse_args()
    main(a)
