"""
Cross-dataset leakage audit: Roboflow potato vs PlantSeg val/test.

Both datasets source imagery from the same public repositories (primarily
PlantVillage), so a Roboflow training image can be a resized, recompressed or
lightly cropped copy of a PlantSeg validation or test image. If that leaks into
training, reported benchmark metrics are inflated and the evaluation is invalid --
this is the single most common way a cross-dataset segmentation paper gets
rejected.

Two-stage protocol:
  1. Exact match on MD5 of the decoded RGB pixels (catches byte-identical files
     even when filenames differ).
  2. Perceptual hash (pHash, 64-bit DCT) with Hamming distance <= 5, which
     survives resizing, recompression and metadata changes.

Any Roboflow image flagged against PlantSeg val or test is reported here and must
be purged from training. PlantSeg val/test are never modified.

Run:
    python scripts/09_phash_dedup.py
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("dedup")

SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}


def iter_plantseg_targets(plantseg_root, plants=("Potato", "Tomato")):
    """Yield (split, image_path, name) for PlantSeg val and test images."""
    md = pd.read_csv(plantseg_root / "Metadata.csv")
    md["split"] = md["Split"].map(SPLIT_MAP)
    sel = md[md.Plant.isin(plants) & md["split"].isin(["val", "test"])]
    for _, row in sel.iterrows():
        path = plantseg_root / "images" / row["split"] / row["Name"]
        if path.exists():
            yield row["split"], path, row["Name"]


def iter_roboflow(roboflow_root):
    for split in ("train", "valid", "test"):
        d = roboflow_root / split
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            yield split, p, p.name


def hashes_for(path, hash_size):
    """Return (md5_of_rgb_pixels, phash). None if the image cannot be decoded."""
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            md5 = hashlib.md5(rgb.tobytes()).hexdigest()
            ph = imagehash.phash(rgb, hash_size=hash_size)
            return md5, ph
    except Exception:
        return None, None


def main(args):
    plantseg = Path(args.plantseg_root)
    roboflow = Path(args.roboflow_root)
    out_dir = Path(args.results_root) / "dedup"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("hashing PlantSeg val/test (%s)", ",".join(args.plants))
    bench_md5, bench_ph = {}, {}
    n_bench = 0
    for split, path, name in iter_plantseg_targets(plantseg, args.plants):
        md5, ph = hashes_for(path, args.hash_size)
        if md5 is None:
            logger.warning("undecodable: %s", path)
            continue
        bench_md5.setdefault(md5, []).append((split, name))
        bench_ph.setdefault(str(ph), []).append((split, name))
        n_bench += 1
    logger.info("PlantSeg val/test images hashed: %d", n_bench)

    logger.info("hashing Roboflow potato images")
    rows = []
    n_rf = 0
    for split, path, name in iter_roboflow(roboflow):
        md5, ph = hashes_for(path, args.hash_size)
        if md5 is None:
            logger.warning("undecodable: %s", path)
            continue
        n_rf += 1

        exact = bench_md5.get(md5, [])
        best_dist, best_match = 64, None
        if ph is not None:
            for cand_str, refs in bench_ph.items():
                cand = imagehash.hex_to_hash(cand_str)
                d = (cand - ph)
                if d < best_dist:
                    best_dist, best_match = d, refs
        near = best_match if (best_dist <= args.hamming) else None

        rows.append({
            "roboflow_split": split,
            "roboflow_file": name,
            "md5": md5,
            "phash": str(ph),
            "exact_match": bool(exact),
            "exact_ref": "|".join(f"{s}:{n}" for s, n in exact),
            "hamming": best_dist,
            "near_match": bool(near),
            "near_ref": "|".join(f"{s}:{n}" for s, n in near) if near else "",
            "flagged": bool(exact or near),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "roboflow_dedup_report.csv", index=False)

    n_exact = int(df["exact_match"].sum())
    n_near = int((df["near_match"] & ~df["exact_match"]).sum())
    n_flag = int(df["flagged"].sum())

    logger.info("")
    logger.info("=== RESULT ===")
    logger.info("Roboflow images scanned        : %d", n_rf)
    logger.info("PlantSeg val/test hashes       : %d", n_bench)
    logger.info("Roboflow EXACT duplicates      : %d", n_exact)
    logger.info("Roboflow NEAR duplicates (<=%d) : %d", args.hamming, n_near)
    logger.info("TOTAL FLAGGED (must be purged) : %d  (%.2f%%)",
                n_flag, 100 * n_flag / max(n_rf, 1))

    # Per-class breakdown, since the plan keeps only Early/Late Blight.
    coco = json.loads((roboflow / "train" / "_annotations.coco.json").read_text())
    fname_to_cat = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img = next(i for i in coco["images"] if i["id"] == img_id)
        fname_to_cat[img["file_name"]] = int(ann["category_id"])

    if n_flag:
        flagged = df[df["flagged"]].copy()
        flagged["category_id"] = flagged["roboflow_file"].map(fname_to_cat)
        flagged.to_csv(out_dir / "roboflow_flagged_images.csv", index=False)
        logger.info("")
        logger.info("Flagged by class: %s",
                    flagged["category_id"].value_counts().to_dict())
        logger.info("per-file list -> %s", out_dir / "roboflow_flagged_images.csv")

    summary = {
        "roboflow_scanned": n_rf,
        "plantseg_benchmark_images": n_bench,
        "exact_duplicates": n_exact,
        "near_duplicates": n_near,
        "hamming_threshold": args.hamming,
        "hash_size": args.hash_size,
        "total_flagged": n_flag,
    }
    (out_dir / "dedup_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("")
    logger.info("report -> %s", out_dir / "roboflow_dedup_report.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plantseg_root", default="D:/pdeu_kaam/plantseg")
    p.add_argument("--roboflow_root",
                   default="D:/pdeu_kaam/potato-leaf-disease-segmentation-4g26g")
    p.add_argument("--results_root",
                   default=str(PROJECT_ROOT / "results"))
    p.add_argument("--plants", nargs="+", default=["Potato", "Tomato"])
    p.add_argument("--hamming", type=int, default=5)
    p.add_argument("--hash_size", type=int, default=8)
    a = p.parse_args()
    main(a)
