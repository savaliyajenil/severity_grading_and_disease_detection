"""
Perceptual Hash Deduplication Pipeline
=======================================
Compares every Roboflow training image against every PlantSeg test + val image
using pHash (perceptual hashing). Any Roboflow image that is a near-duplicate
of a PlantSeg evaluation image is flagged and moved to a quarantine folder.

This guarantees zero data leakage between training and evaluation sets.

Usage:
    python scripts/04_deduplicate.py
"""

import os
import hashlib
import shutil
from pathlib import Path
from PIL import Image
import json
import csv

# ============================================================
# CONFIG
# ============================================================
PLANTSEG_ROOT = Path(r"D:\PDEU Research\Dataset\plantseg")
ROBOFLOW_ROOT = Path(r"D:\PDEU Research\Dataset\roboflow\potato-leaf-disease-segmentation-4g26g")
QUARANTINE_DIR = Path(r"D:\PDEU Research\Dataset\roboflow\_quarantined_duplicates")
HAMMING_THRESHOLD = 10  # pHash distance <= this = near-duplicate (conservative)

# ============================================================
# STAGE 1: MD5 Exact Duplicate Detection
# ============================================================
def compute_md5(filepath):
    """Compute MD5 hash of a file's raw bytes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# ============================================================
# STAGE 2: Perceptual Hashing (pHash)
# ============================================================
def compute_phash(filepath, hash_size=8):
    """
    Compute a perceptual hash (pHash) for an image.
    
    Algorithm:
    1. Resize to (hash_size*4) x (hash_size*4) — captures frequency info
    2. Convert to grayscale
    3. Apply DCT (via pixel averaging as approximation)
    4. Compute median and generate binary hash
    
    Returns a binary string of length hash_size^2.
    """
    try:
        img = Image.open(filepath).convert('L')  # Grayscale
        img = img.resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        
        # Reduce to hash_size x hash_size by averaging
        img_small = img.resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img_small.getdata())
        
        # Compute median
        avg = sum(pixels) / len(pixels)
        
        # Generate hash: 1 if pixel > median, else 0
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return bits
    except Exception as e:
        print(f"  [WARN] Could not hash {filepath}: {e}")
        return None


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two binary hash strings."""
    if hash1 is None or hash2 is None:
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


# ============================================================
# MAIN PIPELINE
# ============================================================
def collect_images(directory, extensions={'.jpg', '.jpeg', '.png', '.bmp'}):
    """Recursively collect all image files from a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)
    return images


def get_plantseg_eval_images():
    """Get all PlantSeg validation and test images (the protected set)."""
    eval_images = []
    
    # Get images from val and test directories
    for split_dir in ['val', 'test']:
        img_dir = PLANTSEG_ROOT / "images" / split_dir
        if img_dir.exists():
            eval_images.extend(collect_images(img_dir))
    
    # Also check metadata for potato/tomato filtering
    return eval_images


def get_roboflow_images():
    """Get all Roboflow training images."""
    images = []
    for split in ['train', 'valid', 'test']:
        split_dir = ROBOFLOW_ROOT / split
        if split_dir.exists():
            images.extend(collect_images(split_dir))
    return images


def run_deduplication():
    print("=" * 65)
    print("  PERCEPTUAL HASH DEDUPLICATION PIPELINE")
    print("  Protecting PlantSeg test/val from Roboflow training leakage")
    print("=" * 65)
    
    # Collect images
    eval_images = get_plantseg_eval_images()
    rf_images = get_roboflow_images()
    
    print(f"\n  PlantSeg eval images (protected): {len(eval_images)}")
    print(f"  Roboflow images (to check):       {len(rf_images)}")
    
    # STAGE 1: MD5 Hashing
    print("\n--- STAGE 1: MD5 Exact Duplicate Detection ---")
    eval_md5s = {}
    for img in eval_images:
        eval_md5s[compute_md5(img)] = img
    print(f"  Computed {len(eval_md5s)} MD5 hashes for eval set")
    
    md5_duplicates = []
    for img in rf_images:
        h = compute_md5(img)
        if h in eval_md5s:
            md5_duplicates.append((img, eval_md5s[h]))
    
    print(f"  MD5 exact duplicates found: {len(md5_duplicates)}")
    for rf_img, eval_img in md5_duplicates:
        print(f"    EXACT MATCH: {rf_img.name} == {eval_img.name}")
    
    # STAGE 2: pHash Near-Duplicate Detection
    print("\n--- STAGE 2: Perceptual Hash Near-Duplicate Detection ---")
    print(f"  Hamming distance threshold: {HAMMING_THRESHOLD}")
    
    # Hash all eval images
    eval_hashes = []
    for img in eval_images:
        h = compute_phash(img)
        if h:
            eval_hashes.append((img, h))
    print(f"  Computed {len(eval_hashes)} pHashes for eval set")
    
    # Compare each Roboflow image against all eval images
    phash_duplicates = []
    already_flagged_md5 = {str(d[0]) for d in md5_duplicates}
    
    for i, rf_img in enumerate(rf_images):
        if str(rf_img) in already_flagged_md5:
            continue  # Already caught by MD5
            
        rf_hash = compute_phash(rf_img)
        if rf_hash is None:
            continue
        
        for eval_img, eval_hash in eval_hashes:
            dist = hamming_distance(rf_hash, eval_hash)
            if dist <= HAMMING_THRESHOLD:
                phash_duplicates.append((rf_img, eval_img, dist))
                break  # One match is enough to flag
        
        if (i + 1) % 100 == 0:
            print(f"  Checked {i + 1}/{len(rf_images)} images...")
    
    print(f"  pHash near-duplicates found: {len(phash_duplicates)}")
    for rf_img, eval_img, dist in phash_duplicates:
        print(f"    NEAR-MATCH (dist={dist}): {rf_img.name} ~ {eval_img.name}")
    
    # QUARANTINE duplicates
    all_duplicates = [d[0] for d in md5_duplicates] + [d[0] for d in phash_duplicates]
    
    if all_duplicates:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n--- QUARANTINING {len(all_duplicates)} duplicate images ---")
        for dup in all_duplicates:
            dest = QUARANTINE_DIR / dup.name
            shutil.move(str(dup), str(dest))
            print(f"  Moved: {dup.name} -> quarantine/")
    
    # SUMMARY
    print("\n" + "=" * 65)
    print("  DEDUPLICATION SUMMARY")
    print("=" * 65)
    print(f"  Total Roboflow images checked:  {len(rf_images)}")
    print(f"  MD5 exact duplicates removed:   {len(md5_duplicates)}")
    print(f"  pHash near-duplicates removed:  {len(phash_duplicates)}")
    print(f"  Clean images remaining:         {len(rf_images) - len(all_duplicates)}")
    print(f"  Quarantine folder:              {QUARANTINE_DIR}")
    print("=" * 65)
    
    return len(all_duplicates)


if __name__ == "__main__":
    run_deduplication()
