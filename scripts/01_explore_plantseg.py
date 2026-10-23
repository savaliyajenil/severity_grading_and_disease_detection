"""
PlantSeg Dataset Explorer - Potato & Tomato Subset
===================================================
This script provides a comprehensive visual exploration of the PlantSeg
dataset, focusing on the Potato and Tomato subsets. It generates:
1. Sample image-mask overlay visualizations
2. Disease distribution charts
3. Mask ratio (severity) distribution analysis
4. Per-disease statistics summary

Run: python scripts/01_explore_plantseg.py
Output: D:\PDEU Research\outputs\exploration\
"""

import os
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2

# ─── Configuration ───────────────────────────────────────────────────────────

DATASET_ROOT = Path("D:/PDEU Research/Dataset/plantseg")
OUTPUT_DIR = Path("D:/PDEU Research/outputs/exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PLANTS = ["Potato", "Tomato"]

# Potato/Tomato disease class -> category_id mapping (from Metadata.csv)
DISEASE_CLASS_MAP = {
    75: "Potato Early Blight",
    76: "Potato Late Blight",
    96: "Tomato Bacterial Leaf Spot",
    97: "Tomato Early Blight",
    98: "Tomato Late Blight",
    99: "Tomato Leaf Mold",
    100: "Tomato Mosaic Virus",
    101: "Tomato Septoria Leaf Spot",
    102: "Tomato Yellow Leaf Curl Virus",
}

# Colors for each disease (for visualization)
DISEASE_COLORS = {
    75: (255, 100, 100),    # Red-ish
    76: (100, 100, 255),    # Blue-ish
    96: (255, 200, 50),     # Yellow
    97: (255, 150, 50),     # Orange
    98: (50, 200, 50),      # Green
    99: (200, 50, 200),     # Purple
    100: (50, 200, 200),    # Cyan
    101: (255, 100, 200),   # Pink
    102: (200, 200, 50),    # Olive
}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_metadata():
    """Load and filter Metadata.csv for target plants."""
    df = pd.read_csv(DATASET_ROOT / "Metadata.csv")
    mask = df["Plant"].isin(TARGET_PLANTS)
    return df[mask].copy()


def load_coco_annotations(split="train"):
    """Load COCO-format annotation JSON for a given split."""
    json_path = DATASET_ROOT / f"annotation_{split}.json"
    print(f"Loading {json_path.name}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_potato_tomato_images(coco_data):
    """Filter COCO data to only potato/tomato images."""
    pt_images = [
        img for img in coco_data["images"]
        if any(plant.lower() in img["file_name"].lower()
               for plant in ["potato", "tomato"])
    ]
    pt_ids = {img["id"] for img in pt_images}
    pt_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in pt_ids
    ]
    return pt_images, pt_annotations


# ─── Visualization Functions ─────────────────────────────────────────────────

def visualize_sample_with_mask(split="train", n_samples=6):
    """
    Visualize n_samples potato/tomato images with their segmentation masks
    overlaid. Shows the original image, the mask, and the overlay side by side.
    """
    coco_data = load_coco_annotations(split)
    pt_images, pt_annotations = get_potato_tomato_images(coco_data)

    # Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in pt_annotations:
        ann_by_image[ann["image_id"]].append(ann)

    # Pick samples from different diseases
    seen_diseases = set()
    selected = []
    for img in pt_images:
        img_anns = ann_by_image.get(img["id"], [])
        if not img_anns:
            continue
        disease_id = img_anns[0]["category_id"]
        if disease_id not in seen_diseases and disease_id in DISEASE_CLASS_MAP:
            seen_diseases.add(disease_id)
            selected.append((img, img_anns))
        if len(selected) >= n_samples:
            break

    if not selected:
        print("No potato/tomato images found!")
        return

    fig, axes = plt.subplots(len(selected), 3, figsize=(18, 5 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "PlantSeg Dataset: Potato & Tomato Disease Samples\n"
        "(Original Image | Segmentation Mask | Overlay)",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for i, (img_info, annotations) in enumerate(selected):
        fname = img_info["file_name"]
        img_path = DATASET_ROOT / "images" / split / fname
        mask_path = DATASET_ROOT / "annotations" / split / fname.replace(".jpg", ".png")

        # Load image
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Load mask (grayscale PNG where pixel value = category_id)
        if mask_path.exists():
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Reconstruct mask from COCO polygon annotations
            mask_raw = np.zeros((h, w), dtype=np.uint8)
            for ann in annotations:
                cat_id = ann["category_id"]
                for seg in ann["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask_raw, [pts], color=int(cat_id))

        # Create colored mask overlay
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cat_id, color in DISEASE_COLORS.items():
            mask_colored[mask_raw == cat_id] = color

        # Create overlay
        overlay = image.copy()
        disease_pixels = mask_raw > 0
        overlay[disease_pixels] = (
            0.5 * overlay[disease_pixels] + 0.5 * mask_colored[disease_pixels]
        ).astype(np.uint8)

        # Draw contours for clarity
        contours, _ = cv2.findContours(
            (mask_raw > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

        # Get disease name
        disease_id = annotations[0]["category_id"]
        disease_name = DISEASE_CLASS_MAP.get(disease_id, f"Unknown ({disease_id})")

        # Calculate severity
        total_pixels = h * w
        disease_pixel_count = np.sum(mask_raw > 0)
        severity_pct = (disease_pixel_count / total_pixels) * 100

        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{disease_name}\n({w}×{h})", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_colored)
        axes[i, 1].set_title(
            f"Segmentation Mask\n{len(annotations)} lesion(s)", fontsize=11
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Overlay | Severity: {severity_pct:.1f}%", fontsize=11
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sample_visualizations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_disease_distribution(metadata_df):
    """Bar chart showing the number of images per disease class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "PlantSeg: Potato & Tomato Disease Distribution",
        fontsize=14, fontweight="bold",
    )

    for idx, plant in enumerate(TARGET_PLANTS):
        plant_df = metadata_df[metadata_df["Plant"] == plant]
        disease_counts = plant_df["Disease"].value_counts()

        colors = plt.cm.Set2(np.linspace(0, 1, len(disease_counts)))
        bars = axes[idx].barh(
            disease_counts.index, disease_counts.values, color=colors, edgecolor="gray"
        )
        axes[idx].set_title(f"{plant} ({len(plant_df)} images)", fontsize=12)
        axes[idx].set_xlabel("Number of Images")

        # Add value labels on bars
        for bar, val in zip(bars, disease_counts.values):
            axes[idx].text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10,
            )

        # Add split breakdown
        for disease in disease_counts.index:
            d_df = plant_df[plant_df["Disease"] == disease]
            splits = d_df["Split"].value_counts()
            split_str = " | ".join(
                f"{s}: {c}" for s, c in splits.items()
            )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "disease_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_severity_distribution(metadata_df):
    """Histogram of mask ratios (severity proxy) per disease."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Disease Severity Distribution (Mask Ratio = Diseased Area / Total Area)",
        fontsize=14, fontweight="bold",
    )

    for idx, plant in enumerate(TARGET_PLANTS):
        plant_df = metadata_df[metadata_df["Plant"] == plant]

        for disease in plant_df["Disease"].unique():
            d_df = plant_df[plant_df["Disease"] == disease]
            mask_ratios = d_df["Mask ratio"].astype(float)
            axes[idx].hist(
                mask_ratios * 100, bins=20, alpha=0.6,
                label=disease.replace(plant.lower() + " ", "").title(),
                edgecolor="black", linewidth=0.5,
            )

        axes[idx].set_title(f"{plant}", fontsize=12)
        axes[idx].set_xlabel("Severity (% of image area diseased)")
        axes[idx].set_ylabel("Number of Images")
        axes[idx].legend(fontsize=8)
        axes[idx].axvline(x=5, color="green", linestyle="--", alpha=0.5, label="Grade 1 threshold")
        axes[idx].axvline(x=25, color="orange", linestyle="--", alpha=0.5, label="Grade 2 threshold")
        axes[idx].axvline(x=50, color="red", linestyle="--", alpha=0.5, label="Grade 3 threshold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "severity_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_statistics(metadata_df):
    """Print a comprehensive summary table."""
    print("\n" + "=" * 70)
    print("PLANTSEG DATASET — POTATO & TOMATO SUMMARY")
    print("=" * 70)

    for plant in TARGET_PLANTS:
        plant_df = metadata_df[metadata_df["Plant"] == plant]
        print(f"\n{'-' * 40}")
        print(f"  {plant.upper()} ({len(plant_df)} total images)")
        print(f"{'-' * 40}")

        for disease in sorted(plant_df["Disease"].unique()):
            d_df = plant_df[plant_df["Disease"] == disease]
            mask_ratios = d_df["Mask ratio"].astype(float)
            splits = d_df["Split"].value_counts()

            print(f"\n  Disease: {disease}")
            print(f"    Count: {len(d_df)} images")
            print(f"    Splits: {', '.join(f'{s}={c}' for s, c in splits.items())}")
            print(f"    Severity (mask ratio):")
            print(f"      Mean: {mask_ratios.mean() * 100:.1f}%")
            print(f"      Median: {mask_ratios.median() * 100:.1f}%")
            print(f"      Min: {mask_ratios.min() * 100:.1f}%")
            print(f"      Max: {mask_ratios.max() * 100:.1f}%")
            print(f"      Std: {mask_ratios.std() * 100:.1f}%")

    # Overall summary
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {len(metadata_df)} images across {metadata_df['Disease'].nunique()} diseases")
    print(f"  Potato: {len(metadata_df[metadata_df['Plant'] == 'Potato'])} images, "
          f"{metadata_df[metadata_df['Plant'] == 'Potato']['Disease'].nunique()} diseases")
    print(f"  Tomato: {len(metadata_df[metadata_df['Plant'] == 'Tomato'])} images, "
          f"{metadata_df[metadata_df['Plant'] == 'Tomato']['Disease'].nunique()} diseases")
    print(f"{'=' * 70}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("PlantSeg Dataset Explorer - Potato & Tomato Focus")
    print("=" * 50)

    # Load metadata
    print("\n[1/4] Loading metadata...")
    metadata_df = load_metadata()
    print(f"  Found {len(metadata_df)} potato/tomato images in Metadata.csv")

    # Print statistics
    print("\n[2/4] Computing summary statistics...")
    print_summary_statistics(metadata_df)

    # Visualize samples
    print("\n[3/4] Generating sample visualizations...")
    visualize_sample_with_mask(split="train", n_samples=6)

    # Distribution plots
    print("\n[4/4] Generating distribution plots...")
    plot_disease_distribution(metadata_df)
    plot_severity_distribution(metadata_df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
