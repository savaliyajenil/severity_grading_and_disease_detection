"""
Visual Mask Quality Audit
=========================
Randomly samples images from the Roboflow potato dataset, renders their
COCO polygon annotations as colored overlays, and saves a grid visualization
for manual quality inspection.

Usage:
    python scripts/05_visual_audit.py
"""

import json
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
ROBOFLOW_ROOT = Path(r"D:\PDEU Research\Dataset\roboflow\potato-leaf-disease-segmentation-4g26g")
OUTPUT_DIR = Path(r"D:\PDEU Research\audit_results")
NUM_SAMPLES = 12  # 4x3 grid
SEED = 42

# Color map for disease classes
COLORS = {
    'Early Blight': (255, 100, 50),    # Orange-red
    'Late Blight':  (50, 100, 255),    # Blue
    'Leaf Miner':   (100, 255, 100),   # Green
    'Nematode':     (255, 255, 50),    # Yellow
}


def load_coco_annotations(split_dir):
    """Load COCO annotations from a split directory."""
    anno_file = split_dir / "_annotations.coco.json"
    with open(anno_file, 'r') as f:
        return json.load(f)


def render_mask_overlay(image, annotations, categories, alpha=0.4):
    """
    Renders polygon annotations as semi-transparent colored overlays on the image.
    Also draws polygon boundaries and class labels.
    
    Returns:
        overlay_image: Image with masks overlaid
        mask_only: Pure mask image (for quality inspection)
    """
    overlay = image.copy()
    mask_only = np.zeros_like(image)
    
    for ann in annotations:
        cat_name = categories.get(ann['category_id'], 'unknown')
        color = COLORS.get(cat_name, (200, 200, 200))
        
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                
                # Fill polygon on overlay
                cv2.fillPoly(overlay, [pts], color)
                cv2.fillPoly(mask_only, [pts], color)
                
                # Draw boundary (thick line for visibility)
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
                
                # Label
                centroid_x = int(np.mean(pts[:, 0]))
                centroid_y = int(np.mean(pts[:, 1]))
                cv2.putText(image, cat_name, (centroid_x - 30, centroid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Blend
    blended = cv2.addWeighted(image, 1.0, overlay, alpha, 0)
    
    return blended, mask_only


def run_audit():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load annotations from train split
    train_dir = ROBOFLOW_ROOT / "train"
    coco = load_coco_annotations(train_dir)
    
    categories = {c['id']: c['name'] for c in coco['categories']}
    
    # Build image_id → annotations mapping
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    # Get images that have annotations
    annotated_imgs = [img for img in coco['images'] if img['id'] in img_anns]
    
    # Filter to only Early Blight and Late Blight (our target classes)
    target_cat_ids = set()
    for cat in coco['categories']:
        if cat['name'] in ('Early Blight', 'Late Blight'):
            target_cat_ids.add(cat['id'])
    
    target_imgs = []
    for img in annotated_imgs:
        anns = img_anns[img['id']]
        if any(ann['category_id'] in target_cat_ids for ann in anns):
            target_imgs.append(img)
    
    if len(target_imgs) < NUM_SAMPLES:
        sample_imgs = target_imgs
    else:
        sample_imgs = random.sample(target_imgs, NUM_SAMPLES)
    
    print(f"Auditing {len(sample_imgs)} randomly sampled images...")
    
    # === GRID 1: Overlay visualization (image + mask) ===
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('ROBOFLOW POTATO DATASET — Mask Quality Audit\n'
                 'Red=Early Blight | Blue=Late Blight | Polygon boundaries in color',
                 fontsize=14, fontweight='bold')
    
    for idx, img_info in enumerate(sample_imgs):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        
        img_path = train_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            ax.set_title(f"MISSING: {img_info['file_name']}")
            ax.axis('off')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        anns = img_anns.get(img_info['id'], [])
        blended, mask_only = render_mask_overlay(image, anns, categories)
        
        # Count polygon vertices to assess tightness
        total_vertices = 0
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    total_vertices += len(seg) // 2
        
        ax.imshow(blended)
        ax.set_title(f"{img_info['file_name']}\n{total_vertices} polygon vertices", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_imgs), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis('off')
    
    plt.tight_layout()
    overlay_path = OUTPUT_DIR / "mask_quality_audit_overlay.png"
    plt.savefig(str(overlay_path), dpi=150, bbox_inches='tight')
    print(f"Saved overlay audit: {overlay_path}")
    
    # === GRID 2: Mask-only visualization (to check polygon precision) ===
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 15))
    fig2.suptitle('MASK-ONLY VIEW — Check polygon tightness and boundary accuracy\n'
                  'Red=Early Blight | Blue=Late Blight',
                  fontsize=14, fontweight='bold')
    
    for idx, img_info in enumerate(sample_imgs):
        row, col = divmod(idx, cols)
        ax = axes2[row][col]
        
        img_path = train_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            ax.axis('off')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        anns = img_anns.get(img_info['id'], [])
        _, mask_only = render_mask_overlay(image.copy(), anns, categories)
        
        ax.imshow(mask_only)
        ax.set_title(f"{img_info['file_name']}", fontsize=8)
        ax.axis('off')
    
    for idx in range(len(sample_imgs), rows * cols):
        row, col = divmod(idx, cols)
        axes2[row][col].axis('off')
    
    plt.tight_layout()
    mask_path = OUTPUT_DIR / "mask_quality_audit_masks.png"
    plt.savefig(str(mask_path), dpi=150, bbox_inches='tight')
    print(f"Saved mask-only audit: {mask_path}")
    
    # === STATS ===
    print("\n--- Annotation Quality Statistics ---")
    vertex_counts = []
    has_segmentation = 0
    bbox_only = 0
    
    for ann in coco['annotations']:
        if ann.get('segmentation') and len(ann['segmentation']) > 0:
            has_segmentation += 1
            for seg in ann['segmentation']:
                vertex_counts.append(len(seg) // 2)
        else:
            bbox_only += 1
    
    print(f"  Annotations with polygon: {has_segmentation}")
    print(f"  Annotations bbox-only:    {bbox_only}")
    if vertex_counts:
        print(f"  Polygon vertex count:")
        print(f"    Min:    {min(vertex_counts)}")
        print(f"    Max:    {max(vertex_counts)}")
        print(f"    Mean:   {sum(vertex_counts)/len(vertex_counts):.1f}")
        print(f"    Median: {sorted(vertex_counts)[len(vertex_counts)//2]}")
    
    # Flag: if median vertices < 6, polygons are likely just bounding boxes
    median_v = sorted(vertex_counts)[len(vertex_counts)//2] if vertex_counts else 0
    if median_v <= 4:
        print("\n  [WARN] Median vertex count <= 4 — polygons may be bounding boxes!")
    elif median_v <= 8:
        print("\n  [CAUTION] Median vertex count is low — polygons may be rough approximations.")
    else:
        print(f"\n  [GOOD] Median vertex count ({median_v}) suggests detailed polygon annotations.")


if __name__ == "__main__":
    run_audit()
