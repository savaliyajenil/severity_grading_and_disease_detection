import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def audit_dataset(dataset_dir, output_path, num_samples=6):
    """
    Reads COCO annotations from a Roboflow dataset and plots random images 
    with their corresponding segmentation masks overlaid.
    """
    dataset_path = Path(dataset_dir)
    
    # Roboflow usually exports into train/valid/test folders
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        print(f"Error: {train_dir} does not exist.")
        return
        
    anno_file = train_dir / "_annotations.coco.json"
    if not anno_file.exists():
        print(f"Error: Annotations not found at {anno_file}")
        return
        
    with open(anno_file, 'r') as f:
        coco = json.load(f)
        
    categories = {c['id']: c['name'] for c in coco['categories']}
    
    # Pick random images that have annotations
    annotated_image_ids = list(set([ann['image_id'] for ann in coco['annotations']]))
    if len(annotated_image_ids) < num_samples:
        num_samples = len(annotated_image_ids)
        
    sample_img_ids = random.sample(annotated_image_ids, num_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_id in enumerate(sample_img_ids):
        # Get image info
        img_info = next(item for item in coco['images'] if item["id"] == img_id)
        img_path = train_dir / img_info['file_name']
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw annotations
        anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
        
        # Create a mask overlay
        mask_overlay = np.zeros_like(img, dtype=np.uint8)
        
        for ann in anns:
            cat_name = categories[ann['category_id']]
            # Roboflow COCO segmentation format: [[x1, y1, x2, y2, ...]]
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    
                    # Random color for mask
                    color = np.random.randint(100, 255, 3).tolist()
                    
                    # Draw polygon
                    cv2.fillPoly(mask_overlay, [pts], color)
                    # Draw boundary
                    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    
                    # Put text
                    cv2.putText(img, cat_name, (pts[0][0], pts[0][1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif 'bbox' in ann:
                # Fallback if only bbox exists
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, cat_name + " (bbox)", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend image and mask
        alpha = 0.5
        blended = cv2.addWeighted(img, 1, mask_overlay, alpha, 0)
        
        axes[i].imshow(blended)
        axes[i].set_title(img_info['file_name'])
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved audit visualization to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    
    audit_dataset(args.dataset_dir, args.out_file)
