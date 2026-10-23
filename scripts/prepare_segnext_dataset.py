#!/usr/bin/env python3
"""
Convert COCO annotations to PNG semantic segmentation masks for mmsegmentation CustomDataset.
Uses the unified index to get class mapping.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import shutil
import argparse

# Class mapping from class_map.json
CLASS_NAMES = [
    "background",
    "potato_early_blight",
    "potato_late_blight",
    "tomato_bacterial_leaf_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus",
]

# COCO category_id -> target class index (0-9)
# From class_map.json: disease_to_class and roboflow_to_class
COCO_TO_CLASS = {
    # PlantSeg mask values (from disease_to_mask_value)
    76: 1,   # potato early blight
    77: 2,   # potato late blight
    97: 3,   # tomato bacterial leaf spot
    98: 4,   # tomato early blight
    99: 5,   # tomato late blight
    100: 6,  # tomato leaf mold
    101: 7,  # tomato mosaic virus
    102: 8,  # tomato septoria leaf spot
    103: 9,  # tomato yellow leaf curl virus
    # Roboflow category_ids (1=Early Blight, 2=Late Blight)
    1: 1,    # Early Blight -> potato_early_blight
    2: 2,    # Late Blight -> potato_late_blight
}

def convert_coco_to_png(coco_json_path, image_dir, output_ann_dir, split_name):
    """Convert COCO annotations to PNG masks."""
    coco = COCO(str(coco_json_path))
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    
    img_ids = coco.getImgIds()
    print(f"Processing {split_name}: {len(img_ids)} images")
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = Path(image_dir) / img_name
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping")
            continue
        h, w = img.shape[:2]
        
        # Create empty mask (background = 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            cat_id = ann['category_id']
            target_class = COCO_TO_CLASS.get(cat_id)
            if target_class is None:
                # Unknown category, skip
                continue
            
            # Rasterize annotation
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    # Polygon format
                    for poly in seg:
                        if len(poly) >= 6:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], target_class)
                elif isinstance(seg, dict):
                    # RLE format
                    if 'counts' in seg:
                        rle = maskUtils.frPyObjects(seg, h, w)
                        m = maskUtils.decode(rle)
                        if m.ndim == 3:
                            m = m[:, :, 0]
                        mask[m == 1] = target_class
        
        # Save mask as PNG
        mask_name = Path(img_name).stem + '_labelTrainIds.png'
        cv2.imwrite(str(output_ann_dir / mask_name), mask)
    
    print(f"Done {split_name}: masks saved to {output_ann_dir}")

def copy_images(src_dir, dst_dir, split_name):
    """Copy images to mmseg structure."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_file in Path(src_dir).glob('*.jpg'):
        shutil.copy2(img_file, dst_dir / img_file.name)
    for img_file in Path(src_dir).glob('*.JPG'):
        shutil.copy2(img_file, dst_dir / img_file.name)
    print(f"Copied {split_name} images to {dst_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/potato-leaf-disease-segmentation-4g26g')
    parser.add_argument('--output_root', default='SegNeXt/data/potato_seg')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n=== Processing {split} ===")
        
        # Paths
        coco_json = data_root / split / '_annotations.coco.json'
        img_dir = data_root / split
        
        out_img_dir = output_root / 'img_dir' / split
        out_ann_dir = output_root / 'ann_dir' / split
        
        if not coco_json.exists():
            print(f"COCO JSON not found: {coco_json}")
            continue
        
        # Copy images
        copy_images(img_dir, out_img_dir, split)
        
        # Convert annotations
        convert_coco_to_png(coco_json, img_dir, out_ann_dir, split)
    
    print("\n=== Dataset preparation complete ===")
    print(f"Images: {output_root}/img_dir/{{train,val,test}}")
    print(f"Annotations: {output_root}/ann_dir/{{train,val,test}}")

if __name__ == '__main__':
    main()