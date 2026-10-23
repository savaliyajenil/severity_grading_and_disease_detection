import json
from pathlib import Path
from collections import Counter

anno = Path(r'D:\PDEU Research\Dataset\roboflow\potato-leaf-disease-segmentation-4g26g\train\_annotations.coco.json')
with open(anno) as f:
    coco = json.load(f)

print('=== Categories in Roboflow Potato Dataset ===')
for cat in coco['categories']:
    print(f"  ID: {cat['id']} | Name: {cat['name']}")

print(f"\nTotal images: {len(coco['images'])}")
print(f"Total annotations: {len(coco['annotations'])}")

cat_counts = Counter(ann['category_id'] for ann in coco['annotations'])
print('\nAnnotations per category:')
for cat in coco['categories']:
    print(f"  {cat['name']}: {cat_counts.get(cat['id'], 0)} annotations")

# Check if segmentation polygons exist
has_seg = sum(1 for ann in coco['annotations'] if ann.get('segmentation'))
has_bbox_only = sum(1 for ann in coco['annotations'] if not ann.get('segmentation'))
print(f"\nAnnotations WITH polygon segmentation: {has_seg}")
print(f"Annotations WITH bbox only (no polygon): {has_bbox_only}")
