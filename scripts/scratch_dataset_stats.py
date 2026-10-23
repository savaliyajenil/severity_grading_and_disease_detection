import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

plantseg_root = Path(r"D:\PDEU Research\Dataset\plantseg")
metadata_path = plantseg_root / "Metadata.csv"

df = pd.read_csv(metadata_path)
df['Plant'] = df['Plant'].str.strip().str.lower()
df['Disease'] = df['Disease'].str.strip()
df['Split'] = df['Split'].str.strip().str.lower()

# Filter to potato and tomato
target = df[df['Plant'].isin(['potato', 'tomato'])]

print("=" * 65)
print("PlantSeg — Image Counts per Crop, Disease, and Split")
print("=" * 65)

for plant in ['potato', 'tomato']:
    pdata = target[target['Plant'] == plant]
    print(f"\n{'='*30}")
    print(f"  {plant.upper()}  (Total: {len(pdata)} images)")
    print(f"{'='*30}")
    for disease, dgroup in pdata.groupby('Disease'):
        train_cnt = len(dgroup[dgroup['Split'] == 'training'])
        val_cnt   = len(dgroup[dgroup['Split'] == 'validation'])
        test_cnt  = len(dgroup[dgroup['Split'] == 'test'])
        total     = len(dgroup)
        print(f"  {disease:<35} train={train_cnt:>3}  val={val_cnt:>3}  test={test_cnt:>3}  total={total:>3}")

# Roboflow potato stats
print("\n\n" + "=" * 65)
print("Roboflow — Potato Leaf Disease Segmentation 4-Class")
print("=" * 65)
rf_anno = Path(r"D:\PDEU Research\Dataset\roboflow\potato-leaf-disease-segmentation-4g26g\train\_annotations.coco.json")
with open(rf_anno) as f:
    coco = json.load(f)

cats = {c['id']: c['name'] for c in coco['categories']}
from collections import Counter
cat_img_counts = defaultdict(set)
for ann in coco['annotations']:
    cat_img_counts[ann['category_id']].add(ann['image_id'])

print(f"\n  Total images: {len(coco['images'])}")
print(f"  Total annotations: {len(coco['annotations'])}")
print("\n  Unique images per class:")
for cat in coco['categories']:
    if cat['id'] in cat_img_counts:
        print(f"    {cat['name']:<35} images={len(cat_img_counts[cat['id']]):>4}")

# Useful images (Early + Late Blight only)
useful_ids = cat_img_counts[1] | cat_img_counts[2]  # Early and Late Blight
print(f"\n  Usable images (Early+Late Blight only): {len(useful_ids)}")
