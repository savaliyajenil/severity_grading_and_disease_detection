# Ablation results — SAMTSeg

Evaluated on the **PlantSeg test split** (160 images, potato + tomato, 9 diseases).
Training data: PlantSeg train (563) + harmonized Roboflow potato (623).
Validation: PlantSeg val (90 images), used for checkpoint selection.

Checkpoints selected by best validation mIoU; the test split was evaluated once per run.

| Configuration | Params | Best val mIoU | **Test mIoU** | Test disease mIoU | Test Dice | Test disease F1 | Test sev. acc | Test sev. ±1 |
|---|---|---|---|---|---|---|---|---|
| Baseline U-Net (no CBAM, no severity head) | 24.55M | 0.2814 | **0.2651** | 0.2025 | 0.3768 | 0.3180 | 0.2250 | 0.6687 |
| + CBAM (severity head off) | 24.59M | 0.2387 | **0.2235** | 0.1601 | 0.3234 | 0.2609 | 0.1875 | 0.3938 |
| + CBAM + severity head (no consistency loss) | 24.59M | 0.2600 | **0.2777** | 0.2161 | 0.3937 | 0.3365 | 0.4563 | 0.8125 |
| Full SAMTSeg (no pretrained encoder) | 24.59M | 0.1939 | **0.1736** | 0.1038 | 0.2468 | 0.1753 | 0.2688 | 0.6812 |
| PlantSeg Novelty (ResNet101, 512², PolyLR, Novelty) | 59.28M | 0.3683 | **0.3309** | 0.2697 | 0.4504 | 0.3963 | 0.3562 | 0.7563 |

## Severity grading accuracy by grade (full model)

Not yet computed — see `severity_grade.png` for the confusion-style view.
