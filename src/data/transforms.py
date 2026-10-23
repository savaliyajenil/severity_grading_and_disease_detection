import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Mean/std for ImageNet-pretrained encoders
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_training_augmentation(image_size=(512, 512)):
    """
    PlantSeg-style training augmentation (adapted for albumentations 2.x):
    - RandomScale: scale_limit=(-0.3, 1.0) with PadIfNeeded to ensure 512px minimum
    - RandomCrop: crop_size=(512, 512)
    - HorizontalFlip: prob=0.5
    - PhotoMetricDistortion (ColorJitter + BrightnessContrast + HueSaturationValue)
    
    This approximates PlantSeg's RandomResize(0.5-2.0) + RandomCrop(512) + Flip + PhotoMetricDistortion
    """
    h, w = image_size
    return A.Compose([
        # Scale + Pad + Crop ensures crop-size output
        A.Compose([
            A.RandomScale(scale_limit=(-0.3, 1.0), p=1.0),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            A.RandomCrop(height=h, width=w, p=1.0),
        ], p=1.0),
        
        # RandomFlip
        A.HorizontalFlip(p=0.5),
        
        # PhotoMetricDistortion approximation
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size=(512, 512)):
    """
    PlantSeg-style validation: Resize to 512x512 (center crop from larger resize)
    Actually, PlantSeg test uses: Resize to (2048, 512) keep_ratio=True, then test on whole image
    But for consistent validation metrics, we resize to the crop size
    """
    h, w = image_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])