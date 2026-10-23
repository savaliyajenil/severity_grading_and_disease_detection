import sys
import importlib.util
from pathlib import Path
import argparse
import logging
import torch

PROJECT_ROOT = Path(r"C:\Users\student\Desktop\pdeu_kaam")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Load the training script module dynamically
spec = importlib.util.spec_from_file_location(
    "train_samtseg", 
    PROJECT_ROOT / "scripts" / "11_train_samtseg.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Pull needed names
get_model = mod.get_model
NUM_CLASSES = mod.NUM_CLASSES
CLASS_NAMES = mod.CLASS_NAMES
get_validation_augmentation = mod.get_validation_augmentation
UnifiedSegDataset = mod.UnifiedSegDataset
DataLoader = mod.DataLoader
run_epoch = mod.run_epoch
per_class_plot = mod.per_class_plot
evaluate_checkpoint = mod.evaluate_checkpoint

# Patch DataLoader to use num_workers=0 to avoid multiprocessing spawn issues
original_DataLoader = DataLoader
mod.DataLoader = lambda *a, **kw: original_DataLoader(*a, num_workers=0, **kw)

# Load best checkpoint
ckpt_path = Path(r"C:\Users\student\Desktop\pdeu_kaam\results\samtseg_v3_boundary_clean\checkpoints\best.pth")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ck_args = argparse.Namespace(**ckpt["args"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir = Path(r"C:\Users\student\Desktop\pdeu_kaam\results\samtseg_v3_boundary_clean")

# Run evaluation
evaluate_checkpoint(ckpt_path, ck_args, device, run_dir)
print("Evaluation complete - per-class plots should now exist")