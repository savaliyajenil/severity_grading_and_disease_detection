#!/usr/bin/env python3
"""
SegNeXt Training Script for Potato Leaf Disease Segmentation
Optimized for: AMD Threadripper Pro 5945WX (12 cores), RTX A4000 (16GB), 64GB RAM
Uses mmseg's proper training system with hook-based LR scheduling
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
SEGNEXT_ROOT = PROJECT_ROOT / "SegNeXt"
sys.path.insert(0, str(SEGNEXT_ROOT))

def check_environment():
    """Verify training environment"""
    print("=" * 60)
    print("TRAINING ENVIRONMENT CHECK")
    print("=" * 60)
    
    # GPU check
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # CPU check
    print(f"CPU Threads: {torch.get_num_threads()}")
    
    # Memory check
    import psutil
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.total / 1024**3:.1f} GB ({mem.available / 1024**3:.1f} GB available)")
    
    # mmseg check
    try:
        import mmcv
        import mmseg
        print(f"MMCV: {mmcv.__version__}")
        print(f"MMSeg: {mmseg.__version__}")
    except Exception as e:
        print(f"Import error: {e}")
        return False
    
    # Config check
    config_path = SEGNEXT_ROOT / "local_configs/segnext/base/segnext_base_potato_512x512_160k.py"
    print(f"Config exists: {config_path.exists()}")
    
    # Data check
    data_root = SEGNEXT_ROOT / "data/potato_seg"
    train_img = data_root / "img_dir/train"
    train_ann = data_root / "ann_dir/train"
    print(f"Train images: {len(list(train_img.glob('*'))) if train_img.exists() else 0}")
    print(f"Train annotations: {len(list(train_ann.glob('*'))) if train_ann.exists() else 0}")
    
    # Pretrained weights check
    pretrained = SEGNEXT_ROOT / "pretrained/mscan_b.pth"
    if pretrained.exists():
        size = pretrained.stat().st_size / 1024**2
        print(f"Pretrained backbone: {size:.1f} MB")
        if size < 10:
            print("  WARNING: Pretrained file too small, may be invalid")
    else:
        print("Pretrained backbone: NOT FOUND (will train from scratch)")
    
    return True
def optimize_config_for_hardware():
    """Create optimized config for the specific hardware"""
    config_path = SEGNEXT_ROOT / "local_configs/segnext/base/segnext_base_potato_512x512_160k.py"
    
    # Read original config
    with open(config_path, 'r') as fp:
        config_content = fp.read()
    
    # Optimizations for RTX A4000 (16GB) + 12-core CPU + 64GB RAM
    optimizations = {
        # Increase batch size to utilize 16GB GPU (4 -> 6)
        'samples_per_gpu=4': 'samples_per_gpu=6',
        # Increase workers for 12-core CPU (4 -> 8)
        'workers_per_gpu=4': 'workers_per_gpu=8',
        # Add gradient clipping for stability
        'optimizer_config = dict()': 'optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))',
    }
    
    for old, new in optimizations.items():
        if old in config_content:
            config_content = config_content.replace(old, new)
            print(f"  Applied: {old} -> {new}")
    
    # Write optimized config
    optimized_config = SEGNEXT_ROOT / "local_configs/segnext/base/segnext_base_potato_512x512_160k_optimized.py"
    with open(optimized_config, 'w') as fp:
        fp.write(config_content)
    
    print(f"Optimized config saved to: {optimized_config}")
    return str(optimized_config)

def run_training(config_path, work_dir, gpus=1):
    """Run training using mmseg's train.py"""
    
    train_script = SEGNEXT_ROOT / "tools/train.py"
    
    # Build command
    cmd = [
        sys.executable, str(train_script),
        config_path,
        f"--work-dir={work_dir}",
        f"--gpu-ids=0",
        "--seed=42",
        "--deterministic",
    ]
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Work directory: {work_dir}")
    print("")
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{SEGNEXT_ROOT}:{env.get('PYTHONPATH', '')}"
    env['OMP_NUM_THREADS'] = '12'
    env['MKL_NUM_THREADS'] = '12'
    
    # Run training
    try:
        process = subprocess.Popen(
            cmd,
            cwd=SEGNEXT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def main():
    print("SegNeXt Potato Leaf Disease Segmentation Training")
    print("Hardware: Threadripper Pro 5945WX | RTX A4000 | 64GB RAM")
    print("")
    
    # Check environment
    if not check_environment():
        print("Environment check failed!")
        return 1
    
    # Create optimized config
    print("\nOptimizing config for hardware...")
    config_path = optimize_config_for_hardware()
    
    # Setup work directory
    work_dir = PROJECT_ROOT / "results/segnext_potato_optimized"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    print(f"\nStarting training in 3 seconds... (Ctrl+C to cancel)")
    import time
    time.sleep(3)
    
    success = run_training(config_path, str(work_dir))
    
    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {work_dir}")
        print("Check training_log.csv and checkpoints/ for results")
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED OR INTERRUPTED")
        print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())