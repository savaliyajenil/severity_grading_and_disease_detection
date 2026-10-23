#!/usr/bin/env python3
"""Download MSCAN-B pretrained weights."""

import requests
import torch
from pathlib import Path

def download_file(url, output_path, description):
    print(f"Downloading {description} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        r = requests.get(url, allow_redirects=True, timeout=120, headers=headers, stream=True)
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        print(f"  Status: {r.status_code}, Content-Type: {content_type}")
        
        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"  Downloaded {total_size / 1024 / 1024:.1f} MB")
        return total_size
    except Exception as e:
        print(f"  Error: {e}")
        return 0

def extract_backbone(full_ckpt_path, backbone_path):
    """Extract backbone weights from full model checkpoint."""
    print(f"Loading checkpoint from {full_ckpt_path}...")
    ckpt = torch.load(full_ckpt_path, map_location='cpu')
    
    # Check structure
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Extract backbone keys
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            # Remove 'backbone.' prefix for loading with init_cfg
            new_k = k[9:]  # len('backbone.') = 9
            backbone_state[new_k] = v
    
    print(f"Extracted {len(backbone_state)} backbone parameters")
    torch.save(backbone_state, backbone_path)
    print(f"Saved backbone to {backbone_path}")

def main():
    pretrained_dir = Path('C:/Users/student/Desktop/pdeu_kaam/SegNeXt/pretrained')
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    # Try TsingHua Cloud for full SegNeXt MSCAN-B model (ADE20K)
    # This is the full model, we'll extract backbone
    urls_to_try = [
        # Full model checkpoints from README
        ("https://cloud.tsinghua.edu.cn/f/1ea8000916284493810b/?dl=1", "SegNeXt MSCAN-B ADE20K full model"),
        ("https://cloud.tsinghua.edu.cn/f/b2d1eb94f5944d60b3d2/?dl=1", "SegNeXt MSCAN-S ADE20K full model"),
        ("https://cloud.tsinghua.edu.cn/f/5da98841b8384ba0988a/?dl=1", "SegNeXt MSCAN-T ADE20K full model"),
    ]
    
    full_model_path = pretrained_dir / 'segnext_b_ade_full.pth'
    backbone_path = pretrained_dir / 'mscan_b.pth'
    
    # If backbone already exists, check if valid
    if backbone_path.exists():
        try:
            ckpt = torch.load(backbone_path, map_location='cpu')
            print(f"Existing backbone found with {len(ckpt)} keys")
            if len(ckpt) > 100:
                print("Backbone looks valid, skipping download")
                return
        except:
            print("Existing backbone invalid, re-downloading")
    
    for url, desc in urls_to_try:
        size = download_file(url, full_model_path, desc)
        if size > 10_000_000:  # At least 10MB for a real model
            print(f"Successfully downloaded {desc}")
            extract_backbone(full_model_path, backbone_path)
            return
        else:
            print(f"Download too small ({size} bytes), trying next...")
    
    print("All downloads failed. Trying alternative...")
    # Try GitHub releases
    try:
        import subprocess
        result = subprocess.run([
            'curl', '-L', '-o', str(full_model_path),
            'https://github.com/Visual-Attention-Network/SegNeXt/releases/download/v1.0/segnext_b_512x512_ade_160k.pth'
        ], capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and full_model_path.stat().st_size > 10_000_000:
            extract_backbone(full_model_path, backbone_path)
            return
    except:
        pass
    
    print("ERROR: Could not download pretrained weights")
    print("Please manually download from: https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/")
    print("And place mscan_b.pth in SegNeXt/pretrained/")

if __name__ == '__main__':
    main()