"""
Roboflow Dataset Downloader
===========================
This script downloads the supplementary Potato and Tomato datasets
from Roboflow Universe to supplement the PlantSeg dataset.

Usage:
1. Pip install roboflow: `pip install roboflow`
2. Get your free Roboflow API key from https://app.roboflow.com/
3. Run: `python scripts/02_download_roboflow.py --api_key YOUR_API_KEY`
"""

import argparse
import os
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: The 'roboflow' package is not installed.")
    print("Please run: pip install roboflow")
    exit(1)

# Dataset configurations based on our research phase
DATASETS = [
    {
        "workspace": "potatocare",
        "project": "potatocare-multi-class-3",
        "version": 1,  # Change if a specific version is needed
        "format": "coco",
        "desc": "PotatoCare Multi-Class (2,155 images)"
    },
    {
        "workspace": "new-workspace-5zciy",
        "project": "potato-leaf-disease-segmentation-4g26g",
        "version": 1,
        "format": "coco",
        "desc": "Potato Leaf Disease 4-class (~1,400 images)"
    },
    {
        "workspace": "segahs-workspace",
        "project": "tomato-leaf-disease-kfy8b",
        "version": 1,
        "format": "coco",
        "desc": "Tomato Leaf Disease by segahs (~500 images)"
    }
]

def download_datasets(api_key, download_dir):
    rf = Roboflow(api_key=api_key)
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading datasets to: {download_path.absolute()}")
    print("-" * 50)
    
    for ds in DATASETS:
        print(f"\nDownloading {ds['desc']}...")
        try:
            project = rf.workspace(ds['workspace']).project(ds['project'])
            dataset = project.version(ds['version']).download(ds['format'], location=str(download_path / ds['project']))
            print(f"Successfully downloaded {ds['project']}!")
        except Exception as e:
            print(f"Failed to download {ds['project']}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download supplementary datasets from Roboflow.")
    parser.add_argument("--api_key", type=str, required=True, help="Your Roboflow API key")
    parser.add_argument("--out_dir", type=str, default="Dataset/roboflow", help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve relative to project root
    project_root = Path(__file__).parent.parent
    out_path = project_root / args.out_dir
    
    download_datasets(args.api_key, out_path)
