"""
Quick script to check where the Medetec images actually are.
Run this in ai_env to find the full 160 images.

    conda activate ai_env
    python check_medetec.py
"""
import os
from pathlib import Path

root = Path(r"C:\ai\gradutionProject\trainmodel\dataset\Medetec_foot_ulcer_224")

print("=== Full directory tree ===")
for p in sorted(root.rglob("*")):
    if p.is_dir():
        n_files = len(list(p.glob("*.*")))
        print(f"  DIR  {p.relative_to(root)}  ({n_files} files)")
    else:
        # Only show first few files per folder to avoid spam
        pass

print("\n=== Image counts per folder ===")
img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
for folder in sorted(root.rglob("*")):
    if folder.is_dir():
        imgs = [f for f in folder.iterdir() if f.suffix.lower() in img_exts]
        if imgs:
            print(f"  {len(imgs):>4} images  in  {folder.relative_to(root)}")

print("\n=== Looking for 'labels' or 'masks' folders ===")
for folder in sorted(root.rglob("*")):
    if folder.is_dir() and folder.name.lower() in {"labels", "masks", "annotations"}:
        imgs = [f for f in folder.iterdir() if f.suffix.lower() in img_exts]
        print(f"  {len(imgs):>4} masks   in  {folder.relative_to(root)}")