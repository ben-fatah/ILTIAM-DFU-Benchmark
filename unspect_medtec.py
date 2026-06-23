"""
inspect_medetec.py
-------------------
Diagnose Medetec mask format so we can fix the loading threshold.
Run this FIRST before re-running run_medetec_test.py.

Run:
  python inspect_medetec.py
"""

import os
import glob
import numpy as np
import cv2

BASE      = r"C:\ai\gradutionProject\trainmodel\dataset"
IMG_DIR   = rf"{BASE}\Medetec_foot_ulcer_224\test\images"
MASK_DIR  = rf"{BASE}\Medetec_foot_ulcer_224\test\labels"
IMG_EXTS  = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_files(folder):
    out = []
    for e in IMG_EXTS:
        out += glob.glob(os.path.join(folder, f"*{e}"))
    return sorted(out)

def mask_for(ip, mdir):
    stem = os.path.splitext(os.path.basename(ip))[0]
    for e in IMG_EXTS:
        c = os.path.join(mdir, stem + e)
        if os.path.exists(c): return c
    return None

imgs  = list_files(IMG_DIR)
masks = list_files(MASK_DIR)

print(f"Images found : {len(imgs)}")
print(f"Masks found  : {len(masks)}")
print()

# Show first 5 image filenames
print("Sample image names:")
for p in imgs[:5]: print(f"  {os.path.basename(p)}")
print()
print("Sample mask names:")
for p in masks[:5]: print(f"  {os.path.basename(p)}")
print()

# Inspect each mask in detail
for ip in imgs[:8]:
    mp = mask_for(ip, MASK_DIR)
    if mp is None:
        print(f"  [NO MASK] {os.path.basename(ip)}")
        continue

    # Load as-is (grayscale)
    m_gray = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    # Load as color (some masks are RGB)
    m_rgb  = cv2.imread(mp, cv2.IMREAD_COLOR)

    if m_gray is None:
        print(f"  [LOAD FAILED] {mp}")
        continue

    uniq = np.unique(m_gray)
    print(f"  {os.path.basename(ip)}")
    print(f"    Grayscale shape : {m_gray.shape}  dtype: {m_gray.dtype}")
    print(f"    Unique values   : {uniq}")
    print(f"    Min/Max         : {m_gray.min()} / {m_gray.max()}")
    print(f"    Non-zero pixels : {(m_gray > 0).sum()} / {m_gray.size}")
    if m_rgb is not None and len(m_rgb.shape) == 3:
        print(f"    RGB unique R    : {np.unique(m_rgb[:,:,2])}")  # Red channel
        print(f"    RGB unique G    : {np.unique(m_rgb[:,:,1])}")
        print(f"    RGB unique B    : {np.unique(m_rgb[:,:,0])}")
    print()

print("="*50)
print("WHAT TO LOOK FOR:")
print("  - If unique values are [0, 1]        → threshold at 0.5 (divide by 1)")
print("  - If unique values are [0, 255]      → threshold at 127 (standard)")
print("  - If unique values are [0, 128, 255] → multi-class, use >0")
print("  - If unique values are [0, 38, 75..] → color-coded classes, need remapping")
print("  - If non-zero pixels = 0             → mask naming mismatch (wrong file paired)")