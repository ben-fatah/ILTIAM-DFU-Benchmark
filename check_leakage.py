"""
overlap_matrix.py
-----------------
Checks ALL 7 wound datasets against EACH OTHER for shared/duplicate images,
so you know which folders can be safely combined for training vs testing.

Output:
  1. A pairwise OVERLAP MATRIX (how many images each dataset shares with each other).
  2. Per-pair sample images saved to ./overlap_pairs/<A>__vs__<B>/ so you can
     eyeball-verify (we learned: never trust the hash blindly).

Method (same rigor as the verified leakage script):
  - exact decoded-pixel MD5  -> true identical images
  - strict dHash (Hamming <= 3) -> resized/recompressed copies, rejects
    merely-similar different wounds.

Usage:
  1. PARENT is already set.
  2. pip install pillow numpy
  3. python overlap_matrix.py
  4. Read the matrix. For any non-zero pair, open the saved samples and
     confirm REAL duplicate vs FALSE positive. Paste the matrix back.

Note: this compares whole datasets (all subfolders pooled per dataset),
ignoring mask/label folders.
"""

import os
import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image

# ============================ CONFIG ============================
PARENT = r"C:\ai\gradutionProject\trainmodel\dataset"

# Each dataset = top-level folder under PARENT. The script maps a top-level
# folder name to a short label. Adjust labels if your folder names differ.
DATASET_DIRS = {
    "AZH":         "azh_wound_care_center_dataset_patches",
    "CO2Wounds":   "CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients",
    "combination_5": "combination_5",
    "data_wound_seg": "data_wound_seg",
    "DFUC2022":    "DFUC2022_train_release",
    "FUSeg":       "Foot Ulcer Segmentation Challenge",
    "Medetec":     "Medetec_foot_ulcer_224",
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DHASH_THRESHOLD = 3          # strict
SAMPLES_PER_PAIR = 6         # how many example overlap images to save per pair
OUT_DIR = "overlap_pairs"
# ===============================================================


def find_images(root):
    out = []
    if not os.path.exists(root):
        print(f"  [WARNING] missing: {root}")
        return out
    for dirpath, _, filenames in os.walk(root):
        low = dirpath.lower()
        if any(k in low for k in ("mask", "label", "annotation", "prediction")):
            continue
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dirpath, f))
    return out


def exact_hash(path):
    try:
        with Image.open(path) as im:
            return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()
    except Exception:
        return None


def dhash(path, size=8):
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.int16)
            return (arr[:, 1:] > arr[:, :-1]).flatten()
    except Exception:
        return None


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def save_pair(p_a, p_b, folder, idx):
    try:
        os.makedirs(folder, exist_ok=True)
        a = Image.open(p_a).convert("RGB").resize((256, 256))
        b = Image.open(p_b).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (522, 256), (255, 255, 255))
        canvas.paste(a, (0, 0)); canvas.paste(b, (266, 0))
        canvas.save(os.path.join(folder, f"pair_{idx:02d}.png"))
    except Exception:
        pass


def main():
    print("=" * 64)
    print("CROSS-DATASET OVERLAP MATRIX (7 datasets)")
    print("=" * 64)

    # Load + fingerprint each dataset once
    data = {}
    for label, sub in DATASET_DIRS.items():
        paths = find_images(os.path.join(PARENT, sub))
        print(f"  {label:14s}: {len(paths)} images")
        exact = {}
        dh = []
        for p in paths:
            h = exact_hash(p)
            if h:
                exact.setdefault(h, p)
            d = dhash(p)
            if d is not None:
                dh.append((p, d))
        data[label] = {"paths": paths, "exact": exact, "dh": dh}
    print("-" * 64)

    labels = list(DATASET_DIRS.keys())
    matrix = {a: {b: 0 for b in labels} for a in labels}

    for a, b in combinations(labels, 2):
        A, B = data[a], data[b]
        if not A["paths"] or not B["paths"]:
            continue
        hits = []
        seen = set()

        # exact
        for h, pa in A["exact"].items():
            if h in B["exact"]:
                key = pa
                if key not in seen:
                    seen.add(key)
                    hits.append((pa, B["exact"][h]))

        # strict dHash
        for pa, da in A["dh"]:
            if pa in seen:
                continue
            for pb, db in B["dh"]:
                if hamming(da, db) <= DHASH_THRESHOLD:
                    seen.add(pa)
                    hits.append((pa, pb))
                    break

        n = len(hits)
        matrix[a][b] = matrix[b][a] = n
        if n:
            folder = os.path.join(OUT_DIR, f"{a}__vs__{b}")
            for i, (pa, pb) in enumerate(hits[:SAMPLES_PER_PAIR]):
                save_pair(pa, pb, folder, i)
            print(f"  {a} <-> {b}: {n} overlap  (samples -> {folder})")

    # print matrix
    print("\n" + "=" * 64)
    print("OVERLAP MATRIX (image count shared between datasets)")
    print("=" * 64)
    head = "            " + "".join(f"{l[:9]:>11s}" for l in labels)
    print(head)
    for a in labels:
        row = f"{a[:11]:11s} "
        for b in labels:
            row += f"{('-' if a==b else matrix[a][b]):>11}"
        print(row)
    print("-" * 64)
    print("Read it: any non-zero pair shares images. Open ./overlap_pairs/")
    print("to confirm REAL duplicates vs FALSE positives, then tell me which")
    print("datasets overlap so we pick a clean train/test split.")
    print("=" * 64)


if __name__ == "__main__":
    main()