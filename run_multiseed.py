"""
run_multiseed.py
----------------
Enhancement #1: statistical reliability via multiple seeds.

Trains each of the 3 models (U-Net, DeepLabV3+, SegFormer-B2) with 3 random
seeds, then reports mean +/- std of in-domain (FUSeg) and cross-dataset
(DFUC2022) Dice/IoU. This turns single-run numbers into credible, publishable
results (e.g. 0.556 +/- 0.012).

It REUSES the exact same data pipeline and protocol as your single-run scripts,
so results are directly comparable.

Output:
  - prints a paper-ready mean+/-std table
  - saves all raw numbers to multiseed_results.csv (so you never lose them)

Runtime: 3 models x 3 seeds = 9 trainings (~3 hours on your GPU). It runs
unattended -- start it and walk away. You can lower EPOCHS or SEEDS to test
the script quickly first (see QUICK TEST note below).

Run:
  python run_multiseed.py
"""

import os
import csv
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================ CONFIG ============================
BASE = r"C:\ai\gradutionProject\trainmodel\dataset"

TRAIN_SOURCES = [
    (rf"{BASE}\Foot Ulcer Segmentation Challenge\train\images",
     rf"{BASE}\Foot Ulcer Segmentation Challenge\train\labels"),
    (rf"{BASE}\azh_wound_care_center_dataset_patches\train\images",
     rf"{BASE}\azh_wound_care_center_dataset_patches\train\labels"),
]
VALID_SOURCES = [
    (rf"{BASE}\Foot Ulcer Segmentation Challenge\validation\images",
     rf"{BASE}\Foot Ulcer Segmentation Challenge\validation\labels"),
]
TEST_SOURCES = [
    (rf"{BASE}\DFUC2022_train_release\DFUC2022_train_images",
     rf"{BASE}\DFUC2022_train_release\DFUC2022_train_masks"),
]

SEEDS   = [42, 123, 2024]          # 3 seeds. QUICK TEST: set to [42] + EPOCHS=2
IMG_SIZE = 256
BATCH    = 8
EPOCHS   = 30
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
CSV_OUT  = "multiseed_results.csv"

# model registry: name -> (builder, lr, optimizer)
def build_unet():       return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
def build_deeplab():    return smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
def build_segformer():  return smp.Segformer("mit_b2", encoder_weights="imagenet", in_channels=3, classes=1)

MODELS = {
    "U-Net (ResNet34)":      (build_unet,      1e-3, "adam"),
    "DeepLabV3+ (ResNet34)": (build_deeplab,   1e-3, "adam"),
    "SegFormer-B2":          (build_segformer, 6e-5, "adamw"),
}
# ===============================================================


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for e in IMG_EXTS:
        out += glob.glob(os.path.join(folder, f"*{e}"))
        out += glob.glob(os.path.join(folder, f"*{e.upper()}"))
    return sorted(out)


def mask_for(ip, mdir):
    stem = os.path.splitext(os.path.basename(ip))[0]
    for e in IMG_EXTS:
        c = os.path.join(mdir, stem + e)
        if os.path.exists(c):
            return c
    return None


def build_pairs(sources):
    pairs = []
    for idir, mdir in sources:
        for ip in list_images(idir):
            mp = mask_for(ip, mdir)
            if mp:
                pairs.append((ip, mp))
    return pairs


class WoundDS(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs = pairs
        if train:
            self.tf = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-20, 20), p=0.4),
                A.Normalize(), ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = self.tf(image=img, mask=msk)
        return out["image"], out["mask"].unsqueeze(0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    d, j, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        inter = (p*y).sum((1,2,3)); ps = p.sum((1,2,3)); ys = y.sum((1,2,3))
        d += ((2*inter+1e-6)/(ps+ys+1e-6)).sum().item()
        j += ((inter+1e-6)/(ps+ys-inter+1e-6)).sum().item()
        n += x.size(0)
    return d/n, j/n


def train_once(name, builder, lr, optim_name, seed, train_dl, valid_dl, test_dl):
    set_seed(seed)
    model = builder().to(DEVICE)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    if optim_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state = 0.0, None
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            (loss_fn(logits, y) + bce(logits, y)).backward()
            opt.step()
        vdice, _ = evaluate(model, valid_dl)
        if vdice > best_val:
            best_val = vdice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # restore best and test cross-dataset
    model.load_state_dict(best_state)
    tdice, tiou = evaluate(model, test_dl)
    print(f"    [{name} | seed {seed}] val Dice {best_val:.4f} | DFUC Dice {tdice:.4f} IoU {tiou:.4f}")
    return best_val, tdice, tiou


def main():
    train_pairs = build_pairs(TRAIN_SOURCES)
    valid_pairs = build_pairs(VALID_SOURCES)
    test_pairs  = build_pairs(TEST_SOURCES)
    print(f"Train {len(train_pairs)} | valid {len(valid_pairs)} | test {len(test_pairs)} | device {DEVICE}")
    if not (train_pairs and valid_pairs and test_pairs):
        print("Empty split -- check paths."); return

    train_dl = DataLoader(WoundDS(train_pairs, True),  batch_size=BATCH, shuffle=True,  num_workers=0)
    valid_dl = DataLoader(WoundDS(valid_pairs, False), batch_size=BATCH, shuffle=False, num_workers=0)
    test_dl  = DataLoader(WoundDS(test_pairs,  False), batch_size=BATCH, shuffle=False, num_workers=0)

    rows = []  # name, seed, val_dice, dfuc_dice, dfuc_iou
    for name, (builder, lr, optim_name) in MODELS.items():
        print(f"\n=== {name} ===")
        for seed in SEEDS:
            v, td, ti = train_once(name, builder, lr, optim_name, seed,
                                    train_dl, valid_dl, test_dl)
            rows.append([name, seed, v, td, ti])

    # save raw
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "val_dice", "dfuc_dice", "dfuc_iou"])
        w.writerows(rows)

    # aggregate mean+/-std
    print("\n" + "=" * 70)
    print("PAPER-READY RESULTS (mean +/- std over seeds)")
    print("=" * 70)
    print(f"{'Model':22s} {'FUSeg Dice':>16s} {'DFUC2022 Dice':>18s} {'DFUC2022 IoU':>16s}")
    for name in MODELS:
        vs = np.array([r[2] for r in rows if r[0] == name])
        ds = np.array([r[3] for r in rows if r[0] == name])
        js = np.array([r[4] for r in rows if r[0] == name])
        print(f"{name:22s} {vs.mean():.3f}+/-{vs.std():.3f}   "
              f"{ds.mean():.3f}+/-{ds.std():.3f}   {js.mean():.3f}+/-{js.std():.3f}")
    print("=" * 70)
    print(f"Raw numbers saved to {CSV_OUT}")


if __name__ == "__main__":
    main()