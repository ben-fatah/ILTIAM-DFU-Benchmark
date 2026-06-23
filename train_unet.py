"""
train_unet.py
-------------
Your FIRST real result. Trains U-Net (ResNet34) on FUSeg + AZH, then tests
cross-dataset on DFUC2022, and prints Dice + IoU for both.

This is the experiment your whole paper rests on:
    TRAIN: FUSeg + AZH  (one combined source)
    VALID: FUSeg validation split
    TEST : DFUC2022      (independent hospital -> the generalization number)

Design choices kept deliberately SIMPLE so it runs and finishes:
  - segmentation_models_pytorch U-Net, ResNet34 encoder, ImageNet weights
  - 256x256, Dice+BCE loss, Adam, ~30 epochs
  - saves best model by validation Dice to best_unet.pth

STEP 0 runs first and ONLY checks paths + counts. If a path is wrong it
tells you exactly what to fix BEFORE wasting time. Run it once; if STEP 0
prints all green, it proceeds to train.

Install (in ai_env):
  pip install segmentation-models-pytorch albumentations torch torchvision opencv-python

Run:
  python train_unet.py
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================ CONFIG: PATHS ============================
# Each entry: (images_folder, masks_folder). Edit if your subfolder names differ.
BASE = r"C:\ai\gradutionProject\trainmodel\dataset"

TRAIN_SOURCES = [
    # FUSeg train
    (rf"{BASE}\Foot Ulcer Segmentation Challenge\train\images",
     rf"{BASE}\Foot Ulcer Segmentation Challenge\train\labels"),
    # AZH train
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

# ============================ CONFIG: TRAINING ========================
IMG_SIZE   = 256
BATCH      = 8
EPOCHS     = 30
LR         = 1e-3
ENCODER    = "resnet34"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "best_unet.pth"
IMG_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# =====================================================================


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    out = []
    for e in IMG_EXTS:
        out += glob.glob(os.path.join(folder, f"*{e}"))
        out += glob.glob(os.path.join(folder, f"*{e.upper()}"))
    return sorted(out)


def mask_for(image_path, mask_folder):
    """Find the mask matching an image by stem, trying common extensions."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for e in IMG_EXTS:
        cand = os.path.join(mask_folder, stem + e)
        if os.path.exists(cand):
            return cand
    return None


def build_pairs(sources):
    pairs = []
    for img_dir, mask_dir in sources:
        for ip in list_images(img_dir):
            mp = mask_for(ip, mask_dir)
            if mp:
                pairs.append((ip, mp))
    return pairs


def step0_check():
    print("=" * 60)
    print("STEP 0 — PATH CHECK (no training yet)")
    print("=" * 60)
    ok = True
    for name, src in [("TRAIN", TRAIN_SOURCES), ("VALID", VALID_SOURCES), ("TEST", TEST_SOURCES)]:
        total = 0
        for img_dir, mask_dir in src:
            imgs = list_images(img_dir)
            img_ok = os.path.isdir(img_dir)
            msk_ok = os.path.isdir(mask_dir)
            # count how many images have a matching mask
            matched = sum(1 for ip in imgs if mask_for(ip, mask_dir))
            total += matched
            flag = "OK " if (img_ok and msk_ok and matched > 0) else "FAIL"
            if flag == "FAIL":
                ok = False
            print(f"  [{flag}] {name}")
            print(f"        images: {img_dir}  ({len(imgs)} found, dir_exists={img_ok})")
            print(f"        masks : {mask_dir}  (dir_exists={msk_ok})")
            print(f"        image+mask pairs matched: {matched}")
        print(f"   -> {name} total usable pairs: {total}")
    print("-" * 60)
    print(f"Device: {DEVICE}")
    if not ok:
        print("\nSome paths failed. Fix the folder names in CONFIG (likely 'labels'")
        print("vs 'masks', or a different subfolder name) and re-run. Training skipped.")
    print("=" * 60)
    return ok


class WoundDS(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs = pairs
        if train:
            self.tf = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=20, p=0.4),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        msk = (msk > 127).astype(np.float32)
        out = self.tf(image=img, mask=msk)
        return out["image"], out["mask"].unsqueeze(0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dice_sum, iou_sum, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = torch.sigmoid(model(x))
        p = (p > 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        psum  = p.sum(dim=(1, 2, 3))
        ysum  = y.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (psum + ysum + 1e-6)
        iou  = (inter + 1e-6) / (psum + ysum - inter + 1e-6)
        dice_sum += dice.sum().item()
        iou_sum  += iou.sum().item()
        n += x.size(0)
    return dice_sum / n, iou_sum / n


def main():
    if not step0_check():
        return

    train_pairs = build_pairs(TRAIN_SOURCES)
    valid_pairs = build_pairs(VALID_SOURCES)
    test_pairs  = build_pairs(TEST_SOURCES)
    print(f"\nTraining on {len(train_pairs)} | validating on {len(valid_pairs)} "
          f"| testing on {len(test_pairs)}\n")

    train_dl = DataLoader(WoundDS(train_pairs, True),  batch_size=BATCH, shuffle=True,  num_workers=0)
    valid_dl = DataLoader(WoundDS(valid_pairs, False), batch_size=BATCH, shuffle=False, num_workers=0)
    test_dl  = DataLoader(WoundDS(test_pairs,  False), batch_size=BATCH, shuffle=False, num_workers=0)

    model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(DEVICE)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y) + bce(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
        vdice, viou = evaluate(model, valid_dl)
        print(f"epoch {ep:02d} | loss {running/len(train_dl):.4f} "
              f"| val Dice {vdice:.4f} IoU {viou:.4f}")
        if vdice > best:
            best = vdice
            torch.save(model.state_dict(), SAVE_PATH)

    # cross-dataset test with best model
    model.load_state_dict(torch.load(SAVE_PATH))
    tdice, tiou = evaluate(model, test_dl)
    print("\n" + "=" * 60)
    print("RESULTS — U-Net (ResNet34)")
    print("=" * 60)
    print(f"  Best VALIDATION (FUSeg)        : Dice {best:.4f}")
    print(f"  CROSS-DATASET TEST (DFUC2022)  : Dice {tdice:.4f}  IoU {tiou:.4f}")
    print("  ^ the gap between these two IS your paper's finding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
