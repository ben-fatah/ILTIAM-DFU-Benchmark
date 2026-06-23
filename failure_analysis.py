"""
ILTIAM — Enhancement #3: Failure Analysis (v2)
Medetec uses ALL 160 images (train+test merged — zero overlap with FUSeg/AZH training data)
Author: Abderrahmane Zine Benfatah (Thabit) | King Saud University

Run:
    conda activate ai_env
    python "failure_analysis_v2.py"
"""

import os, json, tempfile, shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "DFUC2022": {
        "img_dir":  r"C:\ai\gradutionProject\trainmodel\dataset\DFUC2022_train_release\DFUC2022_train_images",
        "mask_dir": r"C:\ai\gradutionProject\trainmodel\dataset\DFUC2022_train_release\DFUC2022_train_masks",
        "medetec":  False,
        "multi_src": False,
    },
    "Medetec": {
        # Both train and test — all 160 images, zero overlap with FUSeg/AZH
        "img_dirs":  [
            r"C:\ai\gradutionProject\trainmodel\dataset\Medetec_foot_ulcer_224\train\images",
            r"C:\ai\gradutionProject\trainmodel\dataset\Medetec_foot_ulcer_224\test\images",
        ],
        "mask_dirs": [
            r"C:\ai\gradutionProject\trainmodel\dataset\Medetec_foot_ulcer_224\train\labels",
            r"C:\ai\gradutionProject\trainmodel\dataset\Medetec_foot_ulcer_224\test\labels",
        ],
        "medetec":   True,
        "multi_src": True,
    },
    "CHECKPOINTS": {
        "U-Net":        r"C:\ai\gradutionProject\checkpoints\U-Net_ResNet34_seed42.pth",
        "DeepLabV3+":   r"C:\ai\gradutionProject\checkpoints\DeepLabV3plus_ResNet34_seed42.pth",
        "SegFormer-B2": r"C:\ai\gradutionProject\checkpoints\SegFormer-B2_seed42.pth",
    },
    "OUTPUT_DIR":  r"C:\ai\gradutionProject\failure_analysis",
    "IMG_SIZE":    256,
    "DEVICE":      "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────
# Dataset  (supports single or multi-source)
# ─────────────────────────────────────────────
class WoundDataset(Dataset):
    def __init__(self, img_dirs, mask_dirs, img_size=256, medetec=False):
        """
        img_dirs / mask_dirs: list of Path-like strings.
        Pairs are matched by stem name within each (img_dir, mask_dir) pair.
        """
        IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        self.pairs   = []   # list of (img_path, mask_path)
        self.medetec = medetec

        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            img_dir  = Path(img_dir)
            mask_dir = Path(mask_dir)
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXT:
                    continue
                mask_path = mask_dir / (img_path.stem + ".png")
                if not mask_path.exists():
                    mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))
                else:
                    print(f"  [WARN] No mask found for {img_path.name}")

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.mask_tf = transforms.Resize(
            (img_size, img_size), interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_t   = self.img_tf(img)
        mask_np = np.array(self.mask_tf(mask), dtype=np.float32)
        mask_np = (mask_np > 0).astype(np.float32) if self.medetec \
                  else (mask_np > 127).astype(np.float32)

        return img_t, torch.tensor(mask_np).unsqueeze(0), str(img_path)


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────
def load_model(name, ckpt_path, device):
    if name == "U-Net":
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=1)
    elif name == "DeepLabV3+":
        model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None,
                                   in_channels=3, classes=1)
    elif name == "SegFormer-B2":
        model = smp.Segformer(encoder_name="mit_b2", encoder_weights=None,
                               in_channels=3, classes=1)
    else:
        raise ValueError(name)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]; break
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    ✓ {name}")
    return model


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred, target, eps=1e-6):
    p = (torch.sigmoid(pred) > 0.5).float()
    return ((2*(p*target).sum() + eps) / (p.sum() + target.sum() + eps)).item()

def iou_score(pred, target, eps=1e-6):
    p = (torch.sigmoid(pred) > 0.5).float()
    inter = (p*target).sum()
    return ((inter + eps) / (p.sum() + target.sum() - inter + eps)).item()


# ─────────────────────────────────────────────
# Per-image scoring
# ─────────────────────────────────────────────
@torch.no_grad()
def score_dataset(model, dataset, device):
    results = []
    for img_t, mask_t, path in dataset:
        pred   = model(img_t.unsqueeze(0).to(device))
        mask_d = mask_t.unsqueeze(0).to(device)
        results.append({
            "path": path,
            "dice": round(dice_score(pred, mask_d), 4),
            "iou":  round(iou_score(pred,  mask_d), 4),
        })
    return results


# ─────────────────────────────────────────────
# Histogram
# ─────────────────────────────────────────────
def plot_histogram(all_results, dataset_name, output_path):
    colors = {"U-Net":"#4C72B0","DeepLabV3+":"#DD8452","SegFormer-B2":"#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Per-Image Dice Distribution — {dataset_name}",
                 fontsize=13, fontweight="bold")
    for ax, (name, results) in zip(axes, all_results.items()):
        dices  = [r["dice"] for r in results]
        mean_d = np.mean(dices)
        n_fail = sum(1 for d in dices if d < 0.4)
        ax.hist(dices, bins=25, color=colors.get(name,"gray"),
                edgecolor="white", alpha=0.85)
        ax.axvline(mean_d, color="red",    ls="--", lw=1.5,
                   label=f"Mean = {mean_d:.3f}")
        ax.axvline(0.4,    color="orange", ls=":",  lw=1.5,
                   label=f"Fail < 0.4  (n={n_fail})")
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Dice Score"); ax.set_ylabel("Image Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved histogram : {Path(output_path).name}")


# ─────────────────────────────────────────────
# Qualitative gallery
# ─────────────────────────────────────────────
MEAN = np.array([0.485,0.456,0.406])
STD  = np.array([0.229,0.224,0.225])

def denorm(t):
    return np.clip(t.permute(1,2,0).numpy() * STD + MEAN, 0, 1)

def pick_cases(scores):
    good_i    = int(np.argmax(scores))
    fail_i    = int(np.argmin(scores))
    mid       = np.median(scores)
    partial_i = int(np.argmin(np.abs(scores - mid)))
    # avoid collision
    used = {good_i, fail_i}
    if partial_i in used:
        candidates = [i for i in range(len(scores)) if i not in used]
        partial_i  = candidates[len(candidates)//2] if candidates else partial_i
    return good_i, partial_i, fail_i

def make_gallery(models_dict, dataset, device, output_path, dataset_name):
    # Score with anchor model (SegFormer — best generalizer)
    anchor = models_dict["SegFormer-B2"]
    scores, items = [], []
    with torch.no_grad():
        for img_t, mask_t, path in dataset:
            pred   = anchor(img_t.unsqueeze(0).to(device))
            scores.append(dice_score(pred, mask_t.unsqueeze(0).to(device)))
            items.append((img_t, mask_t))

    scores      = np.array(scores)
    good_i, partial_i, fail_i = pick_cases(scores)
    case_idx    = [good_i, partial_i, fail_i]
    case_labels = ["GOOD", "PARTIAL", "FAIL"]
    model_names = list(models_dict.keys())

    n_rows, n_models, sub = 3, len(model_names), 3
    fig, axes = plt.subplots(n_rows, n_models * sub,
                             figsize=(sub * n_models * 3.2, n_rows * 3.2))
    fig.suptitle(
        f"{dataset_name} — Segmentation Gallery  (GOOD / PARTIAL / FAIL)",
        fontsize=13, fontweight="bold", y=1.01)

    for row, (idx, label) in enumerate(zip(case_idx, case_labels)):
        img_t, mask_t = items[idx]
        img_np = denorm(img_t)
        gt_np  = mask_t.squeeze().numpy()

        for col_m, (mname, model) in enumerate(models_dict.items()):
            with torch.no_grad():
                pred   = model(img_t.unsqueeze(0).to(device))
                pred_np = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()
            d = dice_score(pred, mask_t.unsqueeze(0).to(device))

            b  = col_m * sub
            ax_i, ax_g, ax_p = axes[row,b], axes[row,b+1], axes[row,b+2]
            ax_i.imshow(img_np);           ax_i.axis("off")
            ax_g.imshow(gt_np,  cmap="gray"); ax_g.axis("off")
            ax_p.imshow(pred_np,cmap="gray"); ax_p.axis("off")

            if row == 0:
                ax_i.set_title(f"{mname}\nInput",  fontsize=8, fontweight="bold", pad=4)
                ax_g.set_title("GT Mask",           fontsize=8, pad=4)
                ax_p.set_title("Prediction",        fontsize=8, pad=4)

            clr = "#2ca02c" if d>=0.7 else ("#ff7f0e" if d>=0.4 else "#d62728")
            ax_p.set_xlabel(f"Dice = {d:.3f}", fontsize=8, color=clr, labelpad=2)

        axes[row,0].set_ylabel(
            f"{label}\n(Dice ≈ {scores[idx]:.2f})",
            fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved gallery   : {Path(output_path).name}")


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
def print_summary(all_results, dataset_name, threshold=0.4):
    print(f"\n{'─'*57}")
    print(f"  Failure Summary — {dataset_name}  (threshold = {threshold})")
    print(f"{'─'*57}")
    print(f"  {'Model':<15} {'n':>5} {'Mean Dice':>10} {'Mean IoU':>10} {'Failed':>8} {'Fail%':>7}")
    print(f"  {'─'*55}")
    for name, results in all_results.items():
        dices  = [r["dice"] for r in results]
        ious   = [r["iou"]  for r in results]
        n      = len(results)
        n_fail = sum(1 for d in dices if d < threshold)
        print(f"  {name:<15} {n:>5} {np.mean(dices):>10.4f} {np.mean(ious):>10.4f} "
              f"{n_fail:>8} {100*n_fail/n:>6.1f}%")
    print(f"{'─'*57}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = cfg["DEVICE"]
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    print(f"\nDevice : {device}")
    print(f"Output : {cfg['OUTPUT_DIR']}\n")

    print("Loading checkpoints...")
    models = {name: load_model(name, ckpt, device)
              for name, ckpt in cfg["CHECKPOINTS"].items()}

    datasets_to_run = {
        "DFUC2022": (
            [cfg["DFUC2022"]["img_dir"]],
            [cfg["DFUC2022"]["mask_dir"]],
            cfg["DFUC2022"]["medetec"],
        ),
        "Medetec": (
            cfg["Medetec"]["img_dirs"],
            cfg["Medetec"]["mask_dirs"],
            cfg["Medetec"]["medetec"],
        ),
    }

    for ds_name, (img_dirs, mask_dirs, is_medetec) in datasets_to_run.items():
        print(f"\n{'═'*57}")
        print(f"  Dataset : {ds_name}")
        print(f"{'═'*57}")

        dataset = WoundDataset(img_dirs, mask_dirs, cfg["IMG_SIZE"], is_medetec)
        print(f"  Total images : {len(dataset)}")

        all_results = {}
        for name, model in models.items():
            print(f"  Scoring {name}...", end=" ", flush=True)
            results = score_dataset(model, dataset, device)
            all_results[name] = results
            print(f"mean Dice = {np.mean([r['dice'] for r in results]):.4f}")

            out = os.path.join(cfg["OUTPUT_DIR"],
                               f"{ds_name}_{name.replace('+','plus')}_scores.json")
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

        print_summary(all_results, ds_name)
        plot_histogram(all_results, ds_name,
                       os.path.join(cfg["OUTPUT_DIR"], f"histogram_{ds_name}.png"))
        make_gallery(models, dataset, device,
                     os.path.join(cfg["OUTPUT_DIR"], f"gallery_{ds_name}.png"),
                     ds_name)

    print(f"\n✅  Done.  Files in {cfg['OUTPUT_DIR']}:\n")
    for f in sorted(os.listdir(cfg["OUTPUT_DIR"])):
        sz = os.path.getsize(os.path.join(cfg["OUTPUT_DIR"], f))
        print(f"  {f:<50} {sz/1024:>8.1f} KB")

if __name__ == "__main__":
    main()