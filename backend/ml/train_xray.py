"""
Chest X-ray pneumonia classifier — training script (from scratch, no pretrained weights).

Dataset layout expected:
  <data_root>/train/NORMAL/*.jpeg        → class 0 (Normal)
  <data_root>/test/PNEUMONIA/*bacteria*  → class 1 (Bacterial)
  <data_root>/test/PNEUMONIA/*virus*     → class 2 (Viral)

Usage:
  python3 -m backend.ml.train_xray
"""

import os, sys, random, pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate truncated JPEGs in the dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.models.xray_cnn import XRayCNN, CLASS_NAMES
from backend.utils.xray_utils import TRAIN_TRANSFORM, INFERENCE_TRANSFORM

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = pathlib.Path("/tmp/xray_inspect/chest xray")
SAVE_PATH = pathlib.Path(__file__).parent.parent / "saved_models" / "xray_model.pth"
EPOCHS    = 30
BATCH     = 32
LR        = 3e-4
SEED      = 42
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────
def collect_samples(data_root: pathlib.Path):
    samples = []
    for p in (data_root / "train" / "NORMAL").glob("*.jpeg"):
        samples.append((p, 0))
    for p in (data_root / "test" / "PNEUMONIA").glob("*.jpeg"):
        name = p.name.lower()
        if "bacteria" in name:
            samples.append((p, 1))
        elif "virus" in name:
            samples.append((p, 2))
    return samples


def stratified_split(samples, seed=42, val_frac=0.12, test_frac=0.08):
    rng = random.Random(seed)
    by_class = {0: [], 1: [], 2: []}
    for path, label in samples:
        by_class[label].append((path, label))
    train, val, test = [], [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_val  = max(2, int(n * val_frac))
        n_test = max(2, int(n * test_frac))
        test  += cls_samples[:n_test]
        val   += cls_samples[n_test:n_test + n_val]
        train += cls_samples[n_test + n_val:]
    return train, val, test


class XRayDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


# ── Training ──────────────────────────────────────────────────────────────────
def compute_class_weights(samples):
    counts = [sum(1 for _, l in samples if l == c) for c in range(3)]
    total  = sum(counts)
    return torch.tensor([total / (3.0 * c) for c in counts], dtype=torch.float32)


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"[XRay] Device: {DEVICE}")
    samples = collect_samples(DATA_ROOT)
    counts  = [sum(1 for _, l in samples if l == c) for c in range(3)]
    print(f"[XRay] Total: {len(samples)} | Normal={counts[0]}, Bacterial={counts[1]}, Viral={counts[2]}")

    train_s, val_s, test_s = stratified_split(samples, seed=SEED)
    print(f"[XRay] Split → train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    class_weights  = compute_class_weights(train_s)
    sample_weights = [float(class_weights[l]) for _, l in train_s]
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(train_s), replacement=True)

    train_ds = XRayDataset(train_s, TRAIN_TRANSFORM)
    val_ds   = XRayDataset(val_s,   INFERENCE_TRANSFORM)
    test_ds  = XRayDataset(test_s,  INFERENCE_TRANSFORM)

    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,   num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,   num_workers=0)

    model     = XRayCNN(num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                vc += (model(imgs).argmax(1) == labels).sum().item()
                vt += len(labels)
        val_acc = vc / vt

        print(f"  Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/total:.4f} | "
              f"train={correct/total:.3f} | val={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_state)
    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            tc += (model(imgs).argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"\n[XRay] Best val_acc={best_val_acc:.3f}  test_acc={test_acc:.3f}")

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":   best_state,
        "class_names":   CLASS_NAMES,
        "num_classes":   3,
        "val_accuracy":  round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
    }, SAVE_PATH)
    print(f"[XRay] Model saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
