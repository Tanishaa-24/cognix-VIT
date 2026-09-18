"""
Train 17-class ECG arrhythmia 1D CNN on MIT-BIH .mat segments.
Saves: backend/saved_models/ecg_model.pth
"""

import os, sys, json, glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ecg_cnn import ECG1DCNN
from utils.ecg_utils import preprocess_signal

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data/ecg/ecg signals/MLII")
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "../saved_models/ecg_model.pth")
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
DEVICE     = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Class folder name → canonical label
CLASS_MAP = {
    "1 NSR": "NSR",   "2 APB": "APB",   "3 AFL": "AFL",
    "4 AFIB": "AFIB", "5 SVTA": "SVTA", "6 WPW": "WPW",
    "7 PVC": "PVC",   "8 Bigeminy": "Bigeminy", "9 Trigeminy": "Trigeminy",
    "10 VT": "VT",    "11 IVR": "IVR",  "12 VFL": "VFL",
    "13 Fusion": "Fusion", "14 LBBBB": "LBBBB", "15 RBBBB": "RBBBB",
    "16 SDHB": "SDHB", "17 PR": "PR",
}


# ── Dataset ─────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mat  = sio.loadmat(self.paths[idx])
        raw  = mat["val"]
        sig  = preprocess_signal(raw)
        return torch.tensor(sig, dtype=torch.float32), self.labels[idx]


def load_dataset():
    paths, labels, label_names = [], [], []
    class_dirs = sorted(os.listdir(DATA_DIR))
    name_to_idx = {}
    idx = 0
    for cd in class_dirs:
        canonical = CLASS_MAP.get(cd)
        if canonical is None:
            continue
        if canonical not in name_to_idx:
            name_to_idx[canonical] = idx
            idx += 1
        class_idx = name_to_idx[canonical]
        for fp in glob.glob(os.path.join(DATA_DIR, cd, "*.mat")):
            paths.append(fp)
            labels.append(class_idx)
            label_names.append(canonical)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    return paths, labels, idx_to_name


# ── Training ────────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    paths, labels, idx_to_name = load_dataset()
    num_classes = len(idx_to_name)
    print(f"Total samples: {len(paths)}, Classes: {num_classes}")
    for i, name in sorted(idx_to_name.items()):
        count = labels.count(i)
        print(f"  [{i:2d}] {name:<12} {count} samples")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.25, stratify=labels, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    # Weighted sampler to handle class imbalance
    counts    = np.bincount(y_tr, minlength=num_classes).astype(float)
    weights   = 1.0 / np.maximum(counts, 1)
    s_weights = torch.tensor([weights[y] for y in y_tr], dtype=torch.float)
    sampler   = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)

    tr_loader  = DataLoader(ECGDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_loader  = DataLoader(ECGDataset(X_te,  y_te),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = ECG1DCNN(num_classes=num_classes).to(DEVICE)
    class_wts = torch.tensor(weights / weights.sum() * num_classes, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(DEVICE))
                preds.extend(out.argmax(1).cpu().tolist())
                trues.extend(yb.tolist())
        val_acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss={total_loss/len(tr_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "idx_to_name": idx_to_name,
                "num_classes": num_classes,
            }, SAVE_PATH)

    # Final test evaluation
    ckpt  = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            out = model(xb.to(DEVICE))
            preds.extend(out.argmax(1).cpu().tolist())
            trues.extend(yb.tolist())
    names = [idx_to_name[i] for i in range(num_classes)]
    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy: {accuracy_score(trues, preds):.4f}")
    print(classification_report(trues, preds, target_names=names, zero_division=0))
    print(f"\nModel saved → {SAVE_PATH}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    train()
