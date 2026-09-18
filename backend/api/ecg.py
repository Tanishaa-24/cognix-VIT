"""ECG arrhythmia classification endpoint."""

import io, os, sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from fastapi import APIRouter, UploadFile, File, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ecg_cnn import ECG1DCNN
from utils.ecg_utils import preprocess_signal, compute_saliency

router = APIRouter(prefix="/ecg", tags=["ECG"])

# Loaded at startup by main.py
_ecg_model   = None
_idx_to_name = None
_device      = None


def load_ecg_model(path: str, device: str):
    global _ecg_model, _idx_to_name, _device
    # Always load weights to CPU first to avoid MPS segfault, then move model
    ckpt         = torch.load(path, map_location="cpu", weights_only=False)
    _idx_to_name = ckpt["idx_to_name"]
    num_classes  = ckpt["num_classes"]
    _ecg_model   = ECG1DCNN(num_classes=num_classes)
    _ecg_model.load_state_dict(ckpt["model_state"])
    _ecg_model = _ecg_model.to(device)
    _ecg_model.eval()
    _device = device
    print(f"[ECG] Model loaded — {num_classes} classes on {device}")


@router.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    if _ecg_model is None:
        raise HTTPException(503, "ECG model not loaded")

    raw_bytes = await file.read()
    ext       = file.filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "mat":
            buf = io.BytesIO(raw_bytes)
            mat = sio.loadmat(buf)
            raw = mat["val"]
        elif ext in ("csv", "txt"):
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            text = arr.tobytes().decode("utf-8", errors="ignore")
            vals = [float(v) for v in text.replace(",", "\n").split() if v.strip()]
            raw  = np.array(vals).reshape(1, -1)
        else:
            raise HTTPException(400, f"Unsupported format: {ext}. Send .mat or .csv")
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    sig_np = preprocess_signal(raw)  # (1, 3600) float32
    x      = torch.tensor(sig_np, dtype=torch.float32).to(_device)  # (1, 3600)

    with torch.no_grad():
        logits = _ecg_model(x.unsqueeze(0))               # (1, C)
        probs  = F.softmax(logits, dim=-1).squeeze(0)
        top_idx = int(probs.argmax())

    predicted_class = _idx_to_name[top_idx]
    confidence      = float(probs[top_idx])

    # Saliency map (gradient-based)
    saliency = compute_saliency(_ecg_model, x, top_idx, _device)

    # Top-5 predictions
    top5 = sorted(
        [{"class": _idx_to_name[i], "probability": round(float(probs[i]), 4)}
         for i in range(len(probs))],
        key=lambda d: d["probability"], reverse=True
    )[:5]

    # ECG waveform for display (downsampled to 360 pts for frontend)
    waveform = sig_np[0][::10].tolist()  # 3600 → 360 pts
    sal_down = saliency[::10].tolist()

    return {
        "predicted_class": predicted_class,
        "confidence":       round(confidence, 4),
        "top5":             top5,
        "waveform":         waveform,
        "saliency":         sal_down,
        "cascade_risk_weight": ECG1DCNN.CASCADE_RISK.get(predicted_class, 0),
        "alert_level": (
            "critical" if ECG1DCNN.CASCADE_RISK.get(predicted_class, 0) == 3 else
            "high"     if ECG1DCNN.CASCADE_RISK.get(predicted_class, 0) == 2 else
            "moderate" if ECG1DCNN.CASCADE_RISK.get(predicted_class, 0) == 1 else
            "normal"
        ),
    }
