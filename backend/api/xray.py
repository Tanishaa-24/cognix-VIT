"""Chest X-ray pneumonia classification endpoint."""

import os, sys
import torch
import torch.nn.functional as F
from fastapi import APIRouter, UploadFile, File, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.xray_cnn import XRayCNN, CLASS_NAMES, ALERT_LEVEL, CLINICAL_NOTE
from utils.xray_utils import preprocess_image

router = APIRouter(prefix="/xray", tags=["XRay"])

_xray_model  = None
_class_names = None
_device      = None


def load_xray_model(path: str, device: str):
    global _xray_model, _class_names, _device
    ckpt         = torch.load(path, map_location="cpu", weights_only=False)
    _class_names = ckpt["class_names"]
    num_classes  = ckpt["num_classes"]
    _xray_model  = XRayCNN(num_classes=num_classes)
    _xray_model.load_state_dict(ckpt["model_state"])
    _xray_model = _xray_model.to(device)
    _xray_model.eval()
    _device = device
    val_acc = ckpt.get("val_accuracy", "n/a")
    print(f"[XRay] Model loaded — {num_classes} classes on {device} (val_acc={val_acc})")


@router.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    if _xray_model is None:
        raise HTTPException(503, "X-ray model not loaded")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png"):
        raise HTTPException(400, f"Unsupported format: {ext}. Send .jpg/.jpeg/.png")

    raw_bytes = await file.read()
    try:
        x = preprocess_image(raw_bytes).to(_device)   # (1, 3, 224, 224)
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    with torch.no_grad():
        logits = _xray_model(x)                        # (1, 3)
        probs  = F.softmax(logits, dim=-1).squeeze(0)
        top_idx = int(probs.argmax())

    predicted_class = _class_names[top_idx]
    confidence      = float(probs[top_idx])

    top3 = sorted(
        [{"class": _class_names[i], "probability": round(float(probs[i]), 4)}
         for i in range(len(probs))],
        key=lambda d: d["probability"], reverse=True
    )

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 4),
        "top3":            top3,
        "alert_level":     ALERT_LEVEL[predicted_class],
        "clinical_note":   CLINICAL_NOTE[predicted_class],
    }
