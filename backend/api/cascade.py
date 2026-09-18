"""Cascade chain prediction, SHAP explanation, and counterfactual endpoints."""

import os, sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cascade_utils import (
    get_cascade_timeline, get_shap_explanation, simulate_counterfactual
)

router = APIRouter(prefix="/cascade", tags=["Cascade"])

_cascade_model = None
_cardiac_model = None
_stroke_model  = None


def load_cascade_models(save_dir: str):
    global _cascade_model, _cardiac_model, _stroke_model
    import joblib
    _cascade_model = joblib.load(os.path.join(save_dir, "cascade_model.joblib"))
    _cardiac_model = joblib.load(os.path.join(save_dir, "cardiac_model.joblib"))
    _stroke_model  = joblib.load(os.path.join(save_dir, "stroke_model.joblib"))
    print("[Cascade] All cascade models loaded (joblib sklearn)")


# ── Request schemas ──────────────────────────────────────────────────────────
class ClinicalData(BaseModel):
    # Demographics
    age:               float
    gender:            int    # 0=female, 1=male
    ever_married:      int    # 0=no, 1=yes
    work_type:         int    # 0-4 encoded
    Residence_type:    int    # 0=rural, 1=urban
    # Clinical measurements
    hypertension:      int    # 0/1
    heart_disease:     int    # 0/1
    avg_glucose_level: float
    bmi:               float
    smoking_status:    int    # 0=never, 1=formerly, 2=smokes
    # ECG results (from ECG module — injected into cascade)
    arrhythmia_name:   str    = "NSR"
    arrhythmia_confidence: float = 1.0


class InterventionRequest(BaseModel):
    clinical:     ClinicalData
    intervention: str   # "anticoagulation" | "beta_blocker" | "lifestyle" | "all"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _clinical_dict(data: ClinicalData) -> dict:
    return {
        "gender":            data.gender,
        "age":               data.age,
        "hypertension":      data.hypertension,
        "heart_disease":     data.heart_disease,
        "ever_married":      data.ever_married,
        "work_type":         data.work_type,
        "Residence_type":    data.Residence_type,
        "avg_glucose_level": data.avg_glucose_level,
        "bmi":               data.bmi,
        "smoking_status":    data.smoking_status,
    }


def _cardiac_risk(data: ClinicalData) -> dict:
    """Run cardiac XGBoost separately for cardiac risk score."""
    import pandas as pd
    feats = _cardiac_model["features"]
    # Map available fields; fill missing with 0
    cardiac_map = {
        "age":      data.age,
        "sex":      data.gender,
        "cp":       1 if data.hypertension else 0,
        "trestbps": min(data.avg_glucose_level * 0.5, 200),  # proxy
        "chol":     200.0,  # default neutral
        "fbs":      1 if data.avg_glucose_level > 120 else 0,
        "restecg":  0,
        "thalach":  max(60, 220 - data.age),  # estimated max HR
        "exang":    int(data.heart_disease),
        "oldpeak":  0.5 if data.heart_disease else 0.0,
        "slope":    1,
        "ca":       1 if data.heart_disease else 0,
        "thal":     2,
    }
    row = pd.DataFrame([[cardiac_map.get(f, 0) for f in feats]], columns=feats)
    prob = float(_cardiac_model["model"].predict_proba(row)[0, 1])
    return {"cardiac_risk": round(prob * 100, 2)}


# ── Endpoints ────────────────────────────────────────────────────────────────
@router.post("/predict")
async def cascade_predict(data: ClinicalData):
    if _cascade_model is None:
        raise HTTPException(503, "Cascade models not loaded")

    clinical = _clinical_dict(data)
    timeline = get_cascade_timeline(
        _cascade_model, clinical, data.arrhythmia_name, data.arrhythmia_confidence
    )
    shap_vals = get_shap_explanation(
        _cascade_model, clinical, data.arrhythmia_name, data.arrhythmia_confidence
    )
    cardiac   = _cardiac_risk(data)

    # Cascade stage (1-4) based on combined signals
    base         = timeline["base_probability"]
    cardiac_risk = cardiac["cardiac_risk"]
    from utils.cascade_utils import STROKE_ELEVATING, CARDIAC_ELEVATING

    combined_risk = (base + cardiac_risk * 0.3) / 1.3
    stage = 1 if combined_risk < 20 else 2 if combined_risk < 40 else 3 if combined_risk < 65 else 4

    if data.arrhythmia_name in CARDIAC_ELEVATING:
        action = "⚠️ Critical arrhythmia detected. Immediate cardiology referral required. Consider antiarrhythmic therapy."
    elif data.arrhythmia_name in STROKE_ELEVATING:
        action = "🩺 Stroke-elevating arrhythmia detected. Evaluate for anticoagulation (CHA₂DS₂-VASc ≥ 2). Cardiology consult within 24 hours."
    elif combined_risk > 50:
        action = "🔴 High cascade risk. Initiate risk factor modification, medication review, and close follow-up within 1 week."
    elif combined_risk > 30:
        action = "🟡 Moderate cascade risk. Review antihypertensive and lipid-lowering therapy. Follow-up within 2 weeks."
    else:
        action = "🟢 Low cascade risk. Continue preventive monitoring. Annual review recommended."

    return {
        "cascade_stage":     stage,
        "timeline":          timeline,
        "shap_explanation":  shap_vals,
        "cardiac_risk":      cardiac["cardiac_risk"],
        "recommended_action": action,
    }


@router.post("/counterfactual")
async def counterfactual(req: InterventionRequest):
    if _cascade_model is None:
        raise HTTPException(503, "Cascade models not loaded")

    clinical = _clinical_dict(req.clinical)
    result   = simulate_counterfactual(
        _cascade_model, clinical,
        req.clinical.arrhythmia_name,
        req.clinical.arrhythmia_confidence,
        req.intervention,
    )
    return result


@router.post("/all-interventions")
async def all_interventions(data: ClinicalData):
    """Run all 4 interventions at once for comparison panel."""
    if _cascade_model is None:
        raise HTTPException(503, "Cascade models not loaded")

    clinical = _clinical_dict(data)
    results  = {}
    for iv in ["anticoagulation", "beta_blocker", "lifestyle", "all"]:
        results[iv] = simulate_counterfactual(
            _cascade_model, clinical,
            data.arrhythmia_name, data.arrhythmia_confidence, iv
        )
    return results
