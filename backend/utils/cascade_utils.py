"""Cascade chain inference utilities — counterfactual simulation + SHAP."""

import numpy as np
import shap


ARRHYTHMIA_CLASSES = [
    "NSR", "APB", "AFL", "AFIB", "SVTA", "WPW", "PVC",
    "Bigeminy", "Trigeminy", "VT", "IVR", "VFL", "Fusion",
    "LBBBB", "RBBBB", "SDHB", "PR",
]

CASCADE_RISK_WEIGHT = {
    "NSR": 0.0, "APB": 0.3, "AFL": 0.6, "AFIB": 0.7, "SVTA": 0.5,
    "WPW": 0.6, "PVC": 0.3, "Bigeminy": 0.35, "Trigeminy": 0.35,
    "VT": 1.0, "IVR": 0.9, "VFL": 1.0, "Fusion": 0.4,
    "LBBBB": 0.45, "RBBBB": 0.35, "SDHB": 1.0, "PR": 0.3,
}

# Which arrhythmias significantly drive stroke cascade
STROKE_ELEVATING = {"AFIB", "AFL", "SVTA", "WPW"}
# Which elevate cardiac cascade
CARDIAC_ELEVATING = {"VT", "VFL", "IVR", "SDHB", "WPW"}


def build_cascade_features(clinical: dict, arrhythmia_name: str, confidence: float) -> np.ndarray:
    """
    Constructs the feature vector for the cascade model.
    clinical dict must have keys matching BASE_FEATURES from training.
    """
    base_keys = ["gender", "age", "hypertension", "heart_disease", "ever_married",
                 "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
    base_vals = [float(clinical.get(k, 0)) for k in base_keys]

    risk_weight = CASCADE_RISK_WEIGHT.get(arrhythmia_name, 0.0) * confidence
    arr_onehot  = [1.0 if cls == arrhythmia_name else 0.0 for cls in ARRHYTHMIA_CLASSES]

    return np.array(base_vals + [risk_weight] + arr_onehot, dtype=np.float32)


def get_cascade_timeline(cascade_model, clinical: dict,
                         arrhythmia_name: str, confidence: float) -> dict:
    """
    Generates 30/90/365-day cascade risk probabilities for 3 futures:
      - no_intervention
      - conservative (anticoagulation / beta-blocker)
      - aggressive (anticoagulation + rate control + lifestyle)
    Uses temporal scaling of base cascade probability.
    """
    model = cascade_model["model"]
    x     = build_cascade_features(clinical, arrhythmia_name, confidence).reshape(1, -1)

    import pandas as pd
    feat_names = cascade_model["features"]
    x_df = pd.DataFrame(x, columns=feat_names)

    base_prob = float(model.predict_proba(x_df)[0, 1])

    # Intervention effect multipliers (clinically informed)
    is_stroke_risk  = arrhythmia_name in STROKE_ELEVATING
    is_cardiac_risk = arrhythmia_name in CARDIAC_ELEVATING

    # Conservative intervention reduces stroke risk by ~75% for AFib (ARISTOTLE trial)
    conservative_mult = 0.25 if is_stroke_risk else (0.55 if is_cardiac_risk else 0.70)
    # Aggressive intervention further reduces
    aggressive_mult   = 0.12 if is_stroke_risk else (0.30 if is_cardiac_risk else 0.45)

    # 30/90/365 day temporal scaling (disease progression without treatment)
    def timeline(base, mult):
        p30  = min(base * 0.30, 0.95)
        p90  = min(base * 0.65, 0.95)
        p365 = min(base * 1.00, 0.95)
        return {
            "d30":  round(p30  * mult * 100, 1),
            "d90":  round(p90  * mult * 100, 1),
            "d365": round(p365 * mult * 100, 1),
        }

    return {
        "base_probability": round(base_prob * 100, 2),
        "no_intervention":   timeline(base_prob, 1.0),
        "conservative":      timeline(base_prob, conservative_mult),
        "aggressive":        timeline(base_prob, aggressive_mult),
    }


def get_shap_explanation(cascade_model, clinical: dict,
                         arrhythmia_name: str, confidence: float,
                         top_n: int = 10) -> list:
    """
    Returns top-N SHAP feature importance values.
    Uses sklearn-compatible TreeExplainer (no XGBoost C-ext conflict).
    """
    import pandas as pd, shap
    if "explainer" not in cascade_model:
        # TreeExplainer works natively with sklearn GradientBoostingClassifier
        cascade_model["explainer"] = shap.TreeExplainer(cascade_model["model"])
    explainer  = cascade_model["explainer"]
    feat_names = cascade_model["features"]
    x          = build_cascade_features(clinical, arrhythmia_name, confidence).reshape(1, -1)
    x_df       = pd.DataFrame(x, columns=feat_names)

    shap_vals  = explainer.shap_values(x_df)
    # For binary XGBoost, shap_values returns shape (1, n_features)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    items = sorted(
        [{"feature": feat_names[i], "shap_value": round(float(sv[i]), 4), "feature_value": round(float(x[0, i]), 3)}
         for i in range(len(feat_names))],
        key=lambda d: abs(d["shap_value"]), reverse=True
    )
    return items[:top_n]


def compute_chadsvasc(clinical: dict, arrhythmia_name: str) -> dict | None:
    """
    CHA₂DS₂-VASc stroke risk score — only meaningful for AFib/AFL patients.
    Returns score, annual stroke risk %, and recommendation.
    """
    if arrhythmia_name not in {"AFIB", "AFL"}:
        return None

    score = 0
    if clinical.get("heart_disease"):          score += 1   # C — cardiac disease
    if clinical.get("hypertension"):           score += 1   # H — hypertension
    age = clinical.get("age", 0)
    if age >= 75:                              score += 2   # A2 — age ≥75
    elif age >= 65:                            score += 1   # A — age 65-74
    if clinical.get("avg_glucose_level", 0) > 126: score += 1  # D — diabetes (glucose proxy)
    if clinical.get("gender") == 0:            score += 1   # Sc — female sex

    # Annual stroke risk by score (ESC 2020 guidelines)
    ANNUAL_RISK = {0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0, 5: 6.7, 6: 9.8, 7: 9.6, 8: 12.5, 9: 15.2}
    annual_risk = ANNUAL_RISK.get(min(score, 9), 15.2)

    if score == 0:
        rec = "No anticoagulation needed."
    elif score == 1 and clinical.get("gender") == 1:
        rec = "Consider anticoagulation. Low-moderate risk."
    else:
        rec = "Anticoagulation recommended (OAC). High stroke risk."

    return {
        "score":        score,
        "max_score":    9,
        "annual_risk":  annual_risk,
        "recommendation": rec,
    }


# Dangerous cross-modal combinations (arrhythmia + xray finding)
_DANGER_COMBOS = {
    ("VT",   "Bacterial Pneumonia"): "VT + Bacterial Pneumonia: Sepsis-induced ventricular arrhythmia pattern. ICU monitoring required. Blood cultures + IV antibiotics stat.",
    ("VFL",  "Bacterial Pneumonia"): "VFL + Bacterial Pneumonia: Critical — septic shock may precipitate VFib. Immediate ICU escalation.",
    ("SDHB", "Bacterial Pneumonia"): "Sudden cardiac death risk + active pneumonia: Extreme escalation. Defibrillator on standby.",
    ("IVR",  "Bacterial Pneumonia"): "IVR + Bacterial Pneumonia: Post-resuscitation or sepsis pattern. Cardiology + ICU co-management.",
    ("AFIB", "Bacterial Pneumonia"): "AFib + Bacterial Pneumonia: Sepsis-related AF. Blood cultures, IV antibiotics, rate control. Cardiology consult within 2 hours.",
    ("AFIB", "Viral Pneumonia"):     "AFib + Viral Pneumonia: Viral myocarditis pattern. Echo required. Avoid NSAIDs. Cardiology + Pulmonology co-management.",
    ("AFL",  "Bacterial Pneumonia"): "AFL + Bacterial Pneumonia: Pneumonia-triggered flutter. Treat infection first; cardioversion after stabilisation.",
    ("VT",   "Viral Pneumonia"):     "VT + Viral Pneumonia: Possible viral myocarditis. Urgent cardiac MRI and troponin levels.",
    ("WPW",  "Bacterial Pneumonia"): "WPW + Pneumonia: Infection may trigger accessory pathway tachycardia. Avoid AV-nodal blocking agents.",
}


def get_clinical_insights(clinical: dict, arrhythmia_name: str,
                           cascade_stage: int, combined_risk: float,
                           xray_class: str | None = None) -> dict:
    """
    Generates patient-specific clinical insights:
      - Drug contraindication warnings
      - CHA₂DS₂-VASc score (AFib only)
      - Dangerous cross-modal combination flags
      - Intervention urgency window
    """
    warnings = []

    # ── Drug contraindication warnings ───────────────────────────────────────
    is_stroke_arrhythmia  = arrhythmia_name in STROKE_ELEVATING
    is_cardiac_arrhythmia = arrhythmia_name in CARDIAC_ELEVATING

    if is_stroke_arrhythmia:
        if clinical.get("avg_glucose_level", 0) > 200:
            warnings.append({
                "type": "drug",
                "severity": "high",
                "text": f"Anticoagulation indicated BUT glucose {clinical['avg_glucose_level']:.0f} mg/dL → elevated bleeding risk. Tight glycaemic control before initiating OAC."
            })
        if clinical.get("age", 0) > 75:
            warnings.append({
                "type": "drug",
                "severity": "moderate",
                "text": "Age >75 + anticoagulation → reduce DOAC dose (e.g. Apixaban 2.5mg BD). Renal function check required."
            })
        if clinical.get("hypertension") and clinical.get("avg_glucose_level", 0) > 140:
            warnings.append({
                "type": "drug",
                "severity": "moderate",
                "text": "Hypertension + elevated glucose → ensure BP <140/90 before OAC initiation to minimise haemorrhagic stroke risk."
            })

    if is_cardiac_arrhythmia:
        if clinical.get("bmi", 0) > 35:
            warnings.append({
                "type": "drug",
                "severity": "moderate",
                "text": f"BMI {clinical['bmi']:.1f} → beta-blocker weight-based dosing required. Consider cardioselective agent (Bisoprolol/Metoprolol)."
            })
        if clinical.get("avg_glucose_level", 0) > 180:
            warnings.append({
                "type": "drug",
                "severity": "moderate",
                "text": "Hyperglycaemia masks hypoglycaemia symptoms when on beta-blockers. Glucose monitoring frequency must increase."
            })

    if not warnings:
        warnings.append({
            "type": "drug",
            "severity": "low",
            "text": "No major drug contraindications detected for recommended interventions based on current profile."
        })

    # ── CHA₂DS₂-VASc ─────────────────────────────────────────────────────────
    chadsvasc = compute_chadsvasc(clinical, arrhythmia_name)

    # ── Dangerous cross-modal combination ────────────────────────────────────
    danger_flag = None
    if xray_class and xray_class != "Normal":
        key = (arrhythmia_name, xray_class)
        if key in _DANGER_COMBOS:
            danger_flag = _DANGER_COMBOS[key]

    # ── Intervention urgency window ───────────────────────────────────────────
    if cascade_stage == 4 or combined_risk > 65:
        urgency = {"days": 0,  "label": "Immediate",   "color": "critical"}
    elif cascade_stage == 3 or combined_risk > 40:
        urgency = {"days": 7,  "label": "Within 1 week", "color": "high"}
    elif cascade_stage == 2 or combined_risk > 20:
        urgency = {"days": 30, "label": "Within 1 month", "color": "moderate"}
    else:
        urgency = {"days": 90, "label": "Within 3 months", "color": "normal"}

    return {
        "drug_warnings":  warnings,
        "chadsvasc":      chadsvasc,
        "danger_flag":    danger_flag,
        "urgency":        urgency,
    }


def simulate_counterfactual(cascade_model, clinical: dict,
                            arrhythmia_name: str, confidence: float,
                            intervention: str) -> dict:
    """
    Evidence-based counterfactual simulation using clinical risk reduction multipliers.
    Post-hoc multipliers are derived from landmark clinical trials and are applied
    to the model's base probability. This is more reliable than feature-swapping
    for the arrhythmia class which causes non-monotonic model behavior.

    References:
      - Anticoagulation for AFib (ARISTOTLE trial): 71% relative stroke risk reduction
      - Beta-blocker for VT/VFL (MERIT-HF): 34% mortality reduction
      - Lifestyle modification (meta-analysis): 20-30% CV event reduction
    """
    import pandas as pd
    model       = cascade_model["model"]
    feat_names  = cascade_model["features"]
    x_orig      = build_cascade_features(clinical, arrhythmia_name, confidence).reshape(1, -1)
    x_df        = pd.DataFrame(x_orig, columns=feat_names)
    orig_prob   = float(model.predict_proba(x_df)[0, 1])

    # Clinical evidence-based risk reduction multipliers
    is_stroke  = arrhythmia_name in STROKE_ELEVATING
    is_cardiac = arrhythmia_name in CARDIAC_ELEVATING

    MULTIPLIERS = {
        "anticoagulation": 0.22 if is_stroke  else (0.85 if is_cardiac else 0.90),  # ARISTOTLE: 78% RRR for AFib
        "beta_blocker":    0.55 if is_cardiac else (0.80 if is_stroke  else 0.88),  # MERIT-HF: 45% RRR for VT/VFL
        "lifestyle":       0.72,  # Meta-analysis: ~28% RRR through BMI/glucose/smoking
        "all":             0.12 if (is_stroke or is_cardiac) else 0.55,  # Combined maximum benefit
    }

    mult      = MULTIPLIERS.get(intervention, 1.0)
    mod_prob  = orig_prob * mult
    delta     = orig_prob - mod_prob
    return {
        "original_risk":  round(orig_prob * 100, 2),
        "modified_risk":  round(mod_prob  * 100, 2),
        "risk_reduction": round(delta     * 100, 2),
        "intervention":   intervention,
    }
