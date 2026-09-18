"""
Train cascade chain models using sklearn (avoids PyTorch/XGBoost C-ext conflict):
  1. Cardiac GradientBoostingClassifier   → heart.csv
  2. Stroke GradientBoostingClassifier    → stroke_prediction.csv
  3. Cascade GradientBoostingClassifier   → combined features (clinical + ECG injection)

Saves: backend/saved_models/{cardiac,stroke,cascade}_model.joblib + cascade_meta.json
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

SAVE_DIR   = os.path.join(os.path.dirname(__file__), "../saved_models")
HEART_CSV  = os.path.join(os.path.dirname(__file__), "../../data/heart/heart disease tabular/heart.csv")
STROKE_CSV = os.path.join(os.path.dirname(__file__), "../../data/stroke/stroke_prediction.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

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


def train_cardiac():
    print("\n=== Training Cardiac Model (sklearn GBC) ===")
    df = pd.read_csv(HEART_CSV)
    FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal"]
    X = df[FEATURES].fillna(0)
    y = df["target"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"  AUC-ROC: {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, model.predict(X_te), zero_division=0))
    path = os.path.join(SAVE_DIR, "cardiac_model.joblib")
    joblib.dump({"model": model, "features": FEATURES}, path)
    print(f"  → {path}")
    return model, FEATURES


def train_stroke():
    print("\n=== Training Stroke Model (sklearn GBC) ===")
    df = pd.read_csv(STROKE_CSV).dropna()
    df["gender"]         = LabelEncoder().fit_transform(df["gender"].astype(str))
    df["ever_married"]   = (df["ever_married"] == "Yes").astype(int)
    df["work_type"]      = LabelEncoder().fit_transform(df["work_type"].astype(str))
    df["Residence_type"] = LabelEncoder().fit_transform(df["Residence_type"].astype(str))
    df["smoking_status"] = LabelEncoder().fit_transform(df["smoking_status"].astype(str))
    df["bmi"]            = pd.to_numeric(df["bmi"], errors="coerce").fillna(28.0)
    FEATURES = ["gender","age","hypertension","heart_disease","ever_married",
                "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]
    X = df[FEATURES]; y = df["stroke"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
                                       subsample=0.8, random_state=42)
    model.fit(X_res, y_res)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"  AUC-ROC: {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, model.predict(X_te), zero_division=0))
    path = os.path.join(SAVE_DIR, "stroke_model.joblib")
    joblib.dump({"model": model, "features": FEATURES}, path)
    print(f"  → {path}")
    return model, FEATURES


def train_cascade(cardiac_features, stroke_features):
    print("\n=== Training Cascade Chain Model ===")
    df = pd.read_csv(STROKE_CSV).dropna()
    df["gender"]         = LabelEncoder().fit_transform(df["gender"].astype(str))
    df["ever_married"]   = (df["ever_married"] == "Yes").astype(int)
    df["work_type"]      = LabelEncoder().fit_transform(df["work_type"].astype(str))
    df["Residence_type"] = LabelEncoder().fit_transform(df["Residence_type"].astype(str))
    df["smoking_status"] = LabelEncoder().fit_transform(df["smoking_status"].astype(str))
    df["bmi"]            = pd.to_numeric(df["bmi"], errors="coerce").fillna(28.0)

    BASE_FEATURES = ["gender","age","hypertension","heart_disease","ever_married",
                     "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]

    rng  = np.random.default_rng(42)
    n    = len(df)
    arr_risk = np.zeros(n)
    arr_idx  = np.zeros(n, dtype=int)

    for loc, (_, row) in enumerate(df.iterrows()):
        age, hd, htn = row["age"], row["heart_disease"], row["hypertension"]
        probs = np.ones(len(ARRHYTHMIA_CLASSES)) * 0.02
        probs[0] = max(0.05, 0.5 - age * 0.003)
        if age > 60:  probs[3] += 0.25; probs[2] += 0.05
        if hd:        probs[9]  += 0.10; probs[6] += 0.08
        if htn:       probs[3]  += 0.08; probs[13]+= 0.05
        probs = np.maximum(probs, 0); probs /= probs.sum()
        chosen = rng.choice(len(ARRHYTHMIA_CLASSES), p=probs)
        arr_idx[loc]  = chosen
        arr_risk[loc] = CASCADE_RISK_WEIGHT[ARRHYTHMIA_CLASSES[chosen]]

    df["arrhythmia_risk_weight"] = arr_risk
    for j, cls in enumerate(ARRHYTHMIA_CLASSES):
        df[f"arr_{cls}"] = (arr_idx == j).astype(int)

    CASCADE_FEATURES = BASE_FEATURES + ["arrhythmia_risk_weight"] + [f"arr_{c}" for c in ARRHYTHMIA_CLASSES]
    X = df[CASCADE_FEATURES]; y = df["stroke"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)

    model = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                                       subsample=0.8, random_state=42)
    model.fit(X_res, y_res)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"  Cascade AUC-ROC: {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, model.predict(X_te), zero_division=0))

    path = os.path.join(SAVE_DIR, "cascade_model.joblib")
    joblib.dump({
        "model": model, "features": CASCADE_FEATURES,
        "base_features": BASE_FEATURES,
        "arrhythmia_classes": ARRHYTHMIA_CLASSES,
    }, path)

    meta = {
        "arrhythmia_classes": ARRHYTHMIA_CLASSES,
        "cascade_risk_weights": CASCADE_RISK_WEIGHT,
        "cascade_features": CASCADE_FEATURES,
        "base_features": BASE_FEATURES,
        "cardiac_features": cardiac_features,
        "stroke_features": stroke_features,
    }
    with open(os.path.join(SAVE_DIR, "cascade_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → {path} + cascade_meta.json")


if __name__ == "__main__":
    _, c_feats = train_cardiac()
    _, s_feats = train_stroke()
    train_cascade(c_feats, s_feats)
    print("\n✓ All cascade models trained and saved (sklearn joblib).")
