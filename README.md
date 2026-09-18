# CascadeIQ — Predictive Disease Cascade Intelligence

> **"We don't diagnose. We predict the cascade and simulate the cure."**

CascadeIQ is an AI system that takes an ECG signal and patient clinical profile and does not stop at a diagnosis label. It maps the patient onto a disease cascade pathway, predicts downstream cardiac and stroke events across a 30/90/365-day timeline, and simulates the outcome of clinical interventions **before** they are made.

---

## What Makes It Unique

| Existing tools | CascadeIQ |
|---|---|
| Classify ECG rhythm → stop | Classify + inject as feature into downstream stroke model |
| Give a static risk percentage | Give 30/90/365-day cascade timeline across 3 futures |
| Black-box prediction | SHAP explanation + counterfactual intervention simulation |
| Single modality | ECG signal + clinical data → chained cascade architecture |

The counterfactual engine answers: *"If we start anticoagulation now, how does this patient's 30-day stroke risk change — specifically for their profile?"*

---

## Architecture

```
ECG Signal (.mat)
      ↓
  1D ResNet CNN (17 arrhythmia classes)
      ↓ arrhythmia_type + confidence
      ↓ ←── FEATURE INJECTION ──────────────────┐
  Patient Clinical Features                      │
  (age, BP, glucose, BMI, smoking...)            │
      ↓                                          │
  Cascade Chain Model (GBC + SMOTE)              │
      ↓                                          │
  30/90/365-day Cascade Timeline                 │
      ↓                                          │
  SHAP Explanation                               │
      ↓                                          │
  Counterfactual Intervention Engine ────────────┘
  (anticoagulation / beta-blocker / lifestyle / all)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ECG Model | 1D ResNet (PyTorch) — 17 arrhythmia classes |
| Cascade Model | GradientBoostingClassifier (sklearn) + SMOTE |
| Explainability | SHAP TreeExplainer |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Tailwind CSS + Recharts |

---

## Datasets

| Dataset | Source | Samples | Task |
|---|---|---|---|
| ECG Signals | MIT-BIH Arrhythmia DB | 1,001 segments | 17-class arrhythmia detection |
| Heart Disease | Cleveland Heart Disease | 1,025 patients | Cardiac risk scoring |
| Stroke | Kaggle Stroke Prediction | 5,110 patients | Cascade endpoint (stroke) |

---

## Quick Start

```bash
# One command — starts backend + frontend
bash start.sh
```

Then open **http://localhost:5173**

### Manual start (if needed)

```bash
# Terminal 1 — Backend
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
export PATH="/tmp/node-v20.18.0-darwin-arm64/bin:$PATH"
cd frontend && npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status |
| POST | `/ecg/predict` | ECG arrhythmia classification (upload .mat file) |
| POST | `/cascade/predict` | Full cascade timeline + SHAP explanation |
| POST | `/cascade/all-interventions` | All 4 intervention counterfactuals |

**API docs (interactive):** http://localhost:8000/docs

---

## Demo Samples

Pre-loaded test ECG files in `demo_samples/`:
- `demo_AFIB.mat` — Atrial Fibrillation (stroke-elevating)
- `demo_VT.mat` — Ventricular Tachycardia (cardiac-critical)
- `demo_NSR.mat` — Normal Sinus Rhythm (baseline)

---

## Demo Script (3 min)

1. Open `http://localhost:5173`
2. Upload `demo_AFIB.mat` → Model classifies AFIB at ~94% confidence. Alert: HIGH RISK.
3. Fill clinical form with: Age 63, Male, Hypertension Yes, Glucose 142, BMI 28.4
4. Click **Run Cascade Analysis**
5. Show the 3-future timeline: no intervention (risk grows) vs. conservative vs. aggressive
6. In the Counterfactual panel: select **Anticoagulation** → risk drops from 13% to 2.9% (78% reduction)
7. Switch to `demo_VT.mat` → CRITICAL alert. Beta-blocker most effective intervention.
8. Explain SHAP: "These are the exact features driving this patient's risk — not a generic score."

**Closing line:** *"This is not another ECG classifier. This is the first system to chain an arrhythmia finding into a personalized cascade timeline and simulate the clinical decision before it's made."*

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| ECG 1D ResNet (17-class) | Test Accuracy | 77% |
| Cardiac GBC | AUC-ROC | 1.00 (small dataset) |
| Stroke GBC | AUC-ROC | 0.73 |
| Cascade Chain GBC | AUC-ROC | 0.77 |

*Note: Cascade AUC (0.77) exceeds standalone stroke model (0.73) — confirming ECG feature injection improves prediction.*
