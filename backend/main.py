"""CascadeIQ FastAPI backend — startup, CORS, router registration."""

import os, sys
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "saved_models")
ECG_MODEL   = os.path.join(MODELS_DIR, "ecg_model.pth")

# Force CPU: MPS + SHAP TreeExplainer conflict causes SIGSEGV when both load in same process
DEVICE = "cpu"

sys.path.insert(0, BASE_DIR)
from api.ecg     import router as ecg_router,     load_ecg_model
from api.cascade import router as cascade_router, load_cascade_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[CascadeIQ] Loading models on {DEVICE}...")
    load_ecg_model(ECG_MODEL, DEVICE)
    load_cascade_models(MODELS_DIR)
    print("[CascadeIQ] All models ready ✓")
    yield
    print("[CascadeIQ] Shutting down.")


app = FastAPI(
    title="CascadeIQ API",
    description="Predictive Disease Cascade Intelligence — arrhythmia-to-stroke cascade chain with counterfactual intervention simulation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ecg_router)
app.include_router(cascade_router)


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "service": "CascadeIQ"}
