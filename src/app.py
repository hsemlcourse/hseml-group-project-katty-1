from __future__ import annotations

from fastapi import FastAPI

from src.inference import BankClientInput, load_model_bundle, predict_one, sample_payload

app = FastAPI(
    title="Bank Deposit Subscription Prediction API",
    description="FastAPI service for bank deposit subscription prediction.",
    version="1.0.0",
)

MODEL_BUNDLE = load_model_bundle()


@app.get("/")
def root() -> dict:
    return {
        "service": "Bank Deposit Subscription Prediction API",
        "docs": "/docs",
        "health": "/health",
        "sample": "/sample",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_name": MODEL_BUNDLE["model_name"],
        "threshold": MODEL_BUNDLE["threshold"],
    }


@app.get("/model-info")
def model_info() -> dict:
    return {
        "task": "binary classification",
        "positive_class": "client subscribes to term deposit",
        "model_name": MODEL_BUNDLE["model_name"],
        "threshold": MODEL_BUNDLE["threshold"],
        "duration_used": False,
        "main_metric": "PR-AUC",
    }


@app.get("/sample")
def sample() -> dict:
    return sample_payload()


@app.post("/predict")
def predict(payload: BankClientInput) -> dict:
    return predict_one(payload, MODEL_BUNDLE)
