from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cp2_final_model.joblib"


class BankClientInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    job: str
    marital: Literal["married", "single", "divorced"]
    education: str
    default: Literal["yes", "no"]
    balance: float
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: str
    day: int = Field(..., ge=1, le=31)
    month: str
    campaign: int = Field(..., ge=1)
    pdays: int
    previous: int = Field(..., ge=0)
    poutcome: str


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run python src/train_cp2.py first."
        )

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict) and "model" in bundle:
        return {
            "model": bundle["model"],
            "threshold": float(bundle.get("threshold", 0.5)),
            "model_name": bundle.get("model_name", "unknown"),
            "drop_duration": bool(bundle.get("drop_duration", True)),
        }

    return {
        "model": bundle,
        "threshold": 0.5,
        "model_name": "unknown",
        "drop_duration": True,
    }


def prepare_inference_features(payload: BankClientInput) -> pd.DataFrame:
    data = payload.model_dump()
    df = pd.DataFrame([data])

    df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)
    df["pdays_clean"] = df["pdays"].replace(-1, 999)
    df["has_any_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)
    df["balance_log_signed"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["campaign_capped"] = df["campaign"].clip(upper=50)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["0_25", "26_35", "36_45", "46_55", "56_65", "66_plus"],
        include_lowest=True,
    ).astype(str)

    month_order = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month_number"] = df["month"].map(month_order).fillna(0).astype(int)

    return df


def predict_one(payload: BankClientInput, model_bundle: dict) -> dict:
    model = model_bundle["model"]
    threshold = float(model_bundle["threshold"])

    features = prepare_inference_features(payload)
    probability = float(model.predict_proba(features)[:, 1][0])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "prediction_label": "yes" if prediction == 1 else "no",
        "probability_yes": probability,
        "threshold": threshold,
        "model_name": model_bundle["model_name"],
        "duration_used": False,
    }


def sample_payload() -> dict:
    return {
        "age": 34,
        "job": "management",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }
