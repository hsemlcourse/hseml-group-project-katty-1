from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw_score = model.decision_function(X)
        return (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)

    return model.predict(X)


def evaluate_at_threshold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    y_score = get_score(model, X)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "model": model_name,
        "threshold": threshold,
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_score),
        "pr_auc": average_precision_score(y, y_score),
    }


def tune_threshold(model, X_val: pd.DataFrame, y_val: pd.Series, model_name: str):
    y_score = get_score(model, X_val)
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= threshold).astype(int)

        rows.append(
            {
                "model": model_name,
                "threshold": round(float(threshold), 2),
                "f1": f1_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
            }
        )

    result = pd.DataFrame(rows)
    best_threshold = float(result.sort_values("f1", ascending=False).iloc[0]["threshold"])

    return best_threshold, result
