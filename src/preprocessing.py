from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    return pd.read_csv(path, sep=";")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка данных и feature engineering.

    duration не удаляется здесь намеренно: его удаляем на этапе split,
    чтобы явно показать, как избегаем data leakage.
    """
    df = df.copy()

    df["y"] = df["y"].map({"no": 0, "yes": 1})

    if "pdays" in df.columns:
        df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)
        df["pdays_clean"] = df["pdays"].replace(-1, 999)

    if {"housing", "loan"}.issubset(df.columns):
        df["has_any_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)

    if "balance" in df.columns:
        df["balance_log_signed"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))

    if "campaign" in df.columns:
        df["campaign_capped"] = df["campaign"].clip(upper=df["campaign"].quantile(0.99))

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["0_25", "26_35", "36_45", "46_55", "56_65", "66_plus"],
            include_lowest=True,
        ).astype(str)

    if "month" in df.columns:
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


def make_split(
    df: pd.DataFrame,
    drop_duration: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    drop_cols = ["y"]

    # Основной честный сценарий: до звонка duration неизвестен.
    if drop_duration and "duration" in df.columns:
        drop_cols.append("duration")

    X = df.drop(columns=drop_cols)
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_one_hot_encoder() -> OneHotEncoder:
    """Совместимость со старыми и новыми версиями sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
