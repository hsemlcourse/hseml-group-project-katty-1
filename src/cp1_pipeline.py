"""
CP1 — Bank Marketing UCI Dataset.

Запуск из корня проекта:
    python src/cp1_pipeline.py

Ожидаемый файл:
    data/raw/bank-full.csv

После запуска создаются:
    report/figures/*.png
    report/cp1_experiment_results.csv
    report/cp1_duration_experiment.csv
    report/report.md
    models/cp1_random_forest_no_duration.joblib
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}\n"
            "Положи bank-full.csv в папку data/raw/."
        )
    return pd.read_csv(data_path, sep=";")


def save_basic_dataset_info(df: pd.DataFrame, report_dir: Path) -> None:
    info_path = report_dir / "cp1_dataset_info.txt"
    with info_path.open("w", encoding="utf-8") as file:
        file.write("CP1 Dataset Info\n")
        file.write("================\n\n")
        file.write(f"Shape: {df.shape}\n\n")
        file.write("Columns:\n")
        for column in df.columns:
            file.write(f"- {column}\n")
        file.write("\nMissing values:\n")
        file.write(df.isna().sum().to_string())
        file.write("\n\nDuplicates:\n")
        file.write(str(df.duplicated().sum()))
        file.write("\n\nTarget distribution:\n")
        file.write(df["y"].value_counts().to_string())
        file.write("\n\nTarget distribution normalized:\n")
        file.write(df["y"].value_counts(normalize=True).to_string())


def plot_target_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    df["y"].value_counts().plot(kind="bar")
    plt.title("Target distribution")
    plt.xlabel("y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close()


def plot_target_rate_by_category(
    df: pd.DataFrame,
    column: str,
    figures_dir: Path,
    target: str = "y",
) -> None:
    target_rate = df.groupby(column)[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    target_rate.plot(kind="bar")
    plt.title(f"Target rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Mean target")
    plt.tight_layout()
    plt.savefig(figures_dir / f"target_rate_by_{column}.png", dpi=150)
    plt.close()


def plot_numeric_distribution(df: pd.DataFrame, column: str, figures_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    df[column].hist(bins=40)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figures_dir / f"distribution_{column}.png", dpi=150)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, figures_dir: Path) -> None:
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_columns].corr()
    plt.figure(figsize=(11, 8))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_matrix.png", dpi=150)
    plt.close()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    if "pdays" in df.columns:
        df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)
        df["pdays_clean"] = df["pdays"].replace(-1, 999)

    if {"housing", "loan"}.issubset(df.columns):
        df["has_any_loan"] = (
            (df["housing"] == "yes") | (df["loan"] == "yes")
        ).astype(int)

    return df


def make_split(
    df: pd.DataFrame,
    drop_duration: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    columns_to_drop = ["y"]

    # duration убираем из основной модели, потому что это возможный data leakage
    if drop_duration and "duration" in df.columns:
        columns_to_drop.append("duration")

    X = df.drop(columns=columns_to_drop)
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "model": model_name,
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_score),
        "pr_auc": average_precision_score(y, y_score),
    }


def train_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, Pipeline]:
    experiments = []

    dummy_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )
    dummy_model.fit(X_train, y_train)
    experiments.append(evaluate_model(dummy_model, X_val, y_val, "DummyClassifier"))

    logreg_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logreg_model.fit(X_train, y_train)
    experiments.append(
        evaluate_model(logreg_model, X_val, y_val, "LogisticRegression no duration")
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_model.fit(X_train, y_train)
    experiments.append(
        evaluate_model(rf_model, X_val, y_val, "RandomForest no duration")
    )

    results = pd.DataFrame(experiments).sort_values("pr_auc", ascending=False)
    return results, rf_model


def run_duration_leakage_experiment(df: pd.DataFrame) -> pd.DataFrame | None:
    if "duration" not in df.columns:
        return None

    X_train, X_val, _, y_train, y_val, _ = make_split(df, drop_duration=False)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    duration_result = evaluate_model(
        model,
        X_val,
        y_val,
        "LogisticRegression with duration",
    )

    return pd.DataFrame([duration_result])


def write_markdown_report(
    report_dir: Path,
    results: pd.DataFrame,
    duration_results: pd.DataFrame | None,
    val_report: str,
    val_confusion_matrix: np.ndarray,
) -> None:
    report_path = report_dir / "report.md"

    text = f"""# CP1. Прогноз отклика клиента на оформление банковского депозита

## 1. Постановка задачи

В проекте решается задача бинарной классификации в банковской сфере.

Цель — предсказать, оформит ли клиент срочный депозит после маркетингового контакта.

Таргет:

- `y = yes` — клиент оформил депозит;
- `y = no` — клиент не оформил депозит.

Практический смысл задачи: банк может заранее выделять клиентов с более высокой вероятностью отклика и эффективнее использовать ресурсы маркетинговой кампании.

## 2. Данные

Используется датасет Bank Marketing UCI.

Основной файл:

```text
data/raw/bank-full.csv
```

В данных есть признаки клиента, признаки текущей кампании и информация о прошлых контактах.

## 3. Подготовка данных

В CP1 выполнено:

- загрузка данных;
- проверка размера датасета;
- проверка типов данных;
- проверка пропусков;
- проверка дублей;
- анализ баланса классов;
- преобразование таргета `y` в бинарный формат;
- создание новых признаков.

Добавленные признаки:

- `was_previously_contacted` — был ли клиент ранее в контакте с банком;
- `pdays_clean` — обработанная версия признака `pdays`;
- `has_any_loan` — есть ли у клиента housing loan или personal loan.

## 4. Data leakage

Признак `duration` показывает длительность последнего звонка.

В реальном сценарии до звонка банк не знает его длительность. Поэтому основная модель обучается без `duration`, чтобы избежать data leakage.

## 5. Метрики

Используются:

- F1-score;
- precision;
- recall;
- ROC-AUC;
- PR-AUC.

Accuracy не выбрана как основная метрика, потому что классы несбалансированы.

## 6. Эксперименты

Основные модели без `duration`:

{results.to_markdown(index=False)}

"""

    if duration_results is not None:
        text += f"""
Дополнительный эксперимент с `duration`:

{duration_results.to_markdown(index=False)}

Этот эксперимент не считается основной честной моделью. Он нужен, чтобы показать влияние признака `duration`.
"""

    text += f"""

## 7. Отчёт по предварительной лучшей модели

```text
{val_report}
```

Confusion matrix:

```text
{val_confusion_matrix}
```

## 8. Выводы CP1

В рамках CP1 был подготовлен полный базовый ML-пайплайн:

- выполнен анализ данных;
- сделаны графики EDA;
- подготовлены признаки;
- сделан корректный stratified train/validation/test split;
- обучены DummyClassifier, LogisticRegression и RandomForestClassifier;
- посчитаны основные метрики качества;
- сохранены результаты и модель.

На CP2 планируется добавить больше моделей, подбор гиперпараметров, настройку threshold и более подробную интерпретацию результатов.
"""

    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = get_project_root()
    data_path = project_root / "data" / "raw" / "bank-full.csv"
    report_dir = project_root / "report"
    figures_dir = report_dir / "figures"
    models_dir = project_root / "models"

    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Loading data:", data_path)

    raw_df = load_data(data_path)
    print("Raw shape:", raw_df.shape)

    if "y" not in raw_df.columns:
        raise ValueError("В данных нет колонки y. Проверь, что используется bank-full.csv.")

    save_basic_dataset_info(raw_df, report_dir)
    plot_target_distribution(raw_df, figures_dir)

    df = prepare_features(raw_df)

    for column in ["job", "marital", "education", "contact", "month", "poutcome"]:
        if column in df.columns:
            plot_target_rate_by_category(df, column, figures_dir)

    for column in ["age", "balance", "campaign", "previous", "duration"]:
        if column in df.columns:
            plot_numeric_distribution(df, column, figures_dir)

    plot_correlation_matrix(df, figures_dir)

    X_train, X_val, X_test, y_train, y_val, y_test = make_split(df, drop_duration=True)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    results, best_model = train_models(X_train, X_val, y_train, y_val)

    results_path = report_dir / "cp1_experiment_results.csv"
    results.to_csv(results_path, index=False)

    model_path = models_dir / "cp1_random_forest_no_duration.joblib"
    joblib.dump(best_model, model_path)

    y_val_pred = best_model.predict(X_val)
    val_report = classification_report(y_val, y_val_pred, target_names=["no", "yes"])
    val_confusion_matrix = confusion_matrix(y_val, y_val_pred)

    duration_results = run_duration_leakage_experiment(df)
    if duration_results is not None:
        duration_results.to_csv(report_dir / "cp1_duration_experiment.csv", index=False)

    write_markdown_report(
        report_dir=report_dir,
        results=results,
        duration_results=duration_results,
        val_report=val_report,
        val_confusion_matrix=val_confusion_matrix,
    )

    print("\nDone.")
    print("Saved figures:", figures_dir)
    print("Saved results:", results_path)
    print("Saved model:", model_path)
    print("Saved report:", report_dir / "report.md")
    print("\nMain results:")
    print(results)


if __name__ == "__main__":
    main()
