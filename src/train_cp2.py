from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from evaluate import evaluate_at_threshold, get_score, tune_threshold
from preprocessing import (
    RANDOM_STATE,
    build_preprocessor,
    load_data,
    make_split,
    prepare_features,
)

warnings.filterwarnings("ignore")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_pipeline(X_train: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ]
    )


def run_grid_search(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print(f"Training {model_name}...")
    search.fit(X_train, y_train)
    print(f"{model_name}: best CV PR-AUC = {search.best_score_:.4f}")
    print(f"{model_name}: best params = {search.best_params_}")

    return search


def plot_model_comparison(results: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = results.sort_values("pr_auc", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["model"], plot_df["pr_auc"])
    plt.xlabel("validation PR-AUC")
    plt.title("CP2 model comparison by PR-AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "cp2_model_pr_auc.png", dpi=150)
    plt.close()


def plot_precision_recall_curve(model, X_test, y_test, figures_dir: Path) -> None:
    y_score = get_score(model, X_test)

    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("precision-recall curve for final model")
    plt.tight_layout()
    plt.savefig(figures_dir / "cp2_precision_recall_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, figures_dir: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["pred no", "pred yes"])
    plt.yticks([0, 1], ["true no", "true yes"])
    plt.colorbar()
    plt.title("confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(figures_dir / "cp2_confusion_matrix.png", dpi=150)
    plt.close()


def write_report(
    report_dir: Path,
    results: pd.DataFrame,
    threshold_results: pd.DataFrame,
    test_results: pd.DataFrame,
    best_model_name: str,
    best_threshold: float,
    classification_text: str,
    cm: np.ndarray,
) -> None:
    report_text = f"""# CP2. Прогноз отклика клиента на оформление банковского депозита

## 1. Что сделано

В CP2 была продолжена задача бинарной классификации банковского отклика.

По сравнению с CP1 добавлено:

- расширенный feature engineering;
- отдельный модуль предобработки `src/preprocessing.py`;
- отдельный модуль оценки `src/evaluate.py`;
- 5 моделей машинного обучения;
- ансамбль VotingEnsemble;
- GridSearchCV для подбора гиперпараметров;
- подбор threshold по F1-score;
- финальная проверка на test split;
- графики для сравнения моделей и анализа финальной модели.

## 2. Метрики

Основная метрика — PR-AUC.

Причина: классы несбалансированы, положительный класс `yes` встречается реже. В задаче банковского маркетинга важно не просто угадать большинство клиентов класса `no`, а хорошо ранжировать клиентов по вероятности положительного отклика.

Дополнительные метрики:

- F1-score — баланс precision и recall;
- precision — какая доля клиентов, предсказанных как перспективные, реально оформила депозит;
- recall — какую долю реальных клиентов с откликом модель смогла найти;
- ROC-AUC — дополнительная общая метрика качества ранжирования.

Предпочтение отдаётся PR-AUC, потому что она лучше отражает качество модели на несбалансированном положительном классе.

## 3. Data leakage

Признак `duration` показывает длительность последнего звонка. До звонка банк не знает эту длительность, поэтому использовать `duration` для предварительного выбора клиентов нельзя. В основной модели этот признак исключён.

## 4. Эксперименты

{results.to_markdown(index=False)}

## 5. Выбор финальной модели

Финальная модель выбрана по максимальному PR-AUC на validation split.

Финальная модель:

```text
{best_model_name}
```

Финальный threshold:

```text
{best_threshold}
```

Лучшие значения threshold по F1:

{threshold_results.sort_values("f1", ascending=False).head(10).to_markdown(index=False)}

## 6. Test split

{test_results.to_markdown(index=False)}

Classification report:

```text
{classification_text}
```

Confusion matrix:

```text
{cm}
```

## 7. Вывод

В CP2 были закрыты основные замечания из CP1: добавлены 4–5 моделей, ансамбль, GridSearchCV, конкретные таблицы метрик, обоснование основной метрики и аргументация выбора финальной модели.

На CP3 нужно оформить финальный отчёт в PDF и добавить деплой через FastAPI.
"""

    (report_dir / "cp2_report.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    root = project_root()
    data_path = root / "data" / "raw" / "bank-full.csv"
    report_dir = root / "report"
    figures_dir = report_dir / "figures"
    models_dir = root / "models"

    report_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Loading:", data_path)
    raw_df = load_data(data_path)
    print("Raw shape:", raw_df.shape)

    df = prepare_features(raw_df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(df, drop_duration=True)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)
    print("Positive class share:", round(float(y_train.mean()), 4))

    models: dict[str, object] = {}
    rows: list[dict] = []

    dummy = make_pipeline(X_train, DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_train, y_train)
    dummy_row = evaluate_at_threshold(dummy, X_val, y_val, "DummyClassifier")
    dummy_row["best_params"] = "{}"
    dummy_row["cv_pr_auc"] = np.nan
    models["DummyClassifier"] = dummy
    rows.append(dummy_row)

    model_specs = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "classifier__C": [0.1, 1.0, 3.0],
                "classifier__solver": ["liblinear"],
            },
        ),
        (
            "RandomForest",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 350],
                "classifier__max_depth": [8, 14],
                "classifier__min_samples_leaf": [3, 8],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 350],
                "classifier__max_depth": [8, 14],
                "classifier__min_samples_leaf": [3, 8],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 180],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__max_depth": [2, 3],
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__max_iter": [120, 220],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__max_leaf_nodes": [15, 31],
            },
        ),
    ]

    for model_name, estimator, grid in model_specs:
        pipe = make_pipeline(X_train, estimator)
        search = run_grid_search(model_name, pipe, grid, X_train, y_train)
        best_model = search.best_estimator_

        row = evaluate_at_threshold(best_model, X_val, y_val, model_name)
        row["best_params"] = str(search.best_params_)
        row["cv_pr_auc"] = float(search.best_score_)

        models[model_name] = best_model
        rows.append(row)

    print("Training VotingEnsemble...")
    voting = VotingClassifier(
        estimators=[
            ("lr", models["LogisticRegression"]),
            ("rf", models["RandomForest"]),
            ("gb", models["GradientBoosting"]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    voting.fit(X_train, y_train)

    voting_row = evaluate_at_threshold(voting, X_val, y_val, "VotingEnsemble")
    voting_row["best_params"] = "soft voting: LogisticRegression + RandomForest + GradientBoosting"
    voting_row["cv_pr_auc"] = np.nan
    models["VotingEnsemble"] = voting
    rows.append(voting_row)

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results.to_csv(report_dir / "cp2_experiment_results.csv", index=False)

    best_model_name = str(results.iloc[0]["model"])
    best_model = models[best_model_name]

    best_threshold, threshold_results = tune_threshold(best_model, X_val, y_val, best_model_name)
    threshold_results.to_csv(report_dir / "cp2_threshold_search.csv", index=False)

    test_row = evaluate_at_threshold(best_model, X_test, y_test, best_model_name, best_threshold)
    test_results = pd.DataFrame([test_row])
    test_results.to_csv(report_dir / "cp2_test_results.csv", index=False)

    y_test_score = get_score(best_model, X_test)
    y_test_pred = (y_test_score >= best_threshold).astype(int)

    classification_text = classification_report(y_test, y_test_pred, target_names=["no", "yes"])
    cm = confusion_matrix(y_test, y_test_pred)

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "model_name": best_model_name,
            "drop_duration": True,
        },
        models_dir / "cp2_final_model.joblib",
    )

    plot_model_comparison(results, figures_dir)
    plot_precision_recall_curve(best_model, X_test, y_test, figures_dir)
    plot_confusion_matrix(cm, figures_dir)

    write_report(
        report_dir=report_dir,
        results=results,
        threshold_results=threshold_results,
        test_results=test_results,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        classification_text=classification_text,
        cm=cm,
    )

    print("\\nDone.")
    print("Best model:", best_model_name)
    print("Best threshold:", best_threshold)
    print("\\nValidation results:")
    print(results)
    print("\\nTest results:")
    print(test_results)


if __name__ == "__main__":
    main()
