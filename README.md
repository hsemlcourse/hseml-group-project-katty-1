[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Прогноз отклика клиента на оформление банковского депозита]

**Студент:** [Байбакова Екатерина Алексеевна]

**Группа:** [БИВ234]


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

Проект посвящён задаче бинарной классификации в банковской сфере.

Нужно предсказать, оформит ли клиент срочный депозит, на основе информации о клиенте и истории маркетинговых контактов.

Практический смысл задачи — помочь банку заранее определить клиентов с наибольшей вероятностью отклика и эффективнее планировать маркетинговую кампанию.

**Тип задачи:** бинарная классификация.

**Целевая переменная:** `y`.

Значения таргета:

- `yes` — клиент оформил депозит;
- `no` — клиент не оформил депозит.

Положительный класс — клиент оформил депозит.

---

## Данные

Используется датасет **Bank Marketing UCI Dataset**.

Основной файл:

```text
data/raw/bank-full.csv
```

Также в `data/raw/` лежат дополнительные файлы датасета:

```text
data/raw/bank.csv
data/raw/bank-names.txt
```

Датасет содержит информацию о клиентах банка, параметрах маркетинговой кампании и результатах прошлых контактов.

В проекте используются:

- демографические признаки клиента;
- финансовые признаки клиента;
- признаки текущего маркетингового контакта;
- признаки прошлых маркетинговых кампаний.

---

## Метрики

Основная метрика — **PR-AUC**.

Почему именно PR-AUC:

- классы несбалансированы;
- положительный класс `yes` встречается заметно реже, чем `no`;
- в банковском маркетинге важно хорошо ранжировать клиентов по вероятности отклика;
- accuracy может быть обманчивой, потому что модель может часто предсказывать самый частый класс `no`.

Дополнительные метрики:

- `F1-score`;
- `precision`;
- `recall`;
- `ROC-AUC`.

`F1-score` используется как дополнительная рабочая метрика, потому что он балансирует `precision` и `recall`.

`precision` показывает, какая доля клиентов, предсказанных как перспективные, реально оформила депозит.

`recall` показывает, какую долю реальных клиентов с положительным откликом модель смогла найти.

---

## Data leakage

В датасете есть признак `duration`.

Он показывает длительность последнего звонка.

В реальном бизнес-сценарии до звонка банк не знает, сколько он будет длиться. Поэтому использовать `duration` для предварительного выбора клиентов нельзя: это привело бы к утечке данных.

В основной модели признак `duration` исключается.

---

## Структура репозитория

```text
.
├── data
│   ├── processed
│   └── raw
│       ├── bank-full.csv
│       ├── bank.csv
│       └── bank-names.txt
│
├── models
│   ├── cp1_random_forest_no_duration.joblib
│   └── cp2_final_model.joblib
│
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_baseline_and_experiments.ipynb
│
├── presentation
│
├── report
│   ├── figures
│   ├── cp1_dataset_info.txt
│   ├── cp1_duration_experiment.csv
│   ├── cp1_experiment_results.csv
│   ├── cp2_experiment_results.csv
│   ├── cp2_report.md
│   ├── cp2_test_results.csv
│   ├── cp2_threshold_search.csv
│   └── report.md
│
├── src
│   ├── __init__.py
│   ├── cp1_pipeline.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   └── train_cp2.py
│
├── tests
├── requirements.txt
├── Makefile
└── README.md
```

---

## Запуск

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить основной CP2-пайплайн:

```bash
python src/train_cp2.py
```

После запуска создаются или обновляются файлы:

```text
report/cp2_experiment_results.csv
report/cp2_threshold_search.csv
report/cp2_test_results.csv
report/cp2_report.md
report/figures/cp2_model_pr_auc.png
report/figures/cp2_precision_recall_curve.png
report/figures/cp2_confusion_matrix.png
models/cp2_final_model.joblib
```

Запустить линтер:

```bash
ruff check src
```

Или через Makefile:

```bash
make lint
```

---

## Что сделано в CP2

В CP2 была продолжена работа над модельной частью проекта.

Было добавлено:

- отдельный модуль предобработки `src/preprocessing.py`;
- отдельный модуль оценки `src/evaluate.py`;
- расширенный feature engineering;
- обучение нескольких моделей;
- подбор гиперпараметров через `GridSearchCV`;
- ансамбль `VotingEnsemble`;
- подбор threshold по `F1-score`;
- финальная проверка лучшей модели на test split;
- сохранение таблиц, графиков и итоговой модели.

Модели, которые сравнивались:

- `DummyClassifier`;
- `LogisticRegression`;
- `RandomForest`;
- `ExtraTrees`;
- `GradientBoosting`;
- `HistGradientBoosting`;
- `VotingEnsemble`.

---

## Результаты

Основная таблица экспериментов сохранена в файле:

```text
report/cp2_experiment_results.csv
```

Лучшая модель по validation PR-AUC:

```text
HistGradientBoosting
```

Лучшие параметры модели:

```text
learning_rate = 0.03
max_iter = 220
max_leaf_nodes = 15
```

Финальный threshold после подбора:

```text
0.21
```

Итоговая проверка на test split:

| Модель | Threshold | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.21 | 0.504 | 0.478 | 0.532 | 0.806 | 0.481 |

Вывод:

`HistGradientBoosting` была выбрана как финальная модель, потому что она показала лучший `PR-AUC` на validation split. После выбора модели был подобран threshold `0.21`, который улучшил баланс между `precision` и `recall`.

---

## Отчёт

Краткий отчёт по CP2:

```text
report/cp2_report.md
```

EDA с выводами по графикам:

```text
notebooks/01_eda.ipynb
```

Baseline и эксперименты:

```text
notebooks/02_baseline_and_experiments.ipynb
```

Финальная модель:

```text
models/cp2_final_model.joblib
```