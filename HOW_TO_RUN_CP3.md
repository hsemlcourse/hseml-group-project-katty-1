# CP3. Запуск деплоя

## Локальный запуск

```powershell
pip install -r requirements.txt
uvicorn src.app:app --reload
```

Открыть:

```text
http://127.0.0.1:8000/docs
```

## Проверка из второго терминала

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body (Get-Content sample_payload.json -Raw)
```

## Docker

```powershell
docker compose up api --build
```

## Что добавить в отчёт

Сделать скриншоты:

```text
report/figures/deploy_docs.png
report/figures/deploy_predict.png
```

Записать короткое видео работы API и вставить ссылку в `report/final_report.md`.
