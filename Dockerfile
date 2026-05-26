FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
