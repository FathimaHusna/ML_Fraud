FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional: for pandas/numPy performance)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ML_Fraud /app/ML_Fraud

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r ML_Fraud/requirements.txt

EXPOSE 8000

ENV MODEL_PATH=models/baseline_xgb.pkl \
    FEATURES_JSON=models/baseline_xgb.features.json \
    STORAGE_MODE=adls

CMD ["uvicorn", "ML_Fraud.scoring.service:app", "--host", "0.0.0.0", "--port", "8000"]

