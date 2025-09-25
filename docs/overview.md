ML_Fraud – Project Overview

Purpose
- End‑to‑end prototype for near‑real‑time e‑payments fraud detection (Sri Lanka GovPay context).
- Covers ingestion → bronze/silver → feature engineering → training → model selection → serving → validation.

Architecture (high level)
- Data: raw CSV/JSON → bronze/silver JSON → gold feature CSV.
- Features: per‑transaction engineered signals (time, velocity, novelty, z‑scores, channel dummies, optional anomaly score).
- Models: baseline XGBoost; stacked ensemble (XGB + IsolationForest features → LogisticRegression).
- Serving: FastAPI service scoring numeric features; policy.yaml controls threshold.
- Storage: local filesystem by default; optional ADLS routing via environment.

Repository Map
- ingestion/: config, storage router (local/ADLS), Pydantic schemas, ingestion CLI.
- features/: feature builder (txn_features.py).
- synthetic/: synthetic data generators (builtin + simple configurable) and CLI.
- models/: training (baseline/ensemble), registry + champion pointers.
- scoring/: FastAPI service, batch scoring, Dockerfile.
- validation/: synthetic dataset checks.
- configs/: policy.yaml (threshold), synth.yaml (simple generator config).
- data/: bronze, silver, gold, models (artifacts), reports.

Common CLIs
- Ingest: `python ML_Fraud/run_txn_ingest.py --input-csv path/to/txns.csv --prefix txn`
- Features: `python ML_Fraud/run_txn_features.py --silver-dir ML_Fraud/data/silver/txn --out ML_Fraud/data/gold/txn_features.csv`
- Synthetic: `python ML_Fraud/run_txn_synth.py --generator simple --config ML_Fraud/configs/synth.yaml`
- Train baseline: `python ML_Fraud/run_train_baseline.py --gold-path ML_Fraud/data/gold/txn_synth_features.csv --promote`
- Train ensemble: `python ML_Fraud/run_train_ensemble.py --gold-path ML_Fraud/data/gold/txn_synth_features.csv --promote`
- Serve: `uvicorn ML_Fraud.scoring.service:app --host 0.0.0.0 --port 8000`

