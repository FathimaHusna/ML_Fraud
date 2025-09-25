ML_Fraud – E‑Payments Fraud (Sri Lanka GovPay)

Contents
- ingestion/: Storage adapters, settings, and transaction schema (silver)
- features/: Transaction feature engineering
- synthetic/: Transaction synthetic data generator + labels
- CLIs: `run_txn_ingest.py`, `run_txn_features.py`, `run_txn_synth.py`

Documentation
- docs/overview.md – high‑level purpose, repo map, key CLIs
- docs/pipeline.md – bronze/silver/gold pipeline and storage modes
- docs/synthetic.md – synthetic generators, config, validation
- docs/features.md – engineered features and design rationale
- docs/training.md – baseline/ensemble training, artifacts, registry
- docs/model_selection.md – metrics, thresholding, champion–challenger
- docs/validation.md – synthetic dataset checks and usage
- docs/serving_deploy.md – service endpoints, configs, deployment paths
- docs/docker_deploy.md – minimal Docker/Compose steps
- docs/mlops.md – experiments (MLflow), AutoML, retraining
- docs/statistics.md – core statistical concepts in this project
- docs/faq.md – quick questions and typical commands

Quickstart
- Install deps: `pip install -r requirements.txt`

Next steps: Features and Synthetic Data
- Compute features from existing txn silver JSONs:
  - `python ML_Fraud/run_txn_features.py --silver-dir ML_Fraud/data/silver/txn --out ML_Fraud/data/gold/txn_features.csv`
- Generate a synthetic transaction set:
  - `python ML_Fraud/run_txn_synth.py --users 300 --days 45 --avg-per-user 80 --fraud 0.25 --out-features ML_Fraud/data/gold/txn_synth_features.csv --out-records ML_Fraud/data/silver/txn_synth_records.json`

Notes
- Transaction features include time, channel, velocity windows (60s/5m/1h/1d), novelty flags, and 30‑day amount z‑scores.
- Synthetic labels mark odd‑hour + location/device anomalies, velocity bursts, and new‑payee/device high‑amount spikes.
- ADLS uploads are optional via `--to-adls` flags (requires `STORAGE_MODE=adls`).

—

e‑Payments Fraud (Sri Lanka GovPay) – Transaction Pipeline

Overview
- Aligns with the product document: hybrid ML with transaction, device/location, behavior, and velocity features.
- Adds a parallel transaction pipeline: ingestion → silver → feature engineering → model training.

Modules
- ingestion/txn_schema.py: Defines raw and normalized transaction schemas.
- run_txn_ingest.py: Ingests transactions from CSV/JSON into bronze/silver (`data/bronze/txn`, `data/silver/txn`).
- features/txn_features.py: Builds per‑transaction features (amount z‑scores, time, channel, velocity counts/sums, device/payee/city novelty, 30‑day stats).
- run_txn_features.py: CLI to compute features from silver and optionally upload to ADLS.
- synthetic/txn_generate.py: Generates realistic transaction streams and fraud patterns.
- run_txn_synth.py: CLI to generate synthetic txn features and records; supports ADLS upload.

Synthetic Generators
- Builtin (default): pattern injections with optional complexity flag inside `synthetic/txn_generate.py`.
- Simple (configurable): seasonality + rules. See `docs/synthetic_simple_generator.md`.
  - Example: `python ML_Fraud/run_txn_synth.py --generator simple --config ML_Fraud/configs/synth.yaml`

Quickstart (Transactions)
- Ingest sample CSV (headers must at least include transactionId,userId,amount,timestamp; optional: channel,deviceId,payeeId,city,is_fraud):
  - `python ML_Fraud/run_txn_ingest.py --input-csv path/to/txns.csv --prefix txn`
  - Outputs per‑txn JSON under `ML_Fraud/data/bronze/txn/` and `ML_Fraud/data/silver/txn/`.
- Build features from silver:
  - `python ML_Fraud/run_txn_features.py --silver-dir ML_Fraud/data/silver/txn --out ML_Fraud/data/gold/txn_features.csv`
- Generate synthetic dataset for prototyping:
  - `python ML_Fraud/run_txn_synth.py --users 300 --days 45 --avg-per-user 80 --fraud 0.25 --out-features ML_Fraud/data/gold/txn_synth_features.csv --out-records ML_Fraud/data/silver/txn_synth_records.json`

Feature Highlights
- Real‑time signals: time‑of‑day, day‑of‑week, is_night, channel encoding.
- Velocity: counts and sums over 60s/5m/1h/1d per user.
- Behavioral: device/payee/city novelty flags (first‑seen), amount z‑score over prior 30 days.
- Designed to combine with a supervised classifier (e.g., XGBoost) and an anomaly detector.

ADLS Integration
- All new CLIs accept `--to-adls` (features/synth) and reuse existing storage routing for uploads.



https://chatgpt.com/c/68c162a5-e304-8332-9a5d-874c8882c334
