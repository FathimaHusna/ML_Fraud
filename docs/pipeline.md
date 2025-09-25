Data Pipeline

Layers
- Bronze: raw per‑transaction JSON (one file per event) as ingested.
- Silver: normalized per‑transaction JSON (consistent field names/types for feature builders).
- Gold: tabular features (CSV) used for modeling/scoring.

Ingestion
- Schema: `ingestion/txn_schema.py` defines `TxnRaw` (bronze) and normalized mapping to silver.
- CLI: `run_txn_ingest.py` supports `--input-csv` or `--input-json`; writes `data/bronze/<prefix>/` and `data/silver/<prefix>/`.
- Storage: `ingestion/storage.py` routes to local filesystem (default) or ADLS when `STORAGE_MODE=adls`.

Entity Aggregates
- Script: `run_entity_aggregates.py` builds users/devices/merchants/payees summaries from silver or records JSON.
- Outputs: `data/silver/entities/{users,devices,merchants,payees}.csv` (used optionally by feature joins).

Gold Features
- Builder: `features/txn_features.py` reads silver JSONs or takes a DataFrame and emits engineered numeric features.
- CLI: `run_txn_features.py` writes `data/gold/txn_features.csv`; can optionally upload to ADLS.

Storage Modes
- Local: default; root under `DATA_ROOT` (defaults to `./ML_Fraud/data`).
- ADLS: set `STORAGE_MODE=adls` and provide either `AZURE_STORAGE_CONNECTION_STRING` or `STORAGE_ACCOUNT_URL` (+ identity); all writes go to Blob via `ingestion/azure_blob.py`.

Conventions
- IDs: `transactionId`, `userId`, `deviceId`, `payeeId`; timestamps are ISO8601 UTC.
- Partitioning: simple flat folders; time partitioning can be added later if needed.

