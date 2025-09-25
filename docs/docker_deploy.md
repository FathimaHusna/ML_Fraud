Docker Deployment – ML_Fraud Scoring Service

Quick Start (Local Image)
- Build: `docker build -t ml-fraud-svc -f ML_Fraud/scoring/Dockerfile .`
- Run: `docker run -p 8000:8000 ml-fraud-svc`
  - Defaults to local artifacts inside the image:
    - `MODEL_PATH=ML_Fraud/data/models/stack_v1.pkl`
    - `FEATURES_JSON=ML_Fraud/data/models/stack_v1.features.json`

With Docker Compose (recommended for protos)
- One command: `docker compose up --build`
- What it does:
  - Publishes `8000` → `8000`
  - Mounts `ML_Fraud/configs/policy.yaml` (edit threshold live, then POST `/reload`)
  - Mounts `ML_Fraud/data/models` (swap models without rebuild)
  - Sets STORAGE_MODE=local and explicit model paths (avoids champion symlink issues)

Environment Overrides
- `MODEL_PATH`: set to a specific artifact (avoid champion symlink), e.g., `ML_Fraud/data/models/stack_v1.pkl`
- `FEATURES_JSON`: expected features list JSON
- `THRESHOLD`: optionally override the policy file threshold
- `STORAGE_MODE=adls`: switch to ADLS if you host artifacts in Azure Blob
  - With ADLS, set either `AZURE_STORAGE_CONNECTION_STRING` or `STORAGE_ACCOUNT_URL` (+ workload identity)

Smoke Tests
- Health: `curl http://localhost:8000/healthz`
- Score sample: `PYTHONPATH=ML_Fraud python ML_Fraud/pilot/smoke_score.py`
- Realtime probe: `PYTHONPATH=ML_Fraud python ML_Fraud/pilot/realtime_probe.py --user user_0001 --amount 250000 --new-device --new-payee`
- Metrics: `curl http://localhost:8000/metrics`

Updating Policy/Model
- Update `ML_Fraud/configs/policy.yaml` → `curl -X POST http://localhost:8000/reload`
- Replace artifacts under `ML_Fraud/data/models` (compose mounts this path) → `curl -X POST /reload`

Notes
- Champion symlinks are not portable across builds; prefer explicit `MODEL_PATH`/`FEATURES_JSON` to concrete files.
- If you don’t need live edits, you can remove the volumes from `docker-compose.yml` and rely on the baked artifacts.
