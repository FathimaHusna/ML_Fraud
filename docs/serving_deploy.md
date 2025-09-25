Serving & Deployment

Service
- File: `scoring/service.py` (FastAPI)
- Endpoints:
  - `GET /healthz` – liveness/readiness
  - `POST /score` – body: `{ "rows": [ {feature: value, ...} ], "threshold": optional }`
  - `GET /metrics` – basic latency and score distribution stats
  - `POST /reload` – reload model/policy
  - `GET /thresholds` – show recommended thresholds JSON if available

Configuration
- MODEL_PATH / FEATURES_JSON: either local file paths (preferred for Docker) or ADLS blob paths/URLs.
- STORAGE_MODE: `local` (default) or `adls`.
- THRESHOLD env (optional): overrides when policy.yaml absent.
- policy.yaml: `configs/policy.yaml` (threshold + action), mounted or baked into image.

Local Run
- `PYTHONPATH=ML_Fraud uvicorn ML_Fraud.scoring.service:app --host 0.0.0.0 --port 8000`

Docker (single container)
- Build: `docker build -t ml-fraud-svc -f ML_Fraud/scoring/Dockerfile .`
- Run: `docker run -p 8000:8000 ml-fraud-svc`
- See also: `docs/docker_deploy.md` and root `docker-compose.yml` for live policy/model mounts.

Azure Options
- App Service for Containers: simplest managed hosting (image only; local artifacts baked in).
- Azure Container Apps: managed container, external ingress, can use ADLS for model artifacts.
- AKS (Kubernetes): for scaling, RBAC, and identity; configure STORAGE_ACCOUNT_URL and managed identity for ADLS.

Security Notes
- Add auth (gateway or API key) if exposed beyond internal networks.
- Keep secrets in Key Vault or platform secrets; prefer managed identity for ADLS.
- Limit POST body size and validate inputs if moving toward production.

