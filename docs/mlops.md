MLOps: Experiments, AutoML, Retraining

Experiments with MLflow
- Tracking URI: local – set `MLFLOW_TRACKING_URI=./mlruns`.
- What to log: params (XGB/stack), metrics (AUC‑PR/ROC, precision@5%, capture@5%, FPR@0.5), artifacts (metrics.json, thresholds.json, features.json, policy.yaml, model.pkl), tags (data_path, git_commit).
- UI: `mlflow ui --backend-store-uri ./mlruns -p 5000`.
- Hosted option: DagsHub – set MLflow URI to `https://dagshub.com/<user>/<repo>.mlflow` and authenticate.

AutoML Options
- Local: FLAML/AutoGluon – quick sweeps on current features; implement forward time splits manually.
- Azure ML AutoML: managed sweeps; point to Gold dataset in ADLS or registered dataset; use forward validation windows (multiple jobs or custom CV).
- Metric: optimize AUC‑PR; ensure class imbalance handling.

Retraining Pipeline (prototype)
1) Generate or ingest fresh silver; build Gold features.
2) Run training (baseline/ensemble or AutoML sweep).
3) Compute thresholds; compare against champion on a forward holdout window.
4) Promote challenger only if win criteria met; update registry and champion.
5) Update policy.yaml as needed; call `POST /reload`.

Scheduling
- Cron/CI weekly job: synth/ingest → features → train → evaluate → (optional) promote.
- Track data fingerprints (hash of Gold CSV) to ensure reproducibility.

Drift & Monitoring
- Serve `/metrics`: latency and recent high-rate; add score drift and PSI if needed.
- Periodically recompute performance on a labeled window; alert on capture@K drops or FPR increases.

