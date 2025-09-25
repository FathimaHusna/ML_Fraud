FAQ

How do I generate a quick synthetic dataset?
- `python ML_Fraud/run_txn_synth.py --generator simple --config ML_Fraud/configs/synth.yaml`

How do I compute features from my own silver JSONs?
- `python ML_Fraud/run_txn_features.py --silver-dir ML_Fraud/data/silver/txn --out ML_Fraud/data/gold/txn_features.csv`

Which features should I drop before training?
- Drop identifiers and `event_ts` (kept for splitting): `transactionId`, `userId`, `event_ts`.

How do I choose a threshold?
- Check `<model>.thresholds.json` for `max_f1`, `top5pct`, and precision targets. Set `configs/policy.yaml` and `POST /reload`.

How do I batch score a CSV of features?
- `python ML_Fraud/scoring/batch_score.py --in-path <features.csv> --model-path <model.pkl> --features-json <features.json> --out-path <scored.csv> --threshold <t>`

How do I simulate a real‑time transaction?
- `python ML_Fraud/pilot/realtime_probe.py --user user_0001 --amount 250000 --new-device --new-payee`

How do I deploy quickly?
- Docker: `docker build -t ml-fraud-svc -f ML_Fraud/scoring/Dockerfile . && docker run -p 8000:8000 ml-fraud-svc`
- Compose: `docker compose up --build`

How do I use ADLS?
- Set `STORAGE_MODE=adls` and either `AZURE_STORAGE_CONNECTION_STRING` or `STORAGE_ACCOUNT_URL` (+ identity). Use blob paths/URLs for MODEL_PATH/FEATURES_JSON.

