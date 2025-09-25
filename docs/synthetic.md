Synthetic Data Generation

Generators
- Builtin (`synthetic/txn_generate.py`): richer scenarios with complexity flag; injects patterns like odd-hour/new-city, velocity bursts, ATO sequences, structuring.
- Simple (`synthetic/simple_generator.py`): rule‑based, fast, configurable via YAML; supports seasonality and patterns including `velocity_burst`, `new_payee_high_amt`, `odd_hour_new_city`, `card_testing` (micro-amount bursts).

Configuration (simple)
- File: `configs/synth.yaml` — set fraud rate, seasonality, and pattern probabilities/params.
- Example parameters: `daily_amp`, `weekly_amp`, `burst_lambda`, `window_s`, `z_min`, `multiplier_min/max`, `night_end_hour`, `n_min/max` for micro‑bursts.

How It Works
- User profiles: city, two devices, payees, lognormal amount parameters, channel mix, merchant preferences.
- Seasonality: non‑homogeneous arrivals via rejection sampling with hour/day sinusoidal weights.
- Amounts: lognormal for strictly positive, heavy‑tailed behavior.
- Fraud selection: randomly choose candidate transactions at configured rate; apply one pattern by probability.
- Micro‑amount (“card testing”): shrinks to 5–50 LKR range, adds K near‑time clones within a window.

Outputs
- Records JSON: `data/silver/txn_synth_records.json` — full events with `is_fraud` and `scenario`.
- Features CSV: `data/gold/txn_synth_features.csv` — numeric features for modeling.

Validation
- Script: `validation/validate_synth.py` checks schema sample, timestamps, duplicates, fraud ratio, heuristic counts (bursts/ATO/struct/micro), and feature separability (AUCs).
- Goal: All checks pass; adjust `configs/synth.yaml` or generator params to reach targets.

Limitations
- Not a full behavioral simulator (no coordinated fraud rings/graphs).
- Seasonality approximate; good enough for prototyping and feature testing.

CLI
- Builtin: `python ML_Fraud/run_txn_synth.py --users 300 --days 30 --avg-per-user 80 --fraud 0.2`
- Simple: `python ML_Fraud/run_txn_synth.py --generator simple --config ML_Fraud/configs/synth.yaml`

