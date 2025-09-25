Feature Engineering

Source
- Input: silver JSON rows with normalized fields (transactionId, userId, amount, timestamp, channel, deviceId, payeeId, city, …).
- Builder: `features/txn_features.py` → `build_txn_features(df)`.

Time & Cyclical
- hour, day‑of‑week, `is_night` (hour<5 or ≥23)
- sin/cos encodings for hour/dow capture periodicity.

Velocity Windows (per user)
- Counts and sums over rolling windows: 60s, 5m, 1h, 1d, 7d, 30d.
- Efficient computation using sorted timestamps, searchsorted, and prefix sums (O(N) per user per window set).

Behavioral Novelty
- `is_new_device`, `is_new_payee`, `is_new_city` via first‑seen flags within the user’s history.
- `time_since_last_s` since last transaction.

Amount Context (30‑day)
- Per‑user rolling mean/std (excluding current), z‑score of current amount (`z_amt_30d`).
- Handles degenerate std=0 by returning NaN or 0 appropriately; downstream filled to 0 when modeling.

Channel Encoding
- One‑hot of channel (web/mobile/ussd → `ch_*`). Missing values become `ch_unknown` (if present) or zeros.

Anomaly Signal (optional)
- IsolationForest fitted on non‑fraud subset (if labels present), ranked to [0,1] with higher = more anomalous (`iforest_score`). Fails open if sklearn absent.

Entity Joins (optional)
- Joins against `data/silver/entities/*` for device user count, merchant txn counts, etc., if present.

Labels & Event Time
- Preserves `is_fraud` when present for supervised training.
- Adds `event_ts` (epoch seconds) for time‑aware splitting downstream.

Output
- Flat numeric DataFrame with consistent order; infinities/NaNs replaced by 0 in training.
- CLI: `run_txn_features.py` writes CSV; can upload to ADLS.

