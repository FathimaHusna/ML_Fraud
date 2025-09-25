Synthetic Data – Simple Rule‑Based Generator

Overview
- Purpose: Lightweight, fast synthetic transaction generator with clear knobs and reproducible behavior.
- Approach: Normal behavior via seasonality‑modulated arrivals and lognormal amounts; fraud injected by configurable rules.
- Where: `synthetic/simple_generator.py` with CLI routing through `run_txn_synth.py --generator simple`.

Quick Start
- Default config:
  - `python ML_Fraud/run_txn_synth.py --generator simple --config ML_Fraud/configs/synth.yaml`
- Minimal overrides (no YAML):
  - `python ML_Fraud/run_txn_synth.py --generator simple --users 300 --days 30 --avg-per-user 80 --fraud 0.02`
- Outputs:
  - Features CSV: `ML_Fraud/data/gold/txn_synth_features.csv` (numeric model features)
  - Records JSON: `ML_Fraud/data/silver/txn_synth_records.json` (full events incl. labels and scenarios)

Config Schema (YAML)
- generator: must be `simple` to select this path.
- base:
  - fraud_rate: target fraction of events to label/inject as fraud (0–1).
  - start: optional ISO8601 UTC start timestamp or date (e.g., `"2025-01-01"`). If omitted, the window defaults to the last `days` relative to now.
- seasonality:
  - daily_amp: amplitude for hour‑of‑day rhythm (0–1 typical).
  - weekly_amp: amplitude for day‑of‑week rhythm (0–1 typical).
- patterns:
  - velocity_burst: { prob, burst_lambda, window_s }
  - new_payee_high_amt: { prob, z_min, multiplier_min, multiplier_max }
  - odd_hour_new_city: { prob, night_end_hour }

Example
  generator: simple
  base:
    fraud_rate: 0.02
    start: "2025-01-01"
  seasonality:
    daily_amp: 0.3
    weekly_amp: 0.2
  patterns:
    velocity_burst:
      prob: 0.005
      burst_lambda: 8
      window_s: 120
    new_payee_high_amt:
      prob: 0.004
      z_min: 2.5
      multiplier_min: 5.0
      multiplier_max: 12.0
    odd_hour_new_city:
      prob: 0.004
      night_end_hour: 4

Data Model
- Records JSON fields: transactionId, userId, payeeId, amount, timestamp (UTC ISO), channel, deviceId, city, is_fraud, scenario (string), fraud_score_sim (0–1 float).
- Features CSV fields: engineered numeric features for modeling; it does not include `scenario` or `fraud_score_sim`.

How It Works
- Users: Each user gets a profile (home city, 2 devices, payees, lognormal amount parameters, channel mix).
- Arrivals (seasonality): For each per‑user transaction, a timestamp is sampled with a probability proportional to
  1 + daily_amp*sin(2π*hour/24) + weekly_amp*sin(2π*dow/7). This concentrates activity at busier hours/days.
- Amounts (lognormal): Positive, skewed distribution matching real spending (many small, few large).
- Fraud selection: A global pool of transactions is sampled with probability `fraud_rate` to become “candidates”.
- Scenario choice: Each candidate picks a pattern by configured probabilities and applies its transformation.

Fraud Patterns
- velocity_burst
  - Idea: Card testing or rapid micro‑transactions.
  - Params: burst_lambda (expected extra txns), window_s (time window).
  - Effect: Shrinks base amount and appends K≈Poisson(burst_lambda) near‑time clones.
  - Severity: fraud_score_sim ≈ 0.4 + 0.02*K capped at 1.
- new_payee_high_amt
  - Idea: New beneficiary plus unusually high amount.
  - Params: z_min (min z‑score above user’s mean), multiplier range.
  - Effect: Switch payee to new, scale amount; ensure z‑score ≥ z_min.
  - Severity: Maps z‑score to ~0.5–1.0.
- odd_hour_new_city
  - Idea: Night‑time transaction from atypical location and device.
  - Params: night_end_hour (exclusive upper hour for “night”).
  - Effect: Snap to 0..night_end_hour, change city, hijacked device.
  - Severity: ~0.6 baseline.

Outputs and Validation
- scenario: Helps audit which rule produced the label; enables per‑scenario metrics.
- fraud_score_sim: Quick proxy for difficulty/severity; can be used for threshold sweeps.
- Sanity checks (suggested):
  - Realized fraud rate ≈ configured fraud_rate.
  - Distribution of scenarios ≈ configured probabilities (allowing variance).
  - Seasonality: more txns during peak hours/days than off‑peak.

Intuition and Terminology
- Poisson arrivals: A simple model for counts of independent events in time; here, we approximate a non‑homogeneous Poisson process by seasonality‑weighted sampling (busy hours get sampled more often).
- Rejection sampling: Draw a time uniformly, accept it with probability proportional to its seasonal weight; repeat if rejected.
- Lognormal amounts: If log(amount) is roughly normal, amount is lognormal—captures “mostly small, occasionally large” behavior and is strictly positive.
- z‑score: (x − mean) / std. Measures how extreme an amount is relative to the user’s recent history.
- Seasonality amplitudes: daily_amp, weekly_amp control how pronounced the cycles are. 0.0 = flat; 0.5 = moderate peaks; ≥1.0 is very peaky.

Extending
- Add a new pattern:
  1) Implement a function `_apply_<name>(records, idx, rng, params, profile)` that edits the base row and/or appends neighbors.
  2) Register it in `generate_simple()` switch and add a `patterns.<name>` block in YAML with a `prob`.

CLI Flags Recap
- --generator {builtin|simple}: choose the engine.
- --config <YAML>: optional parameters for the simple generator.
- --users, --days, --avg-per-user, --fraud, --seed: global controls; `--fraud` overrides base.fraud_rate in YAML.

Limitations
- Seasonality is approximate (not a true NHPP); good enough for prototyping.
- No coordinated fraud rings or graph dynamics; use the builtin complex mode or a graph simulator for that.

Notes on Dependencies
- The simple generator depends on NumPy and pandas. YAML parsing is optional; if PyYAML is not installed, the generator uses internal defaults and CLI overrides. ADLS uploads (optional `--to-adls`) require Azure SDKs but do not affect local generation.
