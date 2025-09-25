Synthetic Dataset Validation

Purpose
- Sanity‑check synthetic data realism and feature separability before training.

Script
- File: `validation/validate_synth.py`
- Inputs: `--records` (JSON array), `--features` (CSV)
- Outputs: JSON report at `data/reports/synth_validation.json` + console summary.

Checks
- Schema sample: validates a random subset against Pydantic schema.
- Timestamps: invalid count, time range.
- Duplicates & nonpositive amounts.
- Labels: class ratio and optional target tolerance.
- Heuristics: counts velocity bursts, ATO‑like users, structuring, micro‑amount txns.
- Feature separability: per‑feature AUCs (or mean gaps if sklearn missing).
- Row alignment: features vs records row counts (≤2% mismatch).

Usage
- `python ML_Fraud/validation/validate_synth.py --records ML_Fraud/data/silver/txn_synth_records.json --features ML_Fraud/data/gold/txn_synth_features.csv --report-out ML_Fraud/data/reports/synth_validation.json`
- Add `--strict` to exit non‑zero if any check fails (for CI).

Interpreting Failures
- Heuristics low → increase pattern probabilities (e.g., `card_testing`, `velocity_burst`).
- Poor feature AUC → revisit feature windows or z‑score context; ensure joins are present.
- Row mismatch → check for invalid timestamps getting dropped during feature build.

