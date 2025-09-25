Statistical Concepts & Rationale

Class Imbalance
- Fraud is rare; optimizing for overall accuracy or ROC AUC can be misleading.
- Prefer AUC‑PR and report precision@K, capture@K, and FPR at operating thresholds.

Precision–Recall & Operating Points
- Precision: TP / (TP + FP); Recall: TP / (TP + FN).
- Threshold choices: max‑F1, target precision, or top‑K% alert rate; choose based on analyst capacity and business risk.

Isolation Forest (Anomaly Score)
- Unsupervised outlier detector; trained on non‑fraud subset (or all data if labels missing).
- `score_samples` → larger is less anomalous; convert to rank‑based [0,1] anomaly score for stability.

Velocity Features
- Sliding windows (60s..30d) of counts and sums; correlate with bursts and structuring.
- Efficient computation via prefix sums and binary search over sorted timestamps.

Z‑Score Context
- Per‑user 30‑day mean/std; `z = (amount - mean) / std` flags unusual amounts.
- Guard against std=0; treat as NaN/0 to avoid numerical blow‑ups.

Cyclical Encoding
- Sine/Cosine transforms capture periodic nature of hour (24) and day‑of‑week (7).

Synthetic Arrivals & Amounts
- Arrivals: approximate non‑homogeneous Poisson using rejection sampling with sinusoidal weights.
- Amounts: lognormal to reflect heavy‑tailed spending (many small, few large amounts).

Time‑Aware Validation
- Forward splits respect temporal leakage; evaluate models on future windows only.

