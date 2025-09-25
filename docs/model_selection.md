Model Selection & Thresholding

Metrics
- AUC‑PR: area under precision‑recall; robust for class imbalance; primary selection metric.
- AUC‑ROC: secondary; can be misleading with heavy imbalance.
- precision@K%: share of positives among top‑K% scores.
- capture@K%: share of all frauds captured in top‑K%.
- FPR@t: false positive rate at a probability threshold t.

Threshold Policies
- max‑F1: pick threshold that maximizes F1 (2*P*R/(P+R)); balances precision & recall.
- target precision: choose smallest threshold achieving precision ≥ X%.
- top‑K%: pick score quantile corresponding to desired alert rate.
- policy.yaml: single source of truth for serving; can be reloaded without restart via `/reload`.

Champion–Challenger
- Keep current serving model as champion; train challengers and compare on a forward holdout window.
- Promote only if challenger improves AUC‑PR and meets operating constraints (e.g., FPR cap, capture@K target).

Operational Trade‑offs
- High precision, low recall → small alert load, low analyst burden (prototype default).
- Lower threshold → higher recall and alert rate; track FPR and team capacity.
- Monitor drift: watch average scores, recent_high_rate from `/metrics`.

Artifacts
- `<name>.thresholds.json`: recommended thresholds summary.
- `configs/policy.yaml`: chosen threshold and action.

