Model Training

Datasets
- Gold features: `ML_Fraud/data/gold/txn_synth_features.csv` (or features built from your silver JSONs).
- Target: `is_fraud` (0/1). Drop identifiers (`transactionId`, `userId`) and `event_ts` from features.

Baseline: XGBoost
- Script: `models/train_baseline.py` (invoked by `run_train_baseline.py`).
- Split: time‑aware (sort by `event_ts`, last 20% as test); fallback to stratified random split.
- Imbalance: `scale_pos_weight = neg/pos`.
- Search: small random search over trees, depth, LR, subsample, colsample, reg_lambda.
- Metrics: AUC‑ROC, AUC‑PR, precision/recall/F1 @0.5, precision@5%, capture@5%, FPR@0.5.
- Artifacts: model pickle, features list JSON, metrics JSON, thresholds recommendations JSON.

Ensemble: Stacked (XGB + IsolationForest + LR)
- Script: `models/train_ensemble.py` (invoked by `run_train_ensemble.py`).
- Outer forward folds: multiple forward time splits to evaluate params on future windows.
- Base learner: XGBClassifier (imbalance handled via `scale_pos_weight`).
- Anomaly channel: IsolationForest score ranked to [0,1].
- Meta learner: LogisticRegression on [xgb_proba, iforest_rank].
- Selection: random param trials; pick best by mean AUC‑PR across folds; fit final on all data.
- Outputs: same artifacts as baseline, plus PR curves JSON.

Threshold Recommendations
- Derived from PR curve: `max_f1`, target precision levels (e.g., ≥0.995, ≥0.99), and `top5pct` (score quantile).
- Saved to `<name>.thresholds.json` and used by serving as guidance for `policy.yaml`.

Registry & Champion
- Registry file: `data/models/registry.json` tracks runs and artifacts.
- Champion pointers: `data/models/champion/{model.pkl,features.json}` (symlinks locally; JSON pointers in ADLS mode).
- Promote: run training with `--promote` to update registry and champion.

CLI
- Baseline: `python ML_Fraud/run_train_baseline.py --gold-path ML_Fraud/data/gold/txn_synth_features.csv --promote`
- Ensemble: `python ML_Fraud/run_train_ensemble.py --gold-path ML_Fraud/data/gold/txn_synth_features.csv --outer-folds 3 --trials 30 --promote`

