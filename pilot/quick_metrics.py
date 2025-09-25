from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _read_policy_threshold(path: Path) -> float:
    try:
        txt = path.read_text(encoding="utf-8")
        # Tip: simple split without YAML dependency
        return float(txt.split("threshold:")[1].splitlines()[0].strip())
    except Exception:
        return 0.5


def main() -> None:
    feat_path = Path("ML_Fraud/data/gold/txn_synth_features.csv")
    scored_path = Path("ML_Fraud/data/gold/txn_synth_scored.csv")
    policy_path = Path("ML_Fraud/configs/policy.yaml")

    missing = [str(p) for p in [feat_path, scored_path] if not p.exists()]
    if missing:
        print(json.dumps({
            "error": "missing input",
            "missing": missing,
            "hint": "Run batch_score to produce ML_Fraud/data/gold/txn_synth_scored.csv",
        }, indent=2))
        return

    f = pd.read_csv(feat_path)
    s = pd.read_csv(scored_path)
    df = pd.concat([f.reset_index(drop=True), s[["score", "decision"]]], axis=1)

    y = df["is_fraud"].astype(int).to_numpy()
    pred = df["decision"].astype(int).to_numpy()
    precision = float(y[pred == 1].mean() if (pred == 1).any() else 0.0)
    recall = float(pred[y == 1].mean() if (y == 1).any() else 0.0)
    pos_rate = float(pred.mean())
    print(json.dumps({
        "mode": "as_scored",
        "precision": precision,
        "recall": recall,
        "pos_rate": pos_rate,
    }, indent=2))

    # Evaluate at current policy threshold (from policy.yaml) using scores
    thr = _read_policy_threshold(policy_path)
    yhat = (s["score"] >= thr).astype(int).to_numpy()
    precision2 = float(y[yhat == 1].mean() if (yhat == 1).any() else 0.0)
    recall2 = float(yhat[y == 1].mean() if (y == 1).any() else 0.0)
    pos_rate2 = float(yhat.mean())
    print(json.dumps({
        "mode": "policy_threshold",
        "threshold": thr,
        "precision": precision2,
        "recall": recall2,
        "pos_rate": pos_rate2,
    }, indent=2))


if __name__ == "__main__":
    main()

