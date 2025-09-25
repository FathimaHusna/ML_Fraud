from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    feats_path = Path("ML_Fraud/data/gold/txn_synth_features.csv")
    if not feats_path.exists():
        raise SystemExit(f"Features file not found: {feats_path}")
    df = pd.read_csv(feats_path).head(5)
    if "is_fraud" in df.columns:
        df = df.drop(columns=["is_fraud"])  # service expects features only
    # Replace NaNs/Infs to make JSON serializable and match service preprocessing
    df = df.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    rows = df.to_dict(orient="records")
    resp = requests.post("http://localhost:8000/score", json={"rows": rows})
    print(resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
