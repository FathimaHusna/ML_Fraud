from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests


def _read_policy_threshold(path: Path) -> float | None:
    try:
        txt = path.read_text(encoding="utf-8")
        return float(txt.split("threshold:")[1].splitlines()[0].strip())
    except Exception:
        return None


def _now_iso_utc() -> str:
    return pd.Timestamp.utcnow().tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _pick_payee(hist: pd.DataFrame, new_payee: bool, rng: np.random.Generator) -> str:
    if new_payee:
        return f"pay-{int(rng.integers(9000, 9999))}"
    if "payeeId" in hist.columns and hist["payeeId"].notna().any():
        vals = hist["payeeId"].dropna().tolist()
        return str(vals[-1])
    return "pay-1001"


def _pick_device(user_id: str, hist: pd.DataFrame, new_device: bool, rng: np.random.Generator) -> str:
    if new_device:
        return f"dev-{user_id}-new-{int(rng.integers(100,999))}"
    if "deviceId" in hist.columns and hist["deviceId"].notna().any():
        vals = hist["deviceId"].dropna().tolist()
        return str(vals[-1])
    return f"dev-{user_id}-base"


def build_features_for_new(
    history: pd.DataFrame,
    new_txn: Dict[str, Any],
):
    # Lazy import to keep script self-contained
    from features.txn_features import build_txn_features

    df_raw = pd.concat([history, pd.DataFrame([new_txn])], ignore_index=True)
    feats = build_txn_features(df_raw)
    # Last row corresponds to the new transaction
    row = feats.iloc[-1].copy()
    if "is_fraud" in row.index:
        row = row.drop(labels=["is_fraud"])  # scoring expects pure features
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the scoring service with a new real-time transaction")
    ap.add_argument("--user", default="user_0001", help="User ID")
    ap.add_argument("--amount", type=float, default=250000.0, help="Transaction amount")
    ap.add_argument("--channel", default="mobile", help="Channel: mobile/web/ussd")
    ap.add_argument("--city", default="Colombo")
    ap.add_argument("--new-device", action="store_true", help="Mark device as new")
    ap.add_argument("--new-payee", action="store_true", help="Mark payee as new")
    ap.add_argument("--timestamp", default=None, help="ISO8601 UTC; default = now")
    ap.add_argument("--txid", default=None, help="Optional transactionId override")
    ap.add_argument("--history", default="ML_Fraud/data/silver/txn_synth_records.json", help="Path to history JSON array")
    ap.add_argument("--history-n", type=int, default=30, help="Number of recent events to use for history features")
    ap.add_argument("--service-url", default="http://localhost:8000/score", help="Scoring endpoint URL")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--print-features", action="store_true", help="Print computed features payload")
    ap.add_argument("--dry-run", action="store_true", help="Compute features but do not call the service")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    user_id = str(args.user)
    ts = args.timestamp or _now_iso_utc()

    # Load recent history for the user (optional but helps velocity/novelty features)
    hist_path = Path(args.history)
    hist = pd.DataFrame()
    if hist_path.exists():
        try:
            hist = pd.read_json(hist_path)
            hist = hist[hist.get("userId", "").eq(user_id)].sort_values("timestamp").tail(max(1, args.history_n))
        except Exception:
            hist = pd.DataFrame()

    device_id = _pick_device(user_id, hist, new_device=args.new_device, rng=rng)
    payee_id = _pick_payee(hist, new_payee=args.new_payee, rng=rng)
    txid = args.txid or f"TXNprobe-{user_id}-{int(rng.integers(10**6, 10**9))}"

    new_txn = {
        "transactionId": txid,
        "userId": user_id,
        "payeeId": payee_id,
        "merchantId": None,
        "merchantCategory": "electronics",
        "amount": float(args.amount),
        "timestamp": ts,
        "channel": str(args.channel).lower(),
        "deviceId": device_id,
        "deviceType": None,
        "deviceOs": None,
        "ip": None,
        "city": str(args.city),
        "region": None,
        "country": "LK",
        "lat": None,
        "lon": None,
        # label unknown for live scoring
        "is_fraud": None,
    }

    feat_row = build_features_for_new(hist, new_txn)
    if args.print_features:
        print(json.dumps(json.loads(feat_row.to_json()), indent=2))

    if args.dry_run:
        print(json.dumps({"note": "dry-run", "rows": 1}, indent=2))
        return 0

    body = {"rows": [json.loads(feat_row.to_json())]}
    try:
        r = requests.post(args.service_url, json=body, timeout=args.timeout)
        r.raise_for_status()
    except Exception as e:
        print(json.dumps({"error": str(e), "hint": "Ensure the scoring service is running on the given URL."}, indent=2))
        return 2

    resp = r.json()
    # Read local policy threshold (optional surface)
    local_thr = _read_policy_threshold(Path("ML_Fraud/configs/policy.yaml"))
    out = {
        "service_url": args.service_url,
        "request_txn": {k: new_txn[k] for k in ["transactionId", "userId", "amount", "timestamp", "channel", "deviceId", "payeeId", "city"]},
        "response": resp,
        "policy_threshold": local_thr,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

