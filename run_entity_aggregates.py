from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _load_silver_dir(silver_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(silver_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Normalize fields we rely on
    df["is_fraud"] = df.get("is_fraud", 0).fillna(0).astype(int)
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    df["ts"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    return df


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def build_device_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["deviceId"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["deviceId", "user_count", "txn_count", "fraud_rate", "first_seen", "last_seen"])  # noqa: E501
    g = d.groupby("deviceId")
    agg = pd.DataFrame({
        "user_count": g["userId"].nunique(),
        "txn_count": g.size(),
        "fraud_rate": g["is_fraud"].mean().fillna(0.0),
        "first_seen": g["ts"].min(),
        "last_seen": g["ts"].max(),
    }).reset_index()
    # Format timestamps as ISO strings
    for c in ("first_seen", "last_seen"):
        if c in agg.columns:
            agg[c] = pd.to_datetime(agg[c], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return agg


def build_merchant_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    m = df.dropna(subset=["merchantId"]).copy()
    if m.empty:
        return pd.DataFrame(columns=["merchantId", "txn_count", "avg_amount", "fraud_rate", "unique_users"])  # noqa: E501
    g = m.groupby("merchantId")
    agg = pd.DataFrame({
        "txn_count": g.size(),
        "avg_amount": g["amount"].mean().fillna(0.0),
        "fraud_rate": g["is_fraud"].mean().fillna(0.0),
        "unique_users": g["userId"].nunique(),
    }).reset_index()
    return agg


def build_user_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "userId", "txn_count", "avg_amount", "device_count", "payee_count", "merchant_count", "fraud_rate"
        ])
    g = df.groupby("userId", dropna=False)
    agg = pd.DataFrame({
        "txn_count": g.size(),
        "avg_amount": g["amount"].mean().fillna(0.0),
        "device_count": g["deviceId"].nunique(dropna=True),
        "payee_count": g["payeeId"].nunique(dropna=True),
        "merchant_count": g["merchantId"].nunique(dropna=True),
        "fraud_rate": g["is_fraud"].mean().fillna(0.0),
    }).reset_index()
    # Ensure numeric types
    for c in ("txn_count", "device_count", "payee_count", "merchant_count"):
        agg[c] = agg[c].astype(int)
    agg["avg_amount"] = agg["avg_amount"].astype(float)
    agg["fraud_rate"] = agg["fraud_rate"].astype(float)
    return agg


def build_payee_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    p = df.dropna(subset=["payeeId"]).copy()
    if p.empty:
        return pd.DataFrame(columns=["payeeId", "txn_count", "unique_users", "fraud_rate"])  # optional
    g = p.groupby("payeeId")
    agg = pd.DataFrame({
        "txn_count": g.size(),
        "unique_users": g["userId"].nunique(),
        "fraud_rate": g["is_fraud"].mean().fillna(0.0),
    }).reset_index()
    return agg


def _load_records_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"records_json not found: {path}")
        return pd.DataFrame()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
    except Exception as e:
        print(f"failed to parse records_json: {path} ({e})")
        return pd.DataFrame()
    if df.empty:
        return df
    df["is_fraud"] = df.get("is_fraud", 0).fillna(0).astype(int)
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    df["ts"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build entity-level aggregates from silver transactions")
    ap.add_argument("--silver-dir", default="data/silver/txn_synth", help="Directory with per-txn silver JSONs")
    ap.add_argument("--records-json", default=None, help="Optional JSON array of records (e.g., data/silver/txn_synth_records.json)")
    ap.add_argument("--out-dir", default="data/silver/entities", help="Output directory for entity CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()

    df = pd.DataFrame()
    if args.records_json:
        df = _load_records_json(Path(args.records_json).resolve())
    if df.empty:
        silver_dir = Path(args.silver_dir).resolve()
        if not silver_dir.exists():
            print(f"silver_dir not found: {silver_dir}")
        df = _load_silver_dir(silver_dir)
    if df.empty:
        print("No records found. Provide --records-json (JSON array) or a valid --silver-dir with JSON files.")
        return

    devices = build_device_aggregates(df)
    merchants = build_merchant_aggregates(df)
    users = build_user_aggregates(df)
    payees = build_payee_aggregates(df)

    _write_csv(devices, out_dir / "devices.csv")
    _write_csv(merchants, out_dir / "merchants.csv")
    _write_csv(users, out_dir / "users.csv")
    # Optional; not currently joined by features, but useful for analysis
    _write_csv(payees, out_dir / "payees.csv")

    print("Entity aggregates written:")
    print(f"  devices:   {out_dir / 'devices.csv'} ({len(devices)} rows)")
    print(f"  merchants: {out_dir / 'merchants.csv'} ({len(merchants)} rows)")
    print(f"  users:     {out_dir / 'users.csv'} ({len(users)} rows)")
    print(f"  payees:    {out_dir / 'payees.csv'} ({len(payees)} rows)")


if __name__ == "__main__":
    main()
