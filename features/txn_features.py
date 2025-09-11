from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _parse_ts(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")


def _is_night(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    h = ts.tz_convert("UTC").hour if ts.tzinfo else ts.hour
    return int(h < 5 or h >= 23)


def _encode_channel(s: pd.Series) -> pd.DataFrame:
    vals = s.fillna("unknown").str.lower()
    return pd.get_dummies(vals, prefix="ch")


def _group_time_windows(df_u: pd.DataFrame, windows: List[pd.Timedelta]) -> Dict[str, List[int]]:
    # Assumes df_u sorted by timestamp ascending and has columns ["ts", "amount", "deviceId", "payeeId", "city"]
    # Convert tz-aware timestamps to epoch ns int64
    ts = df_u["ts"].astype("int64").to_numpy()
    amt = df_u["amount"].to_numpy()
    out_counts: Dict[str, List[int]] = {}
    out_sums: Dict[str, List[float]] = {}
    for w in windows:
        key_c = f"cnt_{int(w.total_seconds())}s"
        key_s = f"sum_{int(w.total_seconds())}s"
        counts: List[int] = []
        sums: List[float] = []
        w_ns = int(w.total_seconds() * 1e9)
        for i in range(len(ts)):
            t0 = ts[i] - w_ns
            # previous only (exclude current i)
            j = np.searchsorted(ts, t0, side="left")
            counts.append(max(0, i - j))
            sums.append(float(amt[j:i].sum()))
        out_counts[key_c] = counts
        out_sums[key_s] = sums
    return {**out_counts, **out_sums}


def _first_seen_flags(df_u: pd.DataFrame, col: str) -> List[int]:
    # 1 if first time this value appears for the user
    first_idx = df_u.groupby(col).cumcount()
    return (first_idx == 0).astype(int).tolist()


def build_txn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transaction features matching the product document themes.

    Requires columns: transactionId, userId, amount, timestamp, channel, deviceId, payeeId, city
    Optional label: is_fraud
    """
    df = df.copy()
    df["ts"] = _parse_ts(df["timestamp"]).dt.tz_convert("UTC")
    # Drop rows with invalid timestamps to avoid overflow in rolling window math
    df = df[~df["ts"].isna()].copy()
    df.sort_values(["userId", "ts", "transactionId"], inplace=True)

    # Basic time features
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_night"] = df["ts"].apply(_is_night)
    df["log_amount"] = np.log1p(df["amount"].astype(float))

    # Per-user rolling stats (30d window for zscore)
    def _user_stats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        windows = [pd.Timedelta(seconds=60), pd.Timedelta(minutes=5), pd.Timedelta(hours=1), pd.Timedelta(days=1)]
        tw = _group_time_windows(g, windows)
        for k, v in tw.items():
            g[k] = v
        # device/payee novelty
        g["is_new_device"] = _first_seen_flags(g, "deviceId")
        g["is_new_payee"] = _first_seen_flags(g, "payeeId")
        g["is_new_city"] = _first_seen_flags(g, "city")
        # 30d mean/std excluding current
        # Build 30d rolling using searchsorted for mean/std
        ts = g["ts"].astype("int64").to_numpy()
        amt = g["amount"].astype(float).to_numpy()
        w_ns = int(pd.Timedelta(days=30).total_seconds() * 1e9)
        mean_30: List[float] = []
        std_30: List[float] = []
        for i in range(len(ts)):
            t0 = ts[i] - w_ns
            j = np.searchsorted(ts, t0, side="left")
            win = amt[j:i]
            if len(win) == 0:
                mean_30.append(np.nan)
                std_30.append(np.nan)
            else:
                mean_30.append(float(np.mean(win)))
                std_30.append(float(np.std(win, ddof=1)) if len(win) > 1 else 0.0)
        g["mean_amt_30d"] = mean_30
        g["std_amt_30d"] = std_30
        # z-score w.r.t. prior 30d
        def _z(a, m, s):
            if np.isnan(m) or s == 0.0:
                return np.nan
            return (a - m) / (s if s > 0 else 1.0)
        g["z_amt_30d"] = [
            _z(float(a), float(m) if not pd.isna(m) else np.nan, float(s) if not pd.isna(s) else 0.0)
            for a, m, s in zip(g["amount"], g["mean_amt_30d"], g["std_amt_30d"])
        ]
        return g

    # Exclude grouping columns inside apply to align with future pandas behavior,
    # then restore userId from the group index.
    df = df.groupby("userId", group_keys=True).apply(_user_stats, include_groups=False)
    df = df.reset_index(level=0)

    # Channel encoding
    ch = _encode_channel(df["channel"])
    feat = pd.concat([
        df[[
            "transactionId",
            "userId",
            "amount",
            "log_amount",
            "hour",
            "dow",
            "is_night",
            "cnt_60s",
            "cnt_300s",
            "cnt_3600s",
            "cnt_86400s",
            "sum_60s",
            "sum_300s",
            "sum_3600s",
            "sum_86400s",
            "is_new_device",
            "is_new_payee",
            "is_new_city",
            "mean_amt_30d",
            "std_amt_30d",
            "z_amt_30d",
        ]],
        ch,
    ], axis=1)

    # Preserve label if present
    if "is_fraud" in df.columns:
        feat["is_fraud"] = df["is_fraud"].astype("Int64")

    return feat


def features_from_silver_dir(silver_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(silver_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return build_txn_features(df)


@dataclass
class TxnFeatureJobConfig:
    silver_dir: Path
    out_path: Path


def run_txn_feature_job(cfg: TxnFeatureJobConfig) -> Tuple[pd.DataFrame, Path]:
    feat = features_from_silver_dir(cfg.silver_dir)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(cfg.out_path, index=False)
    return feat, cfg.out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute features for e-payment transactions")
    parser.add_argument("--silver-dir", default="./ML_Fraud/data/silver/txn", help="Directory with txn silver JSON files")
    parser.add_argument("--out", default="./ML_Fraud/data/gold/txn_features.csv", help="Output CSV path for features")
    args = parser.parse_args()

    cfg = TxnFeatureJobConfig(silver_dir=Path(args.silver_dir).resolve(), out_path=Path(args.out).resolve())
    df, outp = run_txn_feature_job(cfg)
    print(f"Wrote txn features: {outp} ({len(df)} rows)")
