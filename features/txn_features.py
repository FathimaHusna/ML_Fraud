from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path


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
    # Use exclusive prefix sums for O(1) range-sum queries over [j, i-1]
    # psum[k] = sum(amt[0..k-1]), so sum(j..i-1) = psum[i] - psum[j]
    psum = np.concatenate(([0.0], np.cumsum(amt, dtype=float)))
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
            j = int(np.searchsorted(ts, t0, side="left"))
            cnt = max(0, i - j)
            counts.append(cnt)
            # Range sum via prefix sums (handles j==i -> 0)
            sums.append(float(psum[i] - psum[j]))
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
    # Numeric event timestamp (seconds) for temporal splitting downstream
    df["event_ts"] = (df["ts"].astype("int64") / 1e9).astype(float)

    # Basic time features
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["is_night"] = df["ts"].apply(_is_night)
    df["log_amount"] = np.log1p(df["amount"].astype(float))
    # Cyclical encodings for periodicity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Per-user rolling stats (30d window for zscore)
    def _user_stats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        windows = [
            pd.Timedelta(seconds=60),
            pd.Timedelta(minutes=5),
            pd.Timedelta(hours=1),
            pd.Timedelta(days=1),
            pd.Timedelta(days=7),
            pd.Timedelta(days=30),
        ]
        tw = _group_time_windows(g, windows)
        for k, v in tw.items():
            g[k] = v
        # device/payee novelty
        g["is_new_device"] = _first_seen_flags(g, "deviceId")
        g["is_new_payee"] = _first_seen_flags(g, "payeeId")
        g["is_new_city"] = _first_seen_flags(g, "city")
        # time since last txn (seconds)
        ts_ns = g["ts"].astype("int64").to_numpy()
        tsl: List[float] = []
        for i in range(len(ts_ns)):
            if i == 0:
                tsl.append(np.nan)
            else:
                tsl.append(float(ts_ns[i] - ts_ns[i - 1]) / 1e9)
        g["time_since_last_s"] = tsl
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

    # Apply per-user feature computation; keep original columns
    df = df.groupby("userId", group_keys=False).apply(_user_stats)

    # Channel encoding
    ch = _encode_channel(df["channel"])
    feat = pd.concat([
        df[[
            "transactionId",
            "userId",
            "event_ts",
            "amount",
            "log_amount",
            "hour",
            "dow",
            "is_night",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "cnt_60s",
            "cnt_300s",
            "cnt_3600s",
            "cnt_86400s",
            "cnt_604800s",
            "cnt_2592000s",
            "sum_60s",
            "sum_300s",
            "sum_3600s",
            "sum_86400s",
            "sum_604800s",
            "sum_2592000s",
            "is_new_device",
            "is_new_payee",
            "is_new_city",
            "time_since_last_s",
            "mean_amt_30d",
            "std_amt_30d",
            "z_amt_30d",
        ]],
        ch,
    ], axis=1)

    # Preserve label if present
    if "is_fraud" in df.columns:
        feat["is_fraud"] = df["is_fraud"].astype("Int64")

    # Optional anomaly score via Isolation Forest (unsupervised)
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        # Build training mask: use only known non-fraud if labels exist; else use all
        if "is_fraud" in feat.columns and feat["is_fraud"].notna().any():
            mask = (feat["is_fraud"].fillna(0) == 0)
        else:
            mask = pd.Series([True] * len(feat), index=feat.index)

        drop_cols = {"transactionId", "userId", "is_fraud"}
        num_cols = [c for c in feat.columns if c not in drop_cols]
        X_all = feat[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_train = X_all[mask]

        if len(X_train) >= 10:  # avoid fitting on too-few rows
            iforest = IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
                n_jobs=-1,
            )
            iforest.fit(X_train)
            # Higher score = more anomalous (invert score_samples)
            raw = iforest.score_samples(X_all)
            # score_samples: larger is less anomalous; convert to anomaly_score in [0,1] via rank-normalize
            ranks = pd.Series(raw).rank(method="average") / len(raw)
            anomaly = 1.0 - ranks
            feat["iforest_score"] = anomaly.astype(float).values
    except Exception:
        # If sklearn not installed or any failure, skip without breaking pipeline
        pass

    # Optional joins with entity profiles and external signals (if materialized)
    try:
        # Users
        users_csv = Path("ML_Fraud/data/silver/entities/users.csv")
        if users_csv.exists():
            u = pd.read_csv(users_csv)
            cols = [c for c in u.columns if c not in {"first_seen", "last_seen"}]
            feat = feat.merge(u[cols], on="userId", how="left")
        # Devices
        dev_csv = Path("ML_Fraud/data/silver/entities/devices.csv")
        if dev_csv.exists() and "deviceId" in df.columns:
            d = pd.read_csv(dev_csv)
            feat = feat.join(df[["deviceId"]])
            feat = feat.merge(d[["deviceId", "user_count", "txn_count"]], on="deviceId", how="left")
            feat.rename(columns={
                "user_count": "device_user_count",
                "txn_count": "device_txn_count",
            }, inplace=True)
        # Merchants
        mer_csv = Path("ML_Fraud/data/silver/entities/merchants.csv")
        if mer_csv.exists():
            m = pd.read_csv(mer_csv)
            if "merchantId" in df.columns:
                # Merge using the original df to get merchantId
                feat = feat.join(df[["merchantId"]])
                feat = feat.merge(m[["merchantId", "txn_count", "avg_amount", "fraud_rate"]], on="merchantId", how="left")
                feat.rename(columns={
                    "txn_count": "merchant_txn_count",
                    "avg_amount": "merchant_avg_amount",
                    "fraud_rate": "merchant_fraud_rate",
                }, inplace=True)
        # External signals
        ext_dir = Path("ML_Fraud/data/silver/external")
        dev_bl = ext_dir / "device_blacklist.csv"
        ip_bl = ext_dir / "ip_blacklist.csv"
        geo_risk = ext_dir / "geo_risk.csv"
        if dev_bl.exists() and "deviceId" in df.columns:
            db = pd.read_csv(dev_bl)
            feat = feat.join(df[["deviceId"]])
            feat["device_blacklisted"] = feat["deviceId"].isin(set(db["deviceId"]))
        if ip_bl.exists() and "ip" in df.columns:
            ib = pd.read_csv(ip_bl)
            feat = feat.join(df[["ip"]])
            feat["ip_blacklisted"] = feat["ip"].isin(set(ib["ip"]))
        if geo_risk.exists():
            gr = pd.read_csv(geo_risk)
            if "country" in df.columns:
                feat = feat.join(df[["country"]])
                feat = feat.merge(gr, on="country", how="left")
                feat.rename(columns={"risk_score": "country_risk"}, inplace=True)
    except Exception:
        # Never fail feature build due to optional joins
        pass

    # Clean up non-numeric keys accidentally introduced via joins
    for c in ["deviceId", "merchantId", "ip", "country"]:
        if c in feat.columns:
            feat.drop(columns=[c], inplace=True)
    # Cast booleans to ints
    for c in feat.columns:
        if feat[c].dtype == bool:
            feat[c] = feat[c].astype(int)

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
