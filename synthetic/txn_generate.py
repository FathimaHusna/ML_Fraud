from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from features.txn_features import build_txn_features


@dataclass
class TxnSynthConfig:
    users: int = 200
    days: int = 30
    avg_txn_per_user: int = 60
    fraud_ratio: float = 0.2
    seed: int = 42
    out_features: Path = Path("./ML_Fraud/data/gold/txn_synth_features.csv")
    out_records: Path | None = Path("./ML_Fraud/data/silver/txn_synth_records.json")


_CHANNELS = ["web", "mobile", "ussd"]
_CITIES = ["Colombo", "Gampaha", "Kandy", "Galle", "Jaffna"]


def _rand_times(n: int, start: datetime, end: datetime, rng: np.random.Generator) -> List[datetime]:
    s = start.timestamp()
    e = end.timestamp()
    t = s + (e - s) * rng.random(n)
    return [datetime.fromtimestamp(x, tz=timezone.utc) for x in t]


def _mk_user_profile(uid: str, rng: np.random.Generator) -> Dict[str, Any]:
    base_city = rng.choice(_CITIES)
    base_device = f"dev-{uid}-{int(rng.integers(100,999))}"
    alt_device = f"dev-{uid}-alt"
    mean_amt = float(rng.uniform(1500, 15000))
    std_amt = float(rng.uniform(0.2, 0.8) * mean_amt)
    channel_probs = rng.dirichlet(alpha=np.ones(len(_CHANNELS)))
    payees = [f"pay-{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(3, 8)))]
    return {
        "userId": uid,
        "city": base_city,
        "devices": [base_device, alt_device],
        "mean_amt": mean_amt,
        "std_amt": std_amt,
        "channels": dict(zip(_CHANNELS, channel_probs)),
        "payees": payees,
    }


def _sample_amount(mean: float, std: float, rng: np.random.Generator) -> float:
    # Use lognormal to avoid negatives
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    return float(np.clip(rng.lognormal(mu, sigma), 10.0, 2_000_000.0))


def _mk_legit_txn(uid: str, profile: Dict[str, Any], t: datetime, rng: np.random.Generator) -> Dict[str, Any]:
    ch = rng.choice(_CHANNELS, p=list(profile["channels"].values()))
    dev = profile["devices"][0] if rng.random() < 0.9 else profile["devices"][1]
    pay = rng.choice(profile["payees"]) if rng.random() < 0.95 else f"pay-{int(rng.integers(1000,9999))}"
    amt = _sample_amount(profile["mean_amt"], profile["std_amt"], rng)
    city = profile["city"]
    return {
        "transactionId": f"TXN{uid}-{int(rng.integers(10**6, 10**9))}",
        "userId": uid,
        "payeeId": pay,
        "merchantId": None,
        "merchantCategory": None,
        "amount": round(amt, 2),
        "timestamp": t.isoformat().replace("+00:00", "Z"),
        "channel": ch,
        "deviceId": dev,
        "deviceType": None,
        "deviceOs": None,
        "ip": None,
        "city": city,
        "region": None,
        "country": "LK",
        "lat": None,
        "lon": None,
        "is_fraud": 0,
    }


def _apply_fraud_patterns(txn: Dict[str, Any], profile: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    t = pd.to_datetime(txn["timestamp"], utc=True)
    pattern = rng.choice(["odd_hour_new_city", "velocity_burst", "new_payee_high_amt", "device_switch_high_amt"])  # noqa
    tx = txn.copy()
    if pattern == "odd_hour_new_city":
        # Move to night and new city
        night_hour = int(rng.integers(0, 4))
        # Use lowercase 'h' to avoid pandas deprecation for floor frequency
        t2 = t.tz_convert("UTC").floor("h").replace(hour=night_hour)
        tx["timestamp"] = t2.isoformat().replace("+00:00", "Z")
        tx["city"] = rng.choice([c for c in _CITIES if c != profile["city"]])
        tx["deviceId"] = f"dev-hijacked-{int(rng.integers(1000,9999))}"
    elif pattern == "velocity_burst":
        # Mark this as one of a burst; the caller will add neighbors
        tx["amount"] = round(float(tx["amount"]) * float(rng.uniform(0.1, 0.5)), 2)
    elif pattern == "new_payee_high_amt":
        tx["payeeId"] = f"pay-{int(rng.integers(9000,9999))}"
        tx["amount"] = round(float(tx["amount"]) * float(rng.uniform(5, 15)), 2)
    elif pattern == "device_switch_high_amt":
        tx["deviceId"] = f"dev-new-{int(rng.integers(1000,9999))}"
        tx["amount"] = round(float(tx["amount"]) * float(rng.uniform(3, 8)), 2)
    tx["is_fraud"] = 1
    return tx


def generate_txn_dataset(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    users = [f"user_{i:04d}" for i in range(cfg.users)]
    start = datetime.now(tz=timezone.utc) - timedelta(days=cfg.days)
    end = datetime.now(tz=timezone.utc)

    records: List[Dict[str, Any]] = []
    # Legit transactions
    for uid in users:
        profile = _mk_user_profile(uid, rng)
        n = int(max(5, rng.poisson(cfg.avg_txn_per_user)))
        times = _rand_times(n, start, end, rng)
        times.sort()
        for t in times:
            records.append(_mk_legit_txn(uid, profile, t, rng))

    # Inject fraud
    n_fraud = int(len(records) * cfg.fraud_ratio)
    fraud_idx = rng.choice(len(records), size=n_fraud, replace=False)
    for idx in fraud_idx:
        tx = records[idx]
        uid = tx["userId"]
        # Reconstruct user profile for consistency
        profile = _mk_user_profile(uid, rng)
        fr = _apply_fraud_patterns(tx, profile, rng)
        records[idx] = fr
        # For velocity burst, add 2-4 more tiny txns around the same minute
        if float(fr["amount"]) < 1000 and rng.random() < 0.5:
            base_t = pd.to_datetime(fr["timestamp"], utc=True)
            for _ in range(int(rng.integers(2, 5))):
                dt = timedelta(seconds=int(rng.integers(5, 50)))
                t2 = base_t + dt
                clone = fr.copy()
                clone["transactionId"] = f"{fr['transactionId']}-b{int(rng.integers(100,999))}"
                clone["timestamp"] = t2.isoformat().replace("+00:00", "Z")
                records.append(clone)

    df = pd.DataFrame(records)
    feat = build_txn_features(df)
    return feat, df


def run_and_save(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    feat, rec = generate_txn_dataset(cfg)
    cfg.out_features.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(cfg.out_features, index=False)
    if cfg.out_records:
        p = Path(cfg.out_records)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec.to_json(p, orient="records", indent=2)
    return feat, rec, cfg.out_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic e-payment transactions and features")
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--avg-per-user", type=int, default=60)
    parser.add_argument("--fraud", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-features", default="./ML_Fraud/data/gold/txn_synth_features.csv")
    parser.add_argument("--out-records", default="./ML_Fraud/data/silver/txn_synth_records.json")
    args = parser.parse_args()

    cfg = TxnSynthConfig(
        users=args.users,
        days=args.days,
        avg_txn_per_user=args.avg_per_user,
        fraud_ratio=args.fraud,
        seed=args.seed,
        out_features=Path(args.out_features).resolve(),
        out_records=Path(args.out_records).resolve() if args.out_records else None,
    )
    feat, rec, outp = run_and_save(cfg)
    print(f"Wrote txn synthetic features: {outp} ({len(feat)} rows)")
