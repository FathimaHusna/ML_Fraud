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
    complexity: str = "medium"  # one of: simple, medium, complex (builtin generator only)
    # New: generator routing and optional YAML config
    generator: str = "builtin"  # one of: builtin, simple
    config_path: Path | None = None


_CHANNELS = ["web", "mobile", "ussd"]
_CITIES = ["Colombo", "Gampaha", "Kandy", "Galle", "Jaffna"]
_MCC = [
    "groceries",
    "fuel",
    "electronics",
    "restaurants",
    "utilities",
    "travel",
    "health",
    "entertainment",
    "education",
    "online",
]


def _build_merchant_catalog(rng: np.random.Generator, n: int = 400) -> Dict[str, str]:
    """Create a simple merchant catalog mapping merchantId -> category (MCC label)."""
    cats = rng.choice(_MCC, size=n, replace=True)
    catalog: Dict[str, str] = {}
    for i, c in enumerate(cats):
        catalog[f"mer-{i:04d}"] = str(c)
    return catalog


def _rand_times(n: int, start: datetime, end: datetime, rng: np.random.Generator) -> List[datetime]:
    s = start.timestamp()
    e = end.timestamp()
    t = s + (e - s) * rng.random(n)
    return [datetime.fromtimestamp(x, tz=timezone.utc) for x in t]


def _mk_user_profile(uid: str, rng: np.random.Generator, merchants: Dict[str, str]) -> Dict[str, Any]:
    base_city = rng.choice(_CITIES)
    base_device = f"dev-{uid}-{int(rng.integers(100,999))}"
    alt_device = f"dev-{uid}-alt"
    mean_amt = float(rng.uniform(1500, 15000))
    std_amt = float(rng.uniform(0.2, 0.8) * mean_amt)
    channel_probs = rng.dirichlet(alpha=np.ones(len(_CHANNELS)))
    payees = [f"pay-{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(3, 8)))]
    # Merchant preferences: pick 8–20 merchants per user with a Dirichlet mix
    merch_ids = rng.choice(list(merchants.keys()), size=int(rng.integers(8, 21)), replace=False)
    merch_probs = rng.dirichlet(alpha=np.ones(len(merch_ids)))
    return {
        "userId": uid,
        "city": base_city,
        "devices": [base_device, alt_device],
        "mean_amt": mean_amt,
        "std_amt": std_amt,
        "channels": dict(zip(_CHANNELS, channel_probs)),
        "payees": payees,
        "merchants": list(merch_ids),
        "merchant_probs": merch_probs.tolist(),
    }


def _sample_amount(mean: float, std: float, rng: np.random.Generator) -> float:
    # Use lognormal to avoid negatives
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    return float(np.clip(rng.lognormal(mu, sigma), 10.0, 2_000_000.0))


def _mk_legit_txn(uid: str, profile: Dict[str, Any], t: datetime, rng: np.random.Generator, merchants: Dict[str, str]) -> Dict[str, Any]:
    ch = rng.choice(_CHANNELS, p=list(profile["channels"].values()))
    dev = profile["devices"][0] if rng.random() < 0.9 else profile["devices"][1]
    pay = rng.choice(profile["payees"]) if rng.random() < 0.95 else f"pay-{int(rng.integers(1000,9999))}"
    amt = _sample_amount(profile["mean_amt"], profile["std_amt"], rng)
    # Choose merchant per user preference (90% from preferred set)
    if profile.get("merchants"):
        if rng.random() < 0.9:
            mi = rng.choice(profile["merchants"], p=np.array(profile["merchant_probs"]))
        else:
            mi = rng.choice(list(merchants.keys()))
        mcat = merchants.get(mi)
    else:
        mi, mcat = None, None
    city = profile["city"]
    return {
        "transactionId": f"TXN{uid}-{int(rng.integers(10**6, 10**9))}",
        "userId": uid,
        "payeeId": pay,
        "merchantId": mi,
        "merchantCategory": mcat,
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


def _clone_with_time(tx: Dict[str, Any], new_time: datetime, rng: np.random.Generator) -> Dict[str, Any]:
    c = tx.copy()
    c["transactionId"] = f"{tx['transactionId']}-s{int(rng.integers(100,999))}"
    c["timestamp"] = new_time.isoformat().replace("+00:00", "Z")
    return c


def _inject_ato_sequence(records: List[Dict[str, Any]], base_tx: Dict[str, Any], rng: np.random.Generator) -> None:
    """Simulate a simple ATO: new device -> new payee -> small tests -> large transfer."""
    base_t = pd.to_datetime(base_tx["timestamp"], utc=True)
    # Step 1: suspicious login/device change
    t1 = base_t - timedelta(minutes=int(rng.integers(30, 120)))
    s1 = base_tx.copy()
    s1["deviceId"] = f"dev-hijacked-{int(rng.integers(1000,9999))}"
    s1["is_fraud"] = 1
    s1 = _clone_with_time(s1, t1, rng)
    # Step 2: add new payee
    t2 = t1 + timedelta(minutes=int(rng.integers(2, 10)))
    s2 = s1.copy()
    s2["payeeId"] = f"pay-{int(rng.integers(9000,9999))}"
    s2 = _clone_with_time(s2, t2, rng)
    # Step 3: 1-3 small test transfers
    smalls: List[Dict[str, Any]] = []
    for _ in range(int(rng.integers(1, 4))):
        dt = timedelta(minutes=int(rng.integers(1, 5)))
        t3 = t2 + dt
        s3 = s2.copy()
        s3["amount"] = round(float(base_tx["amount"]) * float(rng.uniform(0.05, 0.2)), 2)
        s3 = _clone_with_time(s3, t3, rng)
        smalls.append(s3)
        t2 = t3
    # Step 4: large fraudulent transfer
    t4 = (pd.to_datetime(smalls[-1]["timestamp"], utc=True) if smalls else t2) + timedelta(minutes=int(rng.integers(1, 5)))
    s4 = s2.copy()
    s4["amount"] = round(float(base_tx["amount"]) * float(rng.uniform(5, 15)), 2)
    s4 = _clone_with_time(s4, t4, rng)
    for e in [s1, s2, *smalls, s4]:
        e["is_fraud"] = 1
        records.append(e)


def _inject_card_testing(records: List[Dict[str, Any]], base_tx: Dict[str, Any], rng: np.random.Generator) -> None:
    base_t = pd.to_datetime(base_tx["timestamp"], utc=True)
    n = int(rng.integers(8, 20))
    for _ in range(n):
        dt = timedelta(seconds=int(rng.integers(5, 60)))
        t2 = base_t + dt
        c = base_tx.copy()
        c["amount"] = round(float(rng.uniform(5, 50)), 2)
        c = _clone_with_time(c, t2, rng)
        c["is_fraud"] = 1
        records.append(c)


def _inject_structuring(records: List[Dict[str, Any]], base_tx: Dict[str, Any], rng: np.random.Generator) -> None:
    base_t = pd.to_datetime(base_tx["timestamp"], utc=True)
    n = int(rng.integers(5, 12))
    for _ in range(n):
        dt = timedelta(minutes=int(rng.integers(2, 30)))
        t2 = base_t + dt
        c = base_tx.copy()
        c["amount"] = round(float(rng.uniform(9000, 9900)), 2)
        c = _clone_with_time(c, t2, rng)
        c["is_fraud"] = 1
        records.append(c)


def _generate_txn_dataset_builtin(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    users = [f"user_{i:04d}" for i in range(cfg.users)]
    start = datetime.now(tz=timezone.utc) - timedelta(days=cfg.days)
    end = datetime.now(tz=timezone.utc)

    # Build a global merchant catalog
    merchant_catalog = _build_merchant_catalog(rng, n=max(300, int(cfg.users * 1.5)))

    records: List[Dict[str, Any]] = []
    # Legit transactions
    for uid in users:
        profile = _mk_user_profile(uid, rng, merchant_catalog)
        n = int(max(5, rng.poisson(cfg.avg_txn_per_user)))
        times = _rand_times(n, start, end, rng)
        times.sort()
        for t in times:
            records.append(_mk_legit_txn(uid, profile, t, rng, merchant_catalog))

    # Adjust fraud ratio lightly by complexity if not explicitly set by user
    comp = (cfg.complexity or "medium").lower()
    if comp == "simple":
        base_ratio = 0.1
    elif comp == "complex":
        base_ratio = 0.25
    else:
        base_ratio = 0.2
    n_fraud = int(len(records) * (cfg.fraud_ratio if cfg.fraud_ratio is not None else base_ratio))
    fraud_idx = rng.choice(len(records), size=n_fraud, replace=False)
    for idx in fraud_idx:
        tx = records[idx]
        uid = tx["userId"]
        # Reconstruct user profile for consistency
        profile = _mk_user_profile(uid, rng, merchant_catalog)
        fr = _apply_fraud_patterns(tx, profile, rng)
        records[idx] = fr
        # For velocity burst, add 2-4 more tiny txns around the same minute (more likely for complex)
        prob_burst = 0.2 if comp == "simple" else (0.4 if comp == "medium" else 0.7)
        if float(fr["amount"]) < 1000 and rng.random() < prob_burst:
            base_t = pd.to_datetime(fr["timestamp"], utc=True)
            for _ in range(int(rng.integers(2, 5))):
                dt = timedelta(seconds=int(rng.integers(5, 50)))
                t2 = base_t + dt
                clone = fr.copy()
                clone["transactionId"] = f"{fr['transactionId']}-b{int(rng.integers(100,999))}"
                clone["timestamp"] = t2.isoformat().replace("+00:00", "Z")
                records.append(clone)
        # Inject richer scenarios with probabilities by complexity
        r = rng.random()
        if comp == "simple":
            # no extra injections beyond basic pattern
            pass
        elif comp == "medium":
            if r < 0.15:
                _inject_ato_sequence(records, fr, rng)
            elif r < 0.25:
                _inject_card_testing(records, fr, rng)
        elif comp == "complex":
            if r < 0.25:
                _inject_ato_sequence(records, fr, rng)
            elif r < 0.45:
                _inject_card_testing(records, fr, rng)
            elif r < 0.60:
                _inject_structuring(records, fr, rng)

    df = pd.DataFrame(records)
    feat = build_txn_features(df)
    return feat, df


def generate_txn_dataset(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Router for dataset generation.

    - builtin: current generator (complexity flag controls richness)
    - simple: configurable rule-based generator (seasonality + rules)
    """
    if (cfg.generator or "builtin").lower() == "simple":
        try:
            from .simple_generator import generate_simple
        except Exception as e:
            raise RuntimeError(f"simple generator not available: {e}")
        return generate_simple(cfg)
    return _generate_txn_dataset_builtin(cfg)


def run_and_save(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    feat, rec = generate_txn_dataset(cfg)

    # Clean and normalize records before persisting and recomputing features to keep alignment
    rec_df = rec.copy()
    # Parse timestamps; drop invalid
    rec_df["ts"] = pd.to_datetime(rec_df["timestamp"], utc=True, errors="coerce")
    rec_df = rec_df[~rec_df["ts"].isna()].copy()

    # Enforce unique transactionId by appending a short suffix on collisions
    seen: set[str] = set()
    rng = np.random.default_rng(getattr(cfg, "seed", 42))
    def _uniq(txid: str) -> str:
        if txid not in seen:
            seen.add(txid)
            return txid
        # append up to a few times; extremely unlikely to loop long
        for _ in range(10):
            cand = f"{txid}-u{int(rng.integers(100, 999))}"
            if cand not in seen:
                seen.add(cand)
                return cand
        # fallback: include large random
        cand = f"{txid}-u{int(rng.integers(10**6, 10**9))}"
        seen.add(cand)
        return cand

    rec_df.sort_values(["userId", "ts", "transactionId"], inplace=True)
    rec_df["transactionId"] = [
        _uniq(str(t)) for t in rec_df["transactionId"].astype(str).tolist()
    ]

    # Normalize timestamp string format to ISO8601 Z
    rec_df["timestamp"] = rec_df["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    rec_df.drop(columns=["ts"], inplace=True)

    # Recompute features from cleaned records for alignment
    feat_clean = build_txn_features(rec_df)

    # Persist
    cfg.out_features.parent.mkdir(parents=True, exist_ok=True)
    feat_clean.to_csv(cfg.out_features, index=False)
    if cfg.out_records:
        p = Path(cfg.out_records)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec_df.to_json(p, orient="records", indent=2)
    return feat_clean, rec_df, cfg.out_features


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
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium")
    parser.add_argument("--generator", choices=["builtin", "simple"], default="builtin")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = TxnSynthConfig(
        users=args.users,
        days=args.days,
        avg_txn_per_user=args.avg_per_user,
        fraud_ratio=args.fraud,
        seed=args.seed,
        out_features=Path(args.out_features).resolve(),
        out_records=Path(args.out_records).resolve() if args.out_records else None,
        complexity=args.complexity,
        generator=args.generator,
        config_path=Path(args.config).resolve() if args.config else None,
    )
    feat, rec, outp = run_and_save(cfg)
    print(f"Wrote txn synthetic features: {outp} ({len(feat)} rows)")
