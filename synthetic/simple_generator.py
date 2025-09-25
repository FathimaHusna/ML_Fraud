from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    import yaml  # Optional; simple generator works without YAML
except Exception:  # pragma: no cover
    yaml = None

from .txn_generate import TxnSynthConfig
from features.txn_features import build_txn_features


@dataclass
class _SimpleParams:
    # Seasonality amplitudes in [0, 1]
    daily_amp: float = 0.3
    weekly_amp: float = 0.2
    fraud_rate: float = 0.2
    # Optional UTC start date (ISO8601 string). If None, use "now - days" window.
    start_iso: str | None = None
    patterns: Dict[str, Dict[str, Any]] = None  # type: ignore


def _load_params(cfg: TxnSynthConfig) -> _SimpleParams:
    # Defaults
    defaults = {
        "seasonality": {"daily_amp": 0.3, "weekly_amp": 0.2},
        "base": {"fraud_rate": cfg.fraud_ratio if cfg.fraud_ratio is not None else 0.2},
        "patterns": {
            "velocity_burst": {"prob": 0.005, "burst_lambda": 8, "window_s": 120},
            "new_payee_high_amt": {"prob": 0.004, "z_min": 2.5, "multiplier_min": 5.0, "multiplier_max": 12.0},
            "odd_hour_new_city": {"prob": 0.004, "night_end_hour": 4},
        },
    }
    data: Dict[str, Any] = defaults
    if cfg.config_path and Path(cfg.config_path).exists() and yaml is not None:
        try:
            with open(cfg.config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            # Merge shallowly
            for k in ["seasonality", "base", "patterns"]:
                if k in user_cfg:
                    if isinstance(data.get(k), dict) and isinstance(user_cfg[k], dict):
                        data[k].update(user_cfg[k])
                    else:
                        data[k] = user_cfg[k]
        except Exception:
            # Fall back to defaults silently (including if YAML lib is missing)
            pass
    p = _SimpleParams(
        daily_amp=float(data.get("seasonality", {}).get("daily_amp", 0.3)),
        weekly_amp=float(data.get("seasonality", {}).get("weekly_amp", 0.2)),
        fraud_rate=float(data.get("base", {}).get("fraud_rate", 0.2)),
        start_iso=str(data.get("base", {}).get("start")) if data.get("base", {}).get("start") is not None else None,
        patterns=data.get("patterns", {}),
    )
    return p


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
    cats = rng.choice(_MCC, size=n, replace=True)
    catalog: Dict[str, str] = {}
    for i, c in enumerate(cats):
        catalog[f"mer-{i:04d}"] = str(c)
    return catalog


def _mk_user_profile(uid: str, rng: np.random.Generator, merchants: Dict[str, str]) -> Dict[str, Any]:
    base_city = rng.choice(_CITIES)
    base_device = f"dev-{uid}-{int(rng.integers(100,999))}"
    alt_device = f"dev-{uid}-alt"
    mean_amt = float(rng.uniform(1500, 15000))
    std_amt = float(rng.uniform(0.2, 0.8) * mean_amt)
    channel_probs = rng.dirichlet(alpha=np.ones(len(_CHANNELS)))
    payees = [f"pay-{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(3, 8)))]
    merch_list = list(rng.choice(list(merchants.keys()), size=int(rng.integers(8, 21)), replace=False))
    merch_probs = rng.dirichlet(alpha=np.ones(len(merch_list))) if len(merch_list) > 0 else np.array([])
    return {
        "userId": uid,
        "city": base_city,
        "devices": [base_device, alt_device],
        "mean_amt": mean_amt,
        "std_amt": std_amt,
        "channels": dict(zip(_CHANNELS, channel_probs)),
        "payees": payees,
        "merchants": merch_list,
        "merchant_probs": merch_probs.tolist() if len(merch_list) > 0 else [],
    }


def _sample_amount(mean: float, std: float, rng: np.random.Generator) -> float:
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    return float(np.clip(rng.lognormal(mu, sigma), 10.0, 2_000_000.0))


def _seasonal_accept_prob(ts: datetime, daily_amp: float, weekly_amp: float) -> float:
    # Map hour and day-of-week into [0, 1+amp] space
    h = ts.hour
    dow = ts.weekday()
    # sin peaks around mid-day and mid-week; that’s acceptable for demo purposes
    comp = 1.0
    comp += daily_amp * np.sin(2 * np.pi * (h / 24.0))
    comp += weekly_amp * np.sin(2 * np.pi * (dow / 7.0))
    # Normalize by maximal possible amplitude to keep in [0,1]
    max_comp = 1.0 + abs(daily_amp) + abs(weekly_amp)
    return max(0.0, min(1.0, comp / max_comp))


def _rand_time_seasonal(rng: np.random.Generator, start: datetime, end: datetime, daily_amp: float, weekly_amp: float) -> datetime:
    # Simple rejection sampling; bounded attempts
    s = start.timestamp()
    e = end.timestamp()
    for _ in range(200):
        t = s + (e - s) * rng.random()
        ts = datetime.fromtimestamp(t, tz=timezone.utc)
        p = _seasonal_accept_prob(ts, daily_amp, weekly_amp)
        if rng.random() < p:
            return ts
    # Fallback: uniform
    t = s + (e - s) * rng.random()
    return datetime.fromtimestamp(t, tz=timezone.utc)


def _mk_legit_txn(uid: str, profile: Dict[str, Any], t: datetime, rng: np.random.Generator, merchants: Dict[str, str]) -> Dict[str, Any]:
    ch = rng.choice(_CHANNELS, p=list(profile["channels"].values()))
    dev = profile["devices"][0] if rng.random() < 0.9 else profile["devices"][1]
    pay = rng.choice(profile["payees"]) if rng.random() < 0.95 else f"pay-{int(rng.integers(1000,9999))}"
    amt = _sample_amount(profile["mean_amt"], profile["std_amt"], rng)
    # Merchant choice per preference
    if profile.get("merchants"):
        n_m = len(profile["merchants"])
        probs = np.array(profile.get("merchant_probs", [1.0 / max(1, n_m)] * n_m), dtype=float)
        if probs.sum() <= 0 and n_m > 0:
            probs = np.ones(n_m, dtype=float) / n_m
        else:
            probs = probs / probs.sum()
        if rng.random() < 0.9:
            mid = rng.choice(profile["merchants"], p=probs)
        else:
            mid = rng.choice(list(merchants.keys()))
        mcat = merchants.get(mid)
    else:
        mid, mcat = None, None
    return {
        "transactionId": f"TXN{uid}-{int(rng.integers(10**6, 10**9))}",
        "userId": uid,
        "payeeId": pay,
        "merchantId": mid,
        "merchantCategory": mcat,
        "amount": round(amt, 2),
        "timestamp": t.isoformat().replace("+00:00", "Z"),
        "channel": ch,
        "deviceId": dev,
        "deviceType": None,
        "deviceOs": None,
        "ip": None,
        "city": profile["city"],
        "region": None,
        "country": "LK",
        "lat": None,
        "lon": None,
        "is_fraud": 0,
        "scenario": "legit",
        "fraud_score_sim": 0.0,
    }


def _apply_velocity_burst(records: List[Dict[str, Any]], idx: int, rng: np.random.Generator, params: Dict[str, Any]) -> None:
    base = records[idx]
    # Reduce base amount to small value typical for testing
    base_amt = float(base["amount"]) * float(rng.uniform(0.1, 0.5))
    base["amount"] = round(base_amt, 2)
    base_ts = pd.to_datetime(base["timestamp"], utc=True)
    lam = float(params.get("burst_lambda", 8))
    window_s = int(params.get("window_s", 120))
    k = int(max(0, rng.poisson(lam)))
    base["is_fraud"] = 1
    base["scenario"] = "velocity_burst"
    base["fraud_score_sim"] = float(min(1.0, 0.4 + 0.02 * k))
    for _ in range(k):
        dt = timedelta(seconds=int(rng.integers(1, max(2, window_s))))
        clone = dict(base)
        clone["transactionId"] = f"{base['transactionId']}-b{int(rng.integers(100,999))}"
        clone["timestamp"] = (base_ts + dt).isoformat().replace("+00:00", "Z")
        clone["is_fraud"] = 1
        clone["scenario"] = "velocity_burst"
        # Slight jitter in amounts
        clone["amount"] = round(float(clone["amount"]) * float(rng.uniform(0.8, 1.2)), 2)
        clone["fraud_score_sim"] = base["fraud_score_sim"]
        records.append(clone)


def _apply_new_payee_high_amt(records: List[Dict[str, Any]], idx: int, rng: np.random.Generator, params: Dict[str, Any], profile: Dict[str, Any]) -> None:
    tx = records[idx]
    z_min = float(params.get("z_min", 2.5))
    mult_min = float(params.get("multiplier_min", 5.0))
    mult_max = float(params.get("multiplier_max", 12.0))
    tx["payeeId"] = f"pay-{int(rng.integers(9000,9999))}"
    # Raise amount; also ensure z-score threshold vs. profile stats
    target_amt = float(tx["amount"]) * float(rng.uniform(mult_min, mult_max))
    mu, sigma = float(profile["mean_amt"]), float(profile["std_amt"]) or 1.0
    z_amt = (target_amt - mu) / (sigma if sigma > 0 else 1.0)
    if z_amt < z_min:
        target_amt = mu + z_min * sigma
        z_amt = z_min
    tx["amount"] = round(float(target_amt), 2)
    tx["is_fraud"] = 1
    tx["scenario"] = "new_payee_high_amt"
    # Map z into 0.5..1.0 roughly
    tx["fraud_score_sim"] = float(max(0.5, min(1.0, 0.5 + 0.1 * z_amt)))


def _apply_odd_hour_new_city(records: List[Dict[str, Any]], idx: int, rng: np.random.Generator, params: Dict[str, Any], profile: Dict[str, Any]) -> None:
    tx = records[idx]
    night_end = int(params.get("night_end_hour", 4))
    t = pd.to_datetime(tx["timestamp"], utc=True)
    night_hour = int(rng.integers(0, max(1, night_end)))
    t2 = t.tz_convert("UTC").floor("h").replace(hour=night_hour)
    tx["timestamp"] = t2.isoformat().replace("+00:00", "Z")
    tx["city"] = rng.choice([c for c in _CITIES if c != profile["city"]])
    tx["deviceId"] = f"dev-hijacked-{int(rng.integers(1000,9999))}"
    tx["is_fraud"] = 1
    tx["scenario"] = "odd_hour_new_city"
    tx["fraud_score_sim"] = 0.6


def _apply_card_testing(records: List[Dict[str, Any]], idx: int, rng: np.random.Generator, params: Dict[str, Any]) -> None:
    """Inject a burst of micro-amount transactions (card testing pattern).

    Params (with defaults if missing):
      - n_min, n_max: number of additional txns to append
      - window_s: time window in seconds to spread the burst
      - amount_min, amount_max: micro-amount range (defaults 5..50)
    """
    base = records[idx]
    t0 = pd.to_datetime(base["timestamp"], utc=True)
    n_min = int(params.get("n_min", 4))
    n_max = int(params.get("n_max", 10))
    window_s = int(params.get("window_s", 60))
    amount_min = float(params.get("amount_min", 5.0))
    amount_max = float(params.get("amount_max", 50.0))

    # Ensure base is micro and labeled
    base_amt = float(rng.uniform(amount_min, amount_max))
    base["amount"] = round(base_amt, 2)
    base["is_fraud"] = 1
    base["scenario"] = "card_testing"
    base["fraud_score_sim"] = 0.55

    k = int(rng.integers(max(0, n_min), max(n_min + 1, n_max + 1)))
    for _ in range(k):
        dt = timedelta(seconds=int(rng.integers(1, max(2, window_s))))
        clone = dict(base)
        clone["transactionId"] = f"{base['transactionId']}-ct{int(rng.integers(100,999))}"
        clone["timestamp"] = (t0 + dt).isoformat().replace("+00:00", "Z")
        clone["amount"] = round(float(rng.uniform(amount_min, amount_max)), 2)
        clone["is_fraud"] = 1
        clone["scenario"] = "card_testing"
        clone["fraud_score_sim"] = base["fraud_score_sim"]
        records.append(clone)


def generate_simple(cfg: TxnSynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    params = _load_params(cfg)
    merchant_catalog = _build_merchant_catalog(rng, n=max(300, int(cfg.users * 1.5)))

    users = [f"user_{i:04d}" for i in range(cfg.users)]
    # Base window: either from base.start (ISO) or the last N days
    if params.start_iso:
        try:
            # Accept dates like "2025-01-01" or full ISO; force UTC
            start = pd.to_datetime(params.start_iso, utc=True).to_pydatetime()
        except Exception:
            start = datetime.now(tz=timezone.utc) - timedelta(days=cfg.days)
    else:
        start = datetime.now(tz=timezone.utc) - timedelta(days=cfg.days)
    end = start + timedelta(days=cfg.days)

    records: List[Dict[str, Any]] = []
    # Generate legit traffic with seasonality
    for uid in users:
        profile = _mk_user_profile(uid, rng, merchant_catalog)
        n = int(max(1, rng.poisson(cfg.avg_txn_per_user)))
        for _ in range(n):
            t = _rand_time_seasonal(rng, start, end, params.daily_amp, params.weekly_amp)
            records.append(_mk_legit_txn(uid, profile, t, rng, merchant_catalog))

    if not records:
        df_empty = pd.DataFrame([])
        return df_empty, df_empty

    # Select fraud candidates
    total = len(records)
    n_fraud = int(max(0, min(total, round(total * params.fraud_rate))))
    if n_fraud > 0:
        fraud_idx = set(map(int, rng.choice(total, size=n_fraud, replace=False)))
    else:
        fraud_idx = set()

    # Prepare scenario distribution
    pat_cfg: Dict[str, Dict[str, Any]] = params.patterns or {}
    names = list(pat_cfg.keys())
    probs = np.array([float(max(0.0, pat_cfg[n].get("prob", 0.0))) for n in names], dtype=float)
    if probs.sum() <= 0:
        # Fallback uniform over known three
        names = ["velocity_burst", "new_payee_high_amt", "odd_hour_new_city"]
        probs = np.array([1/3, 1/3, 1/3], dtype=float)
    else:
        probs = probs / probs.sum()

    # Index → user profile cache
    prof_cache: Dict[str, Dict[str, Any]] = {}
    def _get_profile(uid: str) -> Dict[str, Any]:
        if uid not in prof_cache:
            # Pass the merchant catalog to ensure consistent merchant preferences
            prof_cache[uid] = _mk_user_profile(uid, rng, merchant_catalog)
        return prof_cache[uid]

    for idx in range(total):
        if idx not in fraud_idx:
            continue
        name = str(rng.choice(names, p=probs.tolist()))
        uid = records[idx]["userId"]
        profile = _get_profile(uid)
        if name == "velocity_burst":
            _apply_velocity_burst(records, idx, rng, pat_cfg.get(name, {}))
        elif name == "new_payee_high_amt":
            _apply_new_payee_high_amt(records, idx, rng, pat_cfg.get(name, {}), profile)
        elif name == "odd_hour_new_city":
            _apply_odd_hour_new_city(records, idx, rng, pat_cfg.get(name, {}), profile)
        elif name == "card_testing":
            _apply_card_testing(records, idx, rng, pat_cfg.get(name, {}))
        else:
            # Unknown name → treat as odd_hour_new_city for now
            _apply_odd_hour_new_city(records, idx, rng, pat_cfg.get(name, {}), profile)

    df = pd.DataFrame(records)
    feat = build_txn_features(df)
    return feat, df
