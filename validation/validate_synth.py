#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root (ML_Fraud) is importable when running as a script
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]  # ML_Fraud/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd


def _load_records(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _schema_sample_validate(rows: List[dict], sample_n: int) -> tuple[int, int]:
    from ingestion.txn_schema import TxnRaw

    n = len(rows)
    if n == 0:
        return 0, 0
    if n <= sample_n:
        idx = range(n)
    else:
        idx = np.random.default_rng(42).choice(n, size=sample_n, replace=False)
    ok = bad = 0
    for i in idx:
        try:
            TxnRaw.model_validate(rows[int(i)])
            ok += 1
        except Exception:
            bad += 1
    return ok, bad


def _velocity_bursts_per_user(df: pd.DataFrame) -> int:
    def bursts(g: pd.DataFrame) -> int:
        t = (g["ts"].astype("int64") // 10**9).to_numpy()
        i = j = 0
        b = 0
        while i < len(t):
            while j < len(t) and t[j] - t[i] <= 60:
                j += 1
            if j - i >= 3:
                b += 1
            i += 1
        return b

    return int(df.groupby("userId", dropna=False).apply(bursts).sum())


def _ato_like_count(df: pd.DataFrame) -> int:
    ato = 0
    for _, g in df.groupby("userId", dropna=False):
        seen: set[Any] = set()
        prev_dev = None
        am_hist: List[float] = []
        for _, r in g.iterrows():
            new_dev = (prev_dev is not None) and (r.get("deviceId") != prev_dev) and pd.notna(r.get("deviceId"))
            new_pay = (r.get("payeeId") not in seen) if pd.notna(r.get("payeeId")) else False
            big_jump = (len(am_hist) > 0 and r["amount"] > (np.median(am_hist) * 3))
            if new_dev and new_pay and big_jump:
                ato += 1
                break
            seen.add(r.get("payeeId"))
            prev_dev = r.get("deviceId")
            am_hist = (am_hist + [float(r["amount"])])[-5:]
    return int(ato)


def _feature_separability(feat_path: Path, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"present": [], "auc": {}, "means": {}}
    if not feat_path.exists():
        out["error"] = f"features not found: {feat_path}"
        return out
    df = pd.read_csv(feat_path)
    out["rows"] = int(len(df))
    if "is_fraud" not in df.columns:
        out["warning"] = "no is_fraud label in features"
        return out
    y = df["is_fraud"].fillna(0).astype(int).to_numpy()
    cand = [
        c
        for c in [
            "z_amt_30d",
            "cnt_60s",
            "cnt_300s",
            "cnt_3600s",
            "is_new_device",
            "is_new_payee",
            "is_new_city",
            "sum_60s",
            "sum_300s",
            "iforest_score",
        ]
        if c in df.columns
    ]
    out["present"] = cand

    # Try sklearn AUC
    auc_ok = 0
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        for c in cand:
            x = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
            auc = float(roc_auc_score(y, x))
            out["auc"][c] = auc
            if auc >= float(thresholds.get("min_feature_auc", 0.6)):
                auc_ok += 1
    except Exception:
        # Fallback to means by class if sklearn not available
        for c in cand:
            a = float(
                df.loc[df["is_fraud"] == 1, c]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .mean()
            )
            b = float(
                df.loc[df["is_fraud"] == 0, c]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .mean()
            )
            out["means"][c] = {"fraud": a, "nonfraud": b}
            if abs(a - b) >= float(thresholds.get("min_feature_mean_gap", 0.2)):
                auc_ok += 1

    out["passes"] = bool(auc_ok >= int(thresholds.get("min_good_features", 2)))
    out["good_feature_count"] = int(auc_ok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate synthetic dataset (records + features)")
    ap.add_argument("--records", required=True, help="Path to records JSON (array)")
    ap.add_argument("--features", required=True, help="Path to features CSV")
    ap.add_argument("--report-out", default="ML_Fraud/data/reports/synth_validation.json")
    ap.add_argument("--fraud-target", type=float, default=-1.0)
    ap.add_argument("--fraud-tolerance", type=float, default=0.03)
    ap.add_argument("--schema-sample", type=int, default=5000)
    ap.add_argument("--max-dupe-txid", type=int, default=0)
    ap.add_argument("--max-nonpos", type=int, default=0)
    ap.add_argument("--min-bursts", type=int, default=1)
    ap.add_argument("--min-ato", type=int, default=1)
    ap.add_argument("--min-struct", type=int, default=1)
    ap.add_argument("--min-micro", type=int, default=1)
    ap.add_argument("--min-feature-auc", type=float, default=0.6)
    ap.add_argument("--min-good-features", type=int, default=2)
    ap.add_argument(
        "--strict", action="store_true", help="Exit non-zero if any check fails"
    )
    args = ap.parse_args()

    rec_path = Path(args.records)
    feat_path = Path(args.features)

    # Load records and a copy of raw rows for schema sampling
    rec_df = _load_records(rec_path)
    try:
        rows = json.loads(rec_path.read_text(encoding="utf-8"))
    except Exception:
        rows = rec_df.to_dict(orient="records")

    # Schema sample
    schema_ok, schema_bad = _schema_sample_validate(rows, args.schema_sample)

    # Timestamps
    ts = pd.to_datetime(rec_df["timestamp"], utc=True, errors="coerce")
    rec_df = rec_df.copy()
    rec_df["ts"] = ts
    invalid_ts = int(ts.isna().sum())
    tmin = str(ts.min()) if len(ts) else ""
    tmax = str(ts.max()) if len(ts) else ""

    # Labels/ratio
    n = int(len(rec_df))
    pos = int((rec_df.get("is_fraud", 0) == 1).sum())
    ratio = float(pos / n) if n else 0.0
    delta = (ratio - args.fraud_target) if args.fraud_target >= 0 else None

    # Duplicates and amounts
    dup_txid = int(rec_df["transactionId"].duplicated().sum())
    nonpos_amt = int((rec_df["amount"] <= 0).sum())

    # Heuristics
    rec_df = rec_df.dropna(subset=["ts"]).sort_values(["userId", "ts"])
    vel_bursts = _velocity_bursts_per_user(rec_df)
    ato_like = _ato_like_count(rec_df)
    n_struct = int(((rec_df["amount"] >= 9000) & (rec_df["amount"] < 9900)).sum())
    n_micro = int((rec_df["amount"] < 50).sum())

    # Features
    feat_res = _feature_separability(
        feat_path,
        {
            "min_feature_auc": args.min_feature_auc,
            "min_feature_mean_gap": 0.2,
            "min_good_features": args.min_good_features,
        },
    )

    # Row alignment (allow small drift due to invalid ts filtering)
    feat_rows = int(feat_res.get("rows", 0)) if "rows" in feat_res else 0
    row_diff = abs(n - feat_rows)
    row_mismatch_ratio = (row_diff / max(1, n)) if n else 0.0

    # Checks
    checks = {
        "schema_ok": schema_bad == 0,
        "timestamps_ok": invalid_ts == 0,
        "duplicates_ok": dup_txid <= args.max_dupe_txid,
        "nonpositive_amounts_ok": nonpos_amt <= args.max_nonpos,
        "fraud_ratio_ok": True
        if args.fraud_target < 0
        else (abs(delta) <= args.fraud_tolerance),
        "heuristics_ok": (vel_bursts >= args.min_bursts)
        and (ato_like >= args.min_ato)
        and (n_struct >= args.min_struct)
        and (n_micro >= args.min_micro),
        "features_ok": bool(feat_res.get("passes", True)),
        "row_alignment_ok": row_mismatch_ratio <= 0.02,  # <=2% mismatch
    }

    failures = [k for k, v in checks.items() if not v]

    report = {
        "summary": {
            "records": str(rec_path),
            "features": str(feat_path),
            "rows": n,
            "features_rows": feat_rows,
            "row_mismatch_ratio": round(row_mismatch_ratio, 6),
        },
        "metrics": {
            "schema_sample_ok": int(schema_ok),
            "schema_sample_bad": int(schema_bad),
            "invalid_timestamps": invalid_ts,
            "time_range": {"min": tmin, "max": tmax},
            "labels": {
                "pos": pos,
                "ratio": round(ratio, 6),
                "target": args.fraud_target if args.fraud_target >= 0 else None,
                "delta": round(delta, 6) if delta is not None else None,
            },
            "duplicates": {"transactionId_dupes": dup_txid},
            "nonpositive_amounts": nonpos_amt,
            "heuristics": {
                "velocity_bursts": vel_bursts,
                "ato_like_users": ato_like,
                "structuring_txns": n_struct,
                "micro_amount_txns": n_micro,
            },
            "features": feat_res,
        },
        "checks": checks,
        "failures": failures,
        "passed": len(failures) == 0,
    }

    # Print concise console summary
    print("Synthetic Validation Summary")
    print(f"- rows={n}, features_rows={feat_rows}, mismatch_ratio={row_mismatch_ratio:.4f}")
    print(f"- schema_sample: ok={schema_ok}, bad={schema_bad}")
    print(f"- timestamps: invalid={invalid_ts}, range=[{tmin} .. {tmax}]")
    fr = report["metrics"]["labels"]
    if fr["target"] is not None:
        print(
            f"- fraud_ratio={fr['ratio']:.4f} target={fr['target']:.3f} delta={fr['delta']:+.3f}"
        )
    else:
        print(f"- fraud_ratio={fr['ratio']:.4f}")
    print(f"- dup_txid={dup_txid}, nonpositive_amounts={nonpos_amt}")
    h = report["metrics"]["heuristics"]
    print(
        f"- heuristics: bursts={h['velocity_bursts']}, ato_like={h['ato_like_users']}, struct={h['structuring_txns']}, micro={h['micro_amount_txns']}"
    )
    if "auc" in feat_res and feat_res["auc"]:
        top = sorted(feat_res["auc"].items(), key=lambda kv: -kv[1])[:3]
        print(f"- features AUC top3: {top}")
    elif "means" in feat_res and feat_res["means"]:
        print("- features means computed (sklearn not available)")
    if failures:
        print("FAILED checks:", ", ".join(failures))
    else:
        print("All checks passed")

    # Write report
    outp = Path(args.report_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {outp}")

    return 1 if (args.strict and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
