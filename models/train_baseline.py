from __future__ import annotations

import argparse
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


def _read_gold_csv(settings: Settings, gold_path: str) -> pd.DataFrame:
    """Load Gold features from local or ADLS.

    - local: `gold_path` is a filesystem path
    - adls:  `gold_path` is a blob path under the container (e.g., `gold/txn_features.csv`) or a full https URL
    """
    # Always prefer local file if it exists (supports local runs even when STORAGE_MODE=adls)
    p = Path(gold_path)
    if p.exists():
        return pd.read_csv(p)
    if gold_path.startswith("file://"):
        return pd.read_csv(gold_path.replace("file://", ""))
    if settings.storage_mode != "adls":
        return pd.read_csv(gold_path)

    # ADLS mode
    try:
        from azure.storage.blob import BlobClient, BlobServiceClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError("azure-storage-blob is required for ADLS mode") from e

    # Full URL case
    if gold_path.startswith("https://"):
        bc = BlobClient.from_blob_url(gold_path)
        data = bc.download_blob().readall()
        return pd.read_csv(io.BytesIO(data))

    # Container-relative path
    if settings.storage_connection_string:
        svc = BlobServiceClient.from_connection_string(settings.storage_connection_string)
    else:
        if not settings.storage_account_url:
            raise ValueError("STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING must be set for ADLS mode")
        from azure.identity import DefaultAzureCredential

        cred = DefaultAzureCredential()
        svc = BlobServiceClient(account_url=settings.storage_account_url, credential=cred)

    bc = svc.get_blob_client(container=settings.storage_container, blob=gold_path)
    data = bc.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


def _select_features(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    cols = [c for c in df.columns if c not in set(drop_cols + [target_col])]
    X = df[cols].copy()
    y = df[target_col].astype(int)
    # Basic imputation: replace inf and NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y, cols


@dataclass
class TrainConfig:
    gold_path: str
    target_col: str = "is_fraud"
    drop_cols: Tuple[str, ...] = (
        "transactionId",
        "userId",
        "event_ts",
    )
    out_subdir: str = "models"
    out_name: str = "baseline_xgb"
    test_size: float = 0.2
    random_state: int = 42
    time_split: bool = True


def train_and_save(cfg: TrainConfig, settings: Settings) -> Tuple[str, str, str]:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    try:
        from xgboost import XGBClassifier
    except Exception as e:  # pragma: no cover
        raise RuntimeError("xgboost is required. Please install it and retry.") from e

    df = _read_gold_csv(settings, cfg.gold_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset. Ensure labels are present.")

    # Preserve event_ts (if present) for temporal split before dropping from features
    event_ts = df["event_ts"].to_numpy() if "event_ts" in df.columns else None
    X, y, feat_cols = _select_features(df, cfg.target_col, list(cfg.drop_cols))
    if cfg.time_split and event_ts is not None:
        order = np.argsort(event_ts)
        X_sorted, y_sorted = X.iloc[order], y.iloc[order]
        split_idx = int((1.0 - cfg.test_size) * len(X_sorted))
        X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )

    # Handle class imbalance: scale_pos_weight = neg/pos
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        raise ValueError("Training set has zero positive samples; cannot train a classifier.")
    spw = max(1.0, float(neg) / float(pos))

    # Lightweight hyperparameter search with temporal validation (last 20% of train)
    def _sample_params(rng: np.random.Generator) -> Dict[str, float | int]:
        return {
            "n_estimators": int(rng.choice([200, 300, 500])),
            "max_depth": int(rng.choice([3, 4, 5, 6])),
            "learning_rate": float(rng.choice([0.05, 0.1, 0.2])),
            "subsample": float(rng.choice([0.7, 0.8, 1.0])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 2.0])),
        }

    rng = np.random.default_rng(cfg.random_state)
    # Build temporal validation split inside train
    n_train = len(X_train)
    cut = int(max(1, np.floor(0.8 * n_train)))
    X_tr, X_val = X_train.iloc[:cut], X_train.iloc[cut:]
    y_tr, y_val = y_train.iloc[:cut], y_train.iloc[cut:]

    best_params = None
    best_ap = -np.inf
    for _ in range(10):
        params = _sample_params(rng)
        model_tmp = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=cfg.random_state,
            scale_pos_weight=spw,
            **params,
        )
        model_tmp.fit(X_tr, y_tr)
        proba_val = model_tmp.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, proba_val))
        if ap > best_ap:
            best_ap = ap
            best_params = params

    # Fallback to defaults if search failed
    if best_params is None:
        best_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=cfg.random_state,
        scale_pos_weight=spw,
        **best_params,
    )
    model.fit(X_train, y_train)

    # Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
    }
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    # False Positive Rate at 0.5
    tn = int(((y_test == 0) & (pred == 0)).sum())
    fp = int(((y_test == 0) & (pred == 1)).sum())
    fpr = float(fp / max(1, (fp + tn)))
    metrics.update({
        "precision@0.5": float(p),
        "recall@0.5": float(r),
        "f1@0.5": float(f1),
        "fpr@0.5": fpr,
    })
    # Precision@K and capture@K for top 5% risk
    k = max(1, int(np.ceil(0.05 * len(y_test))))
    order = np.argsort(-proba)
    top_idx = order[:k]
    y_top = y_test.iloc[top_idx]
    prec_at_k = float(int((y_top == 1).sum()) / k)
    capture_at_k = float(int((y_top == 1).sum()) / max(1, int((y_test == 1).sum())))
    metrics.update({
        "precision@5pct": prec_at_k,
        "capture@5pct": capture_at_k,
        "k@5pct": k,
    })

    # Threshold recommendations (based on current test split)
    try:
        prec, rec, thr = precision_recall_curve(y_test, proba)
        # max F1
        denom = (prec + rec)
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
        i_f1 = int(np.nanargmax(f1)) if len(f1) else 0
        thr_f1 = float(thr[max(0, i_f1 - 1)]) if len(thr) else 0.5

        def _thr_for_precision(target: float) -> float | None:
            mask = prec >= target
            idx = np.where(mask)[0]
            if len(idx) == 0 or len(thr) == 0:
                return None
            j = int(idx[-1])
            return float(thr[max(0, j - 1)])

        thr_p995 = _thr_for_precision(0.995)
        thr_p99 = _thr_for_precision(0.99)
        thr_top5 = float(np.quantile(proba, 1 - 0.05)) if len(proba) else 0.5

        def _summarize(name: str, t: float | None) -> dict:
            if t is None:
                return {"name": name, "note": "no threshold meets target"}
            yhat = (proba >= float(t)).astype(int)
            return {
                "name": name,
                "threshold": float(t),
                "precision": float((y_test[yhat == 1] == 1).mean() if (yhat == 1).any() else 0.0),
                "recall": float((yhat[y_test == 1] == 1).mean() if (y_test == 1).any() else 0.0),
                "pos_rate": float(yhat.mean()),
            }

        thresholds_obj = {
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "thresholds": [
                _summarize("max_f1", thr_f1),
                _summarize("prec>=0.995", thr_p995),
                _summarize("prec>=0.99", thr_p99),
                _summarize("top5pct", thr_top5),
            ],
        }
    except Exception:
        thresholds_obj = {"note": "failed to compute thresholds"}

    # Persist artifacts (model, feature list, metrics) via storage router
    out_prefix = cfg.out_name
    model_bytes = pickle.dumps({"model": model, "features": feat_cols}, protocol=pickle.HIGHEST_PROTOCOL)
    model_url = storage_route_write_bytes(
        settings,
        subdir=cfg.out_subdir,
        name=f"{out_prefix}.pkl",
        data=model_bytes,
        content_type="application/octet-stream",
    )
    feats_url = storage_route_write_bytes(
        settings,
        subdir=cfg.out_subdir,
        name=f"{out_prefix}.features.json",
        data=json.dumps({"features": feat_cols}, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )
    metrics_url = storage_route_write_bytes(
        settings,
        subdir=cfg.out_subdir,
        name=f"{out_prefix}.metrics.json",
        data=json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )
    # Thresholds recommendations JSON
    _ = storage_route_write_bytes(
        settings,
        subdir=cfg.out_subdir,
        name=f"{out_prefix}.thresholds.json",
        data=json.dumps(thresholds_obj, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )
    return model_url, feats_url, metrics_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline fraud model on Gold features (local or ADLS)")
    parser.add_argument(
        "--gold-path",
        required=True,
        help="Path to Gold features: local file path or ADLS blob path (e.g., gold/txn_features.csv or full https URL)",
    )
    parser.add_argument("--target-col", default="is_fraud")
    parser.add_argument(
        "--drop-cols",
        default="transactionId,userId",
        help="Comma-separated non-feature columns to drop",
    )
    parser.add_argument("--out-subdir", default="models", help="Subdirectory under DATA_ROOT or ADLS container")
    parser.add_argument("--out-name", default="baseline_xgb", help="Output artifact base name (without extension)")
    args = parser.parse_args()

    settings = Settings()
    cfg = TrainConfig(
        gold_path=args.gold_path,
        target_col=args.target_col,
        drop_cols=tuple([c for c in args.drop_cols.split(",") if c]),
        out_subdir=args.out_subdir,
        out_name=args.out_name,
    )
    model_url, feats_url, metrics_url = train_and_save(cfg, settings)
    print("Artifacts written:")
    print(f"  model:   {model_url}")
    print(f"  features:{feats_url}")
    print(f"  metrics: {metrics_url}")


if __name__ == "__main__":
    main()
