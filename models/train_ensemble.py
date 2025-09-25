from __future__ import annotations

import argparse
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)

from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


def _read_gold_csv(settings: Settings, gold_path: str) -> pd.DataFrame:
    p = Path(gold_path)
    if p.exists():
        return pd.read_csv(p)
    if gold_path.startswith("file://"):
        return pd.read_csv(gold_path.replace("file://", ""))
    if settings.storage_mode != "adls":
        return pd.read_csv(gold_path)

    # ADLS mode
    try:
        from azure.storage.blob import BlobClient, BlobServiceClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("azure-storage-blob is required for ADLS mode") from e

    if gold_path.startswith("https://"):
        bc = BlobClient.from_blob_url(gold_path)
        data = bc.download_blob().readall()
        return pd.read_csv(io.BytesIO(data))

    if settings.storage_connection_string:
        svc = BlobServiceClient.from_connection_string(settings.storage_connection_string)
    else:
        if not settings.storage_account_url:
            raise ValueError("STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING must be set for ADLS mode")
        from azure.identity import DefaultAzureCredential  # type: ignore

        cred = DefaultAzureCredential()
        svc = BlobServiceClient(account_url=settings.storage_account_url, credential=cred)

    bc = svc.get_blob_client(container=settings.storage_container, blob=gold_path)
    data = bc.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


def _select_features(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    cols = [c for c in df.columns if c not in set(drop_cols + [target_col])]
    X = df[cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].astype(int)
    return X, y, cols


def _forward_time_splits(event_ts: np.ndarray, n_folds: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(event_ts)
    N = len(order)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_folds + 1):
        split = int(np.floor(k * (N / (n_folds + 1))))
        train_idx = order[:split]
        test_idx = order[split: int(np.floor((k + 1) * (N / (n_folds + 1))))] if k < n_folds else order[split:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    if not folds:
        # Fallback single split 80/20
        cut = int(0.8 * N)
        folds = [(order[:cut], order[cut:])]
    return folds


def _random_params_xgb(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.choice([300, 500])),
        "max_depth": int(rng.choice([4, 5, 6])),
        "learning_rate": float(rng.choice([0.05, 0.1, 0.2])),
        "subsample": float(rng.choice([0.7, 0.8, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.8, 1.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0])),
    }


class StackedEnsemble:
    def __init__(self, xgb_params: Dict[str, Any], random_state: int, scale_pos_weight: float) -> None:
        from xgboost import XGBClassifier  # type: ignore

        self.xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            **xgb_params,
        )
        self.lr = LogisticRegression(max_iter=200, n_jobs=-1, class_weight=None)
        self.iforest = None  # set in fit

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        # Isolation Forest on non-fraud (or all if none labeled)
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore

            mask = (y == 0)
            X_train_if = X[mask] if mask.any() else X
            iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
            iforest.fit(X_train_if)
            self.iforest = iforest
            iso_all = iforest.score_samples(X)
            iso_rank = 1.0 - (pd.Series(iso_all).rank(method="average") / len(X))
            iso_feat = iso_rank.to_numpy().reshape(-1, 1)
        except Exception:
            self.iforest = None
            iso_feat = np.zeros((len(X), 1))

        # Base XGB on full X
        self.xgb.fit(X, y)
        xgb_p = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)

        # Meta LR on stacked features [xgb_p, iforest_score]
        Z = np.hstack([xgb_p, iso_feat])
        self.lr.fit(Z, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        xgb_p = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
        if self.iforest is not None:
            iso_all = self.iforest.score_samples(X)
            iso_rank = 1.0 - (pd.Series(iso_all).rank(method="average") / len(X))
            iso_feat = iso_rank.to_numpy().reshape(-1, 1)
        else:
            iso_feat = np.zeros((len(X), 1))
        Z = np.hstack([xgb_p, iso_feat])
        # Logistic meta outputs probability directly via predict_proba
        # If not available, approximate via decision_function and sigmoid
        if hasattr(self.lr, "predict_proba"):
            return np.vstack([1 - self.lr.predict_proba(Z)[:, 1], self.lr.predict_proba(Z)[:, 1]]).T
        else:
            raw = self.lr.decision_function(Z)
            s = 1.0 / (1.0 + np.exp(-raw))
            return np.vstack([1 - s, s]).T


@dataclass
class TrainEnsembleConfig:
    gold_path: str
    target_col: str = "is_fraud"
    drop_cols: Tuple[str, ...] = (
        "transactionId",
        "userId",
        "event_ts",
    )
    out_subdir: str = "models"
    out_name: str = "stack_v1"
    outer_folds: int = 3
    random_state: int = 42
    trials: int = 30


def train_ensemble_and_save(cfg: TrainEnsembleConfig, settings: Settings) -> Tuple[str, str, Dict[str, Any]]:
    df = _read_gold_csv(settings, cfg.gold_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset. Ensure labels are present.")

    event_ts = df["event_ts"].to_numpy() if "event_ts" in df.columns else np.arange(len(df))
    X_all, y_all, feat_cols = _select_features(df, cfg.target_col, list(cfg.drop_cols))

    # Outer forward folds for evaluation and simple HPO selection
    folds = _forward_time_splits(event_ts, cfg.outer_folds)
    rng = np.random.default_rng(cfg.random_state)

    outer_metrics: List[Dict[str, float]] = []
    best_params: Dict[str, Any] | None = None
    best_ap = -np.inf

    # Lightweight param search across outer folds: pick params yielding best avg AP on val
    for _ in range(cfg.trials):
        params = _random_params_xgb(rng)
        ap_scores: List[float] = []
        for tr_idx, te_idx in folds:
            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
            pos = int((y_tr == 1).sum())
            neg = int((y_tr == 0).sum())
            if pos == 0:
                continue
            spw = max(1.0, float(neg) / float(pos))
            model = StackedEnsemble(xgb_params=params, random_state=cfg.random_state, scale_pos_weight=spw)
            model.fit(X_tr, y_tr.to_numpy())
            proba = model.predict_proba(X_te)[:, 1]
            ap = float(average_precision_score(y_te, proba))
            ap_scores.append(ap)
        if ap_scores:
            mean_ap = float(np.mean(ap_scores))
            if mean_ap > best_ap:
                best_ap = mean_ap
                best_params = params

    if best_params is None:
        best_params = _random_params_xgb(rng)

    # Final evaluation across outer folds with best params
    pr_points: List[Tuple[float, float]] = []  # (recall, precision) averaged later
    auc_prs: List[float] = []
    auc_rocs: List[float] = []
    prec_at5: List[float] = []
    cap_at5: List[float] = []
    fprs: List[float] = []
    for tr_idx, te_idx in folds:
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = max(1.0, float(neg) / float(pos)) if pos > 0 else 1.0
        model = StackedEnsemble(xgb_params=best_params, random_state=cfg.random_state, scale_pos_weight=spw)
        model.fit(X_tr, y_tr.to_numpy())
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        # Metrics
        auc_pr = float(average_precision_score(y_te, proba))
        auc_roc = float(roc_auc_score(y_te, proba))
        auc_prs.append(auc_pr)
        auc_rocs.append(auc_roc)
        k = max(1, int(np.ceil(0.05 * len(y_te))))
        order = np.argsort(-proba)
        top_idx = order[:k]
        y_top = y_te.iloc[top_idx]
        prec_at5.append(float(int((y_top == 1).sum()) / k))
        cap_at5.append(float(int((y_top == 1).sum()) / max(1, int((y_te == 1).sum()))))
        tn = int(((y_te == 0) & (pred == 0)).sum())
        fp = int(((y_te == 0) & (pred == 1)).sum())
        fprs.append(float(fp / max(1, (fp + tn))))
        # PR curve points (store for the last fold to avoid size blow-up)
        r, p, _ = precision_recall_curve(y_te, proba)
        pr_points = list(zip(map(float, r), map(float, p)))

    metrics = {
        "auc_pr_mean": float(np.mean(auc_prs)) if auc_prs else 0.0,
        "auc_roc_mean": float(np.mean(auc_rocs)) if auc_rocs else 0.0,
        "precision@5pct_mean": float(np.mean(prec_at5)) if prec_at5 else 0.0,
        "capture@5pct_mean": float(np.mean(cap_at5)) if cap_at5 else 0.0,
        "fpr@0.5_mean": float(np.mean(fprs)) if fprs else 0.0,
        "outer_folds": len(folds),
        "xgb_params": best_params,
    }

    # Fit final model on all data with best params and save
    pos_all = int((y_all == 1).sum())
    neg_all = int((y_all == 0).sum())
    spw_all = max(1.0, float(neg_all) / float(pos_all)) if pos_all > 0 else 1.0
    final_model = StackedEnsemble(xgb_params=best_params, random_state=cfg.random_state, scale_pos_weight=spw_all)
    final_model.fit(X_all, y_all.to_numpy())

    # Choose a threshold based on PR curve target (maximize F1 on last fold curve as proxy)
    best_thr = 0.5
    best_f1 = -1.0
    for rec, prec in pr_points:
        if prec + rec == 0:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            # Find an approximate threshold by percentile of scores matching recall
            # Leave as 0.5 if not easily derivable; the scoring service can use policy.yaml
    
    # Persist artifacts
    out_prefix = cfg.out_name
    model_bytes = pickle.dumps({"model": final_model, "features": feat_cols}, protocol=pickle.HIGHEST_PROTOCOL)
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
    pr_url = storage_route_write_bytes(
        settings,
        subdir=cfg.out_subdir,
        name=f"{out_prefix}.pr_curves.json",
        data=json.dumps({"pr": pr_points}, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )

    # Threshold recommendations using all-data scores
    try:
        proba_all = final_model.predict_proba(X_all)[:, 1]
        prec, rec, thr = precision_recall_curve(y_all, proba_all)
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
        thr_top5 = float(np.quantile(proba_all, 1 - 0.05)) if len(proba_all) else 0.5

        def _summ(name: str, t: float | None) -> dict:
            if t is None:
                return {"name": name, "note": "no threshold meets target"}
            yhat = (proba_all >= float(t)).astype(int)
            return {
                "name": name,
                "threshold": float(t),
                "precision": float(precision_score(y_all, yhat, zero_division=0)),
                "recall": float(recall_score(y_all, yhat, zero_division=0)),
                "pos_rate": float(yhat.mean()),
            }

        thresholds_obj = {
            "auc_roc": float(roc_auc_score(y_all, proba_all)),
            "auc_pr": float(average_precision_score(y_all, proba_all)),
            "thresholds": [
                _summ("max_f1", thr_f1),
                _summ("prec>=0.995", thr_p995),
                _summ("prec>=0.99", thr_p99),
                _summ("top5pct", thr_top5),
            ],
        }
        _ = storage_route_write_bytes(
            settings,
            subdir=cfg.out_subdir,
            name=f"{out_prefix}.thresholds.json",
            data=json.dumps(thresholds_obj, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
    except Exception:
        pass

    return model_url, feats_url, {"metrics": metrics, "pr_curves": pr_url}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ensemble fraud model with temporal CV and stacking")
    parser.add_argument(
        "--gold-path",
        default="ML_Fraud/data/gold/txn_synth_features.csv",
        help="Path to Gold features CSV (local path or ADLS)",
    )
    parser.add_argument("--out-subdir", default="models", help="Subdirectory under data root or container")
    parser.add_argument("--out-name", default="stack_v1", help="Artifact base name (without extension)")
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    settings = Settings()
    cfg = TrainEnsembleConfig(
        gold_path=args.gold_path,
        out_subdir=args.out_subdir,
        out_name=args.out_name,
        outer_folds=args.outer_folds,
        trials=args.trials,
    )
    model_url, feats_url, extras = train_ensemble_and_save(cfg, settings)
    print("Artifacts written:")
    print(f"  model:    {model_url}")
    print(f"  features: {feats_url}")
    print(f"  metrics:  {settings.data_root / args.out_subdir / (args.out_name + '.metrics.json')}")
    print(f"  pr_curve: {settings.data_root / args.out_subdir / (args.out_name + '.pr_curves.json')}")


if __name__ == "__main__":
    main()
