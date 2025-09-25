from __future__ import annotations

import os
import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Body
from pydantic import BaseModel
import time
from collections import deque

from ingestion.config import Settings


app = FastAPI(title="ML_Fraud Scoring Service", version="0.1.0")


def _read_blob_bytes(settings: Settings, blob_path: str) -> bytes:
    try:
        from azure.storage.blob import BlobClient, BlobServiceClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError("azure-storage-blob is required for ADLS mode") from e

    if blob_path.startswith("https://"):
        bc = BlobClient.from_blob_url(blob_path)
        return bc.download_blob().readall()

    if settings.storage_connection_string:
        svc = BlobServiceClient.from_connection_string(settings.storage_connection_string)
    else:
        if not settings.storage_account_url:
            raise ValueError("STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING must be set for ADLS mode")
        from azure.identity import DefaultAzureCredential

        cred = DefaultAzureCredential()
        svc = BlobServiceClient(account_url=settings.storage_account_url, credential=cred)

    bc = svc.get_blob_client(container=settings.storage_container, blob=blob_path)
    return bc.download_blob().readall()


def _load_pickle_dual(settings: Settings, path: str) -> Any:
    p = Path(path)
    if p.exists():
        return pickle.loads(p.read_bytes())
    if path.startswith("file://"):
        fp = path.replace("file://", "")
        return pickle.loads(Path(fp).read_bytes())
    if settings.storage_mode == "adls":
        data = _read_blob_bytes(settings, path)
        return pickle.loads(data)
    return pickle.loads(Path(path).read_bytes())


def _load_json_dual(settings: Settings, path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    if path.startswith("file://"):
        return json.loads(Path(path.replace("file://", "")).read_text(encoding="utf-8"))
    if settings.storage_mode == "adls":
        data = _read_blob_bytes(settings, path)
        return json.loads(data.decode("utf-8"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_feature_matrix(rows: List[Dict[str, Any]], expected: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ScoreRequest(BaseModel):
    rows: List[Dict[str, Any]]
    threshold: Optional[float] = None


class ScoreResponse(BaseModel):
    scores: List[float]
    decisions: List[int]


class AppState:
    def __init__(self) -> None:
        self.settings = Settings()
        self.model = None
        self.features: List[str] | None = None
        self.threshold: float = float(os.getenv("THRESHOLD", "0.5"))
        # Monitoring buffers
        self.latencies_ms: deque[float] = deque(maxlen=500)
        self.recent_scores: deque[float] = deque(maxlen=2000)
        self.total_requests: int = 0
        self.thresholds_info: Dict[str, Any] | None = None

    def load(self) -> None:
        # Load policy.yaml if present (local only for simplicity)
        pol_path = Path("ML_Fraud/configs/policy.yaml")
        policy_loaded = False
        if pol_path.exists():
            try:
                data = yaml.safe_load(pol_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict) and "threshold" in data:
                    self.threshold = float(data["threshold"])
                policy_loaded = True
            except Exception:
                pass

        # Resolve artifacts: env → champion → baseline
        model_path_env = os.getenv("MODEL_PATH")
        features_json_env = os.getenv("FEATURES_JSON")
        model_path = model_path_env if model_path_env else ""
        features_json = features_json_env if features_json_env else ""

        champion_model = Path("ML_Fraud/data/models/champion/model.pkl")
        champion_feats = Path("ML_Fraud/data/models/champion/features.json")
        baseline_model = Path("ML_Fraud/data/models/baseline_xgb.pkl")
        baseline_feats = Path("ML_Fraud/data/models/baseline_xgb.features.json")

        # If env not set or not found, prefer champion, then baseline
        if not model_path or not Path(model_path).exists():
            if champion_model.exists():
                model_path = str(champion_model)
            elif baseline_model.exists():
                model_path = str(baseline_model)
            else:
                # fallback to relative default (may be ADLS path in other setups)
                model_path = "models/baseline_xgb.pkl"
        if (not features_json or not Path(features_json).exists()):
            if champion_feats.exists():
                features_json = str(champion_feats)
            elif baseline_feats.exists():
                features_json = str(baseline_feats)

        art = _load_pickle_dual(self.settings, model_path)
        self.model = art.get("model", art)
        self.features = art.get("features")
        if features_json:
            fj = _load_json_dual(self.settings, features_json)
            self.features = fj.get("features", self.features)
        if self.features is None:
            raise RuntimeError(
                f"Feature list not found. Ensure model pickle contains it or set FEATURES_JSON. Model path used: {model_path}"
            )

        # Try load threshold recommendations JSON living next to the model (same base name)
        self.thresholds_info = None
        try:
            def _load_thr(pth: Path) -> Dict[str, Any] | None:
                if pth.suffix != ".pkl":
                    return None
                base = pth.with_suffix("")
                tj = Path(base.as_posix() + ".thresholds.json")
                if tj.exists():
                    return json.loads(tj.read_text(encoding="utf-8"))
                return None

            p = Path(model_path)
            info = _load_thr(p)
            if info is None:
                # Try resolved target (handles champion symlinks)
                rp = p.resolve() if p.exists() else p
                info = _load_thr(rp)
            self.thresholds_info = info
        except Exception:
            self.thresholds_info = None

        # Auto-select a reasonable threshold if no policy and no env var provided
        try:
            if (not policy_loaded) and ("THRESHOLD" not in os.environ) and self.thresholds_info:
                ts = self.thresholds_info.get("thresholds", [])
                def _pick(name: str) -> float | None:
                    for t in ts:
                        if t.get("name") == name and "threshold" in t:
                            return float(t.get("threshold"))
                    return None
                chosen = _pick("top5pct") or _pick("max_f1")
                if chosen is not None:
                    self.threshold = float(chosen)
        except Exception:
            pass

    def score(self, rows: List[Dict[str, Any]], threshold: Optional[float]) -> ScoreResponse:
        start = time.perf_counter()
        thr = float(threshold) if threshold is not None else self.threshold
        X = _ensure_feature_matrix(rows, self.features or [])
        m = self.model
        if hasattr(m, "predict_proba"):
            scores = m.predict_proba(X)[:, 1]
        elif hasattr(m, "decision_function"):
            raw = m.decision_function(X)
            scores = _sigmoid(raw)
        else:
            preds = m.predict(X)
            scores = preds.astype(float)
        decisions = (scores >= thr).astype(int)
        # Monitoring updates
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            self.latencies_ms.append(float(elapsed_ms))
            for s in scores:
                self.recent_scores.append(float(s))
            self.total_requests += 1
        except Exception:
            pass
        return ScoreResponse(scores=[float(s) for s in scores], decisions=[int(d) for d in decisions])


state = AppState()


@app.on_event("startup")
def _startup() -> None:  # pragma: no cover
    state.load()


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest = Body(...)) -> ScoreResponse:
    return state.score(req.rows, req.threshold)


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    # Basic operational + drift-ish indicators
    lats = list(state.latencies_ms)
    scores = list(state.recent_scores)
    def _percentile(arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        a = np.array(arr)
        return float(np.percentile(a, q))
    avg_lat = float(np.mean(lats)) if lats else 0.0
    p95_lat = _percentile(lats, 95.0)
    mean_score = float(np.mean(scores)) if scores else 0.0
    high_rate = float(np.mean([1.0 if s >= state.threshold else 0.0 for s in scores])) if scores else 0.0
    return {
        "total_requests": state.total_requests,
        "latency_ms_avg": round(avg_lat, 3),
        "latency_ms_p95": round(p95_lat, 3),
        "recent_score_mean": round(mean_score, 6),
        "recent_high_rate": round(high_rate, 6),
        "buffer_sizes": {"latencies": len(lats), "scores": len(scores)},
    }


@app.post("/reload")
def reload_model() -> Dict[str, Any]:  # pragma: no cover
    try:
        state.load()
        return {
            "status": "reloaded",
            "threshold": state.threshold,
            "has_thresholds_info": bool(state.thresholds_info is not None),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/thresholds")
def thresholds() -> Dict[str, Any]:
    if state.thresholds_info is None:
        return {"present": False}
    return {"present": True, "data": state.thresholds_info}
