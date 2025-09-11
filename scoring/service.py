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

    def load(self) -> None:
        # Load policy.yaml if present (local only for simplicity)
        pol_path = Path("ML_Fraud/configs/policy.yaml")
        if pol_path.exists():
            try:
                data = yaml.safe_load(pol_path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict) and "threshold" in data:
                    self.threshold = float(data["threshold"])
            except Exception:
                pass

        model_path = os.getenv("MODEL_PATH", "models/baseline_xgb.pkl")
        features_json = os.getenv("FEATURES_JSON", "")
        # Prefer local defaults when present to make local runs easy
        local_model = Path("ML_Fraud/data/models/baseline_xgb.pkl")
        local_feats = Path("ML_Fraud/data/models/baseline_xgb.features.json")
        if not Path(model_path).exists() and local_model.exists():
            model_path = str(local_model)
        if (not features_json or not Path(features_json).exists()) and local_feats.exists():
            features_json = str(local_feats)

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

    def score(self, rows: List[Dict[str, Any]], threshold: Optional[float]) -> ScoreResponse:
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
