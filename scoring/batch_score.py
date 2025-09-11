from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


def _read_blob_bytes(settings: Settings, blob_path: str) -> bytes:
    """Read blob bytes from ADLS using connection string or AAD credentials.

    blob_path is a container-relative path like "gold/txn_features.csv" or a full https URL.
    """
    try:
        from azure.storage.blob import BlobClient, BlobServiceClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError("azure-storage-blob is required for ADLS mode") from e

    # Full URL case
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


def _read_csv_dual(settings: Settings, path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    if path.startswith("file://"):
        return pd.read_csv(path.replace("file://", ""))
    if settings.storage_mode == "adls":
        data = _read_blob_bytes(settings, path)
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def _read_json_dual(settings: Settings, path: str) -> Dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    if path.startswith("file://"):
        return json.loads(Path(path.replace("file://", "")).read_text(encoding="utf-8"))
    if settings.storage_mode == "adls":
        data = _read_blob_bytes(settings, path)
        return json.loads(data.decode("utf-8"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_model_dual(settings: Settings, path: str):
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


def _ensure_feature_matrix(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    X = df.copy()
    # Add missing columns with zeros, drop extras, order columns
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_batch(
    settings: Settings,
    in_csv: str,
    model_path: str,
    features_json: str | None,
    out_path: str,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, str]:
    # Load inputs
    df = _read_csv_dual(settings, in_csv)
    model_art = _read_model_dual(settings, model_path)
    model = model_art.get("model", model_art)
    expected_feats = model_art.get("features")
    if features_json:
        fj = _read_json_dual(settings, features_json)
        expected_feats = fj.get("features", expected_feats)
    if expected_feats is None:
        raise ValueError("Expected feature list not found in model pickle; provide --features-json")

    # Build X
    X = _ensure_feature_matrix(df, expected_feats)

    # Score
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        scores = _sigmoid(raw)
    else:
        preds = model.predict(X)
        scores = preds.astype(float)
    decisions = (scores >= float(threshold)).astype(int)

    # Compose output: keep ids if present
    out_cols = []
    for c in ("transactionId", "userId"):
        if c in df.columns:
            out_cols.append(c)
    out = pd.DataFrame({"score": scores, "decision": decisions})
    if out_cols:
        out = pd.concat([df[out_cols].reset_index(drop=True), out], axis=1)

    # Write out
    # Prefer local write if path exists/points to filesystem. Otherwise route via ADLS.
    p = Path(out_path)
    if p.parent.exists() or out_path.startswith("./") or out_path.startswith("/"):
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
        return out, str(p)
    if settings.storage_mode == "adls":
        subdir = str(Path(out_path).parent).strip("./")
        name = Path(out_path).name
        url = storage_route_write_bytes(
            settings,
            subdir=subdir if subdir not in ("", ".") else "gold",
            name=name,
            data=out.to_csv(index=False).encode("utf-8"),
            content_type="text/csv; charset=utf-8",
        )
        return out, url
    # Fallback: attempt local write
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out, str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch score Gold features using a trained model (local or ADLS)")
    parser.add_argument("--in-path", required=True, help="Input features CSV (local path or ADLS blob path)")
    parser.add_argument("--model-path", required=True, help="Model pickle path (local or ADLS blob path)")
    parser.add_argument("--features-json", default=None, help="Optional features JSON path (local or ADLS)")
    parser.add_argument("--out-path", required=True, help="Output scored CSV path (local or ADLS blob path)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    settings = Settings()
    df, dest = score_batch(
        settings=settings,
        in_csv=args.in_path,
        model_path=args.model_path,
        features_json=args.features_json,
        out_path=args.out_path,
        threshold=args.threshold,
    )
    print(f"Wrote scored output: {dest} ({len(df)} rows)")


if __name__ == "__main__":
    main()

