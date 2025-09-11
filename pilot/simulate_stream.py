from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd

from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


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


def _slice_rows(df: pd.DataFrame, feature_cols: List[str], id_cols: List[str]) -> List[Dict]:
    # Keep id columns if present; pass full rows to service
    cols = [c for c in id_cols if c in df.columns] + feature_cols
    sub = df[cols].copy()
    return sub.to_dict(orient="records")


def simulate(
    settings: Settings,
    source_path: str,
    endpoint: str,
    threshold: float,
    rate: float,
    batch_size: int,
    feature_list_path: str,
    metrics_out: str | None = None,
) -> Tuple[int, float, float]:
    df = _read_csv_dual(settings, source_path)
    # Ensure JSON-safe numeric values (no NaN/Inf), since httpx/json is RFC compliant
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Load expected feature list
    feats = json.loads(Path(feature_list_path).read_text(encoding="utf-8")) if Path(feature_list_path).exists() else None
    if feats is None and settings.storage_mode == "adls":
        data = _read_blob_bytes(settings, feature_list_path)
        feats = json.loads(data.decode("utf-8"))
    feature_cols = feats.get("features") if feats else [c for c in df.columns if c not in ("transactionId", "userId", "is_fraud")]

    id_cols = ["transactionId", "userId"]
    client = httpx.Client(timeout=10.0)
    per_req_sleep = 0.0 if rate <= 0 else max(0.0, 1.0 / float(rate))

    t0 = time.perf_counter()
    total = 0
    latencies: List[float] = []
    alerts = 0

    # Prepare rows
    rows = _slice_rows(df, feature_cols, id_cols)
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        # Final safety: coerce any lingering NaN/Inf to 0.0 in batch
        for rec in batch:
            for k, v in list(rec.items()):
                if isinstance(v, float):
                    if np.isnan(v) or np.isinf(v):
                        rec[k] = 0.0
                elif v is None:
                    rec[k] = 0
        payload = {"rows": batch, "threshold": threshold}
        t1 = time.perf_counter()
        r = client.post(endpoint.rstrip("/") + "/score", json=payload)
        r.raise_for_status()
        dt = (time.perf_counter() - t1) * 1000.0
        latencies.append(dt)
        resp = r.json()
        decisions = resp.get("decisions", [])
        alerts += int(np.sum(decisions))
        total += len(batch)
        if per_req_sleep > 0:
            time.sleep(per_req_sleep)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    summary = {
        "total_scored": total,
        "alerts": alerts,
        "alert_rate": float(alerts) / float(total) if total else 0.0,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "threshold": threshold,
        "batch_size": batch_size,
        "rate_per_sec": rate,
    }

    if metrics_out:
        # Write a single-line JSON summary; route via ADLS if requested path is container-relative
        if Path(metrics_out).parent.exists() or metrics_out.startswith("./") or metrics_out.startswith("/"):
            Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
            Path(metrics_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            storage_route_write_bytes(
                settings,
                subdir=str(Path(metrics_out).parent).strip("./") or "pilot",
                name=str(Path(metrics_out).name),
                data=json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json; charset=utf-8",
            )

    return total, p50, p95


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a pilot stream to the scoring API")
    parser.add_argument("--source", required=True, help="Gold features CSV (local or ADLS path)")
    parser.add_argument("--endpoint", required=True, help="Scoring endpoint base URL (e.g., http://localhost:8000)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=20.0, help="Requests per second (approx)")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--features-json", default="models/baseline_xgb.features.json", help="Feature list JSON (local or ADLS)")
    parser.add_argument("--metrics-out", default=None, help="Where to write summary metrics (local or ADLS)")
    args = parser.parse_args()

    settings = Settings()
    total, p50, p95 = simulate(
        settings,
        source_path=args.source,
        endpoint=args.endpoint,
        threshold=args.threshold,
        rate=args.rate,
        batch_size=args.batch_size,
        feature_list_path=args.features_json,
        metrics_out=args.metrics_out,
    )
    print(f"Scored {total} rows. Latency p50={p50:.2f}ms, p95={p95:.2f}ms")


if __name__ == "__main__":
    main()
