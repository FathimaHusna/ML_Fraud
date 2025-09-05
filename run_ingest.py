from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

from ingestion.config import Settings
from ingestion.di_client import analyze_id_document, analyze_id_document_from_file
from ingestion.normalize import normalize_id
from ingestion.storage import write_json_local, storage_route_write_json
from ingestion.azure_blob import BlobHelper


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DI ingestion and normalization")
    parser.add_argument("--submission-id", required=True, help="Submission identifier")
    parser.add_argument("--blob-url", required=False, help="Blob URL to the document (SAS or public)")
    parser.add_argument("--use-sample", action="store_true", help="Use local sample instead of calling Azure")
    parser.add_argument("--upload-path", required=False, help="Local file to upload to Azure Blob (requires STORAGE_MODE=adls)")
    parser.add_argument("--idp-bronze-path", required=False, help="Path or URL to IDP bronze JSON (decoupled mode; skips DI call)")
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_dirs()

    raw: Any
    # Decoupled mode: consume bronze JSON directly
    if args.idp_bronze_path:
        bronze_path = args.idp_bronze_path
        # Local file path
        p = Path(bronze_path)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            if settings.storage_mode == "adls" and not bronze_path.lower().startswith("http"):
                helper = BlobHelper(settings)
                # treat as path under container, e.g., idp/bronze/YYYY/MM/DD/submission.json
                raw = helper.download_json(settings.storage_container, bronze_path)
            else:
                # treat as full URL (SAS/public) and try download
                helper = BlobHelper(settings)
                raw = helper.download_json_from_url(bronze_path)
    else:
        blob_url = args.blob_url
        if args.upload_path:
            if settings.storage_mode != "adls":
                raise SystemExit("--upload-path requires STORAGE_MODE=adls and Storage settings configured")
            helper = BlobHelper(settings)
            local_path = Path(args.upload_path).resolve()
            # upload under incoming/<submission-id>/<filename>
            blob_path = f"incoming/{args.submission_id}/{local_path.name}"
            uploaded_url = helper.upload_file(settings.storage_container, blob_path, local_path)
            # try to generate SAS (only possible with connection string / key auth)
            sas_url = helper.generate_sas_url(settings.storage_container, blob_path, settings.blob_sas_ttl_minutes) or uploaded_url
            blob_url = sas_url
            # Prefer calling DI with local bytes to avoid SAS/network issues
            raw = analyze_id_document_from_file(settings, local_path)
        else:
            raw = analyze_id_document(settings, blob_url=blob_url, use_sample=args.use_sample)
        
    norm = normalize_id(raw, submission_id=args.submission_id)

    # Persist to local bronze/silver
    bronze_out = storage_route_write_json(settings, "bronze", args.submission_id, raw)
    silver_out = storage_route_write_json(settings, "silver", args.submission_id, norm.model_dump())

    print(f"Bronze written: {bronze_out}")
    print(f"Silver written: {silver_out}")


if __name__ == "__main__":
    main()
