from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict

# Optional Azure Functions imports; keep code import-tolerant for local tests
try:  # pragma: no cover
    import azure.functions as func
except Exception:  # pragma: no cover
    func = None  # type: ignore

from ...ingestion.config import Settings
from ...ingestion.azure_blob import BlobHelper
from ...ingestion.normalize import normalize_id
from ...ingestion.storage import storage_route_write_json


def _process_event(data: Dict[str, Any]) -> Dict[str, Any]:
    settings = Settings()
    settings.ensure_dirs()
    helper = BlobHelper(settings)

    submission_id = data.get("submissionId") or data.get("submission_id")
    bronze_url = data.get("bronzeUrl") or data.get("bronze_path") or data.get("bronzePath")
    if not submission_id or not bronze_url:
        raise ValueError("Event missing required fields: submissionId and bronzeUrl/bronzePath")

    # Download bronze JSON
    if bronze_url.startswith("http"):
        raw = helper.download_json_from_url(bronze_url)
    else:
        raw = helper.download_json(settings.storage_container, bronze_url)

    # Normalize to silver
    norm = normalize_id(raw, submission_id=submission_id)

    # Persist bronze/silver to configured storage
    bronze_out = storage_route_write_json(settings, "bronze", submission_id, raw)
    silver_out = storage_route_write_json(settings, "silver", submission_id, norm.model_dump())

    # TODO: features and scoring go here
    logging.info("bronze_out=%s silver_out=%s", bronze_out, silver_out)
    return {"bronze": bronze_out, "silver": silver_out}


def main(event: Any) -> Any:  # pragma: no cover - Azure Functions entrypoint
    if func and isinstance(event, func.EventGridEvent):
        data = event.get_json()
        try:
            result = _process_event(data)
            logging.info("Processed submission %s", data.get("submissionId"))
            return json.dumps({"status": "ok", **result})
        except Exception as e:
            logging.exception("Failed to process event: %s", e)
            raise
    else:
        # Allow local invocation for testing
        if isinstance(event, dict):
            return _process_event(event)
        else:
            raise RuntimeError("Unsupported invocation. Expect EventGridEvent or dict.")

