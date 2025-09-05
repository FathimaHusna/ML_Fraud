from __future__ import annotations
import os
from typing import Any, Dict

from .config import Settings
from .storage import read_json_local
from pathlib import Path

try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
except Exception:  # pragma: no cover - allow import without SDK for local testing
    DefaultAzureCredential = None
    DocumentIntelligenceClient = None
    AnalyzeDocumentRequest = None
    AzureKeyCredential = None


def _build_di_client(settings: Settings) -> DocumentIntelligenceClient:
    if not DocumentIntelligenceClient:
        raise RuntimeError("azure-ai-documentintelligence is not installed")
    if not settings.di_endpoint:
        raise ValueError("DI_ENDPOINT must be set")

    di_key = os.getenv("DI_KEY")
    if di_key and AzureKeyCredential is not None:
        return DocumentIntelligenceClient(settings.di_endpoint, AzureKeyCredential(di_key))
    if DefaultAzureCredential is None:
        raise RuntimeError("No credentials available. Set DI_KEY or install azure-identity for AAD.")
    return DocumentIntelligenceClient(settings.di_endpoint, DefaultAzureCredential())


def analyze_id_document(settings: Settings, blob_url: str | None = None, use_sample: bool = False) -> Dict[str, Any]:
    if use_sample or not (settings.di_endpoint and DocumentIntelligenceClient and blob_url):
        repo_root = Path(__file__).resolve().parents[1]
        sample_path = repo_root / "samples" / "idp_raw_sample.json"
        return read_json_local(sample_path)

    client = _build_di_client(settings)
    # SDK expects positional 'body' or keyword 'body', not 'analyze_request'
    poller = client.begin_analyze_document(
        model_id=settings.model_id,
        body=AnalyzeDocumentRequest(url_source=blob_url),
    )
    result = poller.result()
    # result may be a dict-like; normalize to dict for persistence
    return result.as_dict() if hasattr(result, "as_dict") else result


def analyze_id_document_from_file(settings: Settings, file_path: Path) -> Dict[str, Any]:
    """Analyze a local file by sending raw bytes to DI (no SAS/URL needed)."""
    client = _build_di_client(settings)
    data = file_path.read_bytes()
    poller = client.begin_analyze_document(
        model_id=settings.model_id,
        body=data,
        content_type="application/octet-stream",
    )
    result = poller.result()
    return result.as_dict() if hasattr(result, "as_dict") else result
