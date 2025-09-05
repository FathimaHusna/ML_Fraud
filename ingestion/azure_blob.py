from __future__ import annotations
import json
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .config import Settings

try:
    from azure.storage.blob import (
        BlobServiceClient,
        BlobClient,
        generate_blob_sas,
        BlobSasPermissions,
    )
    from azure.identity import DefaultAzureCredential
except Exception:  # pragma: no cover
    BlobServiceClient = None
    BlobClient = None
    generate_blob_sas = None
    BlobSasPermissions = None
    DefaultAzureCredential = None


class BlobHelper:
    def __init__(self, settings: Settings):
        self.settings = settings
        if BlobServiceClient is None:
            raise RuntimeError("azure-storage-blob not installed")

        if settings.storage_connection_string:
            self.svc = BlobServiceClient.from_connection_string(settings.storage_connection_string)
            self.uses_key_auth = True
        else:
            if not settings.storage_account_url:
                raise ValueError("STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING must be set for ADLS mode")
            cred = DefaultAzureCredential() if DefaultAzureCredential else None
            self.svc = BlobServiceClient(account_url=settings.storage_account_url, credential=cred)
            # SAS generation with account key is not available via MI; we will rely on generate_blob_sas only when using conn string
            self.uses_key_auth = False

    def upload_file(self, container: str, blob_path: str, local_path: Path) -> str:
        content_type, _ = mimetypes.guess_type(local_path.name)
        content_type = content_type or "application/octet-stream"
        bc: BlobClient = self.svc.get_blob_client(container=container, blob=blob_path)
        with local_path.open("rb") as f:
            bc.upload_blob(f, overwrite=True, content_type=content_type)
        return bc.url

    def write_json(self, container: str, blob_path: str, obj: Any) -> str:
        bc: BlobClient = self.svc.get_blob_client(container=container, blob=blob_path)
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        bc.upload_blob(data, overwrite=True, content_type="application/json; charset=utf-8")
        return bc.url

    def generate_sas_url(self, container: str, blob_path: str, ttl_minutes: int) -> Optional[str]:
        if not generate_blob_sas or not BlobSasPermissions:
            return None
        if not self.uses_key_auth:
            # Without account key, we cannot generate SAS here; prefer public access or pre-created SAS.
            return None
        acct_url = self.svc.url  # e.g., https://<acct>.blob.core.windows.net
        # Extract account name from URL
        account_name = acct_url.split("//")[-1].split(".")[0]
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_path,
            account_key=self.svc.credential.account_key,  # type: ignore[attr-defined]
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(minutes=ttl_minutes),
        )
        return f"{acct_url}/{container}/{blob_path}?{sas}"

    def download_json(self, container: str, blob_path: str) -> Any:
        bc: BlobClient = self.svc.get_blob_client(container=container, blob=blob_path)
        data = bc.download_blob().readall()
        return json.loads(data)

    def download_json_from_url(self, url: str) -> Any:
        # Attempt via SAS/anonymous first; then via AAD if identity available
        cred = None
        if DefaultAzureCredential is not None:
            try:
                cred = DefaultAzureCredential()
            except Exception:
                cred = None
        bc = BlobClient.from_blob_url(url, credential=cred) if cred else BlobClient.from_blob_url(url)
        data = bc.download_blob().readall()
        return json.loads(data)
