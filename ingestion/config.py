import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.di_endpoint: str | None = os.getenv("DI_ENDPOINT")
        self.model_id: str = os.getenv("MODEL_ID", "prebuilt-idDocument")
        self.storage_mode: str = os.getenv("STORAGE_MODE", "local").lower()
        self.data_root: Path = Path(os.getenv("DATA_ROOT", "./ML_Fraud/data")).resolve()
        # Azure Storage / ADLS
        self.storage_account_url: str | None = os.getenv("STORAGE_ACCOUNT_URL")  # e.g., https://<acct>.blob.core.windows.net
        self.storage_connection_string: str | None = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.storage_container: str = os.getenv("STORAGE_CONTAINER", "mlfraud")
        # SAS TTL minutes for Document Intelligence access
        self.blob_sas_ttl_minutes: int = int(os.getenv("BLOB_SAS_TTL_MINUTES", "60"))

    def ensure_dirs(self) -> None:
        (self.data_root / "bronze").mkdir(parents=True, exist_ok=True)
        (self.data_root / "silver").mkdir(parents=True, exist_ok=True)
        (self.data_root / "gold").mkdir(parents=True, exist_ok=True)
