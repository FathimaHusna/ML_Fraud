from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .azure_blob import BlobHelper
from .config import Settings


def write_json_local(base: Path, subdir: str, name: str, obj: Any) -> Path:
    out_dir = base / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


def read_json_local(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ADLS stubs (extend later)
class ADLSClient:
    def __init__(self, account_url: str | None = None) -> None:
        self.account_url = account_url

    def write_json(self, container: str, path: str, obj: Any) -> None:
        raise NotImplementedError("Use BlobHelper via storage_route_write_json for ADLS mode")


def storage_route_write_json(settings: Settings, subdir: str, name: str, obj: Any) -> str:
    """Write JSON either to local (data_root/subdir/name.json) or to ADLS (container/subdir/name.json).

    Returns the output path or blob URL as a string.
    """
    if settings.storage_mode == "adls":
        helper = BlobHelper(settings)
        blob_path = f"{subdir}/{name}.json"
        return helper.write_json(settings.storage_container, blob_path, obj)
    else:
        path = write_json_local(settings.data_root, subdir, name, obj)
        return str(path)
