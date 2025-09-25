from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ingestion.config import Settings


@dataclass
class RegistryEntry:
    model_type: str  # "baseline" | "ensemble" | other
    out_subdir: str
    out_name: str
    model_url: str
    features_url: str
    metrics_url: str
    extras: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def update_registry(settings: Settings, entry: RegistryEntry, promote: bool = False) -> Path:
    """Append a run entry to the registry and optionally promote to champion.

    - Registry lives at data_root/models/registry.json
    - Champion lives at data_root/models/champion/{model.pkl,features.json}
    - In ADLS mode, champion symlinks are skipped; champion pointers are stored in JSON only.
    """
    root = settings.data_root
    reg_path = root / "models" / "registry.json"
    reg = _read_json(reg_path)
    runs = reg.get("runs") or []

    run_obj = {
        "time": _now_iso(),
        "model_type": entry.model_type,
        "out_subdir": entry.out_subdir,
        "out_name": entry.out_name,
        "artifacts": {
            "model": entry.model_url,
            "features": entry.features_url,
            "metrics": entry.metrics_url,
        },
    }
    if entry.extras:
        run_obj["extras"] = entry.extras
    if entry.run_id:
        run_obj["run_id"] = entry.run_id
    runs.append(run_obj)
    reg["runs"] = runs

    # Champion promotion
    if promote:
        champion: Dict[str, Any] = {
            "time": _now_iso(),
            "model_type": entry.model_type,
            "out_subdir": entry.out_subdir,
            "out_name": entry.out_name,
            "source": {
                "model": entry.model_url,
                "features": entry.features_url,
                "metrics": entry.metrics_url,
            },
        }
        reg["champion"] = champion

        # Local mode: create/update symlinks to champion artifacts
        if settings.storage_mode != "adls":
            champ_dir = root / "models" / "champion"
            champ_dir.mkdir(parents=True, exist_ok=True)

            def _to_local(p: str) -> Optional[Path]:
                if not p:
                    return None
                if p.startswith("file://"):
                    p = p.replace("file://", "")
                q = Path(p)
                return q if q.exists() else None

            model_p = _to_local(entry.model_url)
            feats_p = _to_local(entry.features_url)
            if model_p is not None:
                link = champ_dir / "model.pkl"
                try:
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(model_p)
                except Exception:
                    # Fallback: copy
                    try:
                        link.write_bytes(model_p.read_bytes())
                    except Exception:
                        pass
            if feats_p is not None:
                link = champ_dir / "features.json"
                try:
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(feats_p)
                except Exception:
                    try:
                        link.write_text(feats_p.read_text(encoding="utf-8"), encoding="utf-8")
                    except Exception:
                        pass

    _write_json(reg_path, reg)
    return reg_path

