from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class FeatureStoreConfig:
    root: Path = Path("./ML_Fraud/data/feature_store").resolve()


class FeatureStore:
    def __init__(self, cfg: Optional[FeatureStoreConfig] = None) -> None:
        self.cfg = cfg or FeatureStoreConfig()
        (self.cfg.root / "entities").mkdir(parents=True, exist_ok=True)
        (self.cfg.root / "transactions").mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cfg.root / "registry.json"
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"version": 1, "artifacts": {}}, indent=2), encoding="utf-8")

    def _update_registry(self, kind: str, name: str, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
        art = reg.setdefault("artifacts", {})
        key = f"{kind}:{name}"
        entry = art.setdefault(key, {"versions": []})
        entry["latest_path"] = str(path)
        meta = {"path": str(path), "timestamp": datetime.utcnow().isoformat() + "Z"}
        if extra:
            meta.update(extra)
        entry["versions"].append(meta)
        self.registry_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_entity(self, name: str, df: pd.DataFrame, version_tag: Optional[str] = None) -> Path:
        out_dir = self.cfg.root / "entities" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = version_tag or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_path = out_dir / f"{name}_{tag}.csv"
        df.to_csv(out_path, index=False)
        self._update_registry("entity", name, out_path, {"rows": len(df)})
        # Also write/update a latest.csv for ease of joining
        latest = out_dir / "latest.csv"
        df.to_csv(latest, index=False)
        return out_path

    def write_transaction_features(self, name: str, df: pd.DataFrame, version_tag: Optional[str] = None) -> Path:
        out_dir = self.cfg.root / "transactions" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = version_tag or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_path = out_dir / f"{name}_{tag}.csv"
        df.to_csv(out_path, index=False)
        self._update_registry("txn_features", name, out_path, {"rows": len(df)})
        latest = out_dir / "latest.csv"
        df.to_csv(latest, index=False)
        return out_path

