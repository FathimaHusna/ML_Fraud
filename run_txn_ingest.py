from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ingestion.config import Settings
from ingestion.storage import storage_route_write_json
from ingestion.txn_schema import TxnRaw, normalize_txn


def _load_csv(path: Path) -> List[dict]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _load_json_or_jsonl(path: Path) -> List[dict]:
    txt = path.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("Unsupported JSON structure")
    except json.JSONDecodeError:
        # Try JSON Lines
        rows: List[dict] = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def ingest_records(records: Iterable[dict], settings: Settings, out_prefix: str = "txn") -> List[str]:
    settings.ensure_dirs()
    written: List[str] = []
    for rec in records:
        bronze = TxnRaw.model_validate(rec)
        silver = normalize_txn(bronze)
        # Persist per-transaction bronze/silver under data/<layer>/txn/<YYYYMM>/
        txid = silver.transactionId
        # no strict date partitioning to keep it simple here
        bronze_path = storage_route_write_json(settings, f"bronze/{out_prefix}", txid, bronze.model_dump(by_alias=True))
        silver_path = storage_route_write_json(settings, f"silver/{out_prefix}", txid, silver.model_dump())
        written.append(silver_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest e-payment transactions to bronze/silver")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-csv", help="CSV with transaction records")
    src.add_argument("--input-json", help="JSON or JSONL with transactions (list or lines)")
    parser.add_argument("--prefix", default="txn", help="Subfolder prefix under bronze/silver")
    args = parser.parse_args()

    settings = Settings()
    if args.input_csv:
        records = _load_csv(Path(args.input_csv))
    else:
        records = _load_json_or_jsonl(Path(args.input_json))

    written = ingest_records(records, settings, out_prefix=args.prefix)
    print(f"Wrote {len(written)} silver records under prefix '{args.prefix}'")


if __name__ == "__main__":
    main()

