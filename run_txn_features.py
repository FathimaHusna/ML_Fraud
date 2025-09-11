from __future__ import annotations

from pathlib import Path
import argparse

from features.txn_features import TxnFeatureJobConfig, run_txn_feature_job
from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute features for e-payment transactions")
    parser.add_argument("--silver-dir", default="./ML_Fraud/data/silver/txn", help="Directory with txn silver JSON files")
    parser.add_argument("--out", default="./ML_Fraud/data/gold/txn_features.csv", help="Output CSV path for features")
    parser.add_argument("--to-adls", action="store_true", help="Upload to ADLS if STORAGE_MODE=adls")
    parser.add_argument("--adls-path", default=None, help="Blob path under container, e.g., gold/txn_features.csv")
    args = parser.parse_args()

    cfg = TxnFeatureJobConfig(silver_dir=Path(args.silver_dir).resolve(), out_path=Path(args.out).resolve())
    df, out_path = run_txn_feature_job(cfg)
    print(f"Wrote txn features (local): {out_path} ({len(df)} rows)")

    settings = Settings()
    if args.to_adls and settings.storage_mode == "adls":
        if args.adls_path:
            parts = Path(args.adls_path)
            subdir = str(parts.parent).strip("./")
            name = parts.name
        else:
            subdir = "gold"
            name = Path(out_path).name
        url = storage_route_write_bytes(
            settings,
            subdir=subdir,
            name=name,
            data=Path(out_path).read_bytes(),
            content_type="text/csv; charset=utf-8",
        )
        print(f"Uploaded txn features to ADLS: {url}")


if __name__ == "__main__":
    main()

