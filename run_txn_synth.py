from __future__ import annotations

from pathlib import Path
import os
import argparse

from synthetic.txn_generate import TxnSynthConfig, run_and_save
from ingestion.config import Settings
from ingestion.storage import storage_route_write_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic e-payment transactions and features")
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--avg-per-user", type=int, default=60)
    parser.add_argument("--fraud", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-features", default="./ML_Fraud/data/gold/txn_synth_features.csv")
    parser.add_argument("--out-records", default="./ML_Fraud/data/silver/txn_synth_records.json")
    parser.add_argument("--to-adls", action="store_true")
    parser.add_argument("--adls-features-path", default=None)
    parser.add_argument("--adls-records-path", default=None)
    # Generator selection and config
    parser.add_argument("--generator", choices=["builtin", "simple"], default="builtin",
                        help="Which synthetic generator to use: builtin (current patterns) or simple (configurable rule-based)")
    parser.add_argument("--config", default=None,
                        help="Optional YAML config path for generator parameters (used by --generator simple)")
    args = parser.parse_args()

    # Allow --out-records to be a directory; default file name inside it
    out_records_path = None
    if args.out_records:
        p = Path(args.out_records).resolve()
        # If path exists and is a directory, or the argument endswith a path separator, treat as directory
        if p.is_dir() or str(args.out_records).endswith((os.sep, "/")):
            p = p / "txn_synth_records.json"
        out_records_path = p

    cfg = TxnSynthConfig(
        users=args.users,
        days=args.days,
        avg_txn_per_user=args.avg_per_user,
        fraud_ratio=args.fraud,
        seed=args.seed,
        out_features=Path(args.out_features).resolve(),
        out_records=out_records_path,
        generator=args.generator,
        config_path=Path(args.config).resolve() if args.config else None,
    )
    feat_df, rec_df, outp = run_and_save(cfg)
    print(f"Wrote txn synthetic features (local): {outp} ({len(feat_df)} rows)")

    settings = Settings()
    if args.to_adls and settings.storage_mode == "adls":
        # features
        if args.adls_features_path:
            f_parts = Path(args.adls_features_path)
            f_subdir, f_name = str(f_parts.parent).strip("./"), f_parts.name
        else:
            f_subdir, f_name = "gold", Path(outp).name
        f_url = storage_route_write_bytes(
            settings,
            subdir=f_subdir,
            name=f_name,
            data=Path(outp).read_bytes(),
            content_type="text/csv; charset=utf-8",
        )
        print(f"Uploaded txn features to ADLS: {f_url}")

        # records
        if args.out_records:
            if args.adls_records_path:
                r_parts = Path(args.adls_records_path)
                r_subdir, r_name = str(r_parts.parent).strip("./"), r_parts.name
            else:
                r_subdir, r_name = "silver", Path(args.out_records).name
            r_url = storage_route_write_bytes(
                settings,
                subdir=r_subdir,
                name=r_name,
                data=Path(args.out_records).read_bytes(),
                content_type="application/json; charset=utf-8",
            )
            print(f"Uploaded txn records to ADLS: {r_url}")


if __name__ == "__main__":
    main()
