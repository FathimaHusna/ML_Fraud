from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.config import Settings
from models.train_baseline import TrainConfig, train_and_save
from models.registry import RegistryEntry, update_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline fraud classifier on Gold features")
    parser.add_argument(
        "--gold-path",
        default="ML_Fraud/data/gold/txn_synth_features.csv",
        help="Path to Gold features CSV (default: synthetic features)",
    )
    parser.add_argument("--out-subdir", default="models", help="Subdirectory under data root for outputs")
    parser.add_argument("--out-name", default="baseline_xgb", help="Base name of model artifacts")
    parser.add_argument("--promote", action="store_true", help="Promote this run to champion and update registry")
    args = parser.parse_args()

    settings = Settings()
    cfg = TrainConfig(
        gold_path=args.gold_path,
        out_subdir=args.out_subdir,
        out_name=args.out_name,
    )
    model_url, feats_url, metrics_url = train_and_save(cfg, settings)
    print("Artifacts written:")
    print(f"  model:    {model_url}")
    print(f"  features: {feats_url}")
    print(f"  metrics:  {metrics_url}")

    # Update registry and optionally champion
    try:
        entry = RegistryEntry(
            model_type="baseline",
            out_subdir=args.out_subdir,
            out_name=args.out_name,
            model_url=model_url,
            features_url=feats_url,
            metrics_url=metrics_url,
        )
        reg_path = update_registry(settings, entry, promote=args.promote)
        print(f"Registry updated: {reg_path}")
        if args.promote:
            champ_dir = Settings().data_root / "models" / "champion"
            print(f"Champion pointers updated under: {champ_dir}")
    except Exception as e:
        print(f"Warning: failed to update registry/champion: {e}")


if __name__ == "__main__":
    main()
