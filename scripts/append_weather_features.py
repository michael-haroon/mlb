"""
Append weather features to an existing game_features.parquet.

Standalone script: loads ERA5 from S3, calibrates azimuths, computes
climatology, joins at game hour, engineers 18 weather features, and writes
the augmented parquet back.

Usage (EC2):
  PYTHONPATH=. python3.11 scripts/append_weather_features.py \
    --parquet artifacts/features_staging/game_features.parquet \
    --source s3://mlb-265753586044-us-east-1-an/data \
    --artifacts-dir artifacts/features_staging
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Append weather features to game_features.parquet")
    parser.add_argument("--parquet", required=True, help="Path to existing game_features.parquet")
    parser.add_argument("--source", default="s3://mlb-265753586044-us-east-1-an/data",
                        help="S3 URI or local path to raw data")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Directory for cached azimuths/climatology (default: same as parquet)")
    args = parser.parse_args()

    import pandas as pd
    from classical_learning.engineering.weather import attach_weather_features

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        log.error(f"Parquet not found: {parquet_path}")
        sys.exit(1)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else parquet_path.parent

    log.info(f"Loading {parquet_path}...")
    t0 = time.time()
    games = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(games):,} games × {games.shape[1]} columns ({time.time()-t0:.1f}s)")

    # Drop any existing weather columns to avoid duplicates on re-run
    weather_cols = [c for c in games.columns if c in (
        "air_density", "air_density_ratio", "wind_toward_cf", "wind_crossfield",
        "wind_speed", "wind_gusts", "precip_6h", "precip_24h",
        "vpd", "humidity", "wet_bulb_f", "temperature_f",
        "air_density_anomaly", "temperature_f_anomaly", "humidity_anomaly",
        "wind_speed_anomaly", "surface_pressure_anomaly", "wind_toward_cf_open",
        "cloud_cover", "visibility",
    )]
    if weather_cols:
        log.info(f"Dropping {len(weather_cols)} existing weather columns for fresh computation")
        games = games.drop(columns=weather_cols)

    # Attach weather features
    t1 = time.time()
    games = attach_weather_features(games, args.source, artifacts_dir)
    log.info(f"Weather features attached ({time.time()-t1:.1f}s)")

    # Write back
    log.info(f"Writing {len(games):,} games × {games.shape[1]} columns to {parquet_path}...")
    games.to_parquet(parquet_path, index=False)
    log.info(f"Done. Total time: {time.time()-t0:.1f}s")

    # Summary of new columns
    new_cols = [c for c in games.columns if c in (
        "air_density", "air_density_ratio", "wind_toward_cf", "wind_crossfield",
        "wind_speed", "wind_gusts", "precip_6h", "precip_24h",
        "vpd", "humidity", "wet_bulb_f", "temperature_f",
        "air_density_anomaly", "temperature_f_anomaly", "humidity_anomaly",
        "wind_speed_anomaly", "surface_pressure_anomaly", "wind_toward_cf_open",
    )]
    log.info(f"Weather columns added: {len(new_cols)}")
    for col in new_cols:
        nna = games[col].notna().sum()
        log.info(f"  {col}: {nna}/{len(games)} non-null ({nna/len(games)*100:.1f}%)")


if __name__ == "__main__":
    main()
