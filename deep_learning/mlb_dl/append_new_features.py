"""Append weather, venue, daily stats, and weather_temporal parquets to an existing feature store.

Usage:
    conda run -n pred python -m deep_learning.mlb_dl.append_new_features \
        --feature-store /path/to/feature_store \
        --source-uri s3://mlb-265753586044-us-east-1-an/data \
        [--artifacts weather_temporal]

Reads the existing game_meta.parquet to determine which games need features,
then builds and writes artifacts alongside the existing ones.
Does NOT rebuild any existing parquets.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("mlb_dl.append_new_features")


def _setup_logging() -> None:
    log.setLevel(logging.DEBUG)

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "append_new_features.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(sh)


def _load_park_azimuths(uri: str) -> dict:
    """Load park_azimuths.json from S3 URI or local path. Returns {} on failure."""
    import json

    try:
        if uri.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse

            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            raw = json.loads(obj["Body"].read())
        else:
            with open(uri) as f:
                raw = json.load(f)
        # JSON keys are strings; convert to int→float
        return {int(k): float(v) for k, v in raw.items()}
    except Exception as exc:
        log.warning("Could not load park_azimuths from %s: %s — using defaults (0°)", uri, exc)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Append new feature parquets to existing feature store")
    parser.add_argument("--feature-store", required=True, help="Path to existing feature store directory")
    parser.add_argument("--source-uri", required=True, help="S3 URI or local path to raw data")
    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=["weather_features", "venue_dimensions", "daily_stats"],
        choices=["weather_features", "venue_dimensions", "daily_stats", "weather_temporal"],
        help="Which artifacts to build (default: weather_features, venue_dimensions, daily_stats)",
    )
    parser.add_argument(
        "--azimuths-uri",
        default="s3://mlb-265753586044-us-east-1-an/classical_learning/artifacts/features/park_azimuths.json",
        help="S3 URI or local path to park_azimuths.json (used by weather_temporal)",
    )
    args = parser.parse_args()

    _setup_logging()

    import pandas as pd

    from .data_sources import ParquetCatalog
    from .feature_store import (
        build_weather_frame,
        build_venue_dimensions_frame,
        build_daily_stats_frame,
        build_multihour_weather_frame,
    )

    fs_path = Path(args.feature_store)
    game_meta_path = fs_path / "game_meta.parquet"

    if not game_meta_path.exists():
        log.error("game_meta.parquet not found at %s", game_meta_path)
        sys.exit(1)

    log.info("Loading game_meta from %s", game_meta_path)
    game_meta = pd.read_parquet(game_meta_path)
    log.info("  %d games loaded", len(game_meta))

    catalog = ParquetCatalog(base_uri=args.source_uri)
    t_total = time.time()

    if "weather_features" in args.artifacts:
        out_path = fs_path / "weather_features.parquet"
        log.info("Building weather_features...")
        t = time.time()
        weather = build_weather_frame(catalog, game_meta)
        weather.to_parquet(out_path, index=False)
        log.info("  wrote %d rows to %s (%.1fs)", len(weather), out_path, time.time() - t)
        del weather

    if "venue_dimensions" in args.artifacts:
        out_path = fs_path / "venue_dimensions.parquet"
        log.info("Building venue_dimensions...")
        t = time.time()
        venue_dims = build_venue_dimensions_frame(catalog)
        venue_dims.to_parquet(out_path, index=False)
        log.info("  wrote %d rows to %s (%.1fs)", len(venue_dims), out_path, time.time() - t)
        del venue_dims

    if "daily_stats" in args.artifacts:
        out_path = fs_path / "daily_stats.parquet"
        log.info("Building daily_stats...")
        t = time.time()
        seasons = sorted(pd.to_datetime(game_meta["game_date"], errors="coerce").dt.year.dropna().unique().tolist())
        daily_stats = build_daily_stats_frame(catalog, game_meta, seasons=seasons)
        daily_stats.to_parquet(out_path, index=False)
        log.info("  wrote %d rows to %s (%.1fs)", len(daily_stats), out_path, time.time() - t)
        del daily_stats

    if "weather_temporal" in args.artifacts:
        out_path = fs_path / "weather_temporal.parquet"
        log.info("Building weather_temporal...")
        t = time.time()
        park_azimuths = _load_park_azimuths(args.azimuths_uri)
        log.info("  loaded %d park azimuths", len(park_azimuths))
        weather_temporal = build_multihour_weather_frame(catalog, game_meta, park_azimuths=park_azimuths)
        weather_temporal.to_parquet(out_path, index=False)
        log.info("  wrote %d rows to %s (%.1fs)", len(weather_temporal), out_path, time.time() - t)
        del weather_temporal

    log.info("Done in %.1fs", time.time() - t_total)


if __name__ == "__main__":
    main()
