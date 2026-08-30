"""Full rebuild of the 15 core DL feature-store tables, to a staging dir.

WHY a FULL rebuild and not `seasons=[2026]`:
build_feature_store writes one flat `<table>.parquet` per output -- none of the 15 core
tables is season-partitioned (unlike the weather_asof/ and wx_hour_offset/ artifacts, which
are `season=Y.parquet`). A season-scoped run therefore produces files containing ONLY that
season; there is no append path and no partition to swap. Refreshing 2026 in place would
mean read-modify-write on 15 tables including a 2.4 GB pitch_sequences, keyed on a `season`
column that several tables (player_bios, venue_dimensions) do not even have. That hand-merge
is exactly the silent-corruption risk the pipeline's own streaming writer avoids, so we pay
the wall-clock instead and rebuild every season from the curated Parquet lake.

WHY a staging dir and not the live prefix:
the existing store is the input to the running A/B and to every prepared tensor. It is
known-good. This writes to STAGE_DIR and uploads to a *_staging S3 prefix; promoting it is a
separate, reversible step taken only after the verification below passes. Nothing here can
damage the current store.

Cheap: this only needs the raw lake (read) and ~3.5 GB of local disk (write). RAM stays flat
because build_feature_store streams per-season into open ParquetWriters.

Usage (shard A, python3.11, from ~/mlb):
    nohup python3.11 deep_learning/rebuild_feature_store_2026.py > ~/fs_rebuild.log 2>&1 &
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deep_learning"))

BUCKET = "mlb-265753586044-us-east-1-an"
SOURCE_URI = f"s3://{BUCKET}/data"
STAGE_DIR = Path.home() / "fs_stage"

log = logging.getLogger("rebuild")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(Path.home() / "fs_rebuild_detail.log")],
    )
    from mlb_dl.feature_store import build_feature_store

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info(f"REBUILD START source={SOURCE_URI} stage={STAGE_DIR}")

    # live_state_dir=None: s3://.../data/live_state/ is empty (0 objects), which is why the
    # current live_snapshots.parquet is 636 bytes. Passing the default path would have the
    # builder hunt for a directory that does not exist; None reproduces the shipped artifact.
    outputs = build_feature_store(
        source_uri=SOURCE_URI,
        output_dir=str(STAGE_DIR),
        seasons=None,
        live_state_dir=None,
    )

    log.info(f"REBUILD DONE in {(time.time()-t0)/3600:.2f} h")
    for name, path in sorted(outputs.items()):
        p = Path(path)
        log.info(f"  {name:<28} {p.stat().st_size/1e6 if p.exists() else -1:>10.1f} MB")

    # Verify the whole point of the rebuild BEFORE anything is promoted: game_meta must now
    # extend past 2026-06-20, the date the shipped store stops at. If it does not, the raw
    # lake backfill did not land where the builder reads and promoting would be a no-op.
    import pandas as pd
    gm = pd.read_parquet(STAGE_DIR / "game_meta.parquet",
                         columns=["game_pk", "game_date", "season"])
    gm["game_date"] = pd.to_datetime(gm["game_date"])
    latest = gm["game_date"].max()
    n2026 = int((gm["season"] == 2026).sum())
    log.info(f"VERIFY game_meta: {len(gm):,} games, latest={latest.date()}, 2026={n2026:,}")
    log.info("VERIFY per-season 2024-2026:\n" +
             gm[gm["season"] >= 2024].groupby("season").size().to_string())
    if latest <= pd.Timestamp("2026-06-20"):
        log.error(f"VERIFY FAILED: latest={latest.date()} did not advance past 2026-06-20")
        sys.exit(1)
    if gm["game_pk"].duplicated().any():
        log.error(f"VERIFY FAILED: {int(gm['game_pk'].duplicated().sum())} duplicate game_pk")
        sys.exit(1)
    (Path.home() / "fs_stage_verify.json").write_text(json.dumps(
        {"games": len(gm), "latest": str(latest.date()), "games_2026": n2026}, indent=2))
    log.info("REBUILD VERIFIED — safe to upload to the staging prefix")


if __name__ == "__main__":
    main()
