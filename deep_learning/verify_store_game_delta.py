"""Locate the 363 games present in the live DL store but absent from the staging rebuild.

Consistent losses across every table point at a coherent set of games, not corruption. The
question is WHERE: an era (pre-Statcast), a season, or the 2026 tail. That determines whether
the rebuild is wrong or the live store contains games the raw lake no longer supports.
"""
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs

B = "mlb-265753586044-us-east-1-an"
s3 = pafs.S3FileSystem(region="us-east-1")
cols = ["game_pk", "game_date", "season"]


def load(prefix):
    f = s3.open_input_file(f"{B}/{prefix}/game_meta.parquet")
    df = pq.ParquetFile(f).read(columns=cols).to_pandas()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


live = load("deep_learning/feature_store")
stage = load("deep_learning/feature_store_staging")
print(f"live  {len(live):,} games  latest={live['game_date'].max()}")
print(f"stage {len(stage):,} games  latest={stage['game_date'].max()}")

lost = live[~live["game_pk"].isin(set(stage["game_pk"]))]
gained = stage[~stage["game_pk"].isin(set(live["game_pk"]))]
print(f"\nin live, NOT in stage: {len(lost):,}")
print(f"in stage, NOT in live: {len(gained):,}")

print("\n=== LOST by season ===")
print(lost.groupby("season").size().sort_values(ascending=False).head(15).to_string())
print("\n=== LOST date range ===")
print(f"  {lost['game_date'].min()} .. {lost['game_date'].max()}")
print("\n=== GAINED by season ===")
print(gained.groupby("season").size().sort_values(ascending=False).head(10).to_string())

print("\n=== per-season counts where they differ ===")
a = live.groupby("season").size().rename("live")
b = stage.groupby("season").size().rename("stage")
j = pd.concat([a, b], axis=1).fillna(0).astype(int)
j["delta"] = j["stage"] - j["live"]
print(j[j["delta"] != 0].to_string())

print("\n=== sample lost game_pks ===")
print(lost.sort_values("game_date")[["game_pk", "game_date", "season"]].head(12).to_string(index=False))
