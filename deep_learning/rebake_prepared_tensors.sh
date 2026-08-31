#!/bin/bash
# Runs ON the GPU box. Rebuilds the prepared tensors against the PROMOTED feature store
# (task #18), so training sees the 916 new 2026 games and no longer sees season 2020.
#
# WHY IT IS SPLIT AT THE WEATHER BOUNDARY (2026-08-31):
# the dataset cache and precollate read the FEATURE STORE, not the as-of weather artifact.
# Only `append_weather_asof_to_prepared` needs the weather, and that script rewrites
# weather_asof.npy and wx_decision_hour.npy from scratch (np.save, full overwrite) while
# merely READING the non-weather arrays it keys off (game_pks / sample_to_game /
# prefix_length). So stages 1-3 here can run CONCURRENTLY with the 2026 backfill chain, and
# whatever weather state precollate happens to bake is guaranteed to be overwritten later.
# That overlap is why this exists as a separate script instead of one linear chain.
#
# The weather artifacts are deliberately EXCLUDED from the sync in stage 1. The chain is
# still writing weather_asof/ and weather_asof_norm.json upstream; pulling them mid-write
# would either catch a partial 2026 season or, worse, install the PRE-2020-removal norm
# sidecar, whose moments were fit over rows training will never see again. They get pulled
# in stage 4, after the chain lands.
#
# Nothing existing is destroyed: output goes to *_new and the swap is a separate, manual
# step once the append has verified coverage. /mnt/fast is an instance store — stopping this
# box wipes all of it, so the swap is also the point at which anything worth keeping should
# already be in S3.
#
# Usage:  nohup bash rebake_prepared_tensors.sh >/dev/null 2>&1 &
# Log:    ~/rebake.log
set -u

LOG=/home/ec2-user/rebake.log
REPO=/home/ec2-user/mlb
PY=/home/ec2-user/miniconda3/envs/pred/bin/python
BUCKET=mlb-265753586044-us-east-1-an
FS=/mnt/fast/feature_store
CACHE=/mnt/fast/dataset_cache_new
PREP=/mnt/fast/prepared_tensors_new

exec >>"$LOG" 2>&1
echo "=== rebake start $(date -u +%FT%TZ) ==="

# --- Preflight -------------------------------------------------------------
# 8.3G cache + 27G prepared observed on the previous build, doubled for headroom since both
# the old and new copies coexist until the swap.
avail=$(df -BG --output=avail /mnt/fast | tail -1 | tr -dc '0-9')
if [ "${avail:-0}" -lt 80 ]; then
  echo "ABORT: only ${avail}G free on /mnt/fast, need ~80G to hold old+new side by side"; exit 3
fi
echo "preflight ok: ${avail}G free"

# --- 1/3 sync the promoted store, weather excluded -------------------------
# s5cmd over `aws s3 sync`: the store is ~3.5G across a few large parquets and s5cmd
# parallelises the multipart reads.
echo "--- 1/3 sync promoted feature store (weather excluded) $(date -u +%FT%TZ) ---"
if ! s5cmd sync --exclude 'weather_asof*' --exclude 'weather_asof/*' \
      "s3://${BUCKET}/deep_learning/feature_store/*" "${FS}/"; then
  echo "SYNC FAILED"; exit 4
fi
# Prove the sync actually moved the population, rather than silently no-opping the way the
# weather backfill did on a stale local snapshot. 163,189 = old 163,552 - 1,279 (all of
# season 2020) + 916 new 2026 games; that arithmetic is the signature of the promotion.
"$PY" - <<PYEOF || { echo "POPULATION CHECK FAILED"; exit 4; }
import sys, pyarrow.parquet as pq, pandas as pd
m = pq.ParquetFile("${FS}/game_meta.parquet")
n = m.metadata.num_rows
df = pq.read_table("${FS}/game_meta.parquet", columns=["game_date"]).to_pandas()
d = pd.to_datetime(df["game_date"])
y2020 = int((d.dt.year == 2020).sum())
latest = d.max()
print(f"game_meta rows={n:,} season2020={y2020} latest={latest:%Y-%m-%d}")
# The absence of 2020 is the real signature of the promotion and is checked exactly; the row
# count is only floored, because asserting 163,189 on the nose would abort the whole overlap
# if a later ingestion legitimately added games, and aborting costs more than it protects.
if y2020 != 0:
    sys.exit(f"season 2020 still present ({y2020} games) — this is NOT the promoted store")
if n < 163000:
    sys.exit(f"only {n:,} rows — expected ~163,189, so this is the pre-promotion snapshot")
PYEOF
echo "sync verified $(date -u +%FT%TZ)"

# --- 2/3 rebuild the dataset cache ----------------------------------------
# ~15 min on 37M rows (measured 14.7 min on 2026-08-29).
echo "--- 2/3 dataset cache $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.dataset_cache build \
        --feature-store "$FS" --output "$CACHE" ); then
  echo "CACHE BUILD FAILED"; exit 5
fi
echo "cache built $(date -u +%FT%TZ)"

# --- 3/3 precollate -------------------------------------------------------
# ~6 min (measured 6.1 min). The weather arrays it writes here are placeholders: stage 4's
# append overwrites both of them, which is exactly why this may run before the chain lands.
echo "--- 3/3 precollate $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.train_unified precollate \
        --dataset-cache "$CACHE" --output "$PREP" --num-workers 8 ); then
  echo "PRECOLLATE FAILED"; exit 6
fi
du -sh "$PREP"

echo "=== REBAKE STAGE 1-3 COMPLETE $(date -u +%FT%TZ) ==="
echo "NEXT (only after the 2026 chain prints CHAIN COMPLETE):"
echo "  s5cmd sync 's3://${BUCKET}/deep_learning/feature_store/weather_asof/*' ${FS}/weather_asof/"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/feature_store/weather_asof_norm.json ${FS}/"
echo "  cd ${REPO}/deep_learning && ${PY} -m mlb_dl.append_weather_asof_to_prepared \\"
echo "      --feature-store ${FS} --prepared-dir ${PREP}"
echo "Then swap: mv /mnt/fast/prepared_tensors /mnt/fast/prepared_tensors_old && \\"
echo "           mv ${PREP} /mnt/fast/prepared_tensors"
