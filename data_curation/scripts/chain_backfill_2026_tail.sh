#!/bin/bash
# Runs ON a shard box. Closes the 2026 tail that the feature-store refresh exposed, then
# rebuilds every artifact that depends on it — in dependency order, with a gate between each.
#
# WHY THIS EXISTS (2026-08-30):
# the weather backfill is driven by the game population, so when game_meta was truncated at
# 2026-06-20 the hrrr_asissued archive stopped at exactly 2026-06-20 too. Promoting the
# rebuilt store added 916 games (890 of them population games over 68 dates, 2026-06-21 ..
# 2026-08-30) that have NO forecast archive behind them. Building weather_asof against that
# state would emit rows whose forecast channel is entirely mask=0 — right row count, right
# schema, no signal, and nothing downstream would raise. So: fetch first, gate, then build.
#
# ASOS obs is NOT backfilled here: it is stored as per-station year=YYYY.parquet and was
# refetched 2026-08-29, already covering through 2026-08-30 (verified on ATL/BOS/LGA). Only
# the HRRR as-issued forecast channel is short.
#
# norm-stats is rebuilt LAST and unconditionally, for a reason unrelated to 2026: dropping
# season 2020 removed those game_pks from train, and build_norm_stats fits on
# `game_date < TRAIN_END_DATE`, so the existing sidecar's moments were accumulated over 2020
# weather rows that training will never see again. Stale, not leaky — but it shifts every
# z-score in training AND live serving, so it must not be left behind.
#
# Usage:  nohup bash chain_backfill_2026_tail.sh >/dev/null 2>&1 &
# Log:    ~/chain_2026_tail.log
set -u

LOG=/home/ec2-user/chain_2026_tail.log
REPO=/home/ec2-user/mlb
SEASON=2026
START=2026-06-21
END=2026-08-30
WORKERS=6
# Interpreter differs by box class: shard boxes are bare AL2023 (python3.11), the GPU box
# only has python inside the `pred` conda env.
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")

exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ) season=$SEASON window=$START..$END py=$PY ==="

# --- Preflight: fail now, not after an hour of fetching ---------------------
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof --help ) >/dev/null 2>&1; then
  echo "ABORT: builder not runnable (mlb_dl not importable from $REPO/deep_learning)"; exit 3
fi
for f in data_curation/scripts/fetch_nwp_asissued.py \
         data_curation/scripts/verify_asof_population_coverage.py; do
  [ -f "$REPO/$f" ] || { echo "ABORT: missing $f"; exit 3; }
done
echo "preflight ok $(date -u +%FT%TZ)"

# --- 1/5 fetch the missing forecast dates ----------------------------------
# flock so a re-arm cannot put two fetchers on the same date range. HERBIE_SAVE_DIR is
# already PID-scoped, but two processes would still duplicate S3 writes and burn API quota.
echo "--- 1/5 hrrr_asissued backfill $START..$END $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO" && flock -n /tmp/nwp_asissued_2026_tail.lock \
        "$PY" data_curation/scripts/fetch_nwp_asissued.py backfill \
          --start "$START" --end "$END" --workers "$WORKERS" ); then
  echo "FETCH FAILED or lock held — stopping before any build reads a short archive"; exit 4
fi
echo "fetch exited $(date -u +%FT%TZ)"

# --- 2/5 coverage gate ------------------------------------------------------
# The authoritative gate. Distinguishes "no CONUS file" (actionable) from "outside the HRRR
# grid" (irreducible, e.g. the Tokyo series). Anything actionable stops the chain: a partial
# forecast channel is exactly the silent corruption this chain exists to avoid.
echo "--- 2/5 coverage gate $SEASON $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO" && "$PY" data_curation/scripts/verify_asof_population_coverage.py \
         --year "$SEASON" ); then
  echo "GATE FAILED $SEASON — refusing to build weather_asof on a short archive."
  echo "  Re-run the fetch for the listed dates (add --force if keys exist but are bad)."
  exit 5
fi

# --- 3/5 rebuild the season's as-of tensor ---------------------------------
echo "--- 3/5 build weather_asof season=$SEASON $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO/deep_learning" && flock -n "/tmp/wx_asof_build_$SEASON.lock" \
        "$PY" -m mlb_dl.build_weather_asof build --season "$SEASON" --workers "$WORKERS" ); then
  echo "BUILD FAILED $SEASON (or lock held by another builder)"; exit 6
fi

# --- 4/5 per-pitch decision-hour offsets ----------------------------------
# Separate artifact keyed on pitch timestamps, so it is short for the same 890 games.
echo "--- 4/5 pitch-offsets season=$SEASON $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO/deep_learning" && \
        "$PY" -m mlb_dl.build_weather_asof pitch-offsets --season "$SEASON" ); then
  echo "PITCH-OFFSETS FAILED $SEASON"; exit 7
fi

# --- 5/5 refit the shared standardizer ------------------------------------
# Must come after the season build: it accumulates over every season parquet present, and
# refuses to run if a train season is absent.
echo "--- 5/5 norm-stats $(date -u +%FT%TZ) ---"
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof norm-stats ); then
  echo "NORM-STATS FAILED"; exit 8
fi

echo "=== CHAIN COMPLETE $(date -u +%FT%TZ) ==="
