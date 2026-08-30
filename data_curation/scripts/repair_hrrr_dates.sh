#!/bin/bash
# Force-refetch specific HRRR as-issued dates whose archive object is incomplete.
#
# Why this exists as a script rather than an ad-hoc command:
#   verify_weather_archives.py distinguishes two kinds of low task-fill. A genuine
#   upstream archive gap (the GRIB was never published) is unfixable and is handled
#   by lead-fallback planning. A date where the probed missing tasks DO exist
#   upstream means OUR extraction dropped them, and the only fix is --force, because
#   run_backfill() skips any date whose S3 key already exists -- a partially written
#   object is indistinguishable from a complete one by head_object. So a normal shard
#   rerun will never repair these; it will skip them forever.
#
# Usage:  ./repair_hrrr_dates.sh 2025-08-25 2025-08-31 ...
#
# Safe to run concurrently with a shard backfill: each date is written with a single
# S3 put_object, which is atomic, and both writers produce identical content, so the
# worst case is a redundant overwrite. It also matches the shard chain's
# "fetch_nwp_asissued.py backfill" pgrep pattern, so a chained build will correctly
# wait for the repair before gating completeness.

set -u
REPO="${REPO:-$HOME/mlb}"
WORKERS="${WORKERS:-3}"
LOG="${LOG:-$HOME/hrrr_repair.log}"
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")

if [ "$#" -eq 0 ]; then
  echo "usage: $0 YYYY-MM-DD [YYYY-MM-DD ...]" >&2
  exit 2
fi

echo "=== repair start $(date -u +%FT%TZ) dates=$* workers=$WORKERS ===" >>"$LOG"
rc_all=0
for d in "$@"; do
  case "$d" in
    [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) : ;;
    *) echo "SKIP malformed date: $d" >>"$LOG"; rc_all=1; continue ;;
  esac
  echo "--- forcing $d ---" >>"$LOG"
  # flock per date so two overlapping repair invocations cannot both burn the
  # (rate-limited, 503-prone) HRRR fetch budget on the same date.
  if ( cd "$REPO" && flock -n "/tmp/hrrr_repair_$d.lock" \
        "$PY" data_curation/scripts/fetch_nwp_asissued.py backfill \
          --start "$d" --end "$d" --workers "$WORKERS" --force ) >>"$LOG" 2>&1; then
    echo "REPAIR OK $d" >>"$LOG"
  else
    echo "REPAIR FAILED $d (rc=$?)" >>"$LOG"
    rc_all=1
  fi
done
echo "=== repair done $(date -u +%FT%TZ) rc=$rc_all ===" >>"$LOG"
exit "$rc_all"
