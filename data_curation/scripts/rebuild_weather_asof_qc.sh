#!/bin/bash
# Runs ON a shard box. Rebuilds the given weather_asof seasons so they pick up the ASOS
# QC rules added in 28a18ea (gust factor, precip corroboration). The HRRR archive is
# already complete, so unlike chain_build_weather_asof.sh this does NOT wait on a fetch --
# that script's observe-then-wait guard would abort here, correctly, because no extraction
# is running or ever will be for these seasons.
#
# Usage:  nohup bash rebuild_weather_asof_qc.sh 2017 2018 >/dev/null 2>&1 &
# Log:    ~/rebuild_qc.log
#
# PRECONDITION, and the whole reason this script refuses to start without checking: the
# box must be carrying the POST-28a18ea weather_asof.py. A rebuild on the old module
# silently reproduces the corrupt values it exists to remove, and the output is
# indistinguishable from a good rebuild without re-auditing every season -- so the
# deployment race is checked here, loudly, at arm time.
set -u

SEASONS="$*"
LOG=/home/ec2-user/rebuild_qc.log
REPO=/home/ec2-user/mlb
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")

exec >>"$LOG" 2>&1
echo "=== rebuild start $(date -u +%FT%TZ) seasons=$SEASONS py=$PY ==="

if [ -z "$SEASONS" ]; then
  echo "ABORT: no seasons given"; exit 2
fi

# --- Preflight -------------------------------------------------------------
for sym in GUST_FACTOR_MAX PRECIP_NO_CODE_MAX_IN STATION_ROLE_PRIMARY; do
  if ! grep -q "^$sym" "$REPO/deep_learning/mlb_dl/weather_asof.py"; then
    echo "ABORT: $sym absent from weather_asof.py — this box has the PRE-QC module."
    echo "       Rebuilding now would rewrite the artifact with the same corrupt values."
    exit 3
  fi
done
# The constant living in weather_asof.py does not prove the BUILDER tags the column, and
# select_asof_obs prefers the primary only when the column is present -- so a half-deployed
# tree rebuilds silently with the old recency-only selection. Check the writer, not just
# the reader.
if ! grep -q 'station_role"\] = role' "$REPO/deep_learning/mlb_dl/build_weather_asof.py"; then
  echo "ABORT: build_weather_asof.py does not tag station_role — half-deployed tree."
  echo "       The rebuild would reproduce the backup-preemption it exists to remove."
  exit 3
fi
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof --help ) >/dev/null 2>&1; then
  echo "ABORT: builder not runnable (mlb_dl not importable from $REPO/deep_learning)"; exit 3
fi
echo "preflight ok: new QC present, builder importable $(date -u +%FT%TZ)"

# --- Build -----------------------------------------------------------------
rc=0
for S in $SEASONS; do
  echo "--- rebuild season $S $(date -u +%FT%TZ) ---"
  # flock so a re-arm racing an earlier invocation cannot put two builders on one season.
  if ( cd "$REPO/deep_learning" && \
       flock -n "/tmp/wx_asof_build_$S.lock" \
         "$PY" -m mlb_dl.build_weather_asof build --season "$S" --workers 6 ); then
    echo "REBUILD OK $S $(date -u +%FT%TZ)"
  else
    echo "REBUILD FAILED $S (or lock held by another builder)"
    rc=1
  fi
done
echo "=== REBUILD COMPLETE $(date -u +%FT%TZ) rc=$rc ==="
exit $rc
