#!/bin/bash
# Runs ON a shard box. Waits for that box's HRRR extraction to finish, verifies the
# seasons it owns, then builds their as-of weather tensors. Chaining keeps the
# extract->build handoff off the orchestrator's critical path.
#
# Usage:  nohup bash chain_build_weather_asof.sh 2023 2024 >/dev/null 2>&1 &
# Log:    ~/chain_build.log
#
# Two failure modes from the 2026-08-30 run are guarded here explicitly; both were
# silent, and both cost a full shard's worth of builds:
#
#   1. A bare "wait for the fetch process to disappear" loop is VACUOUS when the
#      chain is armed during a restart gap. Box E armed at 08:43:06Z while its
#      fetch was momentarily down, so the loop fell through in the same second,
#      ran the 2024 gate against an empty archive, and declared CHAIN COMPLETE
#      nine seconds in. The fix is observe-then-wait: refuse to proceed until the
#      extraction has actually been SEEN running at least once.
#   2. The builder was still being deployed when the chain reached it, so
#      `python -m mlb_dl.build_weather_asof` died on ModuleNotFoundError and the
#      season was skipped. Preflight now runs that import BEFORE the long wait, so
#      a deployment race fails loudly at arm time instead of after 8 idle hours.
set -u

SEASONS="$*"
LOG=/home/ec2-user/chain_build.log
REPO=/home/ec2-user/mlb
FETCH_PAT="[f]etch_nwp_asissued.py backfill"
# Interpreter differs by box class: shard boxes are bare AL2023 (python3.11),
# the GPU box only has python inside the `pred` conda env.
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")

exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ) seasons=$SEASONS py=$PY ==="

if [ -z "$SEASONS" ]; then
  echo "ABORT: no seasons given"; exit 2
fi

# --- Preflight: fail at arm time, not after the wait ------------------------
if ! ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.build_weather_asof --help ) >/dev/null 2>&1; then
  echo "ABORT: builder not runnable (mlb_dl not importable from $REPO/deep_learning)"; exit 3
fi
if [ ! -f "$REPO/data_curation/scripts/verify_weather_archives.py" ]; then
  echo "ABORT: verifier missing"; exit 3
fi
echo "preflight ok $(date -u +%FT%TZ)"

# --- Observe-then-wait -----------------------------------------------------
observed=0
for _ in $(seq 1 90); do          # up to 15 min for the fetch to appear
  if pgrep -f "$FETCH_PAT" >/dev/null; then observed=1; break; fi
  sleep 10
done
if [ "$observed" -eq 0 ]; then
  echo "ABORT: extraction never observed running; refusing to gate an archive that"
  echo "       may still be empty. Re-arm after confirming the fetch is up."
  exit 4
fi
echo "extraction observed $(date -u +%FT%TZ); waiting for it to finish"

# Require two consecutive absences: pgrep can miss a live process across the
# fork/exec window of a worker restart, and one false negative here would gate a
# half-written archive.
misses=0
while [ "$misses" -lt 2 ]; do
  if pgrep -f "$FETCH_PAT" >/dev/null; then misses=0; else misses=$((misses + 1)); fi
  sleep 60
done
echo "extraction exited $(date -u +%FT%TZ)"

# --- Gate, then build ------------------------------------------------------
for S in $SEASONS; do
  echo "--- completeness gate $S $(date -u +%FT%TZ) ---"
  if ! ( cd "$REPO" && "$PY" data_curation/scripts/verify_weather_archives.py \
           completeness --year "$S" --sample 400 ); then
    echo "GATE FAILED $S — refusing to build; archive needs a --force rerun"
    continue
  fi

  echo "--- build season $S $(date -u +%FT%TZ) ---"
  # flock so a duplicate chain (or a re-arm racing an old one) cannot have two
  # builders writing the same season's tensors concurrently.
  if ( cd "$REPO/deep_learning" && \
       flock -n "/tmp/wx_asof_build_$S.lock" \
         "$PY" -m mlb_dl.build_weather_asof build --season "$S" --workers 6 ); then
    touch "/home/ec2-user/.wx_asof_built_$S"
    echo "BUILD OK $S $(date -u +%FT%TZ)"
  else
    echo "BUILD FAILED $S (or lock held by another builder)"
  fi
done
echo "=== CHAIN COMPLETE $(date -u +%FT%TZ) ==="
