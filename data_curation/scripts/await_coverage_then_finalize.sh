#!/bin/bash
# Waits for the HRRR as-issued archive to become complete, then finalizes the as-of
# weather artifact. Runs ON a data box; needs AWS creds and the repo, nothing else.
#
# Usage:  nohup bash data_curation/scripts/await_coverage_then_finalize.sh 2016 2018 2022 2025 &
# Log:    ~/await_finalize.log
#
# WHY THIS EXISTS RATHER THAN POLLING FROM A LAPTOP:
# the six shard boxes each own a date range and finish at different times. Watching them
# by counting S3 objects per season and comparing against a guessed target is what this
# replaces, and that method actively misled: a season whose worker has reached the
# postseason tail (2-4 venues per date) advances its object count slowly while being
# nearly done, which reads as a stall. The authority on "is the archive complete" is
# verify_weather_archives.py coverage, because it classifies every absent date as
# recoverable (its tasks exist upstream, so a rerun would get it) or provably
# unobtainable (genuine pre-domain era gap, or an international game with no venue inside
# the HRRR CONUS grid). Only the recoverable count can ever reach zero, so only the gate
# can say when to stop waiting.
#
# The wait condition is deliberately the ARCHIVE, not the shard processes. A dead worker
# and a finished one look identical from process state, and this box cannot see the other
# boxes anyway (no SSH key). Archive state is the shared, observable truth.
set -uo pipefail

LOG=/home/ec2-user/await_finalize.log
REPO=/home/ec2-user/mlb
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")
INTERVAL=${INTERVAL:-300}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-360}

SEASONS=("$@")
[ ${#SEASONS[@]} -gt 0 ] || { echo "usage: $0 <season> [season...]"; exit 2; }

exec >>"$LOG" 2>&1
echo "=== await+finalize start $(date -u +%FT%TZ) waiting on: ${SEASONS[*]} ==="
cd "$REPO" || { echo "ABORT: no repo at $REPO"; exit 1; }

deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
pending=("${SEASONS[@]}")

while [ ${#pending[@]} -gt 0 ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ABORT: ${MAX_WAIT_MIN}min deadline reached with ${pending[*]} still incomplete."
    echo "       Not finalizing: fitting the standardizer on a partial archive would bias"
    echo "       every z-score in training AND live serving."
    exit 1
  fi

  still=()
  for y in "${pending[@]}"; do
    out=$("$PY" data_curation/scripts/verify_weather_archives.py coverage \
            --year "$y" --workers 16 2>&1)
    if echo "$out" | grep -q "ALL CHECKS PASSED"; then
      echo "$(date -u +%H:%M:%SZ) COMPLETE $y"
    else
      # Report the recoverable count, which is the number that has to reach zero. The
      # "unobtainable" lines are permanent and must not be mistaken for progress stalling.
      n=$(echo "$out" | grep -oE "[0-9]+/[0-9]+ population dates have NO archive" | head -1)
      echo "$(date -u +%H:%M:%SZ) waiting $y: ${n:-still short}"
      still+=("$y")
    fi
  done
  pending=("${still[@]+"${still[@]}"}")
  [ ${#pending[@]} -eq 0 ] && break
  sleep "$INTERVAL"
done

echo "=== archive complete for all requested seasons $(date -u +%FT%TZ) ==="

# Hand off to the finalizer, which owns season builds, the standardizer, and the
# artifact/loader verifiers. It is a separate script because it is also the thing to
# rerun by hand after a repair, independently of any waiting.
bash data_curation/scripts/finalize_weather_asof.sh
rc=$?
echo "=== finalize exit=$rc $(date -u +%FT%TZ) ==="
exit "$rc"
