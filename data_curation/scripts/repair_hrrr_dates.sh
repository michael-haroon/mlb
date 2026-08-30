#!/bin/bash
# Retry-until-written repair for individual HRRR as-issued dates.
#
# Usage:  bash data_curation/scripts/repair_hrrr_dates.sh 2018-08-26 [more dates...]
#         ATTEMPTS=8 WORKERS=6 bash ... 2018-08-26
# Log:    ~/repair_hrrr.log
#
# WHY A LOOP AND NOT A SINGLE RERUN:
# fetch_nwp_asissued withholds a date's write when ANY task fails transiently (the
# MIN_WRITE_FILL guard), because persisting a partially-fetched date would make the loss
# permanent and invisible to every later check. One transient Herbie 'href' error out of
# ~85 tasks is therefore enough to block a date indefinitely under one-shot reruns, even
# though the data IS upstream and the next attempt usually gets it. The completeness gate
# refuses to build the season until the date lands, so the repair must retry rather than
# fire once and report failure.
#
# WHY --force IS REQUIRED HERE:
# these dates already HAVE an archive object -- it is merely under-filled. Skip-on-exists
# would pass straight over them, so a rerun without --force is a no-op. Note a withheld
# attempt leaves the pre-existing object untouched (verified: the 0.93-fill 2018-08-26
# object survived a withheld --force run), so looping can never degrade what is already
# archived. That is what makes an unattended retry loop safe.
#
# Success is read from the tool's own verdict ("N written"), not from a fill number we
# recompute -- the writer owns the accept/withhold decision and duplicating that threshold
# here would let the two drift apart.
set -uo pipefail

LOG=/home/ec2-user/repair_hrrr.log
REPO=${REPO:-/home/ec2-user/mlb}
PY=$(command -v python3.11 || echo "$HOME/miniconda3/envs/pred/bin/python")
ATTEMPTS=${ATTEMPTS:-8}
WORKERS=${WORKERS:-6}

DATES=("$@")
[ ${#DATES[@]} -gt 0 ] || { echo "usage: $0 <YYYY-MM-DD> [YYYY-MM-DD...]"; exit 2; }

exec >>"$LOG" 2>&1
echo "=== repair start $(date -u +%FT%TZ) dates=${DATES[*]} attempts=$ATTEMPTS ==="
cd "$REPO" || { echo "ABORT: no repo at $REPO"; exit 1; }

# Refuse to contend with an extraction already running on this box: two processes on the
# same date race in Herbie's save dir, which is what silently dropped 41% of tasks before.
if pgrep -f "fetch_nwp_asissue[d]" >/dev/null; then
  echo "ABORT: extraction already running here; not adding a competing writer."
  exit 1
fi

rc_all=0
for d in "${DATES[@]}"; do
  written=0
  for i in $(seq 1 "$ATTEMPTS"); do
    out=$("$PY" data_curation/scripts/fetch_nwp_asissued.py backfill \
            --start "$d" --end "$d" --force --workers "$WORKERS" 2>&1 \
          | tr '\r' '\n' | grep -E "Backfill complete|NOT WRITING" | tail -2)
    echo "$(date -u +%H:%M:%SZ) $d attempt $i/$ATTEMPTS: ${out:-no verdict line}"
    if echo "$out" | grep -qE "Backfill complete: [1-9][0-9]* written"; then
      echo "$(date -u +%H:%M:%SZ) WROTE $d on attempt $i"
      written=1
      break
    fi
  done
  if [ "$written" -eq 0 ]; then
    echo "$(date -u +%H:%M:%SZ) EXHAUSTED $d after $ATTEMPTS attempts — still withheld."
    echo "    The remaining failures may be genuine archive gaps rather than transient;"
    echo "    check whether the missing tasks still probe as existing upstream."
    rc_all=1
  fi
done

echo "=== repair done $(date -u +%FT%TZ) rc=$rc_all ==="
exit "$rc_all"
