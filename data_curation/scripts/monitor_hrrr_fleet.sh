#!/bin/bash
# Exception-only monitor for the HRRR as-issued shard fleet.
#
# Prints NOTHING while the fleet is healthy. Silence is the success signal, so this
# can be tailed cheaply (or left in a background job) without generating output that
# has to be read and discarded. It emits a line only for a condition a human would
# act on:
#
#   WITHHELD <box>   fetch_nwp_asissued refused to persist a date (MIN_WRITE_FILL).
#                    Not corruption -- the opposite -- but it means a date is absent
#                    and needs a rerun, so it must not stay quiet.
#   REPAIRFAIL <box> a --force repair exhausted its retries.
#   UNREACHABLE      ssh failed twice in a row. One failure is normal (spot network
#                    blips); two consecutive is worth surfacing.
#   STALLED          a box is working but no new S3 object fleet-wide for STALL_MIN
#                    minutes. Herbie can hang on a 503 storm without exiting.
#   DONE             every box idle. This is NOT proof of success: a dead shard and a
#                    finished one look identical from process state, so DONE is a cue
#                    to run `verify_weather_archives.py coverage`, which compares the
#                    archive against the population and is the only authority.
#
# Usage:  ./monitor_hrrr_fleet.sh            # loop until DONE
#         ONESHOT=1 ./monitor_hrrr_fleet.sh  # single pass, for cron/manual checks

set -u
KEY="${KEY:-$HOME/Documents/SENSITIVE/awstest.pem}"
BUCKET="${BUCKET:-mlb-265753586044-us-east-1-an}"
PREFIX="${PREFIX:-data/weather/source=hrrr_asissued/}"
INTERVAL="${INTERVAL:-600}"
STALL_MIN="${STALL_MIN:-25}"
ONESHOT="${ONESHOT:-0}"

# box:ip. Kept inline rather than in a config file so the monitor is a single
# self-contained artifact -- bash 3.2 on macOS has no associative arrays.
BOXES="${BOXES:-A:54.158.139.25 B:3.92.146.230 C:3.84.170.201 D:98.87.153.153 E:18.209.104.142 F:3.80.199.126}"

# Overridable so the detection paths can be tested against a stub instead of a real
# box -- a monitor whose alarms have never fired is indistinguishable from a broken one.
SSH="${SSH:-ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes}"
say() { echo "$(date -u +%H:%M:%SZ) $*"; }

# Consecutive-miss counters, one file per box: cheaper and bash-3.2 safe.
STATE="${STATE:-${TMPDIR:-/tmp}/hrrr_monitor_state}"
mkdir -p "$STATE"

newest_epoch() {
  # Newest LastModified across the archive, as a unix epoch. Cheap server-side query.
  local ts
  ts=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$PREFIX" \
        --query 'sort_by(Contents,&LastModified)[-1].LastModified' --output text 2>/dev/null)
  if [ -z "$ts" ] || [ "$ts" = "None" ]; then echo 0; return; fi
  # TZ=UTC is load-bearing: S3 reports LastModified in UTC, but BSD `date -j` parses a
  # naive timestamp in the LOCAL zone. Without it, a PDT laptop reads every object as
  # 7 hours in the future, new_age goes negative, and the stall alarm can never fire.
  TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%S" "${ts%%+*}" +%s 2>/dev/null \
    || date -d "$ts" +%s 2>/dev/null || echo 0
}

pass() {
  local anomaly=0 idle=0 nbox=0 active=0
  local last new_age
  last=$(newest_epoch)
  new_age=$(( ( $(date +%s) - last ) / 60 ))

  for spec in $BOXES; do
    local n=${spec%%:*} ip=${spec##*:} out
    nbox=$((nbox + 1))
    out=$($SSH "ec2-user@$ip" '
        echo "procs=$(pgrep -fc "fetch_nwp_asissued.py backfill" 2>/dev/null || echo 0)"
        echo "withheld=$(grep -hc "NOT WRITING" ~/*.log 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)"
        echo "repairfail=$(grep -hc "^REPAIR FAILED" ~/hrrr_repair.log 2>/dev/null || echo 0)"
      ' 2>/dev/null)

    if [ -z "$out" ]; then
      local f="$STATE/miss_$n"; local c=$(( $(cat "$f" 2>/dev/null || echo 0) + 1 ))
      echo "$c" >"$f"
      [ "$c" -ge 2 ] && { say "UNREACHABLE $n ($ip) x$c"; anomaly=1; }
      continue
    fi
    echo 0 >"$STATE/miss_$n"

    local procs withheld rfail
    procs=$(echo "$out" | sed -n 's/^procs=//p'); procs=${procs:-0}
    withheld=$(echo "$out" | sed -n 's/^withheld=//p'); withheld=${withheld:-0}
    rfail=$(echo "$out" | sed -n 's/^repairfail=//p'); rfail=${rfail:-0}

    # Report withheld/failed counts only when they GROW, so a permanent record in a
    # log does not produce a line every interval.
    local pw="$STATE/withheld_$n"; local prev=$(cat "$pw" 2>/dev/null || echo 0)
    if [ "$withheld" -gt "$prev" ]; then
      say "WITHHELD $n $withheld date(s) refused (transient loss) -- rerun needed"
      anomaly=1
    fi
    echo "$withheld" >"$pw"

    local pf="$STATE/rfail_$n"; prev=$(cat "$pf" 2>/dev/null || echo 0)
    if [ "$rfail" -gt "$prev" ]; then say "REPAIRFAIL $n $rfail"; anomaly=1; fi
    echo "$rfail" >"$pf"

    if [ "$procs" -eq 0 ]; then idle=$((idle + 1)); else active=$((active + 1)); fi
  done

  # Staleness is a property of the archive, not of one box, so it is judged once after
  # the loop. Doing it inside meant keying the report to a hardcoded box name, which
  # silently disabled the alarm for any fleet not containing that box.
  if [ "$active" -gt 0 ] && [ "$new_age" -ge "$STALL_MIN" ]; then
    say "STALLED fleet: $active box(es) working but no new archive object for ${new_age}m"
    anomaly=1
  fi

  if [ "$idle" -eq "$nbox" ]; then
    # Every shard idle. Distinguish finished from dead by asking the population, which
    # is the only authority -- S3 counts alone cannot tell "season done" from
    # "season abandoned".
    say "all $nbox boxes idle -- running coverage gate"
    return 2
  fi
  return "$anomaly"
}

while :; do
  pass; rc=$?
  if [ "$rc" -eq 2 ]; then say "DONE (fleet idle)"; exit 0; fi
  [ "$ONESHOT" = "1" ] && exit "$rc"
  sleep "$INTERVAL"
done
