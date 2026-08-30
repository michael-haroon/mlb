#!/bin/bash
# Exception-only watcher for the weather A/B (ec2_weather_ab.sh on the GPU box).
#
# Prints NOTHING while both arms train normally. Silence is the success signal, so this can
# be left in a background job without producing output that has to be read and discarded.
# It emits a line only for something a human would act on, and exits when the run is over.
#
#   ANOMALY driver-gone   the driver exited without printing its comparison block, i.e. an
#                         arm failed. The last non-progress log lines come with it, because
#                         "it died" without the reason costs another round trip.
#   ANOMALY unreachable   two consecutive ssh failures (one is a normal network blip).
#   ANOMALY stalled       driver alive but the log has not grown in STALL_MIN minutes.
#                         A wedged dataloader looks exactly like slow training otherwise.
#   COMPLETE              the comparison block, printed verbatim, then exit 0.
#
# The remote side does all the filtering (grep/tail/cut), so each poll returns one short
# digest instead of a training log: progress-bar lines are ~200 bytes each and there are
# thousands per epoch.
#
# Usage:  bash deep_learning/watch_weather_ab.sh
#         ONESHOT=1 bash deep_learning/watch_weather_ab.sh
set -u

KEY="${KEY:-$HOME/Documents/SENSITIVE/awstest.pem}"
HOST="${HOST:-ec2-user@32.197.253.24}"
LOG="${LOG:-/home/ec2-user/weather_ab.log}"
DRIVER_PAT="${DRIVER_PAT:-[e]c2_weather_ab}"
INTERVAL="${INTERVAL:-900}"
STALL_MIN="${STALL_MIN:-40}"
ONESHOT="${ONESHOT:-0}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -i $KEY"

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
misses=0

# The bracket in DRIVER_PAT is load-bearing and this script must never contain the
# unbracketed literal: pgrep -f scans full command lines including the remote shell
# evaluating the probe, so a plain pattern matches itself and the driver looks alive
# forever. monitor_hrrr_fleet.sh shipped with exactly that bug.
probe() {
  $SSH "$HOST" "
    alive=\$(pgrep -fc '$DRIVER_PAT' 2>/dev/null || echo 0)
    age=\$(( ( \$(date +%s) - \$(stat -c %Y '$LOG' 2>/dev/null || echo 0) ) / 60 ))
    # Epoch summary lines are the only cheap progress signal that is not a progress bar.
    ep=\$(tr '\r' '\n' < '$LOG' 2>/dev/null | grep -E '^\[[0-9:]+\] INFO \[phase' | tail -1)
    done=\$(grep -c 'positive delta = weather helped' '$LOG' 2>/dev/null || echo 0)
    echo \"alive=\$alive age=\$age done=\$done\"
    echo \"ep=\$ep\"
  " 2>/dev/null
}

pass() {
  local out alive age ndone ep
  out=$(probe)
  if [ -z "$out" ]; then
    misses=$((misses + 1))
    [ "$misses" -ge 2 ] && { say "ANOMALY unreachable x$misses ($HOST)"; return 1; }
    return 0
  fi
  misses=0
  alive=$(echo "$out" | sed -n 's/.*alive=\([0-9]*\).*/\1/p')
  age=$(echo "$out" | sed -n 's/.*age=\([0-9-]*\).*/\1/p')
  ndone=$(echo "$out" | sed -n 's/.*done=\([0-9]*\).*/\1/p')
  ep=$(echo "$out" | sed -n 's/^ep=//p')

  if [ "${ndone:-0}" -gt 0 ]; then
    say "COMPLETE — comparison block:"
    $SSH "$HOST" "tr '\r' '\n' < '$LOG' | grep -E 'best_val|treatment - control|vs best unseeded|ARM |arm ' | tail -20" 2>/dev/null
    return 2
  fi
  if [ "${alive:-0}" -eq 0 ]; then
    say "ANOMALY driver-gone without a comparison block — an arm failed. last: $ep"
    $SSH "$HOST" "tr '\r' '\n' < '$LOG' | grep -vE 'it/s|s/it' | tail -15 | cut -c1-190" 2>/dev/null
    return 2
  fi
  if [ "${age:-0}" -ge "$STALL_MIN" ]; then
    say "ANOMALY stalled — driver alive but log idle ${age}m. last: $ep"
    return 1
  fi
  return 0
}

while :; do
  pass; rc=$?
  [ "$rc" -eq 2 ] && exit 0
  [ "$ONESHOT" = "1" ] && exit "$rc"
  sleep "$INTERVAL"
done
