#!/bin/bash
# Exception-only watcher for chain_backfill_2026_tail.sh. Runs LOCALLY, polls the shard box,
# and stays silent until the chain reaches a terminal state — then prints one block and exits.
#
# WHY IT IS SHAPED THIS WAY: the chain is ~1h of fetching plus three builds, and polling it
# from a chat turn burns a full context read per check for a line that almost always says
# "still going". So all the decision logic lives here in bash; the caller learns exactly one
# thing (terminal state + the evidence needed to act on it) exactly once.
#
# Terminal states, distinguished because they need different responses:
#   COMPLETE   chain printed its final marker — read the norm-stats output, move on
#   FAILED     a gated step exited nonzero — the chain stopped itself, tail shows which
#   DIED       no chain process and no terminal marker: the box rebooted or was OOM-killed,
#              which looks identical to "finished" if you only grep for failure strings
#   UNREACHABLE  ssh failed N times running — do not report a data problem for a network one
#
# Usage: nohup bash data_curation/scripts/watch_chain_2026_tail.sh > ~/chain_watch.out 2>&1 &
set -u

HOST=${HOST:-ec2-user@54.158.139.25}
KEY=${KEY:-/Users/michaelharoon/Documents/SENSITIVE/awstest.pem}
LOG=${LOG:-/home/ec2-user/chain_2026_tail.log}
INTERVAL=${INTERVAL:-180}
MAX_SSH_FAIL=${MAX_SSH_FAIL:-5}

ssh_fail=0
while :; do
  # One round trip collects everything: terminal markers, liveness, and the tail. Splitting
  # these into separate ssh calls would let the chain finish between them and race.
  out=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
        "$HOST" "
    # Scope every count to the CURRENT run. The chain opens its log with 'exec >>', so a
    # superseded attempt's failure line lives there forever: the first 2026 launch died at
    # 'GATE FAILED 2026' on a stale game_meta, and grepping the whole file reported that
    # dead failure while the relaunched chain was 17 dates deep and perfectly healthy.
    # Slicing from the last 'chain start' marker is what makes the signal about now.
    s=\$(grep -n '=== chain start' $LOG 2>/dev/null | tail -1 | cut -d: -f1)
    seg=\$(tail -n +\${s:-1} $LOG 2>/dev/null)
    # No '|| echo 0' fallbacks here: grep -c and pgrep -c already PRINT 0 while EXITING 1,
    # so a fallback appends a second token and silently corrupts the STATE line the caller
    # parses. tail -1 + \${x:-0} covers the only real empty case (log not created yet).
    d=\$(printf '%s\n' \"\$seg\" | grep -c 'CHAIN COMPLETE' | tail -1)
    f=\$(printf '%s\n' \"\$seg\" | grep -cE '^(FETCH FAILED|GATE FAILED|BUILD FAILED|PITCH-OFFSETS FAILED|NORM-STATS FAILED|ABORT)' | tail -1)
    a=\$(pgrep -fc 'chain_backfill_2026_tail|fetch_nwp_asissued|build_weather_asof' 2>/dev/null | tail -1)
    n=\$(printf '%s\n' \"\$seg\" | grep -c 'tasks, ' | tail -1)
    echo \"STATE done=\${d:-0} fail=\${f:-0} alive=\${a:-0} dates=\${n:-0}\"
    tail -12 $LOG 2>/dev/null
  " 2>/dev/null)

  if [ -z "$out" ]; then
    ssh_fail=$((ssh_fail + 1))
    if [ "$ssh_fail" -ge "$MAX_SSH_FAIL" ]; then
      echo "=== UNREACHABLE — $MAX_SSH_FAIL consecutive ssh failures to $HOST ==="
      echo "Chain state unknown. Check the box before assuming the data is bad."
      exit 2
    fi
    sleep "$INTERVAL"; continue
  fi
  ssh_fail=0

  state=$(printf '%s\n' "$out" | grep '^STATE ' | head -1)
  eval "$(printf '%s\n' "$state" | sed 's/^STATE //')"

  if [ "${done:-0}" -gt 0 ]; then
    echo "=== CHAIN COMPLETE ($dates dates fetched) ==="; printf '%s\n' "$out"; exit 0
  fi
  if [ "${fail:-0}" -gt 0 ]; then
    echo "=== CHAIN FAILED at a gated step ==="; printf '%s\n' "$out"; exit 1
  fi
  # No marker and no process. Checked last so a just-finished chain reports COMPLETE, not
  # DIED, in the window between the process exiting and the marker being flushed.
  if [ "${alive:-0}" -eq 0 ]; then
    echo "=== CHAIN DIED — no process, no terminal marker ($dates dates fetched) ==="
    echo "Likely OOM or reboot. Nothing downstream was gated on this, so the archive is"
    echo "short but not corrupt: rerun the chain, done dates are skipped via S3 head."
    printf '%s\n' "$out"; exit 3
  fi
  sleep "$INTERVAL"
done
