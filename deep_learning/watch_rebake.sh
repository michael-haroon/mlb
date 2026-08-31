#!/bin/bash
# Exception-only watcher for rebake_prepared_tensors.sh. Runs LOCALLY, silent until terminal.
#
# Same shape and the same two hard-won guards as watch_chain_2026_tail.sh:
#   - slice the log from its last '=== rebake start' marker, because the script opens with
#     'exec >>' and a superseded attempt's failure line would otherwise be read as live state
#   - never write `$(grep -c X) || echo 0`: grep -c and pgrep -c PRINT 0 while EXITING 1, so
#     the fallback emits a second token and corrupts the STATE line the caller eval's
# DIED is checked last so a just-finished run reports COMPLETE, not DIED.
set -u

HOST=${HOST:-ec2-user@32.197.253.24}
KEY=${KEY:-/Users/michaelharoon/Documents/SENSITIVE/awstest.pem}
LOG=${LOG:-/home/ec2-user/rebake.log}
INTERVAL=${INTERVAL:-180}
MAX_SSH_FAIL=${MAX_SSH_FAIL:-5}
# Overridable so the same watcher covers a resumed single stage. Defaults are the full rebake.
START_RE=${START_RE:-=== rebake start}
DONE_RE=${DONE_RE:-REBAKE STAGE 1-3 COMPLETE}
FAIL_RE=${FAIL_RE:-^(ABORT|SYNC FAILED|POPULATION CHECK FAILED|CACHE BUILD FAILED|PRECOLLATE FAILED)}
PROC_RE=${PROC_RE:-rebake_prepared_tensors|dataset_cache|train_unified}

ssh_fail=0
while :; do
  out=$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
        "$HOST" "
    s=\$(grep -n '$START_RE' $LOG 2>/dev/null | tail -1 | cut -d: -f1)
    seg=\$(tail -n +\${s:-1} $LOG 2>/dev/null)
    d=\$(printf '%s\n' \"\$seg\" | grep -c '$DONE_RE' | tail -1)
    f=\$(printf '%s\n' \"\$seg\" | grep -cE '$FAIL_RE' | tail -1)
    a=\$(pgrep -fc '$PROC_RE' 2>/dev/null | tail -1)
    # Memory is reported every poll, not just on failure: the first attempt livelocked at
    # 97.4% and the only reason we could not see it coming was that nothing sampled RAM.
    m=\$(free -g | awk '/^Mem:/{print \$3\"/\"\$2\"GB used\"} /^Swap:/{print \"swap \"\$3\"/\"\$2\"GB\"}' | paste -sd' ')
    echo \"STATE done=\${d:-0} fail=\${f:-0} alive=\${a:-0}\"
    echo \"MEM \$m\"
    tail -15 $LOG 2>/dev/null | cut -c1-200
  " 2>/dev/null)

  if [ -z "$out" ]; then
    ssh_fail=$((ssh_fail + 1))
    if [ "$ssh_fail" -ge "$MAX_SSH_FAIL" ]; then
      echo "=== UNREACHABLE — $MAX_SSH_FAIL consecutive ssh failures to $HOST ==="
      exit 2
    fi
    sleep "$INTERVAL"; continue
  fi
  ssh_fail=0

  eval "$(printf '%s\n' "$out" | grep '^STATE ' | head -1 | sed 's/^STATE //')"

  if [ "${done:-0}" -gt 0 ]; then
    echo "=== REBAKE COMPLETE (stages 1-3) ==="; printf '%s\n' "$out"; exit 0
  fi
  if [ "${fail:-0}" -gt 0 ]; then
    echo "=== REBAKE FAILED at a gated step ==="; printf '%s\n' "$out"; exit 1
  fi
  if [ "${alive:-0}" -eq 0 ]; then
    echo "=== REBAKE DIED — no process, no terminal marker ==="
    echo "Most likely a cgroup OOM kill: build_and_save holds frames + all three split"
    echo "datasets live at once and measured 30.3GB RSS / 97.4% of RAM on 2026-08-31."
    echo "Confirm with: journalctl -k --no-pager | grep -i 'oom\|Killed process'"
    echo "prepared_tensors_new is incomplete; nothing swapped, so the A/B set is intact."
    printf '%s\n' "$out"; exit 3
  fi
  sleep "$INTERVAL"
done
