#!/bin/bash
# Exception-only watcher proving the trading box actually CONVERGES after the 2026-08-30
# sync fan-out incident. This is the regression test for that fix, run against production.
#
# Background: master mirrors the full ~9.4 GB S3 lake into artifacts/raw_cache on its root
# volume. When that volume filled (20 GB, 100%), every `aws s3 sync` failed part-way, so
# _refresh_known_pks never ran, so "N finalized game(s) missing from features" stayed true
# forever, so the ~60s scan re-fired -- and because _sync_s3 was unguarded, each firing
# launched another full-lake sync. 45 were observed in flight at load 51 on 2 vCPUs.
#
# Two fixes shipped: a _sync_lock in features.py (caps concurrency at 1) and growing the
# volume to 48 GB (lets the sync finish at all). The lock alone would only have made the box
# fail quietly forever, so the ONLY convincing proof is convergence, not a low load average.
#
#   COMPLETE            raw_cache object count reached the S3 count AND the runner stopped
#                       reporting missing finalized games. This is the success condition.
#   ANOMALY fan-out     more than one sync in flight. The fix regressed, or a code path
#                       other than _sync_s3 spawns syncs. This is the one that matters most.
#   ANOMALY disk        root volume above DISK_WARN%. The originating cause; the lake grows
#                       daily, so this ceiling WILL be hit again.
#   ANOMALY runner-gone the trading process died.
#   ANOMALY stalled     sync alive but the cached object count has not grown in STALL_MIN.
#
# Deliberately does NOT kill anything: pkill is classifier-blocked in the agent that wrote
# this, and a watcher that mutates production is a watcher you cannot leave running.
#
# Usage:  bash trading/scripts/watch_master_convergence.sh
#         ONESHOT=1 bash trading/scripts/watch_master_convergence.sh
set -u

KEY="${KEY:-$HOME/Documents/SENSITIVE/awstest.pem}"
HOST="${HOST:-ec2-user@3.81.9.231}"
CACHE="${CACHE:-/home/ec2-user/pregame/artifacts/raw_cache}"
BUCKET="${BUCKET:-mlb-265753586044-us-east-1-an}"
# Bracketed so the pattern cannot match the remote shell evaluating it: pgrep -f scans full
# command lines. This script must never contain the unbracketed literals either -- an earlier
# manual `pkill -f "aws s3 sync"` killed its own parent shell before it could print.
SYNC_PAT="${SYNC_PAT:-[a]ws s3 sync}"
RUN_PAT="${RUN_PAT:-[p]regame.trading.runner}"
INTERVAL="${INTERVAL:-300}"
STALL_MIN="${STALL_MIN:-25}"
DISK_WARN="${DISK_WARN:-85}"
ONESHOT="${ONESHOT:-0}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $KEY"

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
misses=0; prev_files=-1; stall_since=0

# S3 side is counted once: 67k ListObjects pages per poll would dominate the cost of this
# watcher, and the target only grows by ~15 games/day.
S3_COUNT=$(aws s3 ls "s3://$BUCKET/data/" --recursive 2>/dev/null | wc -l | tr -d ' ')
[ "${S3_COUNT:-0}" -lt 1000 ] && { say "ANOMALY could not count s3://$BUCKET/data/ (got '$S3_COUNT')"; exit 1; }
say "watching $HOST -> convergence target $S3_COUNT objects"

probe() {
  $SSH "$HOST" "
    echo \"sync=\$(pgrep -fc '$SYNC_PAT' 2>/dev/null)\"
    echo \"run=\$(pgrep -fc '$RUN_PAT' 2>/dev/null)\"
    echo \"files=\$(find '$CACHE' -type f 2>/dev/null | wc -l)\"
    echo \"disk=\$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc '0-9')\"
    echo \"load=\$(cut -d' ' -f1 /proc/loadavg)\"
    # The stuck trigger itself. If this line stops appearing, the loop is broken for real.
    echo \"missing=\$(tmux capture-pane -p -S -400 -t 0:0 2>/dev/null | grep -c 'finalized game(s) missing')\"
  " 2>/dev/null
}

pass() {
  local out sync run files disk load missing
  out=$(probe)
  if [ -z "$out" ]; then
    misses=$((misses + 1))
    [ "$misses" -ge 2 ] && { say "ANOMALY unreachable x$misses ($HOST)"; return 1; }
    return 0
  fi
  misses=0
  sync=$(echo "$out" | sed -n 's/^sync=//p'); run=$(echo "$out" | sed -n 's/^run=//p')
  files=$(echo "$out" | sed -n 's/^files=//p'); disk=$(echo "$out" | sed -n 's/^disk=//p')
  load=$(echo "$out" | sed -n 's/^load=//p'); missing=$(echo "$out" | sed -n 's/^missing=//p')

  [ "${sync:-0}" -gt 1 ] && say "ANOMALY fan-out — $sync concurrent syncs (load $load). _sync_lock regressed."
  [ "${disk:-0}" -ge "$DISK_WARN" ] && say "ANOMALY disk ${disk}% — the originating cause; lake grows daily."
  if [ "${run:-0}" -eq 0 ]; then
    say "ANOMALY runner-gone — trading process not running (load $load, disk ${disk}%)"
    return 2
  fi

  if [ "${files:-0}" -ge "$S3_COUNT" ] && [ "${missing:-1}" -eq 0 ]; then
    say "COMPLETE — converged: $files/$S3_COUNT objects cached, no missing-game trigger, load $load, disk ${disk}%"
    return 2
  fi

  # Stall is measured on cached-object growth, not log mtime: a wedged sync still writes
  # progress chatter, so only the count is trustworthy evidence of forward motion.
  if [ "${files:-0}" -gt "$prev_files" ]; then
    stall_since=0
  elif [ "${sync:-0}" -ge 1 ]; then
    stall_since=$((stall_since + INTERVAL))
    if [ "$stall_since" -ge $((STALL_MIN * 60)) ]; then
      say "ANOMALY stalled — sync alive but cache stuck at $files/$S3_COUNT for $((stall_since/60))m"
      stall_since=0
    fi
  fi
  prev_files=${files:-0}
  return 0
}

while :; do
  pass; rc=$?
  [ "$rc" -eq 2 ] && exit 0
  [ "$ONESHOT" = "1" ] && exit "$rc"
  sleep "$INTERVAL"
done
