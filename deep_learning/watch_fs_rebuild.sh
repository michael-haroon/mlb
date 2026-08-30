#!/bin/bash
# Exception-only watcher for rebuild_feature_store_2026.py on shard A.
#
# Prints NOTHING while the rebuild is progressing. Silence is the success signal, so this can
# sit in a background job without emitting output that has to be read and discarded. It emits
# a line only for something worth acting on, and exits when the run is over.
#
#   COMPLETE            "REBUILD VERIFIED" reached; prints the artifact sizes and the
#                       game_meta verification block, then exits 0.
#   ANOMALY verify-fail the build finished but the 2026-06-20 advance or the duplicate-pk
#                       check failed. This is the case that must never be promoted silently,
#                       so it comes with the failing line.
#   ANOMALY proc-gone   the process left without either banner, i.e. it died (OOM is the
#                       likely cause on a 15 GB box; the tail distinguishes that from a bug).
#   ANOMALY stalled     alive but the log has not grown in STALL_MIN minutes. A wedged S3
#                       read looks exactly like a slow season otherwise.
#   ANOMALY read-errors a season's file loader reported non-zero errors. The builder does not
#                       fail on these -- it silently ships a store with missing rows, which is
#                       precisely the corruption this watch exists to catch.
#
# The remote side does all the filtering. The per-file progress lines are written with \r and
# there are thousands per season, so `tr '\r' '\n'` plus a progress grep -v is what keeps each
# poll to a couple of short lines.
#
# Usage:  bash deep_learning/watch_fs_rebuild.sh
#         ONESHOT=1 bash deep_learning/watch_fs_rebuild.sh
set -u

KEY="${KEY:-$HOME/Documents/SENSITIVE/awstest.pem}"
HOST="${HOST:-ec2-user@54.158.139.25}"
LOG="${LOG:-/home/ec2-user/fs_rebuild.log}"
# Bracketed so the pattern cannot match the remote shell evaluating it: pgrep -f scans full
# command lines. The rest of this probe must never contain the unbracketed literal either.
PROC_PAT="${PROC_PAT:-[r]ebuild_feature_store_2026}"
INTERVAL="${INTERVAL:-900}"
STALL_MIN="${STALL_MIN:-30}"
ONESHOT="${ONESHOT:-0}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -i $KEY"

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
misses=0
prev_err=-1

probe() {
  $SSH "$HOST" "
    alive=\$(pgrep -fc '$PROC_PAT' 2>/dev/null || echo 0)
    age=\$(( ( \$(date +%s) - \$(stat -c %Y '$LOG' 2>/dev/null || echo 0) ) / 60 ))
    ok=\$(grep -c 'REBUILD VERIFIED' '$LOG' 2>/dev/null || echo 0)
    bad=\$(grep -c 'VERIFY FAILED' '$LOG' 2>/dev/null || echo 0)
    # 'N errors' with N>0 in any loader line. The builder logs these and carries on.
    err=\$(tr '\r' '\n' < '$LOG' 2>/dev/null | grep -oE '[0-9]+ errors' | grep -vc '^0 errors' || echo 0)
    last=\$(tr '\r' '\n' < '$LOG' 2>/dev/null | grep -E '^\[[0-9:]+\]' | tail -1 | cut -c1-150)
    echo \"alive=\$alive age=\$age ok=\$ok bad=\$bad err=\$err\"
    echo \"last=\$last\"
  " 2>/dev/null
}

pass() {
  local out alive age nok nbad nerr last
  out=$(probe)
  if [ -z "$out" ]; then
    misses=$((misses + 1))
    [ "$misses" -ge 2 ] && { say "ANOMALY unreachable x$misses ($HOST)"; return 1; }
    return 0
  fi
  misses=0
  alive=$(echo "$out" | sed -n 's/.*alive=\([0-9]*\).*/\1/p')
  age=$(echo "$out" | sed -n 's/.*age=\([0-9-]*\).*/\1/p')
  nok=$(echo "$out" | sed -n 's/.*ok=\([0-9]*\).*/\1/p')
  nbad=$(echo "$out" | sed -n 's/.*bad=\([0-9]*\).*/\1/p')
  nerr=$(echo "$out" | sed -n 's/.*err=\([0-9]*\).*/\1/p')
  last=$(echo "$out" | sed -n 's/^last=//p')

  # Report read errors only when the count GROWS, so a single bad season does not produce an
  # identical line every interval for the rest of the run.
  if [ "$prev_err" -ge 0 ] && [ "${nerr:-0}" -gt "$prev_err" ]; then
    say "ANOMALY read-errors: $nerr loader lines with non-zero errors (store will have holes)"
  fi
  prev_err=${nerr:-0}

  if [ "${nbad:-0}" -gt 0 ]; then
    say "ANOMALY verify-fail — do NOT promote:"
    $SSH "$HOST" "grep -E 'VERIFY' '$LOG' | tail -10 | cut -c1-190" 2>/dev/null
    return 2
  fi
  if [ "${nok:-0}" -gt 0 ]; then
    say "COMPLETE — rebuild verified:"
    $SSH "$HOST" "grep -E 'REBUILD|VERIFY|MB\$|  [a-z_]+ +[0-9]' '$LOG' | tail -30 | cut -c1-190" 2>/dev/null
    return 2
  fi
  if [ "${alive:-0}" -eq 0 ]; then
    say "ANOMALY proc-gone without a verify banner — rebuild died. last: $last"
    $SSH "$HOST" "tr '\r' '\n' < '$LOG' | grep -vE 'files read' | tail -15 | cut -c1-190" 2>/dev/null
    return 2
  fi
  if [ "${age:-0}" -ge "$STALL_MIN" ]; then
    say "ANOMALY stalled — alive but log idle ${age}m. last: $last"
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
