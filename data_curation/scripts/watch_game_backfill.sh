#!/bin/bash
# Exception-only watcher for the download_history.py game-ingestion backfill.
#
# Prints NOTHING while ingestion is progressing. Silence is the success signal, so this
# can sit in a background job without producing output that has to be read and discarded.
# It emits a line only for something worth acting on, and exits when the run is over.
#
#   COMPLETE            "INGESTION COMPLETELY FINISHED" reached; prints the tail and the
#                       new checkpoint count, then exits 0.
#   ANOMALY proc-gone   the process left without that banner, i.e. it died. The last
#                       non-progress-bar lines come with it, because "it died" without
#                       the reason costs another round trip.
#   ANOMALY stalled     process alive but the log has not grown in STALL_MIN minutes.
#                       A wedged HTTP retry loop looks exactly like slow scraping.
#   ANOMALY failures    the retry queue grew, meaning games are being abandoned rather
#                       than ingested. Not fatal -- --retry can sweep them -- but silent
#                       partial ingestion is what produces a feature store with holes.
#
# The remote side does all the filtering. tqdm writes one ~120-byte progress line per
# game with \r, so an unfiltered tail is thousands of lines; `tr '\r' '\n'` plus a
# progress-bar grep -v is what keeps each poll to a couple of short lines.
#
# Usage:  bash data_curation/scripts/watch_game_backfill.sh
#         ONESHOT=1 bash data_curation/scripts/watch_game_backfill.sh
set -u

KEY="${KEY:-$HOME/Documents/SENSITIVE/awstest.pem}"
HOST="${HOST:-ec2-user@54.158.139.25}"
LOG="${LOG:-/home/ec2-user/backfill_games.log}"
# Bracketed so the pattern cannot match the remote shell that is evaluating it: pgrep -f
# scans full command lines. monitor_hrrr_fleet.sh shipped with exactly that bug, and the
# rest of this probe must never contain the unbracketed literal either.
PROC_PAT="${PROC_PAT:-[d]ownload_history.py --live}"
BUCKET="${BUCKET:-mlb-265753586044-us-east-1-an}"
INTERVAL="${INTERVAL:-600}"
STALL_MIN="${STALL_MIN:-20}"
ONESHOT="${ONESHOT:-0}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -i $KEY"

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
misses=0
prev_fail=-1

probe() {
  $SSH "$HOST" "
    alive=\$(pgrep -fc '$PROC_PAT' 2>/dev/null || echo 0)
    age=\$(( ( \$(date +%s) - \$(stat -c %Y '$LOG' 2>/dev/null || echo 0) ) / 60 ))
    done=\$(grep -c 'INGESTION COMPLETELY FINISHED' '$LOG' 2>/dev/null || echo 0)
    fail=\$(grep -c 'mark_failed\|queued for retry' '$LOG' 2>/dev/null || echo 0)
    # Last real milestone, not a progress bar: flush/checkpoint lines are the only
    # cheap evidence that games are actually landing in S3.
    last=\$(tr '\r' '\n' < '$LOG' 2>/dev/null | grep -vE 'Scraping Progress|it/s' | tail -1 | cut -c1-150)
    echo \"alive=\$alive age=\$age done=\$done fail=\$fail\"
    echo \"last=\$last\"
  " 2>/dev/null
}

pass() {
  local out alive age ndone nfail last
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
  nfail=$(echo "$out" | sed -n 's/.*fail=\([0-9]*\).*/\1/p')
  last=$(echo "$out" | sed -n 's/^last=//p')

  # Report the retry queue only when it GROWS, so a permanent record in the log does
  # not produce an identical line every interval.
  if [ "$prev_fail" -ge 0 ] && [ "${nfail:-0}" -gt "$prev_fail" ]; then
    say "ANOMALY failures grew to $nfail (games abandoned, not ingested) -- --retry sweep needed"
  fi
  prev_fail=${nfail:-0}

  if [ "${ndone:-0}" -gt 0 ]; then
    say "COMPLETE — ingestion finished:"
    $SSH "$HOST" "tr '\r' '\n' < '$LOG' | grep -vE 'Scraping Progress|it/s' | tail -12 | cut -c1-170" 2>/dev/null
    # The checkpoint is the authority on what actually landed; the log only says what
    # was attempted. 165,634 was the count before this run.
    aws s3 cp "s3://$BUCKET/data/checkpoint.json" - 2>/dev/null \
      | python3 -c "import json,sys; print('checkpoint completed games now:', len(json.load(sys.stdin).get('completed',[])))" 2>/dev/null
    return 2
  fi
  if [ "${alive:-0}" -eq 0 ]; then
    say "ANOMALY proc-gone without the finish banner -- ingestion died. last: $last"
    $SSH "$HOST" "tr '\r' '\n' < '$LOG' | grep -vE 'Scraping Progress|it/s' | tail -15 | cut -c1-170" 2>/dev/null
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
