#!/bin/bash
# Exception-only watcher for the 4-arm size sweep. Prints NOTHING while arms are healthy;
# speaks up only when an arm reaches a terminal state, and prints the ladder once all four are.
#
# TERMINAL IS DETECTED FROM S3, NOT FROM SSH. Arms B/C/D self-terminate, so their boxes and
# their logs cease to exist; the arm script's EXIT trap uploads sweep_<arm>.log on every exit
# path, which makes "the log object exists" the one signal that survives the box. Presence of
# training_history.json alongside it distinguishes a completed arm from a crashed one.
#
# A hard kill (OOM, spot reclaim) never runs the trap and so never uploads a log. That case is
# caught separately by asking EC2 for the instance state, which is why instance ids are passed
# in rather than rediscovered.
#
# Usage:  bash watch_size_sweep.sh <sweep_id> [interval_s] [max_s]
# Exit:   0 all arms terminal  ·  2 timed out with arms still running
set -uo pipefail

SWEEP_ID=${1:?sweep id, e.g. 20260831}
INTERVAL=${2:-600}
MAX=${3:-36000}
BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
PFX="deep_learning/size_sweep_$SWEEP_ID"

# arm:instance_id — A runs on the persistent GPU box and is expected to stay alive after
# finishing, so it is excluded from the "instance vanished" check.
ARMS="A:i-05b5114c32744b47b B:i-03da63616833d34b7 C:i-0f5bd870f0879b1fb D:i-02e91e14083d4a9bb"

start=$(date +%s)
# A flat string rather than an associative array: this watcher runs on the laptop, where
# /bin/bash is 3.2 and `declare -A` is a syntax error.
REPORTED=""
seen()  { case " $REPORTED " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
mark()  { REPORTED="$REPORTED $1"; }

s3_has() { aws s3api head-object --bucket "$BUCKET" --key "$1" >/dev/null 2>&1; }

while :; do
  all_terminal=1
  for pair in $ARMS; do
    arm=${pair%%:*}; iid=${pair##*:}
    seen "$arm" && continue

    if s3_has "$PFX/$arm/sweep_$arm.log"; then
      if s3_has "$PFX/$arm/training_history.json"; then
        mark "$arm"
        echo "[$(date -u +%FT%TZ)] arm $arm TERMINAL: history present"
      else
        mark "$arm"
        echo "[$(date -u +%FT%TZ)] arm $arm FAILED: log shipped, no training_history.json"
        echo "  aws s3 cp s3://$BUCKET/$PFX/$arm/sweep_$arm.log - | tail -40"
      fi
      continue
    fi

    # No log in S3 yet. Either still training, or killed without running its trap.
    if [ "$arm" != "A" ]; then
      st=$(aws ec2 describe-instances --instance-ids "$iid" \
             --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
      case "$st" in
        terminated|stopped|shutting-down)
          mark "$arm"
          echo "[$(date -u +%FT%TZ)] arm $arm LOST: instance $iid is $st with no log in S3 —"
          echo "  the EXIT trap never ran, so suspect OOM or an external kill."
          continue ;;
      esac
    fi
    all_terminal=0
  done

  [ "$all_terminal" = "1" ] && break
  now=$(date +%s)
  if [ $((now - start)) -ge "$MAX" ]; then
    echo "[$(date -u +%FT%TZ)] WATCHER TIMEOUT after ${MAX}s; still running:"
    for pair in $ARMS; do
      arm=${pair%%:*}; seen "$arm" || echo "  arm $arm"
    done
    exit 2
  fi
  sleep "$INTERVAL"
done

echo
echo "=== SIZE SWEEP $SWEEP_ID: all arms terminal $(date -u +%FT%TZ) ==="
tmp=$(mktemp -d)
for pair in $ARMS; do
  arm=${pair%%:*}
  aws s3 cp "s3://$BUCKET/$PFX/$arm/training_history.json" "$tmp/$arm.json" >/dev/null 2>&1
done
python3 - "$tmp" <<'PYEOF'
import json, sys
from pathlib import Path

# The ladder as launched. head_dim is d_model/8 because n_heads is held at 8; arm D's 16 is
# small enough to be a confound in its own right -- see ec2_size_sweep_arm.sh.
LADDER = {"A": (384, 6), "B": (256, 4), "C": (192, 3), "D": (128, 2)}
rows = []
for arm, (dm, nl) in LADDER.items():
    p = Path(sys.argv[1]) / f"{arm}.json"
    if not p.exists():
        rows.append((arm, dm, nl, None, None, None, None)); continue
    d = json.loads(p.read_text())
    # best_val_loss is per-phase; phases 2/3 report None at 0 epochs, so read phase 1.
    p1 = next((x for x in d.get("phases", []) if x.get("phase") == 1), None)
    if p1 is None:
        rows.append((arm, dm, nl, d.get("n_parameters"), None, None, None)); continue
    hist = p1.get("history", [])
    best = p1.get("best_val_loss")
    best_ep = min((e["epoch"] for e in hist if e.get("val_loss") == best), default=None)
    rows.append((arm, dm, nl, d.get("n_parameters"), best, best_ep, p1.get("epochs_trained")))

print(f"{'arm':<4}{'d_model':>8}{'layers':>7}{'head_dim':>9}{'params':>11}"
      f"{'best_val':>10}{'best_ep':>8}{'epochs':>7}")
for arm, dm, nl, np_, best, be, ep in rows:
    hd = dm // 8
    print(f"{arm:<4}{dm:>8}{nl:>7}{hd:>9}"
          f"{(f'{np_:,}' if np_ else '-'):>11}"
          f"{(f'{best:.4f}' if best is not None else 'MISSING'):>10}"
          f"{(be if be is not None else '-'):>8}{(ep if ep is not None else '-'):>7}")

ok = [r for r in rows if r[4] is not None]
if not ok:
    print("\nno arm produced a val loss")
    raise SystemExit
win = min(ok, key=lambda r: r[4])
print(f"\nlowest val loss: arm {win[0]} (d_model {win[1]}, {win[2]} layers) at {win[4]:.4f}")

# Whether the sweep answered its question at all. It was run because the 18.7M control peaked
# at epoch 3-4 of 12; if every arm still peaks that early, capacity is not what binds and the
# ladder needs to extend further down (or the problem is data, not size).
early = [r[0] for r in ok if r[5] is not None and r[5] <= 3 and (r[6] or 0) >= 8]
if len(early) == len(ok):
    print("EVERY arm still peaks by epoch 3 — shrinking capacity did not stop the overfit; "
          "extend the ladder down or reconsider that capacity is the binding constraint.")
elif early:
    print(f"still peaking by epoch 3: {', '.join(early)}")
if win[0] == "D":
    print("arm D won, and its head_dim is 16 — re-run D at n_heads=2 (head_dim 64) before "
          "concluding anything about depth/width.")
PYEOF
rm -rf "$tmp"
