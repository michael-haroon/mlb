#!/bin/bash
# Runs ON one GPU box. Trains ONE arm of the capacity size-down sweep and ships its results
# to S3. One box per arm; the four arms run concurrently on four boxes.
#
# WHY THIS SWEEP EXISTS:
# the weather A/B (2026-08-30) found both arms reached best val at epoch 3-4 of a 12-epoch
# schedule with an 18.7M-parameter model on ~315k samples. At that point capacity binds, and a
# feature comparison run at a capacity that is already memorising cannot resolve anything --
# whichever way it lands. So the size ladder must be measured before any further feature A/B.
# See deep_learning/ec2_weather_ab.sh and [[project-weather-ab-verdict-2026-08-30]].
#
# WHAT IS HELD CONSTANT, AND WHY IT MATTERS:
#   n_heads=8, lr=4e-4, batch=64, seed=42, 12 epochs, dropout 0.1, weight_decay 0.01.
# The previous architecture sweep was VOIDED because learning rate moved with architecture, so
# "bigger was worse" and "the LR was wrong for that size" were inseparable. Holding LR fixed
# means this sweep answers exactly one question: at the control's optimiser settings, does
# less capacity generalise better? A smaller arm's own LR optimum is a SEPARATE experiment --
# do not fold it in here.
#
# KNOWN CONFOUND, stated rather than hidden: n_heads is fixed at 8, so head_dim = d_model/8
# falls out as 48/32/24/16 across arms A-D. Arm D's head_dim of 16 is small enough to limit
# per-head expressiveness on its own. If D wins, re-run D at n_heads=2 (head_dim 64) before
# concluding anything about depth/width. head_dim could NOT be held at 48 as originally
# planned: 48 divides 384 and 192 but neither 256 nor 128.
#
# Usage (env-driven):
#   ARM=A D_MODEL=384 N_LAYERS=6 SWEEP_ID=20260831 bash ec2_size_sweep_arm.sh
# Log: ~/sweep_<ARM>.log
set -uo pipefail

ARM=${ARM:?set ARM (A|B|C|D)}
D_MODEL=${D_MODEL:?set D_MODEL}
N_LAYERS=${N_LAYERS:?set N_LAYERS}
SWEEP_ID=${SWEEP_ID:?set SWEEP_ID so all arms land under one prefix}
N_HEADS=${N_HEADS:-8}
SEED=${SEED:-42}
BATCH=${BATCH:-64}
LR=${LR:-4e-04}
P1=${P1:-12}
WORKERS=${WORKERS:-4}
SHUTDOWN=${SHUTDOWN:-0}

BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
TENSORS_S3="s3://$BUCKET/deep_learning/prepared_tensors_20260831"
FS_S3="s3://$BUCKET/deep_learning/feature_store"
OUT_S3="s3://$BUCKET/deep_learning/size_sweep_$SWEEP_ID/$ARM"

REPO=/home/ec2-user/mlb
PY=/home/ec2-user/miniconda3/envs/pred/bin/python
FS=/mnt/fast/feature_store
PREPARED=/mnt/fast/prepared_tensors
RUN=/mnt/fast/sweep_$ARM
LOG=/home/ec2-user/sweep_$ARM.log

exec >>"$LOG" 2>&1
echo "=== sweep arm $ARM start $(date -u +%FT%TZ) d_model=$D_MODEL n_layers=$N_LAYERS n_heads=$N_HEADS ==="
fail() { echo "ABORT: $*"; exit 1; }

# Ship the log and whatever artifacts exist on EVERY exit path. On the transient sweep boxes
# the instance store AND the box itself disappear at shutdown, so an un-uploaded log is an
# unexplainable failure. The weather A/B lost a completed arm's checkpoint exactly this way
# before it grew a trap.
ship() {
  local rc=$?
  echo "=== arm $ARM exiting rc=$rc $(date -u +%FT%TZ) ==="
  aws s3 cp "$LOG" "$OUT_S3/sweep_$ARM.log" >/dev/null 2>&1
  for f in training_history.json phase1/best.pt; do
    [ -f "$RUN/$f" ] && aws s3 cp "$RUN/$f" "$OUT_S3/$(basename "$f")" >/dev/null 2>&1
  done
  aws s3 cp "$LOG" "$OUT_S3/sweep_$ARM.log" >/dev/null 2>&1
  if [ "$SHUTDOWN" = "1" ]; then
    echo "shutting down (instance-initiated-shutdown-behavior=terminate)"
    sudo shutdown -h +1
  fi
  return $rc
}
trap ship EXIT

# --- preflight -------------------------------------------------------------
# Bracketed pattern: an unbracketed pgrep -f matches this script's own command line and has
# twice reported phantom processes in this project.
pgrep -f "train_unifie[d]" >/dev/null && fail "a trainer is already on this GPU"
"$PY" -c "import torch; assert torch.cuda.is_available()" || fail "CUDA unavailable"
"$PY" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'cudnn', torch.backends.cudnn.version(), flush=True)
print('gpu', torch.cuda.get_device_name(0), flush=True)"
mountpoint -q /mnt/fast || fail "/mnt/fast is not mounted; epochs would be EBS-bound"

# --- data ------------------------------------------------------------------
# s5cmd when available, aws CLI otherwise. A freshly launched box has no s5cmd until user-data
# installs it, and a missing binary must degrade to a slower transfer rather than kill the arm.
pull() {  # pull <s3_prefix> <local_dir>
  mkdir -p "$2"
  if command -v s5cmd >/dev/null; then s5cmd cp "$1/*" "$2/"; else aws s3 sync "$1/" "$2/"; fi
}
if [ ! -f "$PREPARED/manifest.json" ]; then
  echo "--- pull prepared tensors $(date -u +%FT%TZ) ---"
  pull "$TENSORS_S3" "$PREPARED" || fail "tensor pull failed"
fi
if [ ! -d "$FS/weather_asof" ]; then
  echo "--- pull feature store $(date -u +%FT%TZ) ---"
  pull "$FS_S3" "$FS" || fail "feature store pull failed"
fi

# HARD GATE on the population, not on the path. s3://.../deep_learning/prepared_tensors/ held
# the VOIDED 1950-train set (157,150 train games) until 2026-08-31, and a sweep run against it
# would produce a complete, plausible, meaningless ranking. Assert the corrected count so that
# a wrong path or a stale local copy fails here instead of five hours later.
"$PY" - <<PYEOF || fail "prepared tensors are not the corrected 2026-08-31 population"
import json, sys
m = json.load(open("$PREPARED/manifest.json"))
t = m["splits"]["train"]
print("manifest prepared_at", m.get("prepared_at"),
      "train_games", t["n_games"], "train_samples", t["n_samples"])
if t["n_games"] != 21384 or t["n_samples"] != 315791:
    sys.exit(f"WRONG POPULATION: {t['n_games']} games / {t['n_samples']} samples; "
             "expected 21384 / 315791 (the 1950-train void set has 157150 games)")
PYEOF

avail=$(df --output=avail -k /mnt/fast | tail -1)
[ "$avail" -gt 15000000 ] || fail "/mnt/fast under 15GB free ($avail KB)"
echo "preflight ok: $(( avail / 1024 / 1024 ))GB free on /mnt/fast"

# --- train -----------------------------------------------------------------
mkdir -p "$RUN"
echo "--- train $(date -u +%FT%TZ) ---"
( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.train_unified fit-unified \
    --feature-store "$FS" --prepared-dir "$PREPARED" --output "$RUN" \
    --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size "$BATCH" --learning-rate "$LR" \
    --phase1-epochs "$P1" --phase2-epochs 0 --phase3-epochs 0 \
    --patience 10 --num-workers "$WORKERS" --seed "$SEED" )
rc=$?
echo "train exit=$rc $(date -u +%FT%TZ)"

# Report the arm's own curve here so a per-arm verdict survives even if the collector never
# runs. best_val_loss is recorded PER PHASE with no top-level field, and phases 2/3 report
# None at 0 epochs, so phase 1 is read explicitly rather than taking the last entry.
"$PY" - <<PYEOF
import json
from pathlib import Path
h = Path("$RUN/training_history.json")
if not h.exists():
    print("no training_history.json"); raise SystemExit
d = json.loads(h.read_text())
p1 = next((p for p in d.get("phases", []) if p.get("phase") == 1), None)
if p1 is None:
    print("no phase-1 record"); raise SystemExit
hist = p1.get("history", [])
print(f"arm $ARM d_model=$D_MODEL n_layers=$N_LAYERS params={d.get('n_parameters')}")
print(f"arm $ARM best_val={p1['best_val_loss']} over {p1['epochs_trained']} epochs")
for e in hist:
    print(f"  e{e.get('epoch')} train={e.get('train_loss')} val={e.get('val_loss')}")
best_ep = min((e["epoch"] for e in hist if e.get("val_loss") == p1["best_val_loss"]),
              default=None)
if best_ep is not None:
    print(f"arm $ARM best epoch = {best_ep}/{p1['epochs_trained']}")
    # An arm that still peaks in the first quarter of the schedule has not been shrunk enough
    # to stop overfitting, which is the whole point of the ladder.
    if best_ep <= 3 and p1.get("epochs_trained", 0) >= 8:
        print(f"arm $ARM STILL OVERFITS at this capacity")
PYEOF

echo "=== arm $ARM done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
