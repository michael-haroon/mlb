#!/bin/bash
# Runs ON the GPU box (g5.2xlarge, i-05b5114c32744b47b) after
# data_curation/scripts/finalize_weather_asof.sh reports FINALIZE COMPLETE.
#
# Syncs the as-of weather artifact, bakes it into the existing prepared tensors, then runs
# the two A/B arms and prints both minima against the measured noise floor.
#
# Usage:  nohup bash deep_learning/ec2_weather_ab.sh >/dev/null 2>&1 &
# Log:    ~/weather_ab.log
#
# WHY BOTH ARMS ARE RERUN rather than comparing against the existing control's 4.95209:
# nothing in the trainer was seeded until this change, and three phase-1 runs at the same
# nominal config landed at best_val 4.9330, 5.0289 and 4.9521. The spread between
# supposedly identical runs is a few hundredths of a nat -- the same magnitude as the
# effect the A/B has to detect. Worse, the best of those unseeded runs (4.9330) BEATS the
# nominal control (4.95209), so scoring a treatment against 4.95209 alone could manufacture
# an improvement that an existing no-weather run already exceeded. The arms are therefore a
# paired comparison at a shared --seed, differing only by --no-asof-weather.
#
# --no-asof-weather is what makes it one variable: the append patches each split's
# manifest.json in place, so after it runs there is no directory left that serves the
# legacy geometry. Without the flag, control and treatment would differ in the weather
# channels AND in which prepared directory each read.
set -uo pipefail

LOG=/home/ec2-user/weather_ab.log
REPO=/home/ec2-user/mlb
PY=$HOME/miniconda3/envs/pred/bin/python
FS=/mnt/fast/feature_store
PREPARED=/mnt/fast/prepared_tensors
# Checkpoints go to instance store, not root: the root volume runs ~89% full and two runs
# of a 21M-param model would crowd it. Only the small artifacts are copied back at the end.
RUNS=/mnt/fast/ab_runs
KEEP=/home/ec2-user/output/weather_ab
S3FS=s3://mlb-265753586044-us-east-1-an/deep_learning/feature_store

# Architecture read from the control checkpoint's own state_dict, not guessed:
# d_model 384 / n_layers 6 / n_heads 8 / 21,178,100 params.
SEED=${SEED:-42}
D_MODEL=${D_MODEL:-384}
N_LAYERS=${N_LAYERS:-6}
N_HEADS=${N_HEADS:-8}
BATCH=${BATCH:-64}
LR=${LR:-4e-04}
P1=${P1:-12}
WORKERS=${WORKERS:-4}

exec >>"$LOG" 2>&1
echo "=== weather A/B start $(date -u +%FT%TZ) seed=$SEED ==="
fail() { echo "ABORT: $*"; exit 1; }

cd "$REPO" || fail "no repo at $REPO"
mkdir -p "$RUNS" "$KEEP"

# --- Preflight -------------------------------------------------------------
# pgrep patterns are bracketed: an unbracketed pattern matches this script's own
# command line and has twice reported phantom processes during this project.
if pgrep -f "train_unifie[d]" >/dev/null; then
  fail "a training process is already on the GPU; refusing to contend for VRAM"
fi
[ -d "$PREPARED/train" ] || fail "no prepared tensors at $PREPARED"
"$PY" -c "import torch; assert torch.cuda.is_available()" || fail "CUDA unavailable"

avail=$(df --output=avail -k /mnt/fast | tail -1)
[ "$avail" -gt 20000000 ] || fail "/mnt/fast under 20GB free ($avail KB); checkpoints will not fit"
echo "preflight ok: GPU idle, $(( avail / 1024 / 1024 ))GB free on /mnt/fast"

# --- Stage 1: sync the artifact -------------------------------------------
# The sidecar is fetched FIRST and its absence is fatal here rather than inside the append:
# _load_weather_asof_artifacts falls back to raw physical units with only a warning, and
# the append would bake those in permanently, making the treatment arm train unnormalized
# weather. That failure reads as "weather did not help" instead of as an error.
echo "--- sync artifact $(date -u +%FT%TZ) ---"
aws s3 cp "$S3FS/weather_asof_norm.json" "$FS/weather_asof_norm.json" \
  || fail "no weather_asof_norm.json in S3 — run finalize_weather_asof.sh first"
SYNC=$(command -v s5cmd || echo "")
if [ -n "$SYNC" ]; then
  s5cmd sync "$S3FS/weather_asof/*" "$FS/weather_asof/" || fail "weather_asof sync failed"
  s5cmd sync "$S3FS/wx_hour_offset/*" "$FS/wx_hour_offset/" || fail "wx_hour_offset sync failed"
else
  aws s3 sync "$S3FS/weather_asof/" "$FS/weather_asof/" || fail "weather_asof sync failed"
  aws s3 sync "$S3FS/wx_hour_offset/" "$FS/wx_hour_offset/" || fail "wx_hour_offset sync failed"
fi

# Verify the transfer rather than trusting the exit code: a truncated parquet still exits 0
# on some paths, and a short season would silently reduce as-of coverage.
"$PY" - <<PYEOF || fail "synced artifact failed verification"
import json, sys
from pathlib import Path
import pandas as pd
fs = Path("$FS")
seasons = sorted(fs.glob("weather_asof/season=*.parquet"))
offs = sorted(fs.glob("wx_hour_offset/season=*.parquet"))
if not seasons:
    sys.exit("no weather_asof parquets landed")
norm = json.loads((fs / "weather_asof_norm.json").read_text())
print(f"norm sidecar fit on seasons {norm.get('seasons')} train_end={norm.get('train_end_date')}")
total = 0
for p in seasons:
    df = pd.read_parquet(p, columns=["game_pk", "decision_hour", "target_hour"])
    n_games = df["game_pk"].nunique()
    if len(df) != n_games * 49:
        sys.exit(f"{p.name}: {len(df)} rows for {n_games} games, expected {n_games * 49} "
                 f"(7 decisions x 7 target hours) — file is truncated")
    total += n_games
print(f"verified {len(seasons)} season parquets, {total} games, {len(offs)} offset files")
PYEOF
echo "sync verified"

# --- Stage 2: bake the weather into the prepared tensors ------------------
# Additive and idempotent: writes weather_asof.npy + wx_decision_hour.npy per split and
# patches the manifest. Its own gates (>=95% coverage, <=1% offset truncation) are the ones
# that catch a stale pitches snapshot, so a nonzero exit here must stop the run.
echo "--- append to prepared tensors $(date -u +%FT%TZ) ---"
( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.append_weather_asof_to_prepared \
    --feature-store "$FS" --prepared-dir "$PREPARED" ) \
  || fail "append failed (coverage or offset-truncation gate, or missing sidecar)"

# Confirm the append produced what the model geometry will be built from, and that the
# control arm's override still yields the legacy geometry on this same directory.
"$PY" - <<PYEOF || fail "post-append verification failed"
import sys
sys.path.insert(0, "$REPO/deep_learning")
from mlb_dl.precollate import PreparedDataset
from mlb_dl.train_unified import _resolve_weather_geometry
from mlb_dl.weather_asof import ASOF_CHANNELS, N_TARGET_HOURS
for split in ("train", "val", "test"):
    d = f"$PREPARED/{split}"
    on = PreparedDataset(d)
    if not on._has_weather_asof:
        sys.exit(f"{split}: append ran but the dataset does not serve as-of weather")
    cfg, active = _resolve_weather_geometry(on, use_prepared=True)
    if not active or (cfg.weather_tokens, cfg.weather_dim) != (N_TARGET_HOURS, ASOF_CHANNELS):
        sys.exit(f"{split}: treatment geometry is {cfg.weather_tokens}x{cfg.weather_dim}")
    off = PreparedDataset(d, disable_asof=True)
    _, active_off = _resolve_weather_geometry(off, use_prepared=True)
    if active_off:
        sys.exit(f"{split}: --no-asof-weather did not restore the legacy geometry, so the "
                 f"control arm would not be a control")
    print(f"{split}: treatment {N_TARGET_HOURS}x{ASOF_CHANNELS}, control legacy — both serve")
PYEOF
echo "append verified"

# --- Stage 3: the two arms, sequentially ---------------------------------
# Sequential, not parallel: one A10G. Two concurrent jobs would contend for VRAM and
# bandwidth, roughly doubling each arm's epoch time for no wall-clock gain, and would make
# the arms' throughput differ from each other.
run_arm() {
  local name=$1; shift
  echo "--- arm $name $(date -u +%FT%TZ) ---"
  mkdir -p "$RUNS/$name"
  ( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.train_unified fit-unified \
      --feature-store "$FS" --prepared-dir "$PREPARED" --output "$RUNS/$name" \
      --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" \
      --batch-size "$BATCH" --learning-rate "$LR" \
      --phase1-epochs "$P1" --phase2-epochs 0 --phase3-epochs 0 \
      --num-workers "$WORKERS" --seed "$SEED" "$@" )
  local rc=$?
  echo "arm $name exit=$rc $(date -u +%FT%TZ)"
  return $rc
}

# Control first: if the harness is broken, it fails on the cheaper-to-interpret arm.
run_arm control --no-asof-weather || fail "control arm failed"
run_arm treatment                 || fail "treatment arm failed"

# --- Stage 4: verdict ----------------------------------------------------
"$PY" - <<'PYEOF'
import json
from pathlib import Path

# Three unseeded phase-1 runs at this config. The A/B has to clear this spread, not just
# come out ahead, and 4.9330 is the number to beat because an existing no-weather run
# already reached it.
NOISE = [4.9330, 5.0289, 4.9521]
runs = Path("/mnt/fast/ab_runs")
out = {}
for arm in ("control", "treatment"):
    h = runs / arm / "training_history.json"
    if not h.exists():
        print(f"{arm}: no training_history.json")
        continue
    d = json.loads(h.read_text())
    # best_val_loss is recorded PER PHASE; there is no top-level field. Phases 2 and 3 are
    # requested with 0 epochs and report None, so read phase 1 explicitly rather than
    # taking the last entry.
    p1 = next((p for p in d.get("phases", []) if p.get("phase") == 1), None)
    if p1 is None:
        print(f"{arm}: no phase-1 record")
        continue
    out[arm] = p1["best_val_loss"]
    print(f"{arm}: phase1 best_val={out[arm]} over {p1['epochs_trained']} epochs")

print(f"\nunseeded noise floor: min={min(NOISE):.4f} max={max(NOISE):.4f} "
      f"spread={max(NOISE) - min(NOISE):.4f}")
if len(out) == 2 and None not in out.values():
    delta = out["control"] - out["treatment"]
    print(f"treatment - control = {-delta:+.4f} (positive delta = weather helped)")
    print(f"vs best unseeded no-weather run 4.9330: {4.9330 - out['treatment']:+.4f}")
    verdict = ("weather helps" if delta > (max(NOISE) - min(NOISE))
               else "INCONCLUSIVE: within the unseeded run-to-run spread")
    print(f"verdict: {verdict}")
    (runs / "ab_verdict.json").write_text(json.dumps(
        {**out, "delta": delta, "noise_spread": max(NOISE) - min(NOISE),
         "verdict": verdict}, indent=2))
PYEOF

# Small artifacts only, off the instance store, since stopping this box wipes /mnt/fast.
for arm in control treatment; do
  mkdir -p "$KEEP/$arm"
  cp "$RUNS/$arm/training_history.json" "$KEEP/$arm/" 2>/dev/null
  cp "$RUNS/$arm"/phase1/best.pt "$KEEP/$arm/" 2>/dev/null
done
cp "$RUNS/ab_verdict.json" "$KEEP/" 2>/dev/null
echo "=== weather A/B COMPLETE $(date -u +%FT%TZ); artifacts kept in $KEEP ==="
