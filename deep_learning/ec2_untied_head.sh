#!/bin/bash
# Runs ON the GPU box. Retrains the readout_fix configuration with the pregame team heads UNTIED
# from the live team heads, so the pregame readout is fitted by pregame rows only.
#
# WHY THIS RUNS. `ab619fd` fixed which *representation* a pregame row reads (per-row `ctx_pool`
# instead of last-token padding) and the readout_fix run then trained that path for the first time.
# It did not help: pregame home_win BSS across its 12 epochs was -0.0235, +0.0140, -0.0097, -0.0601,
# -0.0081, -0.0032, -0.0455, -0.0193, -0.0161, -0.0428, -0.0558, -0.0647 — one positive epoch, at a
# COLLAPSED pstd of 0.0878, and actively degrading after e6 as pstd rose to 0.1731.
#
# The reason is one layer up from the representation: a single `head_home_win` read both `ctx_pool`
# (a mean over ~148 context tokens) and `backbone_out[:, -1, :]` (one late-game pitch token). Live
# rows are 93.2% of samples and late-game labels are nearly determined, so that head is fitted to a
# distribution rewarding sharpness, then applied to a pregame input where the honest answer sits
# near the base rate.
#
# THE BOUND, stated before the run so the result cannot be talked up afterwards. On the readout_fix
# e6 checkpoint, over the same test split: the checkpoint's own head emits pregame home_win BSS
# **-0.0054**, while a linear probe refitted on that same FROZEN `ctx_pool` reaches **+0.0082**
# (test) / +0.0113 (pooled val+test). Same trunk, same tensor, different reader — so ~+0.014 is what
# untying can recover, and the probe is the ceiling, not a floor. Expected outcome is pregame moving
# from a small liability to roughly the ~+0.011 BSS that every pregame approach tested lands on.
# This run is a LIABILITY REMOVAL, not a source of new pregame skill; the features do not carry it.
# See [[project-dl-skill-by-prefix-2026-09-01]], [[project-pregame-probe-verdict-2026-08-31]].
#
# ONE VARIABLE. Every hyperparameter below is byte-identical to ec2_readout_fix_rerun.sh, so the
# only difference from that run is the untied heads. In particular the head ARCHITECTURE is
# duplicated exactly rather than resized — the probe found linear >= MLP on a frozen trunk
# (+0.0113 vs +0.0104), so shrinking the pregame head is a plausible NEXT lever, but bundling it in
# here would make this run's result unattributable.
#
# READ THE RESULT ON pregame/*, NEVER ON POOLED VAL. Pooled val is ~73% runs-remaining arithmetic,
# so a pregame change cannot move it ([[project-dl-pregame-collapse-2026-08-31]]).
# pregame/home_win_pstd is the collapse detector: 0.0866 was the collapsed value.
#
# NO SELF-SHUTDOWN, deliberately. /mnt/fast is INSTANCE STORE on i-05b5114c32744b47b; stopping the
# box wipes prepared_tensors and every checkpoint under /mnt/fast without an S3 copy. Reboot is
# safe; stop is not.
#
# Usage:
#   bash ec2_untied_head.sh
# Log: ~/untied_head.log
set -uo pipefail

TAG=${TAG:-untied_head}
RUN_ID=${RUN_ID:-20260901}
SEED=${SEED:-42}
D_MODEL=${D_MODEL:-384}
N_LAYERS=${N_LAYERS:-6}
# 8, matching ec2_readout_fix_rerun.sh:48 -- NOT the argparse default of 12. n_heads changes no
# parameter shape, so getting this wrong trains a different model silently.
N_HEADS=${N_HEADS:-8}
DROPOUT=${DROPOUT:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
BATCH=${BATCH:-64}
LR=${LR:-4e-04}
P1=${P1:-12}
PATIENCE=${PATIENCE:-10}
GRAD_CLIP=${GRAD_CLIP:-5.0}
WORKERS=${WORKERS:-4}

BUCKET=${BUCKET:-mlb-265753586044-us-east-1-an}
OUT_S3="s3://$BUCKET/deep_learning/experiments/untied_head_$RUN_ID/$TAG"

REPO=/home/ec2-user/mlb
PY=/home/ec2-user/miniconda3/envs/pred/bin/python
FS=/mnt/fast/feature_store
PREPARED=${PREPARED:-/mnt/fast/prepared_tensors}
RUN=/mnt/fast/$TAG
LOG=/home/ec2-user/$TAG.log

exec >>"$LOG" 2>&1
echo "=== $TAG start $(date -u +%FT%TZ) d_model=$D_MODEL n_layers=$N_LAYERS n_heads=$N_HEADS seed=$SEED ==="
fail() { echo "ABORT: $*"; exit 1; }

# Ship on EVERY exit path. The log is the only record of why a run failed, and on a box whose
# instance store can vanish an un-uploaded log is an unexplainable failure.
ship() {
  local rc=$?
  echo "=== $TAG exiting rc=$rc $(date -u +%FT%TZ) ==="
  aws s3 cp "$LOG" "$OUT_S3/$TAG.log" >/dev/null 2>&1
  for f in run_config.json training_history.json phase1/best.pt; do
    [ -f "$RUN/$f" ] && aws s3 cp "$RUN/$f" "$OUT_S3/$(basename "$f")" >/dev/null 2>&1
  done
  aws s3 cp "$LOG" "$OUT_S3/$TAG.log" >/dev/null 2>&1
  return $rc
}
trap ship EXIT

# --- preflight -------------------------------------------------------------
# Bracketed pattern: an unbracketed pgrep -f matches this script's own command line and has twice
# reported phantom processes in this project.
pgrep -f "train_unifie[d]" >/dev/null && fail "a trainer is already on this GPU"
pgrep -f "extract_pregame_rep[r]" >/dev/null && fail "the probe extractor is on this GPU"
mountpoint -q /mnt/fast || fail "/mnt/fast is not mounted; epochs would be EBS-bound"
[ -f "$PREPARED/manifest.json" ] || fail "no prepared tensors at $PREPARED"
[ -d "$FS/weather_asof" ] || fail "no feature store at $FS"
"$PY" -c "import torch; assert torch.cuda.is_available()" || fail "CUDA unavailable"

# THE GATE THAT DEFINES THIS RUN. Grepping for the new attribute only proves a string is present.
# The untied suite asserts the property that matters — that a pregame-only loss produces gradient
# in the pregame heads and NONE in the live heads, and vice versa. A 5h run that silently shares
# heads is the single worst outcome available here. The readout suite runs too: untying adds a
# per-row branch on top of a per-row representation choice, and a batch-level implementation of the
# new branch would reintroduce ab619fd one layer up.
echo "--- gate: untied-head + readout parity suites $(date -u +%FT%TZ) ---"
( cd "$REPO" && PYTHONPATH="$REPO/deep_learning" "$PY" -m pytest -q \
    deep_learning/tests/test_pregame_head_untied.py \
    deep_learning/tests/test_pregame_readout_invariance.py \
    deep_learning/tests/test_readout_serving_parity.py ) \
  || fail "untied/readout tests FAILED — this tree does not carry a working untying; do not train"
"$PY" - <<'PYEOF' || fail "the five pregame heads are not present in the model on this box"
import sys
sys.path.insert(0, "/home/ec2-user/mlb")
from deep_learning.mlb_dl.game_transformer import GameTransformer
m = GameTransformer(d_model=64, rating_dim=8, num_backbone_layers=2, num_heads=4)
need = ["head_home_win_pregame", "head_yrfi_pregame", "head_extra_innings_pregame",
        "head_negbin_home_pregame", "head_negbin_away_pregame"]
missing = [n for n in need if not hasattr(m, n)]
if missing:
    sys.exit(f"missing pregame heads: {missing}")
print("pregame heads present:", ", ".join(need))
PYEOF
grep -q "_pregame_metrics" "$REPO/deep_learning/mlb_dl/train_unified.py" \
  || fail "train_unified.py has no _pregame_metrics (a6df6f7) — the run would be unreadable"
grep -q "RUN_CONFIG_FILENAME" "$REPO/deep_learning/mlb_dl/train_unified.py" \
  || fail "train_unified.py does not persist run_config.json — re-rsync the tree"

# HARD GATE on the population, not the path. s3://.../deep_learning/prepared_tensors/ held the
# VOIDED 1950-train set (157,150 train games) until 2026-08-31; training against it would produce
# a complete, plausible, meaningless result five hours from now.
"$PY" - <<PYEOF || fail "prepared tensors are not the corrected 2026-08-31 population"
import json, sys
m = json.load(open("$PREPARED/manifest.json"))
t = m["splits"]["train"]
print("manifest prepared_at", m.get("prepared_at"), "rating_dim", m.get("rating_dim"),
      "has_weather_asof", m.get("has_weather_asof"))
for s, d in m["splits"].items():
    print(f"  {s}: {d['n_games']} games / {d['n_samples']} samples")
if t["n_games"] != 21384 or t["n_samples"] != 315791:
    sys.exit(f"WRONG POPULATION: {t['n_games']} games / {t['n_samples']} samples; "
             "expected 21384 / 315791 (the 1950-train void set has 157150 games)")
PYEOF

avail=$(df --output=avail -k /mnt/fast | tail -1)
[ "$avail" -gt 15000000 ] || fail "/mnt/fast under 15GB free ($avail KB)"
free -g | head -2
echo "preflight ok: $(( avail / 1024 / 1024 ))GB free on /mnt/fast"

# --- train -----------------------------------------------------------------
mkdir -p "$RUN"
echo "--- train $(date -u +%FT%TZ) ---"
( cd "$REPO/deep_learning" && "$PY" -m mlb_dl.train_unified fit-unified \
    --feature-store "$FS" --prepared-dir "$PREPARED" --output "$RUN" \
    --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" \
    --dropout "$DROPOUT" --weight-decay "$WEIGHT_DECAY" \
    --batch-size "$BATCH" --learning-rate "$LR" \
    --phase1-epochs "$P1" --phase2-epochs 0 --phase3-epochs 0 \
    --patience "$PATIENCE" --gradient-clip "$GRAD_CLIP" \
    --num-workers "$WORKERS" --seed "$SEED" --no-asof-weather )
rc=$?
echo "train exit=$rc $(date -u +%FT%TZ)"

# --- verdict ---------------------------------------------------------------
# Thresholds fixed BEFORE the run, and deliberately NOT a max over epochs: taking the best of 12
# epochs is a multiple-comparisons error that already fired once on this exact table (readout_fix
# e2's +0.0140 at a collapsed pstd of 0.0878 was reported as a partial success). Report the
# BEST-VAL epoch — the one a serving deployment would actually ship — plus the mean, and flag any
# epoch whose pstd is still at the collapsed value.
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
print(f"[$TAG] best pooled val={p1.get('best_val_loss')} over {p1.get('epochs_trained')} epochs "
      "(pooled val is ~73% runs-remaining arithmetic — NOT the number this run is about)")
print(f"[$TAG] {'ep':>3} {'train':>9} {'val':>9} {'hw_bss':>8} {'hw_pstd':>8} {'hw_brier':>9} "
      f"{'yrfi_bss':>9} {'xi_bss':>8} {'tr_mae':>8}")

# Pregame metrics live under the epoch record's "val_tasks", not at its top level
# (train_unified.py:1030) -- reading the top level silently yields None for every column.
def pg(e, k):
    return (e.get("val_tasks") or {}).get(f"pregame/{k}")

for e in hist:
    fmt = lambda v, w, p=4: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'--':>{w}}")
    print(f"[$TAG] {e.get('epoch'):>3} {fmt(e.get('train_loss'),9)} {fmt(e.get('val_loss'),9)} "
          f"{fmt(pg(e,'home_win_bss'),8)} {fmt(pg(e,'home_win_pstd'),8)} "
          f"{fmt(pg(e,'home_win_brier'),9,5)} {fmt(pg(e,'yrfi_bss'),9)} "
          f"{fmt(pg(e,'extra_innings_bss'),8)} {fmt(pg(e,'total_runs_mae'),8)}")

vals = [(e.get("epoch"), e.get("val_loss"), pg(e, "home_win_bss"), pg(e, "home_win_pstd"))
        for e in hist if isinstance(e.get("val_loss"), (int, float))]
bsss = [b for _, _, b, _ in vals if isinstance(b, (int, float))]
if not bsss:
    print("[$TAG] VERDICT: NO pregame metric in history — a6df6f7 did not take effect")
    raise SystemExit

ship_ep, _, ship_bss, ship_pstd = min(vals, key=lambda r: r[1])
mean_bss = sum(bsss) / len(bsss)
n_pos = sum(1 for b in bsss if b > 0)
print(f"[$TAG] shipping epoch (best pooled val) = e{ship_ep}: bss={ship_bss:+.4f} "
      f"pstd={ship_pstd if isinstance(ship_pstd,(int,float)) else float('nan'):.4f}")
print(f"[$TAG] across {len(bsss)} epochs: mean bss={mean_bss:+.4f}, "
      f"{n_pos} positive, max={max(bsss):+.4f}, min={min(bsss):+.4f}")

# The pre-registered read, on the SHIPPING epoch.
b, ps = ship_bss, (ship_pstd if isinstance(ship_pstd, (int, float)) else 0.0)
if b >= 0.008 and ps > 0.15:
    v = ("UNTYING WORKED — pregame reached the ~+0.011 probe ceiling with a spread head. "
         "Pregame is no longer a liability; it is NOT new skill. Next lever is the feature set, "
         "not the model.")
elif b >= 0.008:
    v = ("PARTIAL — bss reached the ceiling but pstd is still near collapse; the head is right "
         "on average and underconfident. Check calibration before serving.")
elif b > -0.002:
    v = ("NEUTRAL — pregame is no longer negative but did not reach the +0.0082 the frozen probe "
         "got on this same trunk. The head was not the only binding constraint.")
else:
    v = ("STILL NEGATIVE — untying did not recover the probe gap, so the shared head was NOT the "
         "cause. Do not try a third readout change; the trunk or the label mix is binding.")
print(f"[$TAG] VERDICT: {v}")
print(f"[$TAG] reference: frozen-probe ceiling +0.0082 test / +0.0113 pooled | "
      "readout_fix shared-head -0.0054 | collapsed pstd 0.0866 | market-grade ~+0.056")
PYEOF

echo "=== $TAG done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
