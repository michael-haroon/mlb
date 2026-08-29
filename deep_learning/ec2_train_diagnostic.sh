#!/bin/bash
# Diagnostic training protocol for the unified GameTransformer.
#
# Implements the empirical scaling protocol:
#   Step 1: Baseline (2-layer, 128-dim, ~4.3M params)
#   Step 2: Medium  (4-layer, 256-dim, ~8.6M params)
#   Step 3: Diagnose (compare val Brier, determine bottleneck)
#   Step 4: Large (6-layer, 384-dim, ~18.7M) — only if medium beats baseline
#   Step 5: Depth probe — best width, vary depth 2/4/6/8
#
# Note: param counts are higher than a vanilla transformer because the model
# includes 50K-bucket hash embeddings (3 player types), PerceiverResampler,
# and a full PlayerQueryHead with multinomial output structure. Even the
# smallest config carries ~3M in embeddings/heads. The backbone accounts
# for the scaling difference between sizes.
#
# Date range: 2015+ (clean Statcast). Weather temporal zeros for 2015-2016 games
# is acceptable — the model must learn that zeros can mean "missing."
#
# Instance: g5.2xlarge ($1.20/hr, 1x A10G 24GB VRAM)
# Expected runtime: 12-18 hours for full protocol (~$15-22)
#
# Usage:
#   1. Launch EC2 g5.2xlarge
#   2. rsync repo to instance
#   3. bash deep_learning/ec2_train_diagnostic.sh
#
# Resume: safe to re-run — skips steps whose output directories already exist.

set -euo pipefail

REPO_DIR="$HOME/mlb"
FEATURE_STORE="$REPO_DIR/deep_learning/artifacts/feature_store"
OUTPUT_BASE="$REPO_DIR/deep_learning/artifacts/diagnostic_training"
S3_DATA="s3://mlb-265753586044-us-east-1-an/data"
S3_DL="s3://mlb-265753586044-us-east-1-an/deep_learning/feature_store"
NUM_WORKERS=4
DATASET_CACHE="$REPO_DIR/deep_learning/precomputed_datasets"
PREPARED_DIR="$REPO_DIR/deep_learning/prepared_tensors"
DEBUG_LOG="$HOME/debug_training.log"
GPU_LOG="$HOME/gpu_utilization.log"

# --- Debug helper: timestamps + memory at every step ---
dbg() {
    local msg="$1"
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    local mem=$(free -m 2>/dev/null | awk '/Mem:/{printf "%d/%dMB (%.0f%%)", $3, $2, $3/$2*100}' || echo "n/a")
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%dMB/%dMB", $1, $2}' || echo "n/a")
    echo "[$ts] [RAM: $mem] [GPU: $gpu_mem] $msg" | tee -a "$DEBUG_LOG"
}

# Start fresh debug log
echo "=== DEBUG LOG STARTED $(date) ===" > "$DEBUG_LOG"
dbg "Script starting"

echo "================================================================="
echo " GameTransformer Diagnostic Training Protocol"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 0: Environment
# ---------------------------------------------------------------------------
dbg "STEP 0: Environment setup"
echo "[0/7] Environment setup..."

if ! command -v conda &> /dev/null; then
    echo "  Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^pred "; then
    conda create -n pred python=3.11 -y
fi
conda activate pred

# PyTorch + CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

pip install -q -r "$REPO_DIR/deep_learning/requirements-deep-learning.txt" 2>/dev/null || true

python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# --- GPU utilization monitoring ---
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi dmon -s u -d 5 -f "$GPU_LOG" &
    GPU_LOG_PID=$!
    echo "  GPU logging started (PID=$GPU_LOG_PID, every 5s → $GPU_LOG)"
fi

# ---------------------------------------------------------------------------
# Step 1: Feature store build (2015+)
# ---------------------------------------------------------------------------
# Reduce fragmentation on A10G — extra-inning games create variable-length
# batches that fragment the allocator without this flag.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dbg "STEP 1: Feature store check"
echo "[1/7] Building feature store (2015+)..."

mkdir -p "$FEATURE_STORE"

if [ ! -f "$FEATURE_STORE/pitch_sequences.parquet" ]; then
    cd "$REPO_DIR"
    PYTHONPATH=deep_learning python -m mlb_dl.train build-feature-store \
        --source "$S3_DATA" \
        --output "$FEATURE_STORE" \
        --season-start 2015
    echo "  Feature store built from scratch."
else
    echo "  Feature store exists. Skipping rebuild."
fi

# Weather temporal (separate S3 artifact, 2017-2026)
if [ ! -f "$FEATURE_STORE/weather_temporal.parquet" ]; then
    echo "  Downloading weather_temporal.parquet from S3..."
    aws s3 cp "$S3_DL/weather_temporal.parquet" "$FEATURE_STORE/weather_temporal.parquet"
fi

# Rating sequences (pre-built .npz)
if [ ! -f "$FEATURE_STORE/rating_sequences.npz" ]; then
    echo "  Building rating sequences..."
    cd "$REPO_DIR"
    PYTHONPATH=deep_learning python -m mlb_dl.rating_sequences build \
        --game-features "$S3_DATA/features/game_features.parquet" \
        --output "$FEATURE_STORE/rating_sequences.npz" \
        --train-end 2024-04-01
fi

echo ""

# Data report
dbg "STEP 2: Data statistics"
echo "[2/7] Data statistics..."
cd "$REPO_DIR"
PYTHONPATH=deep_learning python -c "
import pandas as pd
from pathlib import Path

fs = Path('$FEATURE_STORE')
for name in ['pitch_sequences', 'game_targets', 'game_meta', 'team_games',
             'player_batting_history', 'weather_temporal']:
    p = fs / f'{name}.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        date_col = 'game_date' if 'game_date' in df.columns else None
        date_range = ''
        if date_col:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                date_range = f' [{dates.min().date()} .. {dates.max().date()}]'
        print(f'  {name:30s} {len(df):>10,} rows  {len(df.columns):3d} cols{date_range}')
    else:
        print(f'  {name:30s} MISSING')

# Rating sequences
import numpy as np
npz = fs / 'rating_sequences.npz'
if npz.exists():
    data = np.load(npz)
    print(f'  {\"rating_sequences\":30s} {len(data.files)} arrays, {data[data.files[0]].shape}')
"
echo ""

# ---------------------------------------------------------------------------
# Step 3a: Pre-collate tensors (eliminates per-batch data assembly)
# ---------------------------------------------------------------------------
dbg "STEP 3a: Pre-collate tensors"
echo "[3a/7] Pre-collating training tensors..."

if [ ! -f "$PREPARED_DIR/manifest.json" ]; then
    # Check if dataset cache exists first
    if [ -f "$DATASET_CACHE/manifest.json" ]; then
        mkdir -p "$PREPARED_DIR"
        cd "$REPO_DIR"
        PYTHONPATH=deep_learning python -m mlb_dl.train_unified precollate \
            --dataset-cache "$DATASET_CACHE" \
            --output "$PREPARED_DIR" \
            --num-workers $NUM_WORKERS
        dbg "  Pre-collation complete"
        echo "  Prepared tensors saved to $PREPARED_DIR"
    else
        echo "  Dataset cache not found. Training will use cache or feature store directly."
    fi
else
    echo "  Pre-collated tensors already exist. Skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3b: Learning curves — SKIPPED
# samples_per_param_ratio=0.03 already confirms data-limited regime; running
# learning curves at 8% GPU utilization would cost $60+ and confirm what we
# already know. Go straight to baseline fit.
# ---------------------------------------------------------------------------
dbg "STEP 3: Learning curves (skipped)"
echo "[3/7] Learning curves: SKIPPED (data-limited regime already confirmed by stats)"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Baseline model (2-layer, 128-dim, ~500K params)
# ---------------------------------------------------------------------------
dbg "STEP 4: BASELINE model (batch=128, lr=1.2e-3, workers=$NUM_WORKERS)"
echo "[4/7] Step 1: BASELINE model (2-layer, d=128, ~4.3M params)..."
BASELINE_DIR="$OUTPUT_BASE/step1_baseline"

if [ ! -f "$BASELINE_DIR/training_history.json" ]; then
    mkdir -p "$BASELINE_DIR"
    cd "$REPO_DIR"
    dbg "  Launching baseline training..."
    PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
        --feature-store "$FEATURE_STORE" \
        --output "$BASELINE_DIR" \
        --dataset-cache "$DATASET_CACHE" \
        --prepared-dir "$PREPARED_DIR" \
        --d-model 128 \
        --n-layers 2 \
        --n-heads 4 \
        --dropout 0.1 \
        --batch-size 128 \
        --learning-rate 1.2e-3 \
        --weight-decay 0.01 \
        --phase1-epochs 40 \
        --phase2-epochs 20 \
        --phase3-epochs 15 \
        --patience 8 \
        --num-workers $NUM_WORKERS
    dbg "  Baseline training FINISHED (exit $?)"
    echo "  Baseline training complete."
else
    dbg "  Baseline already exists, skipping"
    echo "  Baseline already trained. Skipping."
fi

echo "  Baseline metrics:"
python -c "
import json
with open('$BASELINE_DIR/training_history.json') as f:
    data = json.load(f)
metrics = data.get('test_metrics', {})
for k, v in metrics.items():
    if not isinstance(v, dict):
        print(f'    {k}: {v}')
phases = data.get('phases', [])
for p in phases:
    print(f'    Phase {p[\"phase\"]}: best_val={p[\"best_val_loss\"]:.5f} epochs={p[\"epochs_trained\"]}')
" 2>/dev/null || echo "  (not available yet)"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Medium model (4-layer, 256-dim, ~4M params)
# ---------------------------------------------------------------------------
dbg "STEP 5: MEDIUM model (batch=64, lr=1.2e-3)"
echo "[5/7] Step 2: MEDIUM model (4-layer, d=256, ~8.6M params)..."
MEDIUM_DIR="$OUTPUT_BASE/step2_medium"

if [ ! -f "$MEDIUM_DIR/training_history.json" ]; then
    mkdir -p "$MEDIUM_DIR"
    cd "$REPO_DIR"
    dbg "  Launching medium training..."
    PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
        --feature-store "$FEATURE_STORE" \
        --output "$MEDIUM_DIR" \
        --dataset-cache "$DATASET_CACHE" \
        --prepared-dir "$PREPARED_DIR" \
        --d-model 256 \
        --n-layers 4 \
        --n-heads 8 \
        --dropout 0.1 \
        --batch-size 64 \
        --learning-rate 1.2e-3 \
        --weight-decay 0.01 \
        --phase1-epochs 50 \
        --phase2-epochs 30 \
        --phase3-epochs 20 \
        --patience 10 \
        --num-workers $NUM_WORKERS
    dbg "  Medium training FINISHED (exit $?)"
    echo "  Medium training complete."
else
    dbg "  Medium already exists, skipping"
    echo "  Medium already trained. Skipping."
fi

echo "  Medium metrics:"
python -c "
import json
with open('$MEDIUM_DIR/training_history.json') as f:
    data = json.load(f)
metrics = data.get('test_metrics', {})
for k, v in metrics.items():
    if not isinstance(v, dict):
        print(f'    {k}: {v}')
phases = data.get('phases', [])
for p in phases:
    print(f'    Phase {p[\"phase\"]}: best_val={p[\"best_val_loss\"]:.5f} epochs={p[\"epochs_trained\"]}')
" 2>/dev/null || echo "  (not available yet)"
echo ""

# ---------------------------------------------------------------------------
# Step 6: Diagnose and conditionally train Large
# ---------------------------------------------------------------------------
dbg "STEP 6: Diagnosis + conditional model"
echo "[6/7] Step 3: DIAGNOSIS (compare baseline vs medium)..."

DIAGNOSIS=$(python -c "
import json, sys

try:
    with open('$BASELINE_DIR/training_history.json') as f:
        base = json.load(f)
    with open('$MEDIUM_DIR/training_history.json') as f:
        med = json.load(f)
except FileNotFoundError:
    print('PENDING')
    sys.exit(0)

base_val = base['phases'][-1]['best_val_loss']
med_val = med['phases'][-1]['best_val_loss']

base_train = base['phases'][-1]['history'][-1]['train_loss'] if base['phases'][-1]['history'] else base_val
med_train = med['phases'][-1]['history'][-1]['train_loss'] if med['phases'][-1]['history'] else med_val

improvement = (base_val - med_val) / abs(base_val) * 100
base_gap = abs(base_train - base_val)
med_gap = abs(med_train - med_val)

print(f'IMPROVEMENT={improvement:.2f}')
print(f'BASE_VAL={base_val:.5f}')
print(f'MED_VAL={med_val:.5f}')
print(f'BASE_GAP={base_gap:.5f}')
print(f'MED_GAP={med_gap:.5f}')

# Diagnosis logic
if improvement > 3:
    # Medium is meaningfully better — try larger
    if med_gap > 0.1 * abs(med_val):
        print('VERDICT=OVERFIT_AT_MEDIUM')
        print('ACTION=regularize_before_scaling')
    else:
        print('VERDICT=MODEL_LIMITED')
        print('ACTION=try_large')
elif improvement > 0:
    # Marginal improvement — near saturation
    print('VERDICT=NEAR_SATURATION')
    print('ACTION=depth_probe')
else:
    # No improvement or worse — data-limited
    print('VERDICT=DATA_LIMITED')
    print('ACTION=stop_scaling')

# Brier comparison with classical baseline
base_test = base.get('test_metrics', {})
med_test = med.get('test_metrics', {})
classical_brier = 0.2733  # from calibration-metrics-post-fix memory

for name, test in [('baseline', base_test), ('medium', med_test)]:
    hw = test.get('home_win_brier', None)
    if hw:
        vs_cl = (classical_brier - hw) / classical_brier * 100
        print(f'{name.upper()}_VS_CLASSICAL={vs_cl:.2f}%')
" 2>/dev/null || echo "PENDING")

echo "  $DIAGNOSIS"
echo ""

# Extract verdict for conditional logic
VERDICT=$(echo "$DIAGNOSIS" | grep "^VERDICT=" | cut -d= -f2 || echo "PENDING")
ACTION=$(echo "$DIAGNOSIS" | grep "^ACTION=" | cut -d= -f2 || echo "pending")

if [ "$ACTION" = "try_large" ]; then
    echo "  --> Medium improved over baseline. Training LARGE model..."
    LARGE_DIR="$OUTPUT_BASE/step3_large"

    if [ ! -f "$LARGE_DIR/training_history.json" ]; then
        mkdir -p "$LARGE_DIR"
        cd "$REPO_DIR"
        PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
            --feature-store "$FEATURE_STORE" \
            --output "$LARGE_DIR" \
            --dataset-cache "$DATASET_CACHE" \
            --prepared-dir "$PREPARED_DIR" \
            --d-model 384 \
            --n-layers 6 \
            --n-heads 12 \
            --dropout 0.15 \
            --batch-size 32 \
            --learning-rate 4e-4 \
            --weight-decay 0.02 \
            --phase1-epochs 60 \
            --phase2-epochs 35 \
            --phase3-epochs 25 \
            --patience 12 \
            --num-workers $NUM_WORKERS
    fi
elif [ "$ACTION" = "depth_probe" ]; then
    echo "  --> Near saturation. Running depth probe at d=256..."
    # Depth probe: 6-layer and 8-layer at same width
    for DEPTH in 6 8; do
        DEPTH_DIR="$OUTPUT_BASE/depth_probe_${DEPTH}L"
        if [ ! -f "$DEPTH_DIR/training_history.json" ]; then
            mkdir -p "$DEPTH_DIR"
            cd "$REPO_DIR"
            PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
                --feature-store "$FEATURE_STORE" \
                --output "$DEPTH_DIR" \
                --dataset-cache "$DATASET_CACHE" \
                --prepared-dir "$PREPARED_DIR" \
                --d-model 256 \
                --n-layers $DEPTH \
                --n-heads 8 \
                --dropout 0.12 \
                --batch-size 64 \
                --learning-rate 1e-3 \
                --weight-decay 0.015 \
                --phase1-epochs 50 \
                --phase2-epochs 30 \
                --phase3-epochs 20 \
                --patience 10 \
                --num-workers $NUM_WORKERS
        fi
    done
elif [ "$ACTION" = "regularize_before_scaling" ]; then
    echo "  --> Overfitting at medium. Re-training with stronger regularization..."
    REG_DIR="$OUTPUT_BASE/step2_medium_regularized"
    if [ ! -f "$REG_DIR/training_history.json" ]; then
        mkdir -p "$REG_DIR"
        cd "$REPO_DIR"
        PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
            --feature-store "$FEATURE_STORE" \
            --output "$REG_DIR" \
            --dataset-cache "$DATASET_CACHE" \
            --prepared-dir "$PREPARED_DIR" \
            --d-model 256 \
            --n-layers 4 \
            --n-heads 8 \
            --dropout 0.2 \
            --batch-size 64 \
            --learning-rate 8e-4 \
            --weight-decay 0.05 \
            --phase1-epochs 50 \
            --phase2-epochs 30 \
            --phase3-epochs 20 \
            --patience 10 \
            --num-workers $NUM_WORKERS
    fi
else
    echo "  --> Data-limited or pending. No further scaling."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 7: Summary report
# ---------------------------------------------------------------------------
dbg "STEP 7: Final summary + S3 upload"
echo "[7/7] Final summary..."
echo ""

python -c "
import json
from pathlib import Path

output = Path('$OUTPUT_BASE')
classical_brier = 0.2733  # known baseline

print('='*70)
print(' DIAGNOSTIC TRAINING SUMMARY')
print('='*70)
print()
print(f'{\"Model\":20s} {\"Params\":>10s} {\"Val Loss\":>10s} {\"HW Brier\":>10s} {\"vs Classical\":>12s} {\"Gap\":>8s}')
print('-'*70)

configs = [
    ('step1_baseline',   '~4.3M'),
    ('step2_medium',     '~8.6M'),
    ('step3_large',      '~18.7M'),
    ('step2_medium_regularized', '~8.6M+reg'),
    ('depth_probe_6L',   '~12M'),
    ('depth_probe_8L',   '~16M'),
]

best_model = None
best_brier = 999

for dirname, params in configs:
    hist_file = output / dirname / 'training_history.json'
    if not hist_file.exists():
        continue
    with open(hist_file) as f:
        data = json.load(f)

    val_loss = data['phases'][-1]['best_val_loss']
    test = data.get('test_metrics', {})
    hw_brier = test.get('home_win_brier', None)

    gap = ''
    if data['phases'][-1]['history']:
        last = data['phases'][-1]['history'][-1]
        gap = f'{abs(last[\"train_loss\"] - last[\"val_loss\"]):.4f}'

    vs_cl = ''
    if hw_brier:
        pct = (classical_brier - hw_brier) / classical_brier * 100
        vs_cl = f'{pct:+.1f}%'
        if hw_brier < best_brier:
            best_brier = hw_brier
            best_model = dirname

    hw_str = f'{hw_brier:.5f}' if hw_brier else 'n/a'
    print(f'{dirname:20s} {params:>10s} {val_loss:>10.5f} {hw_str:>10s} {vs_cl:>12s} {gap:>8s}')

print()
if best_model:
    print(f'Best model: {best_model} (Brier={best_brier:.5f})')
    print(f'Classical baseline: {classical_brier:.5f}')
    delta = classical_brier - best_brier
    if delta > 0:
        print(f'DL beats classical by {delta:.5f} ({delta/classical_brier*100:.1f}%)')
    else:
        print(f'Classical still wins by {-delta:.5f} ({-delta/classical_brier*100:.1f}%)')
print()

# Training symptom diagnosis
print('Symptom Analysis:')
for dirname, params in configs:
    hist_file = output / dirname / 'training_history.json'
    if not hist_file.exists():
        continue
    with open(hist_file) as f:
        data = json.load(f)
    phases = data['phases']
    if not phases:
        continue
    last_phase = phases[-1]
    if not last_phase['history']:
        continue
    final = last_phase['history'][-1]
    train_l = final['train_loss']
    val_l = final['val_loss']
    gap = train_l - val_l

    if last_phase['epochs_trained'] < 5:
        symptom = 'CONVERGED_FAST (possibly too simple)'
    elif gap > 0.1 * abs(val_l):
        symptom = 'OVERFITTING (train << val, needs regularization)'
    elif gap < -0.05 * abs(val_l):
        symptom = 'UNDERFITTING (train > val, train longer or add capacity)'
    else:
        symptom = 'WELL_FIT (train ~= val)'

    print(f'  {dirname}: {symptom}')
    print(f'    train={train_l:.5f} val={val_l:.5f} gap={gap:.5f} epochs={last_phase[\"epochs_trained\"]}')

print()
print('='*70)
" 2>/dev/null || echo "  Results not yet available."

echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Artifacts in: $OUTPUT_BASE"
echo ""

# ---------------------------------------------------------------------------
# Upload results to S3
# ---------------------------------------------------------------------------
echo "Uploading results to S3..."
S3_OUTPUT="s3://mlb-265753586044-us-east-1-an/deep_learning/diagnostic_training"

aws s3 sync "$OUTPUT_BASE" "$S3_OUTPUT/" --exclude "*.pt" --quiet
echo "  Training histories and metrics uploaded."

# Upload model weights separately (large files)
find "$OUTPUT_BASE" -name "*.pt" | while read pt_file; do
    rel_path="${pt_file#$OUTPUT_BASE/}"
    aws s3 cp "$pt_file" "$S3_OUTPUT/$rel_path" --quiet
done
echo "  Model weights uploaded."

# Upload training log
aws s3 cp ~/training.log "$S3_OUTPUT/training.log" --quiet 2>/dev/null || true
echo "  Training log uploaded."

# Stop GPU logging and upload
if [ -n "${GPU_LOG_PID:-}" ]; then
    kill $GPU_LOG_PID 2>/dev/null || true
    aws s3 cp "$GPU_LOG" "$S3_OUTPUT/gpu_utilization.log" --quiet 2>/dev/null || true
    echo "  GPU utilization log uploaded."
fi

echo ""
echo "All artifacts saved to: $S3_OUTPUT"
echo "Next steps:"
echo "  1. Review the summary above"
echo "  2. Download results: aws s3 sync $S3_OUTPUT ./diagnostic_results/"
echo "  3. If DL beats classical: deploy best model for inference"
echo "  4. If not: analyze residuals, check feature engineering gaps"
echo ""
echo "Shutting down instance..."
sudo shutdown -h now
