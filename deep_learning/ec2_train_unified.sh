#!/bin/bash
# EC2 training script for the unified GameTransformer model.
#
# Prerequisites:
#   - g5.xlarge (A10G 24GB VRAM) or g5.2xlarge (A10G 24GB + more RAM)
#   - AMI: Deep Learning AMI (Ubuntu) with CUDA 12.x + conda
#   - IAM role: read-write-mlb-s3
#   - Security group: sg-0583a4de608a95a41 (SSH only)
#
# Usage:
#   1. Launch EC2 instance (see below for CLI command)
#   2. rsync this repo to it
#   3. SSH in and run: bash deep_learning/ec2_train_unified.sh
#
# Launch command (run from local):
#   aws ec2 run-instances \
#     --image-id ami-0f47531f8c49bd1c6 \
#     --instance-type g5.xlarge \
#     --key-name awstest \
#     --security-group-ids sg-0583a4de608a95a41 \
#     --subnet-id subnet-013362590293de96a \
#     --iam-instance-profile Name=read-write-mlb-s3 \
#     --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
#     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mlb-dl-training}]'
#
# rsync command (run from local, replace HOSTNAME):
#   rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
#     -e "ssh -i ~/Documents/SENSITIVE/awstest.pem" \
#     ~/Projects/prediction_markets/mlb/ ec2-user@HOSTNAME:~/mlb/

set -euo pipefail

REPO_DIR="$HOME/mlb"
FEATURE_STORE="$REPO_DIR/deep_learning/artifacts/feature_store"
OUTPUT_DIR="$REPO_DIR/deep_learning/artifacts/unified_training"
S3_DATA="s3://mlb-265753586044-us-east-1-an/data"

echo "=== Unified GameTransformer Training Pipeline ==="
echo "Started: $(date)"
echo ""

# --- Step 1: Environment setup ---
echo "[1/6] Setting up environment..."
if ! command -v conda &> /dev/null; then
    echo "Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^pred "; then
    echo "Creating pred environment..."
    conda create -n pred python=3.11 -y
fi

conda activate pred

# Install PyTorch with CUDA (check driver version)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# Install dependencies
pip install -q -r "$REPO_DIR/deep_learning/requirements-deep-learning.txt" 2>/dev/null || true

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# --- Step 2: Build feature store ---
echo "[2/6] Building feature store from S3..."
if [ ! -f "$FEATURE_STORE/pitch_sequences.parquet" ] || \
   [ $(find "$FEATURE_STORE" -name "*.parquet" -mtime +7 | wc -l) -gt 0 ]; then
    echo "  Feature store missing or stale (>7 days). Rebuilding..."
    cd "$REPO_DIR"
    PYTHONPATH=deep_learning python -m mlb_dl.train build-feature-store \
        --source "$S3_DATA" \
        --output "$FEATURE_STORE" \
        --season-start 2015
    echo "  Feature store built."
else
    echo "  Feature store exists and is fresh. Skipping rebuild."
fi
echo ""

# --- Step 3: Report data statistics ---
echo "[3/6] Data statistics..."
cd "$REPO_DIR"
PYTHONPATH=deep_learning python -c "
import pandas as pd
from pathlib import Path

fs = Path('$FEATURE_STORE')
for name in ['pitch_sequences', 'game_targets', 'game_meta', 'team_games', 'player_batting_history']:
    p = fs / f'{name}.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        print(f'  {name}: {len(df):,} rows, {len(df.columns)} cols')
    else:
        print(f'  {name}: MISSING')
"
echo ""

# --- Step 4: Learning curves experiment ---
echo "[4/6] Running learning curve experiment..."
mkdir -p "$OUTPUT_DIR/learning_curves"
cd "$REPO_DIR"
PYTHONPATH=deep_learning python -m mlb_dl.train_unified learning-curves \
    --feature-store "$FEATURE_STORE" \
    --output "$OUTPUT_DIR/learning_curves" \
    --fractions 0.10 0.25 0.50 0.75 1.0 \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --patience 8
echo ""

# --- Step 5: Full phased training ---
echo "[5/6] Full phased training..."
mkdir -p "$OUTPUT_DIR/full_train"
cd "$REPO_DIR"
PYTHONPATH=deep_learning python -m mlb_dl.train_unified fit-unified \
    --feature-store "$FEATURE_STORE" \
    --output "$OUTPUT_DIR/full_train" \
    --d-model 256 \
    --n-layers 6 \
    --n-heads 8 \
    --dropout 0.1 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --phase1-epochs 50 \
    --phase2-epochs 30 \
    --phase3-epochs 20 \
    --patience 10
echo ""

# --- Step 6: Results summary ---
echo "[6/6] Results summary..."
echo ""
echo "=== Learning Curve Diagnosis ==="
cat "$OUTPUT_DIR/learning_curves/learning_curves.json" | python -c "
import sys, json
data = json.load(sys.stdin)
diag = data.get('diagnosis', {})
for k, v in diag.items():
    print(f'  {k}: {v}')
print()
print('Per-fraction results:')
for r in data.get('fractions', []):
    print(f\"  {r['fraction']*100:.0f}%: n={r['n_samples']:,}  val={r['best_val_loss']:.4f}  gap={r['gap_at_best']:.4f}\")
"
echo ""
echo "=== Full Training Test Metrics ==="
cat "$OUTPUT_DIR/full_train/training_history.json" | python -c "
import sys, json
data = json.load(sys.stdin)
metrics = data.get('test_metrics', {})
for k, v in metrics.items():
    if isinstance(v, dict):
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')
    else:
        print(f'  {k}: {v}')
" 2>/dev/null || echo "  (test metrics not yet available)"

echo ""
echo "=== Done ==="
echo "Finished: $(date)"
echo "Artifacts in: $OUTPUT_DIR"
echo ""
echo "To copy results locally:"
echo "  scp -i ~/Documents/SENSITIVE/awstest.pem ec2-user@HOSTNAME:~/mlb/deep_learning/artifacts/unified_training/ ./dl_results/ -r"
