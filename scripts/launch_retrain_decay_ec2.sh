#!/bin/bash
# Launch exponential decay retraining experiment on EC2.
#
# Sequence:
# 1. Append 26 new interpretability features to S3 parquet (if not already done)
# 2. Retrain LightGBM with exponential decay weighting
# 3. Compare with/without new features on held-out 2026 data
#
# Usage:
#   bash scripts/launch_retrain_decay_ec2.sh
#
# Prerequisites:
#   - EC2 instance running with python3.11, lightgbm, shap, boto3 installed
#   - SSH config has "mlb-ec2" host alias
#   - This repo cloned on EC2 at ~/prediction_markets/mlb

set -e

EC2_HOST="mlb-ec2"
REMOTE_DIR="~/prediction_markets/mlb"

echo "=== Syncing scripts to EC2 ==="
scp scripts/append_interpretability_features.py "${EC2_HOST}:${REMOTE_DIR}/scripts/"
scp scripts/retrain_with_decay_ec2.py "${EC2_HOST}:${REMOTE_DIR}/scripts/"

echo ""
echo "=== Step 1: Append new features to S3 parquet ==="
ssh "${EC2_HOST}" "cd ${REMOTE_DIR} && python3.11 scripts/append_interpretability_features.py"

echo ""
echo "=== Step 2: Retrain with decay — home_win (primary target) ==="
ssh "${EC2_HOST}" "cd ${REMOTE_DIR} && python3.11 scripts/retrain_with_decay_ec2.py --target home_win --lambda-decay 0.002"

echo ""
echo "=== Step 3: Retrain with decay — yrfi (temporal drift target) ==="
ssh "${EC2_HOST}" "cd ${REMOTE_DIR} && python3.11 scripts/retrain_with_decay_ec2.py --target yrfi --lambda-decay 0.002"

echo ""
echo "=== DONE ==="
echo "Results uploaded to s3://mlb-265753586044-us-east-1-an/classical_learning/artifacts/retrain_decay/"
