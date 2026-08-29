#!/bin/bash
# EC2 user-data: build weather_temporal.parquet and upload to S3.
# Instance should use: AMI ami-0f47531f8c49bd1c6 (AL2023 ARM64),
# SG sg-0583a4de608a95a41, IAM profile read-write-mlb-s3, key awstest.
# Instance terminates itself on completion.
set -euo pipefail

LOGFILE="/tmp/build_weather_temporal.log"
exec > >(tee -a "$LOGFILE") 2>&1

S3_BUCKET="mlb-265753586044-us-east-1-an"
FS_S3="s3://${S3_BUCKET}/ec2/feature_store/deep_learning/artifacts/feature_store"
CODE_S3="s3://${S3_BUCKET}/deep_learning/code/mlb_dl_append_v2.tar.gz"

echo "=== START: Build weather_temporal at $(date -u) ==="

# Install python3.12
yum install -y python3.12 python3.12-pip

# Install Python deps
python3.12 -m pip install --quiet pandas pyarrow numpy boto3

# Download and extract code archive
mkdir -p /home/ec2-user/mlb
cd /home/ec2-user/mlb
aws s3 cp "$CODE_S3" /tmp/mlb_dl_append_v2.tar.gz
tar xzf /tmp/mlb_dl_append_v2.tar.gz

# Download game_meta.parquet
echo "=== Downloading game_meta.parquet ==="
mkdir -p feature_store
aws s3 cp "${FS_S3}/game_meta.parquet" feature_store/game_meta.parquet
echo "Downloaded $(stat -c%s feature_store/game_meta.parquet) bytes"

# Validate game_meta
python3.12 - <<'PYEOF'
import pandas as pd
gm = pd.read_parquet("feature_store/game_meta.parquet")
print(f"game_meta: {len(gm)} games, {gm.shape[1]} columns")
print(f"Date range: {gm['game_datetime_utc'].min()} → {gm['game_datetime_utc'].max()}")
required = ["game_pk", "venue_id", "game_datetime_utc"]
missing = [c for c in required if c not in gm.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")
print("All required columns present ✓")
PYEOF

# Build weather_temporal
echo "=== Building weather_temporal ==="
python3.12 -m deep_learning.mlb_dl.append_new_features \
    --feature-store feature_store \
    --source-uri "s3://${S3_BUCKET}/data" \
    --artifacts weather_temporal

# Validate output
python3.12 - <<'PYEOF'
import pandas as pd, sys
wt = pd.read_parquet("feature_store/weather_temporal.parquet")
print(f"weather_temporal: {len(wt)} rows, {wt.shape[1]} cols")
print(f"Columns: {list(wt.columns)}")
if "game_pk" not in wt.columns or "hour_offset" not in wt.columns:
    raise SystemExit("Missing required columns game_pk/hour_offset")
games = wt["game_pk"].nunique()
hours = wt["hour_offset"].nunique()
print(f"Unique games: {games}, unique hour_offsets: {hours} (expected 4)")
null_rates = wt.drop(columns=["game_pk","hour_offset"]).isnull().mean()
print(f"Feature null rates (top 5 worst):\n{null_rates.sort_values(ascending=False).head(5)}")
print("=== VALIDATION PASSED ===")
PYEOF

# Upload to S3
echo "=== Uploading to S3 ==="
aws s3 cp feature_store/weather_temporal.parquet "${FS_S3}/weather_temporal.parquet"
echo "  Uploaded weather_temporal.parquet"

# Upload log
aws s3 cp "$LOGFILE" "s3://${S3_BUCKET}/deep_learning/build_weather_temporal.log"

echo "=== DONE at $(date -u) ==="

# Self-terminate (IMDSv2)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1
