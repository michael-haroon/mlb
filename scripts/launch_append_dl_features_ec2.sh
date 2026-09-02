#!/bin/bash
# Append weather, venue, and daily stats features to the existing DL feature store.
#
# Lightweight job: downloads game_meta.parquet (17MB), reads weather+stats+venue
# from S3, writes 3 new parquets back. ~5 minutes, minimal RAM.
#
# Outputs added to: s3://BUCKET/ec2/feature_store/deep_learning/artifacts/feature_store/
#   - weather_features.parquet
#   - venue_dimensions.parquet
#   - daily_stats.parquet
#
# Usage:
#   bash scripts/launch_append_dl_features_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
FEATURE_STORE_PREFIX="ec2/feature_store/deep_learning/artifacts/feature_store"
AMI="ami-0f47531f8c49bd1c6"  # AL2023 ARM64
INSTANCE_TYPE="r8g.medium"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"

# --- Package code ---
echo "Packaging code..."
TARBALL="/tmp/mlb_dl_append.tar.gz"
tar -czf "$TARBALL" \
    deep_learning/mlb_dl/__init__.py \
    deep_learning/mlb_dl/data_sources.py \
    deep_learning/mlb_dl/feature_store.py \
    deep_learning/mlb_dl/targets.py \
    deep_learning/mlb_dl/append_new_features.py \
    deep_learning/__init__.py \
    2>/dev/null

aws s3 cp "$TARBALL" "s3://${BUCKET}/deep_learning/code/mlb_dl_append.tar.gz"
echo "Code uploaded to s3://${BUCKET}/deep_learning/code/mlb_dl_append.tar.gz"

USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/append_dl_features.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: Append DL features at $(date -u) ==="

BUCKET="mlb-265753586044-us-east-1-an"
FEATURE_STORE_PREFIX="ec2/feature_store/deep_learning/artifacts/feature_store"
WORK_DIR="/home/ec2-user/mlb"
FS_DIR="${WORK_DIR}/feature_store"

# Install deps
dnf install -y python3.12 python3.12-pip 2>/dev/null || yum install -y python3.12 python3.12-pip
python3.12 -m pip install --quiet pandas pyarrow numpy boto3 scipy

# Pull code
aws s3 cp "s3://${BUCKET}/deep_learning/code/mlb_dl_append.tar.gz" /tmp/mlb_dl_append.tar.gz
mkdir -p "$WORK_DIR"
tar -xzf /tmp/mlb_dl_append.tar.gz -C "$WORK_DIR"
cd "$WORK_DIR"
mkdir -p data/logs
export PYTHONPATH="$WORK_DIR"

# Download existing game_meta
mkdir -p "$FS_DIR"
echo ""
echo "=== Downloading game_meta.parquet ==="
aws s3 cp "s3://${BUCKET}/${FEATURE_STORE_PREFIX}/game_meta.parquet" "${FS_DIR}/game_meta.parquet"
echo "Downloaded $(stat -c%s "${FS_DIR}/game_meta.parquet" 2>/dev/null || stat -f%z "${FS_DIR}/game_meta.parquet") bytes"

# Pre-check: verify game_meta schema
echo ""
echo "=== Pre-check: game_meta schema ==="
python3.12 -c "
import pandas as pd
df = pd.read_parquet('${FS_DIR}/game_meta.parquet')
print(f'game_meta: {len(df)} games, {df.columns.size} columns')
print(f'Date range: {df[\"game_date\"].min()} → {df[\"game_date\"].max()}')
print(f'Venues: {df[\"venue_id\"].nunique()} unique')
needed = ['game_pk', 'venue_id', 'game_datetime_utc', 'game_date', 'probable_pitcher_home_id', 'probable_pitcher_away_id']
missing = [c for c in needed if c not in df.columns]
assert not missing, f'MISSING COLUMNS: {missing}'
print('All required columns present ✓')
"

# Run the append
echo ""
echo "=== Building new feature parquets ==="
python3.12 -m deep_learning.mlb_dl.append_new_features \
    --feature-store "$FS_DIR" \
    --source-uri "s3://${BUCKET}/data"

# Post-check: validate outputs
echo ""
echo "=== Post-check: validating outputs ==="
python3.12 -c "
import pandas as pd
import numpy as np

fs = '${FS_DIR}'

# Weather features
wx = pd.read_parquet(f'{fs}/weather_features.parquet')
print(f'weather_features: {len(wx)} rows, {wx.columns.size} cols')
print(f'  Non-null rates:')
for col in wx.columns:
    if col == 'game_pk': continue
    rate = wx[col].notna().mean()
    print(f'    {col}: {rate:.1%}')
assert 'game_pk' in wx.columns, 'Missing game_pk!'
assert len(wx) > 0, 'Empty weather_features!'

# Venue dimensions
vd = pd.read_parquet(f'{fs}/venue_dimensions.parquet')
print(f'\nvenue_dimensions: {len(vd)} rows, {vd.columns.size} cols')
print(f'  Columns: {vd.columns.tolist()}')
assert 'venue_id' in vd.columns
assert len(vd) > 0, 'Empty venue_dimensions!'

# Daily stats
ds = pd.read_parquet(f'{fs}/daily_stats.parquet')
print(f'\ndaily_stats: {len(ds)} rows, {ds.columns.size} cols')
print(f'  Columns: {ds.columns.tolist()}')
if len(ds) > 0:
    print(f'  Non-null rates:')
    for col in ds.columns:
        if col == 'game_pk': continue
        rate = ds[col].notna().mean()
        print(f'    {col}: {rate:.1%}')
    assert 'game_pk' in ds.columns, 'Missing game_pk!'

# Cross-check: game_meta coverage
gm = pd.read_parquet(f'{fs}/game_meta.parquet')
games_2015_plus = gm[gm['game_date'] >= '2015-01-01']
wx_coverage = wx['game_pk'].isin(games_2015_plus['game_pk']).sum() / len(games_2015_plus)
print(f'\nWeather coverage for 2015+ games: {wx_coverage:.1%}')

print('\n=== ALL CHECKS PASSED ===')
"

# Upload results
echo ""
echo "=== Uploading to S3 ==="
for artifact in weather_features venue_dimensions daily_stats; do
    if [ -f "${FS_DIR}/${artifact}.parquet" ]; then
        aws s3 cp "${FS_DIR}/${artifact}.parquet" "s3://${BUCKET}/${FEATURE_STORE_PREFIX}/${artifact}.parquet"
        echo "  Uploaded ${artifact}.parquet"
    fi
done

aws s3 cp "$LOG_FILE" "s3://${BUCKET}/deep_learning/append_dl_features.log"

echo ""
echo "=== COMPLETE at $(date -u) ==="
echo "Results at: s3://${BUCKET}/${FEATURE_STORE_PREFIX}/"

shutdown -h now
USERDATA_EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG" \
    --subnet-id "$SUBNET" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=dl_append_features},{Key=Purpose,Value=append_weather_venue_stats}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "Launched → $INSTANCE_ID"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].[State.Name,PublicIpAddress]' --output text"
echo "  # After it gets an IP:"
echo "  ssh -i ~/Documents/SENSITIVE/awstest.pem ec2-user@<ip> 'tail -50 /var/log/append_dl_features.log'"
echo ""
echo "Results (when done):"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/append_dl_features.log /tmp/ && cat /tmp/append_dl_features.log"
echo "  aws s3 ls s3://${BUCKET}/${FEATURE_STORE_PREFIX}/ --human-readable"
