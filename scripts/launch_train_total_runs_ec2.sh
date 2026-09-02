#!/bin/bash
# Launch a single c8g.16xlarge to train all model families for total_runs.
#
# Prerequisites:
#   - sizing_curve_total_runs.json already updated (run launch_sizing_total_runs_ec2.sh first)
#   - Code tarball uploaded: s3://BUCKET/artifacts/code/mlb_train_total_runs.tar.gz
#
# Outputs: s3://BUCKET/artifacts/models/total_runs/
#   - oof_{target}_{family}_A.npy          OOF predictions per family
#   - params_{target}_{family}_A.json      Best Optuna hyperparameters
#   - model_{target}_{family}_A.pkl        Trained model (last LOYO fold)
#   - train_summary_{target}.json          Per-family status/metrics
#
# Usage:
#   bash scripts/launch_train_total_runs_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_train_total_runs.tar.gz"
FEATURES_KEY="artifacts/features/game_features.parquet"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c8g.16xlarge"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TARGET="total_runs"

USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/train_${TARGET}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: training ${TARGET} at \$(date -u) ==="

# Install deps
dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy \
  lightgbm xgboost catboost ydf optuna tqdm

# Pull code (includes data/importance/total_runs/ MDA artifacts)
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb --strip-components=1
cd /home/ec2-user/mlb

# Pull feature store
mkdir -p pregame/artifacts/features
aws s3 cp s3://${BUCKET}/${FEATURES_KEY} pregame/artifacts/features/game_features.parquet
echo "Features downloaded: \$(du -sh pregame/artifacts/features/game_features.parquet)"

# Pull fresh sizing curve (may have just been recomputed by sizing job)
mkdir -p pregame/artifacts/sizing
aws s3 cp s3://${BUCKET}/artifacts/sizing/sizing_curve_${TARGET}.json \
  pregame/artifacts/sizing/sizing_curve_${TARGET}.json
echo "Sizing curve loaded."

# Run training
mkdir -p pregame/artifacts/models

echo "=== Running LOYO training for ${TARGET} ==="
export PYTHONPATH=/home/ec2-user/mlb
python3.12 -m pregame.cli train \
  --target ${TARGET} \
  --features pregame/artifacts/features/game_features.parquet \
  --output pregame/artifacts/models \
  --n-trials 100 \
  --tier A
EXIT_CODE=\$?

echo "=== Training done: exit_code=\${EXIT_CODE} at \$(date -u) ==="

# Upload all model artifacts
aws s3 sync pregame/artifacts/models/ s3://${BUCKET}/artifacts/models/ \
  --exclude "*" --include "*${TARGET}*"
echo "Artifacts uploaded to s3://${BUCKET}/artifacts/models/"

# Log
aws s3 cp "\$LOG_FILE" s3://${BUCKET}/artifacts/models/train_${TARGET}.log

# Summarize result
SUMMARY_FILE="pregame/artifacts/models/training_summary_${TARGET}_A.json"
if [ -f "\$SUMMARY_FILE" ]; then
  echo "=== Training summary ==="
  python3.12 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for fam, r in sorted(d.items()):
    status = r.get('status', 'unknown')
    agg = r.get('aggregate_metrics', {})
    mae = agg.get('mae', 'N/A')
    print(f'  {fam:<25} {status:<12} mae={mae}')
" "\$SUMMARY_FILE"
fi

echo "=== COMPLETE at \$(date -u) ==="
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=train_${TARGET}},{Key=Purpose,Value=train_mda_v1}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched training for $TARGET → $INSTANCE_ID"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].State.Name' --output text"
echo "  aws s3 cp s3://${BUCKET}/artifacts/models/train_${TARGET}.log /tmp/ && tail -50 /tmp/train_${TARGET}.log"
echo "Artifacts:"
echo "  aws s3 ls s3://${BUCKET}/artifacts/models/ | grep ${TARGET}"
