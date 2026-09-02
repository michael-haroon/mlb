#!/bin/bash
# Launch a single c7i.8xlarge (x86_64) to retrain xgboost + ydf_oblique_gbt for total_runs.
#
# Prerequisite: 17 other OOF arrays already on S3 from the Graviton run.
# This instance downloads them, trains only the two failed families, then
# re-runs ensemble over all 19 and uploads everything.
#
# ydf requires x86_64 Linux wheels — will not install on ARM/Graviton.
# xgboost objective fix: reg:pseudohuberror → reg:squarederror (in models.py).
#
# Usage:
#   bash scripts/launch_train_xgb_ydf_x86.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_train_total_runs.tar.gz"
FEATURES_KEY="artifacts/features/game_features.parquet"
AMI="ami-0bdc7d025135d7b49"   # AL2023 x86_64
INSTANCE_TYPE="c7i.8xlarge"   # x86_64, 32 vCPU, 64 GB RAM
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TARGET="total_runs"

USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/train_xgb_ydf.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: xgboost + ydf_oblique_gbt for ${TARGET} at \$(date -u) ==="

# Install deps — ydf needs x86_64 manylinux wheel
dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy \
  lightgbm xgboost catboost ydf optuna tqdm
python3.12 -c "import ydf; print('ydf', ydf.__version__)"
python3.12 -c "import xgboost; print('xgboost', xgboost.__version__)"

# Pull code
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb --strip-components=1
chown -R ec2-user:ec2-user /home/ec2-user/mlb
cd /home/ec2-user/mlb
export PYTHONPATH=/home/ec2-user/mlb

# Pull features
mkdir -p pregame/artifacts/features
aws s3 cp s3://${BUCKET}/${FEATURES_KEY} pregame/artifacts/features/game_features.parquet

# Pull sizing curve
mkdir -p pregame/artifacts/sizing
aws s3 cp s3://${BUCKET}/artifacts/sizing/sizing_curve_${TARGET}.json \
  pregame/artifacts/sizing/sizing_curve_${TARGET}.json

# Pull the 17 completed OOF arrays + params from S3 so ensemble sees them all
mkdir -p pregame/artifacts/models
aws s3 sync s3://${BUCKET}/artifacts/models/ pregame/artifacts/models/ \
  --exclude "*" --include "*${TARGET}*"
echo "Downloaded existing OOF artifacts:"
ls pregame/artifacts/models/oof_${TARGET}_*.npy | wc -l

# Train only the two failed families
echo "=== Training xgboost ==="
python3.12 -m pregame.cli train \
  --target ${TARGET} \
  --features pregame/artifacts/features/game_features.parquet \
  --output pregame/artifacts/models \
  --families xgboost ydf_oblique_gbt \
  --n-trials 100 \
  --tier A
EXIT_CODE=\$?
echo "Training exit_code=\${EXIT_CODE}"

# Upload new OOF + params
aws s3 sync pregame/artifacts/models/ s3://${BUCKET}/artifacts/models/ \
  --exclude "*" --include "*${TARGET}*"

# Run ensemble over all 19 families
echo "=== Building ensemble ==="
python3.12 -m pregame.cli ensemble \
  --models pregame/artifacts/models \
  --features pregame/artifacts/features/game_features.parquet \
  --target ${TARGET} \
  --tier A

# Upload ensemble + summary
aws s3 sync pregame/artifacts/models/ s3://${BUCKET}/artifacts/models/ \
  --exclude "*" --include "*${TARGET}*"

# Print summary
SUMMARY="pregame/artifacts/models/training_summary_${TARGET}_A.json"
if [ -f "\$SUMMARY" ]; then
  python3.12 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for fam, r in sorted(d.items()):
    status = r.get('status','?')
    mae = r.get('aggregate_metrics',{}).get('mae','N/A')
    print(f'  {fam:<25} {status:<12} mae={mae}')
" "\$SUMMARY"
fi

# Upload log
aws s3 cp "\$LOG_FILE" s3://${BUCKET}/artifacts/models/train_xgb_ydf.log

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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=train_xgb_ydf_${TARGET}},{Key=Purpose,Value=train_mda_v1}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched x86 instance for xgboost+ydf → $INSTANCE_ID"
echo ""
echo "Monitor (wait ~5 min for bootstrap):"
echo "  ssh -i /Users/michaelharoon/Documents/SENSITIVE/awstest.pem ec2-user@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].PublicIpAddress' --output text) 'tail -f /var/log/train_xgb_ydf.log'"
echo ""
echo "Or poll S3 log:"
echo "  aws s3 cp s3://${BUCKET}/artifacts/models/train_xgb_ydf.log /tmp/ && tail -30 /tmp/train_xgb_ydf.log"
