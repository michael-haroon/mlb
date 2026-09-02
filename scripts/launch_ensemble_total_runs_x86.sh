#!/bin/bash
# Launch a c7i.4xlarge to build the final 19-family ensemble for total_runs.
#
# All 19 OOF arrays are already on S3. This instance downloads them with an
# exact prefix filter (oof_total_runs_), runs ensemble, uploads the pkl + summary.
#
# Usage:
#   bash scripts/launch_ensemble_total_runs_x86.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_train_total_runs.tar.gz"
FEATURES_KEY="artifacts/features/game_features.parquet"
AMI="ami-0bdc7d025135d7b49"   # AL2023 x86_64
INSTANCE_TYPE="c7i.4xlarge"   # x86_64, 16 vCPU — enough for ensemble refit
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TARGET="total_runs"

USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/ensemble_total_runs.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ensemble ${TARGET} at \$(date -u) ==="

dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy \
  lightgbm xgboost catboost ydf optuna tqdm

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

# Pull sizing curve (needed for model refit in ensemble step)
mkdir -p pregame/artifacts/sizing
aws s3 cp s3://${BUCKET}/artifacts/sizing/sizing_curve_${TARGET}.json \
  pregame/artifacts/sizing/sizing_curve_${TARGET}.json

# Pull ALL total_runs artifacts — explicit sync with exact prefix avoids
# first_5_total_runs contamination (s3 sync --include globs are substring matches)
mkdir -p pregame/artifacts/models
aws s3 sync s3://${BUCKET}/artifacts/models/ pregame/artifacts/models/ \
  --exclude "*" \
  --include "oof_total_runs_*" \
  --include "oof_game_pks_total_runs_*" \
  --include "params_total_runs_*" \
  --include "training_summary_total_runs_*"

echo "OOF arrays downloaded:"
ls pregame/artifacts/models/oof_total_runs_*.npy | wc -l

# Run ensemble over all 19 families
echo "=== Building 19-family ensemble ==="
python3.12 -m pregame.cli ensemble \
  --models pregame/artifacts/models \
  --features pregame/artifacts/features/game_features.parquet \
  --target ${TARGET} \
  --tier A

# Upload ensemble pkl + updated summary
aws s3 sync pregame/artifacts/models/ s3://${BUCKET}/artifacts/models/ \
  --exclude "*" \
  --include "ensemble_${TARGET}_*.pkl" \
  --include "training_summary_${TARGET}_*.json"

echo "Uploaded ensemble to s3://${BUCKET}/artifacts/models/"

# Print result
python3.12 -c "
import json
d = json.load(open('pregame/artifacts/models/training_summary_${TARGET}_A.json'))
ens = d.get('__ensemble__', d.get('total_runs', {}))
print('Ensemble metrics:', ens.get('ensemble_metrics'))
print('Members:', ens.get('members'))
print('Weights:', [round(w,4) for w in ens.get('weights',[])])
" 2>/dev/null || true

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/artifacts/models/ensemble_total_runs.log

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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ensemble_${TARGET}},{Key=Purpose,Value=train_mda_v1}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched ensemble instance → $INSTANCE_ID"
echo ""
echo "Monitor (~5 min bootstrap, ~15 min refit):"
echo "  IP=\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].PublicIpAddress' --output text)"
echo "  ssh -i /Users/michaelharoon/Documents/SENSITIVE/awstest.pem ec2-user@\$IP 'tail -f /var/log/ensemble_total_runs.log'"
echo ""
echo "Or poll S3 when done:"
echo "  aws s3 cp s3://${BUCKET}/artifacts/models/ensemble_total_runs.log /tmp/ && tail -20 /tmp/ensemble_total_runs.log"
