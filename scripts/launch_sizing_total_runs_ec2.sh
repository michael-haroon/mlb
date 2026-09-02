#!/bin/bash
# Launch a single c8g.8xlarge to re-compute sizing curve for total_runs
# using the new MDA-validated cluster-first feature ordering.
#
# Must run BEFORE training because the existing sizing_curve_total_runs.json
# was computed with the old evidence-group ordering — S* is invalid for the
# new cluster-first ordered feature list.
#
# Outputs: s3://BUCKET/artifacts/sizing/sizing_curve_total_runs.json
# Monitor: aws s3 cp s3://BUCKET/artifacts/sizing/sizing_curve_total_runs.json /tmp/check.json && cat /tmp/check.json
#
# Usage:
#   bash scripts/launch_sizing_total_runs_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_train_total_runs.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c8g.8xlarge"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TARGET="total_runs"

USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/sizing_${TARGET}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: sizing ${TARGET} at \$(date -u) ==="

# Install deps (catboost + ydf take ~2m)
dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy \
  lightgbm xgboost catboost ydf optuna tqdm

# Pull code
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb --strip-components=1
cd /home/ec2-user/mlb

echo "=== Running sizing for ${TARGET} ==="
python3.12 scripts/run_sizing_ec2.py --target ${TARGET} --self-terminate
EXIT_CODE=\$?

echo "=== DONE: ${TARGET} exit_code=\${EXIT_CODE} at \$(date -u) ==="
aws s3 cp "\$LOG_FILE" s3://${BUCKET}/artifacts/sizing/sizing_${TARGET}.log
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sizing_${TARGET}},{Key=Purpose,Value=sizing_mda_v1}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched sizing for $TARGET → $INSTANCE_ID"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].State.Name' --output text"
echo "  aws s3 cp s3://${BUCKET}/artifacts/sizing/sizing_${TARGET}.log /tmp/ && tail -30 /tmp/sizing_${TARGET}.log"
echo "Result:"
echo "  aws s3 cp s3://${BUCKET}/artifacts/sizing/sizing_curve_${TARGET}.json /tmp/ && python3 -c \"import json; d=json.load(open('/tmp/sizing_curve_${TARGET}.json')); [print(f'  {k:<25} S*={v[\\\"optimal_S\\\"]}') for k,v in d['per_family'].items() if v.get('optimal_S')]\""
