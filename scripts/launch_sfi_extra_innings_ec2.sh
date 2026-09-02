#!/bin/bash
# Launch one c8g.4xlarge to re-run SFI for extra_innings with class_weight=None fix.
#
# Before running: rebuild and upload the code tarball with the class_weight fix:
#   cd /path/to/mlb
#   tar --exclude='pregame/artifacts' --exclude='.git' --exclude='__pycache__' \
#       --exclude='*.pyc' --exclude='data' \
#       -czf /tmp/mlb_sfi_fix.tar.gz .
#   aws s3 cp /tmp/mlb_sfi_fix.tar.gz \
#       s3://mlb-265753586044-us-east-1-an/artifacts/code/mlb_sfi_fix.tar.gz
#
# After instance terminates (~1–2h), download and re-route locally:
#   aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance/extra_innings/importance_sfi_raw.csv \
#       pregame/artifacts/importance/extra_innings/importance_sfi_raw.csv
#   aws s3 cp s3://mlb-265753586044-us-east-1-an/artifacts/importance/extra_innings/importance_sfi_meta.json \
#       pregame/artifacts/importance/extra_innings/importance_sfi_meta.json
#   conda run -n pred python scripts/regate_and_route.py
#
# Usage:
#   bash scripts/launch_sfi_extra_innings_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_sfi_fix.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c8g.4xlarge"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TARGET="extra_innings"

USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/sfi_extra_innings.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip
python3.11 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

echo "=== Running SFI for ${TARGET} ==="
python3.11 scripts/run_sfi_only_ec2.py --target ${TARGET}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/artifacts/importance/${TARGET}/sfi_only.log

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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=sfi_extra_innings},{Key=Purpose,Value=sfi_class_weight_fix}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched extra_innings SFI → $INSTANCE_ID"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query \"Reservations[*].Instances[*].{State:State.Name}\" --output table"
echo "Log: s3://${BUCKET}/artifacts/importance/${TARGET}/sfi_only.log"
echo "Results: s3://${BUCKET}/artifacts/importance/${TARGET}/importance_sfi_raw.csv"
