#!/bin/bash
# Launch one c8g.24xlarge per target for forward-only LOYO importance run.
#
# Spins up 10 instances (4 classification + 6 regression), each running
# run_importance_ec2_precomputed.py --target <TARGET>, then terminates.
#
# Prerequisites:
#   - awstest key pair in AWS
#   - IAM profile: read-write-mlb-s3
#   - Code tarball already uploaded:
#     s3://mlb-265753586044-us-east-1-an/artifacts/code/mlb_importance_forward_only_loyo.tar.gz
#
# Usage:
#   bash scripts/launch_importance_all_targets.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_importance_forward_only_loyo.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c8g.24xlarge"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"

TARGETS=(
    home_win
    yrfi
    first_5_home_win
    extra_innings
    home_run_diff
    total_runs
    home_runs
    away_runs
    first_5_home_run_diff
    first_5_total_runs
)

for TARGET in "${TARGETS[@]}"; do
    USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e
exec > /var/log/importance_${TARGET}.log 2>&1

# Install deps
dnf install -y python3.11 python3.11-pip git 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5

# Pull code
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb --strip-components=0

cd /home/ec2-user/mlb

# Run importance
python3.11 scripts/run_importance_ec2_precomputed.py --target ${TARGET}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET} exit_code=\${EXIT_CODE} ==="

# Terminate self
INSTANCE_ID=\$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids \$INSTANCE_ID --region us-east-1
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
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=importance_${TARGET}},{Key=Purpose,Value=importance_forward_only_loyo}]" \
        --user-data "$USER_DATA" \
        --query "Instances[0].InstanceId" \
        --output text)

    echo "Launched $TARGET → $INSTANCE_ID"
done

echo ""
echo "All 10 instances launched. Monitor with:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=importance_forward_only_loyo' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results upload to: s3://${BUCKET}/artifacts/importance/<target>/"
