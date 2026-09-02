#!/bin/bash
# Launch one c8g.4xlarge per target for PCA cross-check (MDI + MDA + SFI on PCs).
#
# Prerequisites:
#   - Code tarball already uploaded:
#     s3://mlb-265753586044-us-east-1-an/artifacts/code/mlb_pca_crosscheck.tar.gz
#
# Usage:
#   bash scripts/launch_pca_crosscheck_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="pregame/artifacts/code/mlb_pca_crosscheck.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c8g.4xlarge"
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

# Log to both file and stdout
LOG_FILE="/var/log/pca_crosscheck_${TARGET}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET} at \$(date -u) ==="

# Install deps
dnf install -y python3.12 python3.12-pip
python3.12 -m pip install boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm

# Pull code
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

echo "=== Running PCA cross-check for ${TARGET} ==="

# Run PCA cross-check
python3.12 scripts/run_pca_crosscheck_ec2.py --target ${TARGET}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET} exit_code=\${EXIT_CODE} at \$(date -u) ==="

# Upload log to S3
aws s3 cp "\$LOG_FILE" s3://${BUCKET}/pregame/artifacts/importance/${TARGET}/pca_crosscheck.log

# Shutdown (instance-initiated-shutdown-behavior=terminate handles the rest)
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
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pca_crosscheck_${TARGET}},{Key=Purpose,Value=pca_crosscheck_v4}]" \
        --user-data "$USER_DATA" \
        --query "Instances[0].InstanceId" \
        --output text)

    echo "Launched $TARGET → $INSTANCE_ID"
done

echo ""
echo "All 10 instances launched. Monitor with:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=pca_crosscheck_v4' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Logs upload to: s3://${BUCKET}/pregame/artifacts/importance/<target>/pca_crosscheck.log"
echo "Results at:     s3://${BUCKET}/pregame/artifacts/importance/<target>/kendall_tau.json"
