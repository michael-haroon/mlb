#!/bin/bash
# Launch one c6g.4xlarge per target for CFI-MDI (de Prado per-tree aggregation).
#
# CFI-MDI is in-sample only (fit RF + sum per tree) — much lighter than
# the full importance pipeline, so we use smaller instances.
#
# Prerequisites:
#   - awstest key pair in AWS
#   - IAM profile: read-write-mlb-s3
#   - Code tarball uploaded:
#     s3://mlb-265753586044-us-east-1-an/artifacts/code/mlb_cfi_mdi.tar.gz
#   - Existing cluster_map.json per target in:
#     s3://mlb-265753586044-us-east-1-an/artifacts/importance/{target}/cluster_map.json
#
# Usage:
#   bash scripts/launch_cfi_mdi_all_targets.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="artifacts/code/mlb_cfi_mdi.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
INSTANCE_TYPE="c6g.4xlarge"
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
exec > /var/log/cfi_mdi_${TARGET}.log 2>&1

# Install deps
dnf install -y python3.11 python3.11-pip git 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5

# Pull code
aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb --strip-components=0

cd /home/ec2-user/mlb

# Run CFI-MDI only
python3.11 scripts/run_cfi_mdi_ec2.py --target ${TARGET}
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
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cfi_mdi_${TARGET}},{Key=Purpose,Value=cfi_mdi_deprado}]" \
        --user-data "$USER_DATA" \
        --query "Instances[0].InstanceId" \
        --output text)

    echo "Launched $TARGET → $INSTANCE_ID"
done

echo ""
echo "All 10 instances launched. Monitor with:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=cfi_mdi_deprado' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Check S3 outputs:"
echo "  for t in ${TARGETS[*]}; do echo \"\$t:\"; aws s3 ls s3://${BUCKET}/artifacts/importance/\$t/importance_cfi_mdi_ 2>/dev/null | wc -l; done"
