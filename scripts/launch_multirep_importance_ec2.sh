#!/bin/bash
# Launch 20 EC2 instances: 10 targets × 2 tests (mda, desub_mda) with n_repeats=30.
#
# MDA does not use clusters. DESUB computes ONC locally per instance (~2 min).
# No separate ONC phase needed — just flat distribution.
#
# Instance type: c8g.8xlarge (32 vCPU Graviton3, 64GB) — compute-bound workload.
# Each instance self-terminates after uploading artifacts to S3.
#
# Usage:
#   bash scripts/launch_multirep_importance_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_importance_v6.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
INSTANCE_TYPE="c8g.8xlarge"
N_REPEATS=250

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

TESTS=(
    mda
    desub_mda
)

echo "=== Multi-repeat importance: ${#TARGETS[@]} targets × ${#TESTS[@]} tests = $((${#TARGETS[@]} * ${#TESTS[@]})) instances ==="
echo "  Instance type: ${INSTANCE_TYPE}"
echo "  n_repeats: ${N_REPEATS}"
echo ""

LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for TEST in "${TESTS[@]}"; do
        USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/multirep_${TARGET}_${TEST}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET}/${TEST} n_repeats=${N_REPEATS} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_importance_single_test_ec2.py --target ${TARGET} --test ${TEST} --n-repeats ${N_REPEATS}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET}/${TEST} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/importance/${TARGET}/multirep_${TEST}.log
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
            --instance-initiated-shutdown-behavior stop \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=multirep_${TARGET}_${TEST}},{Key=Purpose,Value=multirep_importance}]" \
            --user-data "$USER_DATA" \
            --query "Instances[0].InstanceId" \
            --output text)

        echo "  ${TARGET}/${TEST} → ${INSTANCE_ID}"
        LAUNCHED=$((LAUNCHED + 1))
    done
done

echo ""
echo "=== ${LAUNCHED} instances launched ==="
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=multirep_importance' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results: s3://${BUCKET}/classical_learning/artifacts/importance/<target>/"
echo "  - importance_mda_raw.csv"
echo "  - importance_mda_repeat_sd.csv"
echo "  - importance_desub_mda_raw.csv"
echo "  - importance_desub_mda_repeat_sd.csv"
