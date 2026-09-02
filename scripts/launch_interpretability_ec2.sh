#!/bin/bash
# Interpretability pipeline: H-stat, ALE, TreeSHAP across all targets.
#
# 30 instances total: 10 targets × 3 tests
#   - h_stat: Friedman's H-statistic (pairwise feature interactions)
#   - ale:    Accumulated Local Effects (response shape curves)
#   - shap:   TreeSHAP (global importance + interaction matrix)
#
# Prerequisites:
#   - MDI results must exist in S3 (from prior importance run) for feature
#     selection. If missing, each test computes MDI internally (slower).
#   - game_features.parquet must be current in S3.
#   - Code tarball must be updated with interpretability module.
#
# Instance sizing:
#   - h_stat: CPU-bound (PD grid × pairs). c8g.8xlarge (~2-4 hrs for top 30).
#   - ale:    CPU-bound (per-feature ALE + CV). c8g.4xlarge (~1-2 hrs).
#   - shap:   Memory-bound (SHAP values matrix). c8g.8xlarge (~1-3 hrs).
#
# Usage:
#   bash scripts/launch_interpretability_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_interpretability_v1.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"

# Instance types per test (sized for ~750 features)
H_STAT_INSTANCE_TYPE="c8g.8xlarge"   # 32 vCPU — parallelized pair computation
ALE_INSTANCE_TYPE="c8g.4xlarge"      # 16 vCPU — per-feature parallelism
SHAP_INSTANCE_TYPE="c8g.8xlarge"     # 32 vCPU, 64GB — SHAP matrix in memory

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
    h_stat
    ale
    shap
)

# Map test to instance type
get_instance_type() {
    case "$1" in
        h_stat) echo "$H_STAT_INSTANCE_TYPE" ;;
        ale)    echo "$ALE_INSTANCE_TYPE" ;;
        shap)   echo "$SHAP_INSTANCE_TYPE" ;;
    esac
}

echo "=== Launching interpretability instances (${#TARGETS[@]} targets × ${#TESTS[@]} tests = $((${#TARGETS[@]} * ${#TESTS[@]}))) ==="
echo ""

LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for TEST in "${TESTS[@]}"; do
        INSTANCE_TYPE=$(get_instance_type "$TEST")

        USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/interpretability_${TARGET}_${TEST}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET}/${TEST} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm shap 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_interpretability_ec2.py --target ${TARGET} --test ${TEST}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET}/${TEST} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/interpretability/${TARGET}/${TEST}.log
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
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=interp_v1_${TARGET}_${TEST}},{Key=Purpose,Value=interpretability_v1}]" \
            --user-data "$USER_DATA" \
            --query "Instances[0].InstanceId" \
            --output text)

        echo "  ${TARGET}/${TEST} (${INSTANCE_TYPE}) → ${INSTANCE_ID}"
        LAUNCHED=$((LAUNCHED + 1))
    done
done

echo ""
echo "=== ${LAUNCHED} instances launched ==="
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=interpretability_v1' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results: s3://${BUCKET}/classical_learning/artifacts/interpretability/<target>/"
