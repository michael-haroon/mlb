#!/bin/bash
# BATCH 3: Final 48 instances.
# Remaining: home_runs (desub_mda, pca_mda, resid_mda) + away_runs, first_5_home_run_diff, first_5_total_runs (full)

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_importance_ratings_v1.tar.gz"
CLUSTER_KEY="classical_learning/artifacts/importance_ratings/cluster_map.json"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
TEST_INSTANCE_TYPE="c8g.4xlarge"
PERM_INSTANCE_TYPE="c8g.8xlarge"
N_REPEATS=250

CV_MODES=(expanding sliding_3)
LAUNCHED=0

launch_test() {
    local TARGET=$1
    local TEST=$2
    local CV_MODE=$3

    case "$TEST" in
        mda|desub_mda|cfi_mda) ITYPE="$PERM_INSTANCE_TYPE" ;;
        *)                     ITYPE="$TEST_INSTANCE_TYPE" ;;
    esac

    USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/importance_ratings_${TARGET}_${TEST}_${CV_MODE}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ratings ${TARGET}/${TEST}/${CV_MODE} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_importance_ratings_ec2.py --target ${TARGET} --test ${TEST} --cluster-key ${CLUSTER_KEY} --n-repeats ${N_REPEATS} --cv-mode ${CV_MODE}
EXIT_CODE=\$?

echo "=== DONE: ratings ${TARGET}/${TEST}/${CV_MODE} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/importance_ratings/${CV_MODE}/${TARGET}/${TEST}.log
shutdown -h now
USERDATA_EOF
)

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI" \
        --instance-type "$ITYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG" \
        --subnet-id "$SUBNET" \
        --iam-instance-profile "Name=$IAM_PROFILE" \
        --instance-initiated-shutdown-behavior stop \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=imp_ratings_${TARGET}_${TEST}_${CV_MODE}},{Key=Purpose,Value=importance_ratings}]" \
        --user-data "$USER_DATA" \
        --query "Instances[0].InstanceId" \
        --output text)

    echo "  ${TARGET}/${TEST}/${CV_MODE} (${ITYPE}) → ${INSTANCE_ID}"
    LAUNCHED=$((LAUNCHED + 1))
}

echo "=== BATCH 3: Launching final 48 importance instances ==="
echo ""

# home_runs remainder
for TEST in desub_mda pca_mda resid_mda; do
    for CV_MODE in "${CV_MODES[@]}"; do
        launch_test home_runs "$TEST" "$CV_MODE"
    done
done

# Full remaining targets
for TARGET in away_runs first_5_home_run_diff first_5_total_runs; do
    for TEST in mdi_cfi_mdi mda cfi_mda sfi desub_mda pca_mda resid_mda; do
        for CV_MODE in "${CV_MODES[@]}"; do
            launch_test "$TARGET" "$TEST" "$CV_MODE"
        done
    done
done

echo ""
echo "=== BATCH 3: ${LAUNCHED} instances launched ==="
