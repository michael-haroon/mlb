#!/bin/bash
# RATINGS SUBSET IMPORTANCE — single entry point for feature importance on the
# 59 rating/ranking features that DL models cannot learn (elo, massey, colley,
# wolfe, log5, pythag, SRS, consensus + their interactions/derivatives).
#
# Phase 1:  1 ONC clustering instance (~5 min on 59 features)
# Phase 1b: 1 PCA cross-check instance (all targets sequentially)
# Phase 2:  140 importance instances (10 targets × 7 tests × 2 CV modes)
#           launch immediately after ONC completes
#
# Total: 142 instances (1 ONC + 1 PCA + 140 importance)
#
# Output: s3://BUCKET/classical_learning/artifacts/importance_ratings/
#
# Usage:
#   bash scripts/launch_ratings_importance_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_importance_ratings_v1.tar.gz"
CLUSTER_KEY="classical_learning/artifacts/importance_ratings/cluster_map.json"
SENTINEL_KEY="classical_learning/artifacts/importance_ratings/clustering_done.sentinel"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
# Smaller instances — 59 features is ~12x less work than full 700+
ONC_INSTANCE_TYPE="c8g.4xlarge"
TEST_INSTANCE_TYPE="c8g.4xlarge"
PERM_INSTANCE_TYPE="c8g.8xlarge"  # mda, desub_mda, cfi_mda need more — 250 repeats
PCA_INSTANCE_TYPE="c8g.4xlarge"
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
    mdi_cfi_mdi
    mda
    cfi_mda
    sfi
    desub_mda
    pca_mda
    resid_mda
)

CV_MODES=(expanding sliding_3)

# --- Clean up stale sentinels ---
echo "Removing stale sentinels (if any)..."
aws s3 rm "s3://${BUCKET}/${SENTINEL_KEY}" 2>/dev/null || true

# =============================================================================
# PHASE 1: Launch ONC clustering instance
# =============================================================================
echo ""
echo "=== PHASE 1: Launching ONC clustering (ratings subset, ~5 min) ==="

ONC_USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -e

LOG_FILE="/var/log/onc_clustering_ratings.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: ONC clustering (ratings subset) at $(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://BUCKET_PLACEHOLDER/CODE_KEY_PLACEHOLDER /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_onc_clustering_ratings_ec2.py
EXIT_CODE=$?

echo "=== DONE: ONC clustering (ratings) exit_code=${EXIT_CODE} at $(date -u) ==="

aws s3 cp "$LOG_FILE" s3://BUCKET_PLACEHOLDER/classical_learning/artifacts/importance_ratings/onc_clustering.log
shutdown -h now
USERDATA_EOF
)

ONC_USER_DATA="${ONC_USER_DATA//BUCKET_PLACEHOLDER/$BUCKET}"
ONC_USER_DATA="${ONC_USER_DATA//CODE_KEY_PLACEHOLDER/$CODE_KEY}"

ONC_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$ONC_INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG" \
    --subnet-id "$SUBNET" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-initiated-shutdown-behavior stop \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=imp_ratings_onc},{Key=Purpose,Value=importance_ratings}]" \
    --user-data "$ONC_USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  ONC clustering (ratings) → ${ONC_INSTANCE_ID}"

# =============================================================================
# PHASE 1b: Launch PCA cross-check (independent of clustering)
# =============================================================================
echo ""
echo "=== Launching PCA cross-check — ratings subset (1 instance, all targets) ==="

PCA_USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -e

LOG_FILE="/var/log/pca_crosscheck_ratings.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: PCA cross-check (ratings subset, all targets) at $(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://BUCKET_PLACEHOLDER/CODE_KEY_PLACEHOLDER /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

for T in home_win yrfi first_5_home_win extra_innings home_run_diff total_runs home_runs away_runs first_5_home_run_diff first_5_total_runs; do
    echo "--- PCA cross-check (ratings): $T ---"
    python3.11 scripts/run_pca_crosscheck_ratings_ec2.py --target $T
done

echo "=== DONE: PCA cross-check (ratings) all targets at $(date -u) ==="

aws s3 cp "$LOG_FILE" s3://BUCKET_PLACEHOLDER/classical_learning/artifacts/importance_ratings/pca_crosscheck_all.log
shutdown -h now
USERDATA_EOF
)

PCA_USER_DATA="${PCA_USER_DATA//BUCKET_PLACEHOLDER/$BUCKET}"
PCA_USER_DATA="${PCA_USER_DATA//CODE_KEY_PLACEHOLDER/$CODE_KEY}"

PCA_INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$PCA_INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG" \
    --subnet-id "$SUBNET" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-initiated-shutdown-behavior stop \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pca_ratings_all_targets},{Key=Purpose,Value=importance_ratings}]" \
    --user-data "$PCA_USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  pca_crosscheck (ratings, all targets) → ${PCA_INSTANCE_ID}"

# =============================================================================
# PHASE 2: Wait for clustering, then launch 140 test instances
# =============================================================================
echo ""
echo "=== PHASE 2: Waiting for ONC clustering (ratings) to complete... ==="
echo "  Polling s3://${BUCKET}/${SENTINEL_KEY} every 30s"
echo ""

while true; do
    if aws s3 ls "s3://${BUCKET}/${SENTINEL_KEY}" >/dev/null 2>&1; then
        echo "  ✓ Clustering complete! Launching test instances..."
        break
    fi
    echo "  $(date '+%H:%M:%S') — still waiting..."
    sleep 30
done

echo ""
echo "=== Launching importance instances (${#TARGETS[@]} targets × ${#TESTS[@]} tests × ${#CV_MODES[@]} cv_modes = $((${#TARGETS[@]} * ${#TESTS[@]} * ${#CV_MODES[@]})) instances) ==="
echo ""

LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for TEST in "${TESTS[@]}"; do
        # Permutation tests (250 repeats) get larger instances
        case "$TEST" in
            mda|desub_mda|cfi_mda) ITYPE="$PERM_INSTANCE_TYPE" ;;
            *)                     ITYPE="$TEST_INSTANCE_TYPE" ;;
        esac

        for CV_MODE in "${CV_MODES[@]}"; do
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

            echo "  ${TARGET}/${TEST}/${CV_MODE} → ${INSTANCE_ID}"
            LAUNCHED=$((LAUNCHED + 1))
        done
    done
done

TOTAL=$((LAUNCHED + 2))
echo ""
echo "=== TOTAL: ${TOTAL} instances launched ==="
echo "      1 ONC + 1 PCA + ${LAUNCHED} importance"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=importance_ratings' 'Name=instance-state-name,Values=running,pending' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results:"
echo "  s3://${BUCKET}/classical_learning/artifacts/importance_ratings/{expanding,sliding_3}/<target>/"
echo "  s3://${BUCKET}/classical_learning/artifacts/importance_ratings/<target>/pca_cross_check.csv"
