#!/bin/bash
# MASTER ANALYSIS SCRIPT — single entry point for the full feature analysis pipeline.
#
# Phase 1:  1 ONC clustering instance (~2-3 hrs on 700 features)
# Phase 1b: 1 PCA cross-check instance (launches with Phase 1, independent)
# Phase 2:  140 importance instances (10 targets × 7 tests × 2 CV modes)
#           launch immediately after ONC completes
# Phase 3:  30 interpretability instances (10 targets × 3 tests: h_stat, ale, shap)
#           launch alongside Phase 2; each polls internally for its target's
#           expanding MDI sentinel before running (no wasted wait between targets)
#
# Total: 172 instances (1 ONC + 1 PCA + 140 importance + 30 interpretability)
#
# Importance tests: mdi_cfi_mdi, mda, cfi_mda, sfi, desub_mda, pca_mda, resid_mda
# CV modes:         expanding (all prior years), sliding_3 (last 3 years only)
# Interp tests:     h_stat (Friedman H-stat), ale (ALE curves), shap (TreeSHAP)
# Targets:          home_win, yrfi, first_5_home_win, extra_innings,
#                   home_run_diff, total_runs, home_runs, away_runs,
#                   first_5_home_run_diff, first_5_total_runs
#
# Usage:
#   bash scripts/launch_importance_and_pca_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_importance_v6.tar.gz"
INTERP_CODE_KEY="classical_learning/artifacts/code/mlb_interpretability_v1.tar.gz"
CLUSTER_KEY="classical_learning/artifacts/importance/cluster_map.json"
SENTINEL_KEY="classical_learning/artifacts/importance/clustering_done.sentinel"
MDI_DONE_PREFIX="classical_learning/artifacts/importance/mdi_done"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
ONC_INSTANCE_TYPE="c8g.metal-24xl"
TEST_INSTANCE_TYPE="c8g.8xlarge"
PCA_INSTANCE_TYPE="c8g.8xlarge"
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
aws s3 rm "s3://${BUCKET}/${MDI_DONE_PREFIX}/" --recursive 2>/dev/null || true

# =============================================================================
# PHASE 1: Launch ONC clustering instance
# =============================================================================
echo ""
echo "=== PHASE 1: Launching ONC clustering instance ==="

ONC_USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -e

LOG_FILE="/var/log/onc_clustering.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: ONC clustering at $(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://BUCKET_PLACEHOLDER/CODE_KEY_PLACEHOLDER /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_onc_clustering_ec2.py
EXIT_CODE=$?

echo "=== DONE: ONC clustering exit_code=${EXIT_CODE} at $(date -u) ==="

aws s3 cp "$LOG_FILE" s3://BUCKET_PLACEHOLDER/classical_learning/artifacts/importance/onc_clustering.log
shutdown -h now
USERDATA_EOF
)

# Substitute placeholders
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=imp_v6_onc_clustering},{Key=Purpose,Value=importance_v6}]" \
    --user-data "$ONC_USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  ONC clustering → ${ONC_INSTANCE_ID}"

# =============================================================================
# PHASE 1b: Launch PCA cross-check (independent of clustering)
# =============================================================================
echo ""
echo "=== Launching PCA cross-check (1 instance, all targets) ==="

PCA_USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -e

LOG_FILE="/var/log/pca_crosscheck_all.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: PCA cross-check (all targets) at $(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://BUCKET_PLACEHOLDER/CODE_KEY_PLACEHOLDER /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

for T in home_win yrfi first_5_home_win extra_innings home_run_diff total_runs home_runs away_runs first_5_home_run_diff first_5_total_runs; do
    echo "--- PCA cross-check: $T ---"
    python3.11 scripts/run_pca_crosscheck_ec2.py --target $T
done

echo "=== DONE: PCA cross-check all targets at $(date -u) ==="

aws s3 cp "$LOG_FILE" s3://BUCKET_PLACEHOLDER/classical_learning/artifacts/importance/pca_crosscheck_all.log
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pca_v6_all_targets},{Key=Purpose,Value=importance_v6}]" \
    --user-data "$PCA_USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  pca_crosscheck (all targets) → ${PCA_INSTANCE_ID}"

# =============================================================================
# PHASE 2: Wait for clustering, then launch 70 test instances
# =============================================================================
echo ""
echo "=== PHASE 2: Waiting for ONC clustering to complete... ==="
echo "  Polling s3://${BUCKET}/${SENTINEL_KEY} every 60s"
echo ""

while true; do
    if aws s3 ls "s3://${BUCKET}/${SENTINEL_KEY}" >/dev/null 2>&1; then
        echo "  ✓ Clustering complete! Launching test instances..."
        break
    fi
    echo "  $(date '+%H:%M:%S') — still waiting..."
    sleep 60
done

echo ""
echo "=== Launching importance instances (${#TARGETS[@]} targets × ${#TESTS[@]} tests × ${#CV_MODES[@]} cv_modes = $((${#TARGETS[@]} * ${#TESTS[@]} * ${#CV_MODES[@]})) instances) ==="
echo ""

LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for TEST in "${TESTS[@]}"; do
        for CV_MODE in "${CV_MODES[@]}"; do
            USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/importance_${TARGET}_${TEST}_${CV_MODE}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET}/${TEST}/${CV_MODE} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5
python3.11 -m pip install --quiet ydf 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_importance_single_test_ec2.py --target ${TARGET} --test ${TEST} --cluster-key ${CLUSTER_KEY} --n-repeats ${N_REPEATS} --cv-mode ${CV_MODE}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET}/${TEST}/${CV_MODE} exit_code=\${EXIT_CODE} at \$(date -u) ==="

# Write per-target MDI done sentinel so interpretability instances can start
if [[ "${TEST}" == "mdi_cfi_mdi" && "${CV_MODE}" == "expanding" && \${EXIT_CODE} -eq 0 ]]; then
    echo "done" | aws s3 cp - s3://${BUCKET}/${MDI_DONE_PREFIX}/${TARGET}.sentinel
    echo "  Wrote MDI done sentinel for ${TARGET}"
fi

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/importance/${CV_MODE}/${TARGET}/${TEST}.log
shutdown -h now
USERDATA_EOF
)

            INSTANCE_ID=$(aws ec2 run-instances \
                --image-id "$AMI" \
                --instance-type "$TEST_INSTANCE_TYPE" \
                --key-name "$KEY_NAME" \
                --security-group-ids "$SG" \
                --subnet-id "$SUBNET" \
                --iam-instance-profile "Name=$IAM_PROFILE" \
                --instance-initiated-shutdown-behavior stop \
                --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=imp_v6_${TARGET}_${TEST}_${CV_MODE}},{Key=Purpose,Value=importance_v6}]" \
                --user-data "$USER_DATA" \
                --query "Instances[0].InstanceId" \
                --output text)

            echo "  ${TARGET}/${TEST}/${CV_MODE} → ${INSTANCE_ID}"
            LAUNCHED=$((LAUNCHED + 1))
        done
    done
done

echo ""
echo "=== ${LAUNCHED} importance instances launched ==="

# =============================================================================
# PHASE 3: Launch interpretability instances (h_stat, ale, shap per target)
# Each instance polls internally for its target's MDI done sentinel before
# running — launches immediately but waits for expanding MDI to complete.
# =============================================================================
echo ""
echo "=== PHASE 3: Launching interpretability instances (${#TARGETS[@]} targets × 3 tests = $((${#TARGETS[@]} * 3)) instances) ==="
echo ""

H_STAT_INSTANCE_TYPE="c8g.8xlarge"
ALE_INSTANCE_TYPE="c8g.4xlarge"
SHAP_INSTANCE_TYPE="c8g.8xlarge"

INTERP_TESTS=(h_stat ale shap)
INTERP_LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for ITEST in "${INTERP_TESTS[@]}"; do
        case "$ITEST" in
            h_stat) ITYPE="$H_STAT_INSTANCE_TYPE" ;;
            ale)    ITYPE="$ALE_INSTANCE_TYPE" ;;
            shap)   ITYPE="$SHAP_INSTANCE_TYPE" ;;
        esac

        INTERP_USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/interpretability_${TARGET}_${ITEST}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: interpretability ${TARGET}/${ITEST} at \$(date -u) ==="

# Wait for expanding MDI to finish for this target
MDI_SENTINEL="s3://${BUCKET}/${MDI_DONE_PREFIX}/${TARGET}.sentinel"
echo "Waiting for MDI sentinel: \${MDI_SENTINEL}"
while ! aws s3 ls "\${MDI_SENTINEL}" >/dev/null 2>&1; do
    echo "  \$(date '+%H:%M:%S') — waiting for MDI..."
    sleep 60
done
echo "MDI sentinel found, proceeding."

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm shap 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${INTERP_CODE_KEY} /tmp/mlb_interp_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_interp_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_interpretability_ec2.py --target ${TARGET} --test ${ITEST}
EXIT_CODE=\$?

echo "=== DONE: interpretability ${TARGET}/${ITEST} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/interpretability/${TARGET}/${ITEST}.log
shutdown -h now
USERDATA_EOF
)

        INTERP_INSTANCE_ID=$(aws ec2 run-instances \
            --image-id "$AMI" \
            --instance-type "$ITYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG" \
            --subnet-id "$SUBNET" \
            --iam-instance-profile "Name=$IAM_PROFILE" \
            --instance-initiated-shutdown-behavior stop \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=interp_v1_${TARGET}_${ITEST}},{Key=Purpose,Value=interpretability_v1}]" \
            --user-data "$INTERP_USER_DATA" \
            --query "Instances[0].InstanceId" \
            --output text)

        echo "  ${TARGET}/${ITEST} (${ITYPE}) → ${INTERP_INSTANCE_ID}"
        INTERP_LAUNCHED=$((INTERP_LAUNCHED + 1))
    done
done

TOTAL=$((LAUNCHED + INTERP_LAUNCHED + 2))
echo ""
echo "=== TOTAL: ${TOTAL} instances launched ==="
echo "      1 ONC + 1 PCA + ${LAUNCHED} importance + ${INTERP_LAUNCHED} interpretability"
echo ""
echo "Monitor all:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=importance_v6' 'Name=instance-state-name,Values=running,pending' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=interpretability_v1' 'Name=instance-state-name,Values=running,pending' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results:"
echo "  s3://${BUCKET}/classical_learning/artifacts/importance/{expanding,sliding_3}/<target>/"
echo "  s3://${BUCKET}/classical_learning/artifacts/interpretability/<target>/"
