#!/bin/bash
# CV Noise Experiment: compare single-fold vs sliding-2 vs sliding-3 vs expanding
#
# 8 instances (4 modes × 2 targets), each runs independently in parallel.
# Each instance: train RF, run 50 permutation repeats, upload results.
#
# Instance: c8g.8xlarge (32 vCPU, 64 GB RAM)
# Expected runtime: ~20 min (all finish in parallel)
#
# Results:
#   s3://mlb-265753586044-us-east-1-an/classical_learning/artifacts/importance/cv_noise_experiment/
#
# Usage:
#   bash scripts/launch_cv_noise_experiment_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="classical_learning/artifacts/code/mlb_cv_noise_exp.tar.gz"
AMI="ami-0f47531f8c49bd1c6"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"
INSTANCE_TYPE="c8g.16xlarge"

TARGETS=(home_win total_runs)
MODES=(single_fold sliding_2 sliding_3 expanding)

# --- Package and upload code ---
echo "=== Packaging code ==="
TARBALL="/tmp/mlb_cv_noise_exp.tar.gz"
STAGING=$(mktemp -d)

mkdir -p "$STAGING/scripts"
mkdir -p "$STAGING/classical_learning/strategy"
mkdir -p "$STAGING/classical_learning/analysis"

cp scripts/run_cv_noise_experiment_ec2.py "$STAGING/scripts/"
echo '"""MLB classical learning."""' > "$STAGING/classical_learning/__init__.py"
cp classical_learning/strategy/__init__.py "$STAGING/classical_learning/strategy/"
cp classical_learning/strategy/config.py "$STAGING/classical_learning/strategy/"
cp classical_learning/strategy/data.py "$STAGING/classical_learning/strategy/"
cp classical_learning/analysis/__init__.py "$STAGING/classical_learning/analysis/"
cp classical_learning/analysis/feature_importance.py "$STAGING/classical_learning/analysis/"
cp classical_learning/analysis/compute.py "$STAGING/classical_learning/analysis/"

tar -czf "$TARBALL" -C "$STAGING" .
rm -rf "$STAGING"

aws s3 cp "$TARBALL" "s3://${BUCKET}/${CODE_KEY}"
echo "  Uploaded to s3://${BUCKET}/${CODE_KEY}"
echo ""

# --- Launch instances ---
echo "=== Launching ${#TARGETS[@]} targets × ${#MODES[@]} modes = $(( ${#TARGETS[@]} * ${#MODES[@]} )) instances ==="
echo ""

LAUNCHED=0

for TARGET in "${TARGETS[@]}"; do
    for MODE in "${MODES[@]}"; do
        USER_DATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -e

LOG_FILE="/var/log/cv_noise_${TARGET}_${MODE}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=== START: ${TARGET}/${MODE} at \$(date -u) ==="

dnf install -y python3.11 python3.11-pip 2>&1 | tail -5
python3.11 -m pip install --quiet boto3 pyarrow scikit-learn joblib pandas numpy scipy tqdm 2>&1 | tail -5

aws s3 cp s3://${BUCKET}/${CODE_KEY} /tmp/mlb_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_code.tar.gz -C /home/ec2-user/mlb

cd /home/ec2-user/mlb

python3.11 scripts/run_cv_noise_experiment_ec2.py --target ${TARGET} --mode ${MODE}
EXIT_CODE=\$?

echo "=== DONE: ${TARGET}/${MODE} exit_code=\${EXIT_CODE} at \$(date -u) ==="

aws s3 cp "\$LOG_FILE" s3://${BUCKET}/classical_learning/artifacts/importance/cv_noise_experiment/${TARGET}_${MODE}.log
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
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cv_noise_${TARGET}_${MODE}},{Key=Purpose,Value=cv_noise_experiment}]" \
            --user-data "$USER_DATA" \
            --query "Instances[0].InstanceId" \
            --output text)

        echo "  ${TARGET}/${MODE} → ${INSTANCE_ID}"
        LAUNCHED=$((LAUNCHED + 1))
    done
done

echo ""
echo "=== ${LAUNCHED} instances launched ==="
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Purpose,Values=cv_noise_experiment' --query \"Reservations[*].Instances[*].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name}\" --output table"
echo ""
echo "Results (after all complete):"
echo "  aws s3 ls s3://${BUCKET}/classical_learning/artifacts/importance/cv_noise_experiment/"
echo ""
echo "Download diagnostics:"
echo "  for f in \$(aws s3 ls s3://${BUCKET}/classical_learning/artifacts/importance/cv_noise_experiment/ | grep diagnostics | awk '{print \$4}'); do echo \"--- \$f ---\"; aws s3 cp s3://${BUCKET}/classical_learning/artifacts/importance/cv_noise_experiment/\$f -; done"
