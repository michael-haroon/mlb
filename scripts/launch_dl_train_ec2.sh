#!/bin/bash
# Launch a GPU instance to train deep learning models (pregame CNN + live HAN).
#
# Pipeline:
#   1. Build feature store from S3 raw data (2015-2026, skip 2020)
#   2. Train pregame CNN (baseline comparison vs classical)
#   3. Run pregame learning curves (10/25/50/75/100%)
#   4. Train live HAN on pitch sequences
#   5. Run live learning curves
#   6. Upload all results to s3://BUCKET/deep_learning/
#
# Outputs: s3://BUCKET/deep_learning/
#   - feature_store/             Feature store parquets
#   - pregame_cnn/               Model checkpoint, history, eval
#   - pregame_curves/            Learning curve diagnostics
#   - live_han/                  Model checkpoint, history, eval
#   - live_curves/               Learning curve diagnostics
#   - train_dl.log               Full training log
#
# Usage:
#   bash scripts/launch_dl_train_ec2.sh

set -e

BUCKET="mlb-265753586044-us-east-1-an"
CODE_KEY="deep_learning/code/mlb_dl_train.tar.gz"
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Amazon Linux 2023)
AMI="ami-00033101c52a6491e"
INSTANCE_TYPE="g5.xlarge"
SG="sg-0583a4de608a95a41"
SUBNET="subnet-013362590293de96a"
IAM_PROFILE="read-write-mlb-s3"
KEY_NAME="awstest"

USER_DATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -eo pipefail

LOG_FILE="/var/log/train_dl.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: DL training pipeline at $(date -u) ==="

# Check GPU
nvidia-smi || echo "WARNING: nvidia-smi failed"

# DL AMI has NVIDIA driver + CUDA 13.2. Use system python + pip venv.
export HOME=/home/ec2-user

# Install Python 3.11 if not present
if ! command -v python3.11 &>/dev/null; then
    echo "Installing Python 3.11..."
    dnf install -y python3.11 python3.11-pip python3.11-devel 2>/dev/null || \
    yum install -y python3.11 python3.11-pip python3.11-devel 2>/dev/null || \
    (echo "Trying deadsnakes PPA..." && \
     apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null)
fi

# Create venv
python3.11 -m venv /home/ec2-user/venv
source /home/ec2-user/venv/bin/activate

# Install PyTorch with CUDA 12.1 (forward-compatible with CUDA 13.2 driver)
echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
pip install pandas pyarrow scikit-learn scipy boto3 s3fs fsspec

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Pull code
BUCKET="mlb-265753586044-us-east-1-an"
aws s3 cp s3://${BUCKET}/deep_learning/code/mlb_dl_train.tar.gz /tmp/mlb_dl_code.tar.gz
mkdir -p /home/ec2-user/mlb
tar -xzf /tmp/mlb_dl_code.tar.gz -C /home/ec2-user/mlb
cd /home/ec2-user/mlb

export PYTHONPATH=/home/ec2-user/mlb:/home/ec2-user/mlb/deep_learning

echo ""
echo "=== STEP 1: Build feature store (2015-2026) ==="
echo "Started at $(date -u)"
python -m mlb_dl.train build-feature-store \
  --source s3://${BUCKET}/data \
  --output /home/ec2-user/mlb/feature_store \
  --season-start 2015

echo ""
echo "=== STEP 2: Train pregame CNN ==="
echo "Started at $(date -u)"
python -m mlb_dl.train fit-pregame \
  --feature-store /home/ec2-user/mlb/feature_store \
  --output /home/ec2-user/mlb/pregame_cnn \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --hidden-dim 128 \
  --patience 7

echo ""
echo "=== STEP 3: Pregame learning curves ==="
echo "Started at $(date -u)"
python -m mlb_dl.train learning-curves-pregame \
  --feature-store /home/ec2-user/mlb/feature_store \
  --output /home/ec2-user/mlb/pregame_curves \
  --fractions 0.10 0.25 0.50 0.75 1.0 \
  --epochs 20 \
  --patience 5

echo ""
echo "=== STEP 4: Train live HAN ==="
echo "Started at $(date -u)"
python -m mlb_dl.train fit-live \
  --feature-store /home/ec2-user/mlb/feature_store \
  --output /home/ec2-user/mlb/live_han \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --d-model 128 \
  --patience 10

echo ""
echo "=== STEP 5: Live learning curves ==="
echo "Started at $(date -u)"
python -m mlb_dl.train learning-curves-live \
  --feature-store /home/ec2-user/mlb/feature_store \
  --output /home/ec2-user/mlb/live_curves \
  --fractions 0.10 0.25 0.50 0.75 1.0 \
  --epochs 20 \
  --patience 7

echo ""
echo "=== STEP 6: Upload results ==="
echo "Started at $(date -u)"
aws s3 sync /home/ec2-user/mlb/feature_store/ s3://${BUCKET}/deep_learning/feature_store/ \
  --exclude "*.parquet" --include "manifest.json" --include "market_specs.json"
# Upload feature store manifests only (parquets are huge)

aws s3 sync /home/ec2-user/mlb/pregame_cnn/ s3://${BUCKET}/deep_learning/pregame_cnn/
aws s3 sync /home/ec2-user/mlb/pregame_curves/ s3://${BUCKET}/deep_learning/pregame_curves/
aws s3 sync /home/ec2-user/mlb/live_han/ s3://${BUCKET}/deep_learning/live_han/
aws s3 sync /home/ec2-user/mlb/live_curves/ s3://${BUCKET}/deep_learning/live_curves/
aws s3 cp "$LOG_FILE" s3://${BUCKET}/deep_learning/train_dl.log

echo ""
echo "=== COMPLETE at $(date -u) ==="
echo "Results at: s3://${BUCKET}/deep_learning/"

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
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=dl_train_pregame_live},{Key=Purpose,Value=deep_learning_v1}]" \
    --user-data "$USER_DATA" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "Launched DL training → $INSTANCE_ID"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].[State.Name,PublicIpAddress]' --output text"
echo "  ssh -i ~/.ssh/awstest.pem ubuntu@<ip> 'tail -50 /var/log/train_dl.log'"
echo ""
echo "Results (when done):"
echo "  aws s3 ls s3://${BUCKET}/deep_learning/ --recursive"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/train_dl.log /tmp/ && tail -100 /tmp/train_dl.log"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/pregame_cnn/history.json /tmp/ && cat /tmp/history.json"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/pregame_curves/learning_curves_pregame.json /tmp/ && cat /tmp/learning_curves_pregame.json"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/live_han/test_metrics.json /tmp/ && cat /tmp/test_metrics.json"
echo "  aws s3 cp s3://${BUCKET}/deep_learning/live_curves/learning_curves_live.json /tmp/ && cat /tmp/learning_curves_live.json"
